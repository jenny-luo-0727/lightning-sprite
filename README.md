# lightning-sprite

Multi-GPU PyTorch Lightning on Intel XPU (DDP + FSDP2).

`lightning-sprite` adds what Lightning 2.6 is missing for Intel Data Center GPU Max (for example, ALCF Aurora):
an XPU accelerator, mixed precision that actually autocasts on XPU, a DDP strategy that does not call CUDA, plain
FSDP2 data parallelism, and a cluster environment for `mpiexec` (PALS) launches.

Not affiliated with Lightning AI or Intel.

> **Status: alpha.** Parity checks pass on A100 and on 1 to 8 Aurora nodes, and checkpoint checks on A100 and
> on 1 and 2 Aurora nodes, one run per configuration.
> See [Tested](#tested), [BF16 parity](#bf16-parity), [Checkpoints](#checkpoints) and [Open items](#open-items).

## Why

| Gap in Lightning 2.6.5 | What this package does |
|---|---|
| No XPU accelerator (Lightning #20938) | `XPUAccelerator` |
| `bf16-mixed` with a custom accelerator autocasts for `"cuda"`, so XPU silently trains in fp32 (#21982) | `XPUMixedPrecision` |
| `DDPStrategy` builds the model wrapper inside `torch.cuda.stream(...)`, which raises without CUDA | `XPUDDPStrategy` |
| `FSDPStrategy` (FSDP1) refuses global-norm gradient clipping and hardcodes CUDA | `FSDP2Strategy`, built on `ModelParallelStrategy` |
| `ModelParallelStrategy`'s `"auto"` mesh is tensor-parallel within a node, so one node shards nothing | `FSDP2Strategy` defaults to data parallelism over every process |
| Without `mpi4py`, an `mpiexec` launch falls back to self-spawning workers | `PALSEnvironment` |

## Install

```bash
pip install lightning-sprite
```

Requires a torch build with XPU support (on Aurora, the `frameworks` module) and `lightning>=2.6.5,<2.7`.

## Use

`trainer_kwargs` returns the accelerator, strategy, devices and plugins:

```python
import lightning as L
from lightning_sprite import trainer_kwargs

trainer = L.Trainer(**trainer_kwargs("ddp", devices=12, num_nodes=2), max_epochs=10)
```

For FSDP2, shard the model in `configure_model`. Put every trainable module, including any task head, inside the
module you shard:

```python
from lightning_sprite import shard_model, trainer_kwargs


class Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = BackboneWithHead()  # e.g. a peft-wrapped Hugging Face decoder plus a head

    def configure_model(self):
        shard_model(self.net, self.device_mesh["data_parallel"])


trainer = L.Trainer(**trainer_kwargs("fsdp", devices=12, num_nodes=2), gradient_clip_val=1.0)
```

`shard_model` shards each block Hugging Face lists in `_no_split_modules`, bottom-up, then the root. Pass
`layer_classes=` for other models. It is idempotent, as Lightning requires: `configure_model` runs again for a later
`validate`/`test`/`predict`, and an already-sharded model is left alone.

Before fitting, `FSDP2Strategy` raises if any trainable parameter is not managed by a `fully_shard` module. Such a
parameter's gradients would never be averaged across ranks. Being a `DTensor` is not enough, since tensor
parallelism alone also produces `DTensor`s.

Under an external launcher, `FSDP2Strategy` also checks the launched process count against
`devices x num_nodes`. Lightning does this only for DDP.

Precision: `"bf16-mixed"` (default) and `"16-mixed"` go through `XPUMixedPrecision`, as do Lightning's legacy
aliases `"bf16"`, `"16"` and `16`. `"32-true"` and `"bf16-true"` pass through unchanged.
`trainer_kwargs("single", devices=[3])` pins tile 3; `"auto"` or `1` pins the current tile.

## Launching on Aurora

```bash
module load frameworks
export ZE_FLAT_DEVICE_HIERARCHY=FLAT        # 12 devices per node, one per tile
mpiexec -n $((NNODES * 12)) --ppn 12 python train.py
```

- Launch one rank per tile.
- Do **not** give each rank its own `ZE_AFFINITY_MASK`. Each rank selects `xpu:<local rank>` from all 12 tiles, and
  `XPUAccelerator` raises if it cannot see its tile.
- `PALSEnvironment` is added automatically when `PALS_RANKID` is set.
- It reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE`, `PALS_RANKID`/`PALS_LOCAL_RANKID`, `PMIX_RANK` and `PMI_*`. All that are
  set must agree; a stale `RANK=0` inherited from an outer launcher raises instead of making every rank rank 0.
- Aurora's `mpiexec` (PALS with PMIx) exports no world-size variable, so the world size comes from `mpi4py`, which the
  `frameworks` module ships. Without `mpi4py`, pass `--env WORLD_SIZE=<ranks>` to `mpiexec`.
- The rendezvous host is `MASTER_ADDR`, else the first host in `PBS_NODEFILE`.
- The port is `MASTER_PORT`, else derived from `PBS_JOBID`. Two `mpiexec` launches running at once inside one job
  need their own `MASTER_PORT`.

## Pitfalls

- **Lightning shuffles multi-GPU training data even with `shuffle=False`.** For training it replaces the sampler with
  `DistributedSampler(shuffle=True)`, so a multi-GPU run sees a different row order than a single-GPU run. To
  control the order, pass `use_distributed_sampler=False` and your own `DistributedSampler`.
- **Lightning's `FSDPStrategy` is not supported.** Use `FSDP2Strategy`.
- **DDP communication hooks** remain CUDA-only, as in Lightning.
- **`manual_backward` runs inside autocast; automatic optimization's backward does not.** In Lightning 2.6.5 the
  precision plugin's autocast context stays open for the whole `training_step`. A `self.manual_backward(loss)` call
  made there therefore runs its backward under BF16 autocast, while automatic optimization runs the backward after
  `training_step` returns, outside autocast.
  - Run the manual reference with
    `with torch.autocast(device_type=self.device.type, enabled=False):` around `manual_backward`.
- **On 2 Aurora nodes, FSDP2 plus a JIT-compiled Triton kernel can segfault at process exit.** All work finishes, then
  ranks fault inside Intel's OpenCL driver (`libigdrcl.so`) during libc exit. The launcher then kills any rank still
  running, so a rank 0 still saving or evaluating loses that work.
  - Seen with ALCF `frameworks` 2026.1.0 (torch 2.13.0a0). It happened in 3 of 3 two-node runs that combined
    `fully_shard` with a freshly compiled Triton kernel, with or without `destroy_process_group()`.
  - Each ingredient alone ran clean on 2 nodes: FSDP2 without Triton, and Triton without FSDP2 (all_reduce, no
    collective, no process group, cached kernel). One node also ran clean.
  - Triton kernels come in through, for example, `fla` or `torch.compile`.
  - Workaround: once a rank's outputs are written and closed, end it with `os._exit(0)`. That skips the exit handlers
    where the fault happens.
  - Upstream: [argonne-lcf/AuroraBugTracking#176](https://github.com/argonne-lcf/AuroraBugTracking/issues/176).

## Tested

| Check | Hardware | Result |
|---|---|---|
| Unit tests (63) | CPU, Python 3.10 and 3.12, torch 2.14.1+cpu, Lightning 2.6.6 | pass |
| FP32 parity (AF): DDP and FSDP2 vs 1 device | 4x A100-SXM4-40GB, torch 2.13.0+cu130, Lightning 2.6.5 | pass |
| FP32 parity (X1F): DDP and FSDP2 vs 1 device | 1 Aurora node, 12 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass |
| FP32 parity (X2F): DDP and FSDP2 vs 1 device | 2 Aurora nodes, 24 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass |
| FP32 parity (X4F): DDP and FSDP2 vs 1 device | 4 Aurora nodes, 48 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass |
| FP32 parity (X8F): DDP and FSDP2 vs 1 device | 8 Aurora nodes, 96 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass |
| BF16 parity, criterion v2 (A440) | 4x A100-SXM4-40GB, torch 2.13.0+cu130, Lightning 2.6.5 | pass; gate and invariants pass; 8 fault cases rejected |
| BF16 parity, criterion v2 (A3a) | 3x A100-SXM4-40GB, torch 2.13.0+cu130, Lightning 2.6.5 | pass; gate and invariants pass |
| BF16 parity, criterion v2 (X1B) | 1 Aurora node, 12 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; gate and invariants pass; 4 fault cases rejected |
| BF16 parity, criterion v2 (X2B) | 2 Aurora nodes, 24 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; gate and invariants pass |
| BF16 parity, criterion v2 (X4B) | 4 Aurora nodes, 48 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; gate and invariants pass; 4 fault cases rejected |
| BF16 parity, criterion v2 (X8B) | 8 Aurora nodes, 96 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; gate and invariants pass; 4 fault cases rejected |
| Checkpoint save and resume, FP32 (AC32) | 4x A100-SXM4-40GB, torch 2.13.0+cu130, Lightning 2.6.5 | pass; 12 faults rejected |
| Checkpoint save and resume, BF16 (ACB) | 4x A100-SXM4-40GB, torch 2.13.0+cu130, Lightning 2.6.5 | pass; 12 faults rejected |
| Checkpoint save and resume, FP32 (X1C32) | 1 Aurora node, 12 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; 12 faults rejected |
| Checkpoint save and resume, BF16 (X1CB) | 1 Aurora node, 12 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; 12 faults rejected |
| Checkpoint save and resume, FP32 (X2C32) | 2 Aurora nodes, 24 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; 12 faults rejected |
| Checkpoint save and resume, BF16 (X2CB) | 2 Aurora nodes, 24 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | pass; 12 faults rejected |

FP32 comparisons against the single-device reference, one run per cell:

| Cell | Arm | Ranks | Global batch | Result | Pre-clip norm rel | Post-clip norm rel | Update rel L2 | Frozen changed |
|---|---|---:|---:|---|---:|---:|---:|---|
| AF | ddp | 4 | 8 | pass | 2.40e-6 | 1.55e-7 | 4.17e-6 | 0 of 290 |
| AF | fsdp | 4 | 8 | pass | 2.93e-6 | 1.55e-7 | 4.10e-6 | 0 of 290 |
| X1F | ddp | 12 | 24 | pass | 3.18e-6 | 2.38e-7 | 4.46e-6 | 0 of 290 |
| X1F | fsdp | 12 | 24 | pass | 2.49e-6 | 2.38e-7 | 4.41e-6 | 0 of 290 |
| X2F | ddp | 24 | 48 | pass | 2.56e-6 | 1.56e-7 | 3.99e-6 | 0 of 290 |
| X2F | fsdp | 24 | 48 | pass | 2.36e-6 | 1.23e-7 | 4.06e-6 | 0 of 290 |
| X4F | ddp | 48 | 96 | pass | 3.26e-5 | 1.90e-7 | 9.09e-6 | 0 of 290 |
| X4F | fsdp | 48 | 96 | pass | 2.99e-5 | 1.90e-7 | 8.48e-6 | 0 of 290 |
| X8F | ddp | 96 | 192 | pass | 1.84e-6 | 1.60e-7 | 2.61e-6 | 0 of 290 |
| X8F | fsdp | 96 | 192 | pass | 1.73e-6 | 1.60e-7 | 2.62e-6 | 0 of 290 |

The parity check runs LoRA sequence classification: Qwen2.5-0.5B with a PEFT LoRA adapter (r=8, q/k/v/o) and a
2-class linear head on the last token, three ways: 1 device, DDP on N devices, and FSDP2 on N devices.
Each row has an int64 class label drawn uniformly from {0, 1}; cross-entropy is averaged over rows.
Seed, initial weights and rows per step are identical in all three, in fp32 with SGD and an active global-norm
clip. It requires every arm to match the 1-device reference within 1e-4 (relative) on:

- per-step gradient norms, before and after clipping;
- the 5-step weight update;

and to leave every frozen weight unchanged. `compare` fails unless all three arms are present and the distributed arms ran
on more than one device, and an arm refuses to overwrite an existing result.

Limits:

- The default check is FP32. BF16 results are under
  [BF16 parity](#bf16-parity). Its single-device `ref` arm records every Linear site (backbone, LoRA A/B
  and head), checking BF16 outputs, FP32 parameters and the expected autocast state at every step.
- Aurora parity validation covers one run per cell on 1, 2, 4 and 8 nodes; checkpoint validation covers 1 and 2
  nodes. More than 8 nodes and a job spread over more than one chassis remain outside that coverage.

## BF16 parity

The `32-true` test is the default. Select `--precision bf16-mixed` to run the BF16 test in
`examples/bf16_parity.py` through the same `parity_check.py` entry point. The scoring rule is **criterion
v2**. Hardware results below are measured, one run per cell.

| Arm | Devices | Computation |
|---|---|---|
| `ref32` | 1 | Full global batch, FP32, TF32 disabled |
| `ref` | 1 | Full global batch, BF16 autocast |
| `split` | 1 | Serial rank shards, BF16 autocast; unscaled FP32 shard gradients summed in rank order, divided once by N |
| `split_premul` | 1 | As `split`, but each shard gradient is multiplied by 1/N before the sum, as DDP's reducer does |
| `ddp` | N | BF16 autocast, one shard per DDP rank |
| `fsdp` | N | BF16 autocast, one shard per FSDP2 rank |

All arms are paired: seed 0, identical initial trainable weights and ordered rows, Qwen2.5-0.5B with a PEFT LoRA
adapter (r=8, q/k/v/o) and a 2-class linear head on the last token, alpha=16, SGD lr=1, five optimizer steps,
and global-norm clip 0.01. The default global batch is 2N. The split arms reproduce the strided
`DistributedSampler` shards, including their order. They do not use loss/N accumulation:
that changes BF16 backward rounding, especially at non-power-of-two world sizes. In this recipe, `lora_B` starts
from a seeded normal draw (std 0.01, seed 1) instead of PEFT's zeros, so that every trainable tensor has a
nonzero gradient and update at step 0; with zeros, every `lora_A` gradient and update at step 0 is exactly zero.

This tests op-level autocast with FP32 parameters and the default FP32 communication configuration.
FSDP parameter casting and compressed communication are outside this check. The report records the
effective FSDP policy and DDP reducer dtype/no-hook configuration; it does not instrument collective
wire buffers or replace the reducer.

### Criterion v2

Scored: DDP and FSDP2 against `split`, at **step 0 only**:

| Check | Rule |
|---|---|
| Pre-clip global gradient norm | relative error at most 1e-3, every rank |
| Each trainable tensor's gradient | L2 difference at most `1e-3 * reference_norm + 1e-12 * sqrt(numel)` |
| Update produced by step 0 (all trainables) | relative L2 difference at most 1e-3 |
| Each trainable tensor's step-0 update | as for gradients, plus `2 * eps_fp32 * norm(initial_tensor)` for endpoint quantization |

Validity gate: `split_premul` must pass the same scored checks against `split`. The two are correct averaging
orders on one device with no collectives. If they disagree, the scored checks cannot separate implementation
error from rounding in that run, and the verdict is **void**, not pass.

Invariants, on every arm and every step: metadata and world size; step count; bit-identical initial trainables;
frozen tensors unchanged; FP32 communication configuration; clip active and post-clip norm within 1e-4 of the
clip; present, finite FP32 gradients; every Linear at the requested dtype under the expected autocast state;
exact rows and rank shards. Synchronization: DDP parameters (after every update) and gradients (before clipping)
must be bitwise equal on every rank. FSDP2 has no replicas, so every trainable parameter and gradient must be a
sharded DTensor whose gathered full value is bitwise equal on every rank. For correctly sharded tensors the
gathered check holds almost by construction. It exists to catch a trainable left outside `fully_shard`, which
never syncs.

Diagnostics, never scored: per-step gaps for all five steps of DDP, FSDP2 and `split_premul` against `split`, and of
DDP and FSDP2 against `split_premul` (largest per-tensor gap, bitwise-equal tensors, fraction of bitwise-equal
elements); five-step updates; full-batch BF16/FP32 differences.

Verdict: **fail** if any invariant fails; otherwise **void** if the validity gate fails; otherwise **pass** only
if DDP and FSDP2 pass every scored check.

Negative controls (`--fault`, never on by default) must be rejected. `FAULTS=missing-rank:wrong-scale` in either
launcher runs them after the clean arms, each in `fault_<name>/` against the clean references, and
`DDP_FAULTS=fp32-backbone:local-adapter:scale-grad:skip-update` adds the four DDP-only controls.
The launchers pass their explicit fault list to `faults --expected-faults`; without that option the standalone
`parity_check.py faults OUTDIR --precision bf16-mixed` requires all six fault kinds (eight arm/fault cases).
Every expected case needs a complete report, passing clean results and reference gate, and a failed check
attributable to that fault. Missing or errored cases are reported as "not executed" and exit non-zero.

| Fault | Arms | What it does |
|---|---|---|
| `missing-rank` | ddp, fsdp | The last rank's loss is multiplied by 0, so its gradient never reaches the average |
| `wrong-scale` | ddp, fsdp | Every rank's loss is multiplied by N, so gradients are summed instead of averaged |
| `fp32-backbone` | ddp | Backbone forward runs without autocast |
| `local-adapter` | ddp | One LoRA tensor keeps its local, unsynchronized gradient |
| `scale-grad` | ddp | Gradients are multiplied by N after synchronization |
| `skip-update` | ddp | Ranks other than 0 skip the last update |

Norm clipping rescales every update to the same length, so a pure gradient scaling leaves the update almost
unchanged; the pre-clip norm and gradient checks are what reject `scale-grad` and `wrong-scale`.

```bash
# Same worker and ordinary resource/account options; third argument selects precision, fourth the world size.
sbatch -A <account> -q debug examples/perlmutter.sbatch <python> <fresh-outdir> bf16-mixed 4
# Aurora keeps the existing default unless PRECISION is supplied.
qsub -A <project> -q debug -v PY=...,OUT=...,MODEL=...,PRECISION=bf16-mixed[,FAULTS=missing-rank:wrong-scale] examples/aurora.pbs
```

Every arm needs a fresh output namespace; partial files are also protected from overwrite.
`compare --precision bf16-mixed` requires all six arms and writes `bf16_report.json`.

### Validation matrix

Each cell below is one run, using seed 0, two rows per rank and all six clean arms. The A100 four-rank
cell also ran all four DDP-only controls. All A100 cells shared one interactive allocation on one node;
the three-rank cell used three of its four GPUs.

| Cell | Hardware | Ranks | Global batch | Verdict | Invariants |
|---|---|---:|---:|---|---|
| A100 n4 | 4x A100-SXM4-40GB, torch 2.13.0+cu130, Lightning 2.6.5 | 4 | 8 | pass | all pass |
| A100 n3 | 3x A100-SXM4-40GB, torch 2.13.0+cu130, Lightning 2.6.5 | 3 | 6 | pass | all pass |
| Aurora 1 node | 1 Aurora node, 12 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | 12 | 24 | pass | all pass |
| Aurora 2 nodes | 2 Aurora nodes, 24 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | 24 | 48 | pass | all pass |

Scored step-0 comparisons against `split`; `gate` is `split_premul`. Per-tensor normalized gaps divide the
L2 difference by the larger of the reference norm and the recorded absolute floor. Acceptance uses the
additive bounds in [Criterion v2](#criterion-v2).

| Cell | Arm | Pre-clip norm rel | Max tensor gradient gap | Update rel L2 | Max tensor update gap |
|---|---|---:|---:|---:|---:|
| A100 n4 | gate | 0 | 0 | 0 | 0 |
| A100 n4 | ddp | 0 | 1.93e-10 | 0 | 0 |
| A100 n4 | fsdp | 0 | 1.69e-10 | 5.63e-7 | 2.01e-5 |
| A100 n3 | gate | 7.20e-9 | 3.47e-7 | 5.79e-7 | 2.05e-5 |
| A100 n3 | ddp | 7.19e-9 | 3.47e-7 | 6.15e-7 | 2.80e-5 |
| A100 n3 | fsdp | 2.58e-9 | 8.17e-8 | 4.21e-7 | 1.76e-5 |
| Aurora 1 node | gate | 3.98e-8 | 1.08e-7 | 6.18e-7 | 3.46e-5 |
| Aurora 1 node | ddp | 8.34e-9 | 7.90e-8 | 5.63e-7 | 5.68e-5 |
| Aurora 1 node | fsdp | 0 | 0 | 0 | 0 |
| Aurora 2 nodes | gate | 3.61e-8 | 1.16e-7 | 5.92e-7 | 7.54e-5 |
| Aurora 2 nodes | ddp | 1.60e-8 | 7.89e-8 | 3.96e-7 | 4.66e-5 |
| Aurora 2 nodes | fsdp | 5.40e-15 | 1.83e-8 | 0 | 0 |

Diagnostic: largest per-tensor gradient gap against `split` at steps 0 to 4.

| Cell | ddp | fsdp | split_premul |
|---|---|---|---|
| A100 n4 | 1.9e-10, 8.2e-9, 1.3e-9, 2.1e-9, 1.9e-9 | 1.7e-10, 8.2e-9, 1.3e-9, 2.1e-9, 4.2e-9 | 0, 0, 0, 0, 0 |
| A100 n3 | 3.5e-7, 7.7e-8, 8.0e-8, 6.2e-8, 7.7e-8 | 8.2e-8, 5.6e-8, 5.9e-8, 7.0e-8, 5.9e-8 | 3.5e-7, 6.8e-8, 5.5e-8, 7.1e-8, 5.7e-8 |
| Aurora 1 node | 7.9e-8, 9.0e-8, 1.1e-7, 8.2e-8, 7.3e-8 | 0, 0, 0, 0, 0 | 1.1e-7, 1.5e-7, 3.6e-7, 1.1e-7, 8.6e-8 |
| Aurora 2 nodes | 7.9e-8, 9.0e-8, 1.2e-7, 9.1e-8, 1.6e-7 | 1.8e-8, 1.3e-8, 2.8e-8, 5.0e-9, 8.3e-9 | 1.2e-7, 1.2e-7, 2.0e-7, 1.2e-7, 1.5e-7 |

| Cell | Fault | Arm | Status | Rejecting evidence | Pre-clip norm rel | Max tensor gradient gap | Tensor gradients over | Tensor updates over |
|---|---|---|---|---|---:|---:|---:|---:|
| A100 n4 | missing-rank | ddp | rejected | pre_norm, gradients | 0.238 | 0.526 | 194/194 | 194/194 |
| A100 n4 | missing-rank | fsdp | rejected | pre_norm, gradients | 0.238 | 0.526 | 194/194 | 194/194 |
| A100 n4 | wrong-scale | ddp | rejected | pre_norm, gradients | 3 | 3 | 194/194 | 0/194 |
| A100 n4 | wrong-scale | fsdp | rejected | pre_norm, gradients | 3 | 3 | 194/194 | 0/194 |
| Aurora 1 node | missing-rank | ddp | rejected | pre_norm, gradients | 0.0469 | 0.566 | 194/194 | 194/194 |
| Aurora 1 node | missing-rank | fsdp | rejected | pre_norm, gradients | 0.0469 | 0.566 | 194/194 | 194/194 |
| Aurora 1 node | wrong-scale | ddp | rejected | pre_norm, gradients | 11 | 11.1 | 194/194 | 140/194 |
| Aurora 1 node | wrong-scale | fsdp | rejected | pre_norm, gradients | 11 | 11.1 | 194/194 | 140/194 |

DDP-only controls in the A100 four-rank cell:

| DDP-only fault (A440) | Status | Rejecting evidence | Supporting observations | Pre-clip norm rel | Max tensor gradient gap | Update rel L2 |
|---|---|---|---|---:|---:|---:|
| fp32-backbone | rejected | dtype | 360/361 invalid Linear sites per rank and step; 20 rank-step observations | 3.37e-3 | 0.265 | 0.0124 |
| local-adapter | rejected | synchronized | gradient replica disagreements 15/20; parameter disagreements 15/20 rank-step observations | 1.24e-6 | 1.32 | 1.19e-3 |
| scale-grad | rejected | pre_norm, gradients | tensor gradients over 194/194; tensor updates over 0/194 | 3 | 3 | 0 |
| skip-update | rejected | synchronized | gradient replica disagreements 0/20; parameter disagreements 3/20 rank-step observations | 0 | 1.93e-10 | 0 |

### Larger-node validation

The larger-node matrix uses the same recipe and limits, with both loss-level faults in each run, one run per
cell. Results below come from the saved reports.

| Cell | Hardware | Ranks | Global batch | Verdict | Invariants |
|---|---|---:|---:|---|---|
| Aurora 4 nodes | 4 Aurora nodes, 48 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | 48 | 96 | pass | all pass |
| Aurora 8 nodes | 8 Aurora nodes, 96 tiles, Max 1550, torch 2.13.0a0 (ALCF `frameworks` 2026.1.0), Lightning 2.6.5 | 96 | 192 | pass | all pass |

Scored step-0 comparisons against `split`; `gate` is `split_premul`. Per-tensor normalized gaps divide the
L2 difference by the larger of the reference norm and the recorded absolute floor. Acceptance uses the
additive bounds in [Criterion v2](#criterion-v2).

| Cell | Arm | Pre-clip norm rel | Max tensor gradient gap | Update rel L2 | Max tensor update gap |
|---|---|---:|---:|---:|---:|
| Aurora 4 nodes | gate | 1.87e-8 | 2.05e-7 | 9.32e-7 | 5.58e-5 |
| Aurora 4 nodes | ddp | 6.88e-9 | 1.58e-7 | 5.84e-7 | 5.09e-5 |
| Aurora 4 nodes | fsdp | 1.00e-12 | 1.62e-8 | 1.42e-12 | 4.15e-10 |
| Aurora 8 nodes | gate | 6.79e-8 | 3.09e-7 | 7.80e-7 | 6.56e-5 |
| Aurora 8 nodes | ddp | 9.12e-9 | 9.87e-8 | 3.87e-7 | 6.56e-5 |
| Aurora 8 nodes | fsdp | 1.19e-11 | 4.24e-8 | 4.66e-8 | 3.82e-6 |

Diagnostic: largest per-tensor gradient gap against `split` at steps 0 to 4.

| Cell | ddp | fsdp | split_premul |
|---|---|---|---|
| Aurora 4 nodes | 1.6e-7, 1.1e-7, 1.4e-7, 1.2e-7, 9.2e-8 | 1.6e-8, 3.3e-8, 8.4e-9, 4.0e-8, 8.9e-9 | 2.0e-7, 2.2e-7, 3.2e-7, 2.1e-7, 2.6e-7 |
| Aurora 8 nodes | 9.9e-8, 8.4e-8, 1.1e-7, 8.6e-8, 1.0e-7 | 4.2e-8, 2.4e-8, 1.6e-8, 1.4e-8, 1.6e-8 | 3.1e-7, 2.4e-7, 2.8e-7, 2.2e-7, 2.9e-7 |

| Cell | Fault | Arm | Status | Rejecting evidence | Pre-clip norm rel | Update rel L2 | Max tensor gradient gap | Tensor gradients over | Tensor updates over |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Aurora 4 nodes | missing-rank | ddp | rejected | pre_norm, gradients | 0.101 | 0.0751 | 0.486 | 194/194 | 194/194 |
| Aurora 4 nodes | missing-rank | fsdp | rejected | pre_norm, gradients | 0.101 | 0.0751 | 0.486 | 194/194 | 194/194 |
| Aurora 4 nodes | wrong-scale | ddp | rejected | pre_norm, gradients | 47 | 2.14e-3 | 47.5 | 194/194 | 129/194 |
| Aurora 4 nodes | wrong-scale | fsdp | rejected | pre_norm, gradients | 47 | 2.14e-3 | 47.5 | 194/194 | 129/194 |
| Aurora 8 nodes | missing-rank | ddp | rejected | pre_norm, gradients | 0.0412 | 0.034 | 0.341 | 194/194 | 194/194 |
| Aurora 8 nodes | missing-rank | fsdp | rejected | pre_norm, gradients | 0.0412 | 0.034 | 0.341 | 194/194 | 194/194 |
| Aurora 8 nodes | wrong-scale | ddp | rejected | pre_norm, gradients | 95 | 1.05e-3 | 95.9 | 194/194 | 88/194 |
| Aurora 8 nodes | wrong-scale | fsdp | rejected | pre_norm, gradients | 95 | 1.05e-3 | 95.9 | 194/194 | 88/194 |

## Checkpoints

`examples/checkpoint_check.py` kills a fit after an end-of-epoch checkpoint and resumes it in a new process, for 1
device, `XPUDDPStrategy`, and `FSDP2Strategy` with its sharded (torch.distributed.checkpoint directory) and full-state
file (`save_distributed_checkpoint=False`) formats. It runs LoRA sequence classification: Qwen2.5-0.5B with a PEFT
LoRA adapter (r=8, q/k/v/o) and a 2-class linear head on the last token (AdamW, a per-step LR schedule, global-norm
clipping) for 2 epochs of 3 steps. Each strategy runs these roles, each in its own
process: an uninterrupted fit, its twin, a fit killed one step past its epoch-1 checkpoint, a resume of that
checkpoint, and a reload of the checkpoint the resume writes. FSDP2 also resumes at half the world size, and the DDP
and full-state FSDP2 checkpoints are loaded on 1 device. Before every load, four frozen tensors are changed, so only the
checkpoint can restore them.

It requires, bit for bit:

- the restored state to equal the state captured inside the save: trainable tensors, AdamW state and its devices,
  param groups and LR scheduler on every rank, digests of every frozen tensor, and the loop position;
- the first resumed step's loss to equal the killed fit's loss for the same step, the LR of every step to match the
  uninterrupted fit, and the same rows to be trained at every step.

Losses, gradient norms and weights after the first resumed step are reported, not required to match. DDP rebuilds
its gradient buckets on a process's second iteration, so a correct resume reduces its first step in a different
order than an uninterrupted fit. Three faults injected after the restore must be rejected: dropped optimizer state, a
reset LR scheduler, and a perturbed weight.

Reported continuation measures loss over the three resumed steps and the final weight update of every trainable
tensor, including the comparison after resharding. Required restore, continuation, reshard and export checks
passed, and all injected faults were rejected, one run per cell.

| Run | Ranks | Global batch | Reshard ranks | Required restore, continuation, reshard, export; faults |
|---|---:|---:|---|---|
| AC32: A100 n4, FP32 | 4 | 8 | 4 → 2 | pass; 12 faults rejected |
| ACB: A100 n4, BF16 | 4 | 8 | 4 → 2 | pass; 12 faults rejected |
| X1C32: Aurora 1 node, FP32 | 12 | 24 | 12 → 6 | pass; 12 faults rejected |
| X1CB: Aurora 1 node, BF16 | 12 | 24 | 12 → 6 | pass; 12 faults rejected |
| X2C32: Aurora 2 nodes, FP32 | 24 | 48 | 24 → 12 | pass; 12 faults rejected |
| X2CB: Aurora 2 nodes, BF16 | 24 | 48 | 24 → 12 | pass; 12 faults rejected |

Reported continuation and reshard gaps, not required for acceptance. Loss and pre-clip norm gaps are the
largest relative gaps over resumed steps; update and Adam moment gaps are the largest per-tensor relative
gaps at the end of continuation, against the uninterrupted fit. Exact zeros are shown as `0`. A resharded
resume runs at half the world size with the same global batch, so each rank's forward covers twice as many rows;
BF16 kernels can depend on batch shape, so BF16 `fsdp_reshard` gaps are not expected to be small.

| Run | Arm | Reported loss gap | Reported pre-clip norm gap | Reported update gap | Reported Adam first-moment gap | Reported Adam second-moment gap |
|---|---|---:|---:|---:|---:|---:|
| AC32: A100 n4, FP32 | single | 0 | 0 | 0 | 0 | 0 |
| AC32: A100 n4, FP32 | ddp | 5.94e-7 | 3.81e-7 | 5.49e-6 | 1.48e-5 | 7.92e-6 |
| AC32: A100 n4, FP32 | fsdp | 0 | 0 | 0 | 0 | 0 |
| AC32: A100 n4, FP32 | fsdp_file | 0 | 0 | 0 | 0 | 0 |
| AC32: A100 n4, FP32 | fsdp_reshard | 6.80e-6 | 6.07e-6 | 1.68e-5 | 1.59e-5 | 1.98e-5 |
| ACB: A100 n4, BF16 | single | 0 | 0 | 0 | 0 | 0 |
| ACB: A100 n4, BF16 | ddp | 0 | 4.81e-13 | 7.44e-9 | 1.44e-8 | 1.58e-9 |
| ACB: A100 n4, BF16 | fsdp | 0 | 0 | 0 | 0 | 0 |
| ACB: A100 n4, BF16 | fsdp_file | 0 | 0 | 0 | 0 | 0 |
| ACB: A100 n4, BF16 | fsdp_reshard | 0.0546 | 0.0403 | 0.135 | 0.306 | 0.219 |
| X1C32: Aurora 1 node, FP32 | single | 0 | 0 | 0 | 0 | 0 |
| X1C32: Aurora 1 node, FP32 | ddp | 5.68e-7 | 6.19e-7 | 1.12e-6 | 4.37e-6 | 4.84e-6 |
| X1C32: Aurora 1 node, FP32 | fsdp | 0 | 0 | 0 | 0 | 0 |
| X1C32: Aurora 1 node, FP32 | fsdp_file | 0 | 0 | 0 | 0 | 0 |
| X1C32: Aurora 1 node, FP32 | fsdp_reshard | 1.98e-6 | 1.58e-6 | 1.29e-5 | 1.09e-5 | 1.43e-5 |
| X1CB: Aurora 1 node, BF16 | single | 0 | 0 | 0 | 0 | 0 |
| X1CB: Aurora 1 node, BF16 | ddp | 0 | 5.23e-9 | 1.03e-8 | 3.45e-8 | 1.98e-8 |
| X1CB: Aurora 1 node, BF16 | fsdp | 0 | 0 | 0 | 0 | 0 |
| X1CB: Aurora 1 node, BF16 | fsdp_file | 0 | 0 | 0 | 0 | 0 |
| X1CB: Aurora 1 node, BF16 | fsdp_reshard | 0.0227 | 8.94e-3 | 0.0204 | 0.0442 | 0.0734 |
| X2C32: Aurora 2 nodes, FP32 | single | 0 | 0 | 0 | 0 | 0 |
| X2C32: Aurora 2 nodes, FP32 | ddp | 1.41e-7 | 1.99e-7 | 2.70e-6 | 6.36e-6 | 3.98e-6 |
| X2C32: Aurora 2 nodes, FP32 | fsdp | 0 | 0 | 0 | 0 | 0 |
| X2C32: Aurora 2 nodes, FP32 | fsdp_file | 0 | 0 | 0 | 0 | 0 |
| X2C32: Aurora 2 nodes, FP32 | fsdp_reshard | 6.69e-7 | 1.29e-7 | 1.21e-5 | 2.39e-5 | 1.64e-5 |
| X2CB: Aurora 2 nodes, BF16 | single | 0 | 0 | 0 | 0 | 0 |
| X2CB: Aurora 2 nodes, BF16 | ddp | 0 | 2.75e-9 | 1.22e-7 | 4.81e-8 | 6.59e-8 |
| X2CB: Aurora 2 nodes, BF16 | fsdp | 0 | 0 | 0 | 0 | 0 |
| X2CB: Aurora 2 nodes, BF16 | fsdp_file | 0 | 0 | 0 | 0 | 0 |
| X2CB: Aurora 2 nodes, BF16 | fsdp_reshard | 2.68e-3 | 2.80e-3 | 0.0126 | 0.047 | 0.0472 |

Results record the checkpoints each role wrote relative to OUTDIR, so `compare` also runs on a copy of it.

Known to fail in Lightning 2.6.5, on any hardware:

- Resuming a mid-epoch checkpoint with an ordinary `DataLoader` restarts the epoch's data from its first batch. Resume
  from epoch boundaries, or use a stateful data loader.
- `python -m lightning.pytorch.utilities.consolidate_checkpoint` fails on `FSDP2Strategy` checkpoints
  (`KeyError: 'model'`); it expects the FSDP1 format. To get a single file, save with
  `save_distributed_checkpoint=False`.
- Resuming a 1-device or DDP checkpoint under `FSDP2Strategy` fails with "Missing optimizer state" when the optimizer
  holds only the trainable parameters, as with LoRA.
- `Trainer.fit(..., weights_only=...)` is not forwarded on the restore path `FSDP2Strategy` uses (found in the source,
  not run).

Not tested: an interrupted checkpoint write, `16-mixed` (its gradient scaler state), `ModelCheckpoint` with a
monitored metric, `save_top_k` or `save_last`, and runs that draw random numbers during training (Lightning does
not checkpoint RNG state).

## Open items

The release hardware matrix is complete: FP32 and BF16 parity on A100 and Aurora, including the listed
negative controls, and FP32/BF16 checkpoint checks on A100 and on one and two Aurora nodes. Each cell is
one run. Additional untested scope is listed under the parity limits and [Checkpoints](#checkpoints).

## Development

```bash
pip install -e ".[test]"
pytest                                   # CPU only
sbatch -A <account> -q debug examples/perlmutter.sbatch <python> <outdir>    # GPU parity (Slurm)
qsub -A <project> -q debug -v PY=...,OUT=...,MODEL=... examples/aurora.pbs   # XPU parity
sbatch -A <account> -q debug examples/checkpoint.sbatch <python> <outdir>    # checkpoints (STAGE=train, then faults)
qsub -A <project> -q debug -v PY=...,OUT=...,MODEL=... examples/checkpoint.pbs   # checkpoints on XPU
```

## Contributing

Issues and pull requests are welcome. For new features or substantial behavioral changes, open an issue first so the
approach can be agreed before you write it.

- Branch from `main` and keep each pull request to one change.
- Before you push, run what CI runs (Python 3.10 and 3.12, CPU only):

  ```bash
  ruff check .
  ruff format --check .
  pytest -q
  ```

- A behavior change needs a CPU test that fails without it.
- A change to the accelerator, a strategy, precision handling or checkpointing also needs a hardware run. Run the
  matching example under [Development](#development) on the devices it affects, and paste the result into the pull
  request: accelerator type, node count, world size, torch and Lightning versions, and the pass or fail lines. If
  you have no access to Aurora or another XPU system, say so; the maintainer will run it there.
- Do not loosen a parity or checkpoint threshold to make a check pass. Report the failure in the pull request.
- Update [Tested](#tested), [Pitfalls](#pitfalls) or [Open items](#open-items) when the change moves what has
  been verified.
- Write commit subjects in the imperative ("Fix …", "Add …"), with the reason in the body.

Contributions are licensed under Apache-2.0, the project's license.

## Maintenance

Best effort, by one maintainer. The Lightning upper bound is raised only after the tests and the parity check pass
on the new release. If Lightning gains native XPU support, this package will be deprecated in its favor.

## License

Apache-2.0.
