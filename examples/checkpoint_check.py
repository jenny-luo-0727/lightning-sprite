"""Checkpoint check: a fit killed after a checkpoint and resumed in a new process must restore exactly what it saved.

Strategy arms:
  single     1 device (CUDA: Lightning's single-device strategy; XPU: trainer_kwargs("single")).
  ddp        N devices, XPUDDPStrategy; rank 0 writes one checkpoint file.
  fsdp       N devices, FSDP2Strategy with its default sharded checkpoint (a torch.distributed.checkpoint directory).
  fsdp_file  N devices, FSDP2Strategy(save_distributed_checkpoint=False); rank 0 writes one full-state file.
Roles, each in its own process:
  full     2 epochs of 3 steps, uninterrupted, no checkpointing.
  twin     full again (run-to-run, reported).
  save     ModelCheckpoint at the end of each epoch; killed by an exception before the second step of epoch 2, so the
           process has taken step 3, one optimizer step past the epoch-1 checkpoint, when it dies.
  resume   a new process: Trainer.fit(ckpt_path=<save's epoch-1 checkpoint>) with the same ModelCheckpoint, which
           writes the epoch-2 checkpoint.
  reload   a new process: fit from resume's epoch-2 checkpoint with max_epochs 3, killed before its first step.
  reshard  (fsdp) resume of fsdp's epoch-1 checkpoint at another world size, same global batch, no checkpointing.
  export   (single, --source ddp or fsdp_file) Trainer.validate(ckpt_path=<the source's epoch-1 checkpoint>).
Every role that loads a checkpoint first adds 1 to four frozen tensors (the restore witness), so only the checkpoint
can put them back. Fault controls rerun resume with a defect injected after the restore:
  drop-optimizer  clear the restored AdamW state.   drop-scheduler  put the LR scheduler back to step 0.
  perturb-weight  scale the restored head weight by 1 + 2**-6, a change that survives a bfloat16 cast; the
                  injection must change the model's output under the training precision to count.

Held fixed: Qwen2.5-0.5B with a PEFT LoRA adapter (r=8, q/k/v/o) and a 2-class linear head on the last token
inside the sharded root (PEFT's zero lora_B); seed 0; global batch B, B/N rows per rank, drawn by an explicit
DistributedSampler(shuffle=True, seed 1234) whose epoch Lightning sets, so a resume that lost the epoch counter
would replay epoch 1's
order (at any world size, the rows of one step are the same block of the permutation); AdamW lr 1e-3, weight decay
0.01, linear per-step decay; global-norm clip 1.0; no dropout and no DataLoader workers, so no random draw reaches
training (Lightning does not checkpoint RNG state). On XPU, torch.use_deterministic_algorithms(True);
CUDA keeps its defaults. Collectives time out after 10 minutes.

Pass criteria (criterion A: exact wherever the comparison does not depend on cross-process reduction arithmetic),
fixed before any hardware run, for each strategy arm:
  R  exact restore, for resume against save's epoch-1 snapshot and for reload against resume's epoch-2 snapshot:
     every trainable tensor and AdamW state tensor (full tensors); on every rank, the optimizer's param groups (members
     in order and hyperparameters), the scheduler state, interval and frequency, the device of each local state
     tensor, and the SHA-256 of the rank's local trainable, optimizer and frozen tensors (under DDP, the same on every
     rank); the SHA-256 of every full frozen tensor and persistent buffer; on_train_start at the checkpoint's global
     step and the next epoch.
     Sequences compare by value: torch's full-state optimizer load returns AdamW's betas as a list, not a tuple.
     reshard: rank-invariant entries against the snapshot's rank 0, no per-rank digests. export: trainable tensors
     and frozen digests against save's snapshot.
  N  not vacuous: save took exactly one checkpoint (epoch 0, global step 3, at the expected path) and was killed at
     global step 4; resume wrote exactly the epoch-1 checkpoint; the witness changed every witness tensor before each
     load; every restored trainable tensor differs from its initial value and has AdamW step 3 and nonzero moments;
     the restored LR is below the base LR; resume trained global steps 3-5, full and twin steps 0-5.
  C  exact continuation: resume's step-3 loss on every rank equals save's (both are a forward pass from the same
     state); the LR of every step of every role equals full's; resume ends with AdamW step 6 for every tensor.
  D  every step of every role trained on full's rows: per rank and in order at full's world size, as a multiset at
     another.
  F  frozen digests unchanged within every role that trains.
  W  single on 1 device, the others on more than 1; every role of an arm on one world size and setup; every result
     written by the code (this script, parity_check.py and the lightning_sprite package) that compares it.
reshard is scored on R, N, C (LR and AdamW steps), D, F and W (a world other than save's). Reported, not scored:
twin against full; save's steps 0-3, resume's steps 3-5 and final state against full; resume against save after
step 3 (loss, pre-clip norm, update, AdamW moments: bit-identical, or the largest relative gap). DDP rebuilds its
gradient buckets on each process's second iteration, so a correct resume can reduce step 3 in another order than an
uninterrupted run. A fault is rejected when R and at least one C check fail and the injection took effect; it counts
only when its arm passes without the fault. A run writes its results only after every rank has finished it.

Usage:
  python checkpoint_check.py run ARM ROLE OUTDIR --device {cuda,xpu,cpu} [--model PATH] [--world N] [--nodes M]
      [--global-batch B] [--precision {32-true,bf16-mixed}] [--fault F] [--source {ddp,fsdp_file}]
  python checkpoint_check.py compare OUTDIR
--world is the number of processes of this launch (1 for single); B must be the same for every launch into OUTDIR.
Launch the multi-device roles with one process per device (srun on Slurm, mpiexec on PBS/PALS).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from parity_check import SEQ, VOCAB_CAP, Net, _full, _is_dtensor, _local
from torch import nn

ARMS = ("single", "ddp", "fsdp", "fsdp_file")
ROLES = ("full", "twin", "save", "resume", "reload", "reshard", "export")
LOADING = ("resume", "reload", "reshard", "export")
EXPORT_SOURCES = ("ddp", "fsdp_file")
FAULTS = ("none", "drop-optimizer", "drop-scheduler", "perturb-weight")
EPOCHS, STEPS_PER_EPOCH = 2, 3
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
LR, WEIGHT_DECAY, CLIP = 1e-3, 0.01, 1.0
# save is killed before batch KILL_BATCH of epoch KILL_EPOCH: one optimizer step past the epoch-0 checkpoint.
KILL_EPOCH, KILL_BATCH = 1, 1
PERTURB = 2.0**-6
DATA_SEED, SAMPLER_SEED = 1234, 1234
TIMEOUT = timedelta(minutes=10)
# Witness tensors, by name suffix among the frozen parameters: the first match of each, except q_proj (the last).
WITNESS = (
    "embed_tokens.weight",
    "layers.0.mlp.down_proj.weight",
    "self_attn.q_proj.base_layer.weight",
    "model.norm.weight",
)
ROW_FAMILIES = ("loss", "pre_clip_norm")
# Entries captured on every rank whose values do not depend on the rank or the world size.
RANK_INVARIANT = ("groups", "scheduler", "scheduler_config", "placement")


class Killed(RuntimeError):
    """Raised inside a fit in place of a job killed after its checkpoint."""


def code_sha256():
    """SHA-256 of the code that writes and scores results: this script, parity_check.py and lightning_sprite."""
    import lightning_sprite

    here = Path(__file__).resolve().parent
    files = [here / "checkpoint_check.py", here / "parity_check.py"]
    files += sorted(Path(lightning_sprite.__file__).resolve().parent.glob("*.py"))
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return digest.hexdigest()


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def digest(t):
    """SHA-256 of a tensor's dtype, shape and bytes."""
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256(repr((str(t.dtype), tuple(t.shape))).encode())
    h.update(memoryview(t.view(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def frozen_tensors(module):
    for n, p in module.named_parameters():
        if not p.requires_grad:
            yield n, p.detach()
    for prefix, sub in module.named_modules():
        for name, buffer in sub._buffers.items():
            if buffer is not None and name not in sub._non_persistent_buffers_set:
                yield f"{prefix}.{name}" if prefix else name, buffer


def frozen_state(module):
    """Digests of every frozen parameter and persistent buffer: as full tensors (on rank 0; a collective on FSDP2), and
    one digest per rank of its local copies."""
    full, local = {}, hashlib.sha256()
    for n, t in frozen_tensors(module):
        if _is_dtensor(t):
            whole = t.full_tensor()
            if is_rank0():
                full[n] = digest(whole)
            del whole
            local_digest = digest(t.to_local())
        else:
            local_digest = digest(t)
            full[n] = local_digest
        local.update(f"{n}:{local_digest};".encode())
    return {"full": full if is_rank0() else {}, "ranks": gather(local.hexdigest())}


def grad_norm(parameters):
    """Global gradient norm, with rank partials gathered and summed in rank order: equal gradients, equal norms."""
    sharded = replicated = 0.0
    for p in parameters:
        if p.grad is not None:
            if _is_dtensor(p.grad):
                sharded += float(_local(p.grad).double().square().sum())
            else:
                replicated += float(p.grad.double().square().sum())
    return math.sqrt(math.fsum(gather(sharded)) + replicated)


def trainable_state(module):
    return {n: _full(p.detach()).cpu().clone() for n, p in module.named_parameters() if p.requires_grad}


def optimizer_state(module, optimizer):
    """Every parameter's optimizer state as full CPU tensors keyed by parameter name."""
    names = {id(p): n for n, p in module.named_parameters()}
    state = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            entry = optimizer.state.get(p, {})
            state[names[id(p)]] = {
                k: _full(v).cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in sorted(entry.items())
            }
    return state


def local_groups(module, optimizer):
    names = {id(p): n for n, p in module.named_parameters()}
    return [
        {"params": [names[id(p)] for p in g["params"]], **{k: copy.deepcopy(v) for k, v in g.items() if k != "params"}}
        for g in optimizer.param_groups
    ]


def local_placement(module, optimizer):
    """The device type of every trainable parameter and of each of its local optimizer state tensors."""
    out = {}
    for n, p in module.named_parameters():
        if p.requires_grad:
            entry = optimizer.state.get(p, {})
            out[n] = {"param": _local(p).device.type} | {
                k: _local(v).device.type for k, v in entry.items() if torch.is_tensor(v)
            }
    return out


def local_replica(module, optimizer):
    """SHA-256 over the rank's local trainable tensors and optimizer state."""
    h = hashlib.sha256()
    for n, p in module.named_parameters():
        if p.requires_grad:
            h.update(f"{n}:{digest(_local(p))};".encode())
            for k, v in sorted(optimizer.state.get(p, {}).items()):
                h.update(f"{k}:{digest(_local(v)) if torch.is_tensor(v) else repr(v)};".encode())
    return h.hexdigest()


def make_probe(arm, role, model_name, global_batch, fault):
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    from lightning_sprite import shard_model

    class Probe(L.LightningModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = Net(model_name)
            self.init = {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}
            self.rows, self.saves, self.start, self.post_step3, self.end = [], [], None, None, None
            self.killed_at, self.fault_tensor, self.fault_effect, self.witness = None, None, None, {}

        def configure_model(self):
            if arm in ("fsdp", "fsdp_file"):
                shard_model(self.net, self.device_mesh["data_parallel"])
            if role in LOADING and not self.witness:
                self.witness = apply_witness(self)

        def dataset(self):
            g = torch.Generator().manual_seed(DATA_SEED)
            n = global_batch * STEPS_PER_EPOCH
            ids = torch.randint(0, VOCAB_CAP, (n, SEQ), generator=g)
            y = torch.randint(0, 2, (n,), generator=g)
            return TensorDataset(torch.arange(n), ids, y)

        def train_dataloader(self):
            world, rank = self.trainer.world_size, self.trainer.global_rank
            dataset = self.dataset()
            if len(dataset) % world or global_batch % world:
                raise ValueError(f"{len(dataset)} rows and batch {global_batch} must divide over {world} ranks")
            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=SAMPLER_SEED)
            return DataLoader(dataset, batch_size=global_batch // world, sampler=sampler)

        def val_dataloader(self):
            return DataLoader(self.dataset(), batch_size=global_batch)

        def training_step(self, batch, batch_idx):
            idx, ids, y = batch
            loss = nn.functional.cross_entropy(self.net(ids), y)
            losses = torch.atleast_1d(self.all_gather(loss.detach())).double()
            by_rank = self.all_gather(idx)
            by_rank = by_rank.unsqueeze(0) if by_rank.dim() == 1 else by_rank
            self.rows.append(
                dict(
                    global_step=self.trainer.global_step,
                    epoch=self.trainer.current_epoch,
                    batch_idx=batch_idx,
                    losses=losses.tolist(),
                    loss=float(losses.mean()),
                    rows_by_rank=by_rank.tolist(),
                )
            )
            return loss

        def validation_step(self, batch, batch_idx):
            pass

        def configure_optimizers(self):
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: 1 - k / TOTAL_STEPS)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        def on_before_optimizer_step(self, optimizer):
            # Lightning calls this before it clips.
            self.rows[-1]["pre_clip_norm"] = grad_norm([p for p in self.parameters() if p.requires_grad])
            self.rows[-1]["lr"] = optimizer.param_groups[0]["lr"]
            self.rows[-1]["lrs"] = gather([g["lr"] for g in optimizer.param_groups])

        def capture(self, training=True, frozen=True):
            """The state R compares. Every rank must call it: it gathers."""
            trainer = self.trainer
            out = dict(global_step=trainer.global_step, epoch=trainer.current_epoch, trainable=trainable_state(self))
            if frozen:
                out["frozen"] = frozen_state(self)
            if training:
                optimizer, config = trainer.optimizers[0], trainer.lr_scheduler_configs[0]
                local = dict(
                    groups=local_groups(self, optimizer),
                    scheduler=copy.deepcopy(config.scheduler.state_dict()),
                    scheduler_config=dict(interval=config.interval, frequency=config.frequency),
                    placement=local_placement(self, optimizer),
                )
                out.update({k: gather(v) for k, v in local.items()})
                out["optimizer"] = optimizer_state(self, optimizer)
                out["replicas"] = gather(local_replica(self, optimizer))
            return out

        def on_save_checkpoint(self, checkpoint):
            # Lightning builds the checkpoint on every rank, so the collectives in capture() are safe here.
            self.saves.append(dict(ckpt_epoch=checkpoint["epoch"], ckpt_global_step=checkpoint["global_step"]))
            self.saves[-1].update(self.capture())

        def on_train_start(self):
            if fault != "none":
                self.fault_tensor, self.fault_effect = inject(fault, self, self.trainer)
            self.start = self.capture()

        def on_train_batch_end(self, *_):
            if self.trainer.global_step == STEPS_PER_EPOCH + 1:
                self.post_step3 = self.capture(frozen=False)

        def on_train_end(self):
            self.end = self.capture()

        def on_validation_start(self):
            if role == "export":
                self.start = self.capture(training=False)

    return Probe()


def apply_witness(module):
    """Add 1 to each witness tensor before a load. Returns, per witness, whether any rank's values changed."""
    frozen = [n for n, p in module.named_parameters() if not p.requires_grad]
    changed = {}
    for i, suffix in enumerate(WITNESS):
        matches = [n for n in frozen if n.endswith(suffix)]
        if not matches:
            raise ValueError(f"no frozen parameter ends with {suffix}")
        name = matches[-1] if i == 2 else matches[0]
        p = module.get_parameter(name)
        before = _local(p.detach()).clone()
        with torch.no_grad():
            p.add_(1.0)
        changed[name] = any(gather(not torch.equal(before, _local(p.detach()))))
    return changed


def inject(fault, module, trainer):
    """Apply a fault after the restore. Returns the perturbed tensor's name (or None) and whether it took effect."""
    optimizer = trainer.optimizers[0]
    if fault == "drop-optimizer":
        had_state = len(optimizer.state) > 0
        optimizer.state.clear()
        return None, all(gather(had_state and len(optimizer.state) == 0))
    if fault == "drop-scheduler":
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        moved = scheduler.last_epoch > 0
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"]
        scheduler.last_epoch = 0
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        return None, all(gather(moved))
    # perturb-weight: it counts only if the model's output changes under the training precision.
    name = "net.head.weight"
    ids = module.dataset().tensors[1][:2].to(module.device)
    before = probe_output(module, ids, trainer)
    # Fetched after the forward: FSDP2 swaps an unsharded root's parameters in and out of the module.
    p = module.get_parameter(name)
    stored = _local(p.detach()).clone()
    with torch.no_grad():
        p.mul_(1 + PERTURB)
    changed = not torch.equal(stored, _local(p.detach()))
    after = probe_output(module, ids, trainer)
    return name, any(gather(changed)) and all(gather(not torch.equal(before, after)))


def probe_output(module, ids, trainer):
    """The model's output on ids under the training precision, leaving FSDP2 parameters sharded as training expects.

    FSDP2 does not reshard the root module after its forward (the backward would gather it again at once), so a
    forward outside training leaves the root's unsharded parameters registered until it is resharded here.
    """
    from torch.distributed.fsdp import FSDPModule

    with torch.no_grad(), trainer.precision_plugin.forward_context():
        out = module.net(ids).float().cpu()
    if isinstance(module.net, FSDPModule):
        module.net.reshard()
    return out


def trainer_device_args(arm, device, devices, nodes, precision):
    from lightning_sprite import FSDP2Strategy, XPUDDPStrategy, trainer_kwargs

    strategy_kwargs = {} if arm == "single" else {"timeout": TIMEOUT}
    if arm == "fsdp_file":
        strategy_kwargs["save_distributed_checkpoint"] = False
    if device == "xpu":
        strategy = {"single": "single", "ddp": "ddp"}.get(arm, "fsdp")
        return trainer_kwargs(strategy, devices=devices, num_nodes=nodes, precision=precision, **strategy_kwargs)
    if arm == "single":
        strategy = "auto"
    else:
        strategy = XPUDDPStrategy(**strategy_kwargs) if arm == "ddp" else FSDP2Strategy(**strategy_kwargs)
    return dict(accelerator=device, devices=devices, num_nodes=nodes, strategy=strategy, precision=precision)


def run_name(arm, role, fault, source):
    if role == "export":
        return f"{source}_export"
    return f"{arm}_{role}" + ("" if fault == "none" else f"_{fault}")


# The role whose checkpoint a loading role reads, and that checkpoint's epoch.
PRODUCERS = {"resume": ("save", 0), "reshard": ("save", 0), "export": ("save", 0), "reload": ("resume", 1)}
SETUP_KEYS = ("device", "global_batch", "precision", "model", "code_sha256", "torch", "lightning")


def checkpoint_source(outdir, arm, role, source):
    """The producing run's meta and the exact checkpoint path a loading role reads."""
    producer, epoch = PRODUCERS[role]
    owner = source if role == "export" else arm
    meta = json.loads((outdir / f"{owner}_{producer}.json").read_text())
    name = f"{owner}_ckpt/epoch{epoch}.ckpt"
    if name not in meta["written"]:
        raise FileNotFoundError(f"{owner}_{producer} did not write {name}: {meta['written']}")
    return meta, str(outdir / name)


def write_once(path, write):
    """Write through this process's own temporary file and publish it with a hard link, which fails if a result
    already exists, so overlapping writers can never replace or interleave each other's results."""
    tmp = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with tmp.open("xb") as stream:
        write(stream)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def run(arm, role, outdir, device, model_name, world, nodes, global_batch, precision, fault, source):
    import lightning as L
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint

    import lightning_sprite

    if (arm == "single") != (world == 1) or nodes < 1 or world % nodes or global_batch % world:
        raise ValueError("single needs --world 1, the others --world >= 2; world divisible by nodes, batch by world")
    if role == "reshard" and arm != "fsdp":
        raise ValueError("reshard applies to the fsdp arm only")
    if (role == "export") != (source is not None) or (role == "export" and arm != "single"):
        raise ValueError(f"export runs on the single arm, with --source one of {EXPORT_SOURCES}; --source only there")
    if fault != "none" and role != "resume":
        raise ValueError("faults apply to the resume role only")
    name = run_name(arm, role, fault, source)
    outdir.mkdir(parents=True, exist_ok=True)
    if any((outdir / f"{name}.{ext}").exists() for ext in ("json", "pt")):
        raise FileExistsError(f"{name} already exists in {outdir}; use a fresh output directory")
    ckpt_dir = outdir / f"{arm}_ckpt"
    saving = role == "save" or (role == "resume" and fault == "none")
    if role == "save" and ckpt_dir.exists():
        raise FileExistsError(f"{ckpt_dir} exists; use a fresh output directory")
    if role == "resume" and saving and (ckpt_dir / "epoch1.ckpt").exists():
        raise FileExistsError(f"{ckpt_dir / 'epoch1.ckpt'} exists; use a fresh output directory")
    setup = dict(
        device=device,
        global_batch=global_batch,
        precision=precision,
        model=model_name,
        code_sha256=code_sha256(),
        torch=torch.__version__,
        lightning=L.__version__,
    )
    ckpt_path = None
    if role in LOADING:
        produced, ckpt_path = checkpoint_source(outdir, arm, role, source)
        mismatch = {k: (produced[k], v) for k, v in setup.items() if produced[k] != v}
        if mismatch:
            raise ValueError(f"{name} must match the run that wrote its checkpoint on {mismatch}")
        same_world = produced["world_size"] == world
        if (role in ("resume", "reload") and not same_world) or (role == "reshard" and same_world):
            raise ValueError(
                f"resume and reload run on the writer's world size ({produced['world_size']}), reshard not"
            )

    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif device == "xpu":
        torch.use_deterministic_algorithms(True)

    class Kill(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if role == "reload" or (trainer.current_epoch == KILL_EPOCH and batch_idx == KILL_BATCH):
                pl_module.killed_at = trainer.global_step
                pl_module.end = pl_module.capture() if role == "save" else None
                raise Killed(f"killed at global step {trainer.global_step}")

    probe = make_probe(arm, role, model_name, global_batch, fault)
    callbacks = []
    if saving:
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="epoch{epoch}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=-1,
                save_on_train_epoch_end=True,
            )
        )
    if role in ("save", "reload"):
        callbacks.append(Kill())
    trainer = L.Trainer(
        **trainer_device_args(arm, device, world // nodes, nodes, precision),
        max_epochs=EPOCHS + (role == "reload"),
        gradient_clip_val=CLIP,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=saving,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=1 if role == "export" else 0,
        use_distributed_sampler=False,
    )
    killed = False
    try:
        if role == "export":
            trainer.validate(probe, ckpt_path=ckpt_path)
        else:
            trainer.fit(probe, ckpt_path=ckpt_path)
    except Killed:
        killed = True
    if killed != (role in ("save", "reload")):
        raise RuntimeError(f"{name}: expected killed={role in ('save', 'reload')}, got {killed}")
    # Results are written only once every rank has got here: a rank that failed earlier leaves no result behind.
    finished = gather(trainer.global_rank)
    if trainer.global_rank != 0:
        return
    # Relative to outdir, so a copied outdir still compares.
    written = sorted(p.relative_to(outdir).as_posix() for p in ckpt_dir.iterdir()) if saving else []
    hardware = {"cuda": torch.cuda.get_device_name, "xpu": lambda: torch.xpu.get_device_name()}
    meta = dict(
        arm=arm,
        role=role,
        fault=fault,
        source=source,
        fault_tensor=probe.fault_tensor,
        fault_effect=probe.fault_effect,
        witness=probe.witness,
        world_size=trainer.world_size,
        num_nodes=trainer.num_nodes,
        ranks_finished=finished,
        **setup,
        strategy=type(trainer.strategy).__name__,
        save_distributed_checkpoint=getattr(trainer.strategy, "_save_distributed_checkpoint", None),
        lightning_sprite=lightning_sprite.__version__,
        hardware=hardware.get(device, platform.processor)(),
        rows=probe.rows,
        killed_at=probe.killed_at,
        n_saves=len(probe.saves),
        loaded=ckpt_path,
        written=written,
    )
    data = dict(init=probe.init, saves=probe.saves, start=probe.start, post_step3=probe.post_step3, end=probe.end)
    write_once(outdir / f"{name}.pt", lambda stream: torch.save(data, stream))
    write_once(outdir / f"{name}.json", lambda stream: stream.write(json.dumps(meta, indent=1).encode()))


def differences(a, b, path=""):
    """Paths at which two nested structures of tensors, containers and scalars differ. Tensors compare bitwise;
    lists and tuples compare by their elements."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        same = torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
        return [] if same and torch.equal(a, b) else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}{{keys}}"]
        return [d for k in a for d in differences(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}[len]"]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in differences(x, y, f"{path}[{i}]")]
    return [] if type(a) is type(b) and a == b else [path]


def restore(snapshot, start, *, same_world=True, weights_only=False, position=None, ddp=False):
    """R: equality of the restored state with the snapshot taken inside the checkpoint save."""
    if start is None:
        return False, {"error": "no state captured after the restore"}
    diffs = differences(snapshot["trainable"], start["trainable"], "trainable")
    diffs += differences(snapshot["frozen"]["full"], start["frozen"]["full"], "frozen")
    if same_world:
        diffs += differences(snapshot["frozen"]["ranks"], start["frozen"]["ranks"], "frozen_ranks")
    if not weights_only:
        diffs += differences(snapshot["optimizer"], start["optimizer"], "optimizer")
        for key in RANK_INVARIANT:
            if same_world:
                diffs += differences(snapshot[key], start[key], key)
            else:
                diffs += [d for r, e in enumerate(start[key]) for d in differences(snapshot[key][0], e, f"{key}[{r}]")]
        if same_world:
            diffs += differences(snapshot["replicas"], start["replicas"], "replicas")
        if ddp and len(set(start["replicas"])) != 1:
            diffs.append("DDP replicas differ across ranks")
    if position is not None and (start["global_step"], start["epoch"]) != position:
        diffs.append(f"started at {(start['global_step'], start['epoch'])}, expected {position}")
    return not diffs, {"n_differences": len(diffs), "differences": diffs[:20]}


def by_step(rows):
    return {r["global_step"]: r for r in rows}


def relative(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


def l2_relative(a, b):
    a, b = a.double(), b.double()
    norm = float(b.norm())
    return float((a - b).norm()) / norm if norm else float((a - b).norm())


def row_report(rows, reference):
    """Reported: per family, bit-identical or the largest relative gap, over the steps both ran."""
    ref = by_step(reference)
    shared = [r for r in rows if r["global_step"] in ref]
    out = {}
    for f in ROW_FAMILIES:
        gaps = [relative(r[f], ref[r["global_step"]][f]) for r in shared]
        identical = all(r[f] == ref[r["global_step"]][f] for r in shared)
        out[f] = {"bit_identical": identical, "max_rel_gap": max(gaps, default=0.0)}
    return out


def state_report(state, reference, init):
    """Reported: per-tensor update and AdamW moment gaps of two captured states."""
    if state is None or reference is None:
        return {"error": "state not captured"}
    updates = [
        l2_relative(state["trainable"][n] - init[n].cpu(), reference["trainable"][n] - init[n].cpu()) for n in init
    ]
    identical = not differences(state["trainable"], reference["trainable"])
    out = {"update": {"bit_identical": identical, "max_rel_gap": max(updates)}}
    for f in ("exp_avg", "exp_avg_sq"):
        a, b = state["optimizer"], reference["optimizer"]
        gaps = [l2_relative(a[n][f], b[n][f]) for n in init]
        out[f] = {"bit_identical": all(torch.equal(a[n][f], b[n][f]) for n in init), "max_rel_gap": max(gaps)}
    return out


def flat(row):
    return [lr for rank in row["lrs"] for lr in rank]


def exact_continuation(save_meta, resume, full_meta, lr_roles, loss_check=True):
    """C: resume's step-3 loss against save's, every role's LR against full's, and the final AdamW step counts."""
    resume_data, resume_meta = resume
    full_lr = {s: r["lr"] for s, r in by_step(full_meta["rows"]).items()}
    end = resume_data["end"]
    checks = {
        # Every group's LR on every rank, against full's rank-0 LR of the same step (full's own ranks included).
        "lr_every_step": all(
            lr == full_lr.get(r["global_step"]) for m in (*lr_roles, full_meta) for r in m["rows"] for lr in flat(r)
        ),
        "adam_step_6": end is not None
        and all("step" in s and float(s["step"]) == TOTAL_STEPS for s in end["optimizer"].values()),
    }
    if loss_check:
        save_rows, resume_rows = by_step(save_meta["rows"]), by_step(resume_meta["rows"])
        step3 = STEPS_PER_EPOCH
        checks["step3_loss_equals_save"] = (
            step3 in save_rows and step3 in resume_rows and save_rows[step3]["losses"] == resume_rows[step3]["losses"]
        )
    return all(checks.values()), checks


def same_rows(metas, full_meta, ordered=True):
    """D: per rank and in order (same world), or as a multiset of the step's rows (another world)."""
    ref = by_step(full_meta["rows"])

    def key(r):
        return r["rows_by_rank"] if ordered else sorted(i for rank in r["rows_by_rank"] for i in rank)

    return all(r["global_step"] in ref and key(r) == key(ref[r["global_step"]]) for m in metas for r in m["rows"])


def load(outdir, name):
    meta_path = outdir / f"{name}.json"
    if not meta_path.exists():
        return None
    return torch.load(outdir / f"{name}.pt", weights_only=False), json.loads(meta_path.read_text())


def witness_ok(meta):
    return bool(meta["witness"]) and len(meta["witness"]) == len(WITNESS) and all(meta["witness"].values())


def completed(meta):
    return meta["ranks_finished"] == list(range(meta["world_size"]))


def setup_ok(metas, code):
    return all(completed(m) and m["code_sha256"] == code for m in metas) and all(
        m[k] == metas[0][k] for m in metas for k in SETUP_KEYS
    )


def not_vacuous(arm, roles):
    (save_data, save_meta), (resume_data, resume_meta) = roles["save"], roles["resume"]
    reload_meta = roles["reload"][1]
    start, init = resume_data["start"], resume_data["init"]
    state = start["optimizer"]
    ckpt = [f"{arm}_ckpt/epoch{e}.ckpt" for e in range(EPOCHS)]
    snapshot = save_data["saves"][0]
    checks = {
        "save_one_checkpoint_epoch0_step3": save_meta["written"] == ckpt[:1]
        and (snapshot["ckpt_epoch"], snapshot["ckpt_global_step"]) == (0, STEPS_PER_EPOCH),
        "save_killed_at_step4": save_meta["killed_at"] == STEPS_PER_EPOCH + KILL_BATCH,
        "resume_wrote_epoch1": resume_meta["written"] == ckpt,
        "witness_changed": witness_ok(resume_meta) and witness_ok(reload_meta),
        "weights_moved": all(not torch.equal(start["trainable"][n], init[n].cpu()) for n in init),
        "adam_state_populated": set(state) >= set(init)
        and all(
            "step" in state[n]
            and float(state[n]["step"]) == STEPS_PER_EPOCH
            and bool(state[n]["exp_avg"].any())
            and bool(state[n]["exp_avg_sq"].any())
            for n in init
        ),
        "lr_decayed": all(g["lr"] < LR for rank in start["groups"] for g in rank),
        "resume_steps_3_to_5": [r["global_step"] for r in resume_meta["rows"]] == [3, 4, 5],
        "full_and_twin_steps_0_to_5": all(
            [r["global_step"] for r in roles[r][1]["rows"]] == list(range(TOTAL_STEPS)) for r in ("full", "twin")
        ),
        "reload_killed_before_a_step": reload_meta["killed_at"] == TOTAL_STEPS and not reload_meta["rows"],
    }
    return all(checks.values()), checks


def frozen_unchanged(roles):
    """F: within every training role, the frozen digests at the end (or the kill) equal those at the start."""
    trained = [roles[r][0] for r in ("full", "twin", "save", "resume")]
    save_data = roles["save"][0]
    return all(d["end"] is not None and d["start"]["frozen"] == d["end"]["frozen"] for d in trained) and all(
        s["frozen"] == save_data["start"]["frozen"] for s in save_data["saves"]
    )


def score_arm(outdir, arm, code):
    roles = {role: load(outdir, f"{arm}_{role}") for role in ("full", "twin", "save", "resume", "reload")}
    missing = [role for role, value in roles.items() if value is None]
    if missing:
        return {"arm": arm, "all_pass": False, "error": f"missing roles {missing}", "faults": {}}
    (full_data, full_meta), (twin_data, twin_meta) = roles["full"], roles["twin"]
    (save_data, save_meta), (resume_data, resume_meta) = roles["save"], roles["resume"]
    if len(save_data["saves"]) != 1 or len(resume_data["saves"]) != 1:
        return {"arm": arm, "all_pass": False, "error": "save and resume must each take one checkpoint", "faults": {}}
    metas = [m for _, m in roles.values()]
    snapshot = save_data["saves"][0]
    ddp = arm == "ddp"
    r_resume = restore(snapshot, resume_data["start"], position=(STEPS_PER_EPOCH, KILL_EPOCH), ddp=ddp)
    r_reload = restore(resume_data["saves"][0], roles["reload"][0]["start"], position=(TOTAL_STEPS, EPOCHS), ddp=ddp)
    n_pass, n_detail = not_vacuous(arm, roles)
    c_pass, c_detail = exact_continuation(save_meta, roles["resume"], full_meta, (twin_meta, save_meta, resume_meta))
    worlds = {m["world_size"] for m in metas}
    checks = {
        "I_init_identical": all(not differences(d["init"], full_data["init"]) for d, _ in roles.values()),
        "R_resume": r_resume[0],
        "R_reload": r_reload[0],
        "N_not_vacuous": n_pass,
        "C_exact_continuation": c_pass,
        "D_same_rows": same_rows((twin_meta, save_meta, resume_meta), full_meta),
        "F_frozen_unchanged": frozen_unchanged(roles),
        "W_world_and_setup": len(worlds) == 1
        and (worlds == {1}) == (arm == "single")
        and len({m["num_nodes"] for m in metas}) == 1
        and setup_ok(metas, code),
    }
    init = full_data["init"]
    reported = {
        "twin_vs_full": {
            **row_report(twin_meta["rows"], full_meta["rows"]),
            **state_report(twin_data["end"], full_data["end"], init),
        },
        "save_vs_full_steps_0_3": row_report(save_meta["rows"], full_meta["rows"]),
        "resume_vs_full_steps_3_5": {
            **row_report(resume_meta["rows"], full_meta["rows"]),
            **state_report(resume_data["end"], full_data["end"], init),
        },
        "resume_vs_save_after_step3": state_report(resume_data["post_step3"], save_data["post_step3"], init),
    }
    report = {
        "arm": arm,
        "world_size": metas[0]["world_size"],
        "strategy": metas[0]["strategy"],
        "all_pass": all(checks.values()),
        "checks": checks,
        "R_resume": r_resume[1],
        "R_reload": r_reload[1],
        "N": n_detail,
        "C": c_detail,
        "reported": reported,
        "faults": {},
    }
    for fault in FAULTS[1:]:
        faulty = load(outdir, f"{arm}_resume_{fault}")
        if faulty is None:
            report["faults"][fault] = {"rejected": False, "error": "missing"}
            continue
        fr = restore(snapshot, faulty[0]["start"], position=(STEPS_PER_EPOCH, KILL_EPOCH), ddp=ddp)
        fc = exact_continuation(save_meta, faulty, full_meta, (faulty[1],))
        valid = bool(faulty[1]["fault_effect"]) and setup_ok([*metas, faulty[1]], code)
        report["faults"][fault] = {
            "rejected": report["all_pass"] and valid and not fr[0] and not fc[0],
            "took_effect": bool(faulty[1]["fault_effect"]),
            "R_failed": not fr[0],
            "C_failed_checks": [k for k, v in fc[1].items() if not v],
            "R_differences": fr[1].get("differences", [])[:5],
        }
    return report


def score_reshard(outdir, code):
    save, resume, full = (load(outdir, f"fsdp_{r}") for r in ("save", "reshard", "full"))
    if save is None or resume is None or full is None:
        return {"arm": "fsdp_reshard", "all_pass": False, "error": "missing fsdp_save, fsdp_reshard or fsdp_full"}
    (save_data, save_meta), (data, meta) = save, resume
    if len(save_data["saves"]) != 1:
        return {"arm": "fsdp_reshard", "all_pass": False, "error": "fsdp_save must take one checkpoint"}
    r_pass, r_detail = restore(
        save_data["saves"][0], data["start"], same_world=False, position=(STEPS_PER_EPOCH, KILL_EPOCH)
    )
    c_pass, c_detail = exact_continuation(save_meta, resume, full[1], (meta,), loss_check=False)
    checks = {
        "R_restore": r_pass,
        "N_witness_and_steps": witness_ok(meta) and [r["global_step"] for r in meta["rows"]] == [3, 4, 5],
        "C_lr_and_adam_steps": c_pass,
        "D_same_rows_multiset": same_rows((meta,), full[1], ordered=False),
        "F_frozen_unchanged": data["end"] is not None and data["start"]["frozen"] == data["end"]["frozen"],
        "W_other_world": 1 < meta["world_size"] != save_meta["world_size"] and setup_ok([save_meta, meta], code),
    }
    return {
        "arm": "fsdp_reshard",
        "world_size": meta["world_size"],
        "saved_world_size": save_meta["world_size"],
        "all_pass": all(checks.values()),
        "checks": checks,
        "R": r_detail,
        "C": c_detail,
        "reported_vs_full": {
            **row_report(meta["rows"], full[1]["rows"]),
            **state_report(data["end"], full[0]["end"], full[0]["init"]),
        },
    }


def score_export(outdir, source, code):
    save, export = load(outdir, f"{source}_save"), load(outdir, f"{source}_export")
    if save is None or export is None or len(save[0]["saves"]) != 1:
        return {"arm": f"{source}_export", "all_pass": False, "error": f"missing {source}_save or {source}_export"}
    r_pass, r_detail = restore(save[0]["saves"][0], export[0]["start"], same_world=False, weights_only=True)
    meta = export[1]
    checks = {
        "R_weights_and_frozen": r_pass,
        "N_witness_changed": witness_ok(meta),
        "W_one_device": meta["world_size"] == 1 < save[1]["world_size"] and setup_ok([save[1], meta], code),
    }
    return {"arm": f"{source}_export", "all_pass": all(checks.values()), "checks": checks, "R": r_detail}


def compare(outdir):
    code = code_sha256()
    reports = [score_arm(outdir, arm, code) for arm in ARMS]
    reports += [score_reshard(outdir, code)] + [score_export(outdir, s, code) for s in EXPORT_SOURCES]
    faults = [f for r in reports for f in r.get("faults", {}).values()]
    faults_rejected = len(faults) == len(ARMS) * (len(FAULTS) - 1) and all(f["rejected"] for f in faults)
    verdict = "pass" if all(r["all_pass"] for r in reports) and faults_rejected else "fail"
    out = {"verdict": verdict, "faults_rejected": faults_rejected, "code_sha256": code, "arms": reports}
    (outdir / "checkpoint_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("arm", choices=ARMS)
    run_parser.add_argument("role", choices=ROLES)
    run_parser.add_argument("outdir", type=Path)
    run_parser.add_argument("--device", choices=["cuda", "xpu", "cpu"], default="cuda")
    run_parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    run_parser.add_argument("--world", type=int, default=4)
    run_parser.add_argument("--nodes", type=int, default=1)
    run_parser.add_argument("--global-batch", type=int, default=8)
    run_parser.add_argument("--precision", choices=["32-true", "bf16-mixed"], default="32-true")
    run_parser.add_argument("--fault", choices=FAULTS, default="none")
    run_parser.add_argument("--source", choices=EXPORT_SOURCES, default=None)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        sys.exit(compare(args.outdir))
    run(
        args.arm,
        args.role,
        args.outdir,
        args.device,
        args.model,
        args.world,
        args.nodes,
        args.global_batch,
        args.precision,
        args.fault,
        args.source,
    )
