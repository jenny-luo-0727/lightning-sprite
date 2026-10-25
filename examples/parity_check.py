"""Hardware parity check: DDP and FSDP2 on N devices must reproduce a 1-device reference.

Runs the same LoRA fine-tune three ways and compares them:
  ref    1 device, batch B (the global batch), single-device strategy.
  ddp    N devices, batch B/N per rank, XPUDDPStrategy (Lightning's DDP on CUDA).
  fsdp   N devices, batch B/N per rank, FSDP2Strategy with shard_model.

Held fixed: Qwen2.5-0.5B with a PEFT LoRA adapter (r=8, q/k/v/o) and a 2-class linear head on the last token
inside the sharded root; seed 0; identical initial weights; the same B rows per step in every arm (an explicit
DistributedSampler(shuffle=False), because Lightning's own training sampler always shuffles); fp32 with TF32 off;
SGD lr 1.0; global-norm clip 0.01, active on every step; 5 steps.

Pass criteria, fixed in advance, each arm against ref (tolerance 1e-4 relative):
  P0 initial trainable weights bit-identical.   P1 clip active on every step.
  P2 pre-clip global grad norm matches, every step.   P3 post-clip global norm equals min(pre, clip).
  P4 the 5-step trainable update matches (relative L2).   P5 frozen weights unchanged, bit-for-bit.
  W  ref ran on 1 device and the arm on more than 1.   D  every step trained on the same rows.
compare fails unless all three arms are present; an arm refuses to write into an OUTDIR that already holds it.

Usage:
  python parity_check.py {ref,ddp,fsdp} OUTDIR --device {cuda,xpu,cpu} [--model PATH] [--world N] [--global-batch B]
      [--nodes M]   (the ddp/fsdp arms run N/M processes on each of M nodes)
  (B defaults to 2N and must be the same for every arm.)
  python parity_check.py compare OUTDIR
Launch the ddp/fsdp arms with one process per device (srun on Slurm, mpiexec on PBS/PALS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

CLIP, LR, STEPS, SEQ, VOCAB_CAP, TOL = 0.01, 1.0, 5, 32, 1000, 1e-4


def _is_dtensor(t):
    from torch.distributed.tensor import DTensor

    return isinstance(t, DTensor)


def _local(t):
    return t.to_local() if _is_dtensor(t) else t


def _full(t):
    return t.full_tensor() if _is_dtensor(t) else t


def _all_reduce_sum(x):
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(x)
    return x


class Net(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        base = AutoModel.from_pretrained(model_name, dtype=torch.float32, attn_implementation="sdpa")
        config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.backbone = get_peft_model(base, config)
        self.head = nn.Linear(base.config.hidden_size, 2)

    def forward(self, ids):
        return self.head(self.backbone(input_ids=ids).last_hidden_state[:, -1])


def make_probe(mode, model_name, world, global_batch):
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    from lightning_sprite import shard_model

    class Probe(L.LightningModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = Net(model_name)
            self.rows, self.init_trainable, self.init_frozen_local, self.grads0 = [], {}, {}, {}

        def configure_model(self):
            if mode == "fsdp":
                shard_model(self.net, self.device_mesh["data_parallel"])

        def train_dataloader(self):
            g = torch.Generator().manual_seed(1234)
            ids = torch.randint(0, VOCAB_CAP, (global_batch * STEPS, SEQ), generator=g)
            y = torch.randint(0, 2, (global_batch * STEPS,), generator=g)
            dataset = TensorDataset(torch.arange(global_batch * STEPS), ids, y)
            if mode == "ref":
                return DataLoader(dataset, batch_size=global_batch, shuffle=False)
            sampler = DistributedSampler(
                dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=False
            )
            return DataLoader(dataset, batch_size=global_batch // world, sampler=sampler)

        def training_step(self, batch, _):
            idx, ids, y = batch
            loss = nn.functional.cross_entropy(self.net(ids), y)
            seen = sorted(int(i) for i in self.all_gather(idx).flatten().tolist())
            self.rows.append({"step": len(self.rows), "rows_seen": seen})
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=LR)

        def _grad_norm(self):
            sharded = torch.zeros((), device=self.device, dtype=torch.float64)
            replicated = torch.zeros((), device=self.device, dtype=torch.float64)
            for p in self.parameters():
                if p.requires_grad and p.grad is not None:
                    if _is_dtensor(p.grad):
                        sharded += _local(p.grad).double().pow(2).sum()
                    else:
                        replicated += p.grad.double().pow(2).sum()
            return float((_all_reduce_sum(sharded) + replicated).sqrt())

        def on_train_start(self):
            for name, p in self.named_parameters():
                if p.requires_grad:
                    self.init_trainable[name] = _full(p.detach()).cpu().clone()
                else:
                    self.init_frozen_local[name] = _local(p.detach()).cpu().clone()

        def on_before_optimizer_step(self, optimizer):
            if len(self.rows) == 1:
                self.grads0 = {
                    n: _full(p.grad.detach()).cpu().clone()
                    for n, p in self.named_parameters()
                    if p.requires_grad and p.grad is not None
                }
            self.rows[-1]["pre_clip_norm"] = self._grad_norm()

        def on_train_batch_end(self, *_):
            self.rows[-1]["post_clip_norm"] = self._grad_norm()

        def on_train_end(self):
            bad = torch.zeros((), device=self.device, dtype=torch.int64)
            self.final_trainable = {}
            for name, p in self.named_parameters():
                if p.requires_grad:
                    self.final_trainable[name] = _full(p.detach()).cpu().clone()
                elif not torch.equal(_local(p.detach()).cpu(), self.init_frozen_local[name]):
                    bad += 1
            self.frozen_mismatches = int(_all_reduce_sum(bad))

    return Probe()


def trainer_device_args(mode, device, world, nodes):
    # Lightning's devices count is per node; the arms spread `world` processes over `nodes` nodes.
    devices, num_nodes = (1, 1) if mode == "ref" else (world // nodes, nodes)
    if device == "xpu":
        from lightning_sprite import trainer_kwargs

        return trainer_kwargs(
            {"ref": "single", "ddp": "ddp", "fsdp": "fsdp"}[mode],
            devices=devices,
            num_nodes=num_nodes,
            precision="32-true",
        )
    from lightning_sprite import FSDP2Strategy, XPUDDPStrategy

    strategy = {"ref": "auto", "ddp": XPUDDPStrategy(), "fsdp": FSDP2Strategy()}[mode]
    return {
        "accelerator": device,
        "devices": devices,
        "num_nodes": num_nodes,
        "strategy": strategy,
        "precision": "32-true",
    }


def run(mode, outdir, device, model_name, world, global_batch, nodes=1):
    import lightning as L

    if world % nodes:
        raise ValueError(f"--world {world} must be divisible by --nodes {nodes}")
    if global_batch % world:
        raise ValueError(f"--global-batch {global_batch} must be divisible by --world {world}")
    if mode != "ref" and world < 2:
        raise ValueError(f"the {mode} arm needs --world >= 2 to test anything distributed")
    if (outdir / f"{mode}.json").exists():
        raise FileExistsError(f"{outdir / f'{mode}.json'} exists; use a fresh OUTDIR so no stale arm is compared")
    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    probe = make_probe(mode, model_name, world, global_batch)
    trainer = L.Trainer(
        **trainer_device_args(mode, device, world, nodes),
        max_steps=STEPS,
        gradient_clip_val=CLIP,
        gradient_clip_algorithm="norm",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        use_distributed_sampler=False,
    )
    trainer.fit(probe)
    if trainer.global_rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": mode,
            "device": device,
            "world_size": trainer.world_size,
            "num_nodes": trainer.num_nodes,
            "global_batch": global_batch,
            "strategy": type(trainer.strategy).__name__,
            "torch": torch.__version__,
            "lightning": L.__version__,
            "rows": probe.rows,
            "frozen_mismatches": probe.frozen_mismatches,
            "n_frozen": len(probe.init_frozen_local),
        }
        (outdir / f"{mode}.json").write_text(json.dumps(meta, indent=1))
        torch.save(
            {"init": probe.init_trainable, "final": probe.final_trainable, "grads0": probe.grads0},
            outdir / f"{mode}.pt",
        )


def compare_arm(outdir, mode):
    ref_meta, arm_meta = (json.loads((outdir / f"{m}.json").read_text()) for m in ("ref", mode))
    ref, arm = torch.load(outdir / "ref.pt"), torch.load(outdir / f"{mode}.pt")
    names = sorted(ref["init"])
    rr, ar = ref_meta["rows"], arm_meta["rows"]
    rel = lambda a, b: abs(a - b) / abs(b)  # noqa: E731
    pre = [rel(a["pre_clip_norm"], r["pre_clip_norm"]) for r, a in zip(rr, ar, strict=True)]
    post = [rel(x["post_clip_norm"], min(x["pre_clip_norm"], CLIP)) for x in rr + ar]
    flat = lambda d: torch.cat([(d["final"][n] - d["init"][n]).flatten().double() for n in names])  # noqa: E731
    d_ref, d_arm = flat(ref), flat(arm)
    update = float((d_arm - d_ref).norm() / d_ref.norm())
    checks = {
        "P0_init_identical": names == sorted(arm["init"])
        and all(torch.equal(ref["init"][n], arm["init"][n]) for n in names),
        "P1_clip_active": len(rr) == len(ar) == STEPS and all(x["pre_clip_norm"] > CLIP for x in rr + ar),
        "P2_pre_norm_match": max(pre) <= TOL,
        "P3_post_norm_is_clip": max(post) <= TOL,
        "P4_update_match": update <= TOL,
        "P5_frozen_unchanged": ref_meta["frozen_mismatches"] == 0 and arm_meta["frozen_mismatches"] == 0,
        "W_multi_device": ref_meta["world_size"] == 1 and arm_meta["world_size"] > 1,
        "D_same_rows_every_step": ref_meta["global_batch"] == arm_meta["global_batch"]
        and [r["rows_seen"] for r in rr] == [a["rows_seen"] for a in ar],
    }
    return {
        "arm": mode,
        "device": arm_meta["device"],
        "world_size": arm_meta["world_size"],
        "num_nodes": arm_meta.get("num_nodes", 1),
        "strategy": arm_meta["strategy"],
        "all_pass": all(checks.values()),
        "checks": checks,
        "max_pre_norm_rel_diff": max(pre),
        "max_post_norm_rel_err": max(post),
        "update_rel_l2": update,
        "frozen_changed": f"{arm_meta['frozen_mismatches']} of {arm_meta['n_frozen']}",
    }


def compare(outdir):
    missing = [mode for mode in ("ref", "ddp", "fsdp") if not (outdir / f"{mode}.json").exists()]
    if missing:
        print(f"FAIL: missing arm(s) {missing} in {outdir}; the check passes only with ref, ddp and fsdp")
        return 1
    reports = [compare_arm(outdir, mode) for mode in ("ddp", "fsdp")]
    if reports[0]["world_size"] != reports[1]["world_size"]:
        reports.append({"all_pass": False, "error": "ddp and fsdp ran at different world sizes"})
    (outdir / "report.json").write_text(json.dumps(reports, indent=1))
    print(json.dumps(reports, indent=1))
    return 0 if all(r["all_pass"] for r in reports) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ref", "ref32", "split", "split_premul", "ddp", "fsdp", "compare", "faults"])
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--device", choices=["cuda", "xpu", "cpu"], default="cuda")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--global-batch", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=1, help="nodes the ddp/fsdp arms span (ref is always 1 device)")
    parser.add_argument("--precision", choices=["32-true", "bf16-mixed"], default="32-true")
    parser.add_argument(
        "--fault",
        choices=["none", "fp32-backbone", "local-adapter", "scale-grad", "skip-update", "missing-rank", "wrong-scale"],
        default="none",
    )
    parser.add_argument(
        "--expected-faults",
        help="colon-separated fault list for the faults summary (default: all six kinds)",
    )
    args = parser.parse_args()
    if args.precision == "bf16-mixed":
        import bf16_parity

        if args.mode == "compare":
            sys.exit(bf16_parity.compare(args.outdir))
        if args.mode == "faults":
            sys.exit(bf16_parity.faults(args.outdir, args.expected_faults))
        bf16_parity.run(
            args.mode,
            args.outdir,
            args.device,
            args.model,
            args.world,
            args.global_batch or 2 * args.world,
            args.nodes,
            args.fault,
        )
        sys.exit(0)
    if args.mode in ("ref32", "split", "split_premul", "faults") or args.fault != "none":
        parser.error("ref32, split, split_premul, faults and fault controls require --precision bf16-mixed")
    if args.mode == "compare":
        sys.exit(compare(args.outdir))
    run(args.mode, args.outdir, args.device, args.model, args.world, args.global_batch or 2 * args.world, args.nodes)
