"""BF16 autocast parity, criterion v2 (step-0 scoring); the original FP32 test remains in parity_check.py.

Criterion v2 scores step 0, checks the split averaging-order validity gate, and reports later trajectory
differences diagnostically. Within-arm synchronization is required on every step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from parity_check import CLIP, LR, SEQ, STEPS, VOCAB_CAP, Net, _all_reduce_sum, _full, _is_dtensor, _local
from torch import nn

CRITERION = "v2"
REL_TOL, GRAD_RMS_FLOOR, POST_TOL = 1e-3, 1e-12, 1e-4
# PEFT starts lora_B at zero, which makes every lora_A gradient and update exactly zero at step 0.
# v2 starts lora_B from a seeded normal draw instead, shared by every arm.
LORA_B_STD, LORA_B_SEED = 0.01, 1
ARMS = ("ref32", "ref", "split", "split_premul", "ddp", "fsdp")
# split sums shard gradients and divides once; split_premul multiplies each by 1/world first, as DDP's reducer does.
SPLIT = ("split", "split_premul")
SCORED = ("ddp", "fsdp")
DDP_FAULTS = ("fp32-backbone", "local-adapter", "scale-grad", "skip-update")
# Injected through the loss, so they apply to DDP and FSDP2 alike.
LOSS_FAULTS = ("missing-rank", "wrong-scale")
FAULTS = ("none", *DDP_FAULTS, *LOSS_FAULTS)


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def norm(parameters):
    sharded = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    replicated = torch.zeros_like(sharded)
    for p in parameters:
        if p.grad is not None:
            if _is_dtensor(p.grad):
                sharded += _local(p.grad).double().square().sum()
            else:
                replicated += p.grad.double().square().sum()
    return float((_all_reduce_sum(sharded) + replicated).sqrt())


def replica_equal(tensors):
    """Bitwise agreement of every tensor with rank 0's copy. Every rank must call it with the same tensors."""
    ok = True
    for value in tensors:
        reference = value.detach().clone()
        dist.broadcast(reference, 0)
        ok = ok and torch.equal(value, reference)
    return ok


def sharded(tensor):
    return _is_dtensor(tensor) and all(placement.is_shard() for placement in tensor.placements)


def make_probe(mode, model_name, world, global_batch, fault):
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    from lightning_sprite import shard_model

    class Probe(L.LightningModule):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.net = Net(model_name)
            generator = torch.Generator().manual_seed(LORA_B_SEED)
            with torch.no_grad():
                for name, p in sorted(self.net.named_parameters()):
                    if ".lora_B." in name:
                        p.copy_(torch.randn(p.shape, generator=generator) * LORA_B_STD)
            self.automatic_optimization = mode not in SPLIT
            self.records, self.gradients, self.observations = [], [], {}
            self.init, self.frozen, self.step0, self.final = {}, {}, {}, {}
            self.trainables = {n: p for n, p in self.named_parameters() if p.requires_grad}
            self.expected_dtype = torch.float32 if mode == "ref32" else torch.bfloat16
            self.local_fault_gradient = None
            self.fault_name = next(n for n in self.trainables if ".q_proj.lora_A." in n)
            self.dtype_names = []
            for name, module in self.net.named_modules():
                if isinstance(module, nn.Linear):
                    self.dtype_names.append(name)
                    module.register_forward_hook(self.dtype_hook(name))
            if fault == "local-adapter":
                self.trainables[self.fault_name].register_hook(self.capture_local)

        def capture_local(self, gradient):
            self.local_fault_gradient = gradient.detach().clone()
            return gradient

        def dtype_hook(self, name):
            def hook(_module, _inputs, output):
                obs = self.observations.setdefault(name, {"calls": 0, "valid": True})
                obs["calls"] += 1
                obs["dtype"], obs["device"] = str(output.dtype), output.device.type
                obs["autocast"] = torch.is_autocast_enabled(output.device.type)
                obs["valid"] &= (
                    output.dtype == self.expected_dtype
                    and output.device.type == self.device.type
                    and obs["autocast"] == (mode != "ref32")
                )

            return hook

        def configure_model(self):
            if mode == "fsdp":
                shard_model(self.net, self.device_mesh["data_parallel"])
                self.trainables = {n: p for n, p in self.named_parameters() if p.requires_grad}

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.trainables.values(), lr=LR)
            if fault == "skip-update":

                def skip_last(opt, _args, _kwargs):
                    if self.trainer.global_rank != 0 and self.global_step == STEPS - 1:
                        for group in opt.param_groups:
                            group["lr"] = 0.0

                optimizer.register_step_pre_hook(skip_last)
            return optimizer

        def train_dataloader(self):
            generator = torch.Generator().manual_seed(1234)
            ids = torch.randint(0, VOCAB_CAP, (global_batch * STEPS, SEQ), generator=generator)
            y = torch.randint(0, 2, (global_batch * STEPS,), generator=generator)
            dataset = TensorDataset(torch.arange(global_batch * STEPS), ids, y)
            if mode in ("ref32", "ref", *SPLIT):
                return DataLoader(dataset, batch_size=global_batch, shuffle=False)
            sampler = DistributedSampler(dataset, num_replicas=world, rank=self.global_rank, shuffle=False)
            return DataLoader(dataset, batch_size=global_batch // world, sampler=sampler)

        def synchronized(self, tensors):
            """Within-arm agreement across ranks. DDP: every replica bitwise equal to rank 0's. FSDP2: every tensor
            sharded, and the gathered full tensor bitwise equal on every rank (catches a trainable left outside
            fully_shard, which never syncs). Single-device arms: trivially true."""
            if mode == "ddp":
                return replica_equal(tensors)
            if mode == "fsdp":
                gathered = replica_equal([_full(t.detach()) for t in tensors])
                return all(sharded(t) for t in tensors) and gathered
            return True

        def on_train_start(self):
            from torch.distributed.fsdp import FSDPModule

            self.communication = {"mode": mode, "parameter_dtype": "torch.float32"}
            if mode == "fsdp":
                policies = [m._get_fsdp_state()._mp_policy for m in self.net.modules() if isinstance(m, FSDPModule)]
                self.communication["fsdp_policies"] = [str(p) for p in policies]
                self.communication["valid"] = bool(policies) and all(
                    p.param_dtype is None and p.reduce_dtype is None and p.output_dtype is None for p in policies
                )
            elif mode == "ddp":
                wrapper = self.trainer.strategy.model
                self.communication["comm_hooks"] = len(wrapper._comm_hooks)
                self.communication["reducer_dtypes"] = wrapper._get_ddp_logging_data().get("dtypes")
                self.communication["valid"] = (
                    not wrapper._comm_hooks and self.communication["reducer_dtypes"] == "float"
                )
            else:
                self.communication["valid"] = True
            for name, p in self.named_parameters():
                if p.requires_grad:
                    self.init[name] = _full(p.detach()).cpu().clone()
                else:
                    self.frozen[name] = _local(p.detach()).cpu().clone()
            self.initial_replicas_equal = self.synchronized(list(self.trainables.values()))

        def forward_loss(self, ids, y):
            if fault == "fp32-backbone":
                with torch.autocast(self.device.type, enabled=False):
                    hidden = self.net.backbone(input_ids=ids).last_hidden_state[:, -1]
                pred = self.net.head(hidden)
            else:
                pred = self.net(ids)
            loss = nn.functional.cross_entropy(pred, y)
            if fault == "missing-rank" and self.global_rank == world - 1:
                return loss * 0.0  # this rank's gradient never reaches the average
            if fault == "wrong-scale":
                return loss * world  # gradients summed instead of averaged
            return loss

        def training_step(self, batch, _batch_idx):
            idx, ids, y = batch
            self.observations = {}
            self.records.append({"step": len(self.records), "rank": self.global_rank})
            if mode not in SPLIT:
                self.records[-1]["shards"] = [idx.tolist()]
                return self.forward_loss(ids, y)
            # Unscaled backward per shard, then FP32 averaging before one clipping/step.
            # Lightning accumulate_grad_batches divides before BF16 backward instead.
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.records[-1]["shards"] = []
            losses, summed = [], {}
            for rank in range(world):
                self.records[-1]["shards"].append(idx[rank::world].tolist())
                loss = self.forward_loss(ids[rank::world], y[rank::world])
                self.manual_backward(loss)
                losses.append(loss.detach())
                if mode == "split_premul":
                    for n, p in self.trainables.items():
                        if p.grad is not None:
                            scaled = p.grad.mul(1.0 / world)
                            summed[n] = summed[n].add_(scaled) if n in summed else scaled
                            p.grad = None
            for n, p in self.trainables.items():
                if mode == "split_premul":
                    p.grad = summed.get(n)
                elif p.grad is not None:
                    p.grad.div_(world)
            self.capture_pre()
            self.clip_gradients(optimizer, gradient_clip_val=CLIP, gradient_clip_algorithm="norm")
            optimizer.step()
            return torch.stack(losses).mean()

        def capture_pre(self):
            if fault == "scale-grad":
                for p in self.trainables.values():
                    if p.grad is not None:
                        p.grad.mul_(world)
            elif fault == "local-adapter":
                self.trainables[self.fault_name].grad.copy_(self.local_fault_gradient)
            record = self.records[-1]
            record["pre_clip_norm"] = norm(list(self.trainables.values()))
            record["gradients_fp32_finite_present"] = all(
                p.dtype == torch.float32
                and p.grad is not None
                and p.grad.dtype == torch.float32
                and bool(torch.isfinite(_local(p.grad)).all())
                for p in self.trainables.values()
            )
            record["grad_replicas_equal"] = self.synchronized(
                [p.grad for p in self.trainables.values() if p.grad is not None]
            )
            self.gradients.append(
                {n: _full(p.grad.detach()).cpu().clone() for n, p in self.trainables.items() if p.grad is not None}
            )

        def on_before_optimizer_step(self, _optimizer):
            if mode not in SPLIT:
                self.capture_pre()

        def on_train_batch_end(self, *_args):
            record = self.records[-1]
            record["post_clip_norm"] = norm(list(self.trainables.values()))
            record["replicas_equal"] = self.synchronized(list(self.trainables.values()))
            if len(self.records) == 1:
                self.step0 = {n: _full(p.detach()).cpu().clone() for n, p in self.trainables.items()}
            record["dtype_sites"] = self.observations
            record["dtype_sites_complete"] = set(self.observations) == set(self.dtype_names)
            record["initial_replicas_equal"] = self.initial_replicas_equal
            record["ranks"] = gather(dict(record))

        def on_train_end(self):
            bad = torch.zeros((), dtype=torch.int64, device=self.device)
            for name, p in self.named_parameters():
                if p.requires_grad:
                    self.final[name] = _full(p.detach()).cpu().clone()
                elif not torch.equal(_local(p.detach()).cpu(), self.frozen[name]):
                    bad += 1
            self.frozen_mismatches = int(_all_reduce_sum(bad))
            self.communication_ranks = gather(self.communication)

    return Probe()


def run(mode, outdir, device, model_name, world, global_batch, nodes=1, fault="none"):
    import platform

    import lightning as L

    from lightning_sprite import FSDP2Strategy, XPUDDPStrategy, trainer_kwargs

    if world < 2 or nodes < 1 or world % nodes or global_batch <= 0 or global_batch % world:
        raise ValueError("Need world >= 2, positive nodes/batch, world divisible by nodes and batch by world")
    if mode not in ARMS or fault not in FAULTS:
        raise ValueError(f"BF16 modes: {'/'.join(ARMS)}; faults: {'/'.join(FAULTS)}")
    if (fault in DDP_FAULTS and mode != "ddp") or (fault in LOSS_FAULTS and mode not in SCORED):
        raise ValueError(f"{fault} applies to {'ddp' if fault in DDP_FAULTS else 'ddp and fsdp'} only")
    outdir.mkdir(parents=True, exist_ok=True)
    if any((outdir / f"{mode}.{ext}").exists() for ext in ("json", "pt")):
        raise FileExistsError(f"{mode} already exists in {outdir}; use a fresh output directory")
    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    single = mode in ("ref32", "ref", *SPLIT)
    precision = "32-true" if mode == "ref32" else "bf16-mixed"
    devices = 1 if single else world // nodes
    if device == "xpu":
        kwargs = trainer_kwargs(
            "single" if single else mode, devices=devices, num_nodes=1 if single else nodes, precision=precision
        )
    else:
        kwargs = dict(
            accelerator=device,
            devices=devices,
            num_nodes=1 if single else nodes,
            precision=precision,
            strategy="auto" if single else XPUDDPStrategy() if mode == "ddp" else FSDP2Strategy(),
        )
    probe = make_probe(mode, model_name, world, global_batch, fault)
    trainer = L.Trainer(
        **kwargs,
        max_steps=STEPS,
        gradient_clip_val=None if mode in SPLIT else CLIP,
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
    communication = probe.communication_ranks
    if trainer.global_rank == 0:
        hardware = {"cuda": torch.cuda.get_device_name, "xpu": lambda: torch.xpu.get_device_name()}
        meta = dict(
            mode=mode,
            criterion=CRITERION,
            device=device,
            world_size=trainer.world_size,
            emulated_world=world,
            num_nodes=trainer.num_nodes,
            global_batch=global_batch,
            precision=precision,
            model=model_name,
            torch=torch.__version__,
            lightning=L.__version__,
            strategy=type(trainer.strategy).__name__,
            seed=0,
            lora_b_init=dict(std=LORA_B_STD, seed=LORA_B_SEED),
            fault=fault,
            fault_tensor=probe.fault_name,
            rows=probe.records,
            frozen_mismatches=probe.frozen_mismatches,
            n_frozen=len(probe.frozen),
            communication=communication,
            hardware=hardware.get(device, platform.processor)(),
        )
        with (outdir / f"{mode}.pt").open("xb") as stream:
            torch.save(dict(init=probe.init, step0=probe.step0, final=probe.final, gradients=probe.gradients), stream)
        with (outdir / f"{mode}.json").open("x") as stream:
            json.dump(meta, stream, indent=1)


def relative(a, b):
    return abs(a - b) / abs(b) if b else (0.0 if a == b else math.inf)


def tensor_metrics(actual, reference, initial=None):
    reports = {}
    for name, ref in reference.items():
        delta = (actual[name].double() - ref.double()).norm().item()
        scale = ref.double().norm().item()
        floor = GRAD_RMS_FLOOR * math.sqrt(ref.numel())
        if initial is not None:
            # FP32 endpoint quantization for weight-update subtraction.
            floor += 2 * torch.finfo(torch.float32).eps * initial[name].double().norm().item()
        reports[name] = dict(
            relative=delta / max(scale, floor),
            absolute_l2=delta,
            reference_l2=scale,
            absolute_floor_l2=floor,
            pass_=math.isfinite(delta) and delta <= REL_TOL * scale + floor,
        )
    return reports


def updates(data, end="final"):
    return {n: data[end][n].double() - p.double() for n, p in data["init"].items()}


def l2_gap(a, b):
    denominator = math.sqrt(sum(float(t.square().sum()) for t in b.values()))
    numerator = math.sqrt(sum(float((a[n] - t).square().sum()) for n, t in b.items()))
    return numerator / denominator if denominator else (0.0 if numerator == 0 else None)


def gap(actual, reference, end="final"):
    return l2_gap(updates(actual, end), updates(reference, end))


def score(meta, data, ref_meta, ref_data):
    """Scored v2 criteria of one arm against split: step-0 gradients, pre-clip norm and the step-0 update."""
    reference_norm = ref_meta["rows"][0]["ranks"][0]["pre_clip_norm"]
    pre = max(relative(r["pre_clip_norm"], reference_norm) for r in meta["rows"][0]["ranks"])
    gradients = tensor_metrics(data["gradients"][0], ref_data["gradients"][0])
    moved = tensor_metrics(updates(data, "step0"), updates(ref_data, "step0"), initial=ref_data["init"])
    update = gap(data, ref_data, "step0")
    checks = dict(
        pre_norm=pre <= REL_TOL,
        gradients=all(t["pass_"] for t in gradients.values()),
        update=update is not None and update <= REL_TOL,
        tensor_updates=all(t["pass_"] for t in moved.values()),
    )
    return dict(
        all_pass=all(checks.values()),
        checks=checks,
        pre_norm_rel=pre,
        update_rel_l2=update,
        max_gradient_relative=max(t["relative"] for t in gradients.values()),
        gradients_failing=sum(not t["pass_"] for t in gradients.values()),
        max_tensor_update_relative=max(t["relative"] for t in moved.values()),
        tensor_updates_failing=sum(not t["pass_"] for t in moved.values()),
        gradients=gradients,
        tensor_updates=moved,
    )


def agreement(actual, reference):
    """Diagnostic: per-step bitwise agreement and largest per-tensor gap of one arm against a reference."""
    steps = []
    for a, b in zip(actual["gradients"], reference["gradients"], strict=True):
        metrics = tensor_metrics(a, b)
        steps.append(
            dict(
                max_relative=max(t["relative"] for t in metrics.values()),
                gradients_failing=sum(not t["pass_"] for t in metrics.values()),
                tensors_equal=sum(torch.equal(a[n], b[n]) for n in b),
                tensors=len(b),
                elements_equal=sum(int((a[n] == b[n]).sum()) for n in b) / sum(b[n].numel() for n in b),
            )
        )
    moved = tensor_metrics(updates(actual), updates(reference), initial=reference["init"])
    return dict(
        steps=steps,
        five_step_update_rel_l2=gap(actual, reference),
        max_five_step_tensor_update_relative=max(t["relative"] for t in moved.values()),
        five_step_tensor_updates_failing=sum(not t["pass_"] for t in moved.values()),
    )


def compare(outdir):
    try:
        meta = {m: json.loads((outdir / f"{m}.json").read_text()) for m in ARMS}
        data = {m: torch.load(outdir / f"{m}.pt", map_location="cpu", weights_only=True) for m in ARMS}
        base = meta["split"]
        world, batch = base["emulated_world"], base["global_batch"]
        names = set(data["split"]["init"])
        invariants = {}
        for mode in ARMS:
            m, d = meta[mode], data[mode]
            ranks = [r for row in m["rows"] for r in row["ranks"]]
            expected_world = world if mode in SCORED else 1
            checks = dict(
                metadata=m["mode"] == mode
                and m.get("criterion") == CRITERION
                and all(m[k] == base[k] for k in ("device", "model", "emulated_world", "global_batch", "seed"))
                and m["lora_b_init"] == base["lora_b_init"]
                and m["world_size"] == expected_world
                and world >= 2
                and m["precision"] == ("32-true" if mode == "ref32" else "bf16-mixed"),
                steps=len(m["rows"]) == len(d["gradients"]) == STEPS
                and all(len(row["ranks"]) == expected_world for row in m["rows"]),
                init=set(d["init"]) == set(d["step0"]) == set(d["final"]) == names
                and all(torch.equal(d["init"][n], data["split"]["init"][n]) for n in names),
                frozen=m["frozen_mismatches"] == 0 and m["n_frozen"] > 0,
                communication=len(m["communication"]) == expected_world and all(c["valid"] for c in m["communication"]),
                clip_active=all(math.isfinite(r["pre_clip_norm"]) and r["pre_clip_norm"] > CLIP for r in ranks),
                clip_norm=all(relative(r["post_clip_norm"], CLIP) <= POST_TOL for r in ranks),
                synchronized=all(
                    r["replicas_equal"] and r["grad_replicas_equal"] and r["initial_replicas_equal"] for r in ranks
                ),
                gradients=all(r["gradients_fp32_finite_present"] for r in ranks)
                and all(set(g) == names for g in d["gradients"]),
                dtype=all(
                    r["dtype_sites_complete"]
                    and bool(r["dtype_sites"])
                    and all(s["valid"] for s in r["dtype_sites"].values())
                    for r in ranks
                ),
            )
            partitions = []
            for step, row in enumerate(m["rows"]):
                expected = [list(range(step * batch + rank, (step + 1) * batch, world)) for rank in range(world)]
                shards = [shard for r in row["ranks"] for shard in r["shards"]]
                full = [list(range(step * batch, (step + 1) * batch))]
                partitions.append(shards == (expected if mode in (*SPLIT, *SCORED) else full))
            checks["rows_and_shards"] = all(partitions)
            checks["all_pass"] = all(checks.values())
            invariants[mode] = checks
        scored = {mode: score(meta[mode], data[mode], base, data["split"]) for mode in SCORED}
        # Validity gate: two correct single-device averaging orders must pass the scored checks against each other.
        gate = score(meta["split_premul"], data["split_premul"], base, data["split"])
        if not all(v["all_pass"] for v in invariants.values()):
            verdict = "fail"
        elif not gate["all_pass"]:
            verdict = "void"
        else:
            verdict = "pass" if all(v["all_pass"] for v in scored.values()) else "fail"
        full_gap = gap(data["ref"], data["ref32"])
        diagnostic = dict(ref_vs_ref32=full_gap, split_vs_ref=gap(data["split"], data["ref"]))
        for mode in SCORED:
            arm_gap = gap(data[mode], data["ref"])
            diagnostic[mode] = dict(versus_ref=arm_gap, original_ratio=arm_gap / full_gap if full_gap else None)
        # All five steps against both references; split_premul vs split is the numerical-noise yardstick.
        diagnostic["versus_split"] = {m: agreement(data[m], data["split"]) for m in (*SCORED, "split_premul")}
        diagnostic["versus_split_premul"] = {m: agreement(data[m], data["split_premul"]) for m in SCORED}
        report = dict(
            criterion=CRITERION,
            verdict=verdict,
            all_pass=verdict == "pass",
            faults={m: meta[m]["fault"] for m in ARMS if meta[m]["fault"] != "none"},
            tolerance=dict(relative=REL_TOL, gradient_rms_floor=GRAD_RMS_FLOOR, post_norm=POST_TOL),
            invariants=invariants,
            validity_gate=gate,
            scored=scored,
            diagnostics=diagnostic,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        report = dict(criterion=CRITERION, verdict="fail", all_pass=False, error=str(error))
    (outdir / "bf16_report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(summary(report))
    return 0 if report["all_pass"] else 1


def summary(report):
    if "error" in report:
        return f"verdict=fail error={report['error']}"
    lines = [f"criterion {report['criterion']} verdict={report['verdict']} faults={report['faults'] or 'none'}"]
    failing = {m: [k for k, v in c.items() if k != "all_pass" and not v] for m, c in report["invariants"].items()}
    lines.append(f"invariants failing: {({m: f for m, f in failing.items() if f}) or 'none'}")
    for name, s in [("split_premul (validity gate)", report["validity_gate"]), *report["scored"].items()]:
        lines.append(
            f"{name}: pass={s['all_pass']} failing={[k for k, v in s['checks'].items() if not v]} "
            f"pre={s['pre_norm_rel']:.3g} update={s['update_rel_l2']:.3g} grad_max={s['max_gradient_relative']:.3g} "
            f"({s['gradients_failing']} over) update_max={s['max_tensor_update_relative']:.3g} "
            f"({s['tensor_updates_failing']} over)"
        )
    for reference in ("split", "split_premul"):
        for mode, result in report["diagnostics"][f"versus_{reference}"].items():
            steps = [f"{s['max_relative']:.2g}" for s in result["steps"]]
            lines.append(f"diagnostic {mode} vs {reference}, largest per-tensor gradient gap per step: {steps}")
    return "\n".join(lines)


def _fault_report(path):
    """Read a complete v2 report; execution failures cannot count as negative controls."""
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    if "error" in report or report.get("criterion") != CRITERION:
        raise ValueError(report.get("error", "missing or incompatible criterion"))
    invariant_keys = {
        "metadata",
        "steps",
        "init",
        "frozen",
        "communication",
        "clip_active",
        "clip_norm",
        "synchronized",
        "gradients",
        "dtype",
        "rows_and_shards",
    }
    score_keys = {"pre_norm", "gradients", "update", "tensor_updates"}

    def checks_complete(checks, expected, aggregate):
        return (
            set(checks) == expected
            and all(type(v) is bool for v in checks.values())
            and type(aggregate) is bool
            and aggregate == all(checks.values())
        )

    if set(report["invariants"]) != set(ARMS) or set(report["scored"]) != set(SCORED):
        raise ValueError("incomplete arm matrix")
    for checks in report["invariants"].values():
        if not checks_complete(
            {k: v for k, v in checks.items() if k != "all_pass"}, invariant_keys, checks["all_pass"]
        ):
            raise ValueError("incomplete invariant checks")
    for result in [report["validity_gate"], *report["scored"].values()]:
        if not checks_complete(result["checks"], score_keys, result["all_pass"]):
            raise ValueError("incomplete scored checks")
    if not isinstance(report["faults"], dict) or any(
        arm not in SCORED or fault not in FAULTS[1:] or (fault in DDP_FAULTS and arm != "ddp")
        for arm, fault in report["faults"].items()
    ):
        raise ValueError("invalid fault identity")
    verdict = (
        "fail"
        if not all(c["all_pass"] for c in report["invariants"].values())
        else "void"
        if not report["validity_gate"]["all_pass"]
        else "pass"
        if all(c["all_pass"] for c in report["scored"].values())
        else "fail"
    )
    if report["verdict"] != verdict or report["all_pass"] is not (verdict == "pass"):
        raise ValueError("inconsistent verdict")
    return report


def faults(outdir, expected_faults=None):
    """Require a passing baseline and fault-specific evidence for every expected arm/fault pair."""
    outdir = Path(outdir)
    expected = FAULTS[1:] if expected_faults is None else tuple(expected_faults.replace(":", " ").split())
    if not expected or len(set(expected)) != len(expected) or set(expected) - set(FAULTS[1:]):
        raise ValueError("expected faults must be a nonempty, unique list of known fault names")
    cases = {
        f"{fault}/{arm}": {"status": "not executed", "rejected": False}
        for fault in expected
        for arm in (SCORED if fault in LOSS_FAULTS else ("ddp",))
    }
    suites = {}
    errors = (OSError, ValueError, KeyError, TypeError, AttributeError)
    try:
        clean = _fault_report(outdir / "bf16_report.json")
        clean_pass = clean["verdict"] == "pass" and not clean["faults"]
        baseline = {"status": "pass" if clean_pass else "fail"}
    except errors as error:
        clean_pass = False
        baseline = {"status": "not executed", "error": str(error)}
    seen = set()
    for directory in sorted(outdir.glob("fault_*")):
        if not directory.is_dir():
            continue
        try:
            report = _fault_report(directory / "bf16_report.json")
            injected = report["faults"]
            if not injected or any(f not in expected for f in injected.values()):
                raise ValueError("missing or unexpected injected fault identity")
            gate = report["validity_gate"]["all_pass"]
            unaffected = all(
                report["invariants"][arm]["all_pass"] and (arm not in SCORED or report["scored"][arm]["all_pass"])
                for arm in ARMS
                if arm not in injected
            )
            suite = {"status": "executed", "gate": gate, "faults": injected}
            for arm, fault in injected.items():
                key = f"{fault}/{arm}"
                if key in seen:
                    raise ValueError(f"duplicate case {key}")
                seen.add(key)
                invariants = report["invariants"][arm]
                scored = report["scored"][arm]["checks"]
                evidence = []
                if fault == "fp32-backbone" and not invariants["dtype"]:
                    evidence.append("dtype")
                if fault in ("local-adapter", "skip-update") and not invariants["synchronized"]:
                    evidence.append("synchronized")
                if fault in (*LOSS_FAULTS, "scale-grad"):
                    evidence.extend(k for k in ("pre_norm", "gradients") if not scored[k])
                complete = all(invariants[k] for k in ("metadata", "steps", "init", "rows_and_shards"))
                rejected = (
                    clean_pass and gate and unaffected and complete and bool(evidence) and report["verdict"] == "fail"
                )
                cases[key] = {
                    "status": "not executed" if not complete else "rejected" if rejected else "not rejected",
                    "rejected": rejected,
                    "suite": directory.name,
                    "evidence": evidence,
                }
            suites[directory.name] = suite
        except errors as error:
            suites[directory.name] = {"status": "not executed", "error": str(error)}
    rejected = (
        clean_pass
        and all(c["rejected"] for c in cases.values())
        and all(s["status"] == "executed" for s in suites.values())
    )
    result = dict(rejected=rejected, baseline=baseline, cases=cases, suites=suites)
    (outdir / "faults_report.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if rejected else 1
