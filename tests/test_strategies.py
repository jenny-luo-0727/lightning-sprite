from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn

import lightning_sprite.strategies as strategies
from lightning_sprite import FSDP2Strategy, PALSEnvironment, XPUDDPStrategy


def test_ddp_wraps_on_xpu_without_cuda_stream(monkeypatch):
    calls = {}

    def fake_ddp(module, device_ids, **kwargs):
        calls.update(module=module, device_ids=device_ids, kwargs=kwargs)
        return module

    def no_cuda_stream(*args, **kwargs):
        raise AssertionError("torch.cuda.Stream must not be used on XPU")

    monkeypatch.setattr(strategies, "DistributedDataParallel", fake_ddp)
    monkeypatch.setattr(torch.cuda, "Stream", no_cuda_stream)
    strategy = XPUDDPStrategy(parallel_devices=[torch.device("xpu", 0)], find_unused_parameters=True)
    model = nn.Linear(2, 2)
    assert strategy._setup_model(model) is model
    assert calls["device_ids"] == [0]
    assert calls["kwargs"] == {"find_unused_parameters": True}


def test_ddp_keeps_lightning_internals_it_relies_on():
    # XPUDDPStrategy reads these private Lightning attributes; a Lightning release that renames them
    # must fail here, not on Aurora.
    strategy = XPUDDPStrategy()
    assert isinstance(strategy._ddp_kwargs, dict)
    assert callable(strategy.determine_ddp_device_ids)


@pytest.mark.parametrize(
    ("kwargs", "nodes", "procs", "expected"),
    [
        ({}, 2, 12, (24, 1)),
        ({"tensor_parallel_size": "auto"}, 1, 12, (12, 1)),
        ({"tensor_parallel_size": 2}, 1, 12, (6, 2)),
        ({"data_parallel_size": 3, "tensor_parallel_size": 4}, 1, 12, (3, 4)),
    ],
)
def test_fsdp2_mesh_defaults_to_pure_data_parallel(kwargs, nodes, procs, expected):
    strategy = FSDP2Strategy(**kwargs)
    strategy.num_nodes = nodes
    strategy.parallel_devices = [torch.device("xpu", i) for i in range(procs)]
    strategy._resolve_mesh_sizes()
    assert (strategy._data_parallel_size, strategy._tensor_parallel_size) == expected


def test_fsdp2_validates_external_launch_size(clean_launch_env, monkeypatch):
    # ModelParallelStrategy builds no launcher for externally started processes, so Lightning never validates.
    monkeypatch.setenv("PALS_RANKID", "0")
    monkeypatch.setenv("PALS_LOCAL_RANKID", "0")
    monkeypatch.setenv("PMI_SIZE", "4")
    strategy = FSDP2Strategy()
    strategy.cluster_environment = PALSEnvironment()
    strategy.parallel_devices = [torch.device("xpu", 0), torch.device("xpu", 1)]
    with pytest.raises(ValueError, match="--ppn"):
        strategy.setup_environment()


@pytest.mark.parametrize(("device", "bound"), [(torch.device("xpu", 1), False), (torch.device("cuda", 1), True)])
def test_fsdp2_binds_no_device_to_the_xpu_process_group(monkeypatch, device, bound):
    # A bound default group sends init_device_mesh down split_group, which XCCL on Aurora rejects.
    calls = {}
    monkeypatch.setattr(
        strategies, "_init_dist_connection", lambda env, backend, **kwargs: calls.setdefault("xpu", kwargs)
    )
    monkeypatch.setattr(
        "lightning.pytorch.strategies.model_parallel._init_dist_connection",
        lambda env, backend, **kwargs: calls.setdefault("lightning", kwargs),
    )
    strategy = FSDP2Strategy(process_group_backend="xccl" if device.type == "xpu" else "nccl")
    strategy.cluster_environment = SimpleNamespace(
        local_rank=lambda: 0,
        node_rank=lambda: 0,
        global_rank=lambda: 0,
        world_size=lambda: 1,
        set_global_rank=lambda r: None,
        set_world_size=lambda w: None,
    )
    strategy.parallel_devices = [device]
    strategy.accelerator = SimpleNamespace(setup_device=lambda device: None)
    strategy._setup_distributed()
    (kwargs,) = calls.values()
    assert ("device_id" in kwargs) == bound
    assert "timeout" in kwargs


@pytest.mark.parametrize(("fn", "raises"), [(TrainerFn.FITTING, True), (TrainerFn.PREDICTING, False)])
def test_fsdp2_checks_sharding_only_when_fitting(monkeypatch, fn, raises):
    monkeypatch.setattr(ModelParallelStrategy, "setup", lambda self, trainer: None)
    strategy = FSDP2Strategy()
    strategy._lightning_module = nn.Linear(2, 2)  # trainable and unsharded
    trainer = SimpleNamespace(state=SimpleNamespace(fn=fn))
    if raises:
        with pytest.raises(RuntimeError, match="outside every fully_shard root"):
            strategy.setup(trainer)
    else:
        strategy.setup(trainer)
