import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture
def fake_xpu(monkeypatch):
    """Make torch report four visible XPUs, so Trainer construction runs on a machine without one."""
    state = {"count": 4, "current": 0}
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: state["count"])
    monkeypatch.setattr(torch.xpu, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.xpu, "set_device", lambda device: None)
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: None)
    return state


@pytest.fixture
def clean_launch_env(monkeypatch):
    """Remove launcher variables that would change Lightning's cluster-environment detection."""
    for name in list(os.environ):
        if name.startswith(("PALS_", "PMI_", "PMIX_", "SLURM_", "PBS_", "OMPI_", "TORCHELASTIC_")) or name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
        ):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gloo_world_of_one(clean_launch_env, monkeypatch):
    """A single-process gloo group and its CPU device mesh."""
    from torch.distributed.device_mesh import init_device_mesh

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29533")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()
