import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.precision import MixedPrecision
from lightning.pytorch.strategies import SingleDeviceStrategy

from lightning_sprite import (
    FSDP2Strategy,
    PALSEnvironment,
    XPUAccelerator,
    XPUDDPStrategy,
    XPUMixedPrecision,
    trainer_kwargs,
)


def _trainer(**kwargs):
    return Trainer(**kwargs, logger=False, enable_checkpointing=False, enable_progress_bar=False)


@pytest.mark.parametrize(("name", "strategy_type"), [("ddp", XPUDDPStrategy), ("fsdp", FSDP2Strategy)])
def test_multi_device_trainer_is_wired_for_xpu(fake_xpu, clean_launch_env, name, strategy_type):
    trainer = _trainer(**trainer_kwargs(name, devices=4))
    assert isinstance(trainer.accelerator, XPUAccelerator)
    assert type(trainer.strategy) is strategy_type
    assert trainer.strategy.root_device == torch.device("xpu", 0)
    assert len(trainer.strategy.parallel_devices) == 4


def test_mixed_precision_autocasts_on_xpu_not_cuda(fake_xpu, clean_launch_env):
    # Guard for Lightning #21982: a CUDA autocast on XPU silently trains in fp32.
    trainer = _trainer(**trainer_kwargs("ddp", devices=2, precision="bf16-mixed"))
    assert isinstance(trainer.precision_plugin, XPUMixedPrecision)
    assert trainer.precision_plugin.device == "xpu"


def test_bare_lightning_would_autocast_on_cuda(fake_xpu, clean_launch_env):
    # Documents the Lightning behavior the plugin exists to correct; if this starts failing, Lightning fixed it.
    trainer = _trainer(
        accelerator=XPUAccelerator(),
        devices=1,
        strategy=SingleDeviceStrategy(torch.device("xpu", 0)),
        precision="bf16-mixed",
    )
    assert isinstance(trainer.precision_plugin, MixedPrecision)
    assert trainer.precision_plugin.device == "cuda"


def test_true_precision_uses_lightning_flag(fake_xpu, clean_launch_env):
    kwargs = trainer_kwargs("ddp", devices=2, precision="32-true")
    assert kwargs["precision"] == "32-true" and "plugins" not in kwargs
    assert not isinstance(_trainer(**kwargs).precision_plugin, MixedPrecision)


def test_single_device_is_pinned_to_current_xpu(fake_xpu, clean_launch_env):
    fake_xpu["current"] = 2
    trainer = _trainer(**trainer_kwargs("single"))
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device == torch.device("xpu", 2)


def test_pals_environment_is_added_under_mpiexec(fake_xpu, clean_launch_env, monkeypatch):
    monkeypatch.setenv("PALS_RANKID", "0")
    monkeypatch.setenv("PALS_LOCAL_RANKID", "0")
    monkeypatch.setenv("PMI_SIZE", "4")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    trainer = _trainer(**trainer_kwargs("fsdp", devices=4))
    assert isinstance(trainer.strategy.cluster_environment, PALSEnvironment)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="'single', 'ddp' or 'fsdp'"):
        trainer_kwargs("deepspeed")


@pytest.mark.parametrize(("alias", "canonical"), [("bf16", "bf16-mixed"), ("16", "16-mixed"), (16, "16-mixed")])
def test_precision_aliases_get_xpu_autocast(fake_xpu, clean_launch_env, alias, canonical):
    # Lightning accepts these legacy spellings and would turn them into CUDA autocast on XPU.
    kwargs = trainer_kwargs("ddp", devices=2, precision=alias)
    (plugin,) = kwargs["plugins"]
    assert isinstance(plugin, XPUMixedPrecision)
    assert (plugin.precision, plugin.device) == (canonical, "xpu")
    assert "precision" not in kwargs


@pytest.mark.parametrize(("devices", "index"), [([3], 3), ("2,", 2), ("auto", 0), (1, 0)])
def test_single_device_honors_explicit_index(fake_xpu, clean_launch_env, devices, index):
    trainer = _trainer(**trainer_kwargs("single", devices=devices))
    assert trainer.strategy.root_device == torch.device("xpu", index)


@pytest.mark.parametrize("devices", [[0, 1], 2, "0,1"])
def test_single_device_rejects_several(fake_xpu, devices):
    with pytest.raises(ValueError, match="needs one device"):
        trainer_kwargs("single", devices=devices)
