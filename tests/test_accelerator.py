import pytest
import torch
from lightning.pytorch import Trainer

from lightning_sprite import XPUAccelerator


@pytest.mark.parametrize(
    ("spec", "expected"),
    [(2, [0, 1]), ("2", [0, 1]), ("0,2", [0, 2]), ("1,", [1]), ([3], [3]), (-1, [0, 1, 2, 3]), ("-1", [0, 1, 2, 3])],
)
def test_parse_devices_follows_lightning_gpu_convention(fake_xpu, spec, expected):
    assert XPUAccelerator.parse_devices(spec) == expected


def test_parallel_devices_are_xpu(fake_xpu):
    assert XPUAccelerator.get_parallel_devices([0, 2]) == [torch.device("xpu", 0), torch.device("xpu", 2)]


def test_setup_device_rejects_non_xpu(fake_xpu):
    with pytest.raises(ValueError, match=r"non-XPU.*trainer_kwargs"):
        XPUAccelerator().setup_device(torch.device("cpu"))


def test_setup_device_explains_per_rank_affinity_mask(fake_xpu):
    fake_xpu["count"] = 1
    with pytest.raises(ValueError, match="ZE_AFFINITY_MASK"):
        XPUAccelerator().setup_device(torch.device("xpu", 3))


def test_trainer_accepts_accelerator_instance(fake_xpu, clean_launch_env):
    trainer = Trainer(accelerator=XPUAccelerator(), devices=1, strategy="auto", logger=False)
    assert isinstance(trainer.accelerator, XPUAccelerator)
