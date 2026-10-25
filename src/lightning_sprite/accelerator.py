"""Lightning accelerator for Intel XPU devices."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.accelerators import Accelerator


def xpu_available() -> bool:
    """Return whether torch can see an Intel XPU."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _parse_devices(devices: Any) -> list[int]:
    """Resolve a Lightning ``devices`` spec to XPU indices, following Lightning's GPU convention.

    An int or a comma-free string is a count (``-1`` = all visible); a string with commas (``"0,2"``,
    ``"1,"``) or a list is explicit indices.
    """
    if isinstance(devices, str):
        spec = devices.strip()
        if "," in spec:
            return [int(token) for token in spec.split(",") if token.strip()]
        devices = int(spec)
    if isinstance(devices, int):
        return list(range(torch.xpu.device_count() if devices == -1 else devices))
    return [int(index) for index in devices]


class XPUAccelerator(Accelerator):
    """Intel XPU accelerator.

    Prefer :func:`lightning_sprite.trainer_kwargs`. Used directly, pass an instance as
    ``Trainer(accelerator=XPUAccelerator(), strategy=...)`` with an explicit strategy: for a single device
    Lightning would otherwise choose a CPU ``SingleDeviceStrategy``, and for several its own ``DDPStrategy``,
    which calls ``torch.cuda`` when wrapping the model. Pass the accelerator only to the Trainer: Lightning
    refuses one set on both the strategy and the Trainer.

    Each process selects its device as ``xpu:<local_rank>`` from the full set of tiles on its node. A
    launcher that masks every rank down to one tile (a per-rank ``ZE_AFFINITY_MASK``) leaves rank ``k`` with
    only ``xpu:0`` and fails in :meth:`setup_device` with an explicit error.
    """

    def setup_device(self, device: torch.device) -> None:
        """Make ``device`` the current XPU.

        Raises:
            ValueError: If ``device`` is not an XPU device or is not visible to this process.
        """
        if device.type != "xpu":
            raise ValueError(
                f"XPUAccelerator received a non-XPU device: {device}. Lightning picks a CPU strategy for an "
                "accelerator it does not know; use lightning_sprite.trainer_kwargs(...) or pass an explicit strategy."
            )
        visible = torch.xpu.device_count()
        if device.index is not None and device.index >= visible:
            raise ValueError(
                f"This process sees {visible} XPU device(s) but was assigned {device}. Ranks select their tile "
                "by local rank, so launch without a per-rank ZE_AFFINITY_MASK (every rank must see all tiles "
                "on its node)."
            )
        torch.xpu.set_device(device)

    def teardown(self) -> None:
        """Release cached allocator blocks."""
        torch.xpu.empty_cache()

    def get_device_stats(self, device: Any) -> dict[str, Any]:
        """Return the XPU allocator statistics for ``device``."""
        return dict(torch.xpu.memory_stats(device))

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        """Resolve a Lightning ``devices`` spec to XPU indices."""
        return _parse_devices(devices)

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[torch.device]:
        """Map XPU indices to ``torch.device`` objects."""
        return [torch.device("xpu", index) for index in _parse_devices(devices)]

    @staticmethod
    def auto_device_count() -> int:
        """Return the number of XPU devices visible to this process."""
        return int(torch.xpu.device_count())

    @staticmethod
    def is_available() -> bool:
        """Return whether an XPU is visible."""
        return xpu_available()

    @staticmethod
    def name() -> str:
        """Return the accelerator's registry name."""
        return "xpu"

    @classmethod
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        """Register this accelerator under ``"xpu"``."""
        accelerator_registry.register("xpu", cls, description=cls.__name__)
