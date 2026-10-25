"""One call that returns the device-related ``Trainer`` arguments for XPU."""

from __future__ import annotations

from typing import Any, Literal

import torch
from lightning.fabric.connector import _convert_precision_to_unified_args
from lightning.fabric.plugins.environments import ClusterEnvironment
from lightning.pytorch.strategies import SingleDeviceStrategy

from lightning_sprite.accelerator import XPUAccelerator
from lightning_sprite.environment import PALSEnvironment
from lightning_sprite.precision import XPUMixedPrecision
from lightning_sprite.strategies import FSDP2Strategy, XPUDDPStrategy

MIXED_PRECISIONS = ("bf16-mixed", "16-mixed")


def trainer_kwargs(
    strategy: Literal["single", "ddp", "fsdp"] = "ddp",
    *,
    devices: int | str | list[int] = "auto",
    num_nodes: int = 1,
    precision: str | int = "bf16-mixed",
    cluster_environment: ClusterEnvironment | None = None,
    **strategy_kwargs: Any,
) -> dict[str, Any]:
    """Return ``accelerator``, ``devices``, ``num_nodes``, ``strategy``, ``plugins`` and ``precision`` for XPU.

    Usage: ``Trainer(**trainer_kwargs("fsdp", devices=12, num_nodes=2), max_epochs=...)``.

    * The accelerator is an :class:`XPUAccelerator` instance, passed only through the Trainer.
    * ``"single"`` pins a ``SingleDeviceStrategy`` to one XPU: the current one for ``devices="auto"`` or ``1``,
      else the one index given (``[3]`` or ``"3,"``). Left to itself, Lightning would build a CPU
      ``SingleDeviceStrategy`` for an accelerator it does not recognize.
    * ``"ddp"`` and ``"fsdp"`` select :class:`XPUDDPStrategy` and :class:`FSDP2Strategy`; ``strategy_kwargs``
      go to their constructors.
    * A mixed precision, including Lightning's legacy aliases (``"bf16"``, ``"16"``, ``16``), becomes an
      :class:`XPUMixedPrecision` plugin with ``precision`` left unset.
    * Under PALS (``mpiexec`` on Aurora), a :class:`PALSEnvironment` is added unless ``cluster_environment``
      is given.

    Returns:
        Keyword arguments for ``lightning.Trainer``.
    """
    plugins: list[Any] = []
    kwargs: dict[str, Any] = {"accelerator": XPUAccelerator(), "devices": devices, "num_nodes": num_nodes}

    if strategy == "single":
        if strategy_kwargs:
            raise TypeError(f"strategy='single' takes no strategy arguments, got {sorted(strategy_kwargs)}")
        if devices in ("auto", 1, "1"):
            index = torch.xpu.current_device()
        else:
            indices = XPUAccelerator.parse_devices(devices) if isinstance(devices, (list, tuple, str)) else []
            if len(indices) != 1:
                raise ValueError(f"strategy='single' needs one device; got devices={devices!r}")
            index = indices[0]
        root = torch.device("xpu", index)
        kwargs.update(strategy=SingleDeviceStrategy(device=root), devices=[root.index])
    elif strategy == "ddp":
        kwargs["strategy"] = XPUDDPStrategy(**strategy_kwargs)
    elif strategy == "fsdp":
        kwargs["strategy"] = FSDP2Strategy(**strategy_kwargs)
    else:
        raise ValueError(f"strategy must be 'single', 'ddp' or 'fsdp', got {strategy!r}")

    precision = _convert_precision_to_unified_args(precision)
    if precision in MIXED_PRECISIONS:
        plugins.append(XPUMixedPrecision(precision))
    else:
        kwargs["precision"] = precision

    if cluster_environment is None and strategy != "single" and PALSEnvironment.detect():
        cluster_environment = PALSEnvironment()
    if cluster_environment is not None:
        plugins.append(cluster_environment)
    if plugins:
        kwargs["plugins"] = plugins
    return kwargs
