"""Multi-GPU PyTorch Lightning on Intel XPU (DDP + FSDP2)."""

import importlib.metadata

from lightning_sprite.accelerator import XPUAccelerator, xpu_available
from lightning_sprite.environment import PALSEnvironment
from lightning_sprite.fsdp import check_trainable_sharded, decoder_layer_classes, shard_model
from lightning_sprite.precision import XPUMixedPrecision
from lightning_sprite.strategies import FSDP2Strategy, XPUDDPStrategy
from lightning_sprite.trainer import trainer_kwargs

try:
    __version__ = importlib.metadata.version("lightning-sprite")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "FSDP2Strategy",
    "PALSEnvironment",
    "XPUAccelerator",
    "XPUDDPStrategy",
    "XPUMixedPrecision",
    "check_trainable_sharded",
    "decoder_layer_classes",
    "shard_model",
    "trainer_kwargs",
    "xpu_available",
]
