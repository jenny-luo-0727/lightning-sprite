"""Mixed precision that autocasts on XPU."""

from __future__ import annotations

from typing import Literal

import torch
from lightning.pytorch.plugins.precision import MixedPrecision


class XPUMixedPrecision(MixedPrecision):
    """``MixedPrecision`` with the autocast device fixed to ``"xpu"``.

    With a non-CUDA accelerator, Lightning 2.6 builds ``MixedPrecision(precision, "cuda")`` for
    ``precision="bf16-mixed"`` or ``"16-mixed"`` (Lightning issue #21982). The CUDA autocast context does
    nothing on XPU, so the model silently trains in fp32. Pass this plugin through ``Trainer(plugins=[...])``
    and leave ``precision`` unset.
    """

    def __init__(
        self,
        precision: Literal["bf16-mixed", "16-mixed"] = "bf16-mixed",
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        super().__init__(precision, device="xpu", scaler=scaler)
