"""FSDP2 sharding for Hugging Face decoders, including peft-wrapped ones."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.nn import Module


def decoder_layer_classes(model: Module) -> set[type[Module]]:
    """Return the classes Hugging Face marks as unsplittable blocks in ``model`` (its ``_no_split_modules``).

    The attribute is searched on every submodule, so a model wrapped by peft (``PeftModel`` ->
    ``LoraModel`` -> the transformers model) is handled the same as the bare model.
    """
    names: set[str] = set()
    for module in model.modules():
        names.update(getattr(module, "_no_split_modules", None) or ())
    return {type(module) for module in model.modules() if type(module).__name__ in names}


def shard_model(
    model: Module,
    mesh: DeviceMesh,
    layer_classes: Iterable[type[Module]] | None = None,
    **fully_shard_kwargs: Any,
) -> Module:
    """Apply FSDP2 ``fully_shard`` to each decoder layer of ``model`` and then to ``model`` itself.

    Call this from ``LightningModule.configure_model`` with ``self.device_mesh["data_parallel"]``. ``model``
    must contain every trainable parameter (backbone, adapters, and any task head);
    :class:`~lightning_sprite.FSDP2Strategy` checks this before training.

    Layers are sharded bottom-up, as ``fully_shard`` requires, so a matched block nested inside another
    matched block is sharded first. The call is idempotent: Lightning runs ``configure_model`` again for
    each later ``validate``/``test``/``predict`` in the same process, and an already-sharded ``model`` is
    returned unchanged.

    Args:
        model: The module to shard; it becomes the FSDP2 root.
        mesh: The data-parallel device mesh.
        layer_classes: Block classes to shard individually. Defaults to :func:`decoder_layer_classes`.
        **fully_shard_kwargs: Passed to every ``fully_shard`` call (e.g. ``mp_policy``,
            ``reshard_after_forward``).

    Returns:
        ``model``, sharded in place.

    Raises:
        ValueError: If no layer classes are given or found.
    """
    from torch.distributed.fsdp import fully_shard

    if isinstance(model, FSDPModule):
        return model
    classes = tuple(layer_classes) if layer_classes is not None else tuple(decoder_layer_classes(model))
    if not classes:
        raise ValueError("No decoder layer classes found in model._no_split_modules; pass layer_classes.")
    # model.modules() is pre-order (parents before children); reversed, every module follows its descendants.
    for layer in reversed([module for module in model.modules() if isinstance(module, classes)]):
        fully_shard(layer, mesh=mesh, **fully_shard_kwargs)
    fully_shard(model, mesh=mesh, **fully_shard_kwargs)
    return model


def check_trainable_sharded(module: Module) -> None:
    """Raise if any trainable parameter of ``module`` is not managed by a ``fully_shard`` module.

    A parameter is managed when it belongs to a module that is, or sits inside, an ``FSDPModule``. Being a
    ``DTensor`` is not enough: tensor parallelism alone also produces ``DTensor`` parameters, whose gradients
    are never reduced across the data-parallel ranks.

    Raises:
        RuntimeError: Naming the unmanaged trainable parameters.
    """
    roots = [name for name, child in module.named_modules() if isinstance(child, FSDPModule)]

    def managed(param_name: str) -> bool:
        owner = param_name.rpartition(".")[0]
        return any(root == "" or owner == root or owner.startswith(root + ".") for root in roots)

    unsharded = [name for name, param in module.named_parameters() if param.requires_grad and not managed(name)]
    if unsharded:
        shown = ", ".join(unsharded[:5]) + (f" (+{len(unsharded) - 5} more)" if len(unsharded) > 5 else "")
        raise RuntimeError(
            f"{len(unsharded)} trainable parameter(s) are outside every fully_shard root, so their gradients "
            f"would never be reduced across ranks: {shown}. Shard a root module that contains them."
        )
