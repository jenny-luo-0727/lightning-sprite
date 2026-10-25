from typing import ClassVar

import pytest
import torch
import torch.distributed.fsdp as fsdp_api
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from lightning_sprite import check_trainable_sharded, decoder_layer_classes, shard_model


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.adapter = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.base(x) + self.adapter(x)


class Decoder(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["Block"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Wrapped(nn.Module):
    """Toy decoder with a two-class head, exercising PEFT-style wrapper nesting."""

    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = Decoder()
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.base_model.model(x))


def _lora_like(model):
    for name, param in model.named_parameters():
        param.requires_grad_("adapter" in name or name.startswith("head"))
    return model


def test_layer_classes_found_through_wrappers():
    assert decoder_layer_classes(Wrapped()) == {Block}


def test_no_layer_classes_is_explicit(gloo_world_of_one):
    with pytest.raises(ValueError, match="layer_classes"):
        shard_model(nn.Linear(2, 2), gloo_world_of_one)


def test_shard_model_shards_each_layer_and_root(gloo_world_of_one):
    model = shard_model(_lora_like(Wrapped()), gloo_world_of_one)
    assert isinstance(model, FSDPModule)
    assert all(isinstance(layer, FSDPModule) for layer in model.base_model.model.layers)
    assert all(isinstance(param, DTensor) for param in model.parameters())
    check_trainable_sharded(model)


def test_mixed_trainable_and_frozen_in_one_unit_trains(gloo_world_of_one):
    torch.manual_seed(0)
    model = shard_model(_lora_like(Wrapped()), gloo_world_of_one)
    frozen = {n: p.detach().to_local().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    nn.functional.cross_entropy(model(torch.randn(3, 4)), labels).backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert all(torch.equal(p.detach().to_local(), frozen[n]) for n, p in model.named_parameters() if n in frozen)


def test_trainable_head_outside_root_is_caught(gloo_world_of_one):
    container = nn.Module()
    container.net = shard_model(_lora_like(Wrapped()), gloo_world_of_one)
    container.extra_head = nn.Linear(4, 2)
    with pytest.raises(RuntimeError, match=r"extra_head\.weight"):
        check_trainable_sharded(container)


def test_dtensor_parameter_without_fully_shard_is_caught(gloo_world_of_one):
    # Tensor parallelism alone yields DTensor parameters whose gradients are never reduced across
    # data-parallel ranks; being a DTensor must not count as sharded.
    model = _lora_like(Wrapped())
    head = model.head
    head.weight = nn.Parameter(distribute_tensor(head.weight.detach(), gloo_world_of_one, [Replicate()]))
    with pytest.raises(RuntimeError, match=r"head\.weight"):
        check_trainable_sharded(model)


def test_shard_model_is_idempotent(gloo_world_of_one):
    # Lightning calls configure_model again for validate/test/predict after fit.
    model = shard_model(_lora_like(Wrapped()), gloo_world_of_one)
    assert shard_model(model, gloo_world_of_one) is model


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.out = nn.Linear(4, 4)

    def forward(self, x):
        return self.out(self.inner(x))


class NestedDecoder(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["Outer", "Inner"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Outer(), Outer()])
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def test_nested_matched_blocks_are_sharded_bottom_up(gloo_world_of_one, monkeypatch):
    order = []
    real = fsdp_api.fully_shard

    def recording(module, **kwargs):
        order.append(type(module).__name__)
        return real(module, **kwargs)

    monkeypatch.setattr(fsdp_api, "fully_shard", recording)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("data_parallel",))["data_parallel"]
    model = shard_model(NestedDecoder(), mesh)
    assert order == ["Inner", "Outer", "Inner", "Outer", "NestedDecoder"]
    labels = torch.tensor([0, 1], dtype=torch.int64)
    nn.functional.cross_entropy(model(torch.randn(2, 4)), labels).backward()
