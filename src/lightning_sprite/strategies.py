"""DDP and FSDP2 strategies that run on XPU."""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.fabric.utilities.distributed import _init_dist_connection
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.strategies import DDPStrategy, ModelParallelStrategy
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.trainer.states import TrainerFn
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from lightning_sprite.fsdp import check_trainable_sharded


class XPUDDPStrategy(DDPStrategy):
    """``DDPStrategy`` that wraps the model without a CUDA stream on XPU.

    Lightning 2.6 builds ``DistributedDataParallel`` inside ``torch.cuda.stream(torch.cuda.Stream())`` for any
    non-CPU device (``lightning/pytorch/strategies/ddp.py``). A torch build without CUDA raises on that call.
    On XPU this strategy builds the wrapper on the default stream; on other devices it defers to Lightning.
    DDP communication hooks stay CUDA-only, as in Lightning.
    """

    def _setup_model(self, model: Module) -> DistributedDataParallel:
        if self.root_device.type != "xpu":
            return super()._setup_model(model)
        return DistributedDataParallel(module=model, device_ids=self.determine_ddp_device_ids(), **self._ddp_kwargs)


class FSDP2Strategy(ModelParallelStrategy):
    """``ModelParallelStrategy`` configured as plain FSDP2 over every process, with a sharding check.

    Two Lightning defaults are changed:

    * ``ModelParallelStrategy``'s ``"auto"`` sizes give tensor parallelism across the processes of a node and
      data parallelism only across nodes. On one 12-tile Aurora node that is a data-parallel mesh of size 1, so
      sharding over ``device_mesh["data_parallel"]`` shards nothing. Here ``data_parallel_size`` defaults to
      every process and ``tensor_parallel_size`` to 1.
    * When fitting, after ``configure_model`` and before the optimizers are built, every trainable parameter
      must be managed by a ``fully_shard`` module. A trainable module outside every ``fully_shard`` root, such
      as a task head beside the backbone, would have gradients that are never reduced across ranks; this
      raises instead.
    * With an external launcher (``mpiexec``, ``srun``), the launched process count is checked against
      ``devices x num_nodes``. Lightning runs that check only in DDP's subprocess launcher, which
      ``ModelParallelStrategy`` skips for externally created processes.
    * On XPU the default process group is created without ``device_id``. A bound group makes
      ``init_device_mesh`` build each mesh dimension with ``split_group``, which ALCF's XCCL build rejects
      ("its backend does not support splitting", Aurora, torch 2.13). Unbound, it uses ``new_group``.

    The model must still be sharded in ``LightningModule.configure_model``, e.g. with
    :func:`lightning_sprite.shard_model`. Lightning's ``FSDPStrategy`` (FSDP1) is not supported: it refuses
    global-norm gradient clipping and hardcodes CUDA autocast.
    """

    def __init__(
        self,
        data_parallel_size: str | int | None = None,
        tensor_parallel_size: str | int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_parallel_size="auto" if data_parallel_size is None else data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
        self._data_parallel_size_all = data_parallel_size is None

    def _resolve_mesh_sizes(self) -> None:
        """Give every process to data parallelism unless the caller set the data-parallel size."""
        if self._data_parallel_size_all:
            if self._tensor_parallel_size == "auto":
                self._tensor_parallel_size = 1
            self._data_parallel_size = self.num_nodes * self.num_processes // int(self._tensor_parallel_size)

    def setup_environment(self) -> None:
        self._resolve_mesh_sizes()
        if self.cluster_environment is not None and self.cluster_environment.creates_processes_externally:
            self.cluster_environment.validate_settings(num_devices=self.num_processes, num_nodes=self.num_nodes)
        super().setup_environment()

    def _setup_distributed(self) -> None:
        if self.root_device.type != "xpu":
            super()._setup_distributed()
            return
        # Lightning 2.6's ModelParallelStrategy._setup_distributed, minus device_id (see the class docstring).
        ParallelStrategy.setup_environment(self)
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def setup(self, trainer: pl.Trainer) -> None:
        if trainer.state.fn == TrainerFn.FITTING:
            check_trainable_sharded(self.lightning_module)
        super().setup(trainer)
