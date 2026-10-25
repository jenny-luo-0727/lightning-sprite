"""Cluster environment for processes started by ``mpiexec`` under PALS (ALCF Aurora)."""

from __future__ import annotations

import os
import socket
import zlib

from lightning.fabric.plugins.environments import ClusterEnvironment

# Every variable that is set is read, and they must agree; the value is taken from the first set one. A
# torch-style value (RANK / LOCAL_RANK / WORLD_SIZE, which job scripts often export) that disagrees with the
# launcher's own variable is usually inherited from an outer launcher, so it raises instead of silently
# making every rank rank 0. Measured on Aurora (mpich 5.0 under PALS with PMIx, 2026-10-23): mpiexec sets
# PALS_RANKID, PALS_LOCAL_RANKID and PMIX_RANK, but no PMI_* variable and nothing that holds the world size
# (PALS_LOCAL_SIZE is per node). The world size therefore falls back to mpi4py, which ALCF's frameworks
# module ships. PMI_* stays for launchers configured with PMI instead of PMIx.
RANK_VARS = ("RANK", "PALS_RANKID", "PMIX_RANK", "PMI_RANK")
LOCAL_RANK_VARS = ("LOCAL_RANK", "PALS_LOCAL_RANKID", "PMI_LOCAL_RANK")
WORLD_SIZE_VARS = ("WORLD_SIZE", "PMI_SIZE")
DEFAULT_PORT = 29500


def _agreed_int(names: tuple[str, ...], what: str) -> int:
    values = {name: int(os.environ[name]) for name in names if os.environ.get(name) not in (None, "")}
    if not values:
        raise RuntimeError(f"No {what} found; set one of {', '.join(names)}.")
    if len(set(values.values())) > 1:
        shown = ", ".join(f"{name}={value}" for name, value in values.items())
        raise RuntimeError(
            f"Conflicting {what} variables ({shown}). A stale value inherited from an outer launcher is the usual "
            "cause; unset it or make it match the mpiexec launch."
        )
    return next(iter(values.values()))


def _mpi_world_size() -> int | None:
    """Return ``MPI.COMM_WORLD``'s size, or None when mpi4py is not installed."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD.Get_size()


def _short(hostname: str) -> str:
    return hostname.split(".")[0]


def _nodefile_hosts() -> list[str]:
    """Return the job's hosts in PBS_NODEFILE order, without duplicates."""
    path = os.environ.get("PBS_NODEFILE")
    if not path or not os.path.exists(path):
        return []
    hosts: list[str] = []
    with open(path) as handle:
        for line in handle:
            host = line.strip()
            if host and host not in hosts:
                hosts.append(host)
    return hosts


class PALSEnvironment(ClusterEnvironment):
    """Rank and rendezvous information for processes that ``mpiexec`` (PALS) has already started.

    Lightning detects an MPI launch only through ``MPIEnvironment``, which needs ``mpi4py``. Without it,
    Lightning falls back to ``LightningEnvironment``, which spawns its own workers. Under ``mpiexec`` every
    rank would then act as rank 0. This environment reads the ranks from the launcher's variables and tells
    Lightning the processes already exist.

    Args:
        main_address: Rendezvous host. Defaults to ``MASTER_ADDR``, else the first host in ``PBS_NODEFILE``.
        main_port: Rendezvous port. Defaults to ``MASTER_PORT``, else a port derived from the full
            ``PBS_JOBID`` (array index included), so every rank of one job agrees and different jobs usually
            differ. Two concurrent ``mpiexec`` launches inside one job derive the same port: give each its own
            ``MASTER_PORT``.
    """

    def __init__(self, main_address: str | None = None, main_port: int | None = None) -> None:
        super().__init__()
        self._main_address = main_address
        self._main_port = main_port
        self._mpi_world_size: int | None = None

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        if self._main_address is None:
            address = os.environ.get("MASTER_ADDR")
            if not address:
                hosts = _nodefile_hosts()
                if not hosts:
                    raise RuntimeError("Set MASTER_ADDR or run inside a PBS job with PBS_NODEFILE.")
                address = hosts[0]
            self._main_address = address
        return self._main_address

    @property
    def main_port(self) -> int:
        if self._main_port is None:
            port = os.environ.get("MASTER_PORT")
            if port:
                self._main_port = int(port)
            else:
                job = os.environ.get("PBS_JOBID")
                self._main_port = 15000 + zlib.crc32(job.encode()) % 20000 if job else DEFAULT_PORT
        return self._main_port

    @staticmethod
    def detect() -> bool:
        return "PALS_RANKID" in os.environ

    def world_size(self) -> int:
        if any(os.environ.get(name) not in (None, "") for name in WORLD_SIZE_VARS):
            return _agreed_int(WORLD_SIZE_VARS, "world size")
        if self._mpi_world_size is None:
            self._mpi_world_size = _mpi_world_size()
            if self._mpi_world_size is None:
                raise RuntimeError(
                    f"No world size found: none of {', '.join(WORLD_SIZE_VARS)} is set and mpi4py is not installed. "
                    "PALS under PMIx exports no world-size variable; install mpi4py or pass --env WORLD_SIZE=<ranks> "
                    "to mpiexec."
                )
        return self._mpi_world_size

    def set_world_size(self, size: int) -> None:
        pass  # fixed by the launcher

    def global_rank(self) -> int:
        return _agreed_int(RANK_VARS, "global rank")

    def set_global_rank(self, rank: int) -> None:
        pass  # fixed by the launcher

    def local_rank(self) -> int:
        return _agreed_int(LOCAL_RANK_VARS, "local rank")

    def node_rank(self) -> int:
        hosts = [_short(host) for host in _nodefile_hosts()]
        here = _short(socket.gethostname())
        if here in hosts:
            return hosts.index(here)
        if len(hosts) <= 1:
            return 0
        raise RuntimeError(f"Host {here!r} is not in PBS_NODEFILE ({len(hosts)} hosts).")

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        expected = num_devices * num_nodes
        if self.world_size() != expected:
            raise ValueError(
                f"mpiexec started {self.world_size()} ranks but the Trainer expects devices={num_devices} x "
                f"num_nodes={num_nodes} = {expected}. Launch one rank per device (e.g. --ppn 12 on Aurora)."
            )
