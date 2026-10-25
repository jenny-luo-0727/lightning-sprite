import socket
import sys
import types
import zlib

import pytest

from lightning_sprite import PALSEnvironment


def fake_mpi4py(monkeypatch, size):
    """Install an mpi4py stand-in whose COMM_WORLD reports ``size`` ranks (None blocks the import)."""
    if size is None:
        monkeypatch.setitem(sys.modules, "mpi4py", None)
        return
    module = types.ModuleType("mpi4py")
    module.MPI = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(Get_size=lambda: size))
    monkeypatch.setitem(sys.modules, "mpi4py", module)


@pytest.fixture
def pals_job(clean_launch_env, monkeypatch, tmp_path):
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("x4102c5s0b0n0.hostmgmt.example\nx4102c5s1b0n0.hostmgmt.example\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("PBS_JOBID", "1234.pbs.example")
    monkeypatch.setenv("PALS_RANKID", "13")
    monkeypatch.setenv("PALS_LOCAL_RANKID", "1")
    monkeypatch.setenv("PMI_SIZE", "24")
    monkeypatch.setattr(socket, "gethostname", lambda: "x4102c5s1b0n0")
    return nodefile


def test_detect(pals_job, monkeypatch):
    assert PALSEnvironment.detect()
    monkeypatch.delenv("PALS_RANKID")
    assert not PALSEnvironment.detect()


def test_ranks_and_rendezvous_from_pals(pals_job):
    env = PALSEnvironment()
    assert env.creates_processes_externally
    assert (env.global_rank(), env.local_rank(), env.world_size(), env.node_rank()) == (13, 1, 24, 1)
    assert env.main_address == "x4102c5s0b0n0.hostmgmt.example"
    assert env.main_port == 15000 + zlib.crc32(b"1234.pbs.example") % 20000


def test_torch_style_variables_used_without_pals(pals_job, monkeypatch):
    for name in ("PALS_RANKID", "PALS_LOCAL_RANKID", "PMI_SIZE"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "12")
    monkeypatch.setenv("MASTER_ADDR", "head")
    monkeypatch.setenv("MASTER_PORT", "1234")
    env = PALSEnvironment()
    assert (env.global_rank(), env.local_rank(), env.world_size()) == (2, 2, 12)
    assert (env.main_address, env.main_port) == ("head", 1234)


def test_agreeing_torch_and_pals_variables_are_accepted(pals_job, monkeypatch):
    monkeypatch.setenv("RANK", "13")
    monkeypatch.setenv("WORLD_SIZE", "24")
    assert (PALSEnvironment().global_rank(), PALSEnvironment().world_size()) == (13, 24)


@pytest.mark.parametrize(("name", "value"), [("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "4")])
def test_inherited_torch_variable_that_disagrees_raises(pals_job, monkeypatch, name, value):
    # An outer launcher's RANK=0 inherited by every mpiexec rank would otherwise make all of them rank 0.
    monkeypatch.setenv(name, value)
    env = PALSEnvironment()
    with pytest.raises(RuntimeError, match="Conflicting"):
        (env.global_rank(), env.local_rank(), env.world_size())


def test_array_jobs_get_different_ports(pals_job, monkeypatch):
    ports = set()
    for job in ("1234[1].pbs", "1234[2].pbs", "21234.pbs"):
        monkeypatch.setenv("PBS_JOBID", job)
        ports.add(PALSEnvironment().main_port)
    assert len(ports) == 3


def test_setters_do_not_override_launcher(pals_job):
    env = PALSEnvironment()
    env.set_global_rank(0)
    env.set_world_size(1)
    assert (env.global_rank(), env.world_size()) == (13, 24)


def test_validate_settings_rejects_rank_count_mismatch(pals_job):
    env = PALSEnvironment()
    env.validate_settings(num_devices=12, num_nodes=2)
    with pytest.raises(ValueError, match="--ppn"):
        env.validate_settings(num_devices=6, num_nodes=2)


def test_aurora_pmix_launch_takes_world_size_from_mpi4py(pals_job, monkeypatch):
    # What Aurora's mpiexec (PALS with PMIx) actually exports: PALS_* and PMIX_RANK, but no PMI_* and no world size.
    monkeypatch.delenv("PMI_SIZE")
    monkeypatch.setenv("PMIX_RANK", "13")
    monkeypatch.setenv("PALS_LOCAL_SIZE", "12")
    fake_mpi4py(monkeypatch, 24)
    env = PALSEnvironment()
    assert (env.global_rank(), env.local_rank(), env.world_size()) == (13, 1, 24)
    env.validate_settings(num_devices=12, num_nodes=2)


def test_world_size_variable_wins_over_mpi4py(pals_job, monkeypatch):
    fake_mpi4py(monkeypatch, 1)
    assert PALSEnvironment().world_size() == 24


def test_disagreeing_pmix_rank_raises(pals_job, monkeypatch):
    monkeypatch.setenv("PMIX_RANK", "0")
    with pytest.raises(RuntimeError, match="Conflicting"):
        PALSEnvironment().global_rank()


def test_missing_world_size_is_explicit(pals_job, monkeypatch):
    monkeypatch.delenv("PMI_SIZE")
    fake_mpi4py(monkeypatch, None)
    with pytest.raises(RuntimeError, match="mpi4py is not installed"):
        PALSEnvironment().world_size()


def test_unknown_host_in_multinode_job_raises(pals_job, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "elsewhere")
    with pytest.raises(RuntimeError, match="not in PBS_NODEFILE"):
        PALSEnvironment().node_rank()
