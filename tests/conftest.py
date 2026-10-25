"""The tests run as on a bare machine, whatever the installation's config/env says: local execution (no SLURM:
a machine whose controller was down failed three benchmark tests with 'could not submit', 2026.10.22), results in
a temporary folder (not the real CFDT_OUT), one rank, no caps. Tests that need SLURM set it up themselves (the
fake SLURM of test_slurm.py), after this fixture."""
from __future__ import annotations

import pytest

_CLEAR = ("CFDT_SLURM_ARGS", "CFDT_SLURM_MIN_MINUTES", "CFDT_MAX_TIMEOUT", "CFDT_MAX_MEM", "CFDT_MAX_PROCS",
          "CFDT_CELLS_PER_RANK", "CFDT_MPIRUN", "SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_MEM_PER_NODE")


@pytest.fixture(autouse=True)
def _bare_machine(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("CFDT_EXECUTOR", "local")
    monkeypatch.setenv("CFDT_OUT", str(tmp_path_factory.mktemp("cfdt_out")))
    monkeypatch.setenv("CFDT_PROCS", "1")
    for k in _CLEAR:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("CFDT_SLURM_NODES", raising=False)
    from cfd_translator import executor
    executor._NODES.clear()                  # sinfo's answer is cached per process
    executor._JOB_NODE.clear()
