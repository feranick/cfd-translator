"""cfdt on llmt-runtime (2026.11.09.2, phase 1): the shared pieces are the runtime's, configured for cfdt.
Version: 2026.11.09.2
"""
import json
import os
from pathlib import Path

import llmt_runtime
from llmt_runtime import memory as M
from llmt_runtime import workers as W

from cfd_translator import __version__, api, executor, foam, workers
from cfd_translator.runtime import RT


def test_runtime_configured_for_cfdt(monkeypatch):
    assert (RT.prefix, RT.cli, RT.package, RT.kind, RT.one_node) == ("CFDT", "cfdt", "cfd_translator", "mpi", True)
    monkeypatch.setenv("CFDT_OUT", "/tmp/cfdt-x")
    assert RT.out_root() == Path("/tmp/cfdt-x") == api.out_root()
    assert RT.hooks.study_procs({"solve": {"procs": 6}}) == 6


def test_runtime_version_reported(monkeypatch):
    monkeypatch.setenv("CFDT_EXECUTOR", "local")
    assert executor.status()["runtime"] == llmt_runtime.__version__


def test_workers_are_the_runtimes_with_foam_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(foam, "kill_all", lambda: calls.append(1))
    monkeypatch.setattr(W, "_HOOKS", [])
    workers._kill_foam()
    assert calls == [1]
    monkeypatch.setattr(W, "watch_parent", lambda ppid, name, **kw: calls.append(name))
    workers.watch_parent(os.getpid())
    assert calls[-1] == "cfdt" and workers._kill_foam in W._HOOKS


def test_final_state_oom_kill_wording_with_accounting(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"procs": 4, "command": ["sbatch", "--parsable", "--mem=8G", "job.sh"]}))
    log = "slurmstepd: error: Detected 2 oom_kill events in StepId=5807.batch. Some of the step tasks have been OOM Killed."
    state, why = executor.final_state("FAILED", log, d)          # accounted as FAILED: the log says oom_kill
    assert state == "out_of_memory" and why.startswith("SLURM killed the job for memory (8G with 4 ranks)")
    assert '"run": {"mem"' in why
    assert executor.final_state(None, "(no job.log)")[0] == "cancelled"
    assert executor.final_state("NODE_FAIL", "(no job.log)")[0] == "failed"


def test_memory_hint_from_the_runtimes_limit(monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setattr(M, "memory_limit_mb", lambda: 4096.0)        # a cgroup limit, outside SLURM
    h = foam.memory_hint("foamRun")
    assert "this job had 4 GB" in h and '"mem": "8G"' in h


def test_request_records_versions_and_status_says_what_ran(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    monkeypatch.setenv("CFDT_EXECUTOR", "local")
    bad = {"2d": True, "geometry": {"box": [1.0, 0.1, 0.01]}, "fluid": {"rho": 1000.0, "mu": 1e-3},
           "bcs": [{"kind": "wall", "faces": {"all": True}}], "mesh": {"cells": [20, 10, 1]}}   # invalid: no driver
    r = api.run({"x": bad}, "ref", wait=1, request_name="run_versions")
    meta = json.loads((tmp_path / "run_versions" / "request.json").read_text())
    assert meta["translator"] == __version__ and meta["runtime"] == llmt_runtime.__version__
    assert r["translator"] == __version__ and "translator_now" not in r
    meta["translator"] = "2026.11.08.2"                              # submitted by an older cfdt
    (tmp_path / "run_versions" / "request.json").write_text(json.dumps(meta))
    again = api.status(str(tmp_path / "run_versions"))
    assert again["translator"] == "2026.11.08.2" and again["translator_now"] == __version__
