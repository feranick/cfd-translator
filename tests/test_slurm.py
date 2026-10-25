"""SLURM execution (llm-translators docs/EXECUTION.md) against a fake SLURM (tests/fake_slurm.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cfd_translator import api, executor

REPO = Path(__file__).resolve().parents[1]
COUETTE = {"geometry": {"box": [400, 100, 10]}, "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
           "bcs": [{"kind": "wall", "name": "lid", "faces": {"ny": 1}, "velocity": [0.01, 0, 0]},
                   {"kind": "wall", "name": "bottom", "faces": {"ny": -1}},
                   {"kind": "opening", "name": "left", "faces": {"nx": -1}},
                   {"kind": "opening", "name": "right", "faces": {"nx": 1}}],
           "mesh": {"cells": [20, 10, 1]}}


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    bindir, state = tmp_path / "bin", tmp_path / "slurm"
    bindir.mkdir()
    state.mkdir()
    for cmd in ("sbatch", "squeue", "scancel", "sacct", "sinfo", "scontrol"):
        w = bindir / cmd
        w.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{REPO / "tests" / "fake_slurm.py"}" {cmd} "$@"\n')
        w.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(state))
    monkeypatch.setenv("CFDT_EXECUTOR", "slurm")
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("CFDT_TIMEOUT", "600")
    monkeypatch.setenv("PYTHONPATH", f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    for k in ("FAKE_PENDING", "FAKE_SACCT_FAIL", "FAKE_SBATCH_FAIL", "FAKE_SBATCH_DELAY", "CFDT_SLURM_MIN_MINUTES",
              "CFDT_PROCS", "CFDT_SLURM_ARGS", "CFDT_MPIRUN", "CFDT_WORKERS", "SLURM_JOB_ID", "SLURM_NTASKS"):
        monkeypatch.delenv(k, raising=False)
    return state


def _job(state: Path, i) -> dict:
    return json.loads((state / f"{i}.json").read_text())


def _set(state: Path, i, **kw):
    r = _job(state, i)
    r.update(kw)
    (state / f"{i}.json").write_text(json.dumps(r))


def _cli(*args, timeout=60):
    r = subprocess.run([sys.executable, "-m", "cfd_translator.cli", *args], capture_output=True, text=True,
                       timeout=timeout, env=os.environ.copy(), cwd=str(REPO))
    return r


def test_submit_then_collected_by_a_new_process(slurm):
    st = dict(COUETTE, queue=True)                     # the reference solver would otherwise run locally
    r = api.run({"couette": st}, "ref", wait=0)
    req = Path(r["request"])
    assert r["pending"] == ["couette"] and r["progress"]["couette"]["state"] in ("queued", "running")
    assert "SLURM jobs" in r["next"] and "1001" in r["next"]
    job = json.loads((req / "couette" / "job.json").read_text())
    assert job["executor"] == "slurm" and job["job_id"] == "1001" and job["time_min"] == 20 and job["procs"] == 1
    sh = (req / "couette" / "job.sh").read_text()
    assert f"#SBATCH --job-name=cfdt:{req.name}:couette" in sh and "#SBATCH --ntasks=1" in sh
    assert "#SBATCH --export=ALL" in sh and f"run-study {req} couette" in sh
    # another process (a new MCP server) collects it
    for _ in range(60):
        out = json.loads(_cli("status", str(req)).stdout)
        if out["done"]:
            break
        time.sleep(1)
    s = out["studies"]["couette"]
    assert out["done"] and s["ok"] and s["reliable"], s
    assert s["executor"] == {"kind": "slurm", "job_id": "1001", "procs": 1}           # the former shape, kept
    ex = s["execution"]
    assert ex["executor"] == "slurm" and ex["slurm_job"] == "1001" and ex["procs_requested"] == 1
    assert ex["procs_used"] == 1 and ex["mpi"] is None                               # the reference solver is serial
    assert "1001" in (req / "couette" / "job.log").read_text() or (req / "couette" / "job.log").exists()


def test_queued_then_cancelled(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_PENDING", "1")
    r = api.run({"a": dict(COUETTE, queue=True), "b": dict(COUETTE, queue=True)}, "ref", wait=0)
    req = r["request"]
    pa, pb = r["progress"]["a"], r["progress"]["b"]
    assert pa["state"] == pb["state"] == "queued" and pa["reason"] == "Resources"
    assert pa["ahead"] == 0 and pb["ahead"] == 1                        # same priority: the older id first
    j = api.jobs()
    assert j["queue"]["queued"] == 2 and j["machine"] == {"cpus": 8, "cpus_allocated": 2}
    assert j["requests"][0]["studies"]["a"]["state"] == "queued"
    c = api.cancel(req, ["a"])
    assert c["cancelled_jobs"] == ["1001"] and c["states"]["a"] == "cancelled"
    s = api.status(req)
    assert s["studies"]["a"]["state"] == "cancelled" and s["pending"] == ["b"]
    assert api.jobs(active=True)["requests"][0]["studies"]["a"]["state"] == "cancelled"


def test_running_then_cancelled_kills_the_job(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_DELAY", "60")       # the job sits in 'sleep' before running the study
    r = api.run({"c": dict(COUETTE, queue=True)}, "ref", wait=0)
    assert r["progress"]["c"]["state"] == "running" and r["progress"]["c"]["job_elapsed_min"] is not None
    pid = _job(slurm, 1001)["pid"]
    t0 = time.time()
    c = api.cancel(r["request"])
    assert c["states"]["c"] == "cancelled" and time.time() - t0 < 15
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split(")")[1][:3]
    assert "CANCELLED" in (Path(r["request"]) / "c" / "job.log").read_text()


@pytest.mark.parametrize("acct, state, words", [("TIMEOUT", "timed_out", "CFDT_TIMEOUT"),
                                                ("OUT_OF_MEMORY", "out_of_memory", "--mem"),
                                                ("CANCELLED by 1000", "cancelled", "1000"),
                                                ("FAILED", "failed", "job.log"),
                                                ("NODE_FAIL", "failed", "NODE_FAIL")])
def test_final_states_from_sacct(slurm, monkeypatch, acct, state, words):
    monkeypatch.setenv("FAKE_PENDING", "1")
    r = api.run({"s": dict(COUETTE, queue=True)}, "ref", wait=0)
    _set(slurm, 1001, pending=False, final=acct)        # it left the queue without a result
    out = api.status(r["request"])
    s = out["studies"]["s"]
    assert out["done"] and s["ok"] is False and s["state"] == state and words in s["why"], s
    assert s["execution"] == {"executor": "slurm", "slurm_job": "1001"}
    assert json.loads((Path(r["request"]) / "s" / "s.json").read_text())["state"] == state   # final: kept


@pytest.mark.parametrize("log, state", [("slurmstepd: error: *** JOB 1001 ON n CANCELLED AT 2026-10-18T12:00:00 DUE TO TIME LIMIT ***", "timed_out"),
                                        ("slurmstepd: error: *** JOB 1001 ON n CANCELLED AT 2026-10-18T12:00:00 ***", "cancelled"),
                                        ("Traceback (most recent call last):\nValueError: boom", "failed")])
def test_final_states_from_job_log_without_accounting(slurm, monkeypatch, log, state):
    monkeypatch.setenv("FAKE_PENDING", "1")
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    r = api.run({"s": dict(COUETTE, queue=True)}, "ref", wait=0)
    (Path(r["request"]) / "s" / "job.log").write_text(log + "\n")
    _set(slurm, 1001, pending=False, final="GONE")      # not in squeue; sacct fails
    s = api.status(r["request"])["studies"]["s"]
    assert s["state"] == state, s
    if state == "failed":
        assert "boom" in s["why"]


def test_sbatch_failure_is_a_result_not_a_local_run(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_FAIL", "Unable to contact slurm controller")
    r = api.run({"s": dict(COUETTE, queue=True)}, "ref", wait=0)
    s = r["studies"]["s"]
    assert r["done"] and s["ok"] is False and s["why"].startswith("could not submit:") and "controller" in s["why"]
    assert not (Path(r["request"]) / "s" / "case").exists()


def test_routing_queue_key_minimum_and_status(slurm, monkeypatch):
    assert api.tool_status()["executor"]["kind"] == "slurm"
    ex = executor.status()
    assert ex["sbatch"].endswith("sbatch") and ex["time_limit_min"] == 20 and ex["min_minutes"] == 0
    assert executor.use_slurm({}, "foam") is True                       # no estimate: SLURM
    assert executor.use_slurm({}, "ref") is True                        # 0.1 min >= 0
    monkeypatch.setenv("CFDT_SLURM_MIN_MINUTES", "1")
    assert executor.use_slurm({}, "ref") is False and executor.use_slurm({"queue": True}, "ref") is True
    assert executor.use_slurm({"queue": False}, "foam") is False
    r = api.run({"loc": dict(COUETTE, queue=False)}, "ref", wait=60)     # runs in this process's pool
    assert r["done"] and r["studies"]["loc"]["ok"] and not (Path(r["request"]) / "loc" / "job.json").exists()
    assert r["studies"]["loc"]["execution"]["executor"] == "local" and r["studies"]["loc"]["execution"]["slurm_job"] is None
    monkeypatch.setenv("CFDT_EXECUTOR", "local")
    assert executor.use_slurm({"queue": True}, "foam") is False and api.tool_status()["executor"]["kind"] == "local"


def test_run_study_by_hand_and_procs(slurm, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_PENDING", "1")
    r = api.run({"h": dict(COUETTE, queue=True, solve={"procs": 3})}, "ref", wait=0)
    req = Path(r["request"])
    assert "#SBATCH --ntasks=1" in (req / "h" / "job.sh").read_text()    # 200 cells: one rank's worth
    p = _cli("run-study", str(req), "h")
    assert p.returncode == 0, p.stderr
    res = json.loads((req / "h" / "h.json").read_text())
    assert res["ok"] and res["executor"]["kind"] == "by hand" and res["executor"]["procs"] == 3
    assert res["execution"]["executor"] == "by hand" and res["execution"]["procs_requested"] == 3
    assert res["execution"]["procs_used"] == 1 and res["execution"]["slurm_job"] is None
    assert res["execution"]["limits"]["timeout_s"] == 600 and res["execution"]["limits"]["from_study"] == []
    assert not list((req / "h").glob(".h.json.*.tmp"))                  # atomic write left nothing behind


def test_local_cancel_flag_kills_the_process_group(tmp_path):
    from cfd_translator.foam import Cancelled, _bash
    flag = tmp_path / "cancel"
    threading.Timer(1.0, flag.write_text, args=("x",)).start()
    t0 = time.time()
    # a sleep length no other process on the machine uses: other test suites (feat, lmpt) run 'sleep 30' too,
    # and a parallel run of theirs made pgrep find one (2026.10.22, a new installation)
    nap = f"{30 + (os.getpid() % 997) / 1000:.3f}"
    with pytest.raises(Cancelled):
        _bash({}, f"sleep {nap} & sleep {nap}; wait", tmp_path, timeout=60, cancel_file=flag)
    assert time.time() - t0 < 10
    for _ in range(20):                                                # killed processes may take a moment to go
        out = subprocess.run(["pgrep", "-f", f"sleep {nap}"], capture_output=True, text=True).stdout.split()
        if not out:
            break
        time.sleep(0.1)
    assert not out                                                     # the whole group is gone


def test_bench_runs_as_slurm_jobs(slurm, tmp_path):
    from cfd_translator import bench
    r = bench.run("ref", ["couette_2d"], tmp_path / "bench")
    assert r["couette_2d"]["passed"], r
    jj = list((tmp_path / "out").glob("run_*/couette_2d/job.json"))
    assert len(jj) == 1 and json.loads(jj[0].read_text())["job_id"] == "1001"


def test_mpirun_maps_by_hwthread_when_the_job_has_fewer_cores(monkeypatch):
    from cfd_translator import foam
    monkeypatch.delenv("CFDT_MPIRUN", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert foam.mpirun_command(64) == "mpirun"                         # outside SLURM: as before
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setattr(foam, "physical_cores", lambda: 5)
    assert foam.mpirun_command(9) == "mpirun --map-by hwthread" and foam.mpirun_command(4) == "mpirun"
    monkeypatch.setenv("CFDT_MPIRUN", "mpirun.openmpi")
    assert foam.mpirun_command(9) == "mpirun.openmpi"


def test_job_asks_a_core_per_rank(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_PENDING", "1")
    monkeypatch.setenv("CFDT_PROCS", "9")
    monkeypatch.setenv("CFDT_CELLS_PER_RANK", "10")                  # 200 cells: enough for 9 ranks here
    r = api.run({"m": dict(COUETTE, queue=True)}, "ref", wait=0)
    sh = (Path(r["request"]) / "m" / "job.sh").read_text()
    assert "#SBATCH --ntasks=9" in sh and "#SBATCH --ntasks-per-core=1" in sh


def test_run_record_for_an_mpi_run(monkeypatch):
    from cfd_translator.runner import execution_record
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    r = execution_record(9, 4, "mpirun --map-by hwthread -np 4", "hwthread")
    assert r["executor"] == "slurm" and r["slurm_job"] == "77" and (r["procs_requested"], r["procs_used"]) == (9, 4)
    assert r["mpi"]["launch"] == "mpirun --map-by hwthread -np 4" and r["mpi"]["binding"] == "hwthread"
    assert r["mpi"]["cores"] >= 1 and r["mpi"]["hwthreads"] >= r["mpi"]["cores"]
    monkeypatch.delenv("SLURM_JOB_ID")
    assert execution_record(1, 1)["executor"] == "local" and execution_record(1, 1)["mpi"] is None


def test_a_job_that_never_started_counts_as_cancelled():
    from cfd_translator.executor import final_state
    st, why = final_state(None, "(no job.log)")
    assert st == "cancelled" and "without starting" in why
    assert final_state(None, "Traceback ...")[0] == "failed"


def test_benchmarks_from_the_mcp_tool_return_at_once_and_are_collected(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_PENDING", "1")
    out = api.bench(["couette_2d", "chain_plate", "nope"], "ref", wait_s=0)
    assert out["done"] is False and out["pending"] == ["couette_2d"] and "bench_status" in out["next"]
    assert "chain_plate" in out["skipped"]
    meta = json.loads((Path(out["bench"]) / "bench_request.json").read_text())
    assert meta["unknown"] == ["nope"] and meta["cases"] == ["couette_2d"]
    monkeypatch.delenv("FAKE_PENDING")
    for i in [p.stem for p in slurm.glob("1*.json")]:              # the queued jobs start now (as SLURM would)
        r = _job(slurm, i)
        _set(slurm, i, pending=False, final=None)
        subprocess.Popen(["bash", r["script"]], cwd=str(Path(r["script"]).parent),
                         env=dict(os.environ, SLURM_JOB_ID=str(i), SLURM_NTASKS="1"),
                         stdout=open(r["output"], "a"), stderr=subprocess.STDOUT).wait(timeout=120)
    res = api.bench_status(out["bench"], 1)
    assert res["done"] and res["passed"] == ["couette_2d"], res
    assert (Path(out["bench"]) / "bench.json").exists()


def test_jobs_ask_only_for_the_ranks_their_mesh_can_use(monkeypatch):
    from cfd_translator import bench
    monkeypatch.setenv("CFDT_PROCS", "9")
    monkeypatch.delenv("CFDT_CELLS_PER_RANK", raising=False)
    C = bench._cases()
    assert executor.cell_estimate(C["porous_channel_1d"]["study"]) == 1600
    assert executor.procs_for_job(C["porous_channel_1d"]["study"]) == 1
    est = executor.cell_estimate(C["pipe_poiseuille"]["study"])          # swept disk: an upper estimate
    assert est and executor.procs_for_job(C["pipe_poiseuille"]["study"]) == min(9, int(-(-est // 5000)))
    big = dict(C["porous_channel_1d"]["study"], mesh={"cells": [400, 400, 1]})
    assert executor.procs_for_job(big) == 9                              # 160k cells: the study's 9
    assert executor.procs_for_job(dict(big, solve={"procs": 4})) == 4
    assert executor.cell_estimate({"geometry": {"step": "x.step"}, "mesh": {"size": 1}}) is None


def test_study_run_overrides_within_the_machine_caps(slurm, monkeypatch):
    from cfd_translator.format import validate_study
    monkeypatch.setenv("FAKE_PENDING", "1")
    monkeypatch.setenv("CFDT_SLURM_ARGS", "--mem=16G --nice=10")
    monkeypatch.setenv("CFDT_MAX_TIMEOUT", "86400")
    monkeypatch.setenv("CFDT_MAX_MEM", "32G")
    monkeypatch.setenv("CFDT_MAX_PROCS", "19")
    monkeypatch.setenv("CFDT_CELLS_PER_RANK", "10")
    st = dict(COUETTE, run={"timeout_s": 30000, "mem": "12G", "procs": 4, "queue": True})
    assert validate_study(st)["ok"], validate_study(st)
    e = " ".join(validate_study(dict(COUETTE, run={"timeout_s": 1e6, "mem": "64G", "procs": 40, "x": 1}))["errors"])
    for w in ("CFDT_MAX_TIMEOUT", "CFDT_MAX_MEM", "CFDT_MAX_PROCS", "unknown keys"):
        assert w in e, w
    assert not validate_study(dict(COUETTE, run={"procs": 4}, solve={"procs": 2}))["ok"]
    assert not validate_study(dict(COUETTE, run={"mem": "lots"}))["ok"]
    r = api.run({"o": st, "d": dict(COUETTE, queue=True)}, "ref", wait=0)
    req = Path(r["request"])
    sh = (req / "o" / "job.sh").read_text()
    assert "#SBATCH --time=510" in sh and "#SBATCH --ntasks=4" in sh             # 30000 s + 10 min; run.procs
    job = json.loads((req / "o" / "job.json").read_text())
    assert job["command"][2:-1] == ["--nice=10", "--mem=12G"] and job["run"]["mem"] == "12G"
    assert job["time_min"] == 510
    d = json.loads((req / "d" / "job.json").read_text())                         # defaults for the other study
    assert d["command"][2:-1] == ["--mem=16G", "--nice=10"] and d["time_min"] == 20          # CFDT_TIMEOUT 600 s + 10 min
    assert executor.status()["caps"] == {"timeout_s": "86400", "mem": "32G", "procs": "19"}
    assert executor.mem_mb("8G") == 8192 and executor.mem_mb("500M") == 500 and executor.mem_mb("x") is None


def test_job_memory_rises_with_the_cell_estimate(monkeypatch):
    monkeypatch.setenv("CFDT_SLURM_ARGS", "--mem=8G --nice=5")
    monkeypatch.delenv("CFDT_MAX_MEM", raising=False)
    big = dict(COUETTE, mesh={"cells": [2000, 2000, 1]})                     # 4M cells: 2 + 4 x 4 = 18 GB
    args, src = executor.sbatch_args(big, executor.cell_estimate(big))
    assert args == ["--nice=5", "--mem=18G"] and src == "estimate"
    monkeypatch.setenv("CFDT_MAX_MEM", "16G")
    assert executor.sbatch_args(big, 4e6)[0][-1] == "--mem=16G"               # within the machine's cap
    small, src = executor.sbatch_args(COUETTE, 200)
    assert small == ["--mem=8G", "--nice=5"] and src == "default"            # the default when it suffices
    auto = dict(COUETTE, run={"mem": "auto"})
    assert executor.sbatch_args(auto, 200)[0][-1] == "--mem=3G" and executor.sbatch_args(auto, 200)[1] == "estimate"


def test_validation_returns_the_job_estimate(monkeypatch):
    """docs/EXECUTION.md 'Estimates in validation': {minutes, procs, mem_mb} from the job sizing."""
    from cfd_translator import bench
    monkeypatch.setenv("CFDT_SOLVER", "foam")
    monkeypatch.setenv("CFDT_PROCS", "19")
    C = bench._cases()
    pipe = C["pipe_poiseuille"]["study"]                 # 2D section swept: ~41k cells
    e = api.validate({"p": pipe})["studies"]["p"]["estimate"]
    assert set(e) >= {"minutes", "procs", "mem_mb"} and e["procs"] == 9          # 41k cells / 5000 per rank
    assert 0.5 < e["minutes"] < 60 and e["mem_mb"] >= 2048
    e2 = api.validate({"p": dict(pipe, run={"mem": "12G", "procs": 4})})["studies"]["p"]["estimate"]
    assert e2["mem_mb"] == 12288 and e2["procs"] == 4 and e2["minutes"] > e["minutes"]
    # transient: steps from the Courant limit on the finest cells
    cyl = C["cylinder_2d2"]["study"]
    assert executor.transient_steps(cyl) == pytest.approx(8.0 / (0.3 * 0.5 * 2.5e-3 / 1.5), rel=1e-6)
    # the reference solver's 2D boxes take seconds; a study it cannot run is estimated as an OpenFOAM run
    monkeypatch.setenv("CFDT_SOLVER", "ref")
    assert api.validate({"c": COUETTE})["studies"]["c"]["estimate"]["minutes"] == 0.1
    assert api.validate({"p": pipe})["studies"]["p"]["estimate"]["minutes"] > 0.1


def test_ranks_are_capped_at_a_nodes_cores_and_jobs_stay_on_this_node(slurm, tmp_path, monkeypatch):
    """mochi: 40 CPUs, 2 threads per core: a 36-rank job was 'Requested node configuration is not available'.
    The sparks (one cluster, two nodes, no shared home): jobs went to the other node and vanished without a log.
    CFDT_SLURM_NODE (llm-translators docs/EXECUTION.md): local (default) | any | a node list."""
    import socket
    host = socket.gethostname().split(".")[0]
    monkeypatch.setenv("FAKE_NODES", f"spark-5da5 {host} 40 2;spark-0808 otherhost 64 1")
    executor._NODES.clear()
    # default local: this machine's node only
    assert executor.local_node() == "spark-5da5" and executor.node_list() == "spark-5da5"
    assert executor.node_cores() == 20 and executor.caps()["procs"] == "20"
    errs = executor.run_errors(dict(COUETTE, run={"procs": 36}))
    assert errs and "at most 20" in errs[0] and "cores" in errs[0]
    monkeypatch.setenv("CFDT_PROCS", "36")
    assert executor.procs_for_job(COUETTE, 1e6) == 20
    js = executor.job_script(tmp_path, "s", 20, COUETTE)
    assert "#SBATCH --nodelist=spark-5da5" in js and "#SBATCH --nodes=1" in js
    node = executor.status()["node"]
    assert node == {"setting": "local", "nodelist": "spark-5da5", "cluster_nodes": 2}
    # any: SLURM chooses; a node list: those; a --nodelist in CFDT_SLURM_ARGS wins
    monkeypatch.setenv("CFDT_SLURM_NODE", "any")
    assert executor.node_list() is None and executor.node_cores() == 64
    monkeypatch.setenv("CFDT_SLURM_NODE", "spark-0808")
    assert executor.node_list() == "spark-0808"
    monkeypatch.setenv("CFDT_SLURM_ARGS", "--mem=16G -w spark-5da5")
    assert executor.node_list() is None and "--nodelist" not in executor.job_script(tmp_path, "s", 4, COUETTE)
    # found by address when the host name is not the node's
    monkeypatch.delenv("CFDT_SLURM_ARGS")
    monkeypatch.delenv("CFDT_SLURM_NODE")
    monkeypatch.setenv("FAKE_NODES", "n1 elsewhere 8 1 127.0.0.1;n2 far 8 1 10.9.9.9")
    executor._NODES.clear()
    monkeypatch.setattr(executor, "_my_addresses", lambda: {"127.0.0.1"})
    assert executor.local_node() == "n1"
    # a one-node cluster: no --nodelist
    monkeypatch.setenv("FAKE_NODES", f"mochi {host} 40 2")
    executor._NODES.clear()
    assert executor.node_list() is None and executor.node_cores() == 20


def test_a_job_lost_on_another_node_says_so(slurm, tmp_path, monkeypatch):
    import socket
    monkeypatch.setenv("FAKE_PENDING", "1")
    monkeypatch.setenv("CFDT_SLURM_NODE", "any")
    monkeypatch.setenv("FAKE_NODES", f"here {socket.gethostname().split('.')[0]} 8 1;far farhost 8 1")
    executor._NODES.clear()
    req = tmp_path / "req"
    (req / "s").mkdir(parents=True)
    info = executor.submit(req, "s", COUETTE)
    _set(slurm, info["job_id"], final="FAILED", node="far", pending=False)
    st = executor.slurm_states(req, ["s"], {"s": COUETTE})
    assert st["s"]["state"] == "failed" and "node far" in st["s"]["why"] and "CFDT_SLURM_NODE=local" in st["s"]["why"]
