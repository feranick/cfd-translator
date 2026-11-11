"""cfdt on llmt-runtime: the shared pieces are the runtime's, configured for cfdt (2026.11.09.2, phase 1: workers,
out of memory, final_state, versions; 2026.11.09.4, phase 2: the SLURM layer, with cfdt's sizing as hooks;
2026.11.09.6, phase 3: the MCP server's stale handling and tool wrapper).
Version: 2026.11.09.6
"""
import json
import os
import subprocess
import threading
import time
import types
from pathlib import Path

import pytest

import llmt_runtime
from llmt_runtime import jobs as J
from llmt_runtime import memory as M
from llmt_runtime import server as LS
from llmt_runtime import slurm as S
from llmt_runtime import workers as W
from llmt_runtime.testing import fake_slurm

from cfd_translator import __version__, api, executor, foam, workers
from cfd_translator.runtime import RT

REPO = Path(__file__).resolve().parents[1]


def test_runtime_configured_for_cfdt(monkeypatch):
    # not one_node since 2026.11.11.2: CFDT_SLURM_NODES lets an MPI job span nodes (llmt-runtime 2026.11.11.1)
    assert (RT.prefix, RT.cli, RT.package, RT.kind, RT.one_node) == ("CFDT", "cfdt", "cfd_translator", "mpi", False)
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


# ---- 2026.11.09.4, phase 2: the SLURM layer (settings, the machine, time limits, job.sh and job.json, submission,
# states, the queue's CPUs, cancel's flags and scancel) is llmt-runtime's; cfdt's sizing comes in as hooks

FAKE_KNOBS = ("FAKE_SBATCH_FAIL", "FAKE_SBATCH_PENDING", "FAKE_PENDING", "FAKE_SBATCH_DELAY", "FAKE_SBATCH_SLEEP",
              "FAKE_SBATCH_SLOW_AFTER", "FAKE_MIN_JOB_AGE", "FAKE_PRIORITY", "FAKE_RUN_NODE", "FAKE_START",
              "FAKE_SACCT_FAIL", "FAKE_SQUEUE_FAIL", "FAKE_SCONTROL_NODES", "FAKE_SCONTROL_FAIL", "FAKE_SCONTROL_DENY",
              "FAKE_SINFO_FAIL", "FAKE_SINFO_PARTITIONS", "FAKE_SINFO_CPUS", "FAKE_TIME_LEFT")
COUETTE = {"geometry": {"box": [400, 100, 10]}, "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
           "bcs": [{"kind": "wall", "name": "lid", "faces": {"ny": 1}, "velocity": [0.01, 0, 0]},
                   {"kind": "wall", "name": "bottom", "faces": {"ny": -1}},
                   {"kind": "opening", "name": "left", "faces": {"nx": -1}},
                   {"kind": "opening", "name": "right", "faces": {"nx": 1}}],
           "mesh": {"cells": [20, 10, 1]}}                                          # 200 cells


@pytest.fixture()
def slurm(tmp_path, monkeypatch):
    """llmt-runtime's fake SLURM (one node fake1: 40 CPUs, 2 threads per core, 64000 MB); returns its state folder."""
    monkeypatch.setenv("CFDT_EXECUTOR", "slurm")
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("CFDT_TIMEOUT", "600")
    monkeypatch.setenv("PYTHONPATH", f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    for k in FAKE_KNOBS + ("CFDT_WORKERS", "SLURM_JOB_ID", "SLURM_NTASKS"):
        monkeypatch.delenv(k, raising=False)
    return fake_slurm(tmp_path, monkeypatch)


def _fake_job(state_dir: Path, jid: str, **rec) -> None:
    """A job in the fake's queue (a record as its sbatch writes them; pid 999999: not running)."""
    d = state_dir / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    base = {"name": "cfdt:r:s", "cpus": 1, "t0": time.time(), "log": "x", "pid": 999999, "node": "fake1"}
    (d / f"{jid}.json").write_text(json.dumps({**base, **rec}))


def _request(state_dir: Path, jid: str, log: str | None = None, job: dict | None = None) -> Path:
    """A request with one SLURM study 's' whose job.json names the job `jid`."""
    req = state_dir.parent / "out" / f"run_{jid}"
    (req / "s").mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"names": ["s"], "solver": "ref", "started": time.time(),
                                                  "studies": {"s": COUETTE}}))
    (req / "s" / "job.json").write_text(json.dumps({"executor": "slurm", "job_id": jid, "procs": 1, **(job or {})}))
    if log is not None:
        (req / "s" / "job.log").write_text(log)
    return req


def test_slurm_layer_is_the_runtimes_with_cfdts_hooks():
    h = RT.hooks
    assert h.time_need_min is None and h.resource_lines is None and h.solves is None    # the runtime's defaults
    assert S.time_need_min(RT, {"run": {"timeout_s": 600}}) == 20 == executor.time_limit_min({"run": {"timeout_s": 600}})
    assert h.procs(COUETTE) == executor.procs_for_job(COUETTE) == 1
    big = dict(COUETTE, mesh={"cells": [2000, 2000, 1]})                                  # 2 + 4 x 4 GB
    assert h.mem_auto(big) == "18G" == executor.mem_auto(big) and h.job_extra(COUETTE) == {"cells_estimate": 200}
    assert h.running_progress(Path("/nonexistent"), {}) == {"stage": "starting"}
    assert executor.FINAL == J.FINAL and executor.write_atomic is J.write_atomic
    assert set(executor.status()) == {"kind", "workers", "procs", "timeout_s", "caps", "runtime"}   # local: as before


def test_slurm_time_and_the_executor_status(slurm, monkeypatch):
    """CFDT_SLURM_TIME (new in cfdt, llmt-runtime's meanings): 'timeout' by default, which is what cfdt always asked
    (ceil(timeout / 60) + 10); max, minutes and estimate on request."""
    ex = executor.status()
    assert ex["time"] == ex["slurm_time"] == "timeout" and ex["time_limit_min"] == 20
    assert ex["runtime"] == llmt_runtime.__version__ and ex["node_cores"] == 20
    assert {"args", "nodes", "node", "caps", "mem", "sbatch", "min_minutes", "procs", "timeout_s"} <= set(ex)
    assert ex["caps"] == {"procs": "20", "procs_from": "a node's cores (SLURM, one rank per core)", "mem": "64000M",
                          "mem_from": "a node's memory (SLURM RealMemory)", "mem_mb": 64000}
    assert executor.job_time(COUETTE)[0] == "20"
    monkeypatch.setenv("CFDT_SLURM_TIME", "max")
    monkeypatch.setenv("FAKE_SINFO_PARTITIONS", "main*|2:00:00")
    assert executor.job_time(COUETTE)[0] == "120" and executor.status()["time"] == "max"
    monkeypatch.setenv("CFDT_SLURM_TIME", "45")
    assert executor.job_time(COUETTE)[0] == "45"
    monkeypatch.setenv("CFDT_SLURM_TIME", "estimate")
    t, info = executor.job_time(COUETTE, 2.0)
    assert t == "20" and info["time_from"] == "estimate" and info["timeout_s"] == 600     # max(20, 3 x 2 + 10) min


def test_job_sh_and_job_json(slurm, monkeypatch):
    """job.sh as cfdt wrote it, with --mem inside (llmt-runtime's fix: it was only on sbatch's command line, so
    `sbatch job.sh` lost it); job.json with cfdt's keys and the runtime's; dependencies through after=."""
    monkeypatch.setenv("FAKE_PENDING", "1")
    req = slurm.parent / "out" / "run_j"
    info = executor.submit(req, "a", COUETTE)
    text = (req / "a" / "job.sh").read_text()
    sh = text.splitlines()
    assert [ln for ln in sh if ln.startswith("#SBATCH")] == [
        "#SBATCH --job-name=cfdt:run_j:a", f"#SBATCH --chdir={req / 'a'}", f"#SBATCH --output={req / 'a' / 'job.log'}",
        "#SBATCH --ntasks=1", "#SBATCH --ntasks-per-core=1", "#SBATCH --time=20",
        "#SBATCH --mem=3G", "#SBATCH --export=ALL"]                       # 200 cells: 2 GB + 0.8 MB, in whole GB
    assert sh[-1].endswith(f"-m cfd_translator.cli run-study {req} a") and sh[-1].startswith("exec ")
    assert text == executor.job_script(req, "a", 1, COUETTE)
    j = json.loads((req / "a" / "job.json").read_text())
    old = {"executor", "job_id", "submitted", "procs", "time_min", "command", "run", "cells_estimate", "mem_from"}
    assert old <= set(j) and j == info and j["job_id"] == "1001" and j["cells_estimate"] == 200
    assert j["mem"] == "3G" and j["mem_from"] == "estimate" and j["time_min"] == 20 and j["procs"] == 1
    assert not any(str(a).startswith("--mem") for a in j["command"])
    assert j["translator"] == __version__ and j["runtime"] == llmt_runtime.__version__
    # the next segment waits for this job (after=: --dependency=afterany) and records it
    nxt = executor.submit(req, "a", COUETTE, after="1001", segment=2)
    assert nxt["job_id"] == "1002" and "--dependency=afterany:1001" in nxt["command"] and nxt["after"] == ["1001"]
    assert nxt["segment"] == 2 and nxt["previous_jobs"] == ["1001"]
    # run.mem 'auto': the estimate, whatever a --mem in CFDT_SLURM_ARGS says; job.json keeps the study's run
    monkeypatch.setenv("CFDT_SLURM_ARGS", "--mem=1G --nice=5")
    c = executor.submit(req, "c", dict(COUETTE, run={"mem": "auto"}))
    assert "#SBATCH --mem=3G" in (req / "c" / "job.sh").read_text() and c["command"][2:-1] == ["--nice=5"]
    assert c["run"] == {"mem": "auto"} and c["mem_from"] == "estimate"


def test_an_ended_job_squeue_still_lists_is_final(slurm, monkeypatch):
    """squeue lists an ended job for MinJobAge (300 s) in its final state: cfdt 2026.11.09.3 read every state but
    PENDING as running. It is now final, from job.log (no accounting here), and written as the study's result."""
    monkeypatch.setenv("FAKE_MIN_JOB_AGE", "300")
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    _fake_job(slurm, "1001", ended=time.time(), final="FAILED")
    req = _request(slurm, "1001", log="Traceback (most recent call last):\nValueError: boom")
    assert executor.squeue(["1001"])["1001"]["state"] == "FAILED"
    st = executor.slurm_states(req, ["s"], {"s": COUETTE})["s"]
    assert st["state"] == "failed" and "ValueError: boom" in st["why"]
    assert json.loads((req / "s" / "s.json").read_text())["state"] == "failed"         # final: kept


def test_queue_summary_counts_each_node_once_with_cfdts_extras(slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SINFO_PARTITIONS", "debug*|1:00:00;long|infinite")       # fake1 in two partitions
    _fake_job(slurm, "1001", state="PENDING", cpus=4, name="cfdt:r:x")
    _fake_job(slurm, "1002", state="PENDING", name="feat:r:y")                       # another translator's
    assert executor.queue_summary() == {"queue": {"running": 0, "queued": 1, "cpus_in_use": 0},
                                        "machine": {"cpus": 8, "cpus_allocated": 2, "cpus_idle": 6},
                                        "starts_now": False, "pending_for_node": 2}


def test_eta_start_still_applied_to_queued_jobs(slurm, monkeypatch):
    """A queued job's start from the running jobs' own ETAs (cfdt's), on top of llmt-runtime's states."""
    import time as _t
    later = _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(_t.time() + 3 * 86400))
    monkeypatch.setenv("FAKE_START", later)                          # SLURM's: from the whole time limits ahead
    _fake_job(slurm, "1002", state="PENDING")
    req = _request(slurm, "1002", job={"segment": 2, "previous_jobs": ["1001"]})
    monkeypatch.setattr(executor, "running_etas", lambda: {"1001": {"node": "fake1", "cpus": 1, "eta_min": 30.0,
                                                                     "name": "r/a"}})
    st = executor.slurm_states(req, ["s"], {"s": COUETTE})["s"]
    assert st["state"] == "queued" and st["ahead"] == 0 and st["expected_start_slurm"] == later
    assert st["expected_start"] < later and "own progress" in st["expected_start_note"]
    # SLURM's one-year placeholder (a job ahead without a time limit, carbonio 2026.11.10): no start, the runtime's note
    monkeypatch.setenv("FAKE_START", _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(_t.time() + 365 * 86400)))
    monkeypatch.setattr(executor, "running_etas", lambda: {})
    st = executor.slurm_states(req, ["s"], {"s": COUETTE})["s"]
    assert "expected_start" not in st and "no time limit" in st["expected_start_note"]


def test_bench_runs_a_case_once_as_a_slurm_job(slurm, tmp_path, monkeypatch):
    """carbonio, `llmt bench cfdt couette_2d` with CFDT_EXECUTOR=slurm: PASS within seconds while the case's job stayed
    queued. bench.run took the cases study_status still returned pending (after its 15 s) for cases to run here, and
    ran them locally as well. It now waits for the job and evaluates the job's result."""
    from cfd_translator import bench
    monkeypatch.setenv("FAKE_PENDING", "1")
    local = []
    monkeypatch.setattr(bench, "run_study", lambda *a, **k: local.append(a[0]) or {"ok": False, "why": "ran here"})
    real = api.status
    monkeypatch.setattr(api, "status", lambda request, wait=0.0, *a, **k: real(request, 0.0, *a, **k))  # no 15 s
    monkeypatch.setattr(bench, "SLURM_POLL_S", 0.2)
    out = {}
    t = threading.Thread(target=lambda: out.update(bench.run("ref", ["couette_2d"], tmp_path / "bench")), daemon=True)
    t.start()
    rec = slurm / "jobs" / "1001.json"
    for _ in range(300):
        if rec.is_file():
            break
        time.sleep(0.1)
    time.sleep(2.0)                                   # bench.run looks at the queued job several times meanwhile
    assert rec.is_file() and t.is_alive() and not local, local
    # the job starts, as SLURM would: RUNNING while it lives, then its result
    r = json.loads(rec.read_text())
    script = Path([a for a in r["args"] if not a.startswith("-")][-1])
    with open(r["log"], "a") as log:
        p = subprocess.Popen(["bash", str(script)], cwd=str(script.parent), stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, SLURM_JOB_ID="1001", SLURM_NTASKS="1"))
    tmp = rec.with_name("1001.tmp")
    tmp.write_text(json.dumps(dict(r, state=None, pid=p.pid)))
    tmp.replace(rec)
    p.wait(timeout=120)
    t.join(120)
    assert not t.is_alive() and not local, local
    assert out["couette_2d"]["passed"], out
    assert [f.name for f in (slurm / "jobs").glob("*.json")] == ["1001.json"]       # one job, run once


def test_an_interrupted_wait_says_which_jobs_are_left(slurm, tmp_path, monkeypatch, capsys):
    """Ctrl-C while `cfdt bench` (or `cfdt run`) waits for SLURM jobs: the jobs left in the queue and how to cancel
    them; they are not cancelled."""
    from cfd_translator import bench
    monkeypatch.setenv("FAKE_PENDING", "1")

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(api, "status", interrupted)
    with pytest.raises(KeyboardInterrupt):
        bench.run("ref", ["couette_2d"], tmp_path / "bench")
    err = capsys.readouterr().err
    assert "1001" in err and "scancel 1001" in err and "cfdt cancel" in err and "nothing was cancelled" in err
    assert json.loads((slurm / "jobs" / "1001.json").read_text())["state"] == "PENDING"        # still queued
    with pytest.raises(KeyboardInterrupt):
        api.run({"r": dict(COUETTE, queue=True)}, "ref")
    err = capsys.readouterr().err
    assert "scancel 1002" in err and "cfdt status" in err


# ---- 2026.11.09.6, phase 3: the MCP server's stale handling, tool wrapper, log and signal handlers are llmt-runtime's
# Server (mcp_server.GUARD, with cfdt's busy and shutdown). No test here may end the process: every exit is recorded
# (GUARD.exit_soon patched) or its thread never started (llmt_runtime.server's threading patched).

@pytest.fixture()
def ms():
    """cfd_translator.mcp_server (needs the MCP SDK)."""
    pytest.importorskip("mcp")
    from cfd_translator import mcp_server
    return mcp_server


def _call(ms, name, args):
    import anyio                                          # (with the MCP SDK)

    async def go():
        return await ms.server.call_tool(name, args)
    return anyio.run(go)


def _answer(r) -> dict:
    """A tool's JSON answer from call_tool (SDK 2.x: a CallToolResult; 1.x: content blocks, or (content, structured))."""
    if isinstance(r, tuple):
        r = r[0]
    return json.loads(getattr(r, "content", r)[0].text)


STALE_WHY = f"version 2000.01.01.1 is installed; this server runs {__version__}"


def _serving_stale(ms, monkeypatch, root: Path, busy: bool) -> list:
    """GUARD as a server started by main() whose cfdt on disk is another version (a checkout updated in place), with
    CFDT_EXIT_WHEN_STALE=1; busy: a study in its pools. Returns the list its exits are recorded in (none is run)."""
    pkg = root / "checkout" / "cfd_translator"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "2000.01.01.1"\n')
    monkeypatch.setattr(ms.GUARD, "pkg_dir", pkg)
    for k, v in (("serving", True), ("checked", 0.0), ("why", None), ("shared", False), ("exiting", False),
                 ("inflight", 0), ("background", 0)):
        monkeypatch.setitem(ms._STATE, k, v)
    monkeypatch.setenv("CFDT_EXIT_WHEN_STALE", "1")
    monkeypatch.setattr(ms, "_pool_busy", lambda: busy)
    exits = []
    monkeypatch.setattr(ms.GUARD, "exit_soon", lambda why, delay: exits.append((why, delay)))
    return exits


def test_the_mcp_server_is_llmt_runtimes(ms, monkeypatch):
    g = ms.GUARD
    assert isinstance(g, LS.Server) and g.rt is RT and g.pkg_dir == Path(api.__file__).resolve().parent
    assert ms._STATE is g.state and ms._LOCK is g.lock and ms.STALE_MARK == LS.STALE_MARK == "STALE_SERVER"
    assert ms.stale_reason is LS.stale_reason
    # the install is looked at every 10 s (cfdt's own: 20 s); CFDT_STALE_CHECK_S sets it
    monkeypatch.delenv("CFDT_STALE_CHECK_S", raising=False)
    assert LS.Server(RT, api.__file__).period == 10.0
    monkeypatch.setenv("CFDT_STALE_CHECK_S", "3")
    assert LS.Server(RT, api.__file__).period == 3.0
    # busy: studies in the pools (the module's _pool_busy, looked up when asked)
    monkeypatch.setitem(ms._STATE, "inflight", 0)
    monkeypatch.setitem(ms._STATE, "background", 0)
    monkeypatch.setattr(ms, "_pool_busy", lambda: True)
    assert not g.idle()
    monkeypatch.setattr(ms, "_pool_busy", lambda: False)
    assert g.idle()
    # the exit (and SIGTERM/SIGHUP/SIGINT) stops the study pool, the aux pool and this process's OpenFOAM runs
    stopped = []
    monkeypatch.setattr(api._POOL, "pool", "study pool")
    monkeypatch.setitem(api._AUX, "pool", "aux pool")
    monkeypatch.setattr(W, "stop_pool", lambda pool, *a, **k: stopped.append(pool))
    monkeypatch.setattr(foam, "kill_all", lambda: stopped.append("foam"))
    g.shutdown()
    assert stopped == ["study pool", "aux pool", "foam"]


def test_a_tools_error_reaches_the_caller_as_tool_error_text(ms, monkeypatch):
    monkeypatch.setitem(ms._STATE, "serving", False)

    def boom(topic=None):
        raise ValueError("no guide 'x'")
    monkeypatch.setattr(api, "guide", boom)
    with pytest.raises(Exception) as e:
        _call(ms, "guide", {"topic": "x"})
    assert type(e.value).__name__ == "ToolError" and "ValueError: no guide 'x'" in str(e.value)


def test_a_stale_idle_server_refuses_and_exits(ms, tmp_path, monkeypatch):
    exits = _serving_stale(ms, monkeypatch, tmp_path, busy=False)
    with pytest.raises(Exception) as e:
        _call(ms, "guide", {"topic": ""})
    assert type(e.value).__name__ == "ToolError"
    assert f"STALE_SERVER: {STALE_WHY}. This server is exiting so that the next one runs the installed version: " \
           "call again." in str(e.value)
    assert exits == [(STALE_WHY, 1.0)] and ms._STATE["inflight"] == 0
    # without CFDT_EXIT_WHEN_STALE a checkout's server keeps serving (by default only shared installs exit)
    monkeypatch.delenv("CFDT_EXIT_WHEN_STALE")
    exits.clear()
    r = _answer(_call(ms, "guide", {"topic": ""}))
    assert "topics" in r and r["server_stale"] == f"{STALE_WHY}: this answer comes from the old code; restart the " \
                                                  "MCP server" and not exits


def test_a_stale_busy_server_answers_with_the_note(ms, tmp_path, monkeypatch):
    exits = _serving_stale(ms, monkeypatch, tmp_path, busy=True)          # a study in the pool: it would die
    r = _answer(_call(ms, "guide", {"topic": ""}))
    assert "topics" in r and not exits
    assert r["server_stale"] == f"{STALE_WHY}: this answer comes from the old code; the server restarts when idle"


def test_a_changed_runtime_on_disk_counts_as_stale(ms, tmp_path, monkeypatch):
    """New with llmt-runtime's Server: in a checkout, llmt-runtime's __init__.py with another version (pulled in place)
    makes the server stale too."""
    pkg = tmp_path / "checkout" / "cfd_translator"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{__version__}"\n')
    rt = tmp_path / "rt" / "llmt_runtime"
    rt.mkdir(parents=True)
    (rt / "__init__.py").write_text('__version__ = "2000.01.01.1"\n')
    monkeypatch.setattr(ms.GUARD, "pkg_dir", pkg)
    monkeypatch.setattr(LS, "__file__", str(rt / "server.py"))
    for k, v in (("checked", 0.0), ("why", None), ("shared", False)):
        monkeypatch.setitem(ms._STATE, k, v)
    assert ms._stale() == ("llmt-runtime 2000.01.01.1 is installed; this server runs llmt-runtime "
                           f"{llmt_runtime.__version__}")
    assert not ms._STATE["shared"]


def test_the_stale_exit_says_so_on_stderr(ms, monkeypatch, capsys):
    """The line llmt's router and `llmt servers` look for; the exit's thread is never started here."""
    started = []

    class _NoThread:
        def __init__(self, target=None, **kw):
            self.target = target

        def start(self):
            started.append(self.target)
    monkeypatch.setattr(LS, "threading", types.SimpleNamespace(Thread=_NoThread))
    monkeypatch.setitem(ms._STATE, "exiting", False)
    ms._exit_soon("version X is installed; this server runs Y", 1.0)
    assert capsys.readouterr().err == ("cfdt: STALE_SERVER: exiting to pick up the new code: version X is installed; "
                                       "this server runs Y\n")
    assert ms._STATE["exiting"] and len(started) == 1
    ms._exit_soon("again", 0.0)                                            # once only
    assert len(started) == 1 and not capsys.readouterr().err


def test_a_tool_runs_under_the_48_s_deadline(ms, monkeypatch):
    """cfdt's tools had no deadline: SLURM commands and submissions now keep to 48 s (slurm.deadline)."""
    monkeypatch.setitem(ms._STATE, "serving", False)
    monkeypatch.setattr(api, "guide", lambda topic=None: {"topics": [], "left": S.remaining()})
    assert S.remaining() is None                                           # outside a call: no limit
    r = _answer(_call(ms, "guide", {"topic": ""}))
    assert 40 < r["left"] <= LS.BUDGET_S == 48.0


def test_waits_keep_to_the_calls_deadline(ms, tmp_path, monkeypatch):
    """study_status, run_studies, bench_status and run_benchmarks wait in api.status; get_case, render_images and
    reprocess in api.in_worker. Both end with 3 s of the call's deadline left (llmt_runtime.requests.until), whatever
    wait_s says (up to 50 s, more than the 48 s deadline)."""
    req = tmp_path / "run_deadline"
    (req / "s").mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"names": ["s"], "solver": "ref", "started": time.time(),
                                                  "studies": {"s": COUETTE}}))

    class _Running:                                       # a study still running in this server's pool
        def ready(self):
            return False

        def wait(self, t):
            waits.append(t)
    waits = []
    monkeypatch.setitem(api._POOL.jobs, str(req), {"s": _Running()})
    monkeypatch.setattr(api, "_progress", lambda d, study: {})

    def study_status_like(request: str) -> dict:
        return api.status(request, 50.0, "summary")
    tool = LS.Server(RT, api.__file__).tool(ms.ToolError, budget_s=6)(study_status_like)

    async def go():
        return await tool(request=str(req))
    import anyio                                          # (with the MCP SDK)
    t0 = time.time()
    r = anyio.run(go)
    assert time.time() - t0 < 5.5 and r["pending"] == ["s"] and r["progress"]["s"]["state"] == "queued"
    # in_worker: the wait is cut to the deadline; outside a call it is wait_s (at most 50)
    monkeypatch.setattr(api, "_aux_pool", lambda: types.SimpleNamespace(apply_async=lambda fn, args: _Running()))
    monkeypatch.setitem(api._AUX, "pending", {})
    with S.deadline(10):
        r = api.in_worker("meshing", "case", {"study": {}, "name": "d"}, 50)
    assert r["state"] == "running" and 6 < waits[0] <= 7
    api.in_worker("meshing", "case", {"study": {}, "name": "d"}, 30)
    assert 29 < waits[1] <= 30
