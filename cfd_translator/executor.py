"""executor.py — where studies run: the local worker pool, or one SLURM job per study.
Version: 2026.10.21.1

Implements llm-translators' docs/EXECUTION.md. Settings (environment):
    CFDT_EXECUTOR          local | slurm (local)
    CFDT_SLURM_MIN_MINUTES studies estimated below this run locally even with slurm (0; no estimate: SLURM)
    CFDT_SLURM_ARGS        extra sbatch arguments, split like a shell does
    CFDT_PROCS             MPI ranks per study (a study's solve.procs overrides it)
    CFDT_TIMEOUT           seconds per study; the job's time limit is ceil(timeout / 60) + 10 minutes
A study's "queue": true (always SLURM) or false (always local) overrides the minimum, not CFDT_EXECUTOR=local.

Every SLURM state is computed from the files and the queue (never from in-process handles), so any server
process, after restarts or from another chat, sees the same thing:
    done           <name>.json exists
    queued         squeue: PENDING (reason, expected_start, ahead)
    running        squeue: RUNNING / COMPLETING (the solver's progress, job_elapsed_min)
    timed_out, out_of_memory, cancelled, failed
                   gone from squeue without a result: sacct, or job.log when sacct knows nothing; these
                   are written as the study's result ({"ok": false, "state", "why"}) so they stay final
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLI = "cfdt"
FINAL = ("timed_out", "out_of_memory", "cancelled", "failed")


# ---- settings ----------------------------------------------------------------------------------
def kind() -> str:
    k = os.environ.get("CFDT_EXECUTOR", "local").strip().lower() or "local"
    return k if k in ("local", "slurm") else "local"


def min_minutes() -> float:
    try:
        return float(os.environ.get("CFDT_SLURM_MIN_MINUTES", "0"))
    except ValueError:
        return 0.0


def timeout_s(study: dict | None = None) -> float:
    """The study's run.timeout_s, else CFDT_TIMEOUT (3600)."""
    t = ((study or {}).get("run") or {}).get("timeout_s")
    return float(t) if t else float(os.environ.get("CFDT_TIMEOUT", "3600"))


def time_limit_min(study: dict | None = None) -> int:
    return int(math.ceil(timeout_s(study) / 60.0)) + 10


def procs_of(study: dict) -> int:
    p = ((study.get("run") or {}).get("procs")) or (study.get("solve") or {}).get("procs")
    return int(p) if p else int(os.environ.get("CFDT_PROCS", "1") or 1)


# ---- per-study run settings (study.run) within the machine's caps ------------------------------------
RUN_KEYS = {"timeout_s", "mem", "procs", "queue"}
_MEM = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)B?$", re.I)


def mem_mb(text) -> float | None:
    """'8G' -> 8192 (SLURM's units: K, M (default), G, T)."""
    m = _MEM.match(str(text).strip())
    if not m:
        return None
    return float(m.group(1)) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2).upper()]


def caps() -> dict:
    """The ceilings set in config/env (CFDT_MAX_TIMEOUT s, CFDT_MAX_MEM, CFDT_MAX_PROCS); unset: none."""
    out = {}
    for k, env in (("timeout_s", "CFDT_MAX_TIMEOUT"), ("mem", "CFDT_MAX_MEM"), ("procs", "CFDT_MAX_PROCS")):
        v = os.environ.get(env, "").strip()
        if v:
            out[k] = v
    return out


def run_errors(st: dict) -> list[str]:
    """Validation of study.run: names, types, and the caps of this machine."""
    r = st.get("run")
    if r is None:
        return []
    if not isinstance(r, dict):
        return ["run: {timeout_s, mem, procs, queue}"]
    errs = []
    bad = set(r) - RUN_KEYS
    if bad:
        errs.append(f"run: unknown keys {sorted(bad)} (timeout_s, mem, procs, queue)")
    cap = caps()
    t = r.get("timeout_s")
    if t is not None:
        if not isinstance(t, (int, float)) or isinstance(t, bool) or t <= 0:
            errs.append("run.timeout_s: seconds (> 0)")
        elif "timeout_s" in cap and t > float(cap["timeout_s"]):
            errs.append(f"run.timeout_s: at most {float(cap['timeout_s']):g} s on this machine (CFDT_MAX_TIMEOUT)")
    m = r.get("mem")
    if m is not None:
        mb = mem_mb(m) if isinstance(m, str) and m.lower() != "auto" else (1 if m == "auto" else None)
        if not mb:
            errs.append("run.mem: SLURM memory per job, e.g. '8G' or '500M', or 'auto' (from the cell estimate)")
        elif "mem" in cap and mem_mb(cap["mem"]) and mb > mem_mb(cap["mem"]):
            errs.append(f"run.mem: at most {cap['mem']} on this machine (CFDT_MAX_MEM)")
    n = r.get("procs")
    if n is not None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            errs.append("run.procs: MPI ranks (integer >= 1)")
        elif "procs" in cap and n > int(cap["procs"]):
            errs.append(f"run.procs: at most {cap['procs']} on this machine (CFDT_MAX_PROCS)")
        sp = (st.get("solve") or {}).get("procs")
        if sp is not None and sp != n:
            errs.append(f"run.procs ({n}) and solve.procs ({sp}) differ: give one")
    q = r.get("queue")
    if q is not None:
        if not isinstance(q, bool):
            errs.append("run.queue: true (always a SLURM job) or false (always local)")
        elif "queue" in st and st["queue"] != q:
            errs.append("run.queue and queue differ: give one")
    return errs


def queue_of(study: dict):
    r = (study.get("run") or {}).get("queue")
    return r if r is not None else study.get("queue")


def mem_estimate_mb(cells: float | None) -> float | None:
    """Memory a study may need, from its cell count: 2 GB plus 4 GB per million cells, an upper bound that
    covers meshing (Gmsh, snappyHexMesh) and a kOmegaSST solve with MPI buffers."""
    return None if not cells else 2048.0 + 4096.0 * cells / 1e6


def _strip_mem(args: list[str]) -> tuple[list[str], float | None]:
    """args without memory requests, and the --mem they held (MB)."""
    out, mem, skip = [], None, None
    for a in args:
        if skip:
            if skip == "--mem":
                mem = mem_mb(a)
            skip = None
            continue
        if a in ("--mem", "--mem-per-cpu"):
            skip = a
            continue
        if a.startswith("--mem="):
            mem = mem_mb(a.split("=", 1)[1])
            continue
        if a.startswith("--mem-per-cpu="):
            continue
        out.append(a)
    return out, mem


def sbatch_args(study: dict, cells: float | None = None) -> tuple[list[str], str | None]:
    """CFDT_SLURM_ARGS with the job's memory: run.mem when given (or 'auto'); otherwise the default --mem,
    raised to the cell estimate's need when that is larger (a 3-6M-cell mesh does not fit 8G), within
    CFDT_MAX_MEM. Returns (args, where the memory came from)."""
    args = shlex.split(os.environ.get("CFDT_SLURM_ARGS", ""))
    mem = (study.get("run") or {}).get("mem")
    rest, default = _strip_mem(args)
    cap = mem_mb(caps()["mem"]) if caps().get("mem") else None
    if mem and str(mem).lower() != "auto":
        return rest + [f"--mem={mem}"], "study"
    need = mem_estimate_mb(cells)
    if need and (str(mem).lower() == "auto" or (default is not None and need > default)):
        mb = min(need, cap) if cap else need
        if default is not None and str(mem).lower() != "auto":
            mb = max(mb, default)
        return rest + [f"--mem={int(math.ceil(mb / 1024))}G"], "estimate"
    return args, "default" if default is not None else None


def cell_estimate(study: dict) -> float | None:
    """An upper estimate of the mesh's cell count before meshing, or None when it cannot be told cheaply.
    Box cells are exact; Gmsh meshes: 2D/swept sections ~1.3 cells per size^2 with quads (2.5 with
    triangles), tetrahedra ~10 per size^3, plus every refine sphere at its size. A goal is built with
    build123d-translator for its volume (a second or so)."""
    import math
    g, m = study.get("geometry") or {}, study.get("mesh") or {}
    if m.get("cells"):
        return float(math.prod(int(c) for c in m["cells"]))
    h = m.get("size")
    if not h:
        return None
    h = float(h)
    dom = study.get("domain") or {}
    vol = thick = None
    if dom.get("kind") == "external" and dom.get("box"):
        b = [float(v) for v in dom["box"]]
        vol, thick = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]), b[5] - b[2]
    elif "box" in g:
        lx, ly, lz = (float(v) for v in g["box"])
        vol, thick = lx * ly * lz, lz
    elif "cylinder" in g:
        r, hh = (float(v) for v in g["cylinder"])
        vol, thick = math.pi * r * r * hh, hh
    elif "annulus" in g:
        ri, ro, hh = (float(v) for v in g["annulus"])
        vol, thick = math.pi * (ro * ro - ri * ri) * hh, hh
    elif "goal" in g:
        try:
            from build123d_translator import translate as T
            solid, _ = T.build(g["goal"])
            bb = solid.bounding_box()
            vol, thick = float(solid.volume), float(bb.max.Z - bb.min.Z)
        except Exception:  # noqa: BLE001
            return None
    if not vol or not thick or thick <= 0:
        return None
    layers = 1 if study.get("2d") else (int(m["extrude"]) if m.get("extrude") else None)
    refine = m.get("refine") or []
    if layers:                                        # a 2D section, swept into layers
        per = 1.3 if m.get("quads", True) else 2.5
        n = vol / thick / h ** 2 * per + sum(math.pi * float(r["r"]) ** 2 / float(r["size"]) ** 2 * per for r in refine)
        return n * layers
    return vol / h ** 3 * 10 + sum(4 / 3 * math.pi * float(r["r"]) ** 3 / float(r["size"]) ** 3 * 10 for r in refine)


def procs_for_job(study: dict, est: float | None = None) -> int:
    """The ranks a SLURM job asks for: the study's (or CFDT_PROCS), but no more than the cell estimate allows
    at CFDT_CELLS_PER_RANK, so small studies do not hold cores they cannot use (the benchmarks on mochi held
    18 CPUs each for 1-3 ranks)."""
    procs = procs_of(study)
    est = cell_estimate(study) if est is None else est
    if est is None:
        return procs
    per = int(os.environ.get("CFDT_CELLS_PER_RANK", "5000") or 5000)
    return max(1, min(procs, int(math.ceil(est / max(per, 1)))))


def estimate_minutes(study: dict, solver: str) -> float | None:
    """The study's own time estimate, or None (cfdt cannot tell an OpenFOAM run's length before meshing;
    the 2D reference solver takes seconds)."""
    return 0.1 if solver == "ref" else None


def use_slurm(study: dict, solver: str) -> bool:
    if kind() != "slurm":
        return False
    q = queue_of(study)
    if q is True:
        return True
    if q is False:
        return False
    est = estimate_minutes(study, solver)
    return est is None or est >= min_minutes()


def status() -> dict:
    """The 'executor' key of tool_status (read by llmt status)."""
    if kind() == "slurm":
        return {"kind": "slurm", "sbatch": shutil.which("sbatch"), "min_minutes": min_minutes(),
                "procs": int(os.environ.get("CFDT_PROCS", "1") or 1), "time_limit_min": time_limit_min(),
                "args": os.environ.get("CFDT_SLURM_ARGS", ""), "timeout_s": timeout_s(), "caps": caps()}
    return {"kind": "local", "workers": int(os.environ.get("CFDT_WORKERS", "2")),
            "procs": int(os.environ.get("CFDT_PROCS", "1") or 1), "timeout_s": timeout_s(), "caps": caps()}


# ---- submitting --------------------------------------------------------------------------------
def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def job_script(request: Path, name: str, procs: int, study: dict | None = None) -> str:
    d = request / name
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={CLI}:{request.name}:{name}",
        f"#SBATCH --chdir={d}",
        f"#SBATCH --output={d / 'job.log'}",
        f"#SBATCH --ntasks={procs}",
        "#SBATCH --ntasks-per-core=1",          # a whole core per rank (OpenMPI 5 maps ranks by core)
        f"#SBATCH --time={time_limit_min(study)}",
        "#SBATCH --export=ALL",
        f"exec {shlex.quote(sys.executable)} -m cfd_translator.cli run-study {shlex.quote(str(request))} {shlex.quote(name)}",
        ""])


def submit(request: Path, name: str, study: dict) -> dict:
    """Writes <request>/<name>/job.sh, submits it, writes job.json. On failure writes the study's result
    {"ok": false, "why": "could not submit: ..."} (never falls back to local)."""
    d = request / name
    d.mkdir(parents=True, exist_ok=True)
    est = cell_estimate(study)
    procs = procs_for_job(study, est)
    script = d / "job.sh"
    script.write_text(job_script(request, name, procs, study))
    script.chmod(0o755)
    sargs, mem_from = sbatch_args(study, est)
    cmd = ["sbatch", "--parsable"] + sargs + [str(script)]
    try:
        r = subprocess.run(cmd, cwd=str(d), capture_output=True, text=True, timeout=60)
        ok, msg = r.returncode == 0 and r.stdout.strip(), (r.stderr or r.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, msg = False, str(exc)
    if not ok:
        res = {"name": name, "ok": False, "state": "failed", "why": f"could not submit: {msg[-500:] or 'sbatch failed'}",
               "execution": {"executor": "slurm", "slurm_job": None}}
        write_atomic(d / f"{name}.json", json.dumps(res, indent=1))
        return res
    job_id = r.stdout.strip().splitlines()[-1].split(";")[0].strip()
    info = {"executor": "slurm", "job_id": job_id, "submitted": time.time(), "procs": procs,
            "time_min": time_limit_min(study), "command": cmd,
            "run": dict(study.get("run") or {}), "cells_estimate": int(est) if est else None, "mem_from": mem_from}
    write_atomic(d / "job.json", json.dumps(info, indent=1))
    return info


def run_study_job(request: str, name: str) -> int:
    """`cfdt run-study REQUEST NAME`: what a SLURM job (or a person) runs. Exit 0 when a result was written."""
    from .runner import run_study

    req = Path(request).expanduser().resolve()
    meta = json.loads((req / "request.json").read_text())
    study = dict(meta["studies"][name])
    n = os.environ.get("SLURM_NTASKS")
    procs = int(n) if n and n.isdigit() else procs_of(study)
    if procs != procs_of(study) or "procs" not in (study.get("solve") or {}):
        study["solve"] = dict(study.get("solve") or {}, procs=procs)
    if not os.environ.get("SLURM_JOB_ID"):
        os.environ["CFDT_EXECUTION"] = "by hand"            # the run record's executor
    t0 = time.perf_counter()
    r = run_study(name, study, req / name, meta.get("solver", "foam"), timeout_s(study))
    r.setdefault("seconds", {})
    if isinstance(r["seconds"], dict):
        r["seconds"]["wall"] = round(time.perf_counter() - t0, 3)
    ex = r.get("execution") or {}
    # 'executor' is the record's former shape, kept for a while for readers of 2026.10.18.2-.5
    r["executor"] = {"kind": ex.get("executor", "slurm" if os.environ.get("SLURM_JOB_ID") else "by hand"),
                     "job_id": ex.get("slurm_job", os.environ.get("SLURM_JOB_ID")), "procs": procs}
    write_atomic(req / name / f"{name}.json", json.dumps(r, indent=1, default=float))
    return 0


# ---- the queue ---------------------------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def _minutes(t: str) -> float | None:
    """SLURM elapsed time [d-]hh:mm:ss or mm:ss -> minutes."""
    t = t.strip()
    if not t or t in ("N/A", "INVALID"):
        return None
    days = 0
    if "-" in t:
        dd, t = t.split("-", 1)
        days = int(dd)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return round(days * 1440 + h * 60 + m + s / 60, 1)


def squeue(ids: list[str]) -> dict:
    """One squeue call for these job ids: {id: {state, reason, start, elapsed_min}}; None when squeue fails."""
    if not ids:
        return {}
    rc, out, err = _run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%r|%S|%M"])
    if rc != 0 and "Invalid job id" not in err:
        return None
    q = {}
    for line in out.splitlines():
        f = (line.split("|") + [""] * 5)[:5]
        q[f[0].strip()] = {"state": f[1].strip(), "reason": f[2].strip(), "start": f[3].strip(),
                           "elapsed_min": _minutes(f[4])}
    return q


def pending_ahead() -> dict:
    """All pending jobs on the machine as (priority, id), to count the jobs ahead of each."""
    rc, out, _ = _run(["squeue", "-h", "-t", "PD", "-o", "%i|%Q"])
    jobs = []
    for line in out.splitlines() if rc == 0 else []:
        i, _, p = line.partition("|")
        try:
            jobs.append((float(p or 0), int(i)))
        except ValueError:
            pass
    return {str(i): sum(1 for pp, ii in jobs if pp > p or (pp == p and ii < i)) for p, i in jobs}


def sacct(ids: list[str]) -> dict | None:
    """{id: state} from accounting, or None when sacct fails or knows nothing."""
    if not ids:
        return {}
    rc, out, _ = _run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "-o", "JobID,State"])
    if rc != 0:
        return None
    st = {}
    for line in out.splitlines():
        i, _, s = line.partition("|")
        if i.strip() and "." not in i:
            st[i.strip()] = s.strip()
    return st or None


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no job.log)"


def final_state(acct: str | None, log: str) -> tuple[str, str]:
    """(state, why) of a job that left the queue without a result."""
    a = (acct or "").upper()
    if a.startswith("TIMEOUT") or (not acct and "DUE TO TIME LIMIT" in log):
        return "timed_out", ("the SLURM time limit ended the job: raise CFDT_TIMEOUT (the job asks for it plus 10 "
                             "minutes), or lower the study's solve.iterations or coarsen its mesh")
    if a.startswith("OUT_OF_MEMORY") or (not acct and ("oom-kill" in log.lower() or "out of memory" in log.lower())):
        return "out_of_memory", "out of memory: add --mem=... to CFDT_SLURM_ARGS, or coarsen the mesh"
    if a.startswith("CANCELLED") or (not acct and "CANCELLED" in log):
        who = a.split("BY", 1)[1].strip() if "BY" in a else None
        return "cancelled", "cancelled" + (f" by user {who}" if who else "")
    if not acct and log == "(no job.log)":     # never started: scancel while queued leaves nothing behind
        return "cancelled", ("left the queue without starting (no job.log): cancelled while queued (scancel), or "
                             "SLURM dropped it; submit it again if it was not cancelled on purpose")
    return "failed", f"the job ended without a result ({acct or 'no accounting'}); the end of job.log:\n{log}"


def slurm_states(request: Path, names: list[str], studies: dict | None = None, write: bool = True) -> dict:
    """{name: {"state", ...}} for the request's studies that have job.json and no result. Finished
    failures become the study's result file (when write)."""
    from .api import _progress

    jobs = {}
    for n in names:
        jf = request / n / "job.json"
        if jf.is_file() and not (request / n / f"{n}.json").is_file():
            try:
                jobs[n] = json.loads(jf.read_text())
            except json.JSONDecodeError:
                pass
    if not jobs:
        return {}
    ids = [str(j["job_id"]) for j in jobs.values()]
    q = squeue(ids)
    gone = [n for n, j in jobs.items() if q is not None and str(j["job_id"]) not in q]
    acct = sacct([str(jobs[n]["job_id"]) for n in gone]) if gone else {}
    ahead = pending_ahead() if q and any(v["state"] == "PENDING" for v in q.values()) else {}
    out = {}
    for n, j in jobs.items():
        jid = str(j["job_id"])
        base = {"job_id": jid, "procs": j.get("procs")}
        if (request / n / f"{n}.json").is_file():               # finished between the calls
            continue
        if q is None:
            out[n] = {"state": "unknown", **base, "why": "squeue failed (is slurmctld running?)"}
            continue
        if jid in q:
            s = q[jid]
            if s["state"] in ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"):
                st = {"state": "queued", **base, "reason": s["reason"], "ahead": ahead.get(jid)}
                if s["start"] and s["start"] not in ("N/A", "Unknown"):
                    st["expected_start"] = s["start"]
            else:
                prog = _progress(request / n, (studies or {}).get(n, {}))
                st = {**prog, "state": "running", **base, "job_elapsed_min": s["elapsed_min"]}
            out[n] = st
            continue
        state, why = final_state((acct or {}).get(jid), _tail(request / n / "job.log"))
        if state == "failed" and (request / n / "cancel").exists():
            state, why = "cancelled", "cancelled with cancel(request) (it left the queue before running)"
        res = {"name": n, "ok": False, "state": state, "why": why, "job_id": jid,
               "execution": {"executor": "slurm", "slurm_job": jid}}
        if write:
            write_atomic(request / n / f"{n}.json", json.dumps(res, indent=1))
        out[n] = res
    return out


# ---- jobs and cancel ---------------------------------------------------------------------------
def queue_summary() -> dict:
    rc, out, _ = _run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T|%C"])
    run = qd = cpus = 0
    for line in out.splitlines() if rc == 0 else []:
        f = (line.split("|") + [""] * 4)[:4]
        if not f[1].startswith(f"{CLI}:"):
            continue
        if f[2] in ("RUNNING", "COMPLETING"):
            run += 1
            cpus += int(f[3]) if f[3].isdigit() else 0
        else:
            qd += 1
    machine = {}
    rc, out, _ = _run(["sinfo", "-h", "-o", "%C"])
    if rc == 0 and out.strip():
        a, i, o, t = (out.strip().splitlines()[0].split("/") + ["0"] * 4)[:4]
        try:
            machine = {"cpus": int(t), "cpus_allocated": int(a)}
        except ValueError:
            pass
    return {"queue": {"running": run, "queued": qd, "cpus_in_use": cpus}, "machine": machine}


def cancel_slurm(request: Path, names: list[str]) -> list[str]:
    ids = []
    for n in names:
        jf = request / n / "job.json"
        if jf.is_file() and not (request / n / f"{n}.json").is_file():
            ids.append(str(json.loads(jf.read_text())["job_id"]))
    if ids:
        _run(["scancel"] + ids)
    return ids
