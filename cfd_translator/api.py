"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.11.12.4

    study_format()                   the study format, for the caller to read
    validate(studies)                static checks, no meshing
    run(studies, solver, wait)       meshes, solves and checks each study in a request folder; with wait,
                                     returns after that many seconds with the rest pending
    start(studies, solver)           the same in the background: returns the request folder
    status(request, wait)            finished studies with results, the others pending (with state)
    jobs(limit, active)              recent requests with SLURM jobs and their states; the queue (llmt-runtime's
                                     job states and queue, with cfdt's progress, ETAs and per-node counts)
    cancel(request, names)           scancel / cancel flag for the request's studies (llmt-runtime's cancel_request;
                                     the local pool's side stays here)
    case(study, name)                mesh and OpenFOAM case of one study, without solving (file texts)
    render_images(folder, images)    new pictures of a finished study from its saved mesh and fields
    bench(names, solver)             the benchmark studies with known answers
    read_file(path)                  a written file as base64 (only under the output folder)
    tool_status()                    versions, OpenFOAM and Gmsh found

Studies run in worker processes with a time limit each. Environment: CFDT_OUT (default ~/cfdt_runs),
CFDT_WORKERS (2), CFDT_TIMEOUT (seconds per study, 3600), CFDT_SOLVER (auto | foam | ref; auto =
OpenFOAM when found, else the reference solver), CFDT_FOAM (OpenFOAM bashrc), CFDT_PROCS (MPI ranks).
"""
from __future__ import annotations

import base64
import json
import re
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

from . import __version__
from .format import study_format as _format
from .format import validate as _validate
from .format import validate_study


def out_root() -> Path:
    return Path(os.environ.get("CFDT_OUT", "~/cfdt_runs")).expanduser()


def _solver(solver: str | None) -> str:
    s = (solver or os.environ.get("CFDT_SOLVER", "auto")).lower()
    if s == "auto":
        from .foam import find_foam
        f = find_foam()
        return "foam" if f and "error" not in f else "ref"
    if s not in ("foam", "ref"):
        raise ValueError("solver: auto, foam or ref")
    return s


def tool_status() -> dict:
    from .foam import find_foam

    from . import executor
    out = {"translator": __version__, "openfoam": find_foam(), "default_solver": _solver(None),
           "procs": int(os.environ.get("CFDT_PROCS", "1")), "executor": executor.status()}
    for mod, key in (("gmsh", "gmsh"), ("build123d_translator", "build123d_translator"), ("fea_translator", "fea_translator")):
        try:
            m = __import__(mod)
            out[key] = getattr(m, "__version__", "yes")
        except ImportError:
            out[key] = None
    from .render_vtk import status as render_status
    out["render"] = render_status()
    return out


def study_format() -> dict:
    return {**_format(), "translator": __version__, "status": tool_status()}


def prepare_studies(studies: dict) -> tuple[dict, dict, dict]:
    """Variants spelled out and geometry references resolved: (studies, groups, errors per study)."""
    from .variants import expand, resolve
    if not isinstance(studies, dict):
        return {}, {}, {}
    ex, groups, errs = expand(studies)
    out = {}
    for n, st in ex.items():
        if isinstance(st, dict):
            st, e = resolve(st)
            if e:
                errs.setdefault(n, []).append(e)
        out[n] = st
    return out, groups, errs


def validate(studies: dict) -> dict:
    ex, groups, errs = prepare_studies(studies)
    v = _validate(ex) if ex else {"ok": not errs, "studies": {}, "errors": [] if errs else
                                   ["studies: a non-empty dict {name: study}"]}
    from .transient import continue_problems, init_problems
    for n, st in ex.items():                 # a run to continue or start from must be on this machine
        if isinstance(st, dict) and n not in errs:
            cp = continue_problems(st) + init_problems(st, list(ex), n)
            if cp:
                errs[n] = cp
    for n, e in errs.items():
        rec = v["studies"].setdefault(n, {"ok": False, "errors": [], "warnings": []})
        rec["ok"] = False
        rec["errors"] = e + rec.get("errors", [])
        v["ok"] = False
    if groups:
        v["variants"] = {b: g["names"] for b, g in groups.items()}
    # what the run will take, for llm-translators' router (docs/EXECUTION.md, 'Estimates in validation')
    from .executor import estimate
    try:
        solver = _solver(None)
    except ValueError:
        solver = "foam"
    for n, rec in (v.get("studies") or {}).items():
        if rec.get("ok") and isinstance(ex.get(n), dict):
            try:
                e = estimate(ex[n], solver)
            except Exception:  # noqa: BLE001 (an estimate must never fail a validation)
                e = None
            if e:
                rec["estimate"] = e
                w = _mem_warning(ex[n], e)
                if w:
                    rec.setdefault("warnings", []).append(w)
    return v


def _mem_warning(study: dict, est: dict) -> str | None:
    """A --mem in CFDT_SLURM_ARGS wins over the estimate (docs/EXECUTION.md): say so when the mesh needs more."""
    from . import executor
    if executor.kind() != "slurm" or not est.get("cells"):
        return None
    mb, src = executor.mem_plan(study, est["cells"])
    need = executor.mem_estimate_mb(est["cells"])
    if src == "CFDT_SLURM_ARGS" and mb and need and need > mb * 1.05:
        return (f"memory: the job asks for {mb / 1024:.0f}G (--mem in CFDT_SLURM_ARGS) but ~{est['cells'] / 1e6:.1f}M "
                f"cells may need {need / 1024:.0f}G: set run.mem (e.g. '{int(need // 1024 + 1)}G'), or have the admin "
                "move --mem from CFDT_SLURM_ARGS to CFDT_SLURM_MEM=auto")
    return None


def guide(topic: str | None = None) -> dict:
    from .guides import guide as _g
    return _g(topic)


def add_geometry(name: str, goal=None, step=None, step_base64=None, stl=None, stl_base64=None, scale=1.0,
                 groups=None) -> dict:
    from .variants import add_geometry as _add
    return _add(name, goal, step, step_base64, stl, stl_base64, scale, groups)


def list_geometry() -> dict:
    from .variants import list_geometry as _list
    return _list()


def _run_one(job: dict) -> dict:
    from .runner import run_study

    t0 = time.perf_counter()
    r = run_study(job["name"], job["study"], Path(job["folder"]) / job["name"], job["solver"], job.get("timeout", 3600))
    r.setdefault("seconds", {})
    if isinstance(r["seconds"], dict):
        r["seconds"]["wall"] = round(time.perf_counter() - t0, 3)
    return r


class _Pool:
    """Worker processes that live across calls; each study writes <request>/<name>/<name>.json when done."""

    def __init__(self):
        self.pool = None
        self.jobs: dict[str, dict] = {}

    def submit(self, folder: Path, jobs: list[dict]) -> None:
        workers = int(os.environ.get("CFDT_WORKERS", "2"))
        from .executor import timeout_s
        if self.pool is None:
            from .workers import watch_parent       # every worker ends with this server
            self.pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=10, initializer=watch_parent,
                                                     initargs=(os.getpid(),))
        self.jobs[str(folder)] = {j["name"]: self.pool.apply_async(_run_one, (dict(j, timeout=timeout_s(j["study"])),))
                                  for j in jobs}


_POOL = _Pool()

# ---- other slow work of a tool call (meshing for get_case, rendering) in worker processes: a call returns within
# its wait_s (at most 50 s) and a repeated call with the same arguments picks the result up (llm-translators'
# router counts on every tool returning in time; a server stuck in a call blocks every other call on it)
_AUX: dict = {"pool": None, "pending": {}}


def _aux_pool():
    if _AUX["pool"] is None:
        from .workers import watch_parent
        _AUX["pool"] = mp.get_context("spawn").Pool(int(os.environ.get("CFDT_AUX_WORKERS", "2") or 2),
                                                     initializer=watch_parent, initargs=(os.getpid(),),
                                                     maxtasksperchild=20)
    return _AUX["pool"]


def _aux_run(fn_name: str, kwargs: dict):
    return globals()[fn_name](**kwargs)


def aux_busy() -> bool:
    return any(not h.ready() for h, _ in _AUX["pending"].values())


def in_worker(what: str, fn_name: str, kwargs: dict, wait: float) -> dict:
    """fn_name(**kwargs) in a worker; waits at most `wait` s (<= 50), and within an MCP call's deadline (3 s left
    for the answer: llmt_runtime.requests.until). Unfinished: {state: running, ...}; the same call again waits for the
    same work."""
    from llmt_runtime.requests import until
    key = f"{fn_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
    if key not in _AUX["pending"]:
        _AUX["pending"][key] = (_aux_pool().apply_async(_aux_run, (fn_name, kwargs)), time.time())
    h, t0 = _AUX["pending"][key]
    h.wait(max(0.0, until(min(float(wait), 50.0)) - time.time()))
    if not h.ready():
        return {"state": "running", "what": what, "elapsed_s": round(time.time() - t0, 1),
                "next": f"still {what}: call {fn_name if fn_name != 'case' else 'get_case'} again with the same arguments "
                        "(the work goes on meanwhile)"}
    _AUX["pending"].pop(key, None)
    try:
        return h.get()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _request_dir(prefix: str) -> Path:
    d = out_root() / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collect(folder: Path) -> dict:
    """Every study of a request: results, or pending with progress and a state. SLURM studies (job.json)
    are judged from their files and the queue, so any process sees them; local ones from this process's pool."""
    from . import executor

    meta = json.loads((folder / "request.json").read_text())
    handles = _POOL.jobs.get(str(folder), {})
    names = meta["names"]
    studies = meta.get("studies", {})
    slurm = executor.slurm_states(folder, names, studies)
    out, pending, progress, job_ids = dict(meta.get("invalid", {})), [], {}, {}
    for name in names:
        f = folder / name / f"{name}.json"
        h = handles.get(name)
        if f.is_file() and (h is None or h.ready()):
            try:
                out[name] = json.loads(f.read_text())
                continue
            except json.JSONDecodeError:
                pass
        if name in slurm:
            st = slurm[name]
            job_ids[name] = st.get("job_id")
            if st["state"] in executor.FINAL:
                out[name] = st
            else:
                pending.append(name)
                progress[name] = st
            continue
        if h is not None and h.ready():
            try:
                out[name] = h.get()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"ok": False, "why": "worker failed", "error": str(exc)[:500]}
        elif h is None and (folder / name / "cancel").exists():
            out[name] = {"ok": False, "state": "cancelled",
                         "why": "cancelled with cancel(request) before it finished (it ran in this server's workers)"}
        elif h is None:
            out[name] = {"ok": False, "state": "failed",
                         "why": "not finished and not running here (the server restarted?); run it again"}
        else:
            pending.append(name)
            pr = _progress(folder / name, studies.get(name, {}))
            pr["state"] = "running" if (folder / name / "case").exists() else "queued"
            if pr["state"] == "queued":
                pr["reason"] = "waiting for a worker of this server (CFDT_WORKERS)"
            progress[name] = pr
    for v in out.values():
        if isinstance(v, dict):
            v.pop("trace", None)
    # the version the studies ran with (their results', else request.json's), and translator_now when this server
    # runs another (llmt-runtime's ran_with; before 2026.11.09.2 the installed version, whatever the run used)
    from llmt_runtime import requests as R

    from .runtime import RT
    res = {"request": str(folder), **R.ran_with(RT, meta, out), "solver": meta["solver"], "done": not pending,
           "pending": pending, "elapsed_s": round(time.time() - meta.get("started", time.time()), 1), "studies": out}
    if pending:
        res["progress"] = progress
        etas = [p["eta_min_max"] for p in progress.values() if p.get("eta_min_max") is not None]
        eta = max(etas) if etas else 0
        sl = [job_ids[n] for n in pending if n in job_ids]
        if sl:
            # a study at its last iteration has eta 0 and is reading results: that is an estimate, not none
            when = (f"the slowest has at most {max(eta, 1):.0f} min left once running"
                    if etas and len(etas) == len([n for n in pending if n in progress]) else
                    "no estimate yet: the time per iteration shows once OpenFOAM iterates (CFDT_TIMEOUT bounds a run)")
            res["next"] = (f"The studies are SLURM jobs on this machine (ids {', '.join(map(str, sl))}). They keep running "
                           f"without this conversation. Tell the user when they should finish ({when}) and stop; "
                           f"collect them later with study_status(request='{folder}') or find them with jobs().")
        else:
            res["next"] = (f"call study_status(request='{folder}') to collect the rest. Polling costs tokens: if eta_min_max "
                           f"(now {eta:.0f} min) is more than about 5 min, tell the user when to come back and stop polling")
    if meta.get("groups"):
        from .variants import compare
        res["variants"] = {b: compare(out, g) for b, g in meta["groups"].items()}
    if not pending:
        (folder / "results.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def _progress(d: Path, study: dict) -> dict:
    """_progress_of, plus what else is running on the job's CPUs right now (cpushare.py), if anything."""
    out = _progress_of(d, study)
    if isinstance(out, dict):
        try:                                   # the mesh it is solving on (runner writes mesh.json after meshing)
            m = json.loads((d / "mesh.json").read_text()) if (d / "mesh.json").is_file() else None
        except (OSError, ValueError):
            m = None
        if m:
            out["mesh"] = {k: m[k] for k in ("cells", "kind", "max_non_orthogonality", "non_ortho_over_70",
                                             "worst_non_ortho", "regions_removed") if k in m}
            lay = m.get("layers") or {}
            if "coverage_pct" in lay:
                out["mesh"]["layers_coverage_pct"] = lay["coverage_pct"]
            try:                               # the estimate the job was sized with, next to the real count
                ce = json.loads((d / "job.json").read_text()).get("cells_estimate") if (d / "job.json").is_file() else None
            except (OSError, ValueError):
                ce = None
            if ce:
                out["mesh"]["cells_estimate"] = ce
            if m.get("warnings"):
                out["mesh_warnings"] = m["warnings"][:6]
        from .foam import diverged, divergence_message
        at = diverged(d / "case" / "log.foamRun") if (d / "case" / "log.foamRun").is_file() else None
        if at is not None:
            out["diverged"] = divergence_message(at) + " (the job stops it at its next check)"
        else:
            slow = _solver_limit(d / "case" / "log.foamRun")
            if slow:
                out["solver_note"] = slow
    f = d / "cpu_share.json"
    try:
        sh = json.loads(f.read_text()) if f.is_file() else None
    except (OSError, ValueError):
        sh = None
    if sh and time.time() - float(sh.get("at", 0)) < 180 and isinstance(out, dict):
        out["shared_cpus"] = {k: sh[k] for k in ("cores_taken", "of", "by") if k in sh}
        out["shared_cpus"]["note"] = "other processes are using this job's CPUs: it runs slower (see warnings)"
    return out


def _solver_limit(log: Path) -> str | None:
    """'p hits its 1000-iteration limit' when the last iteration's pressure solves all ran to the limit."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - 20000))
            tail = f.read().decode(errors="replace")
    except OSError:
        return None
    import re as _re
    last = tail.rsplit("\nTime = ", 1)[0].rsplit("\nTime = ", 1)[-1]
    its = [int(x) for x in _re.findall(r"Solving for p\w*, .*No Iterations (\d+)", last)]
    if its and min(its) >= 1000:
        return ("the pressure solver runs to its 1000-iteration limit every time: it is not converging on this mesh "
                "(bad cells or cut-off pockets: see mesh), so each iteration is very slow")
    return None


def _progress_of(d: Path, study: dict) -> dict:
    """Where a running study is: meshing, or iteration i of at most n with an upper-bound ETA."""
    import re as _re
    log = d / "case" / "log.foamRun"
    if not log.is_file():
        sl = d / "snappy" / "log.snappyHexMesh"
        if sl.is_file() and not (d / "case").exists():
            try:
                t = sl.read_text(errors="replace")[-400000:]
            except OSError:
                t = ""
            stage = ("adding layers" if "Layer addition iteration" in t or "Adding layers" in t else
                     "snapping" if "Morphing phase" in t or "Snapping" in t else "refining")
            return {"stage": f"meshing (snappyHexMesh: {stage})",
                    "elapsed_min": round((time.time() - sl.stat().st_ctime) / 60, 1)}
        ml = d / "case" / "log.mapFields"
        if ml.is_file():                       # init_from: mapping the source's fields onto this mesh (serial)
            try:
                mt = ml.read_text(errors="replace")
            except OSError:
                mt = ""
            if "\nEnd" not in mt:
                done = re.findall(r"^\s+interpolating (\S+)", mt, re.M)
                from .transient import map_timeout
                return {"stage": "mapping fields (init_from, mapFields, serial)",
                        "elapsed_min": round((time.time() - ml.stat().st_ctime) / 60, 1),
                        **({"fields_mapped": done} if done else {}),
                        "limit_min": round(map_timeout() / 60, 1)}
        return {"stage": "meshing / writing the case" if (d / "case").exists() else "starting"}
    try:
        text = log.read_text(errors="replace")[-200000:]
    except OSError:
        return {"stage": "solving"}
    it = _re.findall(r"^Time = ([-+0-9.eE]+)", text, _re.M)
    s = study.get("solve", {})
    trs = s.get("transient")
    # no steady start when the run starts from another run's fields (init_from): reading its time steps as steady
    # iterations showed 'iteration 1 of 500' at 34,000 s an iteration for runs at t = 0.8 and 1.6 s (2026.11.07)
    from .transient import init_ref
    init = (int(trs.get("init", 0 if (study.get("model", "laminar") == "laminar" or init_ref(study)[0]) else 500))
            if trs else 0)
    if trs and (init == 0 or (d / "case" / "log.foamRun.init").exists()):
        from .transient import progress as _tp
        T = float(trs["end_time"])
        pr = _tp(text, T)
        el = max(float(pr.pop("job_clock_s", 0.0)), 1e-3)
        out = {"stage": "solving (transient)", **pr, "elapsed_min": round(el / 60, 1)}
        try:
            head = log.open("r", errors="replace").read(200000)
            t_first = float(_re.findall(r"^Time = ([-+0-9.eE]+)", head, _re.M)[0])
        except (OSError, IndexError, ValueError):
            t_first = None
        if it and t_first is not None and el > 1:
            t1 = float(it[-1])
            rate = (t1 - t_first) / el
            if rate > 0:
                out["eta_min_max"] = round((T - t1) / rate / 60, 1)
                out["note"] = ("eta for the rest of the simulated time at this job's pace; a run longer than one job "
                               "goes on in the next (job.json segment)")
        try:
            out["job_segment"] = json.loads((d / "job.json").read_text()).get("segment", 1)
        except (OSError, json.JSONDecodeError):
            pass
        return out
    if trs:
        s = dict(s, iterations=init)
    elif isinstance(study.get("cavitation"), dict):
        s = dict(s, iterations=int(study["cavitation"].get("iterations", s.get("iterations", 4000))))
    nmax = int(s.get("iterations", 3000 if study.get("model", "laminar") != "laminar" else 2000))
    start = (d / "case" / "log.checkMesh").stat().st_mtime if (d / "case" / "log.checkMesh").exists() else log.stat().st_ctime
    el = max(time.time() - start, 1e-3)
    if not it:
        return {"stage": "solving (starting)", "elapsed_min": round(el / 60, 1)}
    i = float(it[-1])
    per = el / max(i, 1)
    return {"stage": "solving" if i < nmax else "finishing (fields, results, pictures)", "iteration": int(i),
            "max_iterations": nmax, "s_per_iteration": round(per, 3),
            "elapsed_min": round(el / 60, 1), "eta_min_max": round(per * max(nmax - i, 0) / 60, 1),
            "note": "eta is an upper bound: the run stops earlier when it converges"}


from llmt_runtime.requests import REQUEST_NAME  # noqa: E402,F401 (run_<id>: the router names requests)


def start(studies: dict, solver: str | None = None, request_name: str | None = None) -> Path:
    if request_name:                       # chosen by the caller (the router), so it knows the path before the answer
        folder = out_root() / request_name
        folder.mkdir(parents=True, exist_ok=False)
    else:
        folder = _request_dir("run")
    (folder / "studies.json").write_text(json.dumps(studies, indent=1))
    s = _solver(solver)
    studies, groups, pre = prepare_studies(studies)
    from .transient import continue_problems, init_problems
    for n, st in studies.items():
        if isinstance(st, dict) and n not in pre and (continue_problems(st) or init_problems(st, list(studies), n)):
            pre[n] = continue_problems(st) + init_problems(st, list(studies), n)
    invalid, jobs = {}, []
    for n, st in studies.items():
        v = validate_study(st) if n not in pre else {"ok": False, "errors": pre[n]}
        if not v["ok"]:
            invalid[n] = {"ok": False, "why": "invalid study", "errors": v["errors"]}
        else:
            jobs.append({"name": n, "study": st, "folder": str(folder), "solver": s})
    for n, e in pre.items():              # a study whose variants could not be spelled out
        if n not in studies:
            invalid[n] = {"ok": False, "why": "invalid study", "errors": e}
    from llmt_runtime import requests as R

    from . import executor as _ex
    from .runtime import RT
    # with the versions it is submitted with ("translator", llmt-runtime's "runtime"): status says what it ran with
    (folder / "request.json").write_text(json.dumps(R.request_meta(RT, names=[j["name"] for j in jobs], invalid=invalid,
                                                                   solver=s, started=time.time(), executor=_ex.kind(),
                                                                   studies={j["name"]: j["study"] for j in jobs},
                                                                   groups=groups), default=float))
    from . import executor
    from .transient import init_ref
    # a study that starts from another of this request (init_from '<study>': a chained NPSH sweep) goes after it:
    # its SLURM job waits for the other's (afterany), a local one waits in the runner
    same = {j["name"]: init_ref(j["study"])[0] for j in jobs}
    same = {n: r for n, r in same.items() if r and "/" not in r and r in same}
    order, placed = [], set()
    while len(order) < len(jobs):
        ready = [j for j in jobs if j["name"] not in placed and (same.get(j["name"]) in placed or j["name"] not in same)]
        if not ready:                          # a cycle: validation does not allow one, but never loop forever
            ready = [j for j in jobs if j["name"] not in placed]
        for j in ready:
            order.append(j)
            placed.add(j["name"])
    local, ids = [], {}
    for j in order:
        if executor.use_slurm(j["study"], s):
            info = executor.submit(folder, j["name"], j["study"], after=ids.get(same.get(j["name"])))
            if info.get("job_id"):
                ids[j["name"]] = str(info["job_id"])
        else:
            local.append(j)
    if local:
        _POOL.submit(folder, local)
    return folder


# keys that make a result long and that a model rarely needs once the setup is right (the full result stays
# in <request>/<name>/<name>.json and results.json; detail="full" returns it)
_HEAVY_RESULTS = ("feat_loads", "feat_thermal")


def summarize(r: dict) -> dict:
    """A study result without the face-selection dump and per-face lists: what each bc took becomes counts
    and areas; results keep coefficients, forces, flow, pressure, heat, checks, probes, images, averaged
    (without its history), scalars, age, porous, rotating, pump; feat_loads/feat_thermal keep their totals."""
    if not isinstance(r, dict) or "results" not in r and "selection" not in r:
        return r
    out = {k: v for k, v in r.items() if k not in ("selection", "case", "mesh")}
    if isinstance(r.get("selection"), dict):
        out["selection"] = {bc: {"faces": len(v), "area_mm2": round(sum(float(x.get("area_mm2", 0)) for x in v), 4)}
                            for bc, v in r["selection"].items() if isinstance(v, list)}
    m = r.get("mesh") or {}
    if m:
        out["mesh"] = {k: m[k] for k in ("cells", "kind", "max_non_orthogonality", "ok", "layers_y_plus", "layers") if k in m}
        if isinstance(m.get("snappy"), dict):
            out["mesh"]["snappy"] = {k: v for k, v in m["snappy"].items()
                                     if k in ("cells_by_stage", "illegal_faces", "narrow_gaps", "attempts", "surface_mm")}
    res = r.get("results")
    if isinstance(res, dict):
        rs = {k: v for k, v in res.items() if k not in _HEAVY_RESULTS}
        for k in ("feat_loads", "feat_thermal"):
            if isinstance(res.get(k), dict):
                rs[k] = {kk: vv for kk, vv in res[k].items() if kk in ("total", "note")}
                rs[k]["note_summary"] = "per-face bcs left out: study_status(..., detail='full') or get_file"
        if isinstance(rs.get("averaged"), dict):
            rs["averaged"] = {k: v for k, v in rs["averaged"].items() if k != "history"}
        out["results"] = rs
    sv = r.get("solve")
    if isinstance(sv, dict):
        out["solve"] = {k: v for k, v in sv.items() if k not in ("checkMesh", "first_residuals")}
    out["detail"] = "summary"
    return out


def status(request: str, wait: float = 0.0, detail: str = "full", window: list | None = None) -> dict:
    folder = Path(request).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    if window is not None:
        return _window(folder, window)
    from llmt_runtime.requests import until
    t0 = time.time()
    # within an MCP call's deadline (48 s): the last look starts with 3 s left for the answer (outside a call: wait)
    t_end = until(wait)
    while True:
        res = _collect(folder)
        if res["done"] or time.time() >= t_end:
            return _detail(res, detail)
        # SLURM jobs do not need this process: give quick ones up to 15 s, then return them pending
        prog = res.get("progress", {})
        if prog and all("job_id" in v for v in prog.values()) and time.time() - t0 >= 15:
            return _detail(res, detail)
        time.sleep(max(0.0, min(2.0, t_end - time.time())))


def _window(folder: Path, window) -> dict:
    """Averages of the request's finished transient studies over window [t0, t1] (t1 null: to the end)."""
    from .average import window_average
    if not (isinstance(window, (list, tuple)) and len(window) == 2 and isinstance(window[0], (int, float))
            and (window[1] is None or isinstance(window[1], (int, float)))
            and (window[1] is None or window[1] > window[0])):
        return {"error": "window: [t0, t1] in s with t0 < t1 (t1 null: to the end of the run)"}
    meta = json.loads((folder / "request.json").read_text())
    out = {}
    for n in meta["names"]:
        if not (folder / n / f"{n}.json").is_file():
            out[n] = {"error": "not finished"}
            continue
        try:
            out[n] = window_average(folder / n, n, (meta.get("studies") or {}).get(n, {}), float(window[0]),
                                    None if window[1] is None else float(window[1]))
        except Exception as exc:  # noqa: BLE001
            out[n] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"request": str(folder), "translator": __version__, "window": list(window), "studies": out}


def pack(request: str, state: bool = False, out: str | None = None) -> dict:
    """On a cluster: a .tgz of a request's results to bring back (transfer.pack; other/supercomputers/clusters.md)."""
    from .transfer import pack as _pack
    return _pack(request, state, out)


def import_results(path: str, name: str | None = None) -> dict:
    """A request packed on a cluster (cfdt pack), as a request of this machine: then study_status and the rest."""
    from .transfer import import_request
    r = import_request(path, name)
    if "error" in r:
        return r
    st = status(r["request"], 0.0, "summary")
    return {**r, "status": {k: st[k] for k in ("done", "studies", "variants") if k in st}}


def reprocess(request: str, studies: list[str] | None = None) -> dict:
    """Results again for the finished studies of a request (all, or those named), from their cases, with the
    current post-processing: no solving. Running or failed studies are skipped."""
    from .runner import reprocess_study
    folder = Path(request).expanduser().resolve()
    root = out_root().resolve()
    if root not in folder.parents or not (folder / "request.json").is_file():
        return {"error": f"no request {folder} under {root}"}
    names = json.loads((folder / "request.json").read_text())["names"]
    if studies:
        bad = [n for n in studies if n not in names]
        if bad:
            return {"error": f"not in this request: {bad}", "studies": names}
        names = [n for n in names if n in studies]
    out = {}
    for n in names:
        try:
            out[n] = reprocess_study(folder / n, n)
        except Exception as exc:  # noqa: BLE001
            out[n] = {"name": n, "ok": False, "why": f"{type(exc).__name__}: {exc}"}
    return {"request": str(folder), "translator": __version__, "studies": out,
            "next": "study_status(request) gives the new results"}


def brief(r: dict) -> dict:
    """A study result in a few lines: verdict, warnings, the main numbers, mesh size and layers, timing."""
    if not isinstance(r, dict) or "results" not in r:
        return r
    res = r["results"]
    out = {k: r[k] for k in ("ok", "reliable", "unreliable_because", "warnings") if k in r}
    keep = {}
    for k in ("coefficients", "pressure_drop", "total_pressure_drop", "head_loss_m", "y_plus_part_mean",
              "max_mach", "pump", "rotating", "age"):
        if k in res:
            keep[k] = res[k]
    if isinstance(keep.get("coefficients"), dict):
        keep["coefficients"] = {k: v for k, v in keep["coefficients"].items()
                                if k in ("cd", "cl", "cm", "cd_std", "cl_std", "cd_last", "averaged_over",
                                         "cd_walls_only")}
    if "flow" in res:
        keep["flow"] = {b: {k: v for k, v in f.items() if k in ("Q", "mdot", "U_mean")} for b, f in res["flow"].items()}
    if "forces" in res:
        keep["forces"] = {b: {k: f[k] for k in ("fx", "fy", "fz") if k in f} for b, f in res["forces"].items()}
    if "heat" in res:
        keep["heat"] = {b: {k: h[k] for k in ("Q_W", "T_mean_C") if k in h} for b, h in res["heat"].items()}
    out["results"] = keep
    ck = res.get("checks") or {}
    out["checks"] = {k: (v.get("error") if isinstance(v, dict) else v) for k, v in ck.items()
                     if k in ("mass_balance", "momentum_balance", "energy_balance")}
    sv = r.get("solve") or {}
    out["solve"] = {k: sv[k] for k in ("iterations", "converged", "settled", "procs") if k in sv}
    m = r.get("mesh") or {}
    out["mesh"] = {k: m[k] for k in ("cells", "kind", "max_non_orthogonality") if k in m}
    lay = m.get("layers")
    if isinstance(lay, dict):
        out["mesh"]["layers"] = {k: lay[k] for k in ("n", "first_mm", "coverage_pct", "layers_mean",
                                                    "y_plus_first_cell_estimate") if k in lay}
    if r.get("images"):
        out["images"] = {k: v.get("path") for k, v in r["images"].items() if isinstance(v, dict)}
    out["seconds"] = (r.get("seconds") or {}).get("wall")
    out["detail"] = "brief"
    return out


def _detail(res: dict, detail: str) -> dict:
    if detail == "brief":
        return dict(res, studies={n: brief(r) for n, r in (res.get("studies") or {}).items()})
    if detail != "summary":
        return res
    return dict(res, studies={n: summarize(r) for n, r in (res.get("studies") or {}).items()})


def run(studies: dict, solver: str | None = None, wait: float | None = None, detail: str = "full",
        request_name: str | None = None) -> dict:
    """Runs the studies into a new request folder under CFDT_OUT: per study <name>/case (the OpenFOAM
    case with logs and results), <name>_case.tgz, <name>.json; and studies.json, results.json.
    request_name ('run_<anything>'): the folder's name, chosen by the caller so that it knows the request path even
    when the answer does not reach it in time; a call repeating a name that exists returns that request's status
    instead of submitting again (a retry after a time-out cannot run the studies twice)."""
    from llmt_runtime import requests as R

    from .runtime import RT
    t0 = time.time()
    kind_, found = R.check_request_name(RT, request_name)     # llmt-runtime: new | repeat (its status) | error
    if kind_ == "error":
        return found
    if kind_ == "repeat":
        res = status(str(found), 0.0 if wait is None else min(wait, 15.0), detail)
        return dict(res, note=R.REPEAT_NOTE)
    try:
        folder = start(studies, solver, request_name)
    except FileExistsError:                                   # another call took the name between the look and now
        return {"error": f"request_name {request_name!r}: the folder exists but holds no request (still being "
                         "submitted by an earlier call? call again in a few seconds)",
                "request": str(out_root() / request_name)}
    # the wait counts from the call's start: building goals and submitting can take a while on a busy machine,
    # and an MCP call must end within a minute (the router's limit)
    try:
        return status(str(folder), 1e9 if wait is None else max(0.0, wait - (time.time() - t0)), detail)
    except KeyboardInterrupt:          # `cfdt run` interrupted: say which SLURM jobs go on and how to stop them
        from .executor import print_queued_note
        print_queued_note(folder)
        raise


def _requests(limit: int):
    root = out_root()
    if not root.is_dir():
        return []
    reqs = [p for p in root.iterdir() if p.is_dir() and (p / "request.json").is_file()]
    return sorted(reqs, key=lambda p: p.stat().st_mtime, reverse=True)[: max(limit * 5, limit)]


def jobs(limit: int = 20, active: bool = False) -> dict:
    """The newest requests with SLURM jobs (any server process, any chat), each study's state, job id and ETA;
    the queue of this translator's jobs and the machine's CPUs. The states are llmt-runtime's job_states and the
    machine's CPUs its queue_summary (executor.slurm_states, executor.queue_summary), with cfdt's progress, ETAs and
    per-node counts; the answer keeps cfdt's shape."""
    from . import executor

    out = []
    for req in _requests(limit):
        meta = json.loads((req / "request.json").read_text())
        names = meta.get("names", [])
        if not any((req / n / "job.json").is_file() for n in names):
            continue
        st = executor.slurm_states(req, names, meta.get("studies", {}))
        studies = {}
        for n in names:
            jf = req / n / "job.json"
            jid = json.loads(jf.read_text()).get("job_id") if jf.is_file() else None
            if n in st:
                s_ = st[n]
                studies[n] = {k: s_.get(k) for k in ("state", "job_id", "reason", "expected_start", "expected_start_slurm",
                                                      "ahead", "stage", "iteration", "max_iterations", "time", "end_time",
                                                      "eta_min_max", "job_elapsed_min", "why")
                              if s_.get(k) is not None}
                if isinstance(s_.get("mesh"), dict) and s_["mesh"].get("cells"):
                    studies[n]["mesh_cells"] = s_["mesh"]["cells"]
            elif (req / n / f"{n}.json").is_file():
                r = json.loads((req / n / f"{n}.json").read_text())
                studies[n] = {"state": r.get("state", "done" if r.get("ok") else "failed"), "job_id": jid,
                              "ok": r.get("ok"), **({"reliable": r["reliable"]} if "reliable" in r else {})}
            elif (req / n / "cancel").exists():
                studies[n] = {"state": "cancelled", "job_id": jid}
            else:
                studies[n] = {"state": "local", "job_id": jid}
        live = any(v.get("state") in ("queued", "running") for v in studies.values())
        if active and not live:
            continue
        out.append({"request": str(req), "started": meta.get("started"), "active": live, "studies": studies})
        if len(out) >= limit:
            break
    return {"translator": __version__, "executor": executor.status(), **executor.queue_summary(), "requests": out}


def job_progress(folder: Path, study: dict) -> dict:
    """Where a running SLURM study is, as jobs() shows it (llmt-runtime's hooks.running_progress): stage, iteration,
    max_iterations, time, end_time, eta_min_max and mesh_cells from its logs."""
    pr = _progress(Path(folder), study or {})
    if not isinstance(pr, dict):
        return {}
    out = {k: pr[k] for k in ("stage", "iteration", "max_iterations", "time", "end_time", "eta_min_max") if k in pr}
    if isinstance(pr.get("mesh"), dict) and pr["mesh"].get("cells"):
        out["mesh_cells"] = pr["mesh"]["cells"]
    return out


def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops the request's studies (or the named ones): scancel for SLURM jobs; for local studies a cancel
    flag that the running solver (in whatever process runs it) checks every second. The flags and scancel are
    llmt-runtime's cancel_request (2026.11.09.4); the local pool's side stays here."""
    from .runtime import RT

    folder = Path(request).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    meta = json.loads((folder / "request.json").read_text())
    c = RT.jobs.cancel_request(folder, meta, names)
    if "want" not in c:                      # unknown names
        return c
    names, todo, ids = c["want"], c["todo"], c["cancelled_jobs"]
    local = [n for n in todo if not (folder / n / "job.json").is_file()]
    here = [n for n in local if n in _POOL.jobs.get(str(folder), {})]
    time.sleep(1.0 if ids or here else 0.0)
    res = status(str(folder), 3.0 if (ids or here) else 0.0)
    out = {"request": str(folder), "cancelled_jobs": ids, "already_finished": c["already_finished"],
           "states": {n: (res["studies"].get(n, {}).get("state") or ("done" if n in res["studies"] else
                          res.get("progress", {}).get(n, {}).get("state"))) for n in names}}
    if c.get("error"):
        out["error"] = c["error"]
    if [n for n in local if n not in here]:
        out["note"] = ("local studies not run by this server process: a cancel flag was left; the process running "
                       "them (if any) stops them within seconds, otherwise they are not running")
    return out


def run_study_job(request: str, name: str) -> int:
    from . import executor
    return executor.run_study_job(request, name)


def case(study: dict, name: str = "study") -> dict:
    """Meshes the study and writes its OpenFOAM case without solving; returns the dictionaries as text."""
    from .runner import prepare
    from .variants import resolve

    if isinstance(study, dict):           # geometry {"ref": name} as in run_studies
        study, err = resolve(study)
        if err:
            return {"error": "invalid study", "errors": [err]}
    v = validate_study(study)
    if not v["ok"]:
        return {"error": "invalid study", "errors": v["errors"]}
    folder = _request_dir("case") / name
    folder.mkdir(parents=True)
    try:
        mesh, patches, selection, plan, warns = prepare(study, folder)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", **({"warnings": exc.notes} if getattr(exc, "notes", None) else {})}
    c = folder / "case"
    texts = {}
    for rel in ("system/controlDict", "system/fvSchemes", "system/fvSolution", "constant/physicalProperties",
                "constant/momentumTransport", "constant/polyMesh/boundary", "0/U", "0/p", "0/k", "0/omega", "0/nut"):
        p = c / rel
        if p.is_file():
            t = p.read_text()
            texts[rel] = t if len(t) < 8000 else t[:8000] + "\n// ... (truncated; see the file)\n"
    return {"case": str(c), "mesh": plan["quality"], "selection": selection, "inlets": plan["inlets"],
            "warnings": warns + v["warnings"], "files": texts}


def render_images(study_folder: str, images=True) -> dict:
    """New pictures of a finished study (its folder: <request>/<name>) from the saved mesh and fields."""
    from .format import image_errors
    from .render import default_images, load_state, render

    d = Path(study_folder).expanduser().resolve()
    root = out_root().resolve()
    if root not in d.parents:
        return {"error": f"only study folders under {root}"}
    if not (d / "state.npz").is_file():
        return {"error": f"no saved state in {d} (a finished study's folder: <request>/<name>)"}
    mesh, plan, fields, study = load_state(d)
    from .format import _field_names
    errs = image_errors(images, bool(study.get("2d")), _field_names(study),
                        bool((study.get("reference") or {}).get("velocity")))
    if errs:
        return {"error": "invalid images", "errors": errs}
    specs = default_images(study, mesh) if images is True else images
    out, warns = render(study, mesh, plan, fields, d / "images", specs, d.name)
    return {"images": out, "warnings": warns}


def image_path(path: str) -> Path:
    """A PNG written by a run or by render_images (only under the output folder)."""
    p = Path(path).expanduser().resolve()
    root = out_root().resolve()
    if root not in p.parents or p.suffix.lower() != ".png" or not p.is_file():
        raise ValueError(f"no image {p} under {root}")
    return p


def bench(names=None, solver: str | None = None, wait_s: float | None = None) -> dict:
    """wait_s (the MCP tool): submit (SLURM jobs, or the local worker pool) and return after at most wait_s;
    bench_status collects. Without wait_s (the CLI): run them here."""
    from . import bench as B

    s = _solver(solver)
    if wait_s is not None:          # the MCP tool: submitted (SLURM jobs or the local pool), never run in the call
        folder = _request_dir("bench")
        sub = B.submit(s, names)
        sub.update({"solver": s, "translator": __version__})
        (folder / "bench_request.json").write_text(json.dumps(sub, indent=1))
        return bench_status(str(folder), wait_s)
    folder = _request_dir("bench")
    r = B.run(s, names, folder)
    passed = [k for k, v in r.items() if v.get("passed")]
    failed = [k for k, v in r.items() if v.get("passed") is False or v.get("ok") is False]
    out = {"request": str(folder), "solver": s, "translator": __version__, "passed": passed, "failed": failed,
           "skipped": [k for k, v in r.items() if "skipped" in v], "cases": r}
    (folder / "bench.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def bench_status(bench_folder: str, wait_s: float = 45) -> dict:
    """The verdicts of benchmarks submitted by bench(..., wait_s): finished cases are evaluated, the rest
    listed as pending with their progress."""
    from . import bench as B
    folder = Path(bench_folder)
    meta = json.loads((folder / "bench_request.json").read_text())
    if not meta.get("request"):
        return {"bench": str(folder), "done": True, "passed": [], "failed": [], "skipped": meta["skipped"],
                "unknown": meta.get("unknown", [])}
    st = status(meta["request"], wait_s)
    cases = B._cases()
    res = {n: B.evaluate(n, r, cases) for n, r in st["studies"].items()}
    out = {"bench": str(folder), "request": meta["request"], "translator": meta.get("translator"), "done": st["done"],
           "passed": [n for n, v in res.items() if v.get("passed")],
           "failed": [n for n, v in res.items() if not v.get("passed")],
           "pending": st.get("pending", []), "skipped": meta["skipped"], "cases": res}
    if st.get("progress"):
        out["progress"] = st["progress"]
    if not st["done"]:
        out["next"] = (f"call bench_status(bench='{folder}') later; the cases are SLURM jobs and keep running "
                       "without this conversation (polling costs tokens: wait for the longest eta)")
    else:
        (folder / "bench.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def read_file(path: str, max_mb: float = 20) -> dict:
    p = Path(path).expanduser().resolve()
    root = out_root().resolve()
    if root not in p.parents:
        return {"error": f"only files under {root} can be read"}
    if not p.is_file():
        return {"error": f"no file {p}"}
    if p.stat().st_size > max_mb * 1e6:
        return {"error": f"{p.stat().st_size / 1e6:.1f} MB is more than max_mb"}
    return {"name": p.name, "bytes": p.stat().st_size, "base64": base64.b64encode(p.read_bytes()).decode()}
