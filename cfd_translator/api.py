"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.10.19.1

    study_format()                   the study format, for the caller to read
    validate(studies)                static checks, no meshing
    run(studies, solver, wait)       meshes, solves and checks each study in a request folder; with wait,
                                     returns after that many seconds with the rest pending
    start(studies, solver)           the same in the background: returns the request folder
    status(request, wait)            finished studies with results, the others pending (with state)
    jobs(limit, active)              recent requests with SLURM jobs and their states; the queue
    cancel(request, names)           scancel / cancel flag for the request's studies
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
    return out


def study_format() -> dict:
    return {**_format(), "translator": __version__, "status": tool_status()}


def validate(studies: dict) -> dict:
    return _validate(studies)


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
        timeout = float(os.environ.get("CFDT_TIMEOUT", "3600"))
        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=10)
        self.jobs[str(folder)] = {j["name"]: self.pool.apply_async(_run_one, (dict(j, timeout=timeout),)) for j in jobs}


_POOL = _Pool()


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
    res = {"request": str(folder), "translator": __version__, "solver": meta["solver"], "done": not pending,
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
    else:
        (folder / "results.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def _progress(d: Path, study: dict) -> dict:
    """Where a running study is: meshing, or iteration i of at most n with an upper-bound ETA."""
    import re as _re
    log = d / "case" / "log.foamRun"
    if not log.is_file():
        return {"stage": "meshing / writing the case" if (d / "case").exists() else "starting"}
    try:
        text = log.read_text(errors="replace")[-200000:]
    except OSError:
        return {"stage": "solving"}
    it = _re.findall(r"^Time = ([-+0-9.eE]+)", text, _re.M)
    s = study.get("solve", {})
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


def start(studies: dict, solver: str | None = None) -> Path:
    folder = _request_dir("run")
    (folder / "studies.json").write_text(json.dumps(studies, indent=1))
    s = _solver(solver)
    invalid, jobs = {}, []
    for n, st in studies.items():
        v = validate_study(st)
        if not v["ok"]:
            invalid[n] = {"ok": False, "why": "invalid study", "errors": v["errors"]}
        else:
            jobs.append({"name": n, "study": st, "folder": str(folder), "solver": s})
    from . import executor as _ex
    (folder / "request.json").write_text(json.dumps({"names": [j["name"] for j in jobs], "invalid": invalid,
                                                     "solver": s, "started": time.time(), "executor": _ex.kind(),
                                                     "studies": {j["name"]: j["study"] for j in jobs}}, default=float))
    from . import executor
    local = []
    for j in jobs:
        if executor.use_slurm(j["study"], s):
            executor.submit(folder, j["name"], j["study"])
        else:
            local.append(j)
    if local:
        _POOL.submit(folder, local)
    return folder


def status(request: str, wait: float = 0.0) -> dict:
    folder = Path(request).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    t0 = time.time()
    t_end = t0 + max(0.0, float(wait))
    while True:
        res = _collect(folder)
        if res["done"] or time.time() >= t_end:
            return res
        # SLURM jobs do not need this process: give quick ones up to 15 s, then return them pending
        prog = res.get("progress", {})
        if prog and all("job_id" in v for v in prog.values()) and time.time() - t0 >= 15:
            return res
        time.sleep(2.0)


def run(studies: dict, solver: str | None = None, wait: float | None = None) -> dict:
    """Runs the studies into a new request folder under CFDT_OUT: per study <name>/case (the OpenFOAM
    case with logs and results), <name>_case.tgz, <name>.json; and studies.json, results.json."""
    folder = start(studies, solver)
    return status(str(folder), 1e9 if wait is None else wait)


def _requests(limit: int):
    root = out_root()
    if not root.is_dir():
        return []
    reqs = [p for p in root.iterdir() if p.is_dir() and (p / "request.json").is_file()]
    return sorted(reqs, key=lambda p: p.stat().st_mtime, reverse=True)[: max(limit * 5, limit)]


def jobs(limit: int = 20, active: bool = False) -> dict:
    """The newest requests with SLURM jobs (any server process, any chat), each study's state, job id and ETA;
    the queue of this translator's jobs and the machine's CPUs."""
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
                studies[n] = {k: s_.get(k) for k in ("state", "job_id", "reason", "expected_start", "ahead", "iteration",
                                                      "max_iterations", "eta_min_max", "job_elapsed_min", "why")
                              if s_.get(k) is not None}
            elif (req / n / f"{n}.json").is_file():
                r = json.loads((req / n / f"{n}.json").read_text())
                studies[n] = {"state": r.get("state", "done" if r.get("ok") else "failed"), "job_id": jid,
                              "ok": r.get("ok"), **({"reliable": r["reliable"]} if "reliable" in r else {})}
            else:
                studies[n] = {"state": "local", "job_id": jid}
        live = any(v.get("state") in ("queued", "running") for v in studies.values())
        if active and not live:
            continue
        out.append({"request": str(req), "started": meta.get("started"), "active": live, "studies": studies})
        if len(out) >= limit:
            break
    return {"translator": __version__, "executor": executor.status(), **executor.queue_summary(), "requests": out}


def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops the request's studies (or the named ones): scancel for SLURM jobs; for local studies a cancel
    flag that the running solver (in whatever process runs it) checks every second."""
    from . import executor

    folder = Path(request).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    meta = json.loads((folder / "request.json").read_text())
    names = names or meta["names"]
    unknown = [n for n in names if n not in meta["names"]]
    if unknown:
        return {"error": f"no studies {unknown} in this request ({meta['names']})"}
    todo = [n for n in names if not (folder / n / f"{n}.json").is_file()]
    for n in todo:
        (folder / n).mkdir(parents=True, exist_ok=True)
        (folder / n / "cancel").write_text(str(time.time()))
    ids = executor.cancel_slurm(folder, todo)
    local = [n for n in todo if not (folder / n / "job.json").is_file()]
    here = [n for n in local if n in _POOL.jobs.get(str(folder), {})]
    time.sleep(1.0 if ids or here else 0.0)
    res = status(str(folder), 3.0 if (ids or here) else 0.0)
    out = {"request": str(folder), "cancelled_jobs": ids, "already_finished": [n for n in names if n not in todo],
           "states": {n: (res["studies"].get(n, {}).get("state") or ("done" if n in res["studies"] else
                          res.get("progress", {}).get(n, {}).get("state"))) for n in names}}
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

    v = validate_study(study)
    if not v["ok"]:
        return {"error": "invalid study", "errors": v["errors"]}
    folder = _request_dir("case") / name
    folder.mkdir(parents=True)
    try:
        mesh, patches, selection, plan, warns = prepare(study, folder)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
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
    from .format import _scalar_names
    errs = image_errors(images, bool(study.get("2d")), _scalar_names(study))
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


def bench(names=None, solver: str | None = None) -> dict:
    from . import bench as B

    s = _solver(solver)
    folder = _request_dir("bench")
    r = B.run(s, names, folder)
    passed = [k for k, v in r.items() if v.get("passed")]
    failed = [k for k, v in r.items() if v.get("passed") is False or v.get("ok") is False]
    out = {"request": str(folder), "solver": s, "translator": __version__, "passed": passed, "failed": failed,
           "skipped": [k for k, v in r.items() if "skipped" in v], "cases": r}
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
