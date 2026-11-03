"""foam.py — the only module that runs OpenFOAM: find it, run checkMesh and foamRun, read the fields.
Version: 2026.11.03.2

OpenFOAM (openfoam.org, version 11 or later: foamRun + incompressibleFluid) is found as:
CFDT_FOAM=/path/to/etc/bashrc (or 'PATH' when the environment is already sourced), else foamRun on the
PATH, else the newest /opt/openfoam*/etc/bashrc (the Ubuntu packages). Each command runs in its own
bash that sources the bashrc. CFDT_PROCS > 1 runs in parallel (decomposePar, mpirun, reconstructPar).
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import foamio


class FoamError(Exception):
    pass


class Cancelled(FoamError):
    """The study's cancel flag (<request>/<name>/cancel) appeared while OpenFOAM ran."""


class Diverged(FoamError):
    """The solver's residuals became NaN or infinite: it is stopped instead of running on (on aarch64 OpenFOAM
    cannot trap floating point exceptions, so a NaN run goes on at 1000 iterations per solve: s180, 2026.10.31,
    330 s per iteration)."""


WATCH_S = 20.0                               # how often a running solver's log is looked at for NaN
_NAN = re.compile(r"(?:Initial|Final) residual = [-+]?(?:nan|inf)|sum local = [-+]?(?:nan|inf)", re.I)
_TIME_LINE = re.compile(r"^Time = ([-+0-9.eE]+)", re.M)


def diverged(log: Path) -> str | None:
    """The time / iteration at which the log first shows NaN or infinite residuals, or None."""
    try:
        with open(log, "rb") as f:
            text = f.read().decode(errors="replace")
    except OSError:
        return None
    m = _NAN.search(text)
    if not m:
        return None
    t = _TIME_LINE.findall(text[:m.start()])
    return t[-1] if t else "the start"


def divergence_message(at: str) -> str:
    return (f"the solution diverged at iteration {at}: the residuals are NaN. Usual causes: parts of the mesh cut off "
            "from every inlet and outlet (closed pockets: checkMesh 'Number of regions' above 1), very bad cells "
            "(mesh.worst_non_ortho, skewness), or relaxation too high for the mesh (solve.relax)")


_FOUND = {}


def find_foam() -> dict | None:
    if "v" in _FOUND:
        return _FOUND["v"]
    env = os.environ.get("CFDT_FOAM")
    res = None
    if env:
        res = {"bashrc": None if env.upper() == "PATH" else env}
    elif shutil.which("foamRun"):
        res = {"bashrc": None}
    else:
        cands = glob.glob("/opt/openfoam*/etc/bashrc") + glob.glob(os.path.expanduser("~/OpenFOAM/OpenFOAM-*/etc/bashrc"))

        def ver(p):
            m = re.search(r"(?:openfoam|OpenFOAM-)(\d+)", p)
            return int(m.group(1)) if m else -1
        cands = [c for c in cands if ver(c) >= 0]
        if cands:
            res = {"bashrc": max(cands, key=ver)}
    if res is not None:
        try:
            r = _bash(res, 'echo "v=$WM_PROJECT_VERSION"; echo "r=$(command -v foamRun)"', Path.cwd(), timeout=60, capture=True)
            kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if l[:2] in ("v=", "r="))
            res["version"] = kv.get("v") or None
            res["foamRun"] = kv.get("r") or None
            if not res["foamRun"]:
                res = {"error": f"OpenFOAM found ({res.get('bashrc')}) but no foamRun: openfoam.org 11 or later is needed"}
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"cannot source OpenFOAM: {exc}"}
    _FOUND["v"] = res
    return res


_ACTIVE: set = set()                         # running OpenFOAM process groups (Popen) of this process


def kill_all() -> None:
    """Stop every OpenFOAM process group this process started (mpirun and its ranks)."""
    for p in list(_ACTIVE):
        _kill_group(p)


def _on_signal(signum, frame):
    kill_all()
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    """A SLURM job's python: on scancel (SIGTERM) or a hang-up, take the OpenFOAM processes down too. They run in
    their own process group (so a timeout can kill mpirun and every rank), which SLURM's process tracking by
    group does not reach: cancelled jobs left foamRun ranks running on spark1 and spark2 (2026.10.31)."""
    import signal
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_signal)


def _bash(foam: dict, cmd: str, cwd: Path, timeout: float, capture: bool = False, cancel_file: Path | None = None,
          deadline: float | None = None, on_deadline=None, watch_log: Path | None = None):
    """Runs cmd in a bash that sources OpenFOAM. Long commands (not capture) run in their own process group,
    which a timeout or the cancel flag kills as a whole (mpirun and every rank, not only bash). At the epoch
    time deadline, on_deadline() is called once (a transient run asked to write and stop)."""
    src = f"source '{foam['bashrc']}' >/dev/null 2>&1; " if foam.get("bashrc") else ""
    if capture:
        return subprocess.run(["bash", "-c", src + cmd], cwd=str(cwd), timeout=timeout, capture_output=True, text=True)
    env = dict(os.environ, CFDT_STUDY=str(Path(cwd).resolve()))     # marks cfdt's OpenFOAM processes (cpushare.reap)
    p = subprocess.Popen(["bash", "-c", src + cmd], cwd=str(cwd), start_new_session=True, env=env)
    _ACTIVE.add(p)
    try:
        return _wait(p, cmd, timeout, cancel_file, deadline, on_deadline, watch_log)
    finally:
        _ACTIVE.discard(p)


def _wait(p, cmd, timeout, cancel_file, deadline, on_deadline, watch_log):
    t0 = time.time()
    fired = False
    next_look = t0 + WATCH_S
    while True:
        if watch_log is not None and time.time() >= next_look:
            next_look = time.time() + WATCH_S
            at = diverged(watch_log)
            if at is not None:
                _kill_group(p)
                raise Diverged(divergence_message(at))
        if deadline is not None and on_deadline is not None and not fired and time.time() >= deadline:
            fired = True
            on_deadline()
        try:
            return subprocess.CompletedProcess(p.args, p.wait(timeout=1.0))
        except subprocess.TimeoutExpired:
            cancelled = cancel_file is not None and Path(cancel_file).exists()
            if cancelled or time.time() - t0 > timeout:
                _kill_group(p)
                if cancelled:
                    raise Cancelled("cancelled") from None
                raise subprocess.TimeoutExpired(cmd, timeout) from None


def _kill_group(p) -> None:
    import signal
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def _step(foam: dict, cmd: str, logname: str, case: Path, timeout: float, raise_on_error: bool = True,
          deadline: float | None = None, on_deadline=None) -> int:
    try:
        rc = _bash(foam, f"{cmd} > {logname} 2>&1", case, timeout=timeout,
                   cancel_file=Path(case).parent / "cancel", deadline=deadline, on_deadline=on_deadline,
                   watch_log=Path(case) / logname if "foamRun" in cmd else None).returncode
    except subprocess.TimeoutExpired as exc:
        raise FoamError(f"{cmd.split()[0]} took more than {timeout:.0f} s (CFDT_TIMEOUT)") from exc
    if rc != 0 and raise_on_error:
        raise FoamError(f"{cmd} exited with code {rc}:\n" + _tail(case / logname))
    return rc


def run(case: Path, timeout: float = 3600, procs: int | None = None, resume: bool = False,
        deadline: float | None = None) -> dict:
    """checkMesh, then the solver. Returns the parsed logs; raises FoamError on failure.
    resume (transient runs): continue from the latest time (no cleaning, no checkMesh; the decomposition is
    kept when it has as many ranks). deadline (epoch seconds): ask the run to write and stop then
    (transient.request_stop); info['reached_end'] says whether endTime was reached. The fields are
    reconstructed only when it was (or the run is steady)."""
    foam = find_foam()
    if not foam or "error" in (foam or {}):
        raise FoamError((foam or {}).get("error", "OpenFOAM not found (install openfoam.org >= 11, or set CFDT_FOAM)"))
    case = Path(case)
    out = {"openfoam": foam.get("version")}
    if not resume:
        for p in case.iterdir():                 # results of an earlier run must not be read as this one's
            try:
                if p.is_dir() and float(p.name) > 0:
                    shutil.rmtree(p)
            except ValueError:
                pass
        shutil.rmtree(case / "postProcessing", ignore_errors=True)     # monitors of an earlier run (average.py)
        _step(foam, "checkMesh", "log.checkMesh", case, min(timeout, 900), raise_on_error=False)
    else:
        k = len(list(case.glob("log.foamRun.part*"))) + 1
        if (case / "log.foamRun").exists():
            (case / "log.foamRun").rename(case / f"log.foamRun.part{k}")
    cm = case / "log.checkMesh"
    out["checkMesh"] = foamio.parse_checkmesh(cm.read_text(errors="replace")) if cm.exists() else {"ok": True}
    procs = int(procs or os.environ.get("CFDT_PROCS", "1"))
    out["procs"] = procs
    transient = resume or deadline is not None
    dec = decomposed(case)
    if dec and (procs <= 1 or dec["procs"] != procs):
        raise FoamError(f"this case stays decomposed for {dec['procs']} ranks (a sliding mesh's couple was built on the "
                        f"decomposed case: {DECOMPOSED_NOTE}); run it on {dec['procs']} ranks (solve.procs), not {procs}")
    if procs > 1:
        have = len(list(case.glob("processor*")))
        if not (resume and have == procs):
            if resume and have:                  # another rank count: gather the latest time, decompose again
                _step(foam, "reconstructPar -latestTime", "log.reconstructPar", case, timeout)
            (case / "system" / "decomposeParDict").write_text(
                "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      decomposeParDict;\n}\n\n"
                f"numberOfSubdomains {procs};\n\ndecomposer      scotch;\n")
            _step(foam, "decomposePar -force" + (" -latestTime" if resume else ""), "log.decomposePar", case, timeout)
        launch = f"{mpirun_command(procs)} -np {procs}"
        out["launch"] = launch
        out["binding"] = "hwthread" if "--map-by hwthread" in launch else "default"
        solve = (f"{launch} foamRun -parallel", "log.foamRun")
    else:
        solve = ("foamRun", "log.foamRun")
    stop = None
    if deadline is not None:
        from .transient import request_stop
        stop = (lambda: request_stop(case))
    rc = _step(foam, solve[0], solve[1], case, timeout, raise_on_error=False, deadline=deadline, on_deadline=stop)
    log = (case / "log.foamRun").read_text(errors="replace") if (case / "log.foamRun").is_file() else ""
    at = diverged(case / "log.foamRun")
    if at is not None:                         # it ended on its own (short run), but on NaN
        raise Diverged(divergence_message(at))
    out["log"] = foamio.parse_log(log)
    if "fatal" in out["log"]:
        raise FoamError("OpenFOAM stopped:\n" + out["log"]["fatal"])
    if rc != 0:
        raise FoamError(f"{solve[0]} exited with code {rc}:\n" + _tail(case / solve[1]))
    reached = True
    if transient:
        from .transient import latest_time, reset_stop
        reset_stop(case)
        m = re.search(r"endTime\s+([-+0-9.eE]+);", (case / "system" / "controlDict").read_text())
        t = latest_time(case)
        reached = bool(m and t is not None and t >= float(m.group(1)) * (1 - 1e-6))
        out["time"] = t
    out["reached_end"] = reached
    if procs > 1 and reached and not dec:
        _step(foam, "reconstructPar -latestTime", "log.reconstructPar", case, timeout)
    return out


class _Decomposed:
    """Fields of a case that stays decomposed (a sliding mesh, sliding.to_sliding in parallel), put together in the
    layout of the complete conformal mesh without reconstructPar: cells by the decomposition's cellProcAddressing
    (the topology change keeps every cell), each original patch's faces by faceProcAddressing and the patch's start
    before the change (it keeps them in order); both saved in processor*/cfdt_addressing."""

    def __init__(self, case: Path, mesh, procs: int):
        self.case, self.mesh, self.procs = Path(case), mesh, procs
        self.cells, self.patch_faces = [], []
        starts = {P["name"]: P["start"] for P in mesh.patches}
        for k in range(procs):
            d = self.case / f"processor{k}" / "cfdt_addressing"
            self.cells.append(foamio.read_labels(d / "cellProcAddressing"))
            fa = np.abs(foamio.read_labels(d / "faceProcAddressing")) - 1
            pf = {}
            for b in foamio.read_boundary((d / "boundary").read_text(errors="replace")):
                if b["name"] in starts and b["n"]:
                    pf[b["name"]] = fa[b["start"]:b["start"] + b["n"]] - starts[b["name"]]
            self.patch_faces.append(pf)

    def field(self, time_name: str, name: str):
        parts = []
        for k in range(self.procs):
            f = self.case / f"processor{k}" / time_name / name
            if not f.is_file():
                return None
            parts.append(foamio.read_foam(f.read_text()))
        cls = str((parts[0].get("FoamFile") or {}).get("class", ""))
        surface = cls.startswith("surface") or name == "phi"
        width = 3 if "Vector" in cls else 6 if "SymmTensor" in cls else 1
        nc = self.mesh.ncells
        out = {k: v for k, v in parts[0].items() if k not in ("internalField", "boundaryField")}

        def expand(a, n):
            a = np.asarray(a, float)
            if width == 1:
                return np.full(n, float(a)) if a.ndim == 0 else a
            return np.tile(a, (n, 1)) if a.ndim == 1 else a
        if not surface:
            G = np.zeros(nc) if width == 1 else np.zeros((nc, width))
            for k, part in enumerate(parts):
                G[self.cells[k]] = expand(part["internalField"], len(self.cells[k]))
            out["internalField"] = G
        else:
            out["internalField"] = 0.0
        bf = {}
        for P in self.mesh.patches:
            nm = P["name"]
            ent, vals, ok = None, None, True
            for k, part in enumerate(parts):
                idx = self.patch_faces[k].get(nm)
                if idx is None:
                    continue
                e = (part.get("boundaryField") or {}).get(nm)
                if e is None:
                    ok = False
                    continue
                ent = ent or {kk: vv for kk, vv in e.items() if kk != "value"}
                v = e.get("value")
                if v is None or isinstance(v, str):
                    ok = False
                    continue
                a = expand(v, len(idx))
                if vals is None:
                    vals = np.zeros(P["n"]) if width == 1 else np.zeros((P["n"], width))
                vals[idx] = a
            if ent is not None:
                if ok and vals is not None:
                    ent["value"] = vals
                bf[nm] = ent
        out["boundaryField"] = bf
        return out


DECOMPOSED_NOTE = "decomposePar and reconstructPar loop over every pair of the couple's faces"


def decomposed(case: Path) -> dict | None:
    """The case's record of staying decomposed (sliding.to_sliding in parallel), or None."""
    f = Path(case) / "cfdt_decomposed.json"
    if not f.is_file():
        return None
    import json
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def hwthreads() -> int:
    """Hardware threads this process may run on (a SLURM job's cpuset, or the machine)."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def physical_cores() -> int:
    """Physical cores among the CPUs this process may run on (a SLURM job's cpuset)."""
    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    cores = set()
    for c in cpus:
        try:
            t = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
            cores.add(((t / "physical_package_id").read_text().strip(), (t / "core_id").read_text().strip()))
        except OSError:
            cores.add(("?", str(c)))
    return len(cores)


def mpirun_command(procs: int) -> str:
    """CFDT_MPIRUN, or mpirun. Inside a SLURM job, OpenMPI 5 places ranks by core and refuses more ranks
    than cores ('allocation-overload'); when the job's CPUs are hardware threads of fewer cores, map by
    hardware thread instead."""
    if os.environ.get("CFDT_MPIRUN"):
        return os.environ["CFDT_MPIRUN"]
    if os.environ.get("SLURM_JOB_ID") and physical_cores() < procs:
        return "mpirun --map-by hwthread"
    return "mpirun"


def read_fields(case: Path, mesh, plan: dict, mean: bool = False) -> dict:
    """The last time directory's U, p, phi (and wallShearStress, nut, yPlus) in post.results' layout.
    Boundary values not written (zeroGradient, noSlip, slip, symmetry) are evaluated as OpenFOAM does.
    mean (transient runs): UMean and pMean (fieldAverage) stand for U and p when they were written."""
    case = Path(case)
    dec = decomposed(case)
    t = foamio.latest_time(case / "processor0" if dec else case)
    if t is None:
        raise FoamError("no results written (no time directory after 0)")
    if dec:
        joiner = _Decomposed(case, mesh, int(dec["procs"]))

        def load(name):
            return joiner.field(t.name, name)
    else:
        def load(name):
            p = t / name
            return foamio.read_foam(p.read_text()) if p.is_file() else None
    U, p, phi = load("U"), load("p"), load("phi")
    vof = plan.get("vof")
    alpha = p_abs = None
    if vof:
        # cavitation (incompressibleVoF): p and p_rgh in Pa, absolute (equal with g = 0); post reads p kinematic
        # and gauge, here to the inlet's target total pressure, with the liquid's density
        if p is None:
            p = load("p_rgh")
        if p is not None:
            p_abs = np.asarray(p["internalField"], float)
            p = cavitation_kinematic(p, vof)
        alpha = load("alpha.water")
    if U is None or p is None or phi is None:
        raise FoamError(f"{t.name}: U, p or phi missing")
    averaged = False
    if mean:
        Um, pm = load("UMean"), load("pMean")
        if Um is not None and pm is not None:
            U, p, averaged = Um, pm, True
    wss, nut, yp = load("wallShearStress"), load("nut"), load("yPlus")
    T, rho, whf, alphat = load("T"), load("rho"), load("wallHeatFlux"), load("alphat")
    nc = mesh.ncells

    def internal(f, width):
        v = f["internalField"]
        a = np.asarray(v, float)
        if a.ndim == 0 or (width == 3 and a.ndim == 1):
            a = np.tile(a, (nc, 1)) if width == 3 else np.full(nc, float(a))
        return a
    Uc, pc = internal(U, 3), internal(p, 1)
    ac = internal(alpha, 1) if alpha is not None else None
    if p_abs is not None and p_abs.ndim == 0:
        p_abs = np.full(nc, float(p_abs))
    Tc = internal(T, 1) if T else None
    rc = internal(rho, 1) if rho else None
    g = mesh.geometry()
    scal, scal_b = {}, {}
    for sn in (plan.get("scalars") or {}):
        sf = load(sn)
        if sf is not None:
            scal[sn] = (sf, internal(sf, 1))
    patches = {}
    notes = []
    for P in mesh.patches:
        nm, n = P["name"], P["n"]
        f = np.arange(P["start"], P["start"] + n)
        own = mesh.owner[f]
        if P["type"] == "empty":
            patches[nm] = {"U": None, "p": None, "phi": None}
            continue
        nrm = g["Sf"][f] / np.linalg.norm(g["Sf"][f], axis=1)[:, None]

        def bval(field, width, cells):
            e = field["boundaryField"].get(nm, {})
            typ = e.get("type", "")
            if "value" in e and not isinstance(e["value"], str):
                a = np.asarray(e["value"], float)
                if a.ndim == 0 or (width == 3 and a.ndim == 1):
                    a = np.tile(a, (n, 1)) if width == 3 else np.full(n, float(a))
                return a
            if cells is None:
                return np.zeros((n, 3)) if width == 3 else np.zeros(n)
            if typ == "noSlip":
                return np.zeros((n, 3))
            zn = plan["patches"].get(nm, {}).get("zone")
            if typ == "MRFnoSlip" and zn and width == 3:
                zp = plan["zones"][zn]
                d = (g["Cf"][f] - np.asarray(zp["center_mm"])) * 1e-3
                return np.cross(zp["omega_rad_s"] * np.asarray(zp["axis"]), d)
            c = cells[own]
            if typ in ("slip", "symmetry", "symmetryPlane") and width == 3:
                return c - np.einsum("ij,ij->i", c, nrm)[:, None] * nrm
            if typ not in ("zeroGradient", "slip", "symmetry", "symmetryPlane", "inletOutlet"):
                notes.append(f"{nm}: no value for {typ}; used the cell values")
            return c
        rec = {"U": bval(U, 3, Uc), "p": bval(p, 1, pc), "phi": bval(phi, 1, None)}
        if alpha is not None:
            rec["alpha"] = bval(alpha, 1, ac)
        rec["wss"] = bval(wss, 3, None) if (wss and P["type"] == "wall" and nm in wss["boundaryField"]) else None
        rec["nut"] = bval(nut, 1, None) if nut and nm in nut["boundaryField"] else None
        rec["yPlus"] = bval(yp, 1, None) if (yp and P["type"] == "wall" and nm in yp["boundaryField"]) else None
        if T:
            rec["T"] = bval(T, 1, Tc)
            rec["rho"] = bval(rho, 1, rc) if rho else None
            rec["whf"] = bval(whf, 1, None) if (whf and P["type"] == "wall" and nm in whf["boundaryField"]) else None
            rec["alphat"] = bval(alphat, 1, None) if alphat and nm in alphat["boundaryField"] else None
        patches[nm] = rec
        for sn, (sf, scc) in scal.items():          # tracers and age (scalars.py)
            scal_b.setdefault(sn, {})[nm] = bval(sf, 1, scc)
    out = {"U": Uc, "p": pc, "patches": patches, "time": t.name, "notes": notes, "source": "openfoam",
           "averaged": averaged}
    if mean and not averaged:
        notes.append("no UMean/pMean written (the run ended before average_from?): the last time's fields are used")
    if scal:
        out["scalars"] = {sn: {"internal": scc, "patches": scal_b.get(sn, {})} for sn, (sf, scc) in scal.items()}
    if T:
        out["T"], out["rho"] = Tc, rc
    if vof:
        out["p_abs"] = p_abs
        if ac is not None:
            out["alpha"] = ac
            # the vapour fraction as a scalar, so slices can show it (render: field 'vapour')
            out.setdefault("scalars", {})["vapour"] = {
                "internal": 1.0 - ac,
                "patches": {n: (1.0 - np.asarray(r["alpha"], float)) for n, r in patches.items() if r.get("alpha") is not None}}
    return out


def cavitation_kinematic(p: dict, vof: dict) -> dict:
    from .cavitation import to_kinematic
    return to_kinematic(p, float(vof["p_ref_Pa"]), float(vof["rho_l"]))
