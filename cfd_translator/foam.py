"""foam.py — the only module that runs OpenFOAM: find it, run checkMesh and foamRun, read the fields.
Version: 2026.10.18.2

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


def _bash(foam: dict, cmd: str, cwd: Path, timeout: float, capture: bool = False, cancel_file: Path | None = None):
    """Runs cmd in a bash that sources OpenFOAM. Long commands (not capture) run in their own process group,
    which a timeout or the cancel flag kills as a whole (mpirun and every rank, not only bash)."""
    src = f"source '{foam['bashrc']}' >/dev/null 2>&1; " if foam.get("bashrc") else ""
    if capture:
        return subprocess.run(["bash", "-c", src + cmd], cwd=str(cwd), timeout=timeout, capture_output=True, text=True)
    p = subprocess.Popen(["bash", "-c", src + cmd], cwd=str(cwd), start_new_session=True)
    t0 = time.time()
    while True:
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


def _step(foam: dict, cmd: str, logname: str, case: Path, timeout: float, raise_on_error: bool = True) -> int:
    try:
        rc = _bash(foam, f"{cmd} > {logname} 2>&1", case, timeout=timeout,
                   cancel_file=Path(case).parent / "cancel").returncode
    except subprocess.TimeoutExpired as exc:
        raise FoamError(f"{cmd.split()[0]} took more than {timeout:.0f} s (CFDT_TIMEOUT)") from exc
    if rc != 0 and raise_on_error:
        raise FoamError(f"{cmd} exited with code {rc}:\n" + _tail(case / logname))
    return rc


def run(case: Path, timeout: float = 3600, procs: int | None = None) -> dict:
    """checkMesh, then the solver. Returns the parsed logs; raises FoamError on failure."""
    foam = find_foam()
    if not foam or "error" in (foam or {}):
        raise FoamError((foam or {}).get("error", "OpenFOAM not found (install openfoam.org >= 11, or set CFDT_FOAM)"))
    case = Path(case)
    out = {"openfoam": foam.get("version")}
    for p in case.iterdir():                 # results of an earlier run must not be read as this one's
        try:
            if p.is_dir() and float(p.name) > 0:
                shutil.rmtree(p)
        except ValueError:
            pass
    _step(foam, "checkMesh", "log.checkMesh", case, min(timeout, 900), raise_on_error=False)
    out["checkMesh"] = foamio.parse_checkmesh((case / "log.checkMesh").read_text(errors="replace"))
    procs = int(procs or os.environ.get("CFDT_PROCS", "1"))
    out["procs"] = procs
    if procs > 1:
        (case / "system" / "decomposeParDict").write_text(
            "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      decomposeParDict;\n}\n\n"
            f"numberOfSubdomains {procs};\n\ndecomposer      scotch;\n")
        _step(foam, "decomposePar -force", "log.decomposePar", case, timeout)
        mpirun = os.environ.get("CFDT_MPIRUN", "mpirun")       # inside a SLURM job it uses the allocation
        steps = [(f"{mpirun} -np {procs} foamRun -parallel", "log.foamRun"), ("reconstructPar -latestTime", "log.reconstructPar")]
    else:
        steps = [("foamRun", "log.foamRun")]
    for i, (cmd, logname) in enumerate(steps):
        rc = _step(foam, cmd, logname, case, timeout, raise_on_error=False)
        if i == 0:
            log = (case / "log.foamRun").read_text(errors="replace") if (case / "log.foamRun").is_file() else ""
            out["log"] = foamio.parse_log(log)
            if "fatal" in out["log"]:
                raise FoamError("OpenFOAM stopped:\n" + out["log"]["fatal"])
        if rc != 0:
            raise FoamError(f"{cmd} exited with code {rc}:\n" + _tail(case / logname))
    return out


def read_fields(case: Path, mesh, plan: dict) -> dict:
    """The last time directory's U, p, phi (and wallShearStress, nut, yPlus) in post.results' layout.
    Boundary values not written (zeroGradient, noSlip, slip, symmetry) are evaluated as OpenFOAM does."""
    case = Path(case)
    t = foamio.latest_time(case)
    if t is None:
        raise FoamError("no results written (no time directory after 0)")

    def load(name):
        p = t / name
        return foamio.read_foam(p.read_text()) if p.is_file() else None
    U, p, phi = load("U"), load("p"), load("phi")
    if U is None or p is None or phi is None:
        raise FoamError(f"{t.name}: U, p or phi missing")
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
    Tc = internal(T, 1) if T else None
    rc = internal(rho, 1) if rho else None
    g = mesh.geometry()
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
        rec["wss"] = bval(wss, 3, None) if (wss and P["type"] == "wall" and nm in wss["boundaryField"]) else None
        rec["nut"] = bval(nut, 1, None) if nut and nm in nut["boundaryField"] else None
        rec["yPlus"] = bval(yp, 1, None) if (yp and P["type"] == "wall" and nm in yp["boundaryField"]) else None
        if T:
            rec["T"] = bval(T, 1, Tc)
            rec["rho"] = bval(rho, 1, rc) if rho else None
            rec["whf"] = bval(whf, 1, None) if (whf and P["type"] == "wall" and nm in whf["boundaryField"]) else None
            rec["alphat"] = bval(alphat, 1, None) if alphat and nm in alphat["boundaryField"] else None
        patches[nm] = rec
    out = {"U": Uc, "p": pc, "patches": patches, "time": t.name, "notes": notes, "source": "openfoam"}
    if T:
        out["T"], out["rho"] = Tc, rc
    return out
