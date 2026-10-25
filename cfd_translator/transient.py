"""transient.py — time-accurate runs: URANS (kOmegaSST), DES (kOmegaSSTDES) and laminar unsteady flow, with
time averages and runs that continue over several SLURM jobs.
Version: 2026.10.25.3

    settings(study, mesh) -> dict | None                 solve.transient with defaults filled in
    write_transient(case, study, plan)                   controlDict, fvSchemes, fvSolution, momentumTransport
    after_init(case)                                     the steady start's solution becomes time 0
    progress(log_text, end_time) -> dict                 simulated time, time step, Courant number

solve.transient.continue = '<request>/<study>': a later end_time for a finished run (its mesh, case and latest time
are copied; averages go on, or start again when average_from is later than the old one).

A transient study first runs `init` steady iterations (RANS kOmegaSST for both URANS and DES; none for laminar
flow), whose solution is the initial field (as the OpenFOAM motorBike DES tutorial does). Then PIMPLE with an
adjustable time step (maxCo), backward in time, LUST for DES (linearUpwind otherwise). A fieldAverage function
object averages U and p from average_from on (UMean, pMean, UPrime2Mean); results come from the mean fields, and
forces and pressures are averaged in time from the per-step monitors (average.py).

Long runs: each SLURM job has run.timeout_s (or CFDT_TIMEOUT). Before it runs out, controlDict gets
`stopAt writeNow` (runTimeModifiable), OpenFOAM writes and stops, and the job submits its successor (dependent on
it), which continues from the latest time (startFrom latestTime; the averages continue from uniform/). At most
max_jobs jobs.

OpenFOAM 14 (read in its source): no kOmegaSST DDES/IDDES (only kOmegaSSTDES, whose F1/F2 zonal shielding,
FSST 2, protects the boundary layer); residualControl would end a transient run as soon as the residuals are
small, so PIMPLE has none.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import numpy as np

SCALE = 1e-3


def _velocity(study: dict) -> float:
    ref = study.get("reference") or {}
    if ref.get("velocity"):
        return float(ref["velocity"])
    best = 0.0
    for b in study.get("bcs", []):
        v = b.get("velocity") if b.get("kind") == "inlet" else None
        if isinstance(v, (int, float)):
            best = max(best, abs(float(v)))
        elif isinstance(v, list) and len(v) == 3:
            best = max(best, float(np.linalg.norm(v)))
        if b.get("kind") == "wall" and isinstance(b.get("velocity"), list):
            best = max(best, float(np.linalg.norm(b["velocity"])))
    return best or 1.0


def settings(study: dict, mesh=None) -> dict | None:
    tr = (study.get("solve") or {}).get("transient")
    if not tr:
        return None
    T = float(tr["end_time"])
    model = study.get("model", "laminar")
    U = _velocity(study)
    W = float(tr.get("write_interval", T / 10))
    if tr.get("dt"):
        dt = float(tr["dt"])
    else:
        h = 1.0
        if mesh is not None:
            V = mesh.geometry()["V"]
            h = float(np.cbrt(max(float(np.min(V)), 1e-30)))       # mm
        dt = min(0.2 * h * SCALE / U, W / 20)
    return {"end_time": T, "average_from": float(tr.get("average_from", T / 3)), "max_co": float(tr.get("max_co", 1.0)),
            "write_interval": W, "init": int(tr.get("init", 0 if model == "laminar" else 500)),
            "max_jobs": int(tr.get("max_jobs", 10)), "dt": float(f"{dt:.4g}"), "model": model, "velocity": U,
            "length_m": float((study.get("reference") or {}).get("length", 0)) * SCALE or None,
            "max_dt": float(f"{min(W / 10, T / 500):.4g}")}


def _dict(obj: str, location: str, body: str) -> str:
    from .case import _dict as d
    return d(obj, location, body)


def momentum_transport(model: str) -> str:
    if model == "kOmegaSSTDES":
        return ("simulationType  LES;\n\nLES\n{\n    model           kOmegaSSTDES;\n\n    turbulence      on;\n\n"
                "    delta           cubeRootVol;\n\n    cubeRootVol\n    {\n        deltaCoeff      1;\n    }\n}")
    if model == "laminar":
        return "simulationType  laminar;"
    return "simulationType  RAS;\n\nRAS\n{\n    model           kOmegaSST;\n\n    turbulence      on;\n}"


def write_transient(case: Path, study: dict, plan: dict) -> None:
    """The system files and momentumTransport of the transient run (the 0/ fields stay)."""
    case = Path(case)
    tr = plan["transient"]
    model = tr["model"]
    turb = model != "laminar"
    (case / "constant" / "momentumTransport").write_text(_dict("momentumTransport", "constant", momentum_transport(model)))
    funcs = []
    for f in plan.get("functions", []):
        if re.match(r"\s*(wallShearStress|yPlus)\b", f):         # fields: only when written
            f = f.replace("writeControl    writeTime;", "executeControl  writeTime;\n        writeControl    writeTime;")
        funcs.append(f)
    funcs.append("    cfdtAverage\n    {\n        type            fieldAverage;\n"
                 "        libs            (\"libfieldFunctionObjects.so\");\n        writeControl    writeTime;\n"
                 f"        startTime       {tr['average_from']:.8g};\n        restartOnRestart false;\n"
                 "        fields\n        (\n            U { mean on; prime2Mean on; }\n            p { mean on; prime2Mean off; }\n"
                 "        );\n    }")
    cd = (f"solver          incompressibleFluid;\n\nstartFrom       latestTime;\n\nstartTime       0;\n\n"
          f"stopAt          endTime;\n\nendTime         {tr['end_time']:.8g};\n\ndeltaT          {tr['dt']:.6g};\n\n"
          f"adjustTimeStep  yes;\n\nmaxCo           {tr['max_co']:g};\n\nmaxDeltaT       {tr['max_dt']:.6g};\n\n"
          f"writeControl    adjustableRunTime;\n\nwriteInterval   {tr['write_interval']:.8g};\n\npurgeWrite      2;\n\n"
          "writeFormat     ascii;\n\nwritePrecision  10;\n\nwriteCompression off;\n\ntimeFormat      general;\n\n"
          "timePrecision   8;\n\nrunTimeModifiable true;\n\nfunctions\n{\n" + "\n".join(funcs) + "\n}")
    (case / "system" / "controlDict").write_text(_dict("controlDict", "system", cd))
    divU = "Gauss LUST grad(U)" if model == "kOmegaSSTDES" else "Gauss linearUpwind grad(U)"
    fs = (f"ddtSchemes\n{{\n    default         backward;\n}}\n\ngradSchemes\n{{\n    {plan['grad']}\n}}\n\n"
          "divSchemes\n{\n    default         Gauss linear;\n"
          f"    div(phi,U)      {divU};\n"
          "    div(phi,k)      Gauss limitedLinear 1;\n    div(phi,omega)  Gauss limitedLinear 1;\n"
          "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
          f"laplacianSchemes\n{{\n    default         {plan['laplacian']};\n}}\n\n"
          "interpolationSchemes\n{\n    default         linear;\n}\n\n"
          f"snGradSchemes\n{{\n    default         {plan['sngrad']};\n}}\n\nwallDist\n{{\n    method          meshWave;\n}}")
    (case / "system" / "fvSchemes").write_text(_dict("fvSchemes", "system", fs))
    fields = "U|k|omega" if turb else "U"
    fv = ("solvers\n{\n    p\n    {\n        solver          GAMG;\n        tolerance       1e-06;\n"
          "        relTol          0.05;\n        smoother        GaussSeidel;\n    }\n\n"
          "    pFinal\n    {\n        $p;\n        relTol          0;\n    }\n\n"
          "    pcorr\n    {\n        $p;\n        relTol          0;\n    }\n\n"
          f"    \"({fields})\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-07;\n        relTol          0.1;\n    }\n\n"
          f"    \"({fields})Final\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-07;\n        relTol          0;\n    }\n}\n\n"
          f"PIMPLE\n{{\n    nOuterCorrectors 1;\n    nCorrectors     2;\n    nNonOrthogonalCorrectors {plan['nonorth']};\n"
          "    momentumPredictor yes;\n    pRefCell        0;\n    pRefValue       0;\n}\n\n"
          "relaxationFactors\n{\n    equations\n    {\n        \".*\"            1;\n    }\n}")
    (case / "system" / "fvSolution").write_text(_dict("fvSolution", "system", fv))


def _times(d: Path) -> list[tuple[float, Path]]:
    out = []
    for p in Path(d).iterdir():
        if p.is_dir():
            try:
                out.append((float(p.name), p))
            except ValueError:
                pass
    return sorted(out)


def latest_time(case: Path) -> float | None:
    """The latest time written (processor0's when the case is decomposed)."""
    case = Path(case)
    d = case / "processor0" if (case / "processor0").is_dir() else case
    t = _times(d)
    return t[-1][0] if t else None


def after_init(case: Path) -> float | None:
    """The steady start's last iteration becomes time 0 (root and processor directories; uniform/ removed, so
    the time index restarts), its monitors are dropped. Returns the steady iteration count."""
    case = Path(case)
    n = None
    for d in [case] + sorted(case.glob("processor*")):
        t = _times(d)
        if not t or t[-1][0] <= 0:
            continue
        n = t[-1][0]
        z = d / "0"
        if z.exists():
            shutil.rmtree(z)
        t[-1][1].rename(z)
        shutil.rmtree(z / "uniform", ignore_errors=True)
        for tt, p in t[:-1]:
            if tt > 0:
                shutil.rmtree(p, ignore_errors=True)
    shutil.rmtree(case / "postProcessing", ignore_errors=True)
    for lg in ("log.foamRun", "log.decomposePar", "log.reconstructPar"):
        if (case / lg).exists():
            (case / lg).rename(case / f"{lg}.init")
    return n


_TIME = re.compile(r"^Time = ([-+0-9.eE]+)s?\s*$", re.M)
_CO = re.compile(r"Courant Number mean: ([-+0-9.eE]+) max: ([-+0-9.eE]+)")
_DT = re.compile(r"^deltaT = ([-+0-9.eE]+)", re.M)


_CLOCK = re.compile(r"ClockTime = ([-+0-9.eE]+) s")


def progress(text: str, end_time: float) -> dict:
    """From the end of this job's log: simulated time, Courant number, time step, and the job's wall time so far
    (OpenFOAM's ClockTime: the log file's own times change with every write)."""
    t = _TIME.findall(text)
    out = {}
    ck = _CLOCK.findall(text[-20000:])
    if ck:
        out["job_clock_s"] = float(ck[-1])
    if t:
        out["time"] = float(t[-1])
        out["end_time"] = end_time
    co = _CO.findall(text[-20000:])
    if co:
        out["courant_max"] = float(co[-1][1])
    dt = _DT.findall(text[-20000:])
    if dt:
        out["dt"] = float(dt[-1])
    return out


def source_folder(ref: str) -> Path:
    """The study folder that solve.transient.continue names ('<request>/<study>', under CFDT_OUT or absolute)."""
    from .api import out_root
    p = Path(ref).expanduser()
    return p if p.is_absolute() else out_root() / p


def continue_problems(study: dict) -> list[str]:
    """Why solve.transient.continue cannot be used here ([] when it can): the folder must hold a prepared transient
    run of this machine with a time before the new end_time."""
    tr = (study.get("solve") or {}).get("transient") or {}
    ref = tr.get("continue")
    if not ref:
        return []
    src = source_folder(ref)
    if not (src / "prepared.pkl").is_file() or not (src / "transient.json").is_file():
        return [f"solve.transient.continue: no transient run in {src} on this machine"]
    t = latest_time(src / "case") if (src / "case").is_dir() else None
    if t is None:
        return [f"solve.transient.continue: {src} has no written time to continue from"]
    if float(tr["end_time"]) <= t * (1 + 1e-9):
        return [f"solve.transient.continue: end_time must be after the run's latest time ({t:g} s)"]
    return []


def copy_case(src: Path, dst: Path) -> float | None:
    """The case without its older time directories (root and processor*): what continuing needs. Returns the latest
    time."""
    src, dst = Path(src), Path(dst)
    t = latest_time(src)

    def ignore(d, names):
        d = Path(d)
        if not (d == src or (d.parent == src and d.name.startswith("processor"))):
            return []                         # postProcessing/<fo>/<start time> and the rest are copied whole
        out = []
        for n in names:
            try:
                v = float(n)
            except ValueError:
                continue
            if t is not None and v < t * (1 - 1e-12):
                out.append(n)
        return out
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True, ignore=ignore)
    for lg in list(dst.glob("log.foamRun*")):
        if not lg.name.endswith(".previous"):
            lg.rename(dst / f"{lg.name}.previous")
    return t


def restart_averages(case: Path) -> None:
    """Drop the time averages in the latest time (root and processors): fieldAverage starts again at its startTime."""
    case = Path(case)
    for d in [case] + sorted(case.glob("processor*")):
        t = _times(d)
        if not t:
            continue
        last = t[-1][1]
        for f in ("UMean", "pMean", "UPrime2Mean"):
            (last / f).unlink(missing_ok=True)
        for f in (last / "uniform").glob("*Properties") if (last / "uniform").is_dir() else []:
            if f.name.startswith("cfdtAverage"):
                f.unlink()


def request_stop(case: Path) -> bool:
    """controlDict stopAt writeNow: OpenFOAM writes the current time and stops (runTimeModifiable)."""
    f = Path(case) / "system" / "controlDict"
    txt = f.read_text()
    new, n = re.subn(r"stopAt\s+\w+;", "stopAt          writeNow;", txt)
    if n:
        f.write_text(new)
        # the file's time stamp must change visibly for OpenFOAM's timeStamp check
        now = time.time()
        import os
        os.utime(f, (now, now))
    return bool(n)


def reset_stop(case: Path) -> None:
    f = Path(case) / "system" / "controlDict"
    txt = f.read_text()
    f.write_text(re.sub(r"stopAt\s+\w+;", "stopAt          endTime;", txt))
