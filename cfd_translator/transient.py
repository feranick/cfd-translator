"""transient.py — time-accurate runs: URANS (kOmegaSST), DES (kOmegaSSTDES) and laminar unsteady flow, with
time averages and runs that continue over several SLURM jobs.
Version: 2026.11.12.11

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
    # max_co null: a fixed time step (dt, or a sliding mesh's degrees_per_step), no Courant limit: snappyHexMesh leaves
    # a few micrometre cells in narrow gaps, and a Courant limit of 2 took the S50 to 0.02 deg a step (150,000 steps
    # for 8 revolutions); the hand-built v2506 run went at 1 deg a step with Courant ~25 in a handful of cells,
    # mean 0.12, and stayed stable (Perlmutter, 2026.11)
    fixed = "max_co" in tr and tr["max_co"] is None
    out = {"end_time": T, "average_from": float(tr.get("average_from", T / 3)),
           "max_co": None if fixed else float(tr.get("max_co", 1.0)), "fixed_step": fixed,
           "write_interval": W,
           # a run started from another run's fields (init_from) needs no steady start
           "init": int(tr.get("init", 0 if (model == "laminar" or init_ref(study)[0]) else 500)),
           "max_jobs": int(tr.get("max_jobs", 10)), "dt": float(f"{dt:.4g}"), "model": model, "velocity": U,
           "length_m": float((study.get("reference") or {}).get("length", 0)) * SCALE or None,
           "max_dt": float(f"{min(W / 10, T / 500):.4g}"),
           **({"init_from": init_ref(study)[0], "init_fields": init_ref(study)[1]} if init_ref(study)[0] else {}),
           **({"p_below": [float(v) for v in tr["p_below"]], "p_below_regions": list(tr.get("p_below_regions") or []),
               "p_operating": float((study.get("conditions") or {}).get("p", 101325.0))} if tr.get("p_below") else {})}
    if fixed and tr.get("dt"):
        out["dt"] = out["max_dt"] = float(tr["dt"])
    sl = sliding_zone(study)
    if sl:
        # a sliding mesh: at most degrees_per_step of rotation per time step, Courant 2 (OpenFOAM's propeller
        # tutorial), averages over whole revolutions (the window ends at end_time)
        Trev = 60.0 / abs(float(sl["rpm"]))
        deg = float(tr.get("degrees_per_step", 1.0))
        if fixed:                              # exactly degrees_per_step a step
            out["dt"] = out["max_dt"] = float(f"{deg / 360.0 * Trev:.6g}")
        else:
            out["max_co"] = float(tr.get("max_co", 2.0))
            out["max_dt"] = float(f"{min(out['max_dt'], deg / 360.0 * Trev):.4g}")
            out["dt"] = min(out["dt"], out["max_dt"])
        n = int(np.floor((T - out["average_from"]) / Trev + 1e-9))
        if n >= 1:
            out["average_from"] = float(f"{T - n * Trev:.10g}")
        out.update(period=Trev, revolutions=float(f"{T / Trev:.4g}"), average_revolutions=max(n, 0),
                   degrees_per_step=deg, sliding=sl["name"], correct_mesh_phi=bool(tr.get("correct_mesh_phi", True)))
    return out


def p_below_function(tr: dict, rho: float) -> str:
    """A coded function object (OpenFOAM 14: type coded, libutilityFunctionObjects; codeRead gets `dict`, the code
    mesh()) adding, every time step after average_from, the step's length to pBelowTime_<Pa> in each cell whose
    absolute pressure, rho p + p_operating (conditions.p), is below the threshold. The fields are registered
    AUTO_WRITE, so they are written with the run, and read back when a later SLURM job continues it. OpenFOAM has no
    function object that thresholds a field; the code is compiled at the run's start (wmake, allowSystemOperations)."""
    from .foam import p_below_field
    thr = [float(v) for v in tr["p_below"]]
    names = " ".join(p_below_field(v) for v in thr)
    return ("    cfdtPBelow\n    {\n        type            coded;\n"
            "        libs            (\"libutilityFunctionObjects.so\");\n"
            f"        thresholds      ({' '.join(f'{v:.10g}' for v in thr)});\n        names           ({names});\n"
            f"        averageFrom     {tr['average_from']:.10g};\n        pOperating      {tr['p_operating']:.10g};\n"
            f"        rho             {rho:.10g};\n\n"
            "        codeInclude\n        #{\n            #include \"volFields.H\"\n        #};\n\n"
            "        codeData\n        #{\n            scalarList thr_;\n            wordList names_;\n"
            "            scalar t0_ = 0;\n            scalar pOp_ = 0;\n            scalar rho_ = 1;\n"
            "            PtrList<volScalarField> below_;\n        #};\n\n"
            "        codeRead\n        #{\n"
            "            thr_ = dict.lookup<scalarList>(\"thresholds\");\n"
            "            names_ = dict.lookup<wordList>(\"names\");\n"
            "            t0_ = dict.lookup<scalar>(\"averageFrom\");\n"
            "            pOp_ = dict.lookup<scalar>(\"pOperating\");\n"
            "            rho_ = dict.lookup<scalar>(\"rho\");\n"
            "            if (below_.empty())\n            {\n                below_.setSize(names_.size());\n"
            "                forAll(names_, i)\n                {\n"
            "                    typeIOobject<volScalarField> io\n                    (\n"
            "                        names_[i], mesh().time().name(), mesh(),\n"
            "                        IOobject::MUST_READ, IOobject::AUTO_WRITE\n                    );\n"
            "                    if (io.headerOk())\n                    {\n"
            "                        below_.set(i, new volScalarField(io, mesh()));\n                    }\n"
            "                    else\n                    {\n"
            "                        below_.set(i, new volScalarField(IOobject(names_[i], mesh().time().name(), "
            "mesh(), IOobject::NO_READ, IOobject::AUTO_WRITE), mesh(), dimensionedScalar(dimTime, 0)));\n"
            "                    }\n                }\n            }\n        #};\n\n"
            "        codeExecute\n        #{\n            const scalar t = mesh().time().value();\n"
            "            if (t > t0_)\n            {\n                const scalar dt = mesh().time().deltaTValue();\n"
            "                const volScalarField& p = mesh().lookupObject<volScalarField>(\"p\");\n"
            "                const scalarField pa(rho_*p.primitiveField() + pOp_);\n"
            "                forAll(below_, i)\n                {\n"
            "                    below_[i].primitiveFieldRef() += dt*pos0(thr_[i] - pa);\n"
            "                }\n            }\n        #};\n    }")


def _region_mask(reg: dict, C: np.ndarray) -> np.ndarray:
    """Cells (centres C, mm) in a p_below region: {box: [x0, y0, z0, x1, y1, z1]} or {cylinder: {center, axis, radius,
    span, inner_radius}}."""
    if "box" in reg:
        b = np.asarray(reg["box"], float)
        return np.all((C >= b[:3]) & (C <= b[3:]), axis=1)
    c = reg["cylinder"]
    a = np.asarray(c["axis"], float)
    a = a / np.linalg.norm(a)
    d = C - np.asarray(c.get("center", [0, 0, 0]), float)
    s = d @ a
    r = np.linalg.norm(d - np.outer(s, a), axis=1)
    m = (r <= float(c["radius"])) & (r >= float(c.get("inner_radius", 0.0)))
    if "span" in c:
        m &= (s >= float(c["span"][0])) & (s <= float(c["span"][1]))
    return m


def pressure_statistics(study: dict, mesh, plan: dict, fields: dict) -> dict | None:
    """The pressure's statistics over a transient run's averaging window, from the fields read (foam.read_fields):
    p rms (pPrime2Mean) and, per p_below threshold (Pa absolute = conditions.p + the gauge pressure), the fraction of
    the window each cell spent below it: for the whole domain and each p_below_regions region, the volume-time below
    (m3 s), the volume-weighted mean fraction, the largest, and the volume below for over 10 % of the time."""
    tr = plan.get("transient") or {}
    sc = fields.get("scalars") or {}
    V = mesh.geometry()["V"] * 1e-9
    C = mesh.geometry()["C"]
    out = {}
    if "p_rms" in sc:
        pr = np.asarray(sc["p_rms"]["internal"], float)
        k = int(np.argmax(pr))
        out["p_rms_Pa"] = {"max": float(f"{pr[k]:.5g}"), "at_mm": np.round(C[k], 1).tolist(),
                           "mean": float(f"{(pr * V).sum() / max(V.sum(), 1e-300):.5g}")}
    if tr.get("p_below"):
        from .foam import p_below_key
        span = fields.get("p_stats_window_s")
        regions = [{"name": "domain"}] + list(tr.get("p_below_regions") or [])
        below = {}
        for thr in tr["p_below"]:
            key = f"p_below_{p_below_key(thr)}"
            if key not in sc:
                continue
            fr = np.asarray(sc[key]["internal"], float)
            per = {}
            for reg in regions:
                m = np.ones(len(V), bool) if reg["name"] == "domain" else _region_mask(reg, C)
                if not m.any():
                    per[reg["name"]] = {"cells": 0}
                    continue
                vm = V[m]
                per[reg["name"]] = {"volume_time_m3s": float(f"{(fr[m] * vm).sum() * (span or 0):.4g}"),
                                    "mean_fraction": float(f"{(fr[m] * vm).sum() / vm.sum():.4g}"),
                                    "max_fraction": float(f"{fr[m].max():.4g}"),
                                    "volume_over_10pct_mm3": float(f"{vm[fr[m] > 0.1].sum() * 1e9:.4g}")}
            below[f"{thr:g}"] = per
        if below:
            out["below"] = below
            out["window_s"] = span
            out["p_absolute"] = (f"conditions.p ({tr['p_operating']:g} Pa) + the gauge pressure: set conditions.p so "
                                 "the gauge levels (e.g. the outlet's pressure) give the real absolute pressure")
        elif tr["p_below"]:
            out["below_note"] = "no pBelowTime fields were written (the run ended before average_from, or an older run)"
    return out or None


def courant_stats(log: Path, tail_mb: float = 8.0) -> dict | None:
    """The Courant numbers OpenFOAM printed at the end of a transient log (its last tail_mb): the last step's mean and
    max, the largest max and the median of the maxima over those steps. A fixed time step (max_co null) runs with a
    large maximum in a few cells: these say how large and how steady it is."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, Path(log).stat().st_size - int(tail_mb * 1024 * 1024)))
            text = f.read().decode(errors="replace")
    except OSError:
        return None
    co = [(float(a), float(b)) for a, b in _CO.findall(text)]
    if not co:
        return None
    mx = sorted(b for _, b in co)
    return {"mean_last": co[-1][0], "max_last": co[-1][1], "max": mx[-1], "max_median": mx[len(mx) // 2],
            "steps_seen": len(co)}


_SOLVE = re.compile(r"(\w+):\s+Solving for (\w+), Initial residual = [^,]+, Final residual = [^,]+, No Iterations (\d+)")
_PVF = re.compile(r"projected volume fraction = ([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+)")


def step_profile(text: str, last: int = 8) -> dict | None:
    """Where a transient run's time steps go, from its log (the last `last` whole steps): wall seconds per step
    (ClockTime), each solved field's linear-solver iterations per step (all outer correctors; 'Ux' and 'UxFinal'
    together) and its solver, and on a sliding mesh the stitcher's projected volume fraction (max per step).
    OpenFOAM.org prints no timings per part: the iteration counts say which solve dominates (Perlmutter sliding7,
    2026.11.12: 42-55 s a 1-degree step, U's final smoothSolver at 33-118 sweeps)."""
    marks = [m.start() for m in _TIME.finditer(text)]
    if len(marks) < 2:
        return None
    allb = [text[marks[i]:marks[i + 1]] for i in range(len(marks) - 1)]     # the last one may still be running
    blocks = allb[-last:]
    its, solver, pvf = {}, {}, []
    clocks = []
    for b in allb[-(last + 1):]:
        ck = _CLOCK.findall(b)
        clocks.append(float(ck[-1]) if ck else None)
    secs = [b - a for a, b in zip(clocks, clocks[1:]) if a is not None and b is not None and b >= a]
    for b in blocks:
        for sv, field, n in _SOLVE.findall(b):
            its[field] = its.get(field, 0) + int(n)
            solver[field] = sv
        pv = _PVF.findall(b)
        if pv:
            pvf.append(max(float(x[2]) for x in pv))
    nb = len(blocks)
    out = {"steps": nb,
           "iterations_per_step": {f: round(v / nb, 1) for f, v in sorted(its.items(), key=lambda kv: -kv[1])},
           "solvers": solver}
    if secs:
        ss = sorted(secs)
        out["s_per_step"] = round(ss[len(ss) // 2], 2)
        out["s_per_step_max"] = round(ss[-1], 2)
    if pvf:
        out["projected_volume_fraction_max"] = round(max(pvf), 3)
        out["projected_volume_fraction_last"] = round(pvf[-1], 3)
    return out


def stitcher_stats(log: Path, tail_mb: float = 8.0) -> dict | None:
    """The stitcher's projected volume fraction over the end of a sliding run's log: max, last, steps over 0.3."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, Path(log).stat().st_size - int(tail_mb * 1024 * 1024)))
            text = f.read().decode(errors="replace")
    except OSError:
        return None
    v = [float(m[2]) for m in _PVF.findall(text)]
    if not v:
        return None
    return {"projected_volume_fraction_max": round(max(v), 3), "projected_volume_fraction_last": round(v[-1], 3),
            "over_0.3": sum(x > 0.3 for x in v), "seen": len(v)}


def sliding_zone(study: dict) -> dict | None:
    """The study's sliding zone (rotating[].motion 'sliding'), or None."""
    for z in study.get("rotating") or []:
        if isinstance(z, dict) and z.get("motion") == "sliding":
            return z
    return None


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
                 "        fields\n        (\n            U { mean on; prime2Mean on; }\n            p { mean on; prime2Mean on; }\n"
                 "        );\n    }")
    if tr.get("p_below"):
        funcs.append(p_below_function(tr, float(plan.get("rho") or 1.0)))
    cd = (f"solver          incompressibleFluid;\n\nstartFrom       latestTime;\n\nstartTime       0;\n\n"
          f"stopAt          endTime;\n\nendTime         {tr['end_time']:.8g};\n\ndeltaT          {tr['dt']:.6g};\n\n"
          + ("adjustTimeStep  no;\n\nwriteControl    runTime;\n\n" if tr.get("fixed_step") else
             f"adjustTimeStep  yes;\n\nmaxCo           {tr['max_co']:g};\n\nmaxDeltaT       {tr['max_dt']:.6g};\n\n"
             f"writeControl    adjustableRunTime;\n\n") +
          f"writeInterval   {tr['write_interval']:.8g};\n\npurgeWrite      2;\n\n"
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
    slide = bool(tr.get("sliding"))
    # a sliding mesh (OpenFOAM 14 propeller tutorial; cfd.direct 'Using NCC'): the fluxes are corrected after
    # the mesh moves (correctPhi: pcorr) and the mesh fluxes after the couple reconnects (correctMeshPhi: MeshPhi)
    pc = ("    \"pcorr.*\"\n    {\n        $p;\n        tolerance       1e-02;\n        relTol          0;\n    }\n\n"
          "    MeshPhi\n    {\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-02;\n        relTol          0;\n    }\n\n") if slide else \
         "    pcorr\n    {\n        $p;\n        relTol          0;\n    }\n\n"
    fv = ("solvers\n{\n    p\n    {\n        solver          GAMG;\n        tolerance       1e-06;\n"
          "        relTol          0.05;\n        smoother        GaussSeidel;\n    }\n\n"
          "    pFinal\n    {\n        $p;\n        relTol          0;\n    }\n\n" + pc +
          f"    \"({fields})\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-07;\n        relTol          0.1;\n    }\n\n"
          f"    \"({fields})Final\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          # a sliding mesh runs two outer correctors: the last one's momentum solve need not go to 1e-7 (Perlmutter
          # sliding7, 13.1 M cells: 33-118 sweeps a step, 42-55 s a 1-degree step against 5.1 s for a hand-built
          # case on the same cores); a 100-fold drop within the step, or 1e-6
          + ("        tolerance       1e-06;\n        relTol          0.01;\n        maxIter         50;\n    }\n}\n\n"
             if slide else "        tolerance       1e-07;\n        relTol          0;\n    }\n}\n\n") +
          ("PIMPLE\n{\n    correctPhi      yes;\n    correctMeshPhi  " +
           ("yes" if tr.get("correct_mesh_phi", True) else "no") + ";\n    nOuterCorrectors 2;\n    nCorrectors     1;\n"
           if slide else "PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors     2;\n") +
          f"    nNonOrthogonalCorrectors {plan['nonorth']};\n"
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
        out["courant_mean"] = float(co[-1][0])
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


INIT_FIELDS = ("U", "p", "p_rgh", "alpha.water", "k", "omega", "nut", "nuTilda")
MESH_FILES = ("points", "faces", "owner", "neighbour")


def init_ref(study: dict) -> tuple[str | None, str]:
    """(the run a study starts from, which fields): solve.init_from (steady or transient) or
    solve.transient.init_from; init_fields 'last' (default) or 'mean'."""
    s = study.get("solve") or {}
    tr = s.get("transient") or {}
    ref = tr.get("init_from") or s.get("init_from")
    return ref, (tr.get("init_fields") or s.get("init_fields") or "last")


def init_source(ref: str, request: Path | None = None) -> Path:
    """The study folder init_from names: '<request>/<study>' (under CFDT_OUT, or absolute), or '<study>' of the same
    request."""
    if "/" not in ref and request is not None:
        return Path(request) / ref
    return source_folder(ref)


def source_times(src: Path) -> list:
    case = Path(src) / "case"
    return [x for x in _times(case) if x[0] > 0] if case.is_dir() else []


def init_problems(study: dict, names=None, name: str | None = None) -> list[str]:
    """Why init_from cannot be used ([] when it can). '<study>' must be another study of the same request (names;
    it runs first and is checked when this one starts); '<request>/<study>' a finished run of this machine whose case
    has a written time in the case directory itself (reconstructed)."""
    ref, which = init_ref(study)
    if not ref:
        return []
    s = study.get("solve") or {}
    key = "solve.transient.init_from" if (s.get("transient") or {}).get("init_from") else "solve.init_from"
    if (s.get("transient") or {}).get("continue"):
        return ["solve.transient: init_from and continue do not go together (continue carries on the same mesh)"]
    if "/" not in ref:
        if name is not None and ref == name:
            return [f"{key}: a study cannot start from itself"]
        if names is not None and ref not in names:
            return [f"{key}: {ref!r} is not a study of this request (give '<request>/<study>' for an earlier run)"]
        return []
    src = source_folder(ref)
    if not (src / "case").is_dir():
        return [f"{key}: no case in {src} on this machine"]
    t = source_times(src)
    if not t:
        return [f"{key}: {src}/case has no reconstructed time directory after 0 (a sliding-mesh run stays decomposed "
                "and cannot be mapped yet)"]
    if which == "mean" and not (t[-1][1] / "UMean").is_file():
        return [f"{key}: {src}'s latest time ({t[-1][0]:g} s) has no mean fields (UMean); use init_fields 'last'"]
    return []


def _same_mesh(a: Path, b: Path) -> bool:
    """Do two cases have the same polyMesh (byte for byte: a sweep's points meshed from the same input)?"""
    import hashlib
    pa, pb = Path(a) / "constant" / "polyMesh", Path(b) / "constant" / "polyMesh"
    try:
        if any((pa / f).stat().st_size != (pb / f).stat().st_size for f in MESH_FILES):
            return False
        for f in MESH_FILES:
            h = []
            for p in (pa / f, pb / f):
                d = hashlib.sha1()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        d.update(chunk)
                h.append(d.hexdigest())
            if h[0] != h[1]:
                return False
        return True
    except OSError:
        return False


def _splice_internal(src_text: str, dst_text: str) -> str:
    """dst's field file with src's internalField (its boundaryField, dimensions and header kept)."""
    i_s, j_s = src_text.index("\ninternalField"), src_text.index("\nboundaryField")
    i_d, j_d = dst_text.index("\ninternalField"), dst_text.index("\nboundaryField")
    return dst_text[:i_d] + src_text[i_s:j_s] + dst_text[j_d:]


def map_from(case: Path, src: Path | str, which: str, run_step) -> tuple[list[str], dict]:
    """init_from: the source run's U, p, k, omega, nut (and p_rgh, alpha.water between cavitation runs) from its
    latest time ('mean': UMean and pMean for U and p) as this case's start in 0/. On the same mesh (a sweep's
    points) the internal fields are copied; otherwise mapFields maps them (internal fields by cell overlap; every
    patch keeps this case's conditions; cells outside the source's mesh keep their uniform start). Pressure moves
    only between runs of the same kind (single-phase p is kinematic, cavitation p and p_rgh absolute in Pa).
    run_step(cmd, log) runs an OpenFOAM command in case. Returns (notes, {"fields", "time", "same_mesh", "vof"})."""
    case = Path(case)
    src = source_folder(src) if isinstance(src, str) else Path(src)
    src_case = src / "case"
    t, tdir = source_times(src)[-1]
    src_vof, dst_vof = (tdir / "alpha.water").is_file(), (case / "0" / "alpha.water").is_file()
    pairs = []
    for f in INIT_FIELDS:
        srcf = tdir / f
        if which == "mean" and f in ("U", "p") and (tdir / f"{f}Mean").is_file():
            srcf = tdir / f"{f}Mean"
        if f in ("p", "p_rgh") and src_vof != dst_vof:
            continue
        if srcf.is_file() and (case / "0" / f).is_file():
            pairs.append((f, srcf))
    if not pairs:
        raise RuntimeError(f"init_from: none of {INIT_FIELDS} in {tdir} matches this case's fields")
    took = [f + (" (mean)" if srcf.name != f else "") for f, srcf in pairs]
    same = _same_mesh(src_case, case)
    if same:
        for f, srcf in pairs:
            dst = case / "0" / f
            dst.write_text(_splice_internal(srcf.read_text(errors="replace"), dst.read_text(errors="replace")))
        how = "copied (same mesh)"
    else:
        tmp = case.parent / "init_source"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "1").mkdir(parents=True)
        import os
        os.symlink(src_case.resolve() / "constant", tmp / "constant")
        shutil.copytree(src_case / "system", tmp / "system")
        for f, srcf in pairs:
            txt = srcf.read_text(errors="replace")
            if srcf.name != f:                      # UMean -> U: the object name in the header
                txt = re.sub(r"(object\s+)" + re.escape(srcf.name) + r"\s*;", r"\g<1>" + f + ";", txt, count=1)
            (tmp / "1" / f).write_text(txt)
        (case / "system" / "mapFieldsDict").write_text(
            "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      mapFieldsDict;\n}\n\n"
            "// internal fields only: every patch keeps this case's boundary conditions\npatchMap ( );\n\n"
            "cuttingPatches ( );\n")
        try:
            run_step(f"mapFields {tmp} -sourceTime 1", "log.mapFields")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"init_from: mapFields from {src} did not finish ({type(exc).__name__}: {str(exc)[:300]}). It is serial "
                f"and limited to {map_timeout():.0f} s (CFDT_MAP_TIMEOUT); see {case / 'log.mapFields'}. Start without "
                "init_from, or from a run whose mesh is closer to this one") from exc
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        how = "mapped onto this mesh (mapFields)"
    note = f"started from {src} at its time {t:g}: {', '.join(took)} {how}"
    if src_vof != dst_vof:
        note += "; pressure not carried over (single-phase and cavitation runs keep it in different units)"
    return [note], {"fields": [f for f, _ in pairs], "time": t, "same_mesh": same, "vof": src_vof}


def map_timeout() -> float:
    """Seconds mapFields may take (CFDT_MAP_TIMEOUT, default 3600). The Rogue's 911k -> 865k cells took under a
    minute on mochi (2026.11.07); a longer one is stuck, or the meshes hardly overlap."""
    import os
    try:
        return max(60.0, float(os.environ.get("CFDT_MAP_TIMEOUT", "3600")))
    except ValueError:
        return 3600.0


def wait_for(src: Path, timeout: float, cancel: Path | None = None, poll: float = 10.0) -> dict | None:
    """The source study's result once it has one (a study of the same request running before this one), or None
    after timeout seconds / on cancel."""
    f = Path(src) / f"{Path(src).name}.json"
    t_end = time.time() + timeout
    while True:
        if f.is_file():
            try:
                import json
                return json.loads(f.read_text())
            except ValueError:
                pass
        if time.time() >= t_end or (cancel is not None and Path(cancel).exists()):
            return None
        time.sleep(poll)


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
