"""cavitation.py — cavitating liquid flow: OpenFOAM 14 incompressibleVoF with the VoFCavitation fvModel.
Version: 2026.11.03.4

A study with "cavitation": {...} solves liquid and its vapour as one mixture with a vapour fraction (VoF,
alpha.water = liquid fraction), with mass transfer where the pressure falls below the vapour pressure
(Schnerr-Sauer by default; Kunz, Merkle). Pumps: the inlet holds a total pressure from the NPSH asked for,
the outlet the flow rate; MRF zones as in the steady pump; pseudo-steady with local time stepping (LTS).

    errors(study) -> [str]                          validation of the cavitation block and what it needs
    settings(study, rho_l) -> dict                  vapour pressure and properties, NPSH, target total pressure
    write_vof(study, mesh, patches, case, plan, U, kin, k0, om0, turb, zones, g, q) -> plan
    to_kinematic(field, p_ref, rho)                 a p field in Pa (absolute) as post's kinematic gauge layout
    results(study, mesh, plan, fields, res) -> (dict, warnings)   vapour volume and where, NPSHa at the inlet
    npsh_curve(points, head_drop) -> dict           head against NPSH, NPSH3 (the head 3 % below the highest-
                                                    NPSH point, interpolated)

OpenFOAM 14 (tutorials incompressibleVoF/propeller, DTCHull; uniformTotalPressure, flowRateOutletVelocity read
in its source): phaseProperties `phases (water vapour); pSat; sigma`; physicalProperties.water/.vapour; g;
fvModels VoFCavitation {model; <model> {pSat; liquid water; ...}}; fields alpha.water, p_rgh (Pa; = p with g = 0);
LTS: ddt localEuler and PIMPLE maxCo/maxAlphaCo/rDeltaT smoothing and damping.

How a point runs: the inlet's total pressure starts high enough for no vapour and falls to p_v + rho g NPSH over
the first `ramp` of the iterations, while the outlet's flow rate rises from 5 % to the study's over the first tenth: a
cold start at the target suction pressure would boil the whole inlet. The last quarter of the iterations is
averaged (head, torque, power) as for a steady MRF run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCALE = 1e-3
G0 = 9.80665
MODELS = ("SchnerrSauer", "Kunz", "Merkle")
KEYS = {"npsh", "temperature", "p_vapour", "vapour", "model", "iterations", "ramp", "head_drop", "sigma"}
M_WATER = 0.01801528
R_GAS = 8.314462618


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def water_vapour(T_C: float) -> dict:
    """Saturated water vapour at T (1-100 C): pressure (Antoine, Pa), density (ideal gas), viscosity."""
    T = float(T_C)
    p = 10 ** (8.07131 - 1730.63 / (233.426 + T)) * 133.322368
    rho = p * M_WATER / (R_GAS * (T + 273.15))
    mu = 9.0e-6 + 3.5e-8 * T
    return {"p": p, "rho": rho, "mu": mu}


def is_water(fluid) -> bool:
    return fluid == "water" or (isinstance(fluid, dict) and fluid.get("name") == "water")


def errors(st: dict) -> list[str]:
    cv = st.get("cavitation")
    if cv is None:
        return []
    e = []
    if not isinstance(cv, dict):
        return ["cavitation: {npsh (m, or a list for a sweep), temperature (C, 20) | p_vapour (Pa), model, "
                "iterations, ramp, head_drop}"]
    for k in cv:
        if k not in KEYS:
            e.append(f"cavitation: unknown key {k!r} (keys: {sorted(KEYS)})")
    n = cv.get("npsh")
    if isinstance(n, list):
        e.append("cavitation.npsh: a list makes a sweep, which run_studies / validate_studies spell out (one study "
                 "per value); give one value here")
    elif not _num(n) or n <= 0:
        e.append("cavitation.npsh: the net positive suction head at the inlet (m, > 0): the inlet's absolute total "
                 "pressure is p_vapour + rho g NPSH")
    if "temperature" in cv and (not _num(cv["temperature"]) or not 1 <= cv["temperature"] <= 100):
        e.append("cavitation.temperature: 1-100 C (water's vapour pressure and density)")
    if "p_vapour" in cv and (not _num(cv["p_vapour"]) or cv["p_vapour"] <= 0):
        e.append("cavitation.p_vapour: the liquid's vapour pressure (Pa, > 0)")
    if not is_water(st.get("fluid")):
        v = cv.get("vapour")
        if "p_vapour" not in cv or not (isinstance(v, dict) and _num(v.get("rho")) and _num(v.get("mu"))):
            e.append("cavitation: a liquid other than water needs p_vapour (Pa) and vapour: {rho, mu}")
    if "vapour" in cv and (not isinstance(cv["vapour"], dict) or set(cv["vapour"]) - {"rho", "mu"}
                           or not all(_num(v) and v > 0 for v in cv["vapour"].values())):
        e.append("cavitation.vapour: {rho (kg/m3), mu (Pa s)}")
    if cv.get("model", "SchnerrSauer") not in MODELS:
        e.append(f"cavitation.model: one of {MODELS}")
    if "iterations" in cv and (not isinstance(cv["iterations"], int) or not 300 <= cv["iterations"] <= 100000):
        e.append("cavitation.iterations: pseudo-steady (local time step) iterations, 300-100000 (default 4000)")
    if "ramp" in cv and (not _num(cv["ramp"]) or not 0 <= cv["ramp"] <= 0.7):
        e.append("cavitation.ramp: the part of the iterations over which the inlet pressure falls to its target, 0-0.7 "
                 "(default 0.3; the last quarter is averaged)")
    if "head_drop" in cv and (not _num(cv["head_drop"]) or not 0 < cv["head_drop"] < 0.5):
        e.append("cavitation.head_drop: the head drop that defines the NPSH curve's critical point (0.03: NPSH3)")
    if "sigma" in cv and (not _num(cv["sigma"]) or cv["sigma"] < 0):
        e.append("cavitation.sigma: surface tension (N/m, default 0.07)")
    # what the solver path supports
    if st.get("energy"):
        e.append("cavitation: not with energy")
    if (st.get("solve") or {}).get("transient"):
        e.append("cavitation: pseudo-steady (local time stepping) only for now; remove solve.transient (a transient "
                 "and sliding-mesh cavitation run is planned)")
    if any(isinstance(z, dict) and z.get("motion") == "sliding" for z in st.get("rotating") or []):
        e.append("cavitation: with MRF zones only for now (sliding-mesh cavitation is planned)")
    for k in ("scalars", "age", "porous"):
        if st.get(k):
            e.append(f"cavitation: not with {k} yet")
    if st.get("model", "laminar") not in ("laminar", "kOmegaSST"):
        e.append("cavitation: model laminar or kOmegaSST")
    kinds = [b.get("kind") for b in st.get("bcs") or [] if isinstance(b, dict)]
    if kinds.count("inlet") != 1 or kinds.count("outlet") != 1 or "opening" in kinds:
        e.append("cavitation: one inlet (its flow rate) and one outlet, no openings: the inlet then holds the total "
                 "pressure from the NPSH and the outlet the flow rate")
    for b in st.get("bcs") or []:
        if isinstance(b, dict) and b.get("kind") == "inlet":
            if not any(k in b for k in ("flow_rate", "flow_rate_lpm", "velocity")) or isinstance(b.get("velocity"), list):
                e.append("cavitation: the inlet needs flow_rate, flow_rate_lpm or a velocity (m/s, normal): the flow "
                         "rate the outlet then draws")
            if b.get("swirl") or b.get("profile", "uniform") != "uniform":
                e.append("cavitation: the inlet takes a total pressure (no swirl or profile)")
    return e


def settings(study: dict, rho_l: float, mu_l: float) -> dict:
    cv = study["cavitation"]
    T = float(cv.get("temperature", 20.0))
    wv = water_vapour(T)
    pv = float(cv.get("p_vapour", wv["p"]))
    vap = cv.get("vapour") or {}
    rho_v = float(vap.get("rho", pv * M_WATER / (R_GAS * (T + 273.15)) if "p_vapour" in cv else wv["rho"]))
    mu_v = float(vap.get("mu", wv["mu"]))
    npsh = float(cv["npsh"])
    return {"temperature_C": T, "p_vapour_Pa": pv, "rho_l": float(rho_l), "nu_l": float(mu_l) / float(rho_l),
            "rho_v": rho_v, "nu_v": mu_v / rho_v, "npsh_m": npsh, "p0_Pa": pv + float(rho_l) * G0 * npsh,
            "model": cv.get("model", "SchnerrSauer"), "iterations": int(cv.get("iterations", 4000)),
            "ramp": float(cv.get("ramp", 0.3)), "head_drop": float(cv.get("head_drop", 0.03)),
            "sigma": float(cv.get("sigma", 0.07)),
            "p_vapour_from": "given" if "p_vapour" in cv else f"water at {T:g} C (Antoine)"}


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def _table(points: list) -> str:
    return "table (" + " ".join(f"({_fmt(t)} {_fmt(v)})" for t, v in points) + ")"


def fv_models(cv: dict, u_ref: float, l_ref: float) -> str:
    pS = _fmt(cv["p_vapour_Pa"])
    t_inf = l_ref / max(u_ref, 1e-9)
    return (f"VoFCavitation\n{{\n    type    VoFCavitation;\n\n    model   {cv['model']};\n\n"
            f"    SchnerrSauer\n    {{\n        pSat    {pS};\n        liquid  water;\n        n       1.6e+13;\n"
            f"        dNuc    2.0e-06;\n        Cc      1;\n        Cv      1;\n    }}\n\n"
            f"    Kunz\n    {{\n        pSat    {pS};\n        liquid  water;\n        UInf    {_fmt(u_ref)};\n"
            f"        tInf    {_fmt(t_inf)};\n        Cc      1000;\n        Cv      1000;\n    }}\n\n"
            f"    Merkle\n    {{\n        pSat    {pS};\n        liquid  water;\n        UInf    {_fmt(u_ref)};\n"
            f"        tInf    {_fmt(t_inf)};\n        Cc      80;\n        Cv      1e-03;\n    }}\n}}")


def write_vof(study, mesh, patches, case, plan, U, kin, k0, om0, turb, zones, g, q) -> dict:
    """The incompressibleVoF case (fields, constant, system) after write_case has the zones and inlets."""
    from .average import monitor_functions
    from .case import (_dict, _field, _mrf_wall_leaks, _uniform, nonortho_schemes, wall_velocity, write_mrf)
    case = Path(case)
    rho_l, mu_l = float(plan["rho"]), float(plan["mu"])
    cv = settings(study, rho_l, mu_l)
    inlet = next(p for p in patches if p["kind"] == "inlet")["name"]
    outlet = next(p for p in patches if p["kind"] == "outlet")["name"]
    Q = float(plan["inlets"][inlet]["Q"])
    s = study.get("solve") or {}
    iters = int((study["cavitation"] or {}).get("iterations", s.get("iterations", cv["iterations"])))
    n_ramp = max(1, int(round(cv["ramp"] * iters)))
    n_flow = max(1, int(round(0.1 * iters)))
    # a start without vapour: the inlet's total pressure above the lowest pressure the rotor can make
    u_tip = 0.0
    for zn, zp in plan["zones"].items():
        u_tip = max(u_tip, abs(zp["omega_rad_s"]) * float(zones[zn]["radius"]) * SCALE)
    u_ref = max(float(plan.get("U_ref", 1.0)), u_tip)
    p_start = cv["p0_Pa"] + rho_l * max(u_tip, 3.0 * float(plan.get("U_ref", 1.0))) ** 2
    l_ref = 2 * max((float(z["radius"]) for z in zones.values()), default=0.0) * SCALE or \
        float(plan["inlets"][inlet].get("hydraulic_diameter_m", 0.05))
    p0_tab = _table([(0, p_start), (n_ramp, cv["p0_Pa"]), (10 * iters + 1e6, cv["p0_Pa"])])
    # the outlet's flow rises from 5 % (not 0: flowRateOutletVelocity divides its estimate by the flow rate,
    # estimatedFlowRate/flowRate, a 0/0 that stopped the first S50 run on carbonio with SIGFPE, 2026.11.03)
    q_tab = _table([(0, 0.05 * Q), (n_flow, Q), (10 * iters + 1e6, Q)])
    plan["vof"] = {**cv, "p_ref_Pa": cv["p0_Pa"], "p_start_Pa": p_start, "inlet": inlet, "outlet": outlet, "Q": Q,
                   "ramp_iterations": n_ramp, "flow_ramp_iterations": n_flow, "u_tip": u_tip}
    plan["fixed_pressure"] = True
    # fields
    Ub, pb, ab, kb, wb, nb = {}, {}, {}, {}, {}, {}
    for p in patches:
        nm, kind = p["name"], p["kind"]
        if kind in ("empty", "symmetry"):
            for d in (Ub, pb, ab, kb, wb, nb):
                d[nm] = {"type": kind}
            continue
        if kind == "inlet":
            k, om = kin[nm]
            Ub[nm] = {"type": "pressureInletOutletVelocity", "value": _uniform([0, 0, 0])}
            pb[nm] = {"type": "uniformTotalPressure", "p0": p0_tab, "value": _uniform(p_start)}
            ab[nm] = {"type": "fixedValue", "value": _uniform(1)}
            kb[nm] = {"type": "fixedValue", "value": _uniform(k)}
            wb[nm] = {"type": "fixedValue", "value": _uniform(om)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind == "outlet":
            Ub[nm] = {"type": "flowRateOutletVelocity", "volumetricFlowRate": q_tab, "value": _uniform([0, 0, 0])}
            pb[nm] = {"type": "fixedFluxPressure", "value": _uniform(p_start)}
            ab[nm] = {"type": "inletOutlet", "inletValue": _uniform(1), "value": _uniform(1)}
            kb[nm] = {"type": "inletOutlet", "inletValue": _uniform(k0), "value": _uniform(k0)}
            wb[nm] = {"type": "inletOutlet", "inletValue": _uniform(om0), "value": _uniform(om0)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind == "wall":
            Ub[nm] = wall_velocity(p, mesh, g, zones, plan)
            pb[nm] = {"type": "fixedFluxPressure", "value": _uniform(p_start)}
            ab[nm] = {"type": "zeroGradient"}
            kb[nm] = {"type": "kqRWallFunction", "value": _uniform(k0)}
            wb[nm] = {"type": "omegaWallFunction", "value": _uniform(om0)}
            nb[nm] = {"type": "nutkWallFunction", "value": _uniform(0)}
        elif kind == "slip":
            Ub[nm] = {"type": "slip"}
            pb[nm] = {"type": "fixedFluxPressure", "value": _uniform(p_start)}
            for d in (ab, kb, wb):
                d[nm] = {"type": "zeroGradient"}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        else:
            from .polymesh import MeshError
            raise MeshError(f"cavitation: bc kind {kind} is not supported")
    z = case / "0"
    for f in z.iterdir() if z.is_dir() else []:
        if f.is_file():
            f.unlink()
    z.mkdir(parents=True, exist_ok=True)
    (z / "U").write_text(_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", _uniform([0, 0, 0]), Ub))
    (z / "p_rgh").write_text(_field("volScalarField", "p_rgh", "[1 -1 -2 0 0 0 0]", _uniform(p_start), pb))
    (z / "alpha.water").write_text(_field("volScalarField", "alpha.water", "[0 0 0 0 0 0 0]", _uniform(1), ab))
    if turb:
        (z / "k").write_text(_field("volScalarField", "k", "[0 2 -2 0 0 0 0]", _uniform(k0), kb))
        (z / "omega").write_text(_field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", _uniform(om0), wb))
        (z / "nut").write_text(_field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", _uniform(0), nb))
    # constant
    c = case / "constant"
    (c / "physicalProperties").unlink(missing_ok=True)
    (c / "phaseProperties").write_text(_dict("phaseProperties", "constant",
        f"phases          (water vapour);\n\npSat            {_fmt(cv['p_vapour_Pa'])};\n\nsigma           {_fmt(cv['sigma'])};"))
    for ph, nu, rho in (("water", cv["nu_l"], rho_l), ("vapour", cv["nu_v"], cv["rho_v"])):
        (c / f"physicalProperties.{ph}").write_text(_dict(f"physicalProperties.{ph}", "constant",
            f"viscosityModel  constant;\n\nnu              {_fmt(nu)};\n\nrho             {_fmt(rho)};"))
    (c / "g").write_text("FoamFile\n{\n    format      ascii;\n    class       uniformDimensionedVectorField;\n"
                         "    location    \"constant\";\n    object      g;\n}\n\ndimensions      [0 1 -2 0 0 0 0];\n"
                         "value           (0 0 0);\n")
    mt = ("simulationType  RAS;\n\nRAS\n{\n    model           kOmegaSST;\n\n    turbulence      on;\n\n"
          "    printCoeffs     on;\n}") if turb else "simulationType  laminar;"
    (c / "momentumTransport").write_text(_dict("momentumTransport", "constant", mt))
    (c / "fvModels").write_text(_dict("fvModels", "constant", fv_models(cv, u_ref, l_ref)))
    write_mrf(case, zones, plan)
    mesh.write(case)
    # system
    res = float(s.get("residual", 1e-5))
    lap, sn, nonorth, rf = nonortho_schemes(q)
    unstructured = q["max_non_orthogonality"] > 5.0
    if not unstructured:
        lap, sn = "Gauss linear corrected", "corrected"
    grad = ("default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n"
            "    limitedGrad     cellLimited Gauss linear 1;")
    relax = s.get("relax", {})
    rU = float(relax.get("U", 1.0))
    plan.update({"iterations": iters, "residual": res, "relax": {"U": rU, "other": 1.0},
                 "schemes": "unstructured" if unstructured else "orthogonal", "laplacian": lap, "sngrad": sn,
                 "nonorth": nonorth, "grad": grad})
    ptags = {int(t) for t, i in (mesh.tag_info or {}).items() if i.get("on") == "part"}
    external = study.get("domain", {}).get("kind") == "external"
    plan["body_walls"] = [p["name"] for p in patches if p["type"] == "wall"
                          and (not external or (p["tags"] and all(int(t) in ptags for t in p["tags"])))]
    funcs, mon = monitor_functions(study, plan, rho_l)
    plan["monitors"] = mon
    plan["functions"] = funcs
    W = max(1, iters // 4)
    cd = (f"solver          incompressibleVoF;\n\nstartFrom       startTime;\n\nstartTime       0;\n\n"
          f"stopAt          endTime;\n\nendTime         {iters};\n\ndeltaT          1;\n\nwriteControl    timeStep;\n\n"
          f"writeInterval   {W};\n\npurgeWrite      2;\n\nwriteFormat     ascii;\n\nwritePrecision  10;\n\n"
          "writeCompression off;\n\ntimeFormat      general;\n\ntimePrecision   8;\n\nrunTimeModifiable false;\n\n"
          "functions\n{\n" + "\n".join(funcs) + "\n}")
    sy = case / "system"
    (sy / "controlDict").write_text(_dict("controlDict", "system", cd))
    fs = (f"ddtSchemes\n{{\n    default         localEuler;\n}}\n\ngradSchemes\n{{\n    {grad}\n}}\n\n"
          "divSchemes\n{\n    default         none;\n"
          "    div(phi,alpha)  Gauss noInterfaceCompression vanLeer;\n"
          "    div(rhoPhi,U)   Gauss linearUpwind grad(U);\n"
          "    div(phi,k)      Gauss linearUpwind limitedGrad;\n    div(phi,omega)  Gauss linearUpwind limitedGrad;\n"
          "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
          f"laplacianSchemes\n{{\n    default         {lap};\n}}\n\ninterpolationSchemes\n{{\n    default         linear;\n}}\n\n"
          f"snGradSchemes\n{{\n    default         {sn};\n}}\n\nwallDist\n{{\n    method          meshWave;\n}}")
    (sy / "fvSchemes").write_text(_dict("fvSchemes", "system", fs))
    fields = "U|k|omega" if turb else "U"
    fv = ("solvers\n{\n    \"alpha.water.*\"\n    {\n        nCorrectors     2;\n        nSubCycles      1;\n"
          "        MULESCorr       yes;\n        MULES\n        {\n            nIter           5;\n        }\n"
          "        solver          smoothSolver;\n        smoother        symGaussSeidel;\n        tolerance       1e-8;\n"
          "        relTol          0;\n        maxIter         10;\n        minIter         1;\n    }\n\n"
          "    p_rgh\n    {\n        solver          GAMG;\n        smoother        DIC;\n        tolerance       1e-7;\n"
          "        relTol          0.01;\n    }\n\n    p_rghFinal\n    {\n        $p_rgh;\n        relTol          0;\n    }\n\n"
          "    \"pcorr.*\"\n    {\n        $p_rgh;\n        tolerance       1e-5;\n        relTol          0;\n    }\n\n"
          f"    \"({fields}).*\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-7;\n        relTol          0.1;\n        minIter         1;\n    }\n}\n\n"
          "PIMPLE\n{\n    momentumPredictor yes;\n    nOuterCorrectors 1;\n    nCorrectors     2;\n"
          f"    nNonOrthogonalCorrectors {nonorth};\n\n    maxCo           5;\n    maxAlphaCo      2;\n"
          "    rDeltaTSmoothingCoeff 0.05;\n    rDeltaTDampingCoeff 0.5;\n    nAlphaSpreadIter 0;\n"
          "    nAlphaSweepIter 0;\n    maxDeltaT       1;\n}\n\n"
          f"relaxationFactors\n{{\n    equations\n    {{\n        U               {rU:g};\n        \".*\"            1;\n    }}\n}}")
    (sy / "fvSolution").write_text(_dict("fvSolution", "system", fv))
    for f in ("fvConstraints",):
        (sy / f).unlink(missing_ok=True)
    (case / "case.foam").write_text("")
    _mrf_wall_leaks(mesh, g, zones, plan, patches)
    return plan


# ---- results -----------------------------------------------------------------------------------
def to_kinematic(field: dict | None, p_ref: float, rho: float) -> dict | None:
    """A p (or p_rgh) field in Pa, absolute, as the kinematic gauge field post.results reads: (p - p_ref) / rho
    in the cells and in every boundary value written."""
    if field is None:
        return None
    out = dict(field)

    def conv(v):
        a = np.asarray(v, float)
        return (a - p_ref) / rho
    out["internalField"] = conv(field["internalField"])
    bf = {}
    for nm, e in (field.get("boundaryField") or {}).items():
        e = dict(e)
        if "value" in e and not isinstance(e["value"], str):
            e["value"] = conv(e["value"])
        bf[nm] = e
    out["boundaryField"] = bf
    return out


def results(study: dict, mesh, plan: dict, fields: dict, res: dict) -> tuple[dict, list]:
    vf = plan["vof"]
    g = mesh.geometry()
    V = g["V"] * SCALE ** 3
    warns = []
    a = np.asarray(fields.get("alpha", np.ones(mesh.ncells)), float)
    a_min, a_max = float(a.min()), float(a.max())
    vap = (1.0 - np.clip(a, 0, 1)) * V
    Vv = float(vap.sum())
    Vt = float(V.sum())
    rho_l, pv = vf["rho_l"], vf["p_vapour_Pa"]
    out = {"npsh_m": vf["npsh_m"], "p_vapour_Pa": round(pv, 2), "p_vapour_from": vf["p_vapour_from"],
           "model": vf["model"], "inlet_total_pressure_Pa": round(vf["p0_Pa"], 2),
           "vapour_volume_mm3": float(f"{Vv / SCALE ** 3:.5g}"), "vapour_fraction": float(f"{Vv / Vt:.4g}") if Vt else 0.0,
           "cavitating": bool(Vv > 1e-7 * Vt)}
    cells = np.nonzero(a < 0.5)[0]
    if len(cells):
        C = g["C"][cells]
        w = vap[cells]
        out["vapour_region"] = {"cells": int(len(cells)),
                                "centroid_mm": np.round((C * w[:, None]).sum(axis=0) / max(w.sum(), 1e-300), 1).tolist(),
                                "bbox_mm": [np.round(C.min(axis=0), 1).tolist(), np.round(C.max(axis=0), 1).tolist()]}
    pa = fields.get("p_abs")
    if pa is not None and len(pa):
        k = int(np.argmin(pa))
        out["min_pressure_Pa"] = float(f"{float(pa[k]):.5g}")
        out["min_pressure_at_mm"] = np.round(g["C"][k], 1).tolist()
    # NPSH available at the inlet from the flow-weighted total pressure (gauge to the target total pressure)
    pin = (res.get("pressure") or {}).get(vf["inlet"]) or {}
    if pin.get("p_total") is not None:
        pt = vf["p_ref_Pa"] + float(pin["p_total"])
        out["npsh_inlet_m"] = float(f"{(pt - pv) / (rho_l * G0):.4g}")
    if vf.get("u_tip"):
        out["sigma_tip"] = float(f"{(vf['p0_Pa'] - pv) / (0.5 * rho_l * vf['u_tip'] ** 2):.4g}")
    # vapour on the walls: the part of each wall's area whose cells hold mostly vapour
    walls = {}
    for P in mesh.patches:
        if P["type"] != "wall":
            continue
        f = np.arange(P["start"], P["start"] + P["n"])
        A = np.linalg.norm(g["Sf"][f], axis=1)
        wet = a[mesh.owner[f]] < 0.5
        if wet.any():
            walls[P["name"]] = {"area_fraction": float(f"{A[wet].sum() / A.sum():.4g}"), "faces": int(wet.sum())}
    if walls:
        out["walls"] = walls
    ok = a_min > -1e-3 and a_max < 1 + 1e-3
    out["alpha_bounds"] = {"min": float(f"{a_min:.4g}"), "max": float(f"{a_max:.4g}"), "ok": bool(ok)}
    if not ok:
        warns.append(f"the liquid fraction left [0, 1] (min {a_min:.3g}, max {a_max:.3g}): the vapour transport is "
                     "unstable; lower the time step (more iterations, smaller maxCo) or refine where the vapour is")
    if out.get("npsh_inlet_m") is not None and abs(out["npsh_inlet_m"] - vf["npsh_m"]) > 0.05 * max(vf["npsh_m"], 1.0):
        warns.append(f"NPSH at the inlet {out['npsh_inlet_m']} m (flow-weighted total pressure) against "
                     f"{vf['npsh_m']} m asked: the inlet's velocity is not uniform; report npsh_inlet_m")
    if out["cavitating"] and res.get("checks", {}).get("momentum_balance"):
        res["checks"]["momentum_balance"]["note"] = ("vapour in the domain: the balance uses the liquid's density, "
                                                     "so it is not judged")
    if out["cavitating"] and res.get("checks", {}).get("angular_momentum_balance"):
        res["checks"]["angular_momentum_balance"]["note"] = "vapour in the domain: not judged"
    res.setdefault("checks", {})["alpha_bounds"] = out["alpha_bounds"]
    return out, warns


# ---- the NPSH curve ------------------------------------------------------------------------------
def npsh_curve(points: list, head_drop: float = 0.03) -> dict:
    """points: [{name, npsh, head, reliable, vapour_mm3}] (any order). Head against NPSH from the highest NPSH
    down; the reference head is the highest-NPSH point's; NPSH<drop> where the head first falls head_drop below
    it, interpolated linearly between the two points around it."""
    pts = sorted([p for p in points if p.get("head") is not None], key=lambda p: -p["npsh"])
    out = {"points": pts, "head_drop": head_drop}
    if len(pts) < 2:
        out["note"] = "fewer than two finished points"
        return out
    H0 = pts[0]["head"]
    out["reference"] = {"npsh": pts[0]["npsh"], "head": H0}
    target = (1 - head_drop) * H0
    for p in pts:
        p["head_ratio"] = float(f"{p['head'] / H0:.4g}") if H0 else None
    key = f"npsh{round(100 * head_drop):g}_m"
    for a, b in zip(pts[:-1], pts[1:]):
        if a["head"] >= target > b["head"]:
            t = (a["head"] - target) / (a["head"] - b["head"])
            out[key] = float(f"{a['npsh'] + t * (b['npsh'] - a['npsh']):.4g}")
            out["between"] = [a["name"], b["name"]]
            break
    else:
        out[key] = None
        out["note"] = (f"the head did not fall {head_drop:.0%} below the highest-NPSH point's down to NPSH "
                       f"{pts[-1]['npsh']} m: add lower NPSH values" if pts[-1]["head"] >= target else
                       "the head fell already at the second-highest NPSH: add higher NPSH values")
    if pts[0].get("vapour_mm3"):
        out.setdefault("warnings", []).append(
            f"the highest-NPSH point ({pts[0]['npsh']} m) already holds {pts[0]['vapour_mm3']:.3g} mm3 of vapour: add a "
            "higher NPSH so the reference head is free of cavitation")
    if not all(p.get("reliable") for p in pts):
        out.setdefault("warnings", []).append("some points are not reliable (see each study's unreliable_because)")
    return out
