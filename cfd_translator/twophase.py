"""twophase.py — gas-liquid flow with slip: OpenFOAM 14 multiphaseEuler (two-fluid, Euler-Euler), water + air.
Version: 2026.11.20.7

A study with "two_phase": {...} solves water and air as two interpenetrating phases, each with its own velocity:
the air slips through the water under the local pressure gradient (gravity, and in an MRF zone the centrifugal and
Coriolis forces of the rotating frame), held back by drag on bubbles of a given diameter. That is what drives air
to the eye of an impeller and lets it separate in a discharge chamber: a single-velocity (VoF) model cannot.

    errors(study) -> [str]                    validation of the two_phase block
    settings(study) -> dict                   the block with defaults
    write_euler(study, mesh, patches, case, plan, zones, g, q) -> plan   fields, constant, system (from write_case)
    run_and_results(case, study, plan, mesh, timeout, procs, warns, t_job) -> dict   solve and read the monitors
    terminal_velocity(d, accel, rho_l, rho_g, mu_l) -> (u, Re)   Schiller-Naumann drag balance (validation)
    core_fraction(t, R, a, omega, d, rho_l, rho_g, mu_l) -> f    air within r < a of a rotating cylinder (validation)

Dictionaries from the OpenFOAM 14 tutorials multiphaseEuler/aeratedStirredTankMRF (MRF, gas-liquid, kOmegaSSTSato,
phaseTurbulenceStabilisation, semi-implicit MULES with sub-cycling) and bubbleColumnLaminar (perfect-gas air, the
blending of dispersed and segregated drag, prghTotalPressure at an open top): phases (air water), referencePhase
water, pureIsothermalPhaseModel (no energy equation; T fixed), the air a perfect gas (its density follows p: the
1.0-1.2 bar column cases and later vacuum ones), the water rhoConst. The bubble diameter is fixed
(bubble_size {fixed: mm}); {iate: ...} (interfacial area transport, bubbleColumnIATE) is reserved for a second pass.
volFieldValue in OpenFOAM 14 needs a cellZone (all, or a name) and applies weightField to volIntegrate/volAverage.

Monitors (every write_every steps): the air mass in the domain (alpha.air weighted by rho.air) and through every open
patch (alphaRhoPhi.air), the liquid flow through every open patch (alphaPhi.water), each region's gas fraction and
phase-averaged velocities (regions become cellZones tp_<name>), the moment on the rotating walls (forces per
phase), and the static pressure on inlets and outlets. results: two_phase {air_mass_balance, flows, regions,
pump (head, torque, power), reference comparison}; checks.air_mass_balance judges 'reliable' (earlier water + air
runs, compressibleVoF, were flagged for air not conserved).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SCALE = 1e-3
G0 = 9.80665
R_AIR = 287.05
AIR = {"molWeight": 28.96, "Cp": 1007.0, "mu": 1.84e-5, "Pr": 0.7}
KEYS = {"bubble_size", "drag", "virtual_mass", "turbulent_dispersion", "gravity", "p_ref_Pa", "temperature_C",
        "mode", "iterations", "end_time", "write_interval", "write_every", "max_co", "max_delta_t", "inlet",
        "initial", "regions", "reference", "relax", "average_from", "face_momentum"}
DRAG = ("SchillerNaumann", "IshiiZuber")
BALANCE_TOL = 0.01                     # air mass balance: |error| / air in play
REF_TOL = 0.02                         # single-phase consistency at a gas fraction of 0: head and torque


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _vec(x, n=3) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == n and all(_num(v) for v in x)


def settings(study: dict) -> dict:
    tp = dict(study.get("two_phase") or {})
    bs = tp.get("bubble_size", {"fixed": 1.0})
    if _num(bs):
        bs = {"fixed": float(bs)}
    steady = tp.get("mode", "steady") == "steady"
    turb = (study.get("model") or "laminar") != "laminar"
    inl = dict(tp.get("inlet") or {})
    return {"d_mm": float(bs.get("fixed", 1.0)), "bubble_size": bs, "drag": tp.get("drag", "SchillerNaumann"),
            "Cvm": (0.0 if tp.get("virtual_mass") is False else float(tp.get("virtual_mass", 0.5)
                                                                       if _num(tp.get("virtual_mass")) else 0.5)),
            "turbulent_dispersion": bool(tp.get("turbulent_dispersion", turb)) and turb,
            "gravity": [float(v) for v in tp.get("gravity", [0.0, 0.0, -G0])],
            "p_ref": float(tp.get("p_ref_Pa", 101325.0)), "T_K": float(tp.get("temperature_C", 20.0)) + 273.15,
            "steady": steady, "iterations": int(tp.get("iterations", 3000)),
            "end_time": float(tp.get("end_time", 1.0)), "write_interval": tp.get("write_interval"),
            "write_every": int(tp.get("write_every", 1 if not steady else 5)),
            "max_co": float(tp.get("max_co", 1.0 if steady else 0.5)), "max_delta_t": tp.get("max_delta_t"),
            "gas_fraction": float(inl.get("gas_fraction", 0.0)), "air_flow_lpm": inl.get("air_flow_lpm"),
            "initial": dict(tp.get("initial") or {}), "regions": list(tp.get("regions") or []),
            "reference": tp.get("reference"), "relax": dict(tp.get("relax") or {}),
            "average_from": tp.get("average_from"), "turbulent": turb,
            "face_momentum": bool(tp.get("face_momentum", False))}


def errors(st: dict) -> list[str]:
    """Validation of the two_phase block and what it cannot be combined with."""
    tp = st.get("two_phase")
    if tp is None:
        return []
    e = []
    if not isinstance(tp, dict):
        return ["two_phase: {bubble_size: {fixed: mm}, drag, gravity, mode: steady | transient, inlet: {gas_fraction "
                "| air_flow_lpm}, initial, regions, ...} (guide('pumps'))"]
    bad = set(tp) - KEYS
    if bad:
        e.append(f"two_phase: unknown keys {sorted(bad)} ({sorted(KEYS)})")
    bs = tp.get("bubble_size", {"fixed": 1.0})
    if _num(bs):
        bs = {"fixed": bs}
    if not isinstance(bs, dict) or set(bs) - {"fixed", "iate"} or len(bs) != 1:
        e.append("two_phase.bubble_size: {fixed: d (mm)} (or a number); {iate: {...}} is reserved for a later version")
    elif "iate" in bs:
        e.append("two_phase.bubble_size.iate: not available yet (interfacial area transport, bubbleColumnIATE): give "
                 "{fixed: d} and compare several d")
    elif not (_num(bs["fixed"]) and 0.01 <= bs["fixed"] <= 20):
        e.append("two_phase.bubble_size.fixed: the bubble diameter, 0.01-20 mm")
    if tp.get("drag", "SchillerNaumann") not in DRAG:
        e.append(f"two_phase.drag: one of {DRAG}")
    vm = tp.get("virtual_mass", 0.5)
    if not (vm is False or (_num(vm) and 0 <= vm <= 1)):
        e.append("two_phase.virtual_mass: the coefficient Cvm (0.5) or false")
    if "gravity" in tp and not _vec(tp["gravity"]):
        e.append("two_phase.gravity: [gx, gy, gz] m/s2")
    elif "gravity" not in tp and ("fill_level_mm" in (tp.get("initial") or {}) or any(
            isinstance(b, dict) and b.get("kind") == "opening" for b in st.get("bcs") or [])):
        # a free surface or an atmosphere: where 'up' is decides where the air goes. The S50's shaft is horizontal
        # (impeller axis z, y up: gravity [0, -9.81, 0]); a default along -z would have put the reservoir's free
        # surface along the shaft (pump chat, 2026-11-20)
        e.append("two_phase.gravity: required with a free surface (initial.fill_level_mm) or an opening: [gx, gy, gz] "
                 "m/s2 in the study's frame (S50 and S45+50: [0, -9.81, 0], shaft along z, y up)")
    if tp.get("mode", "steady") not in ("steady", "transient"):
        e.append("two_phase.mode: 'steady' (pseudo-transient, local time step) or 'transient'")
    for k, lo in (("iterations", 1), ("write_every", 1)):
        if k in tp and (not isinstance(tp[k], int) or tp[k] < lo):
            e.append(f"two_phase.{k}: an integer >= {lo}")
    for k in ("end_time", "write_interval", "max_co", "max_delta_t", "p_ref_Pa"):
        if k in tp and (not _num(tp[k]) or tp[k] <= 0):
            e.append(f"two_phase.{k}: a positive number")
    if "average_from" in tp and (not _num(tp["average_from"]) or tp["average_from"] < 0):
        e.append("two_phase.average_from: a time (transient) or iteration (steady) from which the monitors are averaged")
    inl = tp.get("inlet") or {}
    if not isinstance(inl, dict) or set(inl) - {"gas_fraction", "air_flow_lpm"} or len(inl) > 1:
        e.append("two_phase.inlet: {gas_fraction: 0-0.6} or {air_flow_lpm: L/min at the inlet's pressure}")
    elif "gas_fraction" in inl and not (_num(inl["gas_fraction"]) and 0 <= inl["gas_fraction"] <= 0.6):
        e.append("two_phase.inlet.gas_fraction: 0-0.6")
    elif "air_flow_lpm" in inl and not (_num(inl["air_flow_lpm"]) and inl["air_flow_lpm"] >= 0):
        e.append("two_phase.inlet.air_flow_lpm: >= 0")
    ini = tp.get("initial") or {}
    if not isinstance(ini, dict) or set(ini) - {"gas_fraction", "fill_level_mm", "velocity"}:
        e.append("two_phase.initial: {gas_fraction (in the liquid), fill_level_mm (liquid below this height against "
                 "gravity, air above), velocity: 'rest' | 'rotation' (solid body with the rotating zone)}")
    else:
        if "gas_fraction" in ini and not (_num(ini["gas_fraction"]) and 0 <= ini["gas_fraction"] < 1):
            e.append("two_phase.initial.gas_fraction: 0-1")
        if "fill_level_mm" in ini and not _num(ini["fill_level_mm"]):
            e.append("two_phase.initial.fill_level_mm: a height (mm) along -gravity")
        if ini.get("velocity", "rest") not in ("rest", "rotation"):
            e.append("two_phase.initial.velocity: 'rest' or 'rotation'")
        elif ini.get("velocity") == "rotation" and not st.get("rotating") and not any(
                isinstance(b, dict) and isinstance(b.get("rotation"), dict) for b in st.get("bcs") or []):
            e.append("two_phase.initial.velocity 'rotation' needs a rotating zone or a wall with its own rotation "
                     "{axis, center, rpm}")
    regs = tp.get("regions") or []
    names = []
    for i, r in enumerate(regs if isinstance(regs, list) else [None]):
        ok = isinstance(r, dict) and isinstance(r.get("name"), str) and r["name"].isidentifier()
        if ok and "cylinder" in r:
            c = r["cylinder"]
            ok = isinstance(c, dict) and _num(c.get("radius")) and c["radius"] > 0 and _vec(c.get("axis", [0, 0, 1])) \
                and _vec(c.get("center", [0, 0, 0])) and ("span" not in c or (_vec(c["span"], 2) and c["span"][0] < c["span"][1])) \
                and ("inner_radius" not in c or (_num(c["inner_radius"]) and 0 <= c["inner_radius"] < c["radius"]))
        elif ok and "box" in r:
            ok = _vec(r["box"], 6) and all(r["box"][k] < r["box"][k + 3] for k in range(3))
        else:
            ok = False
        if not ok:
            e.append(f"two_phase.regions[{i}]: {{name (a word), cylinder: {{center, axis, radius, inner_radius, span}} | "
                     "box: [x0, y0, z0, x1, y1, z1]}} (mm): gas fraction, air mass and the phases' velocities there")
        else:
            names.append(r["name"])
    if len(set(names)) != len(names):
        e.append("two_phase.regions: names must be unique")
    if "face_momentum" in tp and not isinstance(tp["face_momentum"], bool):
        e.append("two_phase.face_momentum: true | false (multiphaseEuler's face-based momentum; default false)")
    if "reference" in tp and not isinstance(tp["reference"], str):
        e.append("two_phase.reference: '<request>/<study>' of a single-phase run on the same mesh (head and torque "
                 "compared at a gas fraction of 0)")
    s = st.get("solve") or {}
    for what, bad_ in (("cavitation", st.get("cavitation")), ("energy", st.get("energy")),
                       ("solve.transient (use two_phase.mode 'transient')", s.get("transient")),
                       ("solve.backend esi", s.get("backend") == "esi"), ("porous zones", st.get("porous")),
                       ("scalars", st.get("scalars")),
                       ("a sliding zone (MRF only)", any(isinstance(z, dict) and z.get("motion") == "sliding"
                                                         for z in st.get("rotating") or [])),
                       ("external flow", (st.get("domain") or {}).get("kind") == "external")):
        if bad_:
            e.append(f"two_phase: not with {what}")
    if st.get("model", "laminar") not in ("laminar", "kOmegaSST"):
        e.append("two_phase: model laminar or kOmegaSST (water kOmegaSSTSato, air kOmegaSST)")
    if not isinstance(st.get("fluid"), str) or st.get("fluid") != "water":
        e.append("two_phase: fluid 'water' (the liquid phase; the gas is air)")
    return e


# ---- analytic answers for the validation cases --------------------------------------------------------------
def slip_velocity(accel: float, d: float, rho_l: float, rho_g: float, mu_l: float) -> tuple[float, float]:
    """Steady slip of a bubble of diameter d (m) driven by an acceleration (m/s2: g, or omega^2 r) against Schiller-
    Naumann drag, Cd = 24/Re (1 + 0.15 Re^0.687): (Delta rho) a pi d^3 / 6 = 3 pi mu d u (1 + 0.15 Re^0.687), Re =
    rho_l u d / mu. Returns (u m/s, Re). The Stokes value is Delta rho a d^2 / (18 mu)."""
    us = (rho_l - rho_g) * abs(accel) * d * d / (18.0 * mu_l)
    u = us
    for _ in range(200):
        re = rho_l * u * d / mu_l
        un = us / (1.0 + 0.15 * re ** 0.687)
        if abs(un - u) <= 1e-12 * max(us, 1e-30):
            u = un
            break
        u = 0.5 * (u + un)
    return u, rho_l * u * d / mu_l


def terminal_velocity(d: float, accel: float, rho_l: float, rho_g: float, mu_l: float) -> tuple[float, float]:
    return slip_velocity(accel, d, rho_l, rho_g, mu_l)


def core_fraction(t: float, R: float, a: float, omega: float, d: float, rho_l: float, rho_g: float,
                  mu_l: float, n: int = 400) -> float:
    """Share of the air within r < a at time t in a cylinder of radius R in solid-body rotation (omega), air uniform
    at t = 0, each bubble moving inward at slip_velocity(omega^2 r): f = (r0*/R)^2 where r0* is where the bubble now
    at a started (integrated backwards, RK4); 1 once r0* >= R."""
    if t <= 0:
        return (a / R) ** 2
    h = t / n
    r = a
    for _ in range(n):
        def v(x):
            return slip_velocity(omega * omega * x, d, rho_l, rho_g, mu_l)[0]
        k1 = v(r)
        k2 = v(r + 0.5 * h * k1)
        k3 = v(r + 0.5 * h * k2)
        k4 = v(r + h * k3)
        r += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if r >= R:
            return 1.0
    return min(1.0, (r / R) ** 2)


# ---- the case -------------------------------------------------------------------------------------------------
def _fmt(x) -> str:
    return f"{float(x):.10g}"


def _vec_s(v) -> str:
    return "(" + " ".join(_fmt(x) for x in v) + ")"


def region_cells(r: dict, C: np.ndarray) -> np.ndarray:
    """Cell indices of a region (cylinder or box, mm) from the cell centres C (mm)."""
    if "box" in r:
        b = np.asarray(r["box"], float)
        return np.nonzero(np.all((C >= b[:3]) & (C <= b[3:]), axis=1))[0]
    c = r["cylinder"]
    a = np.asarray(c.get("axis", [0, 0, 1]), float)
    a /= np.linalg.norm(a)
    d = C - np.asarray(c.get("center", [0, 0, 0]), float)
    s = d @ a
    rad = np.linalg.norm(d - np.outer(s, a), axis=1)
    s0, s1 = c.get("span", [-np.inf, np.inf])
    return np.nonzero((rad <= float(c["radius"])) & (rad >= float(c.get("inner_radius", 0) or 0))
                      & (s >= s0) & (s <= s1))[0]


def _phase_props(rho_l: float, mu_l: float) -> tuple[str, str]:
    water = ("thermoType\n{\n    type            heRhoThermo;\n    mixture         pureMixture;\n    transport       const;\n"
             "    thermo          eConst;\n    equationOfState rhoConst;\n    specie          specie;\n"
             "    energy          sensibleInternalEnergy;\n}\n\nmixture\n{\n    specie\n    {\n        molWeight   18.02;\n"
             f"    }}\n    equationOfState\n    {{\n        rho         {_fmt(rho_l)};\n    }}\n    thermodynamics\n    {{\n"
             "        Cp          4182;\n        Cv          4137;\n        hf          0;\n    }\n    transport\n    {\n"
             f"        mu          {_fmt(mu_l)};\n        Pr          7;\n    }}\n}}")
    air = ("thermoType\n{\n    type            heRhoThermo;\n    mixture         pureMixture;\n    transport       const;\n"
           "    thermo          hConst;\n    equationOfState perfectGas;\n    specie          specie;\n"
           "    energy          sensibleInternalEnergy;\n}\n\nmixture\n{\n    specie\n    {\n"
           f"        molWeight   {_fmt(AIR['molWeight'])};\n    }}\n    thermodynamics\n    {{\n        Cp          {_fmt(AIR['Cp'])};\n"
           f"        hf          0;\n    }}\n    transport\n    {{\n        mu          {_fmt(AIR['mu'])};\n"
           f"        Pr          {_fmt(AIR['Pr'])};\n    }}\n}}")
    return water, air


def phase_properties(cfg: dict) -> str:
    d = cfg["d_mm"] * SCALE
    return ("phases          (air water);\n\nreferencePhase  water;\n\n"
            "air\n{\n    type            pureIsothermalPhaseModel;\n\n    diameterModel\n    {\n        type            constant;\n"
            f"        d               {_fmt(d)};\n    }}\n\n    residualAlpha   1e-6;\n}}\n\n"
            "water\n{\n    type            pureIsothermalPhaseModel;\n\n    diameterModel\n    {\n        type            constant;\n"
            "        d               1e-4;\n    }\n\n    residualAlpha   1e-6;\n}\n\n"
            "blending\n{\n    default\n    {\n        type            linear;\n"
            "        minFullyContinuousAlpha.air 0.7;\n        minPartlyContinuousAlpha.air 0.3;\n"
            "        minFullyContinuousAlpha.water 0.7;\n        minPartlyContinuousAlpha.water 0.3;\n    }\n\n"
            "    segregated\n    {\n        type            linear;\n        minFullyContinuousAlpha.air 0.7;\n"
            "        minPartlyContinuousAlpha.air 0.5;\n        minFullyContinuousAlpha.water 0.7;\n"
            "        minPartlyContinuousAlpha.water 0.5;\n    }\n}\n\n"
            "surfaceTension\n{\n    air_water\n    {\n        type            constant;\n        sigma           0.072;\n    }\n}")


def momentum_transfer(cfg: dict) -> str:
    vm = ""
    if cfg["Cvm"] > 0:
        vm = (f"    air_dispersedIn_water\n    {{\n        type            constantCoefficient;\n        Cvm             {_fmt(cfg['Cvm'])};\n    }}\n\n"
              f"    water_dispersedIn_air\n    {{\n        type            constantCoefficient;\n        Cvm             {_fmt(cfg['Cvm'])};\n    }}\n")
    td = ("    air_dispersedIn_water\n    {\n        type            Burns;\n        sigma           0.7;\n    }\n"
          if cfg["turbulent_dispersion"] else "")
    return ("drag\n{\n    blending        segregated;\n\n"
            f"    air_dispersedIn_water\n    {{\n        type            {cfg['drag']};\n    }}\n\n"
            "    water_dispersedIn_air\n    {\n        type            SchillerNaumann;\n    }\n\n"
            "    air_segregatedWith_water\n    {\n        type            segregated;\n        m               0.5;\n"
            "        n               8;\n    }\n}\n\n"
            f"virtualMass\n{{\n{vm}}}\n\nlift\n{{}}\n\nwallLubrication\n{{}}\n\nturbulentDispersion\n{{\n{td}}}")


def write_euler(study, mesh, patches, case, plan, zones, g, q) -> dict:
    """The multiphaseEuler case after write_case has the zones, patches and inlet flows."""
    from .case import _dict, _field, _nonuniform, _uniform, nonortho_schemes, wall_velocity, write_mrf
    case = Path(case)
    cfg = settings(study)
    rho_l, mu_l = float(plan["rho"]), float(plan["mu"])
    turb = cfg["turbulent"]
    p0, T = cfg["p_ref"], cfg["T_K"]
    gvec = np.asarray(cfg["gravity"], float)
    gmag = float(np.linalg.norm(gvec))
    up = -gvec / gmag if gmag > 0 else np.array([0.0, 0.0, 1.0])
    C = g["C"]
    V = np.abs(g["V"]) * SCALE ** 3
    nc = mesh.ncells
    # initial phase fraction: liquid (with an initial gas fraction) below the fill level, air above
    ini = cfg["initial"]
    a0 = float(ini.get("gas_fraction", 0.0))
    alpha = np.full(nc, a0)
    if "fill_level_mm" in ini:
        alpha[(C @ up) > float(ini["fill_level_mm"])] = 1.0
    rho_g0 = p0 / (R_AIR * T)
    disp = alpha < 0.5
    plan_e = {"phases": ["air", "water"], "d_mm": cfg["d_mm"], "drag": cfg["drag"], "Cvm": cfg["Cvm"],
              "gravity": cfg["gravity"], "p_ref_Pa": p0, "T_K": T, "steady": cfg["steady"], "rho_l": rho_l,
              "mu_l": mu_l, "rho_g_ref": rho_g0, "air_dispersed0_kg": float((alpha[disp] * V[disp]).sum() * rho_g0),
              "air0_kg": float((alpha * V).sum() * rho_g0)}
    # inlets: the liquid's flow from the study's bc, the air's from the gas fraction or the air flow
    inlets = {p["name"]: plan["inlets"][p["name"]] for p in patches if p["kind"] == "inlet"}
    Ql = sum(float(v["Q"]) for v in inlets.values())
    if cfg["air_flow_lpm"] is not None:
        Qg_tot = float(cfg["air_flow_lpm"]) / 60000.0
        gas = Qg_tot / (Ql + Qg_tot) if Ql + Qg_tot > 0 else 0.0
    else:
        gas = cfg["gas_fraction"]
    plan_e.update(inlet_gas_fraction=gas, Q_liquid_m3s=Ql, Q_gas_m3s=Ql * gas / (1 - gas) if gas < 1 else 0.0)
    # regions -> cellZones tp_<name>
    regs = {}
    for r in cfg["regions"]:
        cells = region_cells(r, C)
        if not len(cells):
            from .polymesh import MeshError
            raise MeshError(f"two_phase.regions {r['name']!r} holds no cell centre")
        mesh.zones[f"tp_{r['name']}"] = cells
        regs[r["name"]] = {"cells": int(len(cells)), "volume_m3": float(V[cells].sum())}
    plan_e["regions"] = regs
    for p in patches:                          # where the open patches are (the height between inlet and outlet)
        if p["kind"] in ("inlet", "outlet", "opening"):
            f = mesh.patch_faces(p["name"])
            Af = np.linalg.norm(g["Sf"][f], axis=1)
            plan["patches"][p["name"]]["centroid_mm"] = [float(x) for x in (g["Cf"][f] * Af[:, None]).sum(0) / Af.sum()]
    plan["euler"] = plan_e
    plan["fixed_pressure"] = any(p["kind"] in ("outlet", "opening") for p in patches)
    # fields
    phases = ("air", "water")
    U = {ph: {} for ph in phases}
    A = {ph: {} for ph in phases}
    Tf = {ph: {} for ph in phases}
    K = {ph: {} for ph in phases}
    W = {ph: {} for ph in phases}
    NU = {ph: {} for ph in phases}
    AT = {ph: {} for ph in phases}
    pr, pp = {}, {}
    k0, w0 = 1e-4, 1.0
    kin = {}
    for nm, v in inlets.items():
        Um = float(v.get("U_mean", 1.0))
        kin[nm] = (max(1.5 * (0.05 * Um) ** 2, 1e-8), max(math.sqrt(max(1.5 * (0.05 * Um) ** 2, 1e-8)) /
                                                         (0.09 ** 0.25 * 0.07 * float(v.get("hydraulic_diameter_m", 0.05))), 1e-3))
    if kin:
        k0, w0 = max(x[0] for x in kin.values()), max(x[1] for x in kin.values())
    for p in patches:
        nm, kind, b = p["name"], p["kind"], p["bc"]
        if kind in ("empty", "symmetry"):
            for dct in [*U.values(), *A.values(), *Tf.values(), *K.values(), *W.values(), *NU.values(), *AT.values(), pr, pp]:
                dct[nm] = {"type": kind}
            continue
        pp[nm] = {"type": "calculated", "value": _uniform(p0)}
        if kind == "inlet":
            Qi = float(inlets[nm]["Q"])
            A["air"][nm] = {"type": "fixedValue", "value": _uniform(gas)}
            A["water"][nm] = {"type": "fixedValue", "value": _uniform(1 - gas)}
            U["water"][nm] = {"type": "flowRateInletVelocity", "volumetricFlowRate": _fmt(Qi), "alpha": "alpha.water",
                              "value": _uniform([0, 0, 0])}
            U["air"][nm] = ({"type": "flowRateInletVelocity", "volumetricFlowRate": _fmt(Qi * gas / (1 - gas)),
                             "alpha": "alpha.air", "value": _uniform([0, 0, 0])} if gas > 0 else
                            {"type": "fixedValue", "value": _uniform([0, 0, 0])})
            pr[nm] = {"type": "fixedFluxPressure", "value": _uniform(p0)}
            for ph in phases:
                Tf[ph][nm] = {"type": "fixedValue", "value": _uniform(T)}
                K[ph][nm] = {"type": "fixedValue", "value": _uniform(kin[nm][0])}
                W[ph][nm] = {"type": "fixedValue", "value": _uniform(kin[nm][1])}
        elif kind in ("outlet", "opening"):
            pg = float(b.get("pressure", 0.0))
            top = kind == "opening"
            # an outlet: the static pressure (a valve, a column's foot); an opening: the atmosphere above a free
            # surface, air drawn in where the flow enters (bubbleColumnLaminar's prghTotalPressure)
            pr[nm] = ({"type": "prghTotalPressure", "p0": _uniform(p0 + pg), "U": "U.air", "phi": "phi.air",
                       "rho": "rho.air", "value": _uniform(p0 + pg)} if top else
                      {"type": "prghPressure", "p": _uniform(p0 + pg), "value": _uniform(p0 + pg)})
            A["air"][nm] = {"type": "inletOutlet", "phi": "phi.air", "inletValue": _uniform(1.0 if top else 0.0),
                            "value": _uniform(1.0 if top else 0.0)}
            A["water"][nm] = {"type": "inletOutlet", "phi": "phi.water", "inletValue": _uniform(0.0 if top else 1.0),
                              "value": _uniform(0.0 if top else 1.0)}
            for ph in phases:
                U[ph][nm] = {"type": "pressureInletOutletVelocity", "phi": f"phi.{ph}", "value": _uniform([0, 0, 0])}
                Tf[ph][nm] = {"type": "inletOutlet", "phi": f"phi.{ph}", "inletValue": _uniform(T), "value": _uniform(T)}
                K[ph][nm] = {"type": "inletOutlet", "phi": f"phi.{ph}", "inletValue": _uniform(k0), "value": _uniform(k0)}
                W[ph][nm] = {"type": "inletOutlet", "phi": f"phi.{ph}", "inletValue": _uniform(w0), "value": _uniform(w0)}
        elif kind in ("wall", "slip"):
            pr[nm] = {"type": "fixedFluxPressure", "value": _uniform(p0)}
            for ph in phases:
                A[ph][nm] = {"type": "zeroGradient"}
                Tf[ph][nm] = {"type": "zeroGradient"}
            if kind == "slip":
                U["water"][nm] = U["air"][nm] = {"type": "slip"}
            else:
                uw = wall_velocity(p, mesh, g, zones, plan)
                U["water"][nm] = uw
                # the air slips along walls (aeratedStirredTankMRF: slip, MRFslip on the stirrer); a wall moving on its
                # own keeps its velocity
                U["air"][nm] = ({"type": "MRFslip"} if uw["type"] == "MRFnoSlip" else
                                {"type": "slip"} if uw["type"] == "noSlip" else uw)
            for ph in phases:
                if kind == "wall":
                    K[ph][nm] = {"type": "kqRWallFunction", "value": _uniform(k0)}
                    W[ph][nm] = {"type": "omegaWallFunction", "value": _uniform(w0)}
                    NU[ph][nm] = {"type": "nutkWallFunction", "value": _uniform(0)}
                    AT[ph][nm] = {"type": "compressible::alphatWallFunction", "Prt": "0.85", "value": _uniform(0)}
                else:
                    K[ph][nm] = {"type": "zeroGradient"}
                    W[ph][nm] = {"type": "zeroGradient"}
        else:
            from .polymesh import MeshError
            raise MeshError(f"two_phase: bc kind {kind} is not supported")
        for ph in phases:
            NU[ph].setdefault(nm, {"type": "calculated", "value": _uniform(0)})
            AT[ph].setdefault(nm, {"type": "calculated", "value": _uniform(0)})
    z = case / "0"
    for f in z.iterdir() if z.is_dir() else []:
        if f.is_file():
            f.unlink()
    z.mkdir(parents=True, exist_ok=True)
    # initial velocity: rest, or solid-body rotation with the (first) rotating zone (a validation's container)
    U0 = _uniform([0, 0, 0])
    if ini.get("velocity") == "rotation":
        # the rotating zone's, else the first wall turning on its own (a container spun without MRF: the same test
        # in the absolute frame)
        if plan.get("zones"):
            zp = plan["zones"][next(iter(plan["zones"]))]
            ax, om_s, ctr = zp["axis"], float(zp["omega_rad_s"]), zp["center_mm"]
        else:
            spec = next(p["bc"]["rotation"] for p in patches if isinstance(p["bc"].get("rotation"), dict))
            ax, om_s, ctr = spec["axis"], float(spec["rpm"]) * 2 * math.pi / 60, spec.get("center", [0, 0, 0])
        ax = np.asarray(ax, float)
        ax /= np.linalg.norm(ax)
        U0 = _nonuniform(np.cross(om_s * ax, (C - np.asarray(ctr, float)) * SCALE))
    for ph, af in (("air", alpha), ("water", 1.0 - alpha)):
        uniform_a = np.all(af == af[0])
        (z / f"alpha.{ph}").write_text(_field("volScalarField", f"alpha.{ph}", "[0 0 0 0 0 0 0]",
                                              _uniform(af[0]) if uniform_a else _nonuniform(af), A[ph]))
        (z / f"U.{ph}").write_text(_field("volVectorField", f"U.{ph}", "[0 1 -1 0 0 0 0]", U0, U[ph]))
        (z / f"T.{ph}").write_text(_field("volScalarField", f"T.{ph}", "[0 0 0 1 0 0 0]", _uniform(T), Tf[ph]))
        if turb:
            (z / f"k.{ph}").write_text(_field("volScalarField", f"k.{ph}", "[0 2 -2 0 0 0 0]", _uniform(k0), K[ph]))
            (z / f"omega.{ph}").write_text(_field("volScalarField", f"omega.{ph}", "[0 0 -1 0 0 0 0]", _uniform(w0), W[ph]))
            (z / f"nut.{ph}").write_text(_field("volScalarField", f"nut.{ph}", "[0 2 -1 0 0 0 0]", _uniform(0), NU[ph]))
            (z / f"alphat.{ph}").write_text(_field("volScalarField", f"alphat.{ph}", "[1 -1 -1 0 0 0 0]", _uniform(0), AT[ph]))
    (z / "p").write_text(_field("volScalarField", "p", "[1 -1 -2 0 0 0 0]", _uniform(p0), pp))
    (z / "p_rgh").write_text(_field("volScalarField", "p_rgh", "[1 -1 -2 0 0 0 0]", _uniform(p0), pr))
    # constant
    c = case / "constant"
    for f in ("physicalProperties", "momentumTransport", "phaseProperties", "fvModels"):
        (c / f).unlink(missing_ok=True)
    water, air = _phase_props(rho_l, mu_l)
    (c / "physicalProperties.water").write_text(_dict("physicalProperties.water", "constant", water))
    (c / "physicalProperties.air").write_text(_dict("physicalProperties.air", "constant", air))
    (c / "phaseProperties").write_text(_dict("phaseProperties", "constant", phase_properties(cfg)))
    (c / "momentumTransfer").write_text(_dict("momentumTransfer", "constant", momentum_transfer(cfg)))
    for ph, model in (("water", "kOmegaSSTSato"), ("air", "kOmegaSST")):
        mt = f"simulationType  RAS;\n\nRAS\n{{\n    model           {model};\n}}" if turb else "simulationType  laminar;"
        (c / f"momentumTransport.{ph}").write_text(_dict(f"momentumTransport.{ph}", "constant", mt))
    if turb:
        (c / "fvModels").write_text(_dict("fvModels", "constant", "\n\n".join(
            f"phaseTurbulenceStabilisation{ph.capitalize()}\n{{\n    type    phaseTurbulenceStabilisation;\n\n"
            f"    libs    (\"libmultiphaseEulerFvModels.so\");\n\n    phase   {ph};\n\n    alphaInversion  0.1;\n}}"
            for ph in phases)))
    (c / "g").write_text("FoamFile\n{\n    format      ascii;\n    class       uniformDimensionedVectorField;\n"
                         "    location    \"constant\";\n    object      g;\n}\n\ndimensions      [0 1 -2 0 0 0 0];\n"
                         f"value           {_vec_s(gvec)};\n")
    write_mrf(case, zones, plan)
    mesh.write(case)
    # system
    lap, sn, nonorth, _rf = nonortho_schemes(q)
    if q["max_non_orthogonality"] <= 5.0:
        lap, sn = "Gauss linear corrected", "corrected"
    steady = cfg["steady"]
    w_max = max((abs(zp["omega_rad_s"]) for zp in plan["zones"].values()), default=0.0)
    dt_cap = cfg["max_delta_t"] or (min(1.0, 0.2 / w_max) if w_max > 0 else (1.0 if steady else 0.01))
    funcs, mon = monitor_functions(study, plan, patches, cfg)
    plan_e["monitors"] = mon
    if steady:
        n = cfg["iterations"]
        cd_time = (f"startFrom       startTime;\n\nstartTime       0;\n\nstopAt          endTime;\n\nendTime         {n};\n\n"
                   f"deltaT          1;\n\nwriteControl    timeStep;\n\nwriteInterval   {max(1, n // 4)};\n\n")
    else:
        wi = float(cfg["write_interval"] or cfg["end_time"] / 10)
        cd_time = (f"startFrom       latestTime;\n\nstartTime       0;\n\nstopAt          endTime;\n\n"
                   f"endTime         {_fmt(cfg['end_time'])};\n\ndeltaT          {_fmt(min(1e-4, dt_cap))};\n\n"
                   f"writeControl    adjustableRunTime;\n\nwriteInterval   {_fmt(wi)};\n\nadjustTimeStep  yes;\n\n"
                   f"maxCo           {_fmt(cfg['max_co'])};\n\nmaxDeltaT       {_fmt(dt_cap)};\n\n")
    cd = ("solver          multiphaseEuler;\n\n" + cd_time +
          "purgeWrite      2;\n\nwriteFormat     ascii;\n\nwritePrecision  10;\n\nwriteCompression off;\n\n"
          "timeFormat      general;\n\ntimePrecision   8;\n\nrunTimeModifiable true;\n\n"
          "functions\n{\n" + "\n".join(funcs) + "\n}")
    sy = case / "system"
    (sy / "controlDict").write_text(_dict("controlDict", "system", cd))
    turb_div = ("    div(alphaRhoPhi,k)          Gauss limitedLinear 1;\n    div(alphaRhoPhi,omega)      Gauss limitedLinear 1;\n"
                if turb else "")
    fs = (f"ddtSchemes\n{{\n    default         {'localEuler' if steady else 'Euler'};\n}}\n\n"
          "gradSchemes\n{\n    default         Gauss linear;\n    limited         cellLimited Gauss linear 1;\n"
          "    grad(U)         $limited;\n    grad(k)         $limited;\n    grad(omega)     $limited;\n}\n\n"
          "divSchemes\n{\n    default                     none;\n\n    div(phi,alpha)              Gauss vanLeer;\n"
          "    div(phir,alpha,alpha)       Gauss vanLeer;\n    div(alphaRhoPhi,U)          Gauss linearUpwind limited;\n"
          "    div(phi,U)                  Gauss linearUpwind limited;\n    div(alphaRhoPhi,K)          Gauss limitedLinear 1;\n"
          "    div(alphaRhoPhi,(p|rho))    Gauss limitedLinear 1;\n" + turb_div +
          "    div((((alpha*rho)*nuEff)*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
          f"laplacianSchemes\n{{\n    default         {lap};\n}}\n\ninterpolationSchemes\n{{\n    default         linear;\n}}\n\n"
          f"snGradSchemes\n{{\n    default         {sn};\n}}\n\nwallDist\n{{\n    method meshWave;\n}}")
    (sy / "fvSchemes").write_text(_dict("fvSchemes", "system", fs))
    closed = not any(p["kind"] in ("inlet", "outlet", "opening") for p in patches)
    rU = float(cfg["relax"].get("U", 1.0))
    fields = "U|k|omega" if turb else "U"
    fv = ("solvers\n{\n    \"alpha.*\"\n    {\n        nCorrectors     " + ("1" if steady else "3") +
          ";\n        nSubCycles      " + ("1" if steady else "3") + ";\n\n        MULES\n        {\n"
          "            nIter           1;\n            globalBounds    yes;\n        }\n\n        MULESCorr       yes;\n"
          "        MULESCorrRelax  0.5;\n\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-10;\n        relTol          0;\n        minIter         1;\n    }\n\n"
          "    p_rgh\n    {\n        solver          GAMG;\n        smoother        DIC;\n        tolerance       1e-7;\n"
          "        relTol          0.01;\n    }\n\n    p_rghFinal\n    {\n        $p_rgh;\n        relTol          0;\n    }\n\n"
          "    \"pcorr.*\"\n    {\n        $p_rghFinal;\n        tolerance       1e-4;\n    }\n\n"
          f"    \"({fields}).*\"\n    {{\n        solver          PBiCGStab;\n        preconditioner  DILU;\n"
          "        tolerance       1e-7;\n        relTol          0;\n        minIter         1;\n    }\n}\n\n"
          "PIMPLE\n{\n    nOuterCorrectors    1;\n    nCorrectors         3;\n"
          f"    nNonOrthogonalCorrectors {nonorth};\n\n    correctPhi          yes;\n\n"
          f"    faceMomentum        {'yes' if cfg['face_momentum'] else 'no'};\n"
          "    VmDdtCorrection     yes;\n    dragCorrection      yes;\n"
          + (f"\n    maxCo               {_fmt(cfg['max_co'])};\n    maxDeltaT           {_fmt(dt_cap)};\n"
             "    rDeltaTSmoothingCoeff 0.02;\n" if steady else "")
          + (f"\n    pRefCell            0;\n    pRefValue           {_fmt(p0)};\n" if closed else "")
          + "}\n\nrelaxationFactors\n{\n    equations\n    {\n"
          f"        \"U.*\"           {_fmt(rU)};\n        \".*\"            1;\n    }}\n}}")
    (sy / "fvSolution").write_text(_dict("fvSolution", "system", fv))
    (sy / "fvConstraints").write_text(_dict("fvConstraints", "system",
                                            f"limitp\n{{\n    type       limitPressure;\n\n    min        {_fmt(max(2000.0, 0.02 * p0))};\n}}"))
    plan.update({"iterations": cfg["iterations"] if steady else 0, "residual": 1e-5, "relax": {"U": rU, "other": 1.0},
                 "schemes": "unstructured" if q["max_non_orthogonality"] > 5 else "orthogonal",
                 "laplacian": lap, "sngrad": sn, "nonorth": nonorth, "functions": funcs})
    (case / "case.foam").write_text("")
    return plan


def _fo(name: str, body: str) -> str:
    lines = "\n".join("        " + ln for ln in body.strip().splitlines())
    return f"    {name}\n    {{\n{lines}\n    }}"


def monitor_functions(study: dict, plan: dict, patches: list, cfg: dict) -> tuple[list, dict]:
    """Function objects writing every write_every steps: air mass and volume, flows per open patch, regions, moment,
    patch pressure. Returns (entries, {what: function object name})."""
    every = cfg["write_every"]
    wc = f"writeControl    timeStep;\nwriteInterval   {every};\nlog             false;"
    funcs, mon = [], {"patch_air": {}, "patch_water": {}, "patch_p": {}, "regions": {}, "moment": {}}
    funcs.append(_fo("tpAirMass", f"type            volFieldValue;\nlibs            (\"libfieldFunctionObjects.so\");\n"
                                  f"{wc}\nwriteFields     no;\ncellZone        all;\noperation       volIntegrate;\n"
                                  "weightField     rho.air;\nfields          (alpha.air);"))
    funcs.append(_fo("tpAirVolume", f"type            volFieldValue;\nlibs            (\"libfieldFunctionObjects.so\");\n"
                                    f"{wc}\nwriteFields     no;\ncellZone        all;\noperation       volIntegrate;\n"
                                    "fields          (alpha.air);"))
    mon["air_mass"], mon["air_volume"] = "tpAirMass", "tpAirVolume"
    for p in patches:
        if p["kind"] not in ("inlet", "outlet", "opening"):
            continue
        nm = p["name"]
        for key, field, fo in (("patch_air", "alphaRhoPhi.air", f"tpAirFlow_{nm}"),
                               ("patch_water", "alphaPhi.water", f"tpWaterFlow_{nm}")):
            funcs.append(_fo(fo, f"type            surfaceFieldValue;\nlibs            (\"libfieldFunctionObjects.so\");\n"
                                 f"{wc}\nwriteFields     false;\nsurfaceFormat   none;\npatch           {nm};\n"
                                 f"operation       sum;\nfields          ({field});"))
            mon[key][nm] = fo
        fo = f"tpP_{nm}"
        funcs.append(_fo(fo, f"type            surfaceFieldValue;\nlibs            (\"libfieldFunctionObjects.so\");\n"
                             f"{wc}\nwriteFields     false;\nsurfaceFormat   none;\npatch           {nm};\n"
                             "operation       areaAverage;\nfields          (p);"))
        mon["patch_p"][nm] = fo
    for r in cfg["regions"]:
        rn = r["name"]
        ent = {}
        for what, field, weight in (("alpha", "alpha.air", None), ("U_air", "U.air", "alpha.air"),
                                    ("U_water", "U.water", "alpha.water")):
            fo = f"tpR_{rn}_{what}"
            funcs.append(_fo(fo, f"type            volFieldValue;\nlibs            (\"libfieldFunctionObjects.so\");\n"
                                 f"{wc}\nwriteFields     no;\ncellZone        tp_{rn};\noperation       volAverage;\n"
                                 + (f"weightField     {weight};\n" if weight else "") + f"fields          ({field});"))
            ent[what] = fo
        mon["regions"][rn] = ent
    for zn, zp in (plan.get("zones") or {}).items():
        walls = list(zp.get("rotating_walls") or [])
        if not walls:
            continue
        o = np.asarray(zp["center_mm"], float) * SCALE
        for ph in ("water", "air"):
            fo = f"tpM_{zn}_{ph}"
            funcs.append(_fo(fo, f"type            forces;\nlibs            (\"libforces.so\");\n{wc}\n"
                                 f"patches         ({' '.join(walls)});\np               p;\nU               U.{ph};\n"
                                 f"phase           {ph};\nrho             rhoInf;\nrhoInf          1;\n"
                                 f"CofR            {_vec_s(o)};"))
            mon["moment"].setdefault(zn, {})[ph] = fo
    return funcs, mon


# ---- run and results --------------------------------------------------------------------------------------------
def _series(case: Path, fo: str, width: int = 1):
    """(times, values (n,) or (n, width)) of a function object's .dat file(s)."""
    from .average import read_monitor
    got = read_monitor(case, fo)
    if got is None:
        return None
    t, _h, v = got
    if v.ndim == 1:
        v = v[:, None]
    v = v[:, :width]
    return t, (v[:, 0] if width == 1 else v)


def _mean(t, v, t0):
    m = t >= t0
    if not m.any():
        m = np.ones_like(t, bool)
    return float(np.mean(v[m], axis=0)) if np.ndim(v) == 1 else np.mean(v[m], axis=0)


def air_balance(case: Path, plan: dict) -> dict | None:
    """d(air mass)/dt against the air's net inflow: max |M(t) - M(0) + integral of the net outflow| over the run, as a
    share of the air in play (the air injected, else the initial dispersed air)."""
    mon = plan["euler"]["monitors"]
    m = _series(case, mon["air_mass"])
    if m is None:
        return None
    tm, M = m
    net = np.zeros_like(tm)
    inj = np.zeros_like(tm)
    per = {}
    for nm, fo in mon["patch_air"].items():
        s = _series(case, fo)
        if s is None:
            continue
        f = np.interp(tm, s[0], s[1])
        net += f
        inj += np.minimum(f, 0.0)
        per[nm] = float(f"{np.mean(f[tm >= tm[len(tm) // 2]]):.4g}")
    # each written value is the step's own flux, over the step that ends at its time: a rectangle rule (the
    # trapezoid of 2026.11.20.3 mixed two steps' fluxes; the bubble column's top flux changes sign between steps)
    dt = np.diff(tm)
    cum = np.concatenate([[0.0], np.cumsum(net[1:] * dt)])
    injected = -np.concatenate([[0.0], np.cumsum(inj[1:] * dt)])
    err = (M - M[0]) + cum
    cum_tr = np.concatenate([[0.0], np.cumsum(0.5 * (net[1:] + net[:-1]) * dt)])
    err_tr = (M - M[0]) + cum_tr
    scale = max(float(injected[-1]), float(plan["euler"].get("air_dispersed0_kg") or 0.0), 1e-30)
    steady = plan["euler"]["steady"]
    out = {"error": float(f"{np.abs(err).max() / scale:.3g}"), "error_end": float(f"{err[-1] / scale:.3g}"),
           "air_in_play_kg": float(f"{scale:.4g}"), "air_mass_start_kg": float(f"{M[0]:.4g}"),
           "air_mass_end_kg": float(f"{M[-1]:.4g}"), "air_out_net_kg": float(f"{cum[-1]:.4g}"),
           "patch_air_kg_s_late": per,
           "error_of_all_air": float(f"{np.abs(err).max() / max(float(M[0]), 1e-30):.3g}"),
           "error_trapezoid": float(f"{np.abs(err_tr).max() / scale:.3g}"),
           "series": [[float(f"{tm[i]:.6g}"), float(f"{M[i] - M[0]:.4g}"), float(f"{-cum[i]:.4g}")]
                      for i in np.unique(np.linspace(0, len(tm) - 1, 20).astype(int))],
           "basis": ("pseudo-time (local time step): the time integral is over iterations of differing local steps, "
                     "so only the late net flow (in = out) is meaningful" if steady else
                     "M(t) - M(0) + time integral of the net air outflow; series: [t, M - M0, net inflow so far] (kg)")}
    if steady:                           # in = out at the end, relative to the air flow in
        f_in = -sum(min(v, 0.0) for v in per.values())
        if f_in > 0:
            out["error"] = float(f"{abs(sum(per.values())) / f_in:.3g}")
        elif per:                        # open but no air coming in: nothing to judge in pseudo-time
            out.update(error=None, note="steady with no air flowing in: in = out cannot be judged")
    if max(float(injected[-1]), float(plan["euler"].get("air_dispersed0_kg") or 0.0), float(np.abs(M).max())) < 1e-20:
        # no air in play (zero gas fraction in and at the start: the zero-air consistency check): nothing to balance
        out.update(error=None, applicable=False, note="no air in play: the balance does not apply")
    return out


def _moment_axis(case: Path, mon: dict, plan: dict, t0):
    out = {}
    from .average import _moment_series
    for zn, phs in mon["moment"].items():
        ax = np.asarray(plan["zones"][zn]["axis"], float)
        ax /= np.linalg.norm(ax)
        tot, ser = 0.0, None
        for ph, fo in phs.items():
            s = _moment_series(case, fo)
            if s is None:
                continue
            t, v = s
            mz = v @ ax
            tot += _mean(t, mz, t0)
            ser = (t, mz if ser is None else np.interp(t, ser[0], ser[1]) + mz)
        w = float(plan["zones"][zn]["omega_rad_s"])
        out[zn] = {"torque_Nm": float(f"{tot:.5g}"), "power_W": float(f"{abs(tot * w):.5g}"),
                   "omega_rad_s": w}
    return out


def results(case: Path, study: dict, plan: dict) -> tuple[dict, list]:
    case = Path(case)
    e = plan["euler"]
    mon = e["monitors"]
    warns = []
    cfg = settings(study)
    # the averaging window: the last half, or from average_from
    m = _series(case, mon["air_mass"])
    t_all = m[0] if m else np.array([0.0])
    t0 = float(cfg["average_from"]) if cfg["average_from"] is not None else float(t_all[0] + 0.5 * (t_all[-1] - t_all[0]))
    res = {"two_phase": {"model": "multiphaseEuler (two-fluid)", "bubble_diameter_mm": e["d_mm"], "drag": e["drag"],
                         "inlet_gas_fraction": e["inlet_gas_fraction"], "averaged_from": t0,
                         "time_or_iteration": float(t_all[-1])}}
    tp = res["two_phase"]
    bal = air_balance(case, plan)
    if bal:
        tp["air_mass_balance"] = bal
    flows = {}
    for nm, fo in mon["patch_water"].items():
        s = _series(case, fo)
        if s is not None:
            flows.setdefault(nm, {})["water_m3s"] = float(f"{_mean(s[0], s[1], t0):.5g}")
    for nm, fo in mon["patch_air"].items():
        s = _series(case, fo)
        if s is not None:
            flows.setdefault(nm, {})["air_kg_s"] = float(f"{_mean(s[0], s[1], t0):.5g}")
    tp["flows"] = flows
    regs = {}
    for rn, ent in mon["regions"].items():
        rec = {"volume_m3": e["regions"][rn]["volume_m3"]}
        a = _series(case, ent["alpha"])
        ua, uw = _series(case, ent["U_air"], 3), _series(case, ent["U_water"], 3)
        if a is not None:
            rec["gas_fraction"] = float(f"{_mean(a[0], a[1], t0):.5g}")
            rec["gas_fraction_series"] = [[float(f"{x:.6g}"), float(f"{y:.6g}")] for x, y in
                                          zip(a[0][:: max(1, len(a[0]) // 25)], a[1][:: max(1, len(a[0]) // 25)])]
        if ua is not None and uw is not None:
            va, vw = _mean(ua[0], ua[1], t0), _mean(uw[0], uw[1], t0)
            rec["U_air"] = [float(f"{x:.5g}") for x in va]
            rec["U_water"] = [float(f"{x:.5g}") for x in vw]
            rec["slip"] = [float(f"{x:.5g}") for x in (np.asarray(va) - np.asarray(vw))]
            k = max(1, len(uw[0]) // 25)
            rec["U_water_series"] = [[float(f"{t_:.5g}")] + [float(f"{x:.4g}") for x in v_]
                                     for t_, v_ in zip(uw[0][::k], uw[1][::k])]
        regs[rn] = rec
    tp["regions"] = regs
    av = _series(case, mon["air_volume"])
    if av is not None:
        tp["air_volume_series"] = [[float(f"{x:.6g}"), float(f"{y:.6g}")] for x, y in
                                   zip(av[0][:: max(1, len(av[0]) // 25)], av[1][:: max(1, len(av[0]) // 25)])]
    rot = _moment_axis(case, mon, plan, t0)
    if rot:
        res["rotating"] = rot
    # head: static pressures on inlet and outlet, the liquid's dynamic head, the height between them
    ins = [nm for nm, k in ((p, v["kind"]) for p, v in plan["patches"].items()) if k == "inlet"]
    outs = [nm for nm, k in ((p, v["kind"]) for p, v in plan["patches"].items()) if k in ("outlet",)]
    if ins and outs and mon["patch_p"]:
        try:
            si, so = _series(case, mon["patch_p"][ins[0]]), _series(case, mon["patch_p"][outs[0]])
            pin, pout = _mean(si[0], si[1], t0), _mean(so[0], so[1], t0)
            gv = np.asarray(e["gravity"], float)
            gm = float(np.linalg.norm(gv))
            dz = 0.0
            if gm > 0:
                up = -gv / gm
                ci = np.asarray(plan["patches"][ins[0]].get("centroid_mm", [0, 0, 0]), float) * SCALE
                co = np.asarray(plan["patches"][outs[0]].get("centroid_mm", [0, 0, 0]), float) * SCALE
                dz = float((co - ci) @ up)
            rl = e["rho_l"]
            ui = float(plan["inlets"][ins[0]].get("U_mean", 0.0))
            head = (pout - pin) / (rl * G0) + dz - ui ** 2 / (2 * G0)
            tp["pump"] = {"head_m": float(f"{head:.5g}"), "p_inlet_Pa": float(f"{pin:.6g}"),
                          "p_outlet_Pa": float(f"{pout:.6g}"), "dz_m": float(f"{dz:.4g}"),
                          "note": "head from the static pressures on the inlet and outlet (absolute, Pa) with the liquid's "
                                  "density, the height between them and the inlet's dynamic head; compare ratios with "
                                  "the single-phase head on the same mesh, not absolute heads"}
            if rot:
                zn = next(iter(rot))
                tp["pump"].update(torque_Nm=rot[zn]["torque_Nm"], power_W=rot[zn]["power_W"])
        except (TypeError, KeyError, IndexError):
            pass
    ref = cfg["reference"]
    if ref and "pump" in tp:
        cmp_ = reference_comparison(ref, tp["pump"])
        if cmp_:
            tp["reference"] = cmp_
            if e["inlet_gas_fraction"] == 0 and not cmp_.get("within_tolerance"):
                warns.append(f"two_phase at a gas fraction of 0 differs from the single-phase run {ref}: head "
                             f"{cmp_.get('head_ratio')}, torque {cmp_.get('torque_ratio')} (ratios; within "
                             f"{REF_TOL:.0%} expected): the two-fluid case is not consistent with it yet")
    if bal and bal.get("error") is not None and bal["error"] > BALANCE_TOL:
        warns.append(f"air mass balance off by {bal['error']:.2%} of the air in play (> {BALANCE_TOL:.0%})")
    res["checks"] = {"air_mass_balance": bal or {"error": None}}
    return res, warns


def reference_comparison(ref: str, pump: dict) -> dict | None:
    """Head and torque against a single-phase result ('<request>/<study>' under this machine's output root, or a
    folder): ratios and whether both are within REF_TOL."""
    import json
    from .api import out_root
    p = Path(ref).expanduser()
    if not p.is_absolute():
        p = out_root() / p
    f = p / f"{p.name}.json"
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        return {"error": f"no result {f}"}
    rp = ((r.get("results") or {}).get("pump") or {})
    rr = (r.get("results") or {}).get("rotating") or {}
    h1 = rp.get("head_m")
    t1 = next((z.get("torque_Nm") for z in rr.values()), None)
    out = {"study": str(p), "head_m": h1, "torque_Nm": t1}
    if h1 and pump.get("head_m") is not None:
        out["head_ratio"] = float(f"{pump['head_m'] / h1:.4g}")
    if t1 and pump.get("torque_Nm") is not None:
        out["torque_ratio"] = float(f"{pump['torque_Nm'] / t1:.4g}")
    if "head_ratio" in out and "torque_ratio" in out:
        out["within_tolerance"] = bool(abs(out["head_ratio"] - 1) <= REF_TOL and abs(out["torque_ratio"] - 1) <= REF_TOL)
    return out


def run_and_results(case: Path, study: dict, plan: dict, timeout: float, procs: int, warns: list,
                    t_job: float) -> dict:
    """The solve (one job: a transient run is asked to write and stop before the job's limit) and its results."""
    import time

    from . import foam
    steady = plan["euler"]["steady"]
    margin = min(900.0, max(60.0, 0.1 * timeout))
    info = foam.run(case, max(timeout - (time.time() - t_job), 120.0), procs,
                    deadline=None if steady else t_job + timeout - margin)
    lg = info["log"]
    solve = {"openfoam": info.get("openfoam"), "solver": "multiphaseEuler", "procs": info.get("procs"),
             "iterations": lg["iterations"], "residuals": lg["residuals"], "checkMesh": info["checkMesh"],
             "mode": "steady (local time step)" if steady else "transient"}
    if not steady:
        solve.update(reached_time=info.get("time"), end_time=settings(study)["end_time"],
                     reached_end=info.get("reached_end"))
        if not info.get("reached_end"):
            warns.append(f"the run stopped at t = {info.get('time')} s before end_time (one SLURM job): raise run.timeout_s")
    try:
        from .transient import step_profile
        sp = step_profile((Path(case) / "log.foamRun").read_text(errors="replace")[-400000:])
        if sp:
            solve["step_profile"] = sp
    except OSError:
        pass
    save_plan(case, plan)
    res, w = results(case, study, plan)
    warns += w
    why = verdict(res, plan)
    return {"solve": solve, "results": res, "unreliable_because": why, "info": info}


PLAN_FILE = "twophase_plan.json"


def save_plan(case: Path, plan: dict) -> None:
    """What results() needs (monitors, regions, patches, inlets, zones) beside the case, for reprocess."""
    import json
    keep = {k: plan.get(k) for k in ("euler", "patches", "inlets", "zones")}
    (Path(case).parent / PLAN_FILE).write_text(json.dumps(keep, default=float))


def verdict(res: dict, plan: dict) -> list:
    why = []
    bal = res["checks"]["air_mass_balance"]
    if bal.get("applicable") is False:
        pass
    elif bal.get("error") is None:
        why.append(f"no air mass balance ({bal.get('note') or 'the monitors were not written'})")
    elif bal["error"] > BALANCE_TOL:
        why.append(f"air mass balance off by {bal['error']:.2%}")
    ref = (res["two_phase"].get("reference") or {})
    if plan["euler"]["inlet_gas_fraction"] == 0 and ref and ref.get("within_tolerance") is False:
        why.append("head or torque differs from the single-phase reference by more than 2 % at a gas fraction of 0")
    return why


def reprocess(folder: Path, study: dict, old: dict) -> dict:
    """The two-phase results again from the case's monitors (runner.reprocess_study)."""
    import json
    folder = Path(folder)
    plan = json.loads((folder / PLAN_FILE).read_text())
    res, w = results(folder / "case", study, plan)
    why = verdict(res, plan)
    out = dict(old, results=res, reliable=not why)
    out.pop("unreliable_because", None)
    if why:
        out["unreliable_because"] = why
    out["warnings"] = [x for x in old.get("warnings", []) if not x.startswith("air mass balance")] + \
        [x for x in w if x not in old.get("warnings", [])]
    return out
