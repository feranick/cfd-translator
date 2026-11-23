"""esi_euler.py — two_phase on the ESI backend: OpenFOAM v2506's legacy multiphaseEulerFoam.
Version: 2026.11.23.1

Why: OpenFOAM 14's multiphaseEuler adds angular momentum to a solid-body rotation (+5 % in 1 s, water alone too), ESI's
reacting family loses 6-8 % (MRF unstable); ESI's legacy multiphaseEulerFoam (the 2013 code: constant-density phases,
transportProperties, one mixture turbulence model) holds it like single phase, laminar and with k-omega SST
(-0.44 / -0.51 % with MRF; mrf_hold and esi_legacy, carbonio, 20-22 Nov 2026). Selected with solve.backend 'esi' in a
two_phase study.

What it is and is not (errors() refuses the rest):
- transient only (two_phase.mode 'transient'): the solver has no local time step (localEuler fails: rSubDeltaT);
- MRF zones only (no sliding), laminar or kOmegaSST (on the mixture: k, omega, nut without a phase suffix);
- both phases at constant density: the air's at two_phase.p_ref_Pa and temperature_C (rho = p / (R T)); no turbulent
  dispersion (a note when the study asks for it); drag 'blended' Schiller-Naumann either way round; virtual mass;
- the mesh, the regions' cellZones, the initial fields and the boundary conditions are twophase.write_euler's, turned
  into the legacy dialect here: MRFnoSlip -> fixedValue (ESI's MRF turns the walls in its zones), MRFslip -> slip,
  flowRateInletVelocity without 'alpha' (both phases at Q / (1 - gas fraction) over the area), prghTotalPressure on
  the mixture (U, phi, rho); constant/{transportProperties, turbulenceProperties, MRFProperties (ESI form)};
  system/{controlDict (application multiphaseEulerFoam, ESI function objects), fvSchemes, fvSolution}; no
  fvConstraints (the legacy solver has none: no limitPressure, so checks.pressure_limited never fires).
- monitors: the air's volume (volFieldValue) and per open patch sum(alpha.air phi.air) by weightedSum (the legacy
  solver writes phi.air, not alphaRhoPhi.air), scaled by the air's constant density into kg (plan euler air_scale);
  the water's flow the same way; static pressure per open patch; region averages; the torque from a forces object on
  p / rho_l (an exprField pKin), so the pressure and viscous parts are both in N m.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SCALE = 1e-3
SOLVER = "multiphaseEulerFoam"
LABEL = "multiphaseEulerFoam (ESI v2506, legacy two-fluid)"


def is_legacy(study: dict) -> bool:
    return bool(study.get("two_phase")) and (study.get("solve") or {}).get("backend") == "esi"


def errors(st: dict) -> list[str]:
    """What the legacy solver cannot do (format.validate_study, through esi.errors and twophase.errors)."""
    tp = st.get("two_phase") or {}
    e = []
    why = "the ESI legacy multiphaseEulerFoam"
    if tp.get("mode", "steady") != "transient":
        e.append(f"solve.backend 'esi' with two_phase: {why} has no local time step: give two_phase.mode 'transient' "
                 "(end_time, max_co, write_interval)")
    if st.get("model", "laminar") not in ("laminar", "kOmegaSST"):
        e.append(f"solve.backend 'esi' with two_phase: model laminar or kOmegaSST ({why}: one mixture model)")
    if any(isinstance(z, dict) and z.get("motion") == "sliding" for z in st.get("rotating") or []):
        e.append(f"solve.backend 'esi' with two_phase: MRF zones only ({why})")
    if tp.get("face_momentum"):
        e.append(f"two_phase.face_momentum: OpenFOAM 14's multiphaseEuler only, not {why}")
    if tp.get("drag", "SchillerNaumann") != "SchillerNaumann":
        e.append(f"two_phase.drag: SchillerNaumann on {why} (blended both ways)")
    s = st.get("solve") or {}
    if s.get("transient"):
        e.append("solve.transient: use two_phase.mode 'transient' (two_phase studies keep their own time controls)")
    for key, what in (("energy", "energy"), ("cavitation", "cavitation"), ("porous", "porous zones"),
                      ("scalars", "scalars")):
        if st.get(key):
            e.append(f"solve.backend 'esi' with two_phase: {what} is not available")
    return e


def notes(st: dict) -> list[str]:
    tp = st.get("two_phase") or {}
    out = ["two-phase on ESI v2506's legacy multiphaseEulerFoam: both phases at constant density (the air's at "
           "p_ref_Pa), one mixture turbulence model, transient only"]
    if st.get("model") == "kOmegaSST" and tp.get("turbulent_dispersion", True):
        out.append("two_phase.turbulent_dispersion: not in the legacy solver (left out)")
    return out


# ---- dictionaries ------------------------------------------------------------------------------------------------
def _d(obj: str, loc: str, body: str) -> str:
    from .case import _dict
    return _dict(obj, loc, body)


def transport_properties(cfg: dict, rho_l: float, mu_l: float, rho_g: float, mu_g: float) -> str:
    d = cfg["d_mm"] * SCALE
    cvm = float(cfg.get("Cvm", 0.5) or 0.0)
    body = ("phases\n(\n"
            f"    air\n    {{\n        rho             {rho_g:.8g};\n        nu              {mu_g / rho_g:.8g};\n"
            "        kappa           0;\n        Cp              0;\n\n        diameterModel   constant;\n"
            f"        constantCoeffs\n        {{\n            d               {d:.8g};\n        }}\n    }}\n\n"
            f"    water\n    {{\n        rho             {rho_l:.8g};\n        nu              {mu_l / rho_l:.8g};\n"
            "        kappa           0;\n        Cp              0;\n\n        diameterModel   constant;\n"
            "        constantCoeffs\n        {\n            d               1e-4;\n        }\n    }\n);\n\n"
            "sigmas\n(\n);\n\ninterfaceCompression\n(\n);\n\n"
            "virtualMass\n(\n" + (f"    (air water)     {cvm:g}\n" if cvm > 0 else "") + ");\n\n"
            "drag\n(\n    (air water)\n    {\n        type blended;\n\n"
            "        air\n        {\n            type SchillerNaumann;\n            residualPhaseFraction 0;\n"
            "            residualSlip 0;\n        }\n\n"
            "        water\n        {\n            type SchillerNaumann;\n            residualPhaseFraction 0;\n"
            "            residualSlip 0;\n        }\n\n"
            "        residualPhaseFraction 1e-3;\n        residualSlip 1e-3;\n    }\n);\n\n"
            "// the mixture's turbulence model reads these (a dummy, as in the tutorials)\n"
            "transportModel  Newtonian;\nnu              0;")
    return _d("transportProperties", "constant", body)


def turbulence_properties(turb: bool) -> str:
    body = ("simulationType  RAS;\n\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n"
            "    printCoeffs     on;\n}") if turb else "simulationType  laminar;"
    return _d("turbulenceProperties", "constant", body)


def mrf_properties(plan: dict) -> str | None:
    zs = plan.get("zones") or {}
    if not zs:
        return None
    parts = []
    for name, zp in zs.items():
        o = np.asarray(zp["center_mm"], float) * SCALE
        a = zp["axis"]
        parts.append(f"{name}\n{{\n    cellZone        {name};\n    active          yes;\n    nonRotatingPatches ();\n"
                     f"    origin          ({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});\n"
                     f"    axis            ({a[0]:.12g} {a[1]:.12g} {a[2]:.12g});\n"
                     f"    omega           {float(zp['omega_rad_s']):.12g};\n}}")
    return _d("MRFProperties", "constant", "\n\n".join(parts))


def fv_schemes(turb: bool, lap: str, sn: str) -> str:
    fs = ("ddtSchemes\n{\n    default         Euler;\n}\n\n"
          "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
          "divSchemes\n{\n    default         Gauss limitedLinear 1;\n\n"
          "    \"div\\(phi,alpha.*\\)\"    Gauss vanLeer;\n"
          "    \"div\\(phir,alpha.*,alpha.*\\)\"   Gauss vanLeer;\n"
          "    \"div\\(alphaPhi.*,U.*\\)\" Gauss limitedLinearV 1;\n"
          "    div(Rc)                 Gauss linear;\n"
          "    \"div\\(phi.*,U.*\\)\"      Gauss limitedLinearV 1;\n"
          + ("    div(phi,k)              Gauss limitedLinear 1;\n    div(phi,omega)          Gauss limitedLinear 1;\n"
             "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n" if turb else "") +
          "}\n\n"
          f"laplacianSchemes\n{{\n    default         {lap};\n}}\n\n"
          "interpolationSchemes\n{\n    default         linear;\n}\n\n"
          f"snGradSchemes\n{{\n    default         {sn};\n}}\n\n"
          "wallDist\n{\n    method          meshWave;\n    correctWalls    false;\n}")
    return _d("fvSchemes", "system", fs)


def fv_solution(nonorth: int, closed: bool, p0: float) -> str:
    fv = ("solvers\n{\n    \"alpha.*\"\n    {\n        nAlphaSubCycles 2;\n    }\n\n"
          "    p_rgh\n    {\n        solver          GAMG;\n        smoother        DIC;\n        tolerance       1e-08;\n"
          "        relTol          0.01;\n    }\n\n"
          "    p_rghFinal\n    {\n        $p_rgh;\n        tolerance       1e-09;\n        relTol          0;\n    }\n\n"
          "    \"pcorr.*\"\n    {\n        solver          GAMG;\n        smoother        DIC;\n        tolerance       1e-05;\n"
          "        relTol          0;\n    }\n\n"
          "    \"(U|T|k|omega|Theta).*\"\n    {\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-06;\n        relTol          0;\n    }\n}\n\n"
          f"PIMPLE\n{{\n    nOuterCorrectors 1;\n    nCorrectors     2;\n    nNonOrthogonalCorrectors {max(0, int(nonorth))};\n"
          + (f"    pRefCell        0;\n    pRefValue       {p0:.8g};\n" if closed else "") +
          "}\n\nrelaxationFactors\n{\n    \"(U|T|alpha|Theta|k|omega).*\"   1;\n}")
    return _d("fvSolution", "system", fv)


def control_dict(cfg: dict, dt_cap: float, funcs: list[str]) -> str:
    wi = float(cfg["write_interval"] or cfg["end_time"] / 10)
    cd = (f"application     {SOLVER};\n\nstartFrom       latestTime;\n\nstartTime       0;\n\nstopAt          endTime;\n\n"
          f"endTime         {float(cfg['end_time']):.8g};\n\ndeltaT          {min(1e-4, dt_cap):.6g};\n\n"
          f"writeControl    adjustable;\n\nwriteInterval   {wi:.8g};\n\nadjustTimeStep  yes;\n\n"
          f"maxCo           {float(cfg['max_co']):g};\n\nmaxDeltaT       {dt_cap:.6g};\n\n"
          "purgeWrite      2;\n\nwriteFormat     ascii;\n\nwritePrecision  10;\n\nwriteCompression off;\n\n"
          "timeFormat      general;\n\ntimePrecision   8;\n\nrunTimeModifiable true;\n\n"
          "functions\n{\n" + "\n".join(funcs) + "\n}")
    return _d("controlDict", "system", cd)


def _fo(name: str, body: str) -> str:
    lines = "\n".join("        " + ln for ln in body.strip().splitlines())
    return f"    {name}\n    {{\n{lines}\n    }}"


def monitor_functions(plan: dict, patches: list, cfg: dict, rho_l: float, p0: float) -> tuple[list, dict]:
    """The legacy run's function objects in ESI form (names as twophase.monitor_functions, so its results read them)."""
    every = cfg["write_every"]
    wc = f"writeControl    timeStep;\nwriteInterval   {every};\nlog             false;\nerrors          warn;"
    lib = "libs            (fieldFunctionObjects);"
    funcs, mon = [], {"patch_air": {}, "patch_water": {}, "patch_p": {}, "regions": {}, "moment": {}}
    funcs.append(_fo("tpAirVolume", f"type            volFieldValue;\n{lib}\n{wc}\nwriteFields     false;\n"
                                    "regionType      all;\noperation       volIntegrate;\nfields          (alpha.air);"))
    mon["air_mass"] = mon["air_volume"] = "tpAirVolume"
    for p in patches:
        if p["kind"] not in ("inlet", "outlet", "opening"):
            continue
        nm = p["name"]
        for key, ph in (("patch_air", "air"), ("patch_water", "water")):
            fo = f"tp{'Air' if ph == 'air' else 'Water'}Flow_{nm}"
            funcs.append(_fo(fo, f"type            surfaceFieldValue;\n{lib}\n{wc}\nwriteFields     false;\n"
                                 f"surfaceFormat   none;\nregionType      patch;\nname            {nm};\n"
                                 f"operation       weightedSum;\nweightField     alpha.{ph};\nfields          (phi.{ph});"))
            mon[key][nm] = fo
        fo = f"tpP_{nm}"
        funcs.append(_fo(fo, f"type            surfaceFieldValue;\n{lib}\n{wc}\nwriteFields     false;\n"
                             f"surfaceFormat   none;\nregionType      patch;\nname            {nm};\n"
                             "operation       areaAverage;\nfields          (p);"))
        mon["patch_p"][nm] = fo
    for r in cfg["regions"]:
        rn = r["name"]
        ent = {}
        for what, field, weight in (("alpha", "alpha.air", None), ("U_air", "U.air", "alpha.air"),
                                    ("U_water", "U.water", "alpha.water")):
            fo = f"tpR_{rn}_{what}"
            funcs.append(_fo(fo, f"type            volFieldValue;\n{lib}\n{wc}\nwriteFields     false;\n"
                                 f"regionType      cellZone;\nname            tp_{rn};\n"
                                 + (f"operation       weightedVolAverage;\nweightField     {weight};\n" if weight else
                                    "operation       volAverage;\n") + f"fields          ({field});"))
            ent[what] = fo
        mon["regions"][rn] = ent
    walls_any = any((zp.get("rotating_walls") or []) for zp in (plan.get("zones") or {}).values())
    if walls_any:
        # p is absolute (Pa) in a constant-density two-phase case; forces wants a kinematic pressure with rhoInf
        funcs.append(_fo("tpPKin", "type            exprField;\nlibs            (fieldFunctionObjects);\n"
                                   "field           pKin;\naction          new;\nautowrite       false;\n"
                                   f"expression      \"(p - {p0:.10g})/{rho_l:.10g}\";\n"
                                   "executeControl  timeStep;\nwriteControl    none;\nerrors          warn;"))
    for zn, zp in (plan.get("zones") or {}).items():
        walls = list(zp.get("rotating_walls") or [])
        if not walls:
            continue
        o = np.asarray(zp["center_mm"], float) * SCALE
        fo = f"tpM_{zn}_mix"
        funcs.append(_fo(fo, f"type            forces;\nlibs            (forces);\n{wc}\n"
                             f"patches         ({' '.join(walls)});\np               pKin;\nU               U;\n"
                             f"rho             rhoInf;\nrhoInf          {rho_l:.10g};\n"
                             f"CofR            ({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});"))
        mon["moment"].setdefault(zn, {})["mix"] = fo
    return funcs, mon


# ---- the case: twophase.write_euler's, in the legacy dialect -------------------------------------------------------
def _set(entry: dict, **kw) -> dict:
    out = dict(entry)
    out.update(kw)
    return out


def fields(U: dict, gas: float, inlets: dict) -> dict:
    """U.air / U.water boundary entries for the legacy solver (see the module's notes)."""
    out = {}
    for ph, bcs in U.items():
        new = {}
        for nm, e in bcs.items():
            t = e.get("type")
            if t == "MRFnoSlip":
                e = {"type": "fixedValue", "value": "uniform (0 0 0)"}
            elif t == "MRFslip":
                e = {"type": "slip"}
            elif t == "flowRateInletVelocity":
                Qi = float(inlets[nm]["Q"])
                q = Qi / (1.0 - gas) if gas < 1 else Qi
                e = {"type": "flowRateInletVelocity", "volumetricFlowRate": f"{q:.10g}", "value": "uniform (0 0 0)"}
            elif t == "fixedValue" and nm in inlets:      # the air at zero gas fraction: the water's velocity
                Qi = float(inlets[nm]["Q"])
                e = {"type": "flowRateInletVelocity", "volumetricFlowRate": f"{Qi:.10g}", "value": "uniform (0 0 0)"}
            new[nm] = e
        out[ph] = new
    return out


def pressure(pr: dict) -> dict:
    out = {}
    for nm, e in pr.items():
        if e.get("type") == "prghTotalPressure":
            e = _set(e, U="U", phi="phi", rho="rho")
        out[nm] = e
    return out


def write_constant(case: Path, plan: dict, cfg: dict, rho_l: float, mu_l: float, rho_g: float, mu_g: float,
                   turb: bool) -> None:
    c = Path(case) / "constant"
    for f in ("phaseProperties", "momentumTransfer", "physicalProperties.water", "physicalProperties.air",
              "momentumTransport.water", "momentumTransport.air", "fvModels", "MRFProperties"):
        (c / f).unlink(missing_ok=True)
    (c / "transportProperties").write_text(transport_properties(cfg, rho_l, mu_l, rho_g, mu_g))
    (c / "turbulenceProperties").write_text(turbulence_properties(turb))
    m = mrf_properties(plan)
    if m:
        (c / "MRFProperties").write_text(m)
    from .esi import zone_headers
    zone_headers(c / "polyMesh")
    (Path(case) / "system" / "fvConstraints").unlink(missing_ok=True)


def k_omega_mixture(zdir: Path) -> None:
    """The water's k, omega, nut become the mixture's (k, omega, nut); the air's and alphat go."""
    for n in ("k", "omega", "nut"):
        src = zdir / f"{n}.water"
        if src.is_file():
            txt = src.read_text()
            txt = re.sub(rf"(object\s+){n}\.water;", rf"\g<1>{n};", txt)
            (zdir / n).write_text(txt)
            src.unlink()
        (zdir / f"{n}.air").unlink(missing_ok=True)
    for ph in ("air", "water"):
        (zdir / f"alphat.{ph}").unlink(missing_ok=True)
