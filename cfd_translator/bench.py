"""bench.py — benchmark studies with known answers (closed form or published reference data).
Version: 2026.10.16.3

    run(solver, names, folder) -> {case: {passed, checks, ...}}

  couette_2d       plane Couette flow: Q = U H / 2, wall shear mu U / H (exact on any mesh)
  channel_2d       plane Poiseuille: dp/dx = 12 mu U / H^2, wall force = dp H (2D, parabolic inlet)
  cavity_re100     lid-driven cavity, Re 100: centreline velocities vs Ghia, Ghia & Shin (1982)
  pipe_poiseuille  Hagen-Poiseuille pipe: dp = 8 mu L Q / (pi R^4) (swept disk mesh; Gmsh)
  cylinder_2d1     Schaefer & Turek (1996) 2D-1, cylinder in a channel at Re 20: cD, dp (Gmsh)
  cavity_re1000    lid-driven cavity, Re 1000 vs Ghia (slow; OpenFOAM)
  pipe_turbulent   turbulent pipe, Re 5e4, kOmegaSST: friction factor vs Colebrook (slow; Gmsh)
  chain_plate      build123d-translator plate -> cfd-translator -> fea-translator: the FEA support
                   reaction equals minus the CFD force (needs Gmsh, b3dt and feat)
Slow cases run only when named. The reference solver runs the 2D box cases (couette, channel, cavity).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path


from .runner import run_study

# Ghia, Ghia & Shin (1982), Table I (u on x = 0.5) and Table II (v on y = 0.5)
GHIA_Y = [0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766]
GHIA_U = {100: [-0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332,
                0.23151, 0.68717, 0.73722, 0.78871, 0.84123],
          1000: [-0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080, 0.05702, 0.18719,
                 0.33304, 0.46604, 0.51117, 0.57492, 0.65928]}
GHIA_X = [0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688]
GHIA_V = {100: [0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                -0.10313, -0.08864, -0.07391, -0.05906],
          1000: [0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                 -0.39188, -0.33714, -0.27669, -0.21388]}


def _cavity(re: int, n: int) -> dict:
    L = 1000.0
    probes = [{"name": f"u{k}", "at": [0.0, (y - 0.5) * L, 50.0]} for k, y in enumerate(GHIA_Y)]
    probes += [{"name": f"v{k}", "at": [(x - 0.5) * L, 0.0, 50.0]} for k, x in enumerate(GHIA_X)]
    return {"geometry": {"box": [L, L, 100]}, "2d": True, "fluid": {"rho": 1.0, "mu": 1.0 / re},
            "bcs": [{"kind": "wall", "name": "lid", "faces": {"ny": 1}, "velocity": [1, 0, 0]}],
            "mesh": {"cells": [n, n, 1]}, "solve": {"iterations": 4000, "residual": 1e-6},
            "probes": probes}


def _cavity_checks(re, tol):
    def f(r):
        out = []
        for k, (y, u) in enumerate(zip(GHIA_Y, GHIA_U[re])):
            out.append(("u(y=%.4f)" % y, u, r["probes"][f"u{k}"]["ux"], {"abs": tol}))
        for k, (x, v) in enumerate(zip(GHIA_X, GHIA_V[re])):
            out.append(("v(x=%.4f)" % x, v, r["probes"][f"v{k}"]["uy"], {"abs": tol}))
        return out
    return f


def _cases() -> dict:
    C = {}
    # Couette: H = 0.1 m, U = 0.01 m/s, mu = 1e-3, depth 0.01 m, length 0.4 m
    C["couette_2d"] = {
        "study": {"geometry": {"box": [400, 100, 10]}, "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
                  "bcs": [{"kind": "wall", "name": "lid", "faces": {"ny": 1}, "velocity": [0.01, 0, 0]},
                          {"kind": "wall", "name": "bottom", "faces": {"ny": -1}},
                          {"kind": "opening", "name": "left", "faces": {"nx": -1}},
                          {"kind": "opening", "name": "right", "faces": {"nx": 1}}],
                  "mesh": {"cells": [20, 10, 1]}, "solve": {"residual": 1e-7}},
        "checks": lambda r: [("Q = U H / 2 x depth", 0.01 * 0.1 / 2 * 0.01, r["flow"]["left"]["Q"], {"tol": 0.005}),
                             ("bottom fx = mu U / H x area", 1e-3 * 0.01 / 0.1 * 0.4 * 0.01, r["forces"]["bottom"]["fx"], {"tol": 0.005}),
                             ("lid fx", -1e-3 * 0.01 / 0.1 * 0.4 * 0.01, r["forces"]["lid"]["fx"], {"tol": 0.005})],
        "ref": True, "needs": []}
    # plane Poiseuille: H = 0.1 m, L = 1 m, U = 0.01 m/s, mu = 1e-3
    mu, U, H = 1e-3, 0.01, 0.1
    C["channel_2d"] = {
        "study": {"geometry": {"box": [1000, 100, 10]}, "2d": True, "fluid": {"rho": 1.0, "mu": mu},
                  "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": U, "profile": "parabolic"},
                          {"kind": "outlet", "faces": {"nx": 1}}],
                  "mesh": {"cells": [100, 40, 1]}, "solve": {"residual": 1e-7},
                  "probes": [{"name": "a", "at": [-400, 0, 5]}, {"name": "b", "at": [400, 0, 5]}]},
        "checks": lambda r: [("dp over 0.8 m = 12 mu L U / H^2", 12 * mu * 0.8 * U / H ** 2,
                              r["probes"]["a"]["p"] - r["probes"]["b"]["p"], {"tol": 0.01}),
                             ("wall force = 12 mu U L depth / H", 12 * mu * U * 1.0 * 0.01 / H, r["forces"]["walls"]["fx"], {"tol": 0.01})],
        "ref": True, "needs": []}
    C["cavity_re100"] = {"study": _cavity(100, 64), "checks": _cavity_checks(100, 0.02), "ref": True, "needs": []}
    C["cavity_re1000"] = {"study": _cavity(1000, 128), "checks": _cavity_checks(1000, 0.03), "ref": False,
                          "needs": [], "slow": True}
    # Hagen-Poiseuille: R = 5 mm, water, 0.5 L/min; dp between z = 50 and 150 mm on the axis
    mu_w, Q, R = 1.002e-3, 0.5 / 60000, 0.005
    C["pipe_poiseuille"] = {
        "study": {"geometry": {"cylinder": [5, 200]}, "fluid": "water",
                  "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "flow_rate_lpm": 0.5, "profile": "parabolic"},
                          {"kind": "outlet", "faces": {"nz": 1}}],
                  "mesh": {"size": 0.5, "extrude": 100}, "solve": {"residual": 1e-7},
                  "probes": [{"name": "a", "at": [0, 0, 50]}, {"name": "b", "at": [0, 0, 150]}]},
        "checks": lambda r: [("dp over 0.1 m = 8 mu L Q / (pi R^4)", 8 * mu_w * 0.1 * Q / (math.pi * R ** 4),
                              r["probes"]["a"]["p"] - r["probes"]["b"]["p"], {"tol": 0.02}),
                             ("wall force = 8 pi mu L U_mean", 8 * math.pi * mu_w * 0.2 * Q / (math.pi * R ** 2),
                              r["forces"]["walls"]["fz"], {"tol": 0.02})],
        "ref": False, "needs": ["gmsh"]}
    # Schaefer & Turek 2D-1: channel 2.2 x 0.41 m, cylinder D = 0.1 m at (0.2, 0.2), mean inflow 0.2 m/s, nu = 1e-3
    C["cylinder_2d1"] = {
        "study": {"geometry": {"cylinder": [50, 10]}, "domain": {"kind": "external", "box": [-200, -200, 0, 2000, 210, 10]},
                  "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
                  "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 0.2, "profile": "parabolic"},
                          {"kind": "outlet", "faces": {"box": "xmax"}},
                          {"kind": "wall", "name": "channel", "faces": {"box": ["ymin", "ymax"]}},
                          {"kind": "wall", "name": "cylinder", "faces": {"all": True}}],
                  "mesh": {"size": 12, "refine": [{"at": [0, 0, 5], "r": 150, "size": 2.5}],
                           "layers": {"n": 8, "first": 0.25, "growth": 1.2}},
                  "solve": {"iterations": 3000, "residual": 1e-7},
                  "reference": {"velocity": 0.2, "area": 1000},
                  "probes": [{"name": "front", "at": [-50, 0, 5]}, {"name": "back", "at": [50, 0, 5]}]},
        "checks": lambda r: [("cD (5.57-5.59)", 5.5795, r["coefficients"]["cd"], {"tol": 0.02}),
                             ("cL (0.0104-0.0110)", 0.0106, r["coefficients"]["cl"], {"abs": 0.005}),
                             ("dp front-back (0.1172-0.1176 Pa)", 0.1174, r["probes"]["front"]["p"] - r["probes"]["back"]["p"], {"tol": 0.05})],
        "ref": False, "needs": ["gmsh"]}
    # turbulent pipe: D = 20 mm, Re 5e4 (water), dp between z = 600 and 1100 mm vs Colebrook (smooth)
    rho, D = 998.2, 0.02
    Ub = 5e4 * mu_w / (rho * D)
    f = 0.02
    for _ in range(50):
        f = (-2 * math.log10(2.51 / (5e4 * math.sqrt(f)))) ** -2
    C["pipe_turbulent"] = {
        "study": {"geometry": {"cylinder": [10, 1200]}, "fluid": "water", "model": "kOmegaSST",
                  "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": Ub, "intensity": 0.05},
                          {"kind": "outlet", "faces": {"nz": 1}}],
                  "mesh": {"size": 1.6, "extrude": 120}, "solve": {"iterations": 3000, "residual": 1e-6},
                  "probes": [{"name": "a", "at": [0, 0, 600]}, {"name": "b", "at": [0, 0, 1100]}]},
        "checks": lambda r: [("dp over 25 D = f (L/D) rho U^2 / 2 (Colebrook f = %.4f)" % f,
                              f * 25 * rho * Ub ** 2 / 2, r["probes"]["a"]["p"] - r["probes"]["b"]["p"], {"tol": 0.10})],
        "ref": False, "needs": ["gmsh"], "slow": True}
    C["chain_plate"] = {"study": None, "chain": True, "ref": False, "needs": ["gmsh", "build123d_translator", "fea_translator"]}
    return C


PLATE_GOAL = {"features": [{"kind": "base_box", "params": {"w": 4, "d": 40, "h": 40}}]}


def _chain(folder: Path, solver: str) -> dict:
    """b3dt plate (4 x 40 x 40 mm, standing on z = 0) in a slow viscous flow along X; its CFD face forces
    become fea-translator loads; with the plate's base fixed, the FEA reaction must equal minus the CFD force."""
    study = {"geometry": {"goal": PLATE_GOAL}, "domain": {"kind": "external", "box": [-80, -80, 0, 200, 80, 100]},
             "fluid": {"rho": 1000.0, "mu": 0.1}, "model": "laminar",
             "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 0.1},
                     {"kind": "outlet", "faces": {"box": "xmax"}},
                     {"kind": "slip", "name": "farfield", "faces": {"box": ["ymin", "ymax", "zmax", "zmin"]}},
                     {"kind": "wall", "name": "plate", "faces": {"all": True}}],
             "mesh": {"size": 8, "refine": [{"at": [0, 0, 20], "r": 40, "size": 2}]},
             "solve": {"iterations": 1500, "residual": 1e-6}}
    r = run_study("chain_plate", study, folder / "chain_plate", solver)
    if not r.get("ok"):
        return {"ok": False, "passed": False, "error": r.get("error"), "why": r.get("why")}
    loads = r["results"]["feat_loads"]["bcs"]
    from fea_translator import api as feat
    fstudy = {"part": {"goal": PLATE_GOAL}, "material": "steel", "analysis": "static", "mesh": {"size": 2},
              "bcs": [{"kind": "fixed", "name": "base", "faces": {"nz": -1}}] + loads}
    fr = feat.run({"plate": fstudy})
    s = fr["studies"]["plate"]
    if not s.get("ok") or "reactions" not in s:
        return {"ok": False, "passed": False, "error": s.get("error") or s.get("errors"), "cfd": r["results"]["forces"]["part"]}
    reac = s["reactions"]["base"]
    Fc = r["results"]["forces"]["part"]
    checks = []
    for ax in "xyz":
        exp = -Fc["f" + ax]
        got = reac["f" + ax]
        scale = max(abs(Fc["fx"]), abs(Fc["fy"]), abs(Fc["fz"]))
        checks.append({"what": f"FEA reaction f{ax} = -CFD force f{ax}", "expected": exp, "got": got,
                       "error": abs(got - exp) / scale, "passed": abs(got - exp) <= 0.01 * scale})
    return {"ok": True, "passed": all(c["passed"] for c in checks), "checks": checks, "cfd_force": Fc,
            "feat_loads": len(loads), "warnings": r.get("warnings", []) + s.get("warnings", [])}


def _available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def run(solver: str, names=None, folder: Path = Path(".")) -> dict:
    cases = _cases()
    if names:
        unknown = [n for n in names if n not in cases]
        if unknown:
            return {n: {"ok": False, "passed": False, "error": f"unknown benchmark; known: {list(cases)}"} for n in unknown}
    out = {}
    for n, c in cases.items():
        if names and n not in names:
            continue
        if not names and c.get("slow"):
            out[n] = {"skipped": "slow: run it by name"}
            continue
        if solver == "ref" and not c["ref"]:
            out[n] = {"skipped": "needs OpenFOAM (the reference solver runs the 2D box cases)"}
            continue
        missing = [m for m in c["needs"] if not _available(m)]
        if missing:
            out[n] = {"skipped": f"needs {', '.join(missing)}"}
            continue
        print(f"[cfdt bench] {n} ...", file=sys.stderr, flush=True)
        t0 = time.time()
        if c.get("chain"):
            out[n] = _chain(Path(folder), solver)
            _progress(n, out[n], t0)
            continue
        r = run_study(n, c["study"], Path(folder) / n, solver)
        if not r.get("ok"):
            out[n] = {"ok": False, "passed": False, "why": r.get("why"), "error": r.get("error") or r.get("errors")}
            _progress(n, out[n], t0)
            continue
        checks = []
        for what, exp, got, tol in c["checks"](r["results"]):
            if "abs" in tol:
                err = abs(got - exp)
                ok = err <= tol["abs"]
            else:
                err = abs(got - exp) / abs(exp)
                ok = err <= tol["tol"]
            checks.append({"what": what, "expected": round(exp, 8), "got": round(float(got), 8), "error": float(f"{err:.3g}"),
                           **tol, "passed": bool(ok)})
        out[n] = {"ok": True, "passed": all(c_["passed"] for c_ in checks), "checks": checks,
                  "solve": r.get("solve", {}).get("converged"), "cells": r["mesh"]["cells"],
                  "invariants": {k: r["results"]["checks"][k].get("error") for k in ("mass_balance", "momentum_balance")},
                  "warnings": r.get("warnings", []), "seconds": r.get("seconds")}
        _progress(n, out[n], t0, r)
    return out


def _progress(name, res, t0, r=None):
    """One line per case on stderr (stdout stays JSON)."""
    s = "PASS" if res.get("passed") else "FAIL" if res.get("ok") else "ERROR"
    extra = ""
    if r and r.get("ok"):
        sv = r.get("solve", {})
        extra = f" cells {r['mesh']['cells']}, {sv.get('iterations')} iterations, converged {sv.get('converged')}"
    worst = max((c.get("error", 0) for c in res.get("checks", [])), default=None)
    print(f"[cfdt bench] {name}: {s} in {time.time() - t0:.0f} s{extra}" + (f", largest check error {worst:.3g}" if worst is not None else "")
          + (f" ({str(res.get('error'))[:200]})" if s == "ERROR" else ""), file=sys.stderr, flush=True)
