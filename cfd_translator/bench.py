"""bench.py — benchmark studies with known answers (closed form or published reference data).
Version: 2026.10.18.13

    run(solver, names, folder) -> {case: {passed, checks, ...}}

  couette_2d       plane Couette flow: Q = U H / 2, wall shear mu U / H (exact on any mesh)
  channel_2d       plane Poiseuille: dp/dx = 12 mu U / H^2, wall force = dp H (2D, parabolic inlet)
  cavity_re100     lid-driven cavity, Re 100: centreline velocities vs Ghia, Ghia & Shin (1982)
  pipe_poiseuille  Hagen-Poiseuille pipe: dp = 8 mu L Q / (pi R^4) (swept disk mesh; Gmsh)
  cylinder_2d1     Schaefer & Turek (1996) 2D-1, cylinder in a channel at Re 20: cD, dp (Gmsh)
  taylor_couette_2d   rotating inner cylinder (rotating wall): exact torque and velocity (Gmsh)
  taylor_couette_mrf  the same with the inner region as a rotating zone (MRF): same answers, power = T omega
  swirl_pipe       solid-body swirl at an inlet: exact angular momentum inflow, angular momentum balance (Gmsh)
  pipe_poiseuille_mpi  pipe_poiseuille on 4 MPI ranks (solve.procs)
  conduction_slab_2d   energy: still air between plates at 60 and 20 C, q = k dT / H (exact)
  heated_channel_2d    energy: water, uniform wall heat flux, fully developed Nu = 8.235
  natural_cavity_ra1e4 energy + gravity: de Vahl Davis (1983) square cavity, mean Nu 2.243
  venturi_subsonic     energy, compressible: 2D venturi vs quasi-1D isentropic mass flow and throat pressure
                       (b3dt outline, Gmsh)
  pipe_air_low_mach    energy, air at Mach 0.0015: Hagen-Poiseuille as in the incompressible pipe (Gmsh)
  chain_thermal        CFD heat per face -> fea-translator thermal: a copper block comes out at the CFD wall
                       temperature (needs Gmsh, b3dt, feat)
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
    # Taylor-Couette (2D): inner cylinder R1 = 20 mm turning at 1 rad/s, outer R2 = 40 mm at rest, mu 0.01:
    # u_t = A r + B / r, torque per length 4 pi mu omega R1^2 R2^2 / (R2^2 - R1^2)
    R1, R2, w, mu_tc, depth = 0.02, 0.04, 1.0, 0.01, 0.01
    T_tc = 4 * math.pi * mu_tc * w * R1 ** 2 * R2 ** 2 / (R2 ** 2 - R1 ** 2) * depth
    A_tc, B_tc = -w * R1 ** 2 / (R2 ** 2 - R1 ** 2), w * R1 ** 2 * R2 ** 2 / (R2 ** 2 - R1 ** 2)
    rpm = w * 60 / (2 * math.pi)

    def tc(mrf: bool) -> dict:
        st = {"geometry": {"annulus": [20, 40, 10]}, "2d": True, "fluid": {"rho": 1000.0, "mu": mu_tc},
              "bcs": [{"kind": "wall", "name": "inner", "faces": {"cyl_r": 20},
                       "rotation": "rotor" if mrf else {"axis": [0, 0, 1], "rpm": rpm}},
                      {"kind": "wall", "name": "outer", "faces": {"cyl_r": 40}}],
              "mesh": {"size": 1.0}, "solve": {"iterations": 4000, "residual": 1e-7},
              "probes": [{"name": "mid", "at": [30, 0, 5]}]}
        if mrf:   # zone edge at 25 mm, off the probe: cells are in by centre, so the edge is jagged (mochi: the
            # probe on a 30 mm edge read u_t 7.8 % high)
            st["rotating"] = [{"name": "rotor", "axis": [0, 0, 1], "center": [0, 0, 0], "rpm": rpm, "radius": 25}]
        return st

    def tc_checks(r):
        return [("torque on the inner cylinder", -T_tc, r["forces"]["inner"]["mz"], {"tol": 0.02}),
                ("torque on the outer cylinder", T_tc, r["forces"]["outer"]["mz"], {"tol": 0.02}),
                ("u_t at r = 30 mm (A r + B / r)", A_tc * 0.03 + B_tc / 0.03, r["probes"]["mid"]["uy"], {"tol": 0.02}),
                ("angular momentum balance", 0.0, r["checks"]["angular_momentum_balance"]["error"], {"abs": 0.03})]
    C["taylor_couette_2d"] = {"study": tc(False), "checks": tc_checks, "ref": False, "needs": ["gmsh"]}
    C["taylor_couette_mrf"] = {"study": tc(True), "checks": lambda r: tc_checks(r) + [
        ("power to the fluid = torque x omega", T_tc * w, r["rotating"]["rotor"]["power_W"], {"tol": 0.02})],
        "ref": False, "needs": ["gmsh"]}
    # swirling inflow (solid body, 1 rad/s) into a pipe R = 10 mm at 0.01 m/s: angular momentum flux
    # rho U omega pi R^4 / 2 in, balanced by the wall torque and the outflow
    C["swirl_pipe"] = {
        "study": {"geometry": {"cylinder": [10, 200]}, "fluid": {"rho": 1000.0, "mu": 0.01},
                  "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": 0.01,
                           "swirl": {"axis": [0, 0, 1], "center": [0, 0, 0], "rpm": rpm}},
                          {"kind": "outlet", "faces": {"nz": 1}}],
                  "mesh": {"size": 1.0, "extrude": 40}, "solve": {"residual": 1e-7}},
        "checks": lambda r: [("angular momentum in = rho U omega pi R^4 / 2", 1000 * 0.01 * w * math.pi * 0.01 ** 4 / 2,
                              r["flow"]["inlet"]["angular_momentum_in"][2], {"tol": 0.02}),
                             ("flow rate unchanged by the swirl (U x meshed area)", 0.01 * r["flow"]["inlet"]["area_mm2"] * 1e-6,
                              r["flow"]["inlet"]["Q"], {"tol": 1e-5}),
                             ("angular momentum balance", 0.0, r["checks"]["angular_momentum_balance"]["error"], {"abs": 0.03})],
        "ref": False, "needs": ["gmsh"]}
    C["pipe_poiseuille_mpi"] = dict(C["pipe_poiseuille"], study=dict(C["pipe_poiseuille"]["study"],
                                    solve={"residual": 1e-7, "procs": 4}))
    # ---- energy (OpenFOAM fluid) ----
    from .format import thermo_props
    air = thermo_props({"fluid": "air"})
    water = thermo_props({"fluid": "water"})
    # conduction across still air between plates 10 mm apart at 60 and 20 C: q = k dT / H
    C["conduction_slab_2d"] = {
        "study": {"geometry": {"box": [20, 10, 1]}, "2d": True, "fluid": "air", "energy": True,
                  "bcs": [{"kind": "wall", "name": "hot", "faces": {"ny": -1}, "T": 60},
                          {"kind": "wall", "name": "cold", "faces": {"ny": 1}, "T": 20}],
                  "mesh": {"cells": [4, 20, 1]}, "solve": {"iterations": 3000, "residual": 1e-7},
                  "probes": [{"name": "mid", "at": [0, 0, 0.5]}]},
        "checks": lambda r: [("heat into the fluid at the hot wall = k dT / H x area", air["k"] * 40 / 0.01 * 0.02 * 0.001,
                              r["heat"]["hot"]["Q_W"], {"tol": 0.01}),
                             ("mid-plane temperature (linear profile)", 40.0, r["probes"]["mid"]["T"], {"abs": 0.2}),
                             ("energy balance", 0.0, r["checks"]["energy_balance"]["error"], {"abs": 0.01})],
        "ref": False, "needs": []}
    # water between plates 10 mm apart, both walls 100 W/m2, fully developed laminar: Nu = 8.235 (Dh = 2 H)
    Uw, Hc, qw, xw = 5e-4, 0.01, 100.0, 0.3
    Tb = 20 + 2 * qw * xw / (water["rho"] * Uw * Hc * water["cp"])
    C["heated_channel_2d"] = {
        "study": {"geometry": {"box": [400, 10, 1]}, "2d": True, "fluid": "water", "energy": True,
                  "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": Uw, "profile": "parabolic", "temperature": 20},
                          {"kind": "outlet", "faces": {"nx": 1}},
                          {"kind": "wall", "name": "heated", "faces": [{"ny": 1}, {"ny": -1}], "q": qw}],
                  "mesh": {"cells": [200, 20, 1]}, "solve": {"iterations": 4000, "residual": 1e-5},
                  "probes": [{"name": "wall", "at": [100, 5, 0.5]}]},
        "checks": lambda r: [("Nu at 300 mm (fully developed, both walls heated) = 8.235", 8.235,
                              qw * 2 * Hc / (water["k"] * (r["probes"]["wall"]["T"] - Tb)), {"tol": 0.03}),
                             ("outlet bulk temperature = inlet + heat / (m cp)",
                              20 + 2 * qw * 0.4 / (water["rho"] * Uw * Hc * water["cp"]), r["flow"]["outlet"]["T_mean_C"], {"abs": 0.05}),
                             ("energy balance", 0.0, r["checks"]["energy_balance"]["error"], {"abs": 0.01})],
        "ref": False, "needs": []}
    # de Vahl Davis (1983) square cavity, Ra = 1e4, air (Pr 0.71): mean Nu on the hot wall 2.243
    dT, Tm = 10.0, 293.15
    nu_a = air["mu"] / air["rho"]
    L = (1e4 * nu_a * (nu_a / air["pr"]) * Tm / (9.81 * dT)) ** (1 / 3)          # m
    Lmm = L * 1000
    C["natural_cavity_ra1e4"] = {
        "study": {"geometry": {"box": [Lmm, Lmm, Lmm / 10]}, "2d": True, "fluid": "air", "energy": True,
                  "gravity": [0, -9.81, 0],
                  "bcs": [{"kind": "wall", "name": "hot", "faces": {"nx": -1}, "T": 25},
                          {"kind": "wall", "name": "cold", "faces": {"nx": 1}, "T": 15}],
                  "mesh": {"cells": [48, 48, 1]}, "solve": {"iterations": 6000, "residual": 1e-5}},
        "checks": lambda r: [("mean Nu on the hot wall (de Vahl Davis 1983: 2.243)", 2.243,
                              r["heat"]["hot"]["q_mean"] * L / (air["k"] * dT), {"tol": 0.02}),
                             ("energy balance (hot in = cold out)", 0.0, r["checks"]["energy_balance"]["error"], {"abs": 0.01})],
        "ref": False, "needs": []}
    # subsonic venturi (2D, slip walls): quasi-1D isentropic flow from p0 = conditions + 10 kPa, T0 = 20 C
    g_ = air["gamma"]
    p0, T0, pe = 101325 + 10000, 293.15, 101325.0
    Me = math.sqrt(2 / (g_ - 1) * ((p0 / pe) ** ((g_ - 1) / g_) - 1))

    def area_ratio(M):     # A / A*
        return (1 / M) * ((2 / (g_ + 1)) * (1 + (g_ - 1) / 2 * M * M)) ** ((g_ + 1) / (2 * (g_ - 1)))
    Ae, At_ = 20e-3 * 2e-3, 14e-3 * 2e-3
    mdot_v = Ae * p0 * math.sqrt(g_ / (air["R"] * T0)) * Me * (1 + (g_ - 1) / 2 * Me * Me) ** (-(g_ + 1) / (2 * (g_ - 1)))
    target = At_ / (Ae / area_ratio(Me))
    lo, hi = 1e-4, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if area_ratio(mid) > target else (lo, mid)
    Mt = 0.5 * (lo + hi)
    pt_g = p0 * (1 + (g_ - 1) / 2 * Mt * Mt) ** (-g_ / (g_ - 1)) - 101325
    xs = list(range(-60, 81, 20))
    bump = [[x, -(10 - 3 * (1 + math.cos(math.pi * x / 80)) / 2)] for x in xs]
    top = [[x, 10 - 3 * (1 + math.cos(math.pi * x / 80)) / 2] for x in reversed(xs[:-1])] + [[-80, 10]]
    venturi = {"features": [{"kind": "profile_base", "params": {"h": 2, "outline": {
        "start": [-150, -10], "segs": [["L", -80, -10], ["S", bump], ["L", 150, -10], ["L", 150, 10], ["L", 80, 10],
                                       ["S", top], ["L", -150, 10], ["L", -150, -10]]}}}]}
    C["venturi_subsonic"] = {
        "study": {"geometry": {"goal": venturi}, "2d": True, "fluid": "air", "energy": True,
                  "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "total_pressure": 10000, "total_temperature": 20},
                          {"kind": "outlet", "faces": {"nx": 1}},
                          {"kind": "slip", "name": "walls_slip", "faces": {"all": True}}],
                  "mesh": {"size": 1.0}, "solve": {"iterations": 6000, "residual": 3e-5},
                  "probes": [{"name": "throat", "at": [0, 0, 1]}]},
        "checks": lambda r: [("mass flow (isentropic, M_exit %.3f)" % Me, mdot_v, r["flow"]["inlet"]["mdot"], {"tol": 0.02}),
                             ("throat pressure, gauge (M_throat %.3f)" % Mt, pt_g, r["probes"]["throat"]["p"], {"tol": 0.03}),
                             ("energy balance", 0.0, r["checks"]["energy_balance"]["error"], {"abs": 0.01})],
        "ref": False, "needs": ["gmsh", "build123d_translator"]}
    # low-Mach air pipe with energy on: must reproduce Hagen-Poiseuille (R 5 mm, 0.5 m/s mean)
    C["pipe_air_low_mach"] = {
        "study": {"geometry": {"cylinder": [5, 200]}, "fluid": "air", "energy": True,
                  "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": 0.5, "profile": "parabolic"},
                          {"kind": "outlet", "faces": {"nz": 1}}],
                  "mesh": {"size": 0.5, "extrude": 100}, "solve": {"residual": 1e-5},
                  "probes": [{"name": "a", "at": [0, 0, 50]}, {"name": "b", "at": [0, 0, 150]}]},
        "checks": lambda r: [("dp over 0.1 m = 8 mu L U / R^2", 8 * air["mu"] * 0.1 * 0.5 / 0.005 ** 2,
                              r["probes"]["a"]["p"] - r["probes"]["b"]["p"], {"tol": 0.02})],
        "ref": False, "needs": ["gmsh"]}
    C["chain_thermal"] = {"study": None, "chain": "thermal", "ref": False,
                          "needs": ["gmsh", "build123d_translator", "fea_translator"]}
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
             "mesh": {"size": 14, "refine": [{"at": [0, 0, 20], "r": 32, "size": 3}]},
             "solve": {"iterations": 1500, "residual": 1e-6}}
    import fea_translator
    v = tuple(int(x) for x in fea_translator.__version__.split("."))
    if v < (2026, 10, 16, 6):
        return {"ok": False, "passed": False, "error": f"fea-translator {fea_translator.__version__} is too old: 2026.10.16.6 or "
                "later is needed (earlier versions lose up to 10 % of distributed loads; git pull it)"}
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


BLOCK_GOAL = {"features": [{"kind": "base_box", "params": {"w": 20, "d": 20, "h": 20}}]}


def _chain_thermal(folder: Path, solver: str) -> dict:
    """A 20 mm block at 60 C on the floor of a 0.1 m/s, 20 C air stream: its CFD heat per face becomes
    fea-translator convection bcs (h per face, T_inf), and a heater under the block puts the same total heat
    in. A copper block must then come out at the CFD wall temperature, 60 C (one-way coupling consistent)."""
    study = {"geometry": {"goal": BLOCK_GOAL}, "domain": {"kind": "external", "box": [-60, -50, 0, 140, 50, 60]},
             "fluid": "air", "energy": True,
             "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 0.1, "temperature": 20},
                     {"kind": "outlet", "faces": {"box": "xmax"}},
                     {"kind": "slip", "name": "farfield", "faces": {"box": ["ymin", "ymax", "zmax", "zmin"]}},
                     {"kind": "wall", "name": "block", "faces": {"all": True}, "T": 60}],
             "mesh": {"size": 8, "refine": [{"at": [0, 0, 10], "r": 25, "size": 2}]},
             "solve": {"iterations": 3000, "residual": 1e-6}, "reference": {"T": 20}}
    r = run_study("chain_thermal", study, folder / "chain_thermal", solver)
    if not r.get("ok"):
        return {"ok": False, "passed": False, "error": r.get("error"), "why": r.get("why")}
    ft = r["results"]["feat_thermal"]
    from fea_translator import api as feat
    fstudy = {"part": {"goal": BLOCK_GOAL}, "material": "copper", "analysis": "thermal", "mesh": {"size": 2},
              "bcs": ft["convection"] + [{"kind": "heat", "name": "heater", "faces": {"nz": -1}, "power": ft["total_W"]}]}
    fr = feat.run({"block": fstudy})
    s = fr["studies"]["block"]
    if not s.get("ok"):
        return {"ok": False, "passed": False, "error": s.get("error") or s.get("errors"), "cfd_heat_W": ft["total_W"]}
    checks = []
    for key in ("T_max", "T_min"):
        got = s.get(key)
        checks.append({"what": f"FEA {key} = CFD wall temperature 60 C", "expected": 60.0, "got": got,
                       "error": None if got is None else abs(got - 60.0),
                       "passed": got is not None and abs(got - 60.0) <= 1.0})
    return {"ok": True, "passed": all(c["passed"] for c in checks), "checks": checks, "cfd_heat_W": ft["total_W"],
            "convection_bcs": len(ft["convection"]), "warnings": r.get("warnings", []) + s.get("warnings", [])}


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
    todo = []
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
        todo.append(n)
    # with CFDT_EXECUTOR=slurm the cases run as SLURM jobs (one request); the chains call feat and stay here
    queued = {}
    from . import executor
    if executor.kind() == "slurm":
        from . import api
        jobs = {n: dict(cases[n]["study"], queue=True) for n in todo if not cases[n].get("chain")}
        if jobs:
            req = api.start(jobs, solver)
            print(f"[cfdt bench] {len(jobs)} SLURM jobs in {req}", file=sys.stderr, flush=True)
            queued = api.status(str(req), 1e9)["studies"]
    for n in todo:
        c = cases[n]
        print(f"[cfdt bench] {n} ...", file=sys.stderr, flush=True)
        t0 = time.time()
        if c.get("chain"):
            out[n] = _chain_thermal(Path(folder), solver) if c["chain"] == "thermal" else _chain(Path(folder), solver)
            _progress(n, out[n], t0)
            continue
        r = queued[n] if n in queued else run_study(n, c["study"], Path(folder) / n, solver)
        if not r.get("ok"):
            out[n] = {"ok": False, "passed": False, "why": r.get("why"), "error": r.get("error") or r.get("errors")}
            _progress(n, out[n], t0)
            continue
        checks = []
        for what, exp, got, tol in c["checks"](r["results"]):
            if got is None:
                checks.append({"what": what, "expected": exp, "got": None, "passed": False})
                continue
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
