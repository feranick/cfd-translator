"""bench.py — benchmark studies with known answers (closed form or published reference data).
Version: 2026.10.28.4

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
  mixing_layer_2d      tracer: two streams after a splitter plate vs 1/2 erfc(-y / (2 sqrt(D x / U))) (b3dt, Gmsh)
  tracer_two_streams   tracer: unequal streams, outlet mean = Q1 / (Q1 + Q2) (conservation)
  age_cavity_2d        mean age: channel with a side cavity, outlet mean age = V / Q (exact), dead zone
  porous_channel_1d    porous block (Darcy-Forchheimer) in a slip channel: dp = L (mu d U + rho f U^2 / 2) (exact)
  ergun_bed            the same from a packed bed's particle size and porosity: Ergun's equation
  porous_anisotropic   anisotropic block with axes at 45 degrees: momentum balance and transverse force (rotation)
  porous_slab_external a porous slab across a 2D external domain, the part (a post) slip: the part's drag is the
                       slab's reaction = dp x A (Gmsh)
  chain_thermal        CFD heat per face -> fea-translator thermal: a copper block comes out at the CFD wall
                       temperature (needs Gmsh, b3dt, feat)
  cavity_re1000    lid-driven cavity, Re 1000 vs Ghia (slow; OpenFOAM)
  pipe_turbulent   turbulent pipe, Re 5e4, kOmegaSST: friction factor vs Colebrook (slow; Gmsh)
  chain_plate      build123d-translator plate -> cfd-translator -> fea-translator: the FEA support
                   reaction equals minus the CFD force (needs Gmsh, b3dt and feat)
  sphere_snappy_re100  snappyHexMesh with layers: sphere at Re 100, Cd vs the standard drag curve (b3dt, Gmsh)
  sphere_half_snappy   the same as a half model on a symmetry plane (the cut face meshed by snappy.cut_at_box)
  pipe_snappy          internal flow on a snappyHexMesh mesh: Hagen-Poiseuille as pipe_poiseuille
  cylinder_2d2         transient (laminar, Re 100): Schaefer & Turek 2D-2 vortex shedding, cD max, cL max, Strouhal
  cylinder_2d2_restart the same in 2-minute SLURM jobs: continues from job to job, same answers
  ahmed25_des_mg, ahmed35_des_mg   DES (kOmegaSSTDES), 40 m/s, moving ground, L3 mesh, 0.4 s (slow; by name)
  ahmed25_snappy, ahmed35_snappy, ahmed35_des   the vehicle-aero study's cases (60 m/s, stationary floor): steady L1,
                       DES L2; Cd vs the tunnel (10 / 15 / 12 %) and vs cfdt 2026.10.22.10's values (3 / 3 / 4 %) (slow)
  ahmed25_L1..L3, ahmed35_L1..L3   Ahmed body (half, moving ground, kOmegaSST, layers at y+ ~60), three meshes:
                       Cd vs the wind tunnel (0.285, 0.26) within 10 %, y+ 30-300, layer coverage (slow; by name)
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
    # ---- tracers and age (scalars.py) ----
    # two streams split by a thin plate (x < 0, y = 0), slip walls everywhere: uniform flow, so the tracer layer
    # downstream of the plate's end is the error function c = 1/2 erfc(-y / (2 sqrt(D x / U)))
    split = {"features": [{"kind": "profile_base", "params": {"h": 1, "outline": {"start": [-20, -10], "segs": [
        ["L", 200, -10], ["L", 200, 10], ["L", -20, 10], ["L", -20, 0.25], ["L", 0, 0.25], ["L", 0, -0.25],
        ["L", -20, -0.25], ["L", -20, -10]]}}}]}

    def two_streams(u_top, u_bot, extra=None):
        st = {"geometry": {"goal": split}, "2d": True, "fluid": "water", "model": "laminar",
              "bcs": [{"kind": "inlet", "name": "top", "faces": {"nx": -1, "at": [-20, 5, 0.5]}, "velocity": u_top},
                      {"kind": "inlet", "name": "bottom", "faces": {"nx": -1, "at": [-20, -5, 0.5]}, "velocity": u_bot},
                      {"kind": "outlet", "faces": {"nx": 1}},
                      {"kind": "slip", "name": "walls_slip", "faces": {"all": True}}],
              "scalars": [{"name": "dye", "inlets": {"top": 1}, "diffusivity": 1e-6}],
              "mesh": {"size": 0.5}, "solve": {"iterations": 2000, "residual": 1e-7}}
        st.update(extra or {})
        return st
    Dm, Um, xm = 1e-6, 0.01, 0.1
    ys = [-4, -2, 0, 2, 4]
    erf_c = {y: 0.5 * math.erfc(-y * 1e-3 / (2 * math.sqrt(Dm * xm / Um))) for y in ys}
    C["mixing_layer_2d"] = {
        "study": two_streams(Um, Um, {"probes": [{"name": f"y{y}".replace("-", "m"), "at": [100, y, 0.5]} for y in ys]}),
        "checks": lambda r: [(f"c at x = 100 mm, y = {y} mm (erfc)", erf_c[y], r["probes"][f"y{y}".replace("-", "m")]["dye"],
                              {"abs": 0.02}) for y in ys]
        + [("outlet mean = 1/2 (equal flows; conservation)", 0.5, r["scalars"]["dye"]["outlets"]["outlet"]["mean"], {"abs": 1e-3}),
           ("tracer balance", 0.0, r["checks"]["scalar_balance"]["dye"]["error"], {"abs": 1e-3})],
        "ref": False, "needs": ["gmsh", "build123d_translator"]}
    C["tracer_two_streams"] = {
        "study": two_streams(0.02, 0.01),
        "checks": lambda r: [("outlet mean = Q_top / (Q_top + Q_bottom) = 2/3", 2 / 3,
                              r["scalars"]["dye"]["outlets"]["outlet"]["mean"], {"abs": 1e-3}),
                             ("tracer balance", 0.0, r["checks"]["scalar_balance"]["dye"]["error"], {"abs": 1e-3})],
        "ref": False, "needs": ["gmsh", "build123d_translator"]}
    # mean age in a channel with a side cavity (laminar, Re 100): outlet mean age = V / Q exactly (any flow);
    # the cavity holds old fluid
    cav = {"features": [{"kind": "profile_base", "params": {"h": 1, "outline": {"start": [0, 0], "segs": [
        ["L", 40, 0], ["L", 40, -20], ["L", 60, -20], ["L", 60, 0], ["L", 100, 0], ["L", 100, 10], ["L", 0, 10],
        ["L", 0, 0]]}}}]}
    tau = (100 * 10 + 20 * 20) * 1e-9 / (0.01 * 10e-6)
    C["age_cavity_2d"] = {
        "study": {"geometry": {"goal": cav}, "2d": True, "fluid": "water", "model": "laminar",
                  "bcs": [{"kind": "inlet", "faces": {"nx": -1, "at": [0, 5, 0.5]}, "velocity": 0.01, "profile": "parabolic"},
                          {"kind": "outlet", "faces": {"nx": 1, "at": [100, 5, 0.5]}}],
                  "age": True, "mesh": {"size": 0.5}, "solve": {"iterations": 3000, "residual": 1e-7}},
        "checks": lambda r: [("outlet mean age = V / Q = %.2f s" % tau, tau, r["age"]["mean_residence_time_s"], {"tol": 0.01}),
                             ("age balance (outflow + back-diffusion = V)", 0.0, r["checks"]["age_balance"]["error"], {"abs": 1e-3}),
                             ("the cavity holds fluid older than 2 V/Q", 1.0,
                              1.0 if r["age"]["volume_fraction_older_than_2tau"] > 0.05 else 0.0, {"abs": 0.5})],
        "ref": False, "needs": ["gmsh", "build123d_translator"]}
    # ---- porous zones (porous.py): a block in a slip channel, uniform flow, so dp = L (mu d U + rho f U^2 / 2) ----
    mu_w, rho_w, Up, Lp = 1e-3, 1000.0, 0.01, 0.05

    def porous_channel(zone, extra=None):
        st = {"geometry": {"box": [200, 10, 1]}, "2d": True, "fluid": {"rho": rho_w, "mu": mu_w}, "model": "laminar",
              "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": Up}, {"kind": "outlet", "faces": {"nx": 1}},
                      {"kind": "slip", "faces": [{"ny": 1}, {"ny": -1}]}],
              "porous": [dict({"name": "bed", "region": {"box": [-25, -10, -1, 25, 10, 2]}}, **zone)],
              "mesh": {"cells": [400, 4, 1]}, "solve": {"iterations": 3000, "residual": 1e-8}}
        st.update(extra or {})
        return st
    d0, f0 = 1e8, 1e3
    dp0 = Lp * (mu_w * d0 * Up + 0.5 * rho_w * f0 * Up ** 2)
    C["porous_channel_1d"] = {
        "study": porous_channel({"darcy": d0, "forchheimer": f0}),
        # 1 %: OpenFOAM's collocated grid makes the cell velocity dip in the first and last cells of the zone (mochi,
        # 2026.10.19.1: -2.6 % with 2 of 20 cells at the faces; 100 cells here)
        "checks": lambda r: [("dp across the block = L (mu d U + rho f U^2 / 2)", dp0, r["porous"]["bed"]["pressure_drop_Pa"], {"tol": 0.01}),
                             ("inlet to outlet dp (no other loss)", dp0, r["pressure_drop"], {"tol": 0.01}),
                             ("block force on the fluid = -dp A", -dp0 * 1e-5, r["porous"]["bed"]["force_on_fluid_N"][0], {"tol": 0.01}),
                             ("momentum balance with the porous force", 0.0, r["checks"]["momentum_balance"]["error"], {"abs": 1e-3})],
        "ref": False, "needs": []}
    dpe, eps = 1e-3, 0.4
    ergun = Lp * (150 * mu_w * (1 - eps) ** 2 * Up / (eps ** 3 * dpe ** 2) + 1.75 * rho_w * (1 - eps) * Up ** 2 / (eps ** 3 * dpe))
    C["ergun_bed"] = {
        "study": porous_channel({"ergun": {"particle_diameter": 1.0, "porosity": eps}}),
        "checks": lambda r: [("dp = Ergun (dp 1 mm, porosity 0.4)", ergun, r["porous"]["bed"]["pressure_drop_Pa"], {"tol": 0.01})],
        "ref": False, "needs": []}
    # anisotropic block with its axes at 45 degrees. The flow turns inside it (2D near its ends), so no closed form;
    # the momentum balance closes only if OpenFOAM's rotated resistance equals ours (porous.momentum_source), and the
    # transverse force points to -y (more resistance along e1 = (1, 1)/sqrt 2)
    d1, d2 = 1e8, 1e7
    s2 = math.sqrt(0.5)
    C["porous_anisotropic"] = {
        "study": porous_channel({"darcy": [d1, d2, d2], "axes": [[s2, s2, 0], [-s2, s2, 0]]}),
        "checks": lambda r: [("momentum balance with the rotated resistance", 0.0, r["checks"]["momentum_balance"]["error"], {"abs": 2e-3}),
                             ("transverse force / axial force (sign and size; 1D estimate (d1-d2)/(d1+d2) = 0.82)", 0.82,
                              r["porous"]["bed"]["force_on_fluid_N"][1] / r["porous"]["bed"]["force_on_fluid_N"][0], {"tol": 0.25})],
        "ref": False, "needs": []}
    # a porous slab across a 2D external domain (slip sides, so the flow through it is uniform); the part, a post
    # downstream, is slip, so the part's drag is the slab's reaction alone (2026.10.28.2): dp x A
    dps = 0.05 * (1e-3 * 1e6 * 1.0 + 0.5 * 1.0 * 100 * 1.0 ** 2)          # 52.5 Pa; A = 200 x 10 mm
    C["porous_slab_external"] = {
        "study": {"geometry": {"cylinder": [5, 10]}, "domain": {"kind": "external", "box": [-400, -100, 0, 400, 100, 10]},
                  "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3}, "model": "laminar",
                  "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 1.0},
                          {"kind": "outlet", "faces": {"box": "xmax"}},
                          {"kind": "slip", "name": "sides", "faces": {"box": ["ymin", "ymax"]}},
                          {"kind": "slip", "name": "post", "faces": {"all": True}}],
                  "porous": [{"name": "slab", "region": {"box": [-250, -110, -1, -200, 110, 11]}, "darcy": 1e6,
                              "forchheimer": 100}],
                  "mesh": {"size": 10, "refine": [{"box": [-270, -100, 0, -180, 100, 10], "size": 2.5}]},
                  "solve": {"iterations": 3000, "residual": 1e-7},
                  "reference": {"velocity": 1.0, "area": 2000}},
        "checks": lambda r: [("part drag = the slab's dp x A (2 %)", r["porous"]["slab"]["pressure_drop_Pa"] * 2e-3,
                              r["forces"]["part"]["fx"], {"tol": 0.02}),
                             ("part drag = L (mu d U + rho f U^2 / 2) x A", dps * 2e-3, r["forces"]["part"]["fx"], {"tol": 0.05}),
                             ("walls only: nothing (the post is slip)", 0.0, r["coefficients"]["cd_walls_only"], {"abs": 1e-6}),
                             ("the slab's faces (pressure, momentum) give its force", 0.0,
                              r["porous"]["slab"]["face_balance"]["vs_force"], {"abs": 0.03}),
                             ("momentum balance with the porous force", 0.0, r["checks"]["momentum_balance"]["error"], {"abs": 1e-3})],
        "ref": False, "needs": ["gmsh"]}
    C["chain_thermal"] = {"study": None, "chain": "thermal", "ref": False,
                          "needs": ["gmsh", "build123d_translator", "fea_translator"]}
    C["chain_plate"] = {"study": None, "chain": True, "ref": False, "needs": ["gmsh", "build123d_translator", "fea_translator"]}
    # ---- snappyHexMesh (snappy.py) ----
    # sphere, D = 10 mm, Re 100 (water at 0.01 m/s): Cd 1.09 (standard drag curve, Clift, Grace & Weber 1978:
    # 24/Re (1 + 0.15 Re^0.687) + 0.42 / (1 + 42500 Re^-1.16) = 1.095); 10 D wide domain adds ~2 % blockage
    cd_sph = 24 / 100 * (1 + 0.15 * 100 ** 0.687) + 0.42 / (1 + 42500 * 100 ** -1.16)

    def sphere(half: bool):
        goal = {"features": [{"kind": "base_sphere", "params": {"r": 5}}]}
        if half:
            goal["features"].append({"kind": "split", "params": {"nx": 0, "ny": 1, "nz": 0, "off": 0}})
        y0 = 0 if half else -50
        far = ["ymax", "zmin", "zmax"] + ([] if half else ["ymin"])
        bcs = [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 0.01},
               {"kind": "outlet", "faces": {"box": "xmax"}},
               {"kind": "slip", "name": "farfield", "faces": {"box": far}},
               {"kind": "wall", "name": "sphere", "faces": {"all": True}}]
        if half:
            bcs.insert(3, {"kind": "symmetry", "name": "mid", "faces": {"box": "ymin"}})
        return {"geometry": {"goal": goal}, "domain": {"kind": "external", "box": [-50, y0, -45, 150, 50, 55]},
                "fluid": {"rho": 1000.0, "mu": 1e-3}, "model": "laminar", "bcs": bcs,
                "reference": {"velocity": 0.01, "area": math.pi * 25 / (2 if half else 1)},
                "mesh": {"mesher": "snappy", "size": 10, "surface": 0.625,
                         "refine": [{"distance": 4, "size": 1.25}, {"box": [-10, y0 if half else -12, -7, 60, 12, 17], "size": 2.5}],
                         "layers": {"n": 3, "first": 0.15, "growth": 1.2}},
                "solve": {"iterations": 2000, "residual": 1e-6}}
    C["sphere_snappy_re100"] = {
        "study": sphere(False),
        "checks": lambda r: [("Cd at Re 100 (standard drag curve %.3f)" % cd_sph, cd_sph, r["coefficients"]["cd"], {"tol": 0.06}),
                             ("layers on the sphere (coverage, %)", 100.0, r["_mesh"]["layers"]["coverage_pct"], {"tol": 0.3})],
        "ref": False, "needs": ["gmsh", "build123d_translator"]}
    C["sphere_half_snappy"] = {
        "study": sphere(True),
        "checks": lambda r: [("Cd of the half sphere on a symmetry plane (%.3f)" % cd_sph, cd_sph, r["coefficients"]["cd"],
                              {"tol": 0.06}),
                             ("no lift across the symmetry plane", 0.0, r["forces"]["sphere"]["fz"] / max(abs(r["forces"]["sphere"]["fx"]), 1e-30),
                              {"abs": 0.02})],
        "ref": False, "needs": ["gmsh", "build123d_translator"]}
    C["pipe_snappy"] = {
        "study": {"geometry": {"cylinder": [5, 200]}, "fluid": "water",
                  "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "flow_rate_lpm": 0.5, "profile": "parabolic"},
                          {"kind": "outlet", "faces": {"nz": 1}}],
                  "mesh": {"mesher": "snappy", "size": 2, "surface": 0.5, "layers": {"n": 3, "first": 0.1, "growth": 1.2}},
                  "solve": {"residual": 1e-5, "iterations": 4000},
                  "probes": [{"name": "a", "at": [0, 0, 50]}, {"name": "b", "at": [0, 0, 150]}]},
        "checks": lambda r: [("dp over 0.1 m = 8 mu L Q / (pi R^4)", 8 * mu_w * 0.1 * Q / (math.pi * R ** 4),
                              r["probes"]["a"]["p"] - r["probes"]["b"]["p"], {"tol": 0.03}),
                             ("wall force = 8 pi mu L U_mean", 8 * math.pi * mu_w * 0.2 * Q / (math.pi * R ** 2),
                              r["forces"]["walls"]["fz"], {"tol": 0.03})],
        "ref": False, "needs": ["gmsh"]}
    # ---- transient (transient.py) ----
    # Schaefer & Turek 2D-2: the 2D-1 channel at mean inflow 1 m/s (Re 100), periodic vortex shedding:
    # cD max 3.22-3.24, cL max 0.99-1.01, St = f D / U_mean 0.295-0.305
    def st2d2(extra_run=None):
        st = dict(C["cylinder_2d1"]["study"])
        st["bcs"] = [dict(b, velocity=1.0) if b["kind"] == "inlet" else b for b in st["bcs"]]
        st["reference"] = {"velocity": 1.0, "area": 1000, "length": 100}
        st["solve"] = {"transient": {"end_time": 8.0, "average_from": 5.0, "max_co": 0.5, "write_interval": 1.0,
                                     **({"max_jobs": 30} if extra_run else {})}}
        st.pop("probes", None)
        if extra_run:
            st["run"] = extra_run
        return st
    st_checks = lambda r: [("cD max (3.22-3.24)", 3.23, r["coefficients"]["cd_max"], {"tol": 0.03}),
                           ("cL max (0.99-1.01)", 1.0, r["coefficients"]["cl_max"], {"tol": 0.06}),
                           ("Strouhal f D / U_mean (0.295-0.305)", 0.30, r["coefficients"]["strouhal"], {"tol": 0.03}),
                           ("reliable (the time averages converged)", 1.0, 1.0 if r["_reliable"] else 0.0, {"abs": 0.5})]
    C["cylinder_2d2"] = {"study": st2d2(), "checks": st_checks, "ref": False, "needs": ["gmsh"]}
    # the same in SLURM jobs of 2 minutes each: it must continue from job to job (stopAt writeNow, startFrom
    # latestTime, the averages carried over) and give the same answers
    C["cylinder_2d2_restart"] = {"study": st2d2({"timeout_s": 120, "queue": True}), "checks": st_checks, "ref": False,
                                 "needs": ["gmsh"]}
    # Ahmed body, the fine steady mesh (L3) with DES (kOmegaSSTDES) from a 1000-iteration RANS start, 0.4 s
    # (15 body lengths) averaged from 0.1 s
    for phi, cd_exp in ((25, 0.285), (35, 0.26)):
        st = ahmed_study(phi, 0.5, 19)
        st["model"] = "kOmegaSSTDES"
        st["solve"] = {"transient": {"end_time": 0.4, "average_from": 0.1, "init": 1000, "max_co": 1.0,
                                     "write_interval": 0.05}}
        C[f"ahmed{phi}_des_mg"] = {"study": st, "checks": _ahmed_des_checks(cd_exp), "ref": False,
                                "needs": ["gmsh", "build123d_translator"], "slow": True}
    # ---- the vehicle-aero study's Ahmed cases (examples/cfdt_dev_prompt_vehicle_aero.md): 60 m/s, stationary
    # floor as in the experiment, snappy L1 steady and L2 DES. Two checks each: physics against the wind tunnel
    # (a tolerance for the method's known accuracy) and regression against what cfdt 2026.10.22.10 produced on
    # mochi (a tight tolerance: a change in the tool's own behaviour)
    for phi, cd_t, tol, cd_r, cl_r in ((25, 0.2875, 0.10, 0.289, 0.353), (35, 0.26, 0.15, 0.285, 0.026)):
        C[f"ahmed{phi}_snappy"] = {
            "study": ahmed_va_study(phi), "ref": False, "needs": ["gmsh", "build123d_translator"], "slow": True,
            "checks": _va_checks(cd_t, tol, cd_r, cl_r, 0.03)}
    C["ahmed35_des"] = {
        "study": ahmed_va_study(35, des=True), "ref": False, "needs": ["gmsh", "build123d_translator"], "slow": True,
        "checks": _va_checks(0.26, 0.12, 0.2387, -0.078, 0.04, freq=22.7)}
    # Ahmed body (Ahmed, Ramm & Faltin 1984), half model on a symmetry plane, no stilts, 50 mm over a moving
    # ground, 40 m/s (Re_L 2.8e6, as Lienhart & Becker 2003), kOmegaSST with wall functions and prism layers at
    # y+ ~ 60; three meshes, each sqrt(2) finer than the last. Wind tunnel: Cd 0.285 (25 deg), 0.26 (35 deg).
    for phi, cd_exp in ((25, 0.285), (35, 0.26)):
        for lev, f, procs in ((1, 1.0, 6), (2, 1 / math.sqrt(2), 10), (3, 0.5, 19)):
            C[f"ahmed{phi}_L{lev}"] = {"study": ahmed_study(phi, f, procs), "checks": _ahmed_checks(cd_exp),
                                       "ref": False, "needs": ["gmsh", "build123d_translator"], "slow": True}
    return C


def ahmed_goal(phi: float) -> dict:
    """The Ahmed body's half (y >= 0): 1044 x 389 x 288 mm, front edges rounded R 100, a 222 mm slant at phi."""
    c, t = math.cos(math.radians(phi)), math.tan(math.radians(phi))
    xs = round(522 - 222 * c, 3)
    zc = round(288 - t * (542 - xs), 3)
    return {"features": [
        {"kind": "base_box", "params": {"w": 1044, "d": 389, "h": 288}},
        {"kind": "fillet_edges", "params": {"r": 100, "pts": [[-522, 0, 288], [-522, 0, 0], [-522, 194.5, 144],
                                                              [-522, -194.5, 144]]}},
        {"kind": "profile_pocket", "params": {"nx": 0, "ny": -1, "nz": 0, "x": 0, "y": 0, "z": 0, "outline": {
            "start": [xs, 288], "segs": [["L", 542, zc], ["L", 542, 300], ["L", xs, 300], ["L", xs, 288]]}}},
        {"kind": "split", "params": {"nx": 0, "ny": 1, "nz": 0, "off": 0}}]}


def ahmed_study(phi: float, f: float, procs: int) -> dict:
    """f scales every cell size (1: coarse, 1/sqrt(2), 1/2: fine); the levels stay the same."""
    def r(v):
        return round(v * f, 3)
    U = 40.0
    return {"geometry": {"goal": ahmed_goal(phi)},
            "domain": {"kind": "external", "box": [-3022, 0, -50, 5522, 1800, 1600]},
            "fluid": "air", "model": "kOmegaSST",
            "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": U, "intensity": 0.01},
                    {"kind": "outlet", "faces": {"box": "xmax"}},
                    {"kind": "symmetry", "name": "mid", "faces": {"box": "ymin"}},
                    {"kind": "slip", "name": "far", "faces": {"box": ["ymax", "zmax"]}},
                    {"kind": "wall", "name": "ground", "faces": {"box": "zmin"}, "velocity": [U, 0, 0]},
                    {"kind": "wall", "name": "body", "faces": {"all": True}}],
            "reference": {"velocity": U, "area": 389 * 288 / 2, "length": 1044, "drag_dir": [1, 0, 0],
                          "lift_dir": [0, 0, 1]},
            "mesh": {"mesher": "snappy", "size": r(160), "surface": [r(10), r(20)], "features": r(10),
                     "refine": [{"box": [-800, 0, -50, 1600, 500, 550], "size": r(40)},
                                {"box": [300, 0, -50, 2700, 350, 450], "size": r(20)},
                                {"distance": 60, "size": r(20)}],
                     "layers": {"y_plus": 60}},
            "solve": {"iterations": 4000},
            "run": {"procs": procs},
            "images": True}


def ahmed_va_goal(phi: float) -> dict:
    """The vehicle-aero study's Ahmed half body: 1044 x 389 x 288 mm, front edges R100, a 222 mm slant at phi,
    50 mm above the floor (z = 0), y >= 0."""
    xs = {25: 320.8, 35: 340.15}[phi]
    ze = {25: 176.46, 35: 134.06}[phi]
    inner = {"features": [
        {"kind": "base_box", "params": {"w": 1044, "d": 389, "h": 288}},
        {"kind": "fillet_edges", "params": {"r": 100, "nx": -1, "ny": 0, "nz": 0}},
        {"kind": "profile_pocket", "params": {"nx": 0, "ny": 1, "nz": 0, "off": 0, "x": 0, "y": 0, "z": 0, "outline": {
            "start": [xs, 288], "segs": [["L", 560, ze], ["L", 560, 400], ["L", xs, 400], ["L", xs, 288]]}}}]}
    return {"features": [{"kind": "combine", "params": {"op": "add", "goal": inner, "move": [0, 0, 50]}},
                         {"kind": "split", "params": {"nx": 0, "ny": 1, "nz": 0, "off": 0}}]}


def ahmed_va_study(phi: float, des: bool = False) -> dict:
    """Steady kOmegaSST on the L1 snappy mesh, or (des) kOmegaSSTDES on L2 (sizes x 0.707), 0.4 s averaged from
    0.15 s; as run in the vehicle-aero study on mochi (9 procs)."""
    f = 0.707 if des else 1.0

    def r(v):
        return round(v * f, 2)
    st = {"geometry": {"goal": ahmed_va_goal(phi)}, "domain": {"kind": "external", "box": [-3650, 0, 0, 6800, 1500, 1500]},
          "fluid": "air", "model": "kOmegaSSTDES" if des else "kOmegaSST",
          "bcs": [{"kind": "inlet", "name": "inlet", "faces": {"box": "xmin"}, "velocity": 60, "intensity": 0.01,
                   "length_scale": 45},
                  {"kind": "outlet", "name": "outlet", "faces": {"box": "xmax"}, "pressure": 0},
                  {"kind": "symmetry", "name": "sym", "faces": {"box": "ymin"}},
                  {"kind": "slip", "name": "farfield", "faces": {"box": ["ymax", "zmax"]}},
                  {"kind": "wall", "name": "ground", "faces": {"box": "zmin"}}],
          "mesh": {"mesher": "snappy", "size": r(100), "surface": [r(3), r(6)], "features": r(3) if des else True, "gap": 1,
                   "refine": [{"box": [-700, 0, 0, 800, 350, 500], "size": r(12)},
                              {"box": [522, 0, 0, 1800, 350, 500], "size": r(16)}, {"distance": 20, "size": r(6)}],
                   "layers": {"y_plus": 60, "n": 3 if des else 5, "growth": 1.2}},
          "reference": {"velocity": 60, "area": 56016, "length": 1044, "cofr": [0, 0, 0], "drag_dir": [1, 0, 0],
                        "lift_dir": [0, 0, 1]}}
    if des:
        st["solve"] = {"procs": 9, "transient": {"end_time": 0.4, "average_from": 0.15, "max_co": 1, "init": 500,
                                                 "max_jobs": 10}}
    else:
        st["solve"] = {"iterations": 4000, "procs": 9, "relax": {"U": 0.4, "p": 0.2, "other": 0.4},
                       "consistent": False, "start": "estimate", "average": True}
    return st


def _va_checks(cd_tunnel: float, tol: float, cd_reg: float, cl_reg: float, reg: float, freq: float | None = None):
    """Physics (the tunnel, tol) and regression (cfdt 2026.10.22.10's own values, reg; lift against the drag's
    size, since it can be near zero)."""
    def f(r):
        out = [("Cd vs the wind tunnel (%.4g)" % cd_tunnel, cd_tunnel, r["coefficients"]["cd"], {"tol": tol}),
               ("Cd regression (cfdt 2026.10.22.10: %.4g)" % cd_reg, cd_reg, r["coefficients"]["cd"], {"tol": reg}),
               ("Cl regression (cfdt 2026.10.22.10: %.4g)" % cl_reg, cl_reg, r["coefficients"]["cl"],
                {"abs": reg * cd_reg})]
        if freq:
            out.append(("dominant frequency regression (%.3g Hz)" % freq, freq, r["coefficients"]["frequency_Hz"],
                        {"tol": 0.10}))
        return out
    return f


def _ahmed_des_checks(cd_exp: float):
    return lambda r: [
        ("Cd, time-averaged (wind tunnel %.3f)" % cd_exp, cd_exp, r["coefficients"]["cd"], {"tol": 0.10}),
        ("mean y+ on the body (wall functions want 30-300)", 165.0, r["y_plus_part_mean"], {"abs": 135.0}),
        ("reliable (the time averages converged)", 1.0, 1.0 if r["_reliable"] else 0.0, {"abs": 0.5})]


def _ahmed_checks(cd_exp: float):
    return lambda r: [
        ("Cd, time-averaged over the last iterations (wind tunnel %.3f)" % cd_exp, cd_exp, r["coefficients"]["cd"],
         {"tol": 0.10}),
        ("mean y+ on the body (wall functions want 30-300)", 165.0, r["y_plus_part_mean"], {"abs": 135.0}),
        ("prism-layer coverage on the body (%)", 100.0, r["_mesh"]["layers"]["coverage_pct"], {"tol": 0.4}),
        ("reliable (converged or settled)", 1.0, 1.0 if r["_reliable"] else 0.0, {"abs": 0.5})]


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
        out[n] = evaluate(n, r, cases)
        _progress(n, out[n], t0, r)
    return out


def evaluate(n: str, r: dict, cases: dict | None = None) -> dict:
    """A benchmark's verdict from its study result: every check against its tolerance."""
    c = (cases or _cases())[n]
    if not r.get("ok"):
        return {"ok": False, "passed": False, "why": r.get("why"), "error": r.get("error") or r.get("errors")}
    checks = []
    try:
        triples = c["checks"](dict(r["results"], _mesh=r.get("mesh") or {}, _reliable=bool(r.get("reliable"))))
    except (KeyError, TypeError, ZeroDivisionError) as e:
        return {"ok": True, "passed": False, "error": f"a check could not be evaluated: {type(e).__name__}: {e}"}
    for what, exp, got, tol in triples:
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
    return {"ok": True, "passed": all(c_["passed"] for c_ in checks), "checks": checks,
            "solve": r.get("solve", {}).get("converged"), "cells": r["mesh"]["cells"],
            "invariants": {k: r["results"]["checks"][k].get("error") for k in ("mass_balance", "momentum_balance")},
            "warnings": r.get("warnings", []), "seconds": r.get("seconds")}


def submit(solver: str, names=None) -> dict:
    """SLURM mode for the MCP tool: submit the cases as one request and return at once (the tool call would
    otherwise outlast the client's timeout: mochi, 2026.10.19). Returns {request, cases, skipped}; chains are
    left out (they run fea-translator here: run them by name with the CLI)."""
    cases = _cases()
    skipped, todo = {}, []
    for n, c in cases.items():
        if names and n not in names:
            continue
        if (not names and c.get("slow")) or c.get("chain"):
            skipped[n] = "slow or a chain: run it by name with the CLI (cfdt bench <name>)"
            continue
        missing = [m for m in c["needs"] if not _available(m)]
        if missing:
            skipped[n] = f"needs {', '.join(missing)}"
            continue
        todo.append(n)
    unknown = [n for n in (names or []) if n not in cases]
    from . import api
    req = api.start({n: dict(cases[n]["study"], queue=True) for n in todo}, solver) if todo else None
    return {"request": str(req) if req else None, "cases": todo, "skipped": skipped, "unknown": unknown}


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
