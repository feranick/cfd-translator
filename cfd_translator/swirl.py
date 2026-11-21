"""Swirl in named regions: the mean tangential velocity and the angular momentum about a rotating zone's axis.

Version: 2026.11.20.8

The S50 zero-air gate (pump chat, 20 Nov 2026) compares the swirl of a two-fluid run at zero air with a single-phase
run on the same mesh, in the eye, the pocket in front of the wear plate and the reservoir: a spurious swirl gain drives
air to the eye, which is what the gas-fraction sweep (a) tests. OpenFOAM 14's multiphaseEuler adds angular momentum to
a solid-body rotation (+5 % in 1 s, mrf_hold, 20 Nov 2026), so head and torque alone are not enough.

Per region (cells whose centres fall in a box or cylinder, mm), about the zone's axis (unit a, origin o, omega):
  u_theta_m_s    sum(w V u_theta) / sum(w V)                 mean tangential velocity (+ in the zone's sense)
  swirl_ratio    sum(w V u_theta) / sum(w V omega r)         against the solid body turning with the zone
  angular_momentum_kg_m2_s  rho sum(w V r u_theta)           (w = the liquid's volume fraction in two-fluid runs)
  L_ratio        sum(w V r u_theta) / sum(w V omega r^2)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCALE = 1e-3


def frame_of(plan: dict, zone: str | None = None) -> dict | None:
    """{axis, center_mm, omega} of the named rotating zone, else the first; None without one."""
    zs = plan.get("zones") or {}
    if not zs:
        return None
    zp = zs.get(zone) if zone else zs[next(iter(zs))]
    if zp is None:
        return None
    a = np.asarray(zp["axis"], float)
    return {"axis": a / np.linalg.norm(a), "center_mm": np.asarray(zp["center_mm"], float),
            "omega": float(zp["omega_rad_s"]), "zone": zone or next(iter(zs))}


def _cells(r: dict, C: np.ndarray) -> np.ndarray:
    from .twophase import region_cells
    return region_cells(r, C)


def sums(C_mm: np.ndarray, V_mm3: np.ndarray, U: np.ndarray, w: np.ndarray | None, regions: list, fr: dict) -> dict:
    """Additive sums per region (so processors add up): [sum wV, sum wV u_t, sum wV omega r, sum wV r u_t,
    sum wV omega r^2, cells]."""
    a, o, om = fr["axis"], fr["center_mm"], fr["omega"]
    d = (np.asarray(C_mm, float) - o) * SCALE
    rvec = d - np.outer(d @ a, a)
    r = np.linalg.norm(rvec, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        et = np.cross(a, rvec) / r[:, None]
    et[~np.isfinite(et)] = 0.0
    ut = np.einsum("ij,ij->i", np.asarray(U, float), et) * (1.0 if om >= 0 else -1.0)
    wv = np.abs(np.asarray(V_mm3, float)) * SCALE ** 3 * (np.asarray(w, float) if w is not None else 1.0)
    W = abs(om)
    out = {}
    for reg in regions:
        cs = _cells(reg, C_mm)
        out[reg["name"]] = np.array([wv[cs].sum(), (wv * ut)[cs].sum(), (wv * W * r)[cs].sum(),
                                     (wv * r * ut)[cs].sum(), (wv * W * r * r)[cs].sum(), len(cs)], float)
    return out


def finish(acc: dict, rho: float, fr: dict, basis: str) -> dict:
    out = {}
    for name, s in acc.items():
        if s[5] == 0 or s[0] <= 0:
            out[name] = {"cells": 0, "note": "no cell centre in the region"}
            continue
        out[name] = {"cells": int(s[5]), "volume_m3": float(f"{s[0]:.5g}"),
                     "u_theta_m_s": float(f"{s[1] / s[0]:.5g}"),
                     "swirl_ratio": float(f"{s[1] / s[2]:.5g}") if s[2] > 0 else None,
                     "angular_momentum_kg_m2_s": float(f"{rho * s[3]:.5g}"),
                     "L_ratio": float(f"{s[3] / s[4]:.5g}") if s[4] > 0 else None}
    return {"about": {"zone": fr.get("zone"), "axis": [round(float(v), 6) for v in fr["axis"]],
                      "center_mm": [round(float(v), 4) for v in fr["center_mm"]], "omega_rad_s": fr["omega"]},
            "basis": basis, "regions": out}


def from_fields(mesh, U: np.ndarray, regions: list, fr: dict, rho: float, basis: str, w=None) -> dict:
    g = mesh.geometry()
    return finish(sums(g["C"], g["V"], U, w, regions, fr), rho, fr, basis)


def _scalars(path: Path, n: int) -> np.ndarray | None:
    from . import foamio
    try:
        d = foamio.read_foam(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return None
    a = np.asarray(d.get("internalField"), float)
    if a.ndim == 0 or a.size == 1:
        return np.full(n, float(a.reshape(-1)[0]))
    return a if a.shape == (n,) else None


def from_case(case: Path, regions: list, fr: dict, rho: float, ufield: str = "U.water", wfield: str | None = "alpha.water",
              t_from: float | None = None, latest_only: bool = False) -> dict | None:
    """Swirl from a finished case's written fields: the latest time (latest_only), else the mean over the written
    times at or after t_from. Reads the reconstructed case, or every processor directory when the times are there."""
    from .diverge import _mesh, _times, _vectors
    case = Path(case)
    procs = sorted(case.glob("processor[0-9]*"))
    roots = [case]
    tl = [t for t in _times(case) if t[0] > 0 and (case / t[1] / ufield).is_file()]
    if not tl and procs:
        roots = procs
        tl = [t for t in _times(procs[0]) if t[0] > 0 and (procs[0] / t[1] / ufield).is_file()]
    if not tl:
        return None
    if latest_only or t_from is None:
        sel = [tl[-1]]
    else:
        sel = [t for t in tl if t[0] >= t_from - 1e-9] or [tl[-1]]
    acc = None
    for d in roots:
        mesh = _mesh(d, sel[-1][1])
        g = mesh.geometry()
        n = mesh.ncells
        for _, tn in sel:
            U = _vectors(d / tn / ufield, n)
            if U is None:
                return None
            w = _scalars(d / tn / wfield, n) if wfield else None
            if wfield and w is None:
                w = None
            s = sums(g["C"], g["V"], U, w, regions, fr)
            if acc is None:
                acc = {k: v / len(sel) for k, v in s.items()}
            else:
                for k, v in s.items():
                    acc[k] = acc[k] + v / len(sel)
    for k in acc:                                   # cell counts: per written time, not averaged away
        acc[k][5] = round(acc[k][5])
    basis = (f"{ufield}" + (f" weighted by {wfield}" if wfield else "") +
             (f", time {sel[-1][0]:g}" if len(sel) == 1 else f", mean of {len(sel)} written times {sel[0][0]:g}-{sel[-1][0]:g}"))
    return finish(acc, rho, fr, basis)
