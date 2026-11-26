"""Swirl in named regions: the mean tangential velocity and the angular momentum about a rotating zone's axis.

Version: 2026.11.26.5

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


SNAPSHOTS = 10            # field snapshots over a steady run's averaging window (gate: mean and its standard error)


def snapshot_fo(objects: list, start: int, every: int, end: int | None = None, name: str = "cfdtSwirlSnap") -> str:
    """A writeObjects function object: the named fields every `every` iterations from `start` (the averaging window;
    to `end` when given), so the regions' swirl has a mean and a standard error over the same window as head and
    torque."""
    return (f"    {name}\n    {{\n        type            writeObjects;\n"
            "        libs            (\"libutilityFunctionObjects.so\");\n"
            f"        objects         ({' '.join(objects)});\n        writeControl    timeStep;\n"
            f"        writeInterval   {int(every)};\n        timeStart       {int(start)};\n"
            + (f"        timeEnd         {int(end)};\n" if end is not None else "") + "    }")


def snapshot_plan(iterations: int, window_iterations: int) -> dict:
    """{start, every, n}: SNAPSHOTS writes over the last window_iterations of a steady run."""
    w = max(1, min(int(window_iterations), int(iterations)))
    every = max(1, w // SNAPSHOTS)
    return {"start": int(iterations) - w + 1, "every": every, "n": w // every}


def _vectors_fast(path: Path, n: int):
    """The internalField of an ascii volVectorField, parsed with numpy (snapshots of millions of cells)."""
    try:
        txt = path.read_bytes()
    except OSError:
        return None
    i = txt.find(b"internalField")
    if i < 0:
        return None
    j = txt.find(b"\n", i)
    head = txt[i:j]
    if b"uniform" in head and b"nonuniform" not in head:
        a = np.fromstring(head.split(b"(")[1].split(b")")[0].decode(), sep=" ")
        return np.tile(a, (n, 1)) if a.size == 3 else None
    k = txt.find(b"(", j)
    e = txt.find(b"\n)", k)
    if k < 0 or e < 0:
        return None
    a = np.fromstring(txt[k + 1:e].replace(b"(", b" ").replace(b")", b" ").decode(), sep=" ")
    return a.reshape(-1, 3) if a.size == 3 * n else None


def _scalars_fast(path: Path, n: int):
    try:
        txt = path.read_bytes()
    except OSError:
        return None
    i = txt.find(b"internalField")
    if i < 0:
        return None
    j = txt.find(b"\n", i)
    head = txt[i:j]
    if b"nonuniform" not in head:
        try:
            return np.full(n, float(head.split(b"uniform")[1].strip(b" ;\r")))
        except (ValueError, IndexError):
            return None
    k = txt.find(b"(", j)
    e = txt.find(b"\n)", k)
    a = np.fromstring(txt[k + 1:e].decode(), sep=" ")
    return a if a.size == n else None


def _roots_and_times(case: Path, ufield: str):
    """The directories holding the written times of ufield: the case, or its processors when they hold more times
    (reconstructPar -latestTime leaves the snapshots in the processors)."""
    from .diverge import _times
    case = Path(case)
    procs = sorted(case.glob("processor[0-9]*"), key=lambda p: int(p.name[9:]))
    tl = [t for t in _times(case) if t[0] > 0 and (case / t[1] / ufield).is_file()]
    if procs:
        tp = [t for t in _times(procs[0]) if t[0] > 0 and (procs[0] / t[1] / ufield).is_file()]
        if len(tp) > len(tl):
            return procs, tp
    return [case], tl


def from_case(case: Path, regions: list, fr: dict, rho: float, ufield: str = "U.water", wfield: str | None = "alpha.water",
              t_from: float | None = None, latest_only: bool = False, cleanup: bool = False,
              t_to: float | None = None) -> dict | None:
    """Swirl from a finished case's written fields: the latest time (latest_only), else the mean over the written
    times at or after t_from, with each region's standard error over those snapshots (mean_stderr, as head and
    torque). Reads the reconstructed case, or every processor directory when the times are there. cleanup: remove the
    snapshot directories (all but the latest time) afterwards."""
    from .average import mean_stderr
    from .diverge import _mesh
    roots, tl = _roots_and_times(Path(case), ufield)
    if not tl:
        return None
    if latest_only or t_from is None:
        sel = [tl[-1]]
    else:
        sel = [t for t in tl if t[0] >= t_from - 1e-9 and (t_to is None or t[0] <= t_to + 1e-9)] or [tl[-1]]
    per_time = [None] * len(sel)
    for d in roots:
        mesh = _mesh(d, sel[-1][1])
        g = mesh.geometry()
        n = mesh.ncells
        for it, (_, tn) in enumerate(sel):
            U = _vectors_fast(d / tn / ufield, n)
            if U is None:
                return None
            w = _scalars_fast(d / tn / wfield, n) if wfield else None
            s = sums(g["C"], g["V"], U, w, regions, fr)
            if per_time[it] is None:
                per_time[it] = s
            else:
                for k, v in s.items():
                    per_time[it][k] = per_time[it][k] + v
    acc = {k: sum(pt[k] for pt in per_time) / len(sel) for k in per_time[0]}
    for k in acc:                                   # cell counts: per written time, not averaged away
        acc[k][5] = round(acc[k][5])
    basis = (f"{ufield}" + (f" weighted by {wfield}" if wfield else "") +
             (f", time {sel[-1][0]:g}" if len(sel) == 1 else f", mean of {len(sel)} written times {sel[0][0]:g}-{sel[-1][0]:g}"))
    out = finish(acc, rho, fr, basis)
    if len(sel) >= 4:
        for name, rec in out["regions"].items():
            ut = np.array([pt[name][1] / pt[name][0] for pt in per_time if pt[name][0] > 0])
            sr = np.array([pt[name][1] / pt[name][2] for pt in per_time if pt[name][2] > 0])
            if len(ut) >= 4:
                se, _ = mean_stderr(ut)
                rec["u_theta_stderr"] = float(f"{se:.3g}")
                rec["u_theta_series"] = [float(f"{x:.5g}") for x in ut]
            if len(sr) >= 4:
                se, _ = mean_stderr(sr)
                rec["swirl_ratio_stderr"] = float(f"{se:.3g}")
            rec["snapshots"] = len(ut)
        out["window"] = [sel[0][0], sel[-1][0]]
    if cleanup and len(tl) > 1:
        remove_snapshots(roots, [t for t in sel if t[1] != tl[-1][1]], ufield, wfield)
    return out


def remove_snapshots(roots: list, times: list, *fields) -> None:
    """Delete the snapshot directories (only fields and a 'uniform' folder in them: nothing else is lost)."""
    import shutil
    keep_names = {f for f in fields if f} | {"uniform"}
    for d in roots:
        for _, tn in times:
            td = Path(d) / tn
            try:
                if td.is_dir() and all(p.name in keep_names for p in td.iterdir()):
                    shutil.rmtree(td)
            except OSError:
                pass


def gate_compare(q1: float, q2: float, se1: float | None, se2: float | None, kind: str, tol: float) -> dict:
    """Two runs' means q1 (reference) and q2 against each other: within tol (relative: of |q1|; absolute: in q's own
    units), or within twice their combined standard error (the pump chat's rule, 21 Nov 2026)."""
    diff = q2 - q1
    se = float(np.hypot(se1 or 0.0, se2 or 0.0)) if (se1 is not None or se2 is not None) else None
    within_tol = abs(diff) <= tol * (abs(q1) if kind == "relative" else 1.0) * (1 + 1e-9) + 1e-15
    within_noise = se is not None and se > 0 and abs(diff) <= 2.0 * se
    out = {"reference": q1, "value": q2, "diff": float(f"{diff:.4g}"), "rule": f"{kind} {tol:g} or 2 sigma",
           "ok": bool(within_tol or within_noise), "within_tol": bool(within_tol)}
    if kind == "relative" and abs(q1) > 0:
        out["ratio"] = float(f"{q2 / q1:.4g}")
    if se is not None:
        out["combined_stderr"] = float(f"{se:.3g}")
        out["within_2_sigma"] = bool(within_noise)
    return out


def attach_window(case: Path, study: dict, plan: dict, res: dict, cleanup: bool = True) -> None:
    """A steady single-phase run with snapshots over its averaging window (plan swirl_snap): results.swirl from their
    mean, with each region's standard error, in place of the last iteration's."""
    if ((study.get("solve") or {}).get("average") or {}).get("keep_snapshots") if isinstance(
            (study.get("solve") or {}).get("average"), dict) else False:
        cleanup = False                        # solve.average.keep_snapshots: kept for later analyses (bubbles)
    snap = plan.get("swirl_snap")
    if not snap or not study.get("regions") or not plan.get("zones"):
        return
    groups = {}
    for r in study["regions"]:
        groups.setdefault(r.get("zone"), []).append(r)
    def window(t0, t1=None):
        out_ = None
        for zone, rs in groups.items():
            fr = frame_of(plan, zone)
            if fr is None:
                continue
            try:
                s = from_case(case, rs, fr, float(plan["rho"]), "U", None, t_from=t0, t_to=t1)
            except (OSError, ValueError, KeyError, IndexError):
                s = None
            if s is None:
                continue
            if out_ is None:
                out_ = s
            else:
                out_["regions"].update(s["regions"])
        return out_
    out = window(snap["start"])
    if out:
        res["swirl"] = out
    extra = []                                 # solve.average.snapshot_windows: the same over other windows
    for w in plan.get("swirl_snap_windows") or []:
        s = window(w["start"], w["end"])
        if s:
            extra.append(s)
    if extra:
        res["swirl_windows"] = extra
    if cleanup:                                # every snapshot but the latest time
        roots, tl = _roots_and_times(Path(case), "U")
        if len(tl) > 1:
            remove_snapshots(roots, tl[:-1], "U")
