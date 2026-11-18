"""diverge.py — where a diverged run blew up: the worst cells of the fields written when it stopped.
Version: 2026.11.17.9

When the solver's log shows divergence (foam.divergence), a transient run is asked to write its fields and stop
(stopAt writeNow, up to CFDT_DIVERGE_WRITE_S seconds) before it is killed; report() then reads the latest time
directory (each processor's when the case is decomposed) and ranks the cells by a Courant number estimated from
the written velocity (relative to the rotor inside a rotating zone) and by speed. Each cell comes with its place in
the zone's frame (r, s along the axis, theta = atan2(y, x) for the z axis, as sliding.py and the port's precheck),
its distance from the zone's cylinder and its size and thickness. Pure NumPy, no OpenFOAM utility.

S50 noboth with ESI (spark2, job 643, 2026-11-17): it diverged at the second sliding step (Courant 5.5e7) and the
run said only that; where the cells were had to be guessed from the mesh.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from . import foamio

WRITE_S = 600.0            # CFDT_DIVERGE_WRITE_S: how long a diverged run may take to write its fields (0: kill)
TOP = 8                    # cells listed per ranking
HOT_CO = 100.0             # cells above this Courant estimate count as 'hot'
NEAR_MM = 2.0              # 'at the interface': within this of the zone's cylinder (r) or end planes (s)
FILE = "cfdt_diverged.json"
FRAME = "cfdt_locator.json"     # locator.write_frame: axis, center_mm (+ radius_mm, span_mm, omega_rad_s)


def write_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("CFDT_DIVERGE_WRITE_S", "") or WRITE_S))
    except ValueError:
        return WRITE_S


def _num(p: Path) -> int:
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else 0


def _times(d: Path) -> list:
    out = []
    for p in Path(d).iterdir() if Path(d).is_dir() else []:
        if p.is_dir():
            try:
                out.append((float(p.name), p.name))
            except ValueError:
                pass
    return sorted(out)


def _dt(case: Path) -> float | None:
    """The run's last time step: the log's last 'deltaT = ', else controlDict's deltaT."""
    log = Path(case) / "log.foamRun"
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - 2_000_000))
            m = re.findall(r"^deltaT = ([-+0-9.eE]+)", f.read().decode(errors="replace"), re.M)
        if m:
            return float(m[-1])
    except (OSError, ValueError):
        pass
    try:
        m = re.search(r"^deltaT\s+([-+0-9.eE]+);", (Path(case) / "system" / "controlDict").read_text(), re.M)
        return float(m.group(1)) if m else None
    except (OSError, ValueError):
        return None


def _log_time(case: Path) -> float | None:
    log = Path(case) / "log.foamRun"
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - 2_000_000))
            m = re.findall(r"^Time = ([-+0-9.eE]+)", f.read().decode(errors="replace"), re.M)
        return float(m[-1]) if m else None
    except (OSError, ValueError):
        return None


def _frame(case: Path) -> dict | None:
    try:
        fr = json.loads((Path(case) / FRAME).read_text())
        a = np.asarray(fr["axis"], float)
        fr["a"] = a / np.linalg.norm(a)
        fr["o"] = np.asarray(fr["center_mm"], float)
        return fr
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _vectors(path: Path, n: int) -> np.ndarray | None:
    try:
        d = foamio.read_foam(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return None
    a = np.asarray(d.get("internalField"), float)
    if a.ndim == 1 and a.size == 3:
        return np.tile(a, (n, 1))
    return a if a.shape == (n, 3) else None


def _mesh(d: Path, tname: str):
    """The rank's mesh at time tname: that time's polyMesh where it was written (a moving mesh writes its points),
    else constant/polyMesh. Coordinates in mm."""
    from .polymesh import PolyMesh
    td = d / tname / "polyMesh"
    md = td if (td / "faces").is_file() else d / "constant" / "polyMesh"
    m = foamio.read_mesh_arrays(md)
    pts = m["points"]
    if md != td and (td / "points").is_file():
        pts = foamio.read_points(td / "points")
    nb = len(m["owner"]) - len(m["neighbour"])
    return PolyMesh(points=np.asarray(pts, float) * 1e3, faces=m["faces"], owner=m["owner"],
                    neighbour=m["neighbour"], btag=np.zeros(nb, np.int64), ncells=int(m["ncells"]))


def _basis(a):
    from .sliding import _basis as b
    return b(a)


def _rank(case: Path, d: Path, k: int, tname: str, dt: float | None, fr: dict | None, top: int) -> dict | None:
    mesh = _mesh(d, tname)
    nc = mesh.ncells
    U = _vectors(d / tname / "U", nc)
    if U is None:
        return None
    g = mesh.geometry()
    C, V, Sf = g["C"], np.abs(g["V"]), g["Sf"]
    own, nei = mesh.owner, mesh.neighbour
    ni = len(nei)
    out = {"cells": nc}
    Ur = U.copy()
    r = s = th = None
    inz = np.zeros(nc, bool)
    if fr is not None:
        a, o = fr["a"], fr["o"]
        e1, e2 = _basis(a)
        dd = C - o
        s = dd @ a
        rad = dd - np.outer(s, a)
        r = np.linalg.norm(rad, axis=1)
        th = np.degrees(np.arctan2(rad @ e2, rad @ e1)) % 360
        if fr.get("radius_mm") is not None:
            s0, s1 = fr.get("span_mm") or [-np.inf, np.inf]
            inz = (r <= float(fr["radius_mm"])) & (s >= s0) & (s <= s1)
            w = float(fr.get("omega_rad_s") or 0.0)
            if w:
                Ur[inz] -= w * np.cross(a, dd[inz]) * 1e-3          # m/s
    bad_cell = ~np.isfinite(U).all(axis=1)
    Uf = Ur[own].copy()
    Uf[:ni] = 0.5 * (Ur[own[:ni]] + Ur[nei])
    flux = np.abs(np.einsum("ij,ij->i", np.nan_to_num(Uf, nan=np.inf, posinf=np.inf, neginf=np.inf), Sf)) * 1e-6
    sphi = np.bincount(own, flux, nc) + np.bincount(nei, flux[:ni], nc)
    with np.errstate(divide="ignore", invalid="ignore"):
        co = 0.5 * sphi * (dt or 0.0) / (V * 1e-9) if dt else np.full(nc, np.nan)
    amax = np.zeros(nc)
    am = np.linalg.norm(Sf, axis=1)
    np.maximum.at(amax, own, am)
    np.maximum.at(amax, nei, am[:ni])
    speed = np.linalg.norm(U, axis=1)
    key_co = np.where(np.isfinite(co), co, np.inf)
    key_co[bad_cell] = np.inf
    key_sp = np.where(np.isfinite(speed), speed, np.inf)

    def cell(c):
        rec = {"proc": k, "cell": int(c),
               "Co": float(f"{co[c]:.4g}") if np.isfinite(co[c]) else None,
               "speed_m_s": float(f"{speed[c]:.4g}") if np.isfinite(speed[c]) else None,
               "at_mm": [round(float(x), 2) for x in C[c]],
               "volume_mm3": float(f"{V[c]:.3g}"), "size_mm": float(f"{np.cbrt(V[c]):.3g}"),
               "thickness_mm": float(f"{V[c] / amax[c]:.3g}") if amax[c] > 0 else None}
        if fr is not None:
            rec.update(r_mm=round(float(r[c]), 3), s_mm=round(float(s[c]), 3), theta_deg=round(float(th[c]), 1))
            if fr.get("radius_mm") is not None:
                rec.update(in_zone=bool(inz[c]), dr_cylinder_mm=round(float(r[c]) - float(fr["radius_mm"]), 3))
        return rec
    n = min(top, nc)
    out["by_co"] = [cell(c) for c in np.argsort(-key_co)[:n]]
    out["by_speed"] = [cell(c) for c in np.argsort(-key_sp)[:n]]
    out["key_co"] = [float(key_co[c]) for c in np.argsort(-key_co)[:n]]
    out["key_sp"] = [float(key_sp[c]) for c in np.argsort(-key_sp)[:n]]
    out["nonfinite"] = int(bad_cell.sum())
    hot = key_co > HOT_CO
    out["hot"] = int(hot.sum())
    if fr is not None and hot.any():
        out["hot_r"] = [float(r[hot].min()), float(r[hot].max())]
        out["hot_s"] = [float(s[hot].min()), float(s[hot].max())]
        out["hot_theta"] = th[hot]
        if fr.get("radius_mm") is not None:
            s0, s1 = fr.get("span_mm") or [-np.inf, np.inf]
            near = (np.abs(r - float(fr["radius_mm"])) <= NEAR_MM) | (
                (r <= float(fr["radius_mm"]) + NEAR_MM) & ((np.abs(s - s0) <= NEAR_MM) | (np.abs(s - s1) <= NEAR_MM)))
            out["hot_near"] = int((hot & near).sum())
    return out


def report(case: Path, top: int = TOP) -> dict | None:
    """The worst cells of the latest fields: {time, log_time, dt, cells, nonfinite_cells, hot: {Co_above, cells,
    near_interface, r_mm, s_mm, theta_deg}, worst_courant: [...], fastest: [...], files}; None without fields."""
    case = Path(case)
    dirs = sorted(case.glob("processor*"), key=_num) or [case]
    t = _times(dirs[0])
    if not t:
        return None
    tval, tname = t[-1]
    dt = _dt(case)
    fr = _frame(case)
    ranks = []
    for k, d in enumerate(dirs):
        try:
            rk = _rank(case, d, k if d != case else -1, tname, dt, fr, top)
        except (OSError, ValueError, IndexError) as exc:
            rk = {"error": f"{d.name}: {exc}"}
        if rk is not None:
            ranks.append(rk)
    good = [rk for rk in ranks if "error" not in rk]
    if not good:
        return {"time": tval, "error": "; ".join(rk["error"] for rk in ranks) or "no velocity field written"}

    def merge(lst, key):
        rows = [(kv, c) for rk in good for kv, c in zip(rk[key], rk[lst])]
        rows.sort(key=lambda x: -x[0])
        return [c for _, c in rows[:top]]
    out = {"time": tval, "log_time": _log_time(case), "dt": dt, "fields": str(dirs[0] / tname) if dirs[0] != case
           else str(case / tname), "cells": sum(rk["cells"] for rk in good),
           "nonfinite_cells": sum(rk["nonfinite"] for rk in good)}
    nh = sum(rk["hot"] for rk in good)
    hot = {"Co_above": HOT_CO, "cells": nh}
    if nh and fr is not None:
        hs = [rk for rk in good if rk["hot"]]
        hot["r_mm"] = [round(min(rk["hot_r"][0] for rk in hs), 2), round(max(rk["hot_r"][1] for rk in hs), 2)]
        hot["s_mm"] = [round(min(rk["hot_s"][0] for rk in hs), 2), round(max(rk["hot_s"][1] for rk in hs), 2)]
        thv = np.concatenate([rk["hot_theta"] for rk in hs])
        hot["theta_deg"] = _arc(thv)
        if any("hot_near" in rk for rk in hs):
            hot["near_interface"] = sum(rk.get("hot_near", 0) for rk in hs)
    out["hot"] = hot
    out["worst_courant"] = merge("by_co", "key_co")
    out["fastest"] = merge("by_speed", "key_sp")
    if fr is not None and fr.get("omega_rad_s"):
        out["note"] = ("Courant numbers estimated from the written cell velocities (face average), relative to the "
                       "rotor inside the zone's cylinder (r, s, theta in mm and degrees about its axis and centre)")
    errs = [rk["error"] for rk in ranks if "error" in rk]
    if errs:
        out["errors"] = errs[:4]
    return out


def _arc(th: np.ndarray) -> list:
    """The smallest arc [from, to] (degrees) holding the angles."""
    if not len(th):
        return []
    s = np.sort(np.asarray(th, float) % 360)
    gaps = np.diff(np.concatenate([s, [s[0] + 360]]))
    i = int(np.argmax(gaps))
    lo, hi = s[(i + 1) % len(s)], s[i]
    return [round(float(lo), 1), round(float(hi), 1)]


def summary(rep: dict | None) -> str:
    """One sentence for the divergence message."""
    if not rep:
        return ""
    if "error" in rep and "worst_courant" not in rep:
        return f" No worst-cell report: {rep['error']}."
    w = (rep.get("worst_courant") or [{}])[0]
    where = (f"r {w['r_mm']} s {w['s_mm']} theta {w['theta_deg']}" if "r_mm" in w else f"at {w.get('at_mm')} mm")
    if w.get("dr_cylinder_mm") is not None:
        where += f" ({w['dr_cylinder_mm']:+g} mm from the zone's cylinder)"
    co = w.get("Co")
    txt = (f" Fields written at time {rep['time']:g} ({rep['fields']}); worst cell: Courant "
           f"{co if co is not None else 'not finite'}, {w.get('speed_m_s')} m/s, {where}, size {w.get('size_mm')} mm, "
           f"thickness {w.get('thickness_mm')} mm.")
    h = rep.get("hot") or {}
    if h.get("cells"):
        txt += f" {h['cells']} cells above Courant {h['Co_above']:g}"
        if "r_mm" in h:
            txt += f" (r {h['r_mm'][0]}-{h['r_mm'][1]}, s {h['s_mm'][0]}-{h['s_mm'][1]}, theta {h['theta_deg']}"
            if "near_interface" in h:
                txt += f"; {h['near_interface']} within {NEAR_MM:g} mm of the zone's surface"
            txt += ")"
        txt += "."
    if rep.get("nonfinite_cells"):
        txt += f" {rep['nonfinite_cells']} cells have a NaN or infinite velocity."
    return txt + " Details: results 'diverged'."


def attach(exc, case: Path):
    """A Diverged with the report: the message gets its summary, .report the dict (results 'diverged'), and the
    case keeps it as cfdt_diverged.json."""
    try:
        rep = report(case)
    except Exception as e:  # noqa: BLE001 (a report must never hide the divergence)
        rep = {"error": f"{type(e).__name__}: {e}"}
    new = type(exc)(str(exc) + summary(rep))
    new.report = rep
    if rep is not None:
        try:
            (Path(case) / FILE).write_text(json.dumps(rep, indent=1, default=float))
        except OSError:
            pass
    return new
