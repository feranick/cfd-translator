"""sliding.py — a rotating zone whose cells turn: OpenFOAM 14 non-conformal coupling (NCC), transient.
Version: 2026.11.18.9

An MRF zone freezes the rotor: its results depend on where the blades stand against the cutwater or the
stator vanes, and every rotor-stator interaction is lost. With rotating[].motion 'sliding' the zone's cells turn
with the rotor (a solidBody mover) and slide along the zone's cylinder, where the two sides are coupled
non-conformally. The steps (OpenFOAM's propeller tutorial; cfd.direct, 'Using NCC in OpenFOAM'):

    check(mesh, g, zone, plan, patches)    before the case is written: the zone's boundary is a surface of
                                           revolution in fluid, the rotor's walls are inside it, stationary
                                           ones outside (unless they are surfaces of revolution)
    to_sliding(case, plan, mesh, foam)     once, before the transient run (after the steady MRF start):
                                           faceZone at the zone's boundary -> createBaffles (rotor and stator
                                           patches, slip boundary conditions) -> splitBaffles ->
                                           createNonConformalCouples; MRF replaced by constant/dynamicMeshDict;
                                           the rotor's walls movingWallVelocity

The steady start (solve.transient.init) runs on the conformal mesh with MRF, as before; its solution becomes
time 0 of the sliding run. createBaffles and createNonConformalCouples run in serial before decomposePar (the
mesh is decomposed again for the transient run).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

SCALE = 1e-3


from .polymesh import MeshError


class SlidingError(MeshError):
    pass


def names(zone: str) -> dict:
    """Patch and zone names made for a sliding zone."""
    return {"group": f"{zone}_slide", "face_zone": f"{zone}_interface", "rotor": f"{zone}_rotor",
            "stator": f"{zone}_stator"}


def interface(mesh, cells) -> tuple[np.ndarray, np.ndarray]:
    """The internal faces between the zone's cells and the rest, and their flip (True where the owner is outside
    the zone): createBaffles then gives the zone's side to the 'owner' (rotor) patch."""
    inz = np.zeros(mesh.ncells, bool)
    inz[np.asarray(cells, np.int64)] = True
    ni = mesh.ninternal
    a, b = inz[mesh.owner[:ni]], inz[mesh.neighbour]
    f = np.nonzero(a != b)[0]
    return f, ~a[f]


def _frame(z: dict):
    a = np.asarray(z["axis"], float)
    a = a / np.linalg.norm(a)
    return a, np.asarray(z.get("center", [0, 0, 0]), float)


def _crossing(Cf, Sf, a, o) -> np.ndarray:
    """|sin| of the angle between each face's normal and the plane through the axis (0: the face is part of a
    surface of revolution, which the rotation moves along itself)."""
    t = np.cross(a, Cf - o)
    tn = np.linalg.norm(t, axis=1)
    sn = np.linalg.norm(Sf, axis=1)
    ok = (tn > 1e-9 * max(float(tn.max()) if len(tn) else 0.0, 1e-30)) & (sn > 0)
    out = np.zeros(len(Cf))
    out[ok] = np.abs(np.einsum("ij,ij->i", Sf[ok], t[ok])) / (sn[ok] * tn[ok])
    return out


def _places(Cf, w, n=3) -> str:
    k = np.argsort(-w)[:n]
    return ", ".join(str(np.round(Cf[i], 1).tolist()) for i in k)


def ring_ratio(Cf, Lin, Lout, a, o, h: float) -> dict:
    """The worst cell-size ratio a rotor cell meets as it turns: interface faces grouped in rings about the axis
    (axial position and radius in bins of the fine cell size h); in each ring every zone-side cell passes every
    other-side cell over a revolution. S50 (2026.11.04): 0.25 mm boxes at the vane tips at fixed angles in 1.25 mm
    cells looked fine facing each other at t = 0."""
    d = Cf - o
    sx = d @ a
    rr = np.linalg.norm(d - np.outer(sx, a), axis=1)
    h = max(h, 1e-9)
    key = np.floor(sx / h).astype(np.int64) * 1_000_003 + np.floor(rr / h).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    in_max, in_min = np.full(n, 0.0), np.full(n, np.inf)
    out_max, out_min = np.full(n, 0.0), np.full(n, np.inf)
    np.maximum.at(in_max, inv, Lin)
    np.minimum.at(in_min, inv, Lin)
    np.maximum.at(out_max, inv, Lout)
    np.minimum.at(out_min, inv, Lout)
    rg = np.maximum(in_max / np.maximum(out_min, 1e-300), out_max / np.maximum(in_min, 1e-300))
    face_r = rg[inv]
    k = int(np.argmax(face_r))
    return {"max": round(float(rg.max()), 2), "at_mm": np.round(Cf[k], 1).tolist(),
            "axial_mm": round(float(sx[k]), 2), "radius_mm": round(float(rr[k]), 2),
            "faces_over_2": round(float((face_r > 2).mean()), 4)}


PVF_WARN = 0.3          # OpenFOAM's fvMeshStitcher warns above this (minWarnProjectedVolumeFraction_)
PVF_REFUSE = 1.0        # spark1, 2026.11.04: 1.06 and pcorr NaN; Perlmutter sliding7: 0.54-0.64 warned, diverged


def _rotate(P, a, o, deg: float):
    """Points P turned deg degrees about the axis a through o (Rodrigues)."""
    t = np.radians(deg)
    d = P - o
    return o + d * np.cos(t) + np.cross(a, d) * np.sin(t) + np.outer(d @ a, a) * (1 - np.cos(t))


def projected_fraction(Cf, Sf, cin, cout, V, a, o, step_deg: float, top: int = 4) -> dict:
    """An estimate, before the run, of OpenFOAM's 'projected volume fraction' |sum(Sf . Cf) / 3V - 1| in the cells on
    each side of the couple once the rotor has turned. The stitcher replaces the owner side's coupled faces by their
    intersections with the neighbour's surface, so each owner cell's volume is off by about the gap between the two
    faceted surfaces (delta) times the coupled area: pvf ~ sum(delta A) / V on the owner side. delta: the distance
    from the zone's face centres, turned by the step and by angles up to about one face's width, to the plane of the
    nearest face of the other side (the two sides are the same faces at rest). Also the share of turned faces that
    find no face of the other side within half a face's width (they leave the other surface: OpenFOAM's coverage
    drops there, at the interface's ends). Owner choice: the side whose cells take less of it."""
    from scipy.spatial import cKDTree
    A = np.linalg.norm(Sf, axis=1)
    n = Sf / np.maximum(A, 1e-300)[:, None]
    w = np.sqrt(A)                                          # a face's width
    d = Cf - o
    rr = np.linalg.norm(d - np.outer(d @ a, a), axis=1)
    pitch = float(np.degrees(np.median(w / np.maximum(rr, 1e-9))))
    step = max(float(step_deg), 1e-3)
    angles = sorted({round(step, 6), *[round(float(x), 6) for x in np.linspace(0, max(step, 1.5 * pitch), 7)[1:]]})
    tree = cKDTree(Cf)
    delta = np.zeros(len(Cf))
    off = np.zeros(len(Cf), bool)
    for ang in angles:
        for sgn in (1.0, -1.0):                             # the rotor turned, or the stator seen from the rotor
            P = _rotate(Cf, a, o, sgn * ang)
            dist, j = tree.query(P)
            dn = np.abs(np.einsum("ij,ij->i", n[j], P - Cf[j]))
            lateral = np.sqrt(np.maximum(dist ** 2 - dn ** 2, 0))
            delta = np.maximum(delta, dn)
            off |= lateral > 0.75 * w[j]
    err = delta * A
    nc = len(V)
    out = {"step_deg": round(step, 4), "angles_deg": [round(x, 3) for x in angles],
           "face_pitch_deg": round(pitch, 4), "gap_max_mm": round(float(delta.max()), 5),
           "uncovered_share": round(float(off.mean()), 4)}
    for side, cells in (("zone_owner", cin), ("other_owner", cout)):
        e = np.bincount(cells, err, nc)[cells]
        frac = e / np.maximum(np.abs(V[cells]), 1e-300)
        k = np.argsort(-frac)
        places = []
        sep = 10 * float(np.median(w))
        for i in k[:2000]:
            if all(np.linalg.norm(Cf[i] - np.asarray(p_["at"])) > sep for p_ in places):
                places.append({"at": np.round(Cf[i], 2).tolist(), "pvf": round(float(frac[i]), 3)})
                if len(places) >= top:
                    break
        out[side] = {"max": round(float(frac.max()), 3), "p99": round(float(np.percentile(frac, 99)), 3),
                     "over_warn": round(float((frac > PVF_WARN).mean()), 4), "places": places}
    if off.any():
        k = np.nonzero(off)[0]
        out["uncovered_at"] = np.round(Cf[k[np.argsort(-delta[k])][:3]], 2).tolist()
    return out


def off_surface(Cf, Sf, h, a, o, R: float, s0: float, s1: float, top: int = 6) -> dict:
    """How far each interface face lies from the zone's ideal cylinder (radius R, ends at s0 and s1 along the
    axis), in cells (h: the smaller cell at the face), and how far its normal turns from that surface's normal.
    Faces belong to the nearest part: the rim, or an end face."""
    n = Sf / np.maximum(np.linalg.norm(Sf, axis=1), 1e-300)[:, None]
    d = Cf - o
    sx = d @ a
    radial = d - np.outer(sx, a)
    rr = np.linalg.norm(radial, axis=1)
    er = radial / np.maximum(rr, 1e-300)[:, None]
    big = 1e12
    d_rim = np.where((sx >= s0) & (sx <= s1), np.abs(rr - R), big)
    d_0 = np.where(rr <= R, np.abs(sx - s0), big) if s0 > -1e11 else np.full(len(Cf), big)
    d_1 = np.where(rr <= R, np.abs(sx - s1), big) if s1 < 1e11 else np.full(len(Cf), big)
    D = np.column_stack([d_rim, d_0, d_1])
    part = np.argmin(D, axis=1)
    dist = D[np.arange(len(Cf)), part]
    # beyond every part (outside the span and the radius): the distance to the nearest circular edge
    corner = dist >= big
    if corner.any():
        ds = np.minimum(np.abs(sx[corner] - s0), np.abs(sx[corner] - s1))
        dist[corner] = np.hypot(rr[corner] - R, ds)
        part[corner] = 0
    ideal = np.where((part == 0)[:, None], er, a[None, :])
    cosang = np.abs(np.einsum("ij,ij->i", n, ideal))
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    cells = dist / np.maximum(h, 1e-12)
    score = np.maximum(cells / 0.5, ang / 30.0)
    names = np.array(["rim", "end at s0", "end at s1"])
    order = np.argsort(-score)[:5000]
    places, sep = [], 10 * float(np.median(h))
    for k in order:
        if score[k] < 1:
            break
        if all(np.linalg.norm(Cf[k] - np.asarray(p["at"])) > sep for p in places):
            places.append({"at": np.round(Cf[k], 1).tolist(), "cells": round(float(cells[k]), 2),
                           "deg": round(float(ang[k]), 1), "part": str(names[part[k]]),
                           "radius_mm": round(float(rr[k]), 2), "axial_mm": round(float(sx[k]), 2)})
            if len(places) >= top:
                break
    return {"max_cells": round(float(cells.max()), 2), "max_deg": round(float(ang.max()), 1),
            "share_over_half_cell": round(float((cells > 0.5).mean()), 4),
            "share_over_30_deg": round(float((ang > 30).mean()), 4), "places": places}


COVER_RATIO = 2.0        # a part of the zone's cylinder in cells over this many times its finest band's: uncovered


def _contains(r: dict, P: np.ndarray) -> np.ndarray:
    """Which points P (mm) a mesh.refine region (box, cylinder or hollow cylinder, sphere) holds."""
    if "box" in r:
        b = np.asarray(r["box"], float)
        return np.all((P >= b[:3]) & (P <= b[3:]), axis=1)
    if "cylinder" in r:
        c = r["cylinder"]
        ca = np.asarray(c["axis"], float)
        ca = ca / np.linalg.norm(ca)
        dd = P - np.asarray(c.get("center", [0, 0, 0]), float)
        ax = dd @ ca
        rad = np.linalg.norm(dd - np.outer(ax, ca), axis=1)
        s0, s1 = c.get("span", [-np.inf, np.inf])
        return ((rad <= float(c["radius"])) & (rad >= float(c.get("inner_radius", 0) or 0)) &
                (ax >= float(s0)) & (ax <= float(s1)))
    if "at" in r:
        return np.linalg.norm(P - np.asarray(r["at"], float), axis=1) <= float(r["r"])
    return np.zeros(len(P), bool)


def refine_coverage(zone: dict, refine: list | None, ratio: float = COVER_RATIO) -> dict | None:
    """Does mesh.refine cover the sliding zone's cylinder (its side and both ends) with its finest band all over?
    A point counts as covered by a region when the region holds it and the points 2 of the region's cells either
    side across the surface. Returns None when it does (or when no band reaches the cylinder: the interface is then
    even at the zone surface's own level), else {finest_mm, side: [{axial_mm: [from, to], size_mm}], ends: [{end,
    axial_mm, r_mm: [from, to], size_mm}]} for the parts in cells more than ratio x coarser (size_mm None: no region,
    the zone surface's own level).
    S50 noboth with ESI (spark2, job 643, 2026-11-17): rc_noboth's refinements, made for the MRF zone, left the
    side from z -82.25 to -77.1 and the floor in 1.25 mm cells while the rest of the side was 0.25 mm; the interface
    had unmatched faces, a stray end face 90 degrees off and 141 one-cell floor faces, and the run diverged at its
    second sliding step. The hand-built sliding7 mesh (0.125 mm ring r 59.2-60.2 over the whole span, 0.25 mm bands
    over both ends) ran ESI at 1 degree on all five designs."""
    if zone.get("region") or "span" not in zone or "radius" not in zone:
        return None
    regs = [(i, r, float(r["size"])) for i, r in enumerate(refine or [])
            if isinstance(r, dict) and "distance" not in r and isinstance(r.get("size"), (int, float)) and r["size"] > 0]
    if not regs:
        return None
    a, o = _frame(zone)
    e1, e2 = _basis(a)
    R = float(zone["radius"])
    s0, s1 = map(float, zone["span"])
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    D = np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2)            # (16, 3) radial directions
    ds = min(0.25, (s1 - s0) / 100)
    s = np.unique(np.concatenate([np.arange(s0, s1, ds), [s1]]))
    rr = np.linspace(0, R, 41)

    def sizes(P, n):
        out = np.full(len(P), np.inf)
        for _, r, h in regs:
            m = 2 * h
            ok = _contains(r, P) & _contains(r, P + m * n) & _contains(r, P - m * n)
            out[ok] = np.minimum(out[ok], h)
        return out
    P = (o + np.outer(s, a))[:, None, :] + R * D[None, :, :]            # side: (len(s), 16, 3)
    side = sizes(P.reshape(-1, 3), np.tile(D, (len(s), 1))).reshape(len(s), len(th)).max(axis=1)
    ends = {}
    for nm, se in (("start", s0), ("end", s1)):
        Q = (o + se * a) + rr[:, None, None] * D[None, :, :]             # (41, 16, 3)
        ends[nm] = (se, sizes(Q.reshape(-1, 3), np.tile(a, (Q.shape[0] * Q.shape[1], 1))).reshape(len(rr), len(th))
                    .max(axis=1))
    fin = np.concatenate([side] + [v for _, v in ends.values()])
    if not np.isfinite(fin).any():
        return None
    finest = float(fin[np.isfinite(fin)].min())
    lim = ratio * finest * (1 + 1e-9)
    ax0 = float(o @ a)

    def coarsest(v):
        return None if not np.isfinite(v).all() else float(v.max())
    out = {"finest_mm": finest, "side": [], "ends": []}
    bad = side > lim
    k = 0
    while k < len(s):
        if not bad[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(s) and bad[j + 1]:
            j += 1
        out["side"].append({"axial_mm": [round(float(ax0 + s[k]), 3), round(float(ax0 + s[j]), 3)], "size_mm": coarsest(side[k:j + 1])})
        k = j + 1
    for nm, (se, v) in ends.items():
        b = v > lim
        if b.any():
            out["ends"].append({"end": nm, "axial_mm": round(float(ax0 + se), 3),
                                "r_mm": [round(float(rr[b].min()), 2), round(float(rr[b].max()), 2)],
                                "size_mm": coarsest(v[b])})
    return out if out["side"] or out["ends"] else None


def coverage_message(name: str, cov: dict, axis_name: str = "the axis", motion: str = "sliding") -> str:
    def sz(x):
        return f"{x:g} mm cells" if x is not None else "no band (the zone surface's own level)"
    parts = [f"the side from {axis_name} {p['axial_mm'][0]:g} to {p['axial_mm'][1]:g} ({sz(p['size_mm'])})"
             for p in cov["side"][:4]]
    parts += [f"the {e['end']} face at {axis_name} {e['axial_mm']:g}, r {e['r_mm'][0]:g}-{e['r_mm'][1]:g} "
              f"({sz(e['size_mm'])}; an end inside a wall does not matter)" for e in cov["ends"]]
    why = ("The interface's cells must match all along and around it: where it crosses coarser cells it gets unmatched "
           "faces and stray end faces (S50 noboth with ESI diverged at its second step this way)" if motion == "sliding"
           else "The frozen-rotor zone's surface is where the relative and absolute frames meet; where it crosses "
           "coarser cells the transfer is smeared (S50 noboth, MRF: 2026 W on rc_noboth's MRF refinements, 8.26 M cells; "
           "1671 W with sliding7's interface set, 13.03 M cells, at the same head within 0.75 m)")
    return (f"rotating {name!r}: mesh.refine leaves part of the {motion if motion == 'sliding' else 'MRF'} zone's "
            f"cylinder outside its finest band ({cov['finest_mm']:g} mm): {'; '.join(parts)}. {why}. Give the zone "
            "interface_refine: <band size mm> (a 0.5x ring at its radius and a band over its side and both ends, as "
            "sliding7), or cover the whole side with a hollow mesh.refine cylinder on the axis (inner_radius; the radius "
            "+- 2 cells or more) over the full span, and each end with a cylinder 2 cells or more either side of it")


IFR_CELLS = 4            # interface_refine: the band reaches this many band cells either side of the zone's surface


def interface_regions(zone: dict) -> list:
    """mesh.refine regions for rotating[].interface_refine (h, mm): sliding7's interface set (the hand-built S50 mesh
    that ran ESI at 1 degree on five designs and gave the reliable frozen rotor 648) scaled to h:
    - a ring of h/2 cells, radius +- 2 h, over the span and 4 h beyond each end (sliding7: 0.125 mm, r 59.2-60.2);
    - a band of h cells, radius +- 5 h, over the same span (sliding7: 0.25 mm, r 58.5-61);
    - a cylinder of h cells over each end, out to radius + 5 h, 4 h either side of it (sliding7: 0.25 mm over the
      floor, -83.5 to -81, and the top, -47.5 to -45).
    [] when the zone has no interface_refine, or is not a cylinder."""
    h = zone.get("interface_refine")
    if not isinstance(h, (int, float)) or isinstance(h, bool) or h <= 0 or zone.get("region") \
            or "span" not in zone or "radius" not in zone:
        return []
    h = float(h)
    R = float(zone["radius"])
    s0, s1 = map(float, zone["span"])
    c = [float(x) for x in zone.get("center", [0, 0, 0])]
    ax = [float(x) for x in zone["axis"]]
    e = IFR_CELLS * h

    def cyl(r, span, size, inner=None):
        d = {"center": c, "axis": ax, "radius": round(r, 6), "span": [round(span[0], 6), round(span[1], 6)]}
        if inner is not None:
            d["inner_radius"] = round(max(0.0, inner), 6)
        return {"cylinder": d, "size": size}
    return [cyl(R + 2 * h, [s0 - e, s1 + e], h / 2, R - 2 * h),
            cyl(R + 5 * h, [s0 - e, s1 + e], h, R - 5 * h),
            cyl(R + 5 * h, [s0 - e, s0 + e], h),
            cyl(R + 5 * h, [s1 - e, s1 + e], h)]


def with_interface_regions(study: dict) -> dict:
    """The study with each rotating zone's interface_refine regions in mesh.refine (once: regions already there are
    not added again). The study itself is not changed."""
    if not isinstance(study, dict):
        return study
    extra = [r for z in study.get("rotating") or [] if isinstance(z, dict) for r in interface_regions(z)]
    m = study.get("mesh")
    if not extra or not isinstance(m, dict):
        return study
    have = list(m.get("refine") or []) if isinstance(m.get("refine") or [], list) else None
    if have is None:
        return study
    add = [r for r in extra if r not in have]
    if not add:
        return study
    return dict(study, mesh=dict(m, refine=have + add))


def not_round_regions(refine: list, Cf, a, o) -> list:
    """Refinement regions (mesh.refine) that reach the interface's faces Cf and are not round about the axis
    (boxes; cylinders or spheres off the axis or tilted). Distance bands follow the walls and are left out."""
    out = []
    for i, r in enumerate(refine):
        if not isinstance(r, dict) or "distance" in r:
            continue
        size = float(r.get("size", 0))
        m = 2 * size
        if "box" in r:
            b = np.asarray(r["box"], float)
            inside = np.all((Cf >= b[:3] - m) & (Cf <= b[3:] + m), axis=1)
            shape, round_ = "box", False
        elif "cylinder" in r:
            c = r["cylinder"]
            ca = np.asarray(c["axis"], float)
            ca = ca / np.linalg.norm(ca)
            cc = np.asarray(c.get("center", [0, 0, 0]), float)
            dd = Cf - cc
            ax = dd @ ca
            rad = np.linalg.norm(dd - np.outer(ax, ca), axis=1)
            s0, s1 = c.get("span", [-1e9, 1e9])
            inside = (rad <= float(c["radius"]) + m) & (rad >= float(c.get("inner_radius", 0) or 0) - m) & \
                (ax >= s0 - m) & (ax <= s1 + m)
            off = cc - o
            round_ = bool(abs(ca @ a) > 1 - 1e-6 and
                          np.linalg.norm(off - (off @ a) * a) < 1e-3 * max(float(c["radius"]), 1.0))
            shape = "cylinder"
        elif "at" in r:
            cc = np.asarray(r["at"], float)
            inside = np.linalg.norm(Cf - cc, axis=1) <= float(r["r"]) + m
            off = cc - o
            round_ = bool(np.linalg.norm(off - (off @ a) * a) < 1e-3 * max(float(r["r"]), 1.0))
            shape = "sphere"
        else:
            continue
        if inside.any() and not round_:
            out.append({"index": i, "shape": shape, "size": size, "faces": int(inside.sum())})
    return out


# The hold-back of 2026.11.14.7, lifted: the backward flow beside the S50's vane tips at the start (-59 m/s against a
# mesh speed of +18) is physical start-up flow, fluid displaced over the tips by vanes starting at full speed. ESI
# v2506's cyclicAMI gives -55 m/s at the same point in its first step (Perlmutter 61358002), and the v14 band
# disappears without vane displacement (STATIC) and flips with the rotation (REVERSE). The S50's v14 blow-up was the
# sliding rate (8.5 interface cells a step), which cells_per_step caps.
NCC_UNDER_INVESTIGATION = ""
# every sliding result: the start from rest at full speed
STARTUP_NOTE = ("sliding mesh started from rest at full speed: for the first steps the flow beside the blade tips and "
                "in narrow gaps runs at tens of m/s (the blades displace the fluid at once; the S50: -55 m/s beside the "
                "vane tips in the first step, in both OpenFOAM 14 and ESI cyclicAMI), decaying over the first "
                "revolution. The averages must start after it (average_from at 1 revolution or more)")
CELLS_PER_STEP_NOTE = 2.0   # interface cells the rotor may pass per step (solve.transient.cells_per_step)
END_SHARE_WARN = 0.9     # the share of an end face's interface faces that should lie exactly on it


def end_faces(Cf, Sf, a, o, span, h, tol_mm: float = 1e-3) -> list:
    """For each end face of the zone (span [s0, s1] along the axis): the interface faces near it (normal along the
    axis, centre within half a cell of it) and how many lie exactly on it (within tol_mm, a micrometre: points snapped
    onto an STL end face carry its single-precision coordinates, -79.0 as -79.0000036; the S50 port's precheck of
    job 61295862 counted 2 of 222,720 faces at 1e-6 mm, all of them at 1e-4): grid faces there, not snapped ones."""
    if not span:
        return []
    n = Sf / np.maximum(np.linalg.norm(Sf, axis=1), 1e-300)[:, None]
    sx = (Cf - o) @ a
    axial = np.abs(n @ a) > 0.9
    out = []
    for s_ in span:
        s_ = float(s_)
        if abs(s_) > 1e11:
            continue
        near = axial & (np.abs(sx - s_) < 0.5 * h)
        exact = near & (np.abs(sx - s_) < tol_mm)
        k = int(near.sum())
        rec = {"s_mm": s_, "faces_near": k, "faces_exact": int(exact.sum()),
               "share_exact": float(f"{exact.sum() / k:.4g}") if k else None}
        if k:
            # the plane most of them share: a face lying flat on a grid plane next to the asked one is conformal
            # too (S50 v14 port, job 61295862: 2 faces exactly on -79.0 while the rest sat on one plane near it)
            zz, cnt = np.unique(np.round(sx[near], 4), return_counts=True)
            j = int(np.argmax(cnt))
            rec.update(plane_mm=float(zz[j]), plane_offset_mm=round(float(zz[j] - s_), 4),
                       share_on_plane=float(f"{cnt[j] / k:.4g}"))
        out.append(rec)
    return out


END_GAP_MIN = 2          # whole cells between an end face of the zone and the nearest wall along the axis, each side
END_GAP_SEARCH = 4       # cells walked from the end face (more: reported as END_GAP_SEARCH + 1, 'no wall near')


def _toward(mesh, g, cells_mask, a):
    """For each cell in cells_mask: its face whose outward normal points most along +a and most along -a, as
    (face, the cell beyond it or -1) arrays indexed by cell (-1 where not computed)."""
    ni, nf = mesh.ninternal, len(mesh.owner)
    own, nei = mesh.owner, mesh.neighbour
    sel_b = cells_mask[own[ni:]]
    sel_i = cells_mask[own[:ni]] | cells_mask[nei]
    fi = np.nonzero(sel_i)[0]
    fb = ni + np.nonzero(sel_b)[0]
    n = g["Sf"] / np.maximum(np.linalg.norm(g["Sf"], axis=1), 1e-300)[:, None]
    na = n @ a
    cells = np.concatenate([own[fi], nei[fi], own[fb]])
    faces = np.concatenate([fi, fi, fb])
    score = np.concatenate([na[fi], -na[fi], na[fb]])           # the outward normal's component along a
    other = np.concatenate([nei[fi], own[fi], np.full(len(fb), -1, np.int64)])
    keep = cells_mask[cells]
    cells, faces, score, other = cells[keep], faces[keep], score[keep], other[keep]
    out = []
    for sgn in (1.0, -1.0):
        o = np.lexsort((-sgn * score, cells))
        first = np.unique(cells[o], return_index=True)[1]
        pick = o[first]
        bf = np.full(mesh.ncells, -1, np.int64)
        bc = np.full(mesh.ncells, -1, np.int64)
        bf[cells[pick]] = faces[pick]
        bc[cells[pick]] = other[pick]
        out.append((bf, bc))
    return out


STRAY_FRACTION = 0.1     # an axial interface face this far (in its cells' height along the axis) off every end plane


def _axial(mesh, g, f, inz, a, o) -> dict | None:
    """The zone's interface faces whose normal lies along the axis: face labels, position along the axis, which way
    the stator side lies, the cells on each side, the cells' height along the axis (volume over the face's area, the
    smaller side), and the faces' centres."""
    Sf, Cf = g["Sf"][f], g["Cf"][f]
    A = np.linalg.norm(Sf, axis=1)
    n = Sf / np.maximum(A, 1e-300)[:, None]
    zown = inz[mesh.owner[f]]
    n_out = np.where(zown[:, None], n, -n)                       # from the zone into the rest
    ax = np.abs(n_out @ a) > 0.9
    if not ax.any():
        return None
    fa = f[ax]
    cin = np.where(zown[ax], mesh.owner[fa], mesh.neighbour[fa])
    cout = np.where(zown[ax], mesh.neighbour[fa], mesh.owner[fa])
    V = np.abs(g["V"])
    h = np.minimum(V[cin], V[cout]) / np.maximum(A[ax], 1e-300)
    return {"f": fa, "s": (Cf[ax] - o) @ a, "up": np.sign(n_out[ax] @ a), "cin": cin, "cout": cout, "h": h,
            "Cf": Cf[ax]}


def _planes_used(s, h, planes) -> list:
    """The planes the end faces really lie on: each intended plane, and the plane most of the axial faces within a
    cell of it share when that is a different one (a conformal end face on the grid plane next to the asked one)."""
    out = []
    for p_ in planes:
        out.append(float(p_))
        near = np.abs(s - p_) < np.maximum(h, 1e-3)
        if near.any():
            zz, cnt = np.unique(np.round(s[near], 4), return_counts=True)
            k = int(np.argmax(cnt))
            if cnt[k] >= 0.5 * near.sum() and abs(zz[k] - p_) > 1e-3:
                out.append(float(zz[k]))
    return out


def _match(s, h, used) -> np.ndarray:
    """Index into used of the plane each face lies on (within STRAY_FRACTION of its cells' height, at least a
    micrometre), -1 for none."""
    tol = np.maximum(STRAY_FRACTION * h, 1e-3)
    d = np.abs(s[:, None] - np.asarray(used)[None, :])
    k = np.argmin(d, axis=1)
    return np.where(d[np.arange(len(s)), k] <= tol, k, -1)


def _place(c, a, o, e1, e2) -> dict:
    d = c - o
    rad = d - (d @ a) * a
    return {"r_mm": round(float(np.linalg.norm(rad)), 3), "s_mm": round(float(d @ a), 3),
            "theta_deg": round(float(np.degrees(np.arctan2(rad @ e2, rad @ e1))) % 360, 1)}


def _basis(a):
    """Two unit vectors across the axis: x (or y) with its axial part removed, then a x e1; for the z axis x and y,
    so theta is atan2(y, x)."""
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - (t @ a) * a
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def stray_end_faces(mesh, g, f, inz, a, o, planes, top: int = 8) -> dict:
    """Interface faces facing along the axis that lie on none of the zone's end planes (more than STRAY_FRACTION of
    their cells' height off each): an end face's edge dragged by snapping. S50 v14 port, mesh 621 (spark2,
    2026-11-16): 15 such 'planes' between z -46.11 and -46.27 with a few faces each, at the top rim corner (r
    59.60-59.62), where 286 of the mesh's 343 faces over 65 degrees of non-orthogonality also were; the coupling's
    coverage was 0 to 1.95 at rest and the run diverged within 17 steps. planes: the intended end planes (the zone's
    span ends, a stepped zone's step)."""
    ax = _axial(mesh, g, f, inz, a, o)
    rec = {"planes": [float(p_) for p_ in planes], "faces": 0, "places": []}
    if ax is None:
        return rec
    used = _planes_used(ax["s"], ax["h"], planes)
    k = _match(ax["s"], ax["h"], used)
    bad = np.nonzero(k < 0)[0]
    rec["planes_used"] = used
    rec["faces"] = int(len(bad))
    if len(bad):
        d = np.min(np.abs(ax["s"][bad][:, None] - np.asarray(used)[None, :]), axis=1)
        rec["max_offset_mm"] = round(float(d.max()), 4)
        rec["max_offset_cells"] = round(float((d / np.maximum(ax["h"][bad], 1e-300)).max()), 2)
        e1, e2 = _basis(a)
        for i in bad[np.argsort(-d)][:top]:
            q = _place(ax["Cf"][i], a, o, e1, e2)
            q["offset_mm"] = round(float(np.min(np.abs(ax["s"][i] - np.asarray(used)))), 4)
            rec["places"].append(q)
    return rec


def end_gap_cells(mesh, g, f, inz, wall_faces, a, o, planes=None, top: int = 6) -> list:
    """Whole cells between each end face of the zone and the nearest wall along the axis, on each side: from the cell
    on the end face, cell by cell through the face facing along the axis, until a wall face (1: the end face's own
    cell touches the wall). planes: the intended end planes (the zone's span ends, a stepped zone's step); the
    interface faces facing along the axis on them (or on the grid plane next to one) are counted, the others are
    stray_end_faces' business. Without planes every cluster of such faces counts as a plane (as before 2026.11.16.2).
    The S50 v14 port's long-cell mesh with 2 mm z bands (OGRID_ANISO, spark2, 2026-11-16) had its top rim corner
    dragged by snapping (coverage 0 to 1.95 at rest, projected volume fraction 0.344; diverged within 17 steps); the
    level-5 mesh had 0.125 mm cells there and coupled at 1/1/1."""
    ni = mesh.ninternal
    ax = _axial(mesh, g, f, inz, a, o)
    if ax is None:
        return []
    if planes is not None:
        used = _planes_used(ax["s"], ax["h"], planes)
        k = _match(ax["s"], ax["h"], used)
        # a grid plane next to an intended one reports under the intended one
        lut = [min(range(len(planes)), key=lambda j: abs(float(planes[j]) - u)) for u in used]
        label = np.where(k >= 0, np.asarray(lut + [-1])[k], -1)
        keep = label >= 0
        group = label[keep]
        names = [float(p_) for p_ in planes]
    else:
        keep = np.ones(len(ax["s"]), bool)
        rounded = np.round(ax["s"], 2)
        names = [float(v) for v in np.unique(rounded)]
        group = np.searchsorted(np.asarray(names), rounded)
    if not keep.any():
        return []
    cin, cout, up, Cfa = ax["cin"][keep], ax["cout"][keep], ax["up"][keep], ax["Cf"][keep]
    # the cells the walk can reach: END_GAP_SEARCH layers around the end faces' cells
    near = np.zeros(mesh.ncells, bool)
    near[cin] = near[cout] = True
    own, nei = mesh.owner[:ni], mesh.neighbour
    for _ in range(END_GAP_SEARCH):
        m = near[own] | near[nei]
        near[own[m]] = True
        near[nei[m]] = True
    (pf, pc), (mf, mc) = _toward(mesh, g, near, a)
    wall = np.zeros(len(mesh.owner), bool)
    if len(wall_faces):
        wall[np.asarray(wall_faces, np.int64)] = True
    e1, e2 = _basis(a)

    def walk(start, along_plus):
        cnt = np.full(len(start), END_GAP_SEARCH + 1, np.int64)
        c = start.copy()
        live = np.ones(len(start), bool)
        for k_ in range(1, END_GAP_SEARCH + 1):
            bf = np.where(along_plus, pf[c], mf[c])
            bc = np.where(along_plus, pc[c], mc[c])
            hit = live & (bf >= 0) & wall[np.maximum(bf, 0)]
            cnt[hit] = k_
            live &= ~hit & (bc >= 0) & near[np.maximum(bc, 0)]
            c = np.where(live, bc, c)
            if not live.any():
                break
        return cnt

    sides = {"stator_side": walk(cout, up > 0), "rotor_side": walk(cin, up < 0)}
    out = []
    for gi, pz in enumerate(names):
        on = group == gi
        if not on.any():
            continue
        rec = {"s_mm": pz, "faces": int(on.sum())}
        for side_, cnt in sides.items():
            cs = cnt[on]
            bad = np.nonzero(on)[0][cs < END_GAP_MIN]
            r_ = {"min_cells": int(cs.min()), "faces_under_min": int(len(bad)), "places": []}
            for i in bad[:top]:
                q = _place(Cfa[i], a, o, e1, e2)
                r_["places"].append({k_: round(v, 2) if k_ != "theta_deg" else v for k_, v in q.items()})
            rec[side_] = r_
        out.append(rec)
    return out


THIN_LAYERS = 2          # cells from the interface searched for snapped slivers at walls
THIN_WARN = 0.35         # a wall cell thinner than this fraction of the interface's cells: noted
THIN_UNRELIABLE = 0.2    # ... thinner than this: the run is unreliable (S50 v14 port: 0.02 mm against 0.12, diverged)


def thin_wall_cells(mesh, g, f, inz, wall_faces, a, o, top: int = 8) -> dict:
    """Snapped slivers at walls near the interface, per side: cells within THIN_LAYERS cells of the interface that
    touch a wall, with their thickness h (volume over largest face area) against the median h of the cells on the
    interface (h_ref). The S50 v14 port's sliding runs diverged from such cells, where small wall features (a
    bore-entry taper, a sloped floor at the bore's exit, rounded vane tips) lay within two cells of the interface."""
    ni = mesh.ninternal
    V = np.abs(g["V"])
    Af = np.linalg.norm(g["Sf"], axis=1)
    amax = np.zeros(mesh.ncells)
    np.maximum.at(amax, mesh.owner, Af)
    np.maximum.at(amax, mesh.neighbour, Af[:ni])
    h = V / np.maximum(amax, 1e-300)
    wall = np.zeros(mesh.ncells, bool)
    if len(wall_faces):
        wall[mesh.owner[np.asarray(wall_faces, np.int64)]] = True
    own, nei = mesh.owner[:ni], mesh.neighbour
    out = {"layers": THIN_LAYERS, "warn_fraction": THIN_WARN, "unreliable_fraction": THIN_UNRELIABLE}
    worst = 1.0
    for side_, mine in (("rotor_side", inz), ("stator_side", ~inz)):
        first = np.where(mine[mesh.owner[f]], mesh.owner[f], mesh.neighbour[f])
        layer = np.zeros(mesh.ncells, bool)
        layer[first] = True
        for _ in range(THIN_LAYERS - 1):
            nxt = layer.copy()
            m = (layer[own] | layer[nei]) & mine[own] & mine[nei]
            nxt[own[m]] = True
            nxt[nei[m]] = True
            layer = nxt
        h_ref = float(np.median(h[first])) if len(first) else 0.0
        cand = np.nonzero(layer & wall)[0]
        rec = {"cells_at_walls": int(len(cand)), "h_ref_mm": float(f"{h_ref:.4g}"), "thin": 0, "places": []}
        if len(cand) and h_ref > 0:
            hc = h[cand]
            rec["h_min_mm"] = float(f"{hc.min():.4g}")
            rec["h_min_fraction"] = float(f"{hc.min() / h_ref:.3g}")
            worst = min(worst, hc.min() / h_ref)
            bad = cand[hc < THIN_WARN * h_ref]
            rec["thin"] = int(len(bad))
            e1, e2 = _basis(a)
            for c in bad[np.argsort(h[bad])][:top]:
                d = g["C"][c] - o
                sx = float(d @ a)
                rad = d - sx * a
                rec["places"].append({"h_mm": float(f"{h[c]:.3g}"), "at_mm": np.round(g["C"][c], 2).tolist(),
                                      "r_mm": round(float(np.linalg.norm(rad)), 2), "s_mm": round(sx, 2),
                                      "theta_deg": round(float(np.degrees(np.arctan2(rad @ e2, rad @ e1))) % 360, 1)})
        out[side_] = rec
    out["worst_fraction"] = float(f"{worst:.3g}")
    out["unreliable"] = bool(worst < THIN_UNRELIABLE)
    return out


def check(mesh, g, zone: dict, plan: dict, patches: list, refine: list | None = None) -> None:
    """Can this zone slide? Raises SlidingError with what to change; small deviations become plan notes.
    Records plan['sliding'] (names, interface size, moving walls)."""
    zn = zone["name"]
    cells = mesh.zones[zn]
    inz = np.zeros(mesh.ncells, bool)
    inz[cells] = True
    a, o = _frame(zone)
    notes = plan.setdefault("notes", [])
    n0 = len(notes)
    try:
        _check(mesh, g, zone, plan, patches, refine, cells, inz, a, o, notes)
    finally:
        if plan.get("backend") == "esi":          # cyclicAMI: the NCC-only notes go, the rest lose NCC examples
            from .esi import adapt_notes
            notes[n0:] = adapt_notes(notes[n0:])


def _check(mesh, g, zone, plan, patches, refine, cells, inz, a, o, notes) -> None:
    from .case import _same_rotation, rotation_key
    zn = zone["name"]
    # 1. the boundary between the zone and the rest: a surface of revolution about the axis
    f, _flip = interface(mesh, cells)
    if not len(f):
        raise SlidingError(f"sliding zone {zn!r} has no internal boundary with the rest of the fluid (it fills the "
                           "domain, or touches the rest only through walls): it must be a cylinder in the fluid")
    Sf, Cf = g["Sf"][f], g["Cf"][f]
    A = np.linalg.norm(Sf, axis=1)
    sn = _crossing(Cf, Sf, a, o)
    mean_deg = float(np.degrees(np.arcsin(min((sn * A).sum() / A.sum(), 1.0))))
    steep = sn > np.sin(np.radians(20.0))
    steep_frac = float(A[steep].sum() / A.sum())
    if mean_deg > 5.0 or steep_frac > 0.01:
        raise SlidingError(
            f"sliding zone {zn!r}: its boundary is not a surface of revolution about the axis (faces lean "
            f"{mean_deg:.1f} degrees on average from it, {steep_frac:.1%} of its area more than 20 degrees, e.g. at "
            f"{_places(Cf, sn * A)} mm): the cells could not slide along it. The zone's cylinder must lie in open "
            "fluid, clear of walls (snappyHexMesh snaps its faces to the cylinder only there); check center, "
            "radius and span against the casing, and refine where it passes through narrow gaps")
    # the cells on both sides of the couple: the stitcher intersects their faces and corrects the mesh fluxes in
    # them. S50 on spark1 (2026.11.04): 0.2 mm cells on one side of the zone's end, 0.5 mm on the other; OpenFOAM
    # warned 'maximum projected volume fraction ... may cause instability', the fraction grew from 0.3 to 1.06 in
    # 0.26 degrees of rotation and the flux correction (pcorr) gave NaN. Report their quality and size match
    from .polymesh import face_skewness
    side = np.unique(np.concatenate([mesh.owner[f], mesh.neighbour[f]]))
    near = np.zeros(mesh.ncells, bool)
    near[side] = True
    ni = mesh.ninternal
    fi = np.nonzero(near[mesh.owner[:ni]] | near[mesh.neighbour])[0]
    q_if = {}
    if len(fi):
        d = g["C"][mesh.neighbour[fi]] - g["C"][mesh.owner[fi]]
        S = g["Sf"][fi]
        cosang = np.einsum("ij,ij->i", d, S) / np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(S, axis=1), 1e-300)
        no = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        k = int(np.argmax(no))
        sk = face_skewness(g["Cf"][fi], S, g["C"], mesh.owner[fi], mesh.neighbour[fi], len(fi))
        j = int(np.argmax(sk))
        V = np.abs(g["V"])
        vin = np.where(inz[mesh.owner[f]], V[mesh.owner[f]], V[mesh.neighbour[f]])          # zone side
        vout = np.where(inz[mesh.owner[f]], V[mesh.neighbour[f]], V[mesh.owner[f]])
        Lin, Lout = np.cbrt(vin), np.cbrt(vout)
        ratio = np.maximum(Lin, Lout) / np.maximum(np.minimum(Lin, Lout), 1e-300)    # cell size, facing now
        r = int(np.argmax(ratio))
        ring = ring_ratio(g["Cf"][f], Lin, Lout, a, o, float(np.median(np.minimum(Lin, Lout))))
        q_if = {"faces": int(len(f)), "max_non_ortho": round(float(no[k]), 1), "non_ortho_at": np.round(g["Cf"][fi[k]], 1).tolist(),
                "over_70": int((no > 70).sum()), "max_skewness": round(float(sk[j]), 2),
                "skewness_at": np.round(g["Cf"][fi[j]], 1).tolist(),
                "size_ratio_max": round(float(ratio[r]), 2), "size_ratio_p95": round(float(np.percentile(ratio, 95)), 2),
                "size_ratio_at": np.round(g["Cf"][f[r]], 1).tolist(),
                "mean_cell_mm": {"zone_side": round(float(np.cbrt(vin.mean())), 3),
                                 "other_side": round(float(np.cbrt(vout.mean())), 3)}}
        frac_big = float((ratio > 2.0).mean())
        q_if["size_ratio_over_2"] = round(frac_big, 4)
        q_if["ring"] = ring
        # how round the interface is, and where it leans (S50 6th try, 2026.11.04: the projected volume fraction
        # rose with the rotation until the rotor had turned one 0.25 mm cell, then k went negative and pcorr NaN;
        # the stitcher reported coupled faces at up to 117 degrees to each other: the two sides stop matching
        # where the snapped surface is not a surface of revolution)
        s0, s1 = zone.get("span", [-1e12, 1e12])
        off = off_surface(g["Cf"][f], g["Sf"][f], np.minimum(Lin, Lout), a, o, float(zone["radius"]),
                          float(s0), float(s1))
        q_if["off_surface"] = off
        # the stitcher's volume error before the run: which side should own the couple, and whether it is safe
        cin = np.where(inz[mesh.owner[f]], mesh.owner[f], mesh.neighbour[f])
        cout = np.where(inz[mesh.owner[f]], mesh.neighbour[f], mesh.owner[f])
        step_deg = float((plan.get("transient") or {}).get("degrees_per_step") or 1.0)
        pvf = projected_fraction(g["Cf"][f], g["Sf"][f], cin, cout, g["V"], a, o, step_deg)
        q_if["projected_volume_fraction"] = pvf
        # how many interface cells the rotor passes in one time step (NCC ladder, rung 1 on carbonio, 2026.11.14:
        # the propeller ran at 0.8 and 2.3 cells a step and diverged at step 3 at 8.5, the S50's rate at 1 degree)
        side_f = sn < np.sin(np.radians(20.0))
        if side_f.any():
            rr = np.linalg.norm((Cf - o) - np.outer((Cf - o) @ a, a), axis=1)[side_f]
            h_t = float(np.median(np.sqrt(A[side_f])))
            arc = float(np.median(rr)) * np.radians(step_deg)
            cps = arc / h_t
            q_if["cells_per_step"] = {"value": round(cps, 2), "degrees_per_step": step_deg,
                                      "radius_mm": round(float(np.median(rr)), 2), "cell_mm": round(h_t, 4)}
            tr_ = plan.get("transient") or {}
            lim = tr_.get("cells_per_step", CELLS_PER_STEP_NOTE) if "cells_per_step" in tr_ else CELLS_PER_STEP_NOTE
            if lim and cps > lim:
                # the step capped so that the rotor passes at most lim interface cells: the S50 v14 port ran at 1.5
                # cells per step (0.17 degrees, job 61305325) and diverged at 8.5 (1 degree); the propeller tutorial
                # passed at 2.3 and diverged at 8.5 (NCC ladder, 2026.11.14)
                new_deg = float(f"{step_deg * lim / cps:.4g}")
                trev = tr_.get("period")
                if trev:
                    if tr_.get("fixed_step"):
                        tr_["dt"] = tr_["max_dt"] = float(f"{new_deg / 360.0 * trev:.6g}")
                    else:
                        tr_["max_dt"] = float(f"{min(tr_.get('max_dt', 1e30), new_deg / 360.0 * trev):.4g}")
                        tr_["dt"] = min(tr_.get("dt", tr_["max_dt"]), tr_["max_dt"])
                    tr_["degrees_per_step"] = new_deg
                    q_if["cells_per_step"].update(capped_from_degrees=step_deg, degrees_per_step=new_deg,
                                                  value=round(cps * new_deg / step_deg, 2), limit=lim)
                    notes.append(f"sliding interface: at {step_deg:g} degrees a step the rotor would pass {cps:.1f} "
                                 f"interface cells (r {float(np.median(rr)):.1f} mm, cells about {h_t:.3g} mm); the step "
                                 f"is capped to {new_deg:g} degrees ({lim:g} cells, solve.transient.cells_per_step; "
                                 f"{360.0 / new_deg:.0f} steps a revolution): OpenFOAM 14's coupling ran the S50 at 1.5 "
                                 "and diverged at 8.5 (NCC ladder, 2026.11.14)")
            elif cps > CELLS_PER_STEP_NOTE and plan.get("backend") != "esi":
                notes.append(f"sliding interface: the rotor passes {cps:.1f} interface cells per time step "
                             f"({step_deg:g} degrees at r {float(np.median(rr)):.1f} mm, cells about {h_t:.3g} mm; "
                             "solve.transient.cells_per_step null: not capped): OpenFOAM 14's coupling ran the S50 at "
                             "1.5 and diverged at 8.5 (NCC ladder, 2026.11.14)")
        tr0 = plan.get("transient") or {}
        if tr0.get("period"):
            notes.append(STARTUP_NOTE)
            if float(tr0.get("average_from", 0.0)) < float(tr0["period"]) * (1 - 1e-9):
                notes.append(f"sliding mesh: the averages start at {tr0.get('average_from', 0.0):.4g} s, within the first "
                             f"revolution ({tr0['period']:.4g} s), so they include the start from rest: raise "
                             "solve.transient.average_from (and end_time) so that at least one revolution passes first")
        q_if["end_faces"] = end_faces(g["Cf"][f], g["Sf"][f], a, o, zone.get("span"), np.minimum(Lin, Lout))
        for ef in q_if["end_faces"]:
            if ef["faces_near"] and ef["share_exact"] < END_SHARE_WARN <= ef.get("share_on_plane", 0):
                notes.append(f"sliding zone {zn!r}: its end face s {ef['s_mm']:g} mm lies on the cells' plane at "
                             f"{ef['plane_mm']:g} mm ({ef['plane_offset_mm']:+g} mm; {ef['share_on_plane']:.1%} of its "
                             "faces): flat and conformal, only shifted")
            elif ef["faces_near"] and ef["share_exact"] < END_SHARE_WARN:
                notes.append(f"sliding zone {zn!r}: only {ef['faces_exact']} of the {ef['faces_near']} interface faces at "
                             f"its end face s {ef['s_mm']:g} mm lie exactly on it ({ef['share_exact']:.1%}): that face is "
                             "snapped from cells that do not have a face there (S50 v14 port, job 61293745: 96 faces on "
                             "its raised end face against 289,478 on the other; estimated projected volume fraction "
                             "0.75-0.88 at its corners). Put the end face on a plane of the cells around it, or refine "
                             "there so that it is one")
        wall_faces = np.concatenate([mesh.patch_faces(p_["name"]) for p_ in patches if p_.get("kind") == "wall"]
                                    or [np.zeros(0, np.int64)])
        ends = [float(v) for v in (zone.get("span") or []) if abs(float(v)) < 1e11]
        gaps = end_gap_cells(mesh, g, f, inz, wall_faces, a, o, planes=ends)
        q_if["end_gap_cells"] = gaps
        stray = stray_end_faces(mesh, g, f, inz, a, o, ends)
        q_if["stray_end_faces"] = stray
        if stray["faces"]:
            notes.append(
                f"sliding zone {zn!r}: {stray['faces']} interface faces facing along the axis lie on none of its end "
                f"planes ({', '.join(f'{v:g}' for v in ends)} mm; up to {stray['max_offset_mm']} mm, "
                f"{stray['max_offset_cells']} cells off, at " +
                "; ".join(f"r {q['r_mm']} s {q['s_mm']} theta {q['theta_deg']}" for q in stray["places"][:3]) +
                "): an end face's edge dragged by snapping, usually where it meets the cylinder next to a wall (S50 "
                "v14 port, mesh 621: the top rim corner, coverage 0 to 1.95 at rest, diverged within 17 steps). "
                "Finer cells along the axis there, or move the end face away from the wall")
        for eg in gaps:
            for side_ in ("stator_side", "rotor_side"):
                rec = eg[side_]
                if rec["faces_under_min"]:
                    notes.append(
                        f"sliding zone {zn!r}: {rec['faces_under_min']} of the {eg['faces']} faces of its end face at s "
                        f"{eg['s_mm']:g} mm have fewer than {END_GAP_MIN} whole cells between them and a wall on the "
                        f"{side_.replace('_', ' ')} (min {rec['min_cells']}, at " +
                        "; ".join(f"r {q['r_mm']} s {q['s_mm']} theta {q['theta_deg']}" for q in rec["places"][:3]) +
                        "): snapping a cell there onto the wall can drag the interface's edge (S50 v14 port, long-cell "
                        "mesh: the top rim corner next to the wear plate, coverage 0 to 1.95 at rest, diverged). "
                        "Refine there (finer cells along the axis) or move the end face away from the wall")
        thin = thin_wall_cells(mesh, g, f, inz, wall_faces, a, o)
        q_if["thin_wall_cells"] = thin
        for side_, rec in ((k_, thin[k_]) for k_ in ("rotor_side", "stator_side")):
            if rec["thin"]:
                notes.append(
                    f"sliding interface, {side_.replace('_', ' ')}: {rec['thin']} cells next to walls within "
                    f"{THIN_LAYERS} cells of the interface are thinner than {THIN_WARN:g} of the interface's cells "
                    f"(thinnest {rec['h_min_mm']} mm against {rec['h_ref_mm']} mm, at " +
                    "; ".join(f"r {q['r_mm']} s {q['s_mm']} theta {q['theta_deg']}" for q in rec["places"][:3]) +
                    "): a wall feature there (a taper, a sloped floor, a rounded edge) is snapped onto, not meshed "
                    "by whole cells. Such cells seeded the S50's NCC divergence (v14 port, 2026.11.14); keep the "
                    "interface 3 or more cells away from such features, or make them cylinders and planes")
        plan["sliding_quality"] = q_if
        notes.append(f"sliding interface cells: worst non-orthogonality {q_if['max_non_ortho']} deg, skewness "
                     f"{q_if['max_skewness']} (at {q_if['skewness_at']} mm), cell sizes across the interface differ up to "
                     f"{q_if['size_ratio_max']}x facing each other (95 % within {q_if['size_ratio_p95']}x) and up to "
                     f"{ring['max']}x anywhere around a ring as the rotor turns")
        if q_if["max_non_ortho"] > 70 or q_if["max_skewness"] > 4:
            notes.append(f"cells at the sliding interface are poor ({q_if['over_70']} faces above 70 degrees, worst "
                         f"{q_if['max_non_ortho']} at {q_if['non_ortho_at']} mm; skewness {q_if['max_skewness']} at "
                         f"{q_if['skewness_at']} mm). The interface's mesh-flux correction works on these cells: refine "
                         "there evenly on both sides of the zone's boundary, or move the boundary into regular cells")
        if q_if["size_ratio_p95"] > 2 or frac_big > 0.01:
            notes.append(f"cells on the two sides of the sliding interface differ in size ({frac_big:.1%} of its faces "
                         f"more than 2x, up to {q_if['size_ratio_max']}x at {q_if['size_ratio_at']} mm): the mismatch "
                         "puts large volume errors on the smaller cells (OpenFOAM's 'projected volume fraction'), "
                         "which can blow up at once (S50: NaN after 0.26 degrees). Refine both sides of the zone's "
                         "boundary alike")
        if off["share_over_half_cell"] > 0.005 or off["share_over_30_deg"] > 0.005:
            # a stepped interface cannot slide (S50, 2026.11.04: faces 87 degrees off the cylinder by the vane tips,
            # NaN at the 28th step): stop here rather than after hours of meshing and a solve
            raise SlidingError(
                f"sliding zone {zn!r}: snappyHexMesh did not snap the zone's boundary onto its cylinder everywhere: "
                f"{off['share_over_half_cell']:.1%} of its faces lie more than half a cell off it and "
                f"{off['share_over_30_deg']:.1%} turn more than 30 degrees from it (worst {off['max_cells']} cells, "
                f"{off['max_deg']} degrees), e.g. at " + "; ".join(f"{p_['at']} mm ({p_['cells']} cells, {p_['deg']} deg, "
                f"{p_['part']})" for p_ in off["places"][:5]) + ". Such steps stop matching the other side as soon as "
                "the rotor turns one cell. snappyHexMesh leaves the zone's faces stepped where moving them onto the "
                "cylinder would break the mesh: where the cylinder runs closer to walls than about two cells (vane "
                "tips, a casing bore). Put the zone's boundary where the gap holds 4 or more cells, or refine that "
                "gap (a hollow mesh.refine cylinder over it) until it does")
        if len(f) > 400_000:
            # Perlmutter sliding7 (2026.11.12): 1.47 M interface faces (0.9 M in a 0.125 mm band), 13 M couplings once
            # turned, ~35 of a 44 s step spent changing the mesh against ~7 s in the linear solves
            notes.append(f"the sliding interface has {len(f):,} faces: every time step re-intersects them and rebuilds "
                         "the mesh around them (sliding7: 1.47 M faces, ~35 s of a 44 s step on 512 ranks). Cells twice "
                         "as large along the interface give 4x fewer faces; a band finer than the cells it must match "
                         "on both sides costs time without helping the coupling")
        zo, oo = pvf["zone_owner"], pvf["other_owner"]
        best = min(zo["max"], oo["max"])
        own = "the rotor (zone) side" if zo["max"] <= oo["max"] else "the stator side"
        pick = zo if zo["max"] <= oo["max"] else oo
        where = "; ".join(f"{p_['pvf']} at {p_['at']} mm" for p_ in pick["places"][:3])
        if best > PVF_REFUSE and plan.get("backend") != "esi":   # cyclicAMI does not stitch cells: no such limit
            # Perlmutter sliding7 (2026.11.12): OpenFOAM warned 0.54-0.64 at every step and the run diverged after
            # 4-5 steps (Courant 62 -> 1.3e5); spark1 (2026.11.04): 1.06 and pcorr NaN. Stop before the solve
            raise SlidingError(
                f"sliding zone {zn!r}: as the rotor turns, the stitched interface would distort the cells next to it "
                f"too much: estimated projected volume fraction {best:g} with {own} owning the couple ({where}; "
                f"rotor side {zo['max']:g}, stator side {oo['max']:g}; OpenFOAM warns above {PVF_WARN:g}, runs blew up "
                f"at 0.54-1.06). The two sides' faceted surfaces lie up to {pvf['gap_max_mm']:g} mm apart once turned, "
                "which is large against the thin cells there (layers, cells in narrow gaps). Make the interface's faces "
                "smaller along it (refine the zone's cylinder evenly: a hollow mesh.refine cylinder on the axis), keep "
                "prism layers and narrow gaps away from it (the zone's radius and span), or make the cells next to it "
                "thicker")
        if best > PVF_WARN:
            notes.append(f"sliding interface: estimated projected volume fraction {best:g} with {own} owning the "
                         f"couple (rotor side {zo['max']:g}, stator side {oo['max']:g}; OpenFOAM warns above "
                         f"{PVF_WARN:g}), worst at {where}: the stitched cells there are distorted as the rotor turns; "
                         "refine the zone's cylinder evenly, or keep thin cells (layers, narrow gaps) away from it")
        if pvf["uncovered_share"] > 0.001:
            notes.append(f"{pvf['uncovered_share']:.2%} of the interface's faces, once turned, find no face of the other "
                         f"side (e.g. at {pvf.get('uncovered_at', [])[:2]} mm): the interface ends or bends there, so "
                         "the coupling's coverage drops below 1 (fluxes leak into the walls). The zone's cylinder must "
                         "end in open fluid, clear of walls and narrow gaps")
        if off["max_cells"] > 0.3 or off["max_deg"] > 30:
            notes.append(f"the sliding interface strays from its cylinder: up to {off['max_cells']} of a cell and "
                         f"{off['max_deg']} degrees, e.g. at " + "; ".join(f"{p_['at']} mm ({p_['part']})"
                                                                          for p_ in off["places"][:3]))
        if ring["max"] > 2 and ring["faces_over_2"] > 0.01:
            notes.append(f"as the rotor turns, its cells at the interface pass stator cells of other sizes: up to "
                         f"{ring['max']}x around the ring at {ring['at_mm']} mm (axial {ring['axial_mm']} mm, radius "
                         f"{ring['radius_mm']} mm; {ring['faces_over_2']:.1%} of the interface in rings over 2x). "
                         "Refinement near the interface must be round about the axis (a cylinder on the axis, all "
                         "the way round), not boxes or spheres at fixed angles")
    for w in not_round_regions(refine or [], g["Cf"][f], a, o):
        notes.append(f"mesh.refine[{w['index']}] ({w['shape']}, {w['size']} mm) touches the sliding interface "
                     f"({w['faces']} of its faces) but is not round about the axis: as the rotor turns, the cells it "
                     "refines on one side move past coarser cells on the other. Use a cylinder on the zone's axis "
                     "(all the way round) there instead")
    plan.setdefault("zones", {}).setdefault(zn, {})
    # 2. the walls: rotating ones inside, stationary ones outside (or surfaces of revolution)
    rev = set((mesh.info.get("revolution_tags") or {}).get(rotation_key(zone["axis"], zone.get("center")), []))
    zp = plan["zones"][zn]
    moving, spin = [], []
    for p in patches:
        if p["kind"] != "wall":
            continue
        pf = mesh.patch_faces(p["name"])
        if not len(pf):
            continue
        r = p["bc"].get("rotation")
        Aw = np.linalg.norm(g["Sf"][pf], axis=1)
        tot = float(Aw.sum()) or 1.0
        inside = inz[mesh.owner[pf]]
        tags = mesh.btag[pf - mesh.ninternal]
        on_rev = np.isin(tags, list(rev)) if rev else np.zeros(len(pf), bool)
        # faces moving along themselves count as surfaces of revolution too (snapped facets)
        on_rev |= _crossing(g["Cf"][pf], g["Sf"][pf], a, o) < np.sin(np.radians(6.0))
        turns = r == zn or (isinstance(r, dict) and zp.get("omega_rad_s") is not None and _same_rotation(r, zp))
        if turns:
            out = ~inside
            bad = out & ~on_rev
            fb = float(Aw[bad].sum() / tot)
            if fb > 0.01:
                raise SlidingError(
                    f"wall {p['name']!r} turns with sliding zone {zn!r} but {fb:.1%} of its area (not a surface of "
                    f"revolution) bounds cells outside the zone (e.g. at {_places(g['Cf'][pf[bad]], Aw[bad])} mm): "
                    "those faces would stand still while the rest turns. Enlarge the zone (radius, span) so that it "
                    "holds the whole rotor with at least one cell between the rotor's tips and the zone's ends; in a "
                    "narrow side gap, put the zone's end mid-gap and refine the gap (mesh.refine) to 3 cells or more")
            if bad.any():
                notes.append(f"{fb:.2%} of wall {p['name']!r} (rotor) bounds cells just outside sliding zone {zn!r}: "
                             "those faces stand still")
            fo = float(Aw[out & on_rev].sum() / tot)
            all_rev = float(Aw[~on_rev].sum() / tot) < 0.001
            if all_rev and fo > 0.001:
                # hub, shaft, shrouds only (s50 'impeller_back', 68 % outside the zone, 2026.11.03): every face moves
                # along itself, so OpenFOAM's rotatingWallVelocity (omega x r at the faces' current centres, normal
                # part removed) is right inside the zone and outside it, and its disk friction stays in the torque
                spin.append(p["name"])
                notes.append(f"wall {p['name']!r} (rotor, all surfaces of revolution) lies {fo:.0%} outside sliding "
                             f"zone {zn!r}: it gets the rotating-wall velocity everywhere (rotatingWallVelocity), so "
                             "its friction counts in the torque")
                continue
            if fo > 0.001:
                notes.append(f"{fo:.1%} of wall {p['name']!r} (rotor; its hub, shaft or shroud faces) lies outside "
                             f"sliding zone {zn!r} and stands still there (no disk friction from it in the torque): "
                             "the same bc also holds blades, which must move with the mesh. Give those round faces "
                             "a bc of their own with the same rotation, or enlarge the zone")
            if float(Aw[inside].sum() / tot) > 0.5:
                moving.append(p["name"])
            elif isinstance(r, str):
                raise SlidingError(f"wall {p['name']!r} turns with sliding zone {zn!r} but lies mostly outside it")
        else:
            bad = inside & ~on_rev
            fb = float(Aw[bad].sum() / tot)
            if fb > 0.01:
                raise SlidingError(
                    f"wall {p['name']!r} does not turn with sliding zone {zn!r}, but {fb:.1%} of its area (not a "
                    f"surface of revolution about the axis) bounds cells inside the zone (e.g. at "
                    f"{_places(g['Cf'][pf[bad]], Aw[bad])} mm): it would turn with the zone. The zone (radius, span) "
                    "must hold only the rotor; stationary walls (casing, cutwater, vanes) stay outside it")
            if bad.any():
                notes.append(f"{fb:.2%} of stationary wall {p['name']!r} bounds cells inside sliding zone {zn!r}: "
                             "those faces turn with it")
    nm = names(zn)
    plan["sliding"] = {"zone": zn, **nm, **({"interface_cells": plan.pop("sliding_quality")}
                                          if "sliding_quality" in plan else {}), "interface_faces": int(len(f)), "interface_area_mm2": float(A.sum()),
                       "interface_lean_deg": round(mean_deg, 2), "moving_walls": moving,
                       "spinning_walls": spin,
                       "omega_rad_s": zp.get("omega_rad_s"), "axis": a.tolist(), "center_mm": o.tolist()}
    tr = plan.get("transient") or {}
    if tr.get("average_revolutions") is not None and tr["average_revolutions"] < 3:
        notes.append(f"the averages cover {tr['average_revolutions']} whole revolution(s) "
                     f"({tr['end_time'] - tr['average_from']:.4g} s of {tr['period']:.4g} s each): 3 or more give "
                     "a standard error (raise end_time or lower average_from)")


# ---- the case ------------------------------------------------------------------------------------
def dynamic_mesh_dict(sl: dict) -> str:
    from .case import _dict
    o = np.asarray(sl["center_mm"], float) * SCALE
    ax = sl["axis"]
    rpm = float(sl["omega_rad_s"]) * 60.0 / (2 * np.pi)
    body = (f"mover\n{{\n    type            solidBody;\n\n    libs            (\"libpointMeshMovers.so\");\n\n"
            f"    cellZone        {sl['zone']};\n\n    solidBodyMotionFunction\n    {{\n        type            rotatingMotion;\n\n"
            f"        origin          ({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});\n"
            f"        axis            ({ax[0]:.12g} {ax[1]:.12g} {ax[2]:.12g});\n\n"
            f"        omega           {rpm:.12g} [rpm];\n    }}\n}}")
    return _dict("dynamicMeshDict", "constant", body)


SOLVED = ("U", "p")
SOLVED_TURB = ("U", "p", "k", "omega", "nut")


def _turbulent(plan: dict) -> bool:
    return (plan.get("transient") or {}).get("model", plan.get("model", "laminar")) != "laminar"


def create_baffles_dict(sl: dict, turb: bool) -> str:
    from .case import _dict
    pf = ("        U\n        {\n            type            movingWallSlipVelocity;\n            value           uniform (0 0 0);\n        }\n"
          "        p\n        {\n            type            zeroGradient;\n        }\n")
    if turb:
        pf += ("        k\n        {\n            type            zeroGradient;\n        }\n"
               "        omega\n        {\n            type            zeroGradient;\n        }\n"
               "        nut\n        {\n            type            zeroGradient;\n        }\n")
    pf = "\n".join("        " + ln if ln else ln for ln in pf.splitlines())

    def side(key, name):
        return (f"            {key}\n            {{\n                name            {name};\n                type            patch;\n\n"
                f"                patchFields\n                {{\n{pf}\n                }}\n            }}\n")
    body = (f"internalFacesOnly true;\n\nfields          true;\n\nbaffles\n{{\n    {sl['group']}\n    {{\n"
            f"        type            faceZone;\n        zoneName        {sl['face_zone']};\n\n        patches\n        {{\n"
            + side("owner", sl["rotor"]) + "\n" + side("neighbour", sl["stator"]) + "        }\n    }\n}")
    return _dict("createBafflesDict", "system", body)


def write_face_zone(poly: Path, name: str, faces: np.ndarray, flip: np.ndarray) -> None:
    from .polymesh import header
    lines = [header("faceZoneList", "faceZones", "constant/polyMesh"), "1", "(", name, "{", "    type faceZone;",
             f"    faceLabels List<label> {len(faces)}", "("]
    lines += [str(int(i)) for i in faces]
    lines += [")", ";", f"    flipMap List<bool> {len(flip)}", "("]
    lines += ["1" if b else "0" for b in flip]
    lines += [")", ";", "}", ")", "", "// " + "*" * 73 + " //", ""]
    (Path(poly) / "faceZones").write_text("\n".join(lines))


def set_patch_type(text: str, patch: str, new_type: str) -> tuple[str, bool]:
    """A field file with patch's boundary condition type replaced (its value kept)."""
    m = re.search(r"(?m)^\s*" + re.escape(patch) + r"\s*\n\s*\{", text)
    if not m:
        return text, False
    t = re.compile(r"type\s+[^;]+;").search(text, m.end())
    if not t:
        return text, False
    return text[:t.start()] + f"type            {new_type};" + text[t.end():], True


def _boundary(poly: Path) -> list[dict]:
    from .foamio import read_boundary
    return read_boundary((Path(poly) / "boundary").read_text(errors="replace"))


DECOMPOSED = "cfdt_decomposed.json"


def to_sliding(case: Path, plan: dict, mesh, run_step, procs: int = 1, mpirun: str | None = None) -> list[str]:
    """Turns the case into a sliding-mesh case (see the module's notes). run_step(command, log) runs an OpenFOAM
    utility in the case (raising on failure). With procs > 1 the conformal case is decomposed first and the three
    utilities run in parallel (as OpenFOAM's propeller tutorial does): decomposePar on a mesh that already has the
    couple loops over every pair of faces of the two sides (domainDecompositionDecompose.C, processInterCyclics),
    which for s180 (1.24 M faces a side) never finished. The case then stays decomposed; foam.read_fields reads the
    processor directories (cfdt_decomposed.json). Returns notes."""
    import json
    case = Path(case)
    sl = plan["sliding"]
    zero = case / "0"
    # the solution of the steady start, without what does not survive the topology change
    for d in sorted(case.glob("processor*")):
        shutil.rmtree(d, ignore_errors=True)                # decomposed again for the transient run
    (case / DECOMPOSED).unlink(missing_ok=True)
    # createBaffles (fields true) reads every field in 0/ and needs a boundary condition for each on the new
    # patches: only the solved fields stay; the rest (phi, wallShearStress, yPlus, the monitors' total(p): s180 on
    # spark2, 2026.11.02, "keyword total(p) is undefined") go to 0.steady/, and the transient run writes its own
    keep = set(SOLVED_TURB if _turbulent(plan) else SOLVED)
    aside = case / "0.steady"
    for f in sorted(zero.iterdir()) if zero.is_dir() else []:
        if f.is_file() and f.name not in keep:
            aside.mkdir(exist_ok=True)
            shutil.move(str(f), str(aside / f.name))
    # the rotor's walls move with the mesh
    uf = zero / "U"
    txt = uf.read_text()
    for w in sl["moving_walls"]:
        txt, ok = set_patch_type(txt, w, "movingWallVelocity")
        if not ok:
            raise SlidingError(f"0/U has no entry for wall {w!r}")
    for w in sl.get("spinning_walls", []):
        o = np.asarray(sl["center_mm"], float) * SCALE
        ax = sl["axis"]
        txt, ok = set_patch_type(txt, w, "rotatingWallVelocity;\n        origin          "
                                 f"({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});\n        axis            "
                                 f"({ax[0]:.12g} {ax[1]:.12g} {ax[2]:.12g});\n        omega           "
                                 f"{float(sl['omega_rad_s']):.12g}")
        if not ok:
            raise SlidingError(f"0/U has no entry for wall {w!r}")
    uf.write_text(txt)
    # MRF off, the mover on
    (case / "constant" / "MRFProperties").unlink(missing_ok=True)
    (case / "constant" / "dynamicMeshDict").write_text(dynamic_mesh_dict(sl))
    # the interface: faceZone -> baffles -> split -> non-conformal couple
    poly = case / "constant" / "polyMesh"
    f, flip = interface(mesh, mesh.zones[sl["zone"]])
    write_face_zone(poly, sl["face_zone"], f, flip)
    (case / "system" / "createBafflesDict").write_text(create_baffles_dict(sl, _turbulent(plan)))
    before = {p["name"]: p for p in mesh.patches}
    par = ""
    if procs > 1:
        (case / "system" / "decomposeParDict").write_text(
            "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      decomposeParDict;\n}\n\n"
            f"numberOfSubdomains {procs};\n\ndecomposer      scotch;\n")
        run_step("decomposePar -force", "log.decomposePar")
        # the conformal decomposition's addressing: cells keep it through the topology change, and each original
        # patch keeps its faces in order, so the results can be put together without reconstructPar
        for k in range(procs):
            d = case / f"processor{k}"
            src = d / "constant" / "polyMesh"
            dst = d / "cfdt_addressing"
            dst.mkdir(exist_ok=True)
            for nm in ("cellProcAddressing", "faceProcAddressing", "boundary"):
                if not (src / nm).is_file():
                    raise SlidingError(f"decomposePar wrote no {src / nm}")
                shutil.copyfile(src / nm, dst / nm)
        # launcher: foam.launch(procs) ('mpirun -np N', or CFDT_LAUNCH's 'srun -n N ...'); a bare 'mpirun' (older
        # callers) gets its rank count
        lp = (mpirun or "mpirun").strip()
        par = (lp if re.search(r"(^|\s)(-np|-n)\s+\d+", lp) else f"{lp} -np {procs}") + " "
    run_step(f"{par}createBaffles" + (" -parallel" if par else ""), "log.createBaffles")
    run_step(f"{par}splitBaffles" + (" -parallel" if par else ""), "log.splitBaffles")
    # the side with the smaller cells second (the neighbour): OpenFOAM's stitcher puts the volume error of the
    # couple on the owner side ('Volumetric distortion can be minimised by making the side of the interface with
    # smaller or more finely layered cells the neighbour', fvMeshStitcher.C; S50, 2026.11.04)
    # cfdt 2026.11.12.11: by the estimated projected volume fraction on each side (projected_fraction), which
    # follows the cells' thickness at the faces, not their mean size (Perlmutter sliding7: 0.54-0.64 at every step
    # with the rotor owning the couple); the mean cell size when there is no estimate
    pv = (sl.get("interface_cells") or {}).get("projected_volume_fraction") or {}
    mc = (sl.get("interface_cells") or {}).get("mean_cell_mm") or {}
    if pv.get("zone_owner") and pv.get("other_owner"):
        stator_owns = pv["other_owner"]["max"] < pv["zone_owner"]["max"]
        sl["ncc_owner_by"] = "estimated projected volume fraction"
    else:
        stator_owns = bool(mc) and mc.get("zone_side", 1) < mc.get("other_side", 1)
        sl["ncc_owner_by"] = "mean cell size"
    pair = (sl["stator"], sl["rotor"]) if stator_owns else (sl["rotor"], sl["stator"])
    sl["ncc_owner"] = pair[0]
    run_step(f"{par}createNonConformalCouples" + (" -parallel" if par else "") +
             f" -fields {pair[0]} {pair[1]}", "log.createNonConformalCouples")
    # the patches the results read must be the same faces in the same order (createBaffles keeps their order)
    if par:
        after = {}
        for k in range(procs):
            for p in _boundary(case / f"processor{k}" / "constant" / "polyMesh"):
                after.setdefault(p["name"], {"name": p["name"], "n": 0, "type": p.get("type")})["n"] += p["n"]
    else:
        after = {p["name"]: p for p in _boundary(poly)}
    for nm, p in before.items():
        q = after.get(nm)
        if q is None or q["n"] != p["n"]:
            raise SlidingError(f"after createBaffles patch {nm!r} has {q['n'] if q else 'no'} faces instead of {p['n']}")
    for nm in (sl["rotor"], sl["stator"]):
        if after.get(nm, {}).get("n", 0) != len(f):
            raise SlidingError(f"createBaffles made {after.get(nm, {}).get('n', 0)} faces on {nm!r}, expected {len(f)}")
    sl["patches_added"] = sorted(n for n in set(after) - set(before) if not n.startswith(("procBoundary", "nonConformalProcessorCyclic")))
    sl["ncc_patches"] = {n: after[n].get("type") for n in sl["patches_added"]}
    if par:
        (case / DECOMPOSED).write_text(json.dumps({"procs": procs, "why": "sliding mesh: the couple is built on the "
                                                  "decomposed case; results are read from processor*"}))
        sl["decomposed"] = procs
    return [f"sliding mesh: zone {sl['zone']!r} turns at {sl['omega_rad_s'] * 60 / (2 * np.pi):g} rpm; {len(f)} faces "
            f"on its cylinder coupled non-conformally ({sl['rotor']} | {sl['stator']})" +
            (f", built on the case decomposed for {procs} ranks (it stays decomposed)" if par else "")]


# ---- after the run: the flux through the interface ----------------------------------------------------------------
BALANCE_WARN = 1e-3        # of the through-flow: noted
BALANCE_UNRELIABLE = 1e-2  # the run is unreliable (the v2506 S50 MRF run lost 11.4 m3/h at its AMI interface)


def interface_balance(case: Path, sl: dict, q_through: float | None) -> dict | None:
    """What crosses the sliding interface in the last written fields (phi on the couple's patches, every processor):
    the rotor side's and the stator side's sums (equal and opposite when the coupling conserves the flow), their
    imbalance, the flux into the couple's error patches, and the flow crossing one way. Patches are told apart by
    name: OpenFOAM's 'nonConformalCyclic_on_<rotor>' and the processor and error patches derived from it."""
    from .foamio import read_foam
    case = Path(case)
    if not case.is_dir():
        return None
    dirs = sorted(case.glob("processor*")) or [case]

    def times(d):
        out = []
        for x in d.iterdir():
            try:
                t = float(x.name)
            except ValueError:
                continue
            if t > 0 and (x / "phi").is_file():
                out.append((t, x.name))
        return sorted(out)
    tl = times(dirs[0])
    if not tl:
        return None
    t, tname = tl[-1]
    on_r, on_s = f"on_{sl['rotor']}", f"on_{sl['stator']}"
    rot = sto = err = cross = 0.0
    found = 0
    for d in dirs:
        f = d / tname / "phi"
        if not f.is_file():
            continue
        bf = (read_foam(f.read_text(errors="replace")).get("boundaryField") or {})
        for name, e in bf.items():
            typ = str((e or {}).get("type", ""))
            if "nonConformal" not in typ and "nonConformal" not in name:
                continue
            v = (e or {}).get("value")
            if v is None or isinstance(v, str):
                continue
            a = np.atleast_1d(np.asarray(v, float))
            found += a.size
            if "rror" in typ or "rror" in name:
                err += float(a.sum())
            elif name.endswith(on_r) or on_r + "_" in name or name.endswith(on_r + "]"):
                rot += float(a.sum())
                cross += float(a[a > 0].sum())
            elif name.endswith(on_s) or on_s + "_" in name:
                sto += float(a.sum())
    if not found:
        return None
    q = abs(q_through) if q_through else None
    imb = rot + sto
    out = {"time": t, "rotor_side_m3s": float(f"{rot:.6g}"), "stator_side_m3s": float(f"{sto:.6g}"),
           "imbalance_m3s": float(f"{imb:.4g}"), "error_patches_m3s": float(f"{err:.4g}"),
           "crossing_one_way_m3s": float(f"{cross:.6g}"), "faces": found}
    if q:
        out["through_flow_m3s"] = float(f"{q:.6g}")
        out["error"] = float(f"{max(abs(imb), abs(err)) / q:.4g}")
    return out

