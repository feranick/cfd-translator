"""onset.py — where a steady (MRF) single-phase run says vapour or air would appear, to screen impellers before
two-phase sweeps.
Version: 2026.11.13.2

    settings(study) -> dict | None                   c, thresholds (Pa abs, named), regions (as given)
    regions(study, mesh, plan, res) -> {name: mask, ...}, {name: how}   derived and given regions
    evaluate(study, mesh, plan, fields, res) -> (block, p_eff, warnings)

Per cell, the effective pressure p_eff = p_abs - c rho k (c = 0.3 by default: the turbulent pressure fluctuation is
about 0.2-0.4 rho k), p_abs = conditions.p + the gauge pressure. Per threshold (the vapour pressure, an air-release
saturation pressure, or Pa; onset.npsh sets the absolute level so the inlet's total pressure is p_vapour + rho g
NPSH, else conditions.p) and per region (the domain, boxes and cylinders as in p_below_regions, and regions cfdt
derives from the rotating zone and its walls: eye, tip_clearance, vane_leading_edges): the volume below, the largest
depth below, where, and the region's smallest p_eff (its margin to each threshold). Transient runs with p_below give
the time fraction below the same thresholds: compare the two to calibrate c.

Derived regions (per rotating zone; '<name>_<zone>' with several zones):
  vane_leading_edges  cells within distance_mm (default 8 % of the blades' outer radius) of the blades' leading
                      edges: blade faces are the rotating walls' faces that face round the axis (|n . e_theta| > 0.3);
                      the leading edges are those within the first 15 % of the blades' extent from the inlet side
                      (radius for radial and mixed-flow impellers; the axial coordinate for axial ones)
  eye                 radial impellers: cells inside the blades' leading-edge radius, over the blades' axial extent
                      and half that radius upstream (towards the inlet)
  tip_clearance       cells in gaps between rotating and stationary walls: distance to a rotating wall plus distance
                      to a stationary wall at most gap_mm (default 3 % of the blades' outer radius)
"""
from __future__ import annotations

import numpy as np

DERIVED = ("eye", "tip_clearance", "vane_leading_edges")
C_DEFAULT = 0.3
# cells left out of the robust statistics (S50 impeller variants on spark1, 2026.11.13: one cell at -822 kPa absolute
# in the disc-rim gap at the rotating zone's edge, skewness up to 24.7 nearby, set 'domain' and 'tip_clearance')
EXCLUDE = {"skewness": 4.0, "zone_boundary_cells": 2}
QUANTILES = (0.001, 0.01)        # volume fractions: the p_eff below which this much of a region's volume lies


def settings(study: dict) -> dict | None:
    on = study.get("onset")
    if not on:
        return None
    on = {} if on is True else dict(on)
    ex = on.get("exclude", True)
    ex = {} if ex is True else (None if ex is False else dict(ex))
    if ex is not None:
        ex = {"skewness": float(ex.get("skewness", EXCLUDE["skewness"])),
              "zone_boundary_cells": int(ex.get("zone_boundary_cells", EXCLUDE["zone_boundary_cells"]))}
    return {"c": float(on.get("c", C_DEFAULT)), "below": list(on.get("below") or ["vapour"]),
            "regions": list(on.get("regions") or []), "p_vapour": on.get("p_vapour"), "npsh": on.get("npsh"),
            "exclude": ex}


def errors(study: dict) -> list[str]:
    """Validation of study.onset."""
    on = study.get("onset")
    if on is None or on is True:
        return []
    if not isinstance(on, dict):
        return ["onset: true or {c, below: ['vapour' | {air_release: Pa | 'inlet', ratios} | Pa], regions, npsh, "
                "p_vapour, exclude}"]
    errs = []
    bad = set(on) - {"c", "below", "regions", "p_vapour", "npsh", "exclude"}
    if bad:
        errs.append(f"onset: unknown keys {sorted(bad)} (c, below, regions, npsh, p_vapour, exclude)")
    ex = on.get("exclude", True)
    if not (isinstance(ex, bool) or (isinstance(ex, dict) and not set(ex) - set(EXCLUDE) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in ex.values()))):
        errs.append("onset.exclude: true (default: skewness > 4 and cells within 2 of a rotating zone's boundary left "
                    "out of the robust statistics), false, or {skewness, zone_boundary_cells}")
    if "npsh" in on and (not isinstance(on["npsh"], (int, float)) or isinstance(on["npsh"], bool) or on["npsh"] <= 0):
        errs.append("onset.npsh: m (> 0): the absolute level is set so the inlet's total pressure is p_vapour + rho g NPSH")
    c = on.get("c", C_DEFAULT)
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0 <= c <= 2:
        errs.append("onset.c: 0-2 (p_eff = p_abs - c rho k; 0.3 by default)")
    if "p_vapour" in on and (not isinstance(on["p_vapour"], (int, float)) or on["p_vapour"] <= 0):
        errs.append("onset.p_vapour: Pa (absolute), for liquids other than water")
    below = on.get("below", ["vapour"])
    if not isinstance(below, list) or not 1 <= len(below) <= 8:
        errs.append("onset.below: 1-8 thresholds: 'vapour', {air_release: Pa (the saturation pressure)}, or Pa absolute")
    else:
        for i, b in enumerate(below):
            ar = b.get("air_release") if isinstance(b, dict) else None
            rt = b.get("ratios", []) if isinstance(b, dict) else []
            ok = b == "vapour" or (isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0) or (
                isinstance(b, dict) and set(b) <= {"air_release", "ratios"} and
                (ar == "inlet" or (isinstance(ar, (int, float)) and not isinstance(ar, bool) and ar > 0)) and
                isinstance(rt, list) and len(rt) <= 6 and all(isinstance(r, (int, float)) and 0 < r <= 1 for r in rt))
            if not ok:
                errs.append(f"onset.below[{i}]: 'vapour', {{air_release: Pa (saturation) | 'inlet' (saturated at the "
                            "inlet's pressure), ratios: [0.5, 0.25] (also below these fractions of it)}} or a pressure "
                            "in Pa (absolute)")
    fl = study.get("fluid")
    if "vapour" in (below if isinstance(below, list) else []) and "p_vapour" not in on and fl != "water" and \
            not (isinstance(fl, dict) and fl.get("name") == "water"):
        errs.append("onset.below 'vapour': the fluid is not water: give onset.p_vapour (Pa)")
    regs = on.get("regions", [])
    if not isinstance(regs, list):
        errs.append("onset.regions: ['eye' | 'tip_clearance' | 'vane_leading_edges' | {name, box} | {name, cylinder}]")
        return errs
    from .format import _num, _vec
    for i, rg in enumerate(regs):
        if isinstance(rg, str):
            if rg not in DERIVED:
                errs.append(f"onset.regions[{i}]: {rg!r} is none of {', '.join(DERIVED)}")
            continue
        if not isinstance(rg, dict) or not isinstance(rg.get("name"), str) or rg["name"] == "domain":
            errs.append(f"onset.regions[{i}]: a name (not 'domain') with box or cylinder, or one of {', '.join(DERIVED)}")
            continue
        if rg["name"] in DERIVED and not ({"box", "cylinder"} & set(rg)):
            extra = set(rg) - {"name", "gap_mm", "distance_mm"}
            if extra or any(k in rg and not (_num(rg[k]) and rg[k] > 0) for k in ("gap_mm", "distance_mm")):
                errs.append(f"onset.regions[{i}]: {rg['name']} takes gap_mm (tip_clearance) or distance_mm "
                            "(vane_leading_edges), mm > 0")
            continue
        ok = (_vec(rg.get("box"), 6) and all(rg["box"][j] < rg["box"][j + 3] for j in range(3))) or (
            isinstance(rg.get("cylinder"), dict) and _vec(rg["cylinder"].get("axis"))
            and _num(rg["cylinder"].get("radius")) and rg["cylinder"]["radius"] > 0)
        if not ok:
            errs.append(f"onset.regions[{i}]: {{name, box: [x0,y0,z0,x1,y1,z1]}} or {{name, cylinder: {{center, axis, "
                        "radius, span, inner_radius}}}} (mm)")
    if study.get("cavitation"):
        errs.append("onset: for single-phase runs (screening before a cavitation sweep), not with cavitation")
    return errs


def thresholds(study: dict, on: dict, p_inlet: float | None = None) -> tuple[dict, dict]:
    """({name: Pa absolute}, {name: what}): 'vapour' (water at conditions.T, or onset.p_vapour); 'air_release' (or
    'air_release_<Pa>' with several) at the saturation pressure given, or 'inlet': the inlet's static pressure (water
    saturated where it enters: release only where the pressure falls below that); with ratios, also
    'air_release_x<r>' at r times it (a supersaturation p_sat / p of 1 / r); '<Pa>'."""
    out, what = {}, {}
    T = float((study.get("conditions") or {}).get("T", 20.0))
    rel = [b for b in on["below"] if isinstance(b, dict)]
    for b in on["below"]:
        if b == "vapour":
            if on.get("p_vapour"):
                out["vapour"] = float(on["p_vapour"])
            else:
                from .cavitation import water_vapour
                out["vapour"] = float(water_vapour(T)["p"])
            what["vapour"] = "the vapour pressure"
        elif isinstance(b, dict):
            if b["air_release"] == "inlet":
                if p_inlet is None:
                    continue
                v, src = float(p_inlet), "the inlet's static pressure (saturated there)"
            else:
                v, src = float(b["air_release"]), f"saturation at {float(b['air_release']):g} Pa"
            nm = "air_release" if len(rel) == 1 else f"air_release_{'inlet' if b['air_release'] == 'inlet' else f'{v:g}'}"
            out[nm], what[nm] = v, f"air release: {src}"
            for r in b.get("ratios") or []:
                out[f"{nm}_x{r:g}"] = v * float(r)
                what[f"{nm}_x{r:g}"] = f"{r:g} x {nm} (supersaturation {1 / float(r):.3g})"
        else:
            out[f"{float(b):g}"] = float(b)
            what[f"{float(b):g}"] = "given (Pa absolute)"
    return out, what


def flagged_cells(mesh, study: dict, ex: dict | None) -> tuple[np.ndarray, dict]:
    """Cells left out of the robust statistics: skewness above ex['skewness'] (checkMesh's measure, the largest of a
    cell's faces) or within ex['zone_boundary_cells'] cells of a rotating zone's boundary (the MRF frame jumps there).
    Returns (mask, {skewness: cells, zone_boundary: cells})."""
    n = mesh.ncells
    if not ex:
        return np.zeros(n, bool), {}
    from .polymesh import face_skewness
    g = mesh.geometry()
    ni = mesh.ninternal
    sk = face_skewness(g["Cf"], g["Sf"], g["C"], mesh.owner, mesh.neighbour, ni)
    cs = np.zeros(n)
    np.maximum.at(cs, mesh.owner, sk)
    np.maximum.at(cs, mesh.neighbour, sk[:ni])
    skew = cs > ex["skewness"]
    near = np.zeros(n, bool)
    for z in study.get("rotating") or []:
        if not isinstance(z, dict):
            continue
        cells = (mesh.zones or {}).get(z.get("name"))
        if cells is None:
            from .transient import _region_mask
            cyl = {"cylinder": {"center": z.get("center", [0, 0, 0]), "axis": z.get("axis", [0, 0, 1]),
                                "radius": z.get("radius", 0), **({"span": z["span"]} if z.get("span") else {})}}
            inz = _region_mask(cyl, g["C"]) if z.get("radius") else np.zeros(n, bool)
        else:
            inz = np.zeros(n, bool)
            inz[np.asarray(cells, np.int64)] = True
        own, nei = mesh.owner[:ni], mesh.neighbour
        f = inz[own] != inz[nei]
        layer = np.zeros(n, bool)
        layer[own[f]] = True
        layer[nei[f]] = True
        for _ in range(max(0, ex["zone_boundary_cells"] - 1)):
            grow = layer[own] | layer[nei]
            nxt = layer.copy()
            nxt[own[grow]] = True
            nxt[nei[grow]] = True
            layer = nxt
        near |= layer
    return skew | near, {"skewness": int(skew.sum()), "zone_boundary": int(near.sum()),
                         "skewness_above": ex["skewness"], "zone_boundary_cells": ex["zone_boundary_cells"]}


def vquantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    """The value below which a fraction q of the weight (volume) lies."""
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    k = int(np.searchsorted(cw, q * cw[-1]))
    return float(v[o][min(k, len(o) - 1)])


def _walls(study: dict, mesh, res: dict, zone: str) -> tuple[list, list]:
    """(rotating wall patch names, stationary wall patch names) for a zone, from the results' rotating block."""
    rot = set(((res.get("rotating") or {}).get(zone) or {}).get("rotating_walls") or [])
    for b in study.get("bcs") or []:
        if b.get("kind") == "wall" and b.get("rotation") == zone and b.get("name"):
            rot.add(b["name"])
    walls = [p["name"] for p in mesh.patches if p["type"] == "wall"]
    return [w for w in walls if w in rot], [w for w in walls if w not in rot]


def _faces(mesh, names) -> np.ndarray:
    out = [np.arange(p["start"], p["start"] + p["n"]) for p in mesh.patches if p["name"] in names]
    return np.concatenate(out) if out else np.zeros(0, int)


def derived_regions(study: dict, mesh, res: dict, want: dict) -> tuple[dict, dict]:
    """want: {derived name: options}. Returns ({region name: cell mask}, {region name: how / why not})."""
    from scipy.spatial import cKDTree
    masks, how = {}, {}
    zones = [z for z in (study.get("rotating") or []) if isinstance(z, dict)]
    if not zones:
        for nm in want:
            how[nm] = "no rotating zone: nothing to derive it from"
        return masks, how
    g = mesh.geometry()
    C, Cf, Sf = g["C"], g["Cf"], g["Sf"]
    inlets = [b.get("name") for b in study.get("bcs") or [] if b.get("kind") == "inlet"]
    fin = _faces(mesh, [n for n in inlets if n])
    for z in zones:
        suf = f"_{z['name']}" if len(zones) > 1 else ""
        a = np.asarray(z.get("axis", [0, 0, 1]), float)
        a = a / np.linalg.norm(a)
        o = np.asarray(z.get("center", [0, 0, 0]), float)
        rot, stat = _walls(study, mesh, res, z["name"])
        fr = _faces(mesh, rot)
        if not len(fr):
            for nm in want:
                how[nm + suf] = f"zone {z['name']!r} has no rotating walls"
            continue
        d = Cf[fr] - o
        s = d @ a
        rv = d - np.outer(s, a)
        r = np.linalg.norm(rv, axis=1)
        et = np.cross(a, rv) / np.maximum(r, 1e-12)[:, None]
        n = Sf[fr] / np.maximum(np.linalg.norm(Sf[fr], axis=1), 1e-300)[:, None]
        blade = np.abs(np.einsum("ij,ij->i", n, et)) > 0.3
        if blade.sum() < 3:
            for nm in want:
                how[nm + suf] = f"zone {z['name']!r}: its rotating walls have no blade faces (facing round the axis)"
            continue
        rb, sb = r[blade], s[blade]
        r_le, r_out = float(np.percentile(rb, 2)), float(np.percentile(rb, 98))
        radial = (r_out - r_le) > 0.2 * r_out
        # the inlet side along the axis: where the inlet patch lies relative to the blades
        up = float(np.sign(((Cf[fin] - o) @ a).mean() - sb.mean())) if len(fin) else -1.0
        up = up or -1.0
        Cd = C - o
        Cs = Cd @ a
        Cr = np.linalg.norm(Cd - np.outer(Cs, a), axis=1)
        if "vane_leading_edges" in want:
            if radial:
                le = blade.copy()
                le[blade] = rb <= r_le + 0.15 * (r_out - r_le)
                crit = f"radius {r_le:.3g}-{r_le + 0.15 * (r_out - r_le):.3g} mm"
            else:
                m_ = sb * up                                      # larger: nearer the inlet
                top = float(np.percentile(m_, 98))
                ext = top - float(np.percentile(m_, 2))
                le = blade.copy()
                le[blade] = m_ >= top - 0.15 * ext
                crit = f"the first {0.15 * ext:.3g} mm of the blades from the inlet side"
            dist = float(want["vane_leading_edges"].get("distance_mm") or 0.08 * r_out)
            tree = cKDTree(Cf[fr][le])
            dd, _ = tree.query(C, distance_upper_bound=dist)
            masks["vane_leading_edges" + suf] = np.isfinite(dd)
            how["vane_leading_edges" + suf] = (f"cells within {dist:.3g} mm of {int(le.sum())} leading-edge blade faces "
                                              f"({crit}) of zone {z['name']!r}")
        if "eye" in want:
            if radial:
                lo, hi = float(sb.min()), float(sb.max())
                if up > 0:
                    hi += 0.5 * r_le
                else:
                    lo -= 0.5 * r_le
                masks["eye" + suf] = (Cr <= r_le) & (Cs >= lo) & (Cs <= hi)
                how["eye" + suf] = (f"cells within the leading-edge radius {r_le:.3g} mm, axial {lo:.3g}..{hi:.3g} mm "
                                    f"from the zone's centre (zone {z['name']!r})")
            else:
                how["eye" + suf] = f"zone {z['name']!r}: an axial impeller (no eye inside the blades)"
        if "tip_clearance" in want:
            fs = _faces(mesh, stat)
            gap = float(want["tip_clearance"].get("gap_mm") or 0.03 * r_out)
            if not len(fs):
                how["tip_clearance" + suf] = "no stationary walls"
            else:
                near = Cr <= r_out + 2 * gap
                d1, _ = cKDTree(Cf[fr]).query(C[near], distance_upper_bound=gap)
                d2, _ = cKDTree(Cf[fs]).query(C[near], distance_upper_bound=gap)
                m = np.zeros(len(C), bool)
                m[np.nonzero(near)[0]] = np.isfinite(d1) & np.isfinite(d2) & (d1 + d2 <= gap)
                masks["tip_clearance" + suf] = m
                how["tip_clearance" + suf] = (f"cells between rotating and stationary walls, at most {gap:.3g} mm apart "
                                              f"(zone {z['name']!r})")
    return masks, how


def evaluate(study: dict, mesh, plan: dict, fields: dict, res: dict) -> tuple[dict | None, np.ndarray | None, list]:
    """The onset block, the p_eff field (Pa absolute) and warnings."""
    on = settings(study)
    if not on:
        return None, None, []
    warns = []
    g = mesh.geometry()
    V, C = g["V"], g["C"]
    en = plan.get("energy")
    p = np.asarray(fields["p"], float)
    if en:
        p_abs = p
        rho = np.asarray(fields.get("rho") if fields.get("rho") is not None else np.full(len(p), en.get("rho", 1.0)),
                         float)
    else:
        rho = np.full(len(p), float(plan["rho"]))
        p_abs = float((study.get("conditions") or {}).get("p", 101325.0)) + rho * p
    thr0, _ = thresholds(study, on)
    level = {"from": "conditions.p + the gauge pressure"}
    zero = float((study.get("conditions") or {}).get("p", 101325.0))
    inl = next((b.get("name") for b in study.get("bcs") or [] if b.get("kind") == "inlet" and b.get("name")), None)
    if on.get("npsh") and not en:
        # the absolute level from NPSH: the inlet's flow-weighted total pressure = p_vapour + rho g NPSH
        pt = ((res.get("pressure") or {}).get(inl) or {}).get("p_total") if inl else None
        pv = thr0.get("vapour")
        if pv is None:
            from .cavitation import water_vapour
            pv = float(on.get("p_vapour") or water_vapour(float((study.get("conditions") or {}).get("T", 20.0)))["p"])
        if pt is None:
            warns.append("onset.npsh: no inlet total pressure in the results: conditions.p sets the level instead")
        else:
            from .cavitation import G0
            shift = pv + float(plan["rho"]) * G0 * float(on["npsh"]) - float(pt)
            p_abs = shift + rho * p
            zero = shift
            level = {"from": f"NPSH {on['npsh']:g} m at inlet {inl!r}: its total pressure {pv + float(plan['rho']) * G0 * float(on['npsh']):.6g} Pa",
                     "gauge_zero_Pa": float(f"{shift:.6g}")}
    pin = ((res.get("pressure") or {}).get(inl) or {}).get("p_mean") if inl else None
    p_inlet = (float(pin) if en else zero + float(pin)) if pin is not None else None
    if p_inlet is not None:
        level["inlet_static_Pa"] = float(f"{p_inlet:.6g}")
    thr, what = thresholds(study, on, p_inlet)
    if any(isinstance(b, dict) and b.get("air_release") == "inlet" for b in on["below"]) and p_inlet is None:
        warns.append("onset: air_release 'inlet' needs the inlet's pressure in the results: left out")
    k = (fields.get("scalars") or {}).get("k")
    kk = np.asarray(k["internal"], float) if k else np.zeros(len(p))
    if not k and study.get("model", "laminar") != "laminar":
        warns.append("onset: no k field was read: p_eff is the absolute pressure alone")
    p_eff = p_abs - on["c"] * rho * kk
    want = {}
    given = []
    for rg in on["regions"]:
        if isinstance(rg, str):
            want[rg] = {}
        elif rg.get("name") in DERIVED and not ({"box", "cylinder"} & set(rg)):
            want[rg["name"]] = rg
        else:
            given.append(rg)
    masks, how = derived_regions(study, mesh, res, want) if want else ({}, {})
    from .transient import _region_mask
    masks = {"domain": np.ones(len(V), bool), **masks}
    for rg in given:
        masks[rg["name"]] = _region_mask(rg, C)
        how[rg["name"]] = "as given"
    flag, fl_count = flagged_cells(mesh, study, on["exclude"])
    Va = np.abs(V)
    regions, below = {}, {t: {} for t in thr}
    for nm, m in masks.items():
        if not m.any():
            regions[nm] = {"cells": 0, "how": how.get(nm, "the whole fluid")}
            continue
        idx = np.nonzero(m)[0]
        j = idx[int(np.argmin(p_eff[idx]))]
        clean = m & ~flag
        rec = {"cells": int(m.sum()), "volume_mm3": float(f"{Va[m].sum():.5g}"),
               "p_eff_min_Pa": float(f"{p_eff[j]:.6g}"), "at_mm": np.round(C[j], 2).tolist(),
               "p_abs_min_Pa": float(f"{p_abs[idx].min():.6g}"), **({"how": how[nm]} if nm in how else {})}
        if flag[j]:
            rec["min_cell_flagged"] = True
        ic = np.nonzero(clean)[0]
        if len(ic):
            jc = ic[int(np.argmin(p_eff[ic]))]
            rec["robust"] = {"p_eff_min_Pa": float(f"{p_eff[jc]:.6g}"), "at_mm": np.round(C[jc], 2).tolist(),
                             **{f"p_eff_q{q * 100:g}pct_Pa": float(f"{vquantile(p_eff[ic], Va[ic], q):.6g}")
                                for q in QUANTILES},
                             "cells_left_out": int((m & flag).sum()),
                             "volume_left_out_mm3": float(f"{Va[m & flag].sum():.4g}")}
        regions[nm] = rec
        for t, v in thr.items():
            bm = m & (p_eff < v)
            r = {"margin_Pa": float(f"{p_eff[j] - v:.5g}"), "volume_mm3": float(f"{Va[bm].sum():.5g}") if bm.any() else 0.0}
            if bm.any():
                r.update(cells=int(bm.sum()), depth_max_Pa=float(f"{v - p_eff[j]:.5g}"),
                         deepest_at_mm=np.round(C[j], 2).tolist(),
                         region_fraction=float(f"{Va[bm].sum() / Va[m].sum():.4g}"))
            if len(ic):
                bc = clean & (p_eff < v)
                q1 = rec["robust"]["p_eff_q1pct_Pa"]
                r["robust"] = {"volume_mm3": float(f"{Va[bc].sum():.5g}"),
                               "region_fraction": float(f"{Va[bc].sum() / max(Va[clean].sum(), 1e-300):.4g}"),
                               "depth_max_Pa": float(f"{max(v - rec['robust']['p_eff_min_Pa'], 0.0):.5g}"),
                               "margin_q1pct_Pa": float(f"{q1 - v:.5g}"),
                               **({"supersaturation_q1pct": float(f"{v / q1:.4g}")} if t.startswith("air_release")
                                  and q1 > 0 else {})}
            below[t][nm] = r
    for nm, h in how.items():
        if nm not in regions:
            regions[nm] = {"cells": 0, "how": h}
    out = {"c": on["c"], "thresholds_Pa": {t: float(f"{v:.6g}") for t, v in thr.items()}, "thresholds": what,
           "p_level": level,
           "left_out": {**fl_count, "note": "robust: statistics without these cells (onset.exclude); the plain "
                                            "values keep every cell"} if on["exclude"] else {"note": "every cell kept"},
           "regions": regions,
           "below": below,
           "p_eff": "p_abs - c rho k per cell (Pa absolute); p_abs = conditions.p + the gauge pressure",
           "note": "a steady screen: compare with p_below of a transient run (pressure_statistics) to calibrate c"}
    hit = [f"{t} in {', '.join(n for n, r in below[t].items() if (r.get('robust') or r).get('volume_mm3'))}" for t in thr
           if any((r.get("robust") or r).get("volume_mm3") for r in below[t].values())]
    lone = [n for n, r in regions.items() if r.get("min_cell_flagged")]
    if lone:
        warns.append(f"onset: the lowest p_eff of {', '.join(lone)} lies in a flagged cell (skewness above "
                     f"{on['exclude']['skewness']:g} or at a rotating zone's boundary): see each region's robust values")
    if hit:
        warns.append("onset: p_eff falls below " + "; ".join(hit) + " (results.onset)")
    return out, p_eff, warns
