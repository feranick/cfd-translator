"""onset.py — where a steady (MRF) single-phase run says vapour or air would appear, to screen impellers before
two-phase sweeps.
Version: 2026.11.12.17

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


def settings(study: dict) -> dict | None:
    on = study.get("onset")
    if not on:
        return None
    on = {} if on is True else dict(on)
    return {"c": float(on.get("c", C_DEFAULT)), "below": list(on.get("below") or ["vapour"]),
            "regions": list(on.get("regions") or []), "p_vapour": on.get("p_vapour"), "npsh": on.get("npsh")}


def errors(study: dict) -> list[str]:
    """Validation of study.onset."""
    on = study.get("onset")
    if on is None or on is True:
        return []
    if not isinstance(on, dict):
        return ["onset: true or {c, below: ['vapour' | {air_release: Pa} | Pa], regions, npsh, p_vapour}"]
    errs = []
    bad = set(on) - {"c", "below", "regions", "p_vapour", "npsh"}
    if bad:
        errs.append(f"onset: unknown keys {sorted(bad)} (c, below, regions, npsh, p_vapour)")
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
            ok = b == "vapour" or (isinstance(b, (int, float)) and not isinstance(b, bool) and b > 0) or (
                isinstance(b, dict) and set(b) == {"air_release"} and isinstance(b["air_release"], (int, float))
                and b["air_release"] > 0)
            if not ok:
                errs.append(f"onset.below[{i}]: 'vapour', {{air_release: Pa}} or a pressure in Pa (absolute)")
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


def thresholds(study: dict, on: dict) -> dict:
    """{name: Pa absolute}: 'vapour' (water at conditions.T, or onset.p_vapour), 'air_release' (or
    'air_release_<Pa>' with several), '<Pa>'."""
    out = {}
    T = float((study.get("conditions") or {}).get("T", 20.0))
    rel = [b for b in on["below"] if isinstance(b, dict)]
    for b in on["below"]:
        if b == "vapour":
            if on.get("p_vapour"):
                out["vapour"] = float(on["p_vapour"])
            else:
                from .cavitation import water_vapour
                out["vapour"] = float(water_vapour(T)["p"])
        elif isinstance(b, dict):
            v = float(b["air_release"])
            out["air_release" if len(rel) == 1 else f"air_release_{v:g}"] = v
        else:
            out[f"{float(b):g}"] = float(b)
    return out


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
    thr = thresholds(study, on)
    level = {"from": "conditions.p + the gauge pressure"}
    if on.get("npsh") and not en:
        # the absolute level from NPSH: the inlet's flow-weighted total pressure = p_vapour + rho g NPSH
        inl = next((b.get("name") for b in study.get("bcs") or [] if b.get("kind") == "inlet" and b.get("name")), None)
        pt = ((res.get("pressure") or {}).get(inl) or {}).get("p_total") if inl else None
        pv = thr.get("vapour")
        if pv is None:
            from .cavitation import water_vapour
            pv = float(on.get("p_vapour") or water_vapour(float((study.get("conditions") or {}).get("T", 20.0)))["p"])
        if pt is None:
            warns.append("onset.npsh: no inlet total pressure in the results: conditions.p sets the level instead")
        else:
            from .cavitation import G0
            shift = pv + float(plan["rho"]) * G0 * float(on["npsh"]) - float(pt)
            p_abs = shift + rho * p
            level = {"from": f"NPSH {on['npsh']:g} m at inlet {inl!r}: its total pressure {pv + float(plan['rho']) * G0 * float(on['npsh']):.6g} Pa",
                     "gauge_zero_Pa": float(f"{shift:.6g}")}
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
    regions, below = {}, {t: {} for t in thr}
    for nm, m in masks.items():
        if not m.any():
            regions[nm] = {"cells": 0, "how": how.get(nm, "the whole fluid")}
            continue
        idx = np.nonzero(m)[0]
        j = idx[int(np.argmin(p_eff[idx]))]
        regions[nm] = {"cells": int(m.sum()), "volume_mm3": float(f"{V[m].sum():.5g}"),
                       "p_eff_min_Pa": float(f"{p_eff[j]:.6g}"), "at_mm": np.round(C[j], 2).tolist(),
                       "p_abs_min_Pa": float(f"{p_abs[idx].min():.6g}"),
                       **({"how": how[nm]} if nm in how else {})}
        for t, v in thr.items():
            bm = m & (p_eff < v)
            rec = {"margin_Pa": float(f"{p_eff[j] - v:.5g}")}
            if bm.any():
                rec.update(volume_mm3=float(f"{V[bm].sum():.5g}"), cells=int(bm.sum()),
                           depth_max_Pa=float(f"{v - p_eff[j]:.5g}"), deepest_at_mm=np.round(C[j], 2).tolist(),
                           region_fraction=float(f"{V[bm].sum() / V[m].sum():.4g}"))
            else:
                rec.update(volume_mm3=0.0)
            below[t][nm] = rec
    for nm, h in how.items():
        if nm not in regions:
            regions[nm] = {"cells": 0, "how": h}
    out = {"c": on["c"], "thresholds_Pa": {t: float(f"{v:.6g}") for t, v in thr.items()}, "p_level": level,
           "regions": regions,
           "below": below,
           "p_eff": "p_abs - c rho k per cell (Pa absolute); p_abs = conditions.p + the gauge pressure",
           "note": "a steady screen: compare with p_below of a transient run (pressure_statistics) to calibrate c"}
    hit = [f"{t} in {', '.join(n for n, r in below[t].items() if r.get('volume_mm3'))}" for t in thr
           if any(r.get("volume_mm3") for r in below[t].values())]
    if hit:
        warns.append("onset: p_eff falls below " + "; ".join(hit) + " (results.onset)")
    return out, p_eff, warns
