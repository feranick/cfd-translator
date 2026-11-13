"""sections.py — flow through internal surfaces: a plane across a passage (holes, a gap, an opening), within discs or
a box, from the case's face fluxes. Liquid and gas apart in cavitation and air runs.
Version: 2026.11.13.8

    "sections": [{"name": "wear_plate_holes", "plane": {"axis": "z", "at": -40} | {"point": [x,y,z], "normal": [..]},
                  "within": [{"name": "h1", "center": [x, y, z], "radius": r} | {"name": "o", "box": [x0,y0,z0,
                              x1,y1,z1]}, ...] | {"box": [...]}}]

The section is the set of mesh faces between cells on either side of the plane (cell centres), with the face centre
inside the discs (each centre projected onto the plane) or the box. Their flux (phi, m3/s, owner to neighbour) is
summed with the sign along the normal, so the flow is exact by continuity whatever the cells' shape: a staircase of
faces across a passage carries what the passage carries, as long as the region spans the whole passage (give a disc
a little larger than the hole: outside it there are no cells). A ring just outside each region is checked: faces
crossing the plane there mean the region cuts fluid (warning).

results.sections.<name>: faces, area_mm2, flow_m3s / flow_m3h (net along the normal), forward / backward (the two
directions apart: a hole may carry both), mean_velocity_m_s; with VoF: liquid_m3s, liquid_kg_s, gas_m3s (at the
local pressure), gas_kg_s, gas_volume_fraction (of the volume flow), alpha_from (alphaPhi written by the solver, or
the upwind cell's liquid fraction); parts per disc. From the last time's fields (a steady run's final iteration).
"""
from __future__ import annotations

import numpy as np

SCALE = 1e-3
KEYS = {"name", "plane", "within"}
RING_MM = 1.0              # the ring checked outside each region (at least two cells)


def errors(study: dict) -> list[str]:
    sec = study.get("sections")
    if sec is None:
        return []
    if not isinstance(sec, list) or not sec:
        return ["sections: a list of {name, plane: {axis, at} | {point, normal}, within: [discs] | {box}}"]
    errs, names = [], set()
    for i, s in enumerate(sec):
        w = f"sections[{i}]"
        if not isinstance(s, dict):
            errs.append(f"{w}: a dict")
            continue
        errs += [f"{w}: unknown key {k!r} (keys: {sorted(KEYS)})" for k in s if k not in KEYS]
        nm = s.get("name")
        if not isinstance(nm, str) or not nm.strip():
            errs.append(f"{w}.name: a name")
        elif nm in names:
            errs.append(f"{w}.name: {nm!r} twice")
        names.add(nm)
        pl = s.get("plane")
        if not isinstance(pl, dict):
            errs.append(f"{w}.plane: {{axis: x|y|z, at: mm}} or {{point: [x, y, z], normal: [nx, ny, nz]}}")
        elif "axis" in pl:
            if pl.get("axis") not in ("x", "y", "z") or not _num(pl.get("at")) or set(pl) - {"axis", "at"}:
                errs.append(f"{w}.plane: {{axis: x|y|z, at: mm}}")
        elif not (_vec(pl.get("point")) and _vec(pl.get("normal")) and np.linalg.norm(pl["normal"]) > 0) \
                or set(pl) - {"point", "normal"}:
            errs.append(f"{w}.plane: {{point: [x, y, z], normal: [nx, ny, nz]}} (mm; the normal not zero)")
        wi = s.get("within")
        if wi is None:
            continue
        if isinstance(wi, dict):
            if set(wi) != {"box"} or not _box(wi.get("box")):
                errs.append(f"{w}.within: {{box: [x0, y0, z0, x1, y1, z1]}} (mm, x0 < x1 ...)")
        elif isinstance(wi, list) and wi:
            pn = set()
            for j, d in enumerate(wi):
                disc = isinstance(d, dict) and _vec(d.get("center")) and _num(d.get("radius")) and d["radius"] > 0 \
                    and not set(d) - {"name", "center", "radius"}
                box = isinstance(d, dict) and _box(d.get("box")) and not set(d) - {"name", "box"}
                if not (disc or box):
                    errs.append(f"{w}.within[{j}]: {{name, center: [x, y, z], radius}} or {{name, box: [x0, y0, z0, "
                                "x1, y1, z1]}} (mm, x0 < x1 ...)")
                elif d.get("name") in pn:
                    errs.append(f"{w}.within[{j}].name: {d['name']!r} twice")
                pn.add(d.get("name") if isinstance(d, dict) else None)
        else:
            errs.append(f"{w}.within: a list of discs [{{center, radius}}] or {{box}}")
    return errs


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _box(b) -> bool:
    return isinstance(b, list) and len(b) == 6 and all(_num(v) for v in b) and all(b[k] < b[k + 3] for k in range(3))


def _vec(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(_num(x) for x in v)


def _plane(pl: dict):
    if "axis" in pl:
        n = np.zeros(3)
        k = "xyz".index(pl["axis"])
        n[k] = 1.0
        p0 = np.zeros(3)
        p0[k] = float(pl["at"])
        return p0, n
    n = np.asarray(pl["normal"], float)
    return np.asarray(pl["point"], float), n / np.linalg.norm(n)


def _regions(s: dict, p0, n, Cf, grow: float = 0.0) -> list:
    """[(name, mask over Cf)] for the section's discs or box (grown by `grow` mm), or the whole plane."""
    wi = s.get("within")
    if wi is None:
        return [(None, np.ones(len(Cf), bool))]
    if isinstance(wi, dict):
        b = np.asarray(wi["box"], float)
        lo, hi = b[:3] - grow, b[3:] + grow
        return [(None, np.all((Cf >= lo) & (Cf <= hi), axis=1))]
    out = []
    for j, d in enumerate(wi):
        if "box" in d:
            b = np.asarray(d["box"], float)
            out.append((d.get("name") or f"box{j + 1}", np.all((Cf >= b[:3] - grow) & (Cf <= b[3:] + grow), axis=1)))
            continue
        c = np.asarray(d["center"], float)
        c = c - np.dot(c - p0, n) * n                     # the centre on the plane
        r = Cf - c
        r = r - np.outer(r @ n, n)
        out.append((d.get("name") or f"disc{j + 1}", np.linalg.norm(r, axis=1) <= float(d["radius"]) + grow))
    return out


def results(study: dict, mesh, plan: dict, fields: dict) -> tuple[dict, list]:
    """results.sections and warnings. Needs fields['phi_internal'] (foam.read_fields(..., internal_flux=True))."""
    secs = study.get("sections") or []
    if not secs:
        return {}, []
    phi = fields.get("phi_internal")
    ni = len(mesh.neighbour)
    if phi is None or np.ndim(phi) != 1 or len(phi) != ni:
        return {}, ["sections: the case's internal face fluxes (phi) could not be read (a case that stays decomposed, "
                    "or no case): no section flows"]
    g = mesh.geometry()
    C, Cf, Sf = g["C"], g["Cf"][:ni], g["Sf"][:ni]
    own, nei = mesh.owner[:ni], mesh.neighbour
    vf = plan.get("vof") or {}
    gas = vf.get("gas")
    alpha = fields.get("alpha")
    aphi = fields.get("alphaPhi_internal")
    pa = fields.get("p_abs")
    out, warns = {}, []
    for s in secs:
        p0, n = _plane(s["plane"])
        do, dn = (C[own] - p0) @ n, (C[nei] - p0) @ n
        cross = (do < 0) != (dn < 0)
        sign = np.where(do < 0, 1.0, -1.0)                 # owner below the plane: phi counts along the normal
        parts = {}
        allm = np.zeros(ni, bool)
        for pn, m in _regions(s, p0, n, Cf):
            f = np.nonzero(cross & m)[0]
            allm |= cross & m
            parts[pn] = _flows(f, sign, phi, Sf, n, alpha, aphi, pa, own, nei, vf, gas)
        grown = np.zeros(ni, bool)
        for _, m in _regions(s, p0, n, Cf, grow=RING_MM):
            grown |= m
        ring = np.nonzero(cross & grown & ~allm)[0]
        f_all = np.nonzero(allm)[0]
        rec = _flows(f_all, sign, phi, Sf, n, alpha, aphi, pa, own, nei, vf, gas)
        rec["normal"] = np.round(n, 4).tolist()
        if len(parts) > 1:
            rec["parts"] = parts
        if not len(f_all):
            warns.append(f"sections.{s['name']}: no mesh faces cross the plane there (no fluid, or the plane misses "
                         "the passage)")
        if len(ring):
            q_ring = float((sign[ring] * phi[ring]).sum())
            rec["ring_outside"] = {"faces": int(len(ring)), "flow_m3s": float(f"{q_ring:.4g}")}
            warns.append(f"sections.{s['name']}: {len(ring)} faces cross the plane within {RING_MM:g} mm outside the "
                         f"region (flow {q_ring:.3g} m3/s): it cuts fluid; widen it to span the whole passage")
        out[s["name"]] = rec
    out["_note"] = (f"from the fields at time {fields.get('time')} (a steady run's final iteration; not averaged); "
                    "positive along each section's normal")
    return out, warns


def _flows(f, sign, phi, Sf, n, alpha, aphi, pa, own, nei, vf, gas) -> dict:
    q = sign[f] * phi[f]
    area = float(np.abs(Sf[f] @ n).sum())
    rec = {"faces": int(len(f)), "area_mm2": float(f"{area:.5g}"),
           "flow_m3s": float(f"{q.sum():.6g}"), "flow_m3h": float(f"{q.sum() * 3600:.5g}"),
           "forward_m3s": float(f"{q[q > 0].sum():.5g}"), "backward_m3s": float(f"{q[q < 0].sum():.5g}")}
    if area > 0:
        rec["mean_velocity_m_s"] = float(f"{q.sum() / (area * SCALE ** 2):.4g}")
    if alpha is None or not len(f):
        return rec
    a = np.asarray(alpha, float)
    if aphi is not None:
        ql = sign[f] * np.asarray(aphi, float)[f]
        how = "alphaPhi (the solver's liquid flux)"
    else:
        up = np.where(phi[f] >= 0, own[f], nei[f])             # the upwind cell
        ql = q * np.clip(a[up], 0.0, 1.0)
        how = "the upwind cell's liquid fraction"
    qg = q - ql
    rho_l = float(vf.get("rho_l", 998.2))
    rec.update(liquid_m3s=float(f"{ql.sum():.6g}"), liquid_kg_s=float(f"{rho_l * ql.sum():.6g}"),
               alpha_from=how)
    if gas:
        if pa is not None:
            p = 0.5 * (np.asarray(pa, float)[own[f]] + np.asarray(pa, float)[nei[f]])
            rho_g = np.clip(p, 1.0, None) / (float(gas["R"]) * float(gas["T_K"]))
        else:
            rho_g = np.full(len(f), float(vf.get("rho_v", 1.2)))
        rec.update(gas_m3s=float(f"{qg.sum():.5g}"), gas_kg_s=float(f"{(rho_g * qg).sum():.5g}"),
                   gas_volume_fraction=float(f"{qg.sum() / q.sum():.4g}") if abs(q.sum()) > 0 else None)
    else:
        rec["vapour_m3s"] = float(f"{qg.sum():.5g}")
    return rec
