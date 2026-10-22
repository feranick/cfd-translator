"""porous.py — porous zones: packed beds, catalyst monoliths, filters and screens (Darcy-Forchheimer).
Version: 2026.10.21.2

A porous zone is the set of cells whose centres lie in a region (box, cylinder, sphere). The fluid there feels
a resistance per unit volume  -(mu d + 1/2 rho |U| f) U  with d (1/m2) and f (1/m) given along the zone's axes
e1, e2, e3; U is the superficial velocity (flow rate / total area), as in OpenFOAM. The coefficients come
directly (darcy, forchheimer), from the Ergun equation of a packed bed, or from laminar flow in the channels of
a monolith (resistance along e1; across, 1000 times more).

OpenFOAM 14 (checked against the source, src/fvModels/general/porosityForce and porosityModel/fixedCoeff):
constant/fvModels  <name> {type porosityForce; porosityForce {cellZone; type fixedCoeff; alpha; beta; rhoRef;
coordinateSystem {type cartesian; origin; coordinateRotation {type axesRotation; e1; e3}}}}. fixedCoeff's
force per unit volume is rhoRef (alpha + beta |U|) U, so alpha = (mu / rho) d and beta = f / 2; it needs no
viscosity field (DarcyForchheimer looks up nu or mu fields, which incompressibleFluid may not register). For
the kinematic (incompressible) momentum equation rhoRef must stay 1; for the fluid solver it is the density.

    validate(study, errs)
    build_zones(study, mesh, plan)              cells per zone into mesh.zones and plan["porous"]
    fv_model_entries(plan, energy)              constant/fvModels entries
    momentum_source(plan, mesh, U) -> (n, 3)    force of the zones on the fluid per cell (N), for the balance
    results(study, mesh, plan, fields, out, warns)
"""
from __future__ import annotations

import numpy as np

SCALE = 1e-3


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vec(v, n=3) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(_num(x) for x in v)


def validate(st: dict, errs: list) -> None:
    pz = st.get("porous")
    if pz is None:
        return
    if not isinstance(pz, list) or not pz:
        errs.append("porous: a list of {name, region, darcy/forchheimer | ergun | monolith}")
        return
    names = {z["name"] for z in st.get("rotating") or [] if isinstance(z, dict) and "name" in z}
    for i, z in enumerate(pz):
        w = f"porous[{i}]"
        if not isinstance(z, dict):
            errs.append(f"{w}: {{name, region, darcy/forchheimer | ergun | monolith}}")
            continue
        bad = set(z) - {"name", "region", "darcy", "forchheimer", "ergun", "monolith", "axes"}
        if bad:
            errs.append(f"{w}: unknown keys {sorted(bad)}")
        n = z.get("name")
        if not isinstance(n, str) or not n.replace("_", "").isalnum() or not n[:1].isalpha():
            errs.append(f"{w}.name: letters, digits, _ (starting with a letter)")
        elif n in names:
            errs.append(f"{w}.name: {n!r} is used twice (porous and rotating zones share names)")
        else:
            names.add(n)
        r = z.get("region")
        if not isinstance(r, dict) or len(r) != 1 or next(iter(r)) not in ("box", "cylinder", "sphere"):
            errs.append(f"{w}.region: {{box: [x0, y0, z0, x1, y1, z1]}} | {{cylinder: {{center, axis, radius, span}}}} | "
                        "{sphere: {center, radius}} (mm)")
        elif "box" in r and (not _vec(r["box"], 6) or any(r["box"][k] >= r["box"][k + 3] for k in range(3))):
            errs.append(f"{w}.region.box: [x0, y0, z0, x1, y1, z1] mm with x0 < x1 ...")
        elif "cylinder" in r:
            c = r["cylinder"]
            if not isinstance(c, dict) or not _vec(c.get("center")) or not _vec(c.get("axis")) or not _num(c.get("radius")) \
                    or c["radius"] <= 0 or ("span" in c and not _vec(c["span"], 2)):
                errs.append(f"{w}.region.cylinder: {{center: [x, y, z], axis: [ax, ay, az], radius, span: [s0, s1]}} mm")
        elif "sphere" in r:
            c = r["sphere"]
            if not isinstance(c, dict) or not _vec(c.get("center")) or not _num(c.get("radius")) or c["radius"] <= 0:
                errs.append(f"{w}.region.sphere: {{center: [x, y, z], radius}} mm")
        how = [k for k in ("ergun", "monolith") if k in z] + (["darcy/forchheimer"] if ("darcy" in z or "forchheimer" in z) else [])
        if len(how) != 1:
            errs.append(f"{w}: give one of darcy/forchheimer, ergun, monolith (got {how or 'none'})")
        for k in ("darcy", "forchheimer"):
            if k in z and not ((_num(z[k]) and z[k] >= 0) or (_vec(z[k]) and min(z[k]) >= 0 and max(z[k]) > 0)):
                errs.append(f"{w}.{k}: a number >= 0 or [along e1, e2, e3] (1/m2 for darcy, 1/m for forchheimer)")
        if "ergun" in z:
            e = z["ergun"]
            if not isinstance(e, dict) or not _num(e.get("particle_diameter")) or e["particle_diameter"] <= 0 \
                    or not _num(e.get("porosity")) or not 0 < e["porosity"] < 1:
                errs.append(f"{w}.ergun: {{particle_diameter: mm, porosity: 0..1}}")
        if "monolith" in z:
            m = z["monolith"]
            if not isinstance(m, dict) or not _num(m.get("channel")) or m["channel"] <= 0 or not _num(m.get("open_fraction")) \
                    or not 0 < m["open_fraction"] < 1 or m.get("shape", "square") not in ("round", "square"):
                errs.append(f"{w}.monolith: {{channel: mm (hydraulic diameter), open_fraction: 0..1, shape: round | square}}")
        if "axes" in z:
            a = z["axes"]
            if not (isinstance(a, list) and len(a) == 2 and _vec(a[0]) and _vec(a[1])):
                errs.append(f"{w}.axes: [[e1], [e2]] (e3 = e1 x e2)")
            else:
                e1, e2 = np.asarray(a[0], float), np.asarray(a[1], float)
                if np.linalg.norm(e1) == 0 or np.linalg.norm(e2) == 0 or abs(e1 @ e2) > 1e-6 * np.linalg.norm(e1) * np.linalg.norm(e2):
                    errs.append(f"{w}.axes: e1 and e2 non-zero and perpendicular")
        elif "monolith" in z:
            errs.append(f"{w}.axes: a monolith needs axes [[e1 = channel direction], [e2]]")


def coefficients(z: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """(d [1/m2], f [1/m]) along e1, e2, e3, and how they were obtained."""
    if "ergun" in z:
        dp, eps = z["ergun"]["particle_diameter"] * SCALE, z["ergun"]["porosity"]
        d = 150 * (1 - eps) ** 2 / (eps ** 3 * dp ** 2)
        f = 3.5 * (1 - eps) / (eps ** 3 * dp)
        return np.full(3, d), np.full(3, f), "Ergun (packed bed)"
    if "monolith" in z:
        m = z["monolith"]
        dh, eps = m["channel"] * SCALE, m["open_fraction"]
        c = 32.0 if m.get("shape", "square") == "round" else 28.4
        d1 = c / (eps * dh ** 2)
        return np.array([d1, 1000 * d1, 1000 * d1]), np.zeros(3), f"monolith ({m.get('shape', 'square')} channels, laminar)"
    d, f = z.get("darcy", 0.0), z.get("forchheimer", 0.0)
    d = np.full(3, float(d)) if _num(d) else np.asarray(d, float)
    f = np.full(3, float(f)) if _num(f) else np.asarray(f, float)
    return d, f, "given"


def _axes(z: dict) -> np.ndarray:
    if "axes" in z:
        e1 = np.asarray(z["axes"][0], float)
        e2 = np.asarray(z["axes"][1], float)
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)
        return np.vstack([e1, e2, np.cross(e1, e2)])
    return np.eye(3)


def _cells(region: dict, C: np.ndarray) -> np.ndarray:
    if "box" in region:
        b = np.asarray(region["box"], float)
        return np.nonzero(np.all((C >= b[:3]) & (C <= b[3:]), axis=1))[0]
    if "sphere" in region:
        s = region["sphere"]
        return np.nonzero(np.linalg.norm(C - np.asarray(s["center"], float), axis=1) <= float(s["radius"]))[0]
    c = region["cylinder"]
    a = np.asarray(c["axis"], float)
    a /= np.linalg.norm(a)
    d = C - np.asarray(c["center"], float)
    ax = d @ a
    rad = np.linalg.norm(d - np.outer(ax, a), axis=1)
    s0, s1 = c.get("span", [-np.inf, np.inf])
    return np.nonzero((rad <= float(c["radius"])) & (ax >= s0) & (ax <= s1))[0]


def build_zones(study: dict, mesh, plan: dict) -> None:
    from .polymesh import MeshError
    pz = study.get("porous") or []
    plan.pop("porous", None)
    if not pz:
        return
    g = mesh.geometry()
    taken = np.zeros(mesh.ncells, bool)
    for cells in (mesh.zones or {}).values():
        taken[cells] = True
    plan["porous"] = {}
    for z in pz:
        cells = _cells(z["region"], g["C"])
        snapped = (mesh.info.get("snapped_zones") or {}).get(z["name"])
        if snapped is not None and len(snapped):
            cells = np.asarray(snapped, np.int64)          # snappyHexMesh's cellZone: faces on the region
        if not len(cells):
            raise MeshError(f"porous zone {z['name']!r} contains no cell centres (check its region)")
        if taken[cells].any():
            raise MeshError(f"porous zone {z['name']!r} overlaps another zone ({int(taken[cells].sum())} cells)")
        taken[cells] = True
        mesh.zones[z["name"]] = cells
        d, f, how = coefficients(z)
        plan["porous"][z["name"]] = {"cells": int(len(cells)), "volume_mm3": float(g["V"][cells].sum()),
                                     "darcy_1_m2": d.tolist(), "forchheimer_1_m": f.tolist(), "from": how,
                                     "axes": _axes(z).tolist()}


def _alpha_beta(zp: dict, nu: float) -> tuple[np.ndarray, np.ndarray]:
    return nu * np.asarray(zp["darcy_1_m2"], float), 0.5 * np.asarray(zp["forchheimer_1_m"], float)


def fv_model_entries(plan: dict, energy: dict | None) -> list[str]:
    out = []
    nu = (energy["mu"] / energy["rho"]) if energy else plan["nu"]
    rho_ref = float(energy["rho"]) if energy else 1.0         # 1: the incompressible equation is per unit mass
    for name, zp in (plan.get("porous") or {}).items():
        al, be = _alpha_beta(zp, nu)
        e = np.asarray(zp["axes"], float)
        v = lambda x: f"({x[0]:.12g} {x[1]:.12g} {x[2]:.12g})"  # noqa: E731
        out.append(f"porous_{name}\n{{\n    type            porosityForce;\n\n    porosityForce\n    {{\n"
                   f"        cellZone        {name};\n        type            fixedCoeff;\n"
                   f"        alpha           {v(al)};\n        beta            {v(be)};\n"
                   f"        rhoRef          {rho_ref:.12g};\n\n        coordinateSystem\n        {{\n"
                   "            type            cartesian;\n            origin          (0 0 0);\n"
                   "            coordinateRotation\n            {\n                type            axesRotation;\n"
                   f"                e1              {v(e[0])};\n                e3              {v(e[2])};\n"
                   "            }\n        }\n    }\n}")
    return out


def momentum_source(plan: dict, mesh, U: np.ndarray) -> np.ndarray | None:
    """Force of the porous zones on the fluid per cell (N): -rho (alpha + beta |U|) U V in the zone axes."""
    pz = plan.get("porous") or {}
    if not pz:
        return None
    en = plan.get("energy")
    rho = float(en["rho"]) if en else float(plan["rho"])
    nu = (en["mu"] / en["rho"]) if en else plan["nu"]
    V = mesh.geometry()["V"] * SCALE ** 3
    F = np.zeros((mesh.ncells, 3))
    for name, zp in pz.items():
        cells = mesh.zones[name]
        al, be = _alpha_beta(zp, nu)
        E = np.asarray(zp["axes"], float)                      # rows e1, e2, e3
        u = np.asarray(U, float)[cells]
        ul = u @ E.T                                           # components along the axes
        mag = np.linalg.norm(u, axis=1)[:, None]
        fl = -(al + be * mag) * ul
        F[cells] = rho * (fl @ E) * V[cells][:, None]
    return F


def results(study: dict, mesh, plan: dict, fields: dict, out: dict, warns: list) -> None:
    pz = plan.get("porous") or {}
    if not pz:
        return
    from .post import _r
    g = mesh.geometry()
    en = plan.get("energy")
    U = np.asarray(fields["U"], float)
    p = np.asarray(fields["p"], float)
    p_pa = p - en["p_op"] if en else p * plan["rho"]
    F = momentum_source(plan, mesh, U)
    ni = mesh.ninternal
    own, nei = mesh.owner[:ni], mesh.neighbour[:ni]
    Sf = g["Sf"][:ni] * SCALE ** 2
    res = {}
    for name, zp in pz.items():
        cells = mesh.zones[name]
        inz = np.zeros(mesh.ncells, bool)
        inz[cells] = True
        V = g["V"][cells]
        um = np.linalg.norm(U[cells], axis=1)
        rec = {"cells": zp["cells"], "volume_mm3": _r(zp["volume_mm3"]), "from": zp["from"],
               "darcy_1_m2": _r(zp["darcy_1_m2"]), "forchheimer_1_m": _r(zp["forchheimer_1_m"]),
               "superficial_velocity": {"mean": _r(float((um * V).sum() / V.sum())), "min": _r(um.min()), "max": _r(um.max())},
               "force_on_fluid_N": _r(F[cells].sum(axis=0))}
        # faces between the zone and the rest: flow in (upstream side) and out (downstream side)
        edge = inz[own] != inz[nei]
        if edge.any():
            fo, fn = own[edge], nei[edge]
            Sz = np.where(inz[fo][:, None], Sf[edge], -Sf[edge])      # normal out of the zone
            uf = 0.5 * (U[fo] + U[fn])
            qn = np.einsum("ij,ij->i", uf, Sz)                         # m3/s out of the zone
            outside = np.where(inz[fo], fn, fo)
            qin, qout = np.clip(-qn, 0, None), np.clip(qn, 0, None)
            if qin.sum() > 0 and qout.sum() > 0:
                p_in = float((qin * p_pa[outside]).sum() / qin.sum())
                p_out = float((qout * p_pa[outside]).sum() / qout.sum())
                A = np.linalg.norm(Sz, axis=1)
                un = qout / np.where(A > 0, A, 1)
                m = qout > 0
                ubar = float((un[m] * A[m]).sum() / A[m].sum())
                gamma = 1 - float((np.abs(un[m] - ubar) * A[m]).sum()) / (2 * ubar * float(A[m].sum())) if ubar > 0 else None
                rec.update({"pressure_drop_Pa": _r(p_in - p_out), "flow_in_m3s": _r(qin.sum()),
                            "uniformity_index_out": _r(gamma)})
        # cells on the zone's faces: OpenFOAM's collocated grid makes their velocity dip, so the resistance and the
        # pressure drop come out low (porous_channel_1d on mochi: -2.6 % with 2 of every 20 cells along the flow)
        if edge.any():
            n_edge = len(np.unique(np.concatenate([own[edge], nei[edge]])[inz[np.concatenate([own[edge], nei[edge]])]]))
            frac = n_edge / max(len(cells), 1)
            rec["face_cell_fraction"] = _r(frac)
            if frac > 0.05:
                warns.append(f"porous zone {name!r}: {frac:.0%} of its cells lie on its faces, where the cell velocity "
                             f"dips; its pressure drop may be low by about {0.25 * frac:.1%}: refine the mesh across the zone")
        res[name] = rec
    out["porous"] = res
