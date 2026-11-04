"""case.py — an OpenFOAM case (incompressibleFluid, steady SIMPLE) from a study and its mesh.
Version: 2026.11.03.6

    write_case(study, mesh, patches, case_dir) -> plan

Everything is computed here, not left to OpenFOAM: inlet velocities face by face (uniform or the fully
developed laminar profile, scaled so the flow rate is exact), moving-wall velocities (tangential part),
turbulence inlet values (k = 1.5 (I U)^2, omega = sqrt(k) / (Cmu^0.25 L)), kinematic pressures (p / rho).
Files: system/{controlDict, fvSchemes, fvSolution}, constant/{physicalProperties, momentumTransport,
polyMesh/*}, 0/{U, p[, k, omega, nut]}, case.foam. Target: OpenFOAM (openfoam.org) 11 or later.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .format import fluid_props, thermo_props
from .polymesh import SCALE, MeshError, header

CMU = 0.09


def _uniform(v) -> str:
    if np.ndim(v) == 0:
        return f"uniform {float(v):.12g}"
    return "uniform (" + " ".join(f"{float(x):.12g}" for x in v) + ")"


def _nonuniform(a: np.ndarray) -> str:
    a = np.asarray(a, float)
    if a.ndim == 1:
        body = "\n".join(f"{x:.12g}" for x in a)
        return f"nonuniform List<scalar>\n{len(a)}\n(\n{body}\n)"
    body = "\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in a)
    return f"nonuniform List<vector>\n{len(a)}\n(\n{body}\n)"


def _field(cls: str, obj: str, dims: str, internal: str, patches: dict) -> str:
    lines = [header(cls, obj, "0"), f"dimensions      {dims};", "", f"internalField   {internal};", "",
             "boundaryField", "{"]
    for name, entries in patches.items():
        lines += [f"    {name}", "    {"]
        for k, v in entries.items():
            lines.append(f"        {k:<15} {v};")
        lines.append("    }")
    lines += ["}", "", "// " + "*" * 73 + " //", ""]
    return "\n".join(lines)


def _dict(obj: str, location: str, body: str) -> str:
    return header("dictionary", obj, location) + "\n" + body.strip() + "\n\n// " + "*" * 73 + " //\n"


# ---- inlet profiles ----------------------------------------------------------------------------
def _plane_frame(n: np.ndarray):
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _boundary_edges(rings: np.ndarray) -> np.ndarray:
    """Edges (pairs of node indices) used by exactly one face of the set."""
    e = []
    for r in rings:
        k = int((r >= 0).sum())
        for i in range(k):
            a, b = int(r[i]), int(r[(i + 1) % k])
            e.append((min(a, b), max(a, b)))
    e = np.array(e, np.int64)
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def inlet_shape(mesh, faces: np.ndarray, two_d: bool, profile: str):
    """Relative velocity (mean 1 over the area) of each inlet face, the inlet's hydraulic diameter (m),
    and a description."""
    g = mesh.geometry()
    Sf, Cf = g["Sf"][faces], g["Cf"][faces]
    A = np.linalg.norm(Sf, axis=1)
    n = Sf / A[:, None]
    rings = mesh.faces[faces]
    nodes = np.unique(rings[rings >= 0])
    P = mesh.points[nodes]
    edges = _boundary_edges(rings)
    perim = float(np.linalg.norm(mesh.points[edges[:, 0]] - mesh.points[edges[:, 1]], axis=1).sum())
    Atot = float(A.sum())
    planar = bool(np.all(n @ n[0] > 1 - 1e-6))
    if two_d:
        t = np.cross([0, 0, 1.0], n[0])
        if np.linalg.norm(t) < 1e-9:
            raise MeshError("2d inlet faces must be edge-on (normal in the XY plane)")
        t /= np.linalg.norm(t)
        s = P @ t
        W = float(s.max() - s.min())
        dh = 2 * W * SCALE
        desc = {"shape": "2D channel", "width_mm": W}
        if profile == "parabolic":
            if not planar:
                raise MeshError("a parabolic 2D inlet must be straight")
            sc = Cf @ t
            mid = 0.5 * (s.max() + s.min())
            rel = np.clip(1 - (2 * (sc - mid) / W) ** 2, 0, None)
        else:
            rel = np.ones(len(faces))
    else:
        dh = 4 * Atot / max(perim, 1e-300) * SCALE
        desc = {"shape": "inlet", "area_mm2": Atot, "perimeter_mm": perim}
        rel = np.ones(len(faces))
        if profile == "parabolic":
            if not planar:
                raise MeshError("a parabolic inlet must be planar")
            e1, e2 = _plane_frame(n[0])
            c0 = (Cf * A[:, None]).sum(axis=0) / Atot
            u, v = (P - c0) @ e1, (P - c0) @ e2
            R = float(np.sqrt(u * u + v * v).max())
            uc, vc = (Cf - c0) @ e1, (Cf - c0) @ e2
            if abs(Atot / (np.pi * R * R) - 1) < 0.03:
                rel = np.clip(1 - (uc * uc + vc * vc) / R ** 2, 0, None)
                desc = {"shape": "circle", "radius_mm": R}
            else:
                # rectangle: principal axes of the boundary points
                w, V = np.linalg.eigh(np.cov(np.column_stack([u, v]).T))
                a1 = V[:, 1]
                pu, pv = np.column_stack([u, v]) @ a1, np.column_stack([u, v]) @ np.array([-a1[1], a1[0]])
                la, lb = float(pu.max() - pu.min()), float(pv.max() - pv.min())
                if abs(Atot / (la * lb) - 1) > 0.03:
                    raise MeshError("a parabolic inlet must be a circle, a rectangle, or 2D (this one is neither)")
                q = np.column_stack([uc, vc])
                y = q @ a1 - 0.5 * (pu.max() + pu.min())
                z = q @ np.array([-a1[1], a1[0]]) - 0.5 * (pv.max() + pv.min())
                a, b = la / 2, lb / 2
                acc = np.zeros(len(faces))
                for i in range(1, 80, 2):
                    acc += (-1) ** ((i - 1) // 2) * (1 - np.cosh(i * np.pi * z / (2 * a)) / np.cosh(i * np.pi * b / (2 * a))) \
                        * np.cos(i * np.pi * y / (2 * a)) / i ** 3
                rel = np.clip(acc, 0, None)
                desc = {"shape": "rectangle", "sides_mm": [la, lb]}
    mean = float((rel * A).sum() / Atot)
    if mean <= 0:
        raise MeshError("inlet profile has zero mean (inlet too coarse: refine it)")
    return rel / mean, dh, desc


def swirl_velocity(Cf_mm: np.ndarray, spec: dict) -> np.ndarray:
    """Tangential velocity (m/s) at face centres about spec's axis through spec's centre (mm): solid body
    (rpm: omega x r) or uniform speed (tangential, m/s); right-handed about the axis."""
    a = np.asarray(spec["axis"], float)
    a /= np.linalg.norm(a)
    d = (Cf_mm - np.asarray(spec.get("center", [0, 0, 0]), float)) * SCALE
    r = d - np.outer(d @ a, a)                                     # radial vector, m
    t = np.cross(a, r)                                             # |t| = r
    if "rpm" in spec:
        return float(spec["rpm"]) * 2 * np.pi / 60 * t
    rn = np.linalg.norm(r, axis=1)
    return float(spec["tangential"]) * t / np.maximum(rn, 1e-300)[:, None]


def facet_tangential(U: np.ndarray, n: np.ndarray, angle_deg: float = 6.0) -> np.ndarray:
    """U with its normal part removed on faces it crosses by less than angle_deg (facets of a surface it moves
    along); faces it really crosses (blades, vanes) keep it."""
    un = np.einsum("ij,ij->i", U, n)
    small = np.abs(un) <= np.sin(np.radians(angle_deg)) * np.linalg.norm(U, axis=1)
    out = U.copy()
    out[small] -= un[small, None] * n[small]
    return out


def _same_rotation(spec: dict, zp: dict) -> bool:
    """Does a wall's own rotation {axis, center, rpm} turn exactly with an MRF zone (plan entry)?"""
    a = np.asarray(spec["axis"], float)
    a /= np.linalg.norm(a)
    b = np.asarray(zp["axis"], float)
    b /= np.linalg.norm(b)
    if "rpm" not in spec or abs(a @ b) < 1 - 1e-9:
        return False
    w = float(spec["rpm"]) * 2 * np.pi / 60 * (1.0 if a @ b > 0 else -1.0)     # -axis at -rpm is the same turn
    if abs(w - zp["omega_rad_s"]) > 1e-9 * max(abs(w), 1.0):
        return False
    d = np.asarray(zp["center_mm"], float) - np.asarray(spec.get("center", [0, 0, 0]), float)
    return float(np.linalg.norm(d - (d @ a) * a)) < 1e-6


def same_speed_walls(named_bcs: list, zp: dict) -> list:
    """Wall bcs with their own rotation {axis, center, rpm} turning exactly with zone zp (plan entry)."""
    return [n for n, b in named_bcs if isinstance(b, dict) and b.get("kind", "wall") == "wall"
            and isinstance(b.get("rotation"), dict) and _same_rotation(b["rotation"], zp)]


def rotation_key(axis, center) -> str:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(center if center is not None else [0, 0, 0], float)
    c = c - (c @ a) * a                                          # the axis line, not a point on it
    return ",".join(f"{v:.6g}" for v in np.concatenate([a, c]) + 0.0)


def revolution_tags(pts: np.ndarray, surf: list, axis, center, max_deg: float = 15.0, mean_deg: float = 5.0) -> list:
    """CAD faces (tags) of a triangulated surface that are surfaces of revolution about the axis: their triangles'
    normals have (almost) no part along the direction of rotation (mean under mean_deg, none over max_deg)."""
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    o = np.asarray(center if center is not None else [0, 0, 0], float)
    out = []
    for tg, tri in surf:
        if not len(tri):
            continue
        P = pts[tri]
        nv = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        ar = np.linalg.norm(nv, axis=1)
        c = P.mean(axis=1) - o
        t = np.cross(a, c)
        tn = np.linalg.norm(t, axis=1)
        ok = (ar > 0) & (tn > 1e-9 * max(float(tn.max()), 1e-30))
        if not ok.any():
            out.append(int(tg))                                  # on the axis: nothing moves across it
            continue
        sn = np.abs(np.einsum("ij,ij->i", nv[ok] / ar[ok, None], t[ok] / tn[ok, None]))
        if float((sn * ar[ok]).sum() / ar[ok].sum()) < np.sin(np.radians(mean_deg)) and \
                float(sn.max()) < np.sin(np.radians(max_deg)):
            out.append(int(tg))
    return sorted(out)


def own_rotation_velocity(mesh, g, f, spec: dict, plan: dict) -> np.ndarray:
    """The velocity of a wall with its own rotation {axis, center, rpm} at its faces f (m/s).
    - Faces whose cell turns with it (an MRF zone at the same axis and speed): the full Omega x r, which the zone's
      relative flux cancels exactly.
    - Faces of a CAD face that is a surface of revolution about the axis (mesh.info['revolution_tags'], found on
      the CAD's own triangulation by the snappy mesher): it moves along itself, so the normal part is removed face
      by face, however jagged the snapped faces are.
    - Anything else (blades, vanes) keeps the full motion, except across facets of less than 6 degrees.
    s180 (2026.11.01): 'impeller_back' (hub, shaft, discs, rim r 137) fed 0.036 m3/s through its snapped faces."""
    U = swirl_velocity(g["Cf"][f], spec)
    Sf = g["Sf"][f]
    n = Sf / np.maximum(np.linalg.norm(Sf, axis=1), 1e-300)[:, None]
    same = np.zeros(len(f), bool)
    for zn, zp in (plan.get("zones") or {}).items():
        if _same_rotation(spec, zp):
            same |= np.isin(mesh.owner[f], mesh.zones[zn])
    rev = set((mesh.info.get("revolution_tags") or {}).get(rotation_key(spec["axis"], spec.get("center")), []))
    tags = mesh.btag[f - mesh.ninternal]
    on_rev = np.isin(tags, list(rev)) & ~same if rev else np.zeros(len(f), bool)
    out = U.copy()
    un = np.einsum("ij,ij->i", U, n)
    out[on_rev] -= un[on_rev, None] * n[on_rev]
    rest = ~same & ~on_rev
    out[rest] = facet_tangential(U[rest], n[rest])
    return out


def _tangential(U: np.ndarray, n: np.ndarray) -> np.ndarray:
    return U - np.einsum("ij,ij->i", U, n)[:, None] * n


# ---- the case ----------------------------------------------------------------------------------
def _quality_message(q: dict, mesh, g) -> str:
    """Why the mesh cannot be solved on, where, and what to do (a refine sphere at each bad place)."""
    why = []
    if q.get("min_volume_mm3", 1) <= 0:
        why.append(f"cells with zero or negative volume (smallest {q['min_volume_mm3']:.3g} mm3)")
    if q.get("max_non_orthogonality", 0) >= 89:
        why.append(f"faces folded past 89 degrees (worst {q['max_non_orthogonality']:.1f}; {q.get('non_ortho_over_70', 0)} "
                   "above 70): inverted or nearly flat cells")
    if q.get("max_open_cell", 0) >= 1e-6:
        why.append(f"cells that are not closed (openness {q['max_open_cell']:.3g})")
    if not q.get("boundary_outward", True) or not q.get("upper_triangular", True):
        why.append("faces ordered or oriented wrongly")
    msg = "the mesh cannot be solved on: " + "; ".join(why or ["failed checks"])
    worst = q.get("worst_non_ortho") or []
    if worst:
        h = float(np.cbrt(np.median(np.abs(g["V"]))))
        bad = [w for w in worst if w["deg"] >= 80] or worst[:3]
        sph = [{"at": w["at"], "r": round(3 * h, 3), "size": round(h / 3, 3)} for w in bad]
        msg += (f". {q.get('folded_faces', 0)} faces folded past 89 degrees; the bad places (worst first): " +
                "; ".join(f"{w['deg']} deg at {w['at']} mm" for w in worst) +
                ". Usually small features (screw heads, thin lips, tiny gaps) smaller than the cells around them: "
                "remove them from the geometry, or refine there, e.g. mesh.refine " + json.dumps(sph) +
                ". Next to an inlet or outlet with snappy prism layers: mesh.layers.at_openings false")
    return msg


def _mrf_wall_leaks(mesh, g, zones: dict, plan: dict, patches: list) -> None:
    """Walls the flow passes through. A wall's velocity U_w (Omega x r for one that rotates, with its zone's
    MRFnoSlip or its own rotation {axis, rpm}; 0 for a stationary one) and the frame of the cell it bounds (Omega x r
    in an MRF zone, which OpenFOAM 14 subtracts from the flux of every face there, walls included:
    MRFZoneTemplates.C) give each wall face a flux (U_w - V_frame) . S. It vanishes on surfaces of revolution about
    the axis, and for a rotating wall inside its own zone; anything else feeds or drains the flow. In the s180 pump
    (2026.11.01) 0.30 m3/s left the outlet for 0.069 m3/s in; 'impeller_back' (its own rotation, outside the zone)
    carried 64 % of the inflow. Leaks above 5 % of the inflow stop the run, smaller ones are noted."""
    rot_walls = [p for p in patches if p["kind"] == "wall" and p["bc"].get("rotation")]
    if not zones and not rot_walls:
        return
    q_in = sum(abs(float(v.get("Q", 0.0))) for v in plan.get("inlets", {}).values()) or None
    Sf = g["Sf"] * SCALE ** 2
    # the frame velocity of each cell (m/s): Omega x r in a zone, 0 elsewhere
    frame_zone = np.full(mesh.ncells, -1)
    zlist = list(zones)
    rev_all = set().union(*[set(v) for v in (mesh.info.get("revolution_tags") or {}).values()]) \
        if mesh.info.get("revolution_tags") else set()
    for k, zn in enumerate(zlist):
        frame_zone[mesh.zones[zn]] = k
    bad = []
    for p in patches:
        if p["kind"] != "wall":
            continue
        f = mesh.patch_faces(p["name"])
        r = p["bc"].get("rotation")
        if isinstance(r, str):
            zp = plan["zones"][r]
            Uw = swirl_velocity(g["Cf"][f], {"axis": zp["axis"], "center": zp["center_mm"],
                                             "rpm": zp["omega_rad_s"] * 60 / (2 * np.pi)})
            how = f"rotating with zone {r!r}"
        elif isinstance(r, dict):
            Uw = own_rotation_velocity(mesh, g, f, r, plan)              # as the case writes it
            how = "rotating (its own rotation)"
        else:
            Uw = np.zeros((len(f), 3))
            how = "stationary"
        Vf = np.zeros((len(f), 3))
        zc = frame_zone[mesh.owner[f]]
        for k, zn in enumerate(zlist):
            m = zc == k
            if m.any():
                zp = plan["zones"][zn]
                Vf[m] = swirl_velocity(g["Cf"][f[m]], {"axis": zp["axis"], "center": zp["center_mm"],
                                                       "rpm": zp["omega_rad_s"] * 60 / (2 * np.pi)})
        rel = Uw - Vf
        leak = np.einsum("ij,ij->i", rel, Sf[f])                                    # m3/s per face
        # what counts: faces whose relative motion crosses them by more than ~6 degrees, except on CAD faces that
        # are surfaces of revolution about the axis (their snapped or faceted faces carry fluxes of both signs that
        # cancel next to each other: s180's 'impeller_back')
        tags = mesh.btag[f - mesh.ninternal]
        across = np.abs(leak) > 0.1 * np.linalg.norm(rel, axis=1) * np.linalg.norm(Sf[f], axis=1)
        if rev_all:
            across &= ~np.isin(tags, list(rev_all))
        if not across.any():
            continue
        lk, ff, zz = leak[across], f[across], zc[across]
        k = int(np.argmax(np.abs(lk)))
        n_in = int((zz >= 0).sum())
        where = ("inside zone " + ", ".join(repr(zlist[i]) for i in sorted(set(zz[zz >= 0].tolist())))
                 if n_in == len(zz) else "outside the rotating zones" if n_in == 0 else
                 f"{n_in} inside a rotating zone, {len(zz) - n_in} outside")
        bad.append((p["name"], how, float(np.abs(lk).sum()), len(ff), where, np.round(g["Cf"][ff][k], 1).tolist()))
    if not bad:
        return
    total = sum(b[2] for b in bad)
    frac = total / q_in if q_in else None
    parts = "; ".join(f"wall {nm!r} ({how}): {n} faces {where}, {gross:.3g} m3/s through them (e.g. at {at} mm)"
                      for nm, how, gross, n, where, at in bad)
    msg = (f"fluid would flow through walls ({total:.3g} m3/s" + (f", {frac:.0%} of the inflow" if frac is not None
           else "") + f"): {parts}. A wall may move across its own surface only where the cells' frame moves with "
           "it: rotating walls (by zone or their own rotation) belong inside their MRF zone, stationary walls "
           "outside every zone, unless the wall is a surface of revolution about the axis")
    if frac is not None and frac > 0.05:
        raise MeshError(msg)
    if frac is None or frac > 0.002:
        plan.setdefault("notes", []).append(msg)


def _check_rotating_wall(nm: str, zn: str, f, mesh, g, zones: dict, plan: dict) -> None:
    """A wall that rotates with an MRF zone should lie in it. OpenFOAM's MRFnoSlip gives every face of the patch
    Omega x r whichever cell it touches (MRFnoSlipFvPatchVectorField.C: makeAbsolute on the whole patch), so faces
    whose cells fall just outside the zone are fine: a semi-open impeller's vane tips face the zone's flat ends
    across a narrow side gap (s180, 2026.10.31: 2 % of the impeller's faces, 0.5 mm from the ends with 1 mm cells).
    Faces farther out than two of their cells mean the zone is misplaced."""
    out = ~np.isin(mesh.owner[f], mesh.zones[zn])
    if not out.any():
        return
    z = zones[zn]
    a = np.asarray(z["axis"], float)
    a /= np.linalg.norm(a)
    d = g["Cf"][f[out]] - np.asarray(z.get("center", [0, 0, 0]), float)
    ax = d @ a
    rad = np.linalg.norm(d - np.outer(ax, a), axis=1)
    s0, s1 = z.get("span", [-np.inf, np.inf])
    beyond = np.maximum.reduce([rad - float(z["radius"]), s0 - ax, ax - s1, np.zeros(len(ax))])
    h = np.cbrt(np.abs(g["V"][mesh.owner[f[out]]]))
    far = beyond > 2 * h
    frac = float(out.mean())
    if far.any() or frac > 0.2:
        k = int(np.argmax(beyond - 2 * h))
        raise MeshError(
            f"wall {nm!r} rotates with zone {zn!r} but {frac:.0%} of its faces are outside the zone, some "
            f"{float(beyond[k]):.3g} mm beyond it (e.g. at {np.round(g['Cf'][f[out]][k], 1).tolist()} mm): the zone "
            "(center, radius, span) must cover the rotating walls; its ends and rim belong in fluid (mid-gap) or in "
            "the solid, clear of stationary walls")
    plan.setdefault("notes", []).append(
        f"{frac:.1%} of wall {nm!r}'s faces touch cells just outside zone {zn!r} (within two cells of its boundary: "
        "vane tips facing its ends across a narrow gap); they still move at Omega x r (MRFnoSlip)")


def wall_velocity(p: dict, mesh, g, zones: dict, plan: dict) -> dict:
    """The velocity boundary condition of a wall patch: noSlip, a moving wall's tangential velocity, a rigid
    rotation (omega x r), or MRFnoSlip for a wall turning with an MRF zone (plan['patches'] notes it)."""
    nm, b = p["name"], p["bc"]
    vw = b.get("velocity")
    if not (b.get("rotation") or (vw and np.linalg.norm(vw) > 0)):
        return {"type": "noSlip"}
    f = mesh.patch_faces(nm)
    Sf = g["Sf"][f]
    n = Sf / np.linalg.norm(Sf, axis=1)[:, None]
    if b.get("rotation"):
        # rigid rotation: the full omega x r (blades move through the fluid). On the facets of a
        # surface of revolution it is tangential up to the faceting; that small normal part is
        # dropped (it fed s180's 'impeller_back' cylinders, r 137 at 1450 rpm, with a flux that
        # cancels only on average)
        spec = zones[b["rotation"]] if isinstance(b["rotation"], str) else b["rotation"]
        Ut = (own_rotation_velocity(mesh, g, f, spec, plan) if isinstance(b["rotation"], dict)
              else facet_tangential(swirl_velocity(g["Cf"][f], spec), n))
        if isinstance(b["rotation"], str) and zones[b["rotation"]].get("motion") != "sliding":
            _check_rotating_wall(nm, b["rotation"], f, mesh, g, zones, plan)
    else:
        Ut = _tangential(np.tile(np.asarray(vw, float), (len(f), 1)), n)
    same = np.allclose(Ut, Ut[0], rtol=0, atol=1e-12 * max(float(np.abs(Ut).max()), 1e-300))
    out = {"type": "fixedValue", "value": _uniform(Ut[0]) if same else _nonuniform(Ut)}
    if isinstance(b.get("rotation"), str):        # rotating with an MRF zone (OpenFOAM 14: MRFnoSlip)
        out["type"] = "MRFnoSlip"
        plan["patches"][nm]["zone"] = b["rotation"]
    plan["patches"][nm]["moving"] = True
    return out


def write_mrf(case: Path, zones: dict, plan: dict) -> None:
    """constant/MRFProperties for the rotating zones (removed when there are none)."""
    mrf = Path(case) / "constant" / "MRFProperties"
    if zones:
        body = []
        for name, z in zones.items():
            zp = plan["zones"][name]
            o = np.asarray(zp["center_mm"]) * SCALE
            body.append(f"{name}\n{{\n    cellZone        {name};\n"
                        f"    origin          ({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});\n"
                        f"    axis            ({zp['axis'][0]:.12g} {zp['axis'][1]:.12g} {zp['axis'][2]:.12g});\n"
                        f"    omega           {zp['omega_rad_s']:.12g};   // rad/s ({zones[name]['rpm']:g} rpm)\n}}")
        mrf.write_text(_dict("MRFProperties", "constant", "\n\n".join(body)))
    elif mrf.exists():
        mrf.unlink()


def nonortho_schemes(q: dict) -> tuple[str, str, int, float]:
    """(laplacian, snGrad, non-orthogonal correctors, relaxation factor) for the mesh's worst
    non-orthogonality. 70-80 degrees: limited 0.33 and two correctors; above 80 (Gmsh tetrahedra reached 88 and
    diverged): three correctors and relaxation scaled by 0.7."""
    mx = float(q.get("max_non_orthogonality", 0))
    if mx > 80:
        return "Gauss linear limited corrected 0.33", "limited corrected 0.33", 3, 0.7
    if mx > 70:
        return "Gauss linear limited corrected 0.33", "limited corrected 0.33", 2, 1.0
    return "Gauss linear limited corrected 0.5", "limited corrected 0.5", 1, 1.0


def write_case(study: dict, mesh, patches: list, case: Path) -> dict:
    """Writes the case; returns the plan (inlet values, patches, schemes chosen) for the results."""
    case = Path(case)
    for d in ("system", "constant", "0"):
        (case / d).mkdir(parents=True, exist_ok=True)
    energy = bool(study.get("energy"))
    th = thermo_props(study) if energy else None
    fl = fluid_props(study["fluid"])
    rho, mu = (th["rho"], th["mu"]) if energy else (float(fl["rho"]), float(fl["mu"]))
    nu = mu / rho
    model = study.get("model", "laminar")
    turb = model != "laminar"
    two_d = bool(study.get("2d"))
    external = study.get("domain", {}).get("kind") == "external"
    mesh.set_patches([{"name": p["name"], "type": p["type"], "tags": p["tags"]} for p in patches])
    g = mesh.geometry()
    q = mesh.quality()
    if not q["ok"]:
        raise MeshError(_quality_message(q, mesh, g))
    plan = {"rho": rho, "nu": nu, "mu": mu, "model": model, "patches": {}, "inlets": {}, "quality": q, "zones": {}}
    zones = {z["name"]: z for z in study.get("rotating", [])}
    mesh.zones = {}
    for name, z in zones.items():
        a = np.asarray(z["axis"], float)
        a /= np.linalg.norm(a)
        d = g["C"] - np.asarray(z.get("center", [0, 0, 0]), float)
        ax = d @ a
        rad = np.linalg.norm(d - np.outer(ax, a), axis=1)
        s0, s1 = z.get("span", [-np.inf, np.inf])
        cells = np.nonzero((rad <= float(z["radius"])) & (ax >= s0) & (ax <= s1))[0]
        snapped = (mesh.info.get("snapped_zones") or {}).get(name)
        if snapped is not None and len(snapped):
            cells = np.asarray(snapped, np.int64)          # snappyHexMesh's cellZone: faces on the cylinder
        if not len(cells):
            raise MeshError(f"rotating zone {name!r} contains no cell centres (check center, radius, span)")
        mesh.zones[name] = cells
        omega = float(z["rpm"]) * 2 * np.pi / 60
        walls = [p["name"] for p in patches if p["kind"] == "wall" and p["bc"].get("rotation") == name]
        plan["zones"][name] = {"cells": int(len(cells)), "omega_rad_s": omega, "axis": a.tolist(),
                               "center_mm": list(map(float, z.get("center", [0, 0, 0]))),
                               "volume_mm3": float(g["V"][cells].sum()), "rotating_walls": walls,
                               "motion": z.get("motion", "mrf")}
        # walls with their own rotation at the zone's axis and speed (an impeller's back shroud, hub, shaft): their
        # torque is the rotor's too (disk friction), so they count in rotating.<zone>.torque_Nm and power
        plan["zones"][name]["rotating_walls"] += same_speed_walls([(p["name"], p["bc"]) for p in patches],
                                                                  plan["zones"][name])
        plan["zones"][name]["monitor_walls"] = list(plan["zones"][name]["rotating_walls"])   # the torque monitor's
    from .porous import build_zones as _porous_zones
    _porous_zones(study, mesh, plan)                    # porous cellZones (after the rotating ones)
    # inlets first (they set the turbulence reference)
    U = {}
    kin = {}
    for p in patches:
        f = mesh.patch_faces(p["name"])
        b = p["bc"]
        plan["patches"][p["name"]] = {"kind": p["kind"], "type": p["type"], "faces": int(len(f)),
                                      "tags": [int(t) for t in p["tags"]]}
        if p["kind"] != "inlet":
            continue
        Sf = g["Sf"][f]
        A = np.linalg.norm(Sf, axis=1)
        n = Sf / A[:, None]
        Am2 = float(A.sum()) * SCALE ** 2
        rel, dh, desc = inlet_shape(mesh, f, two_d, b.get("profile", "uniform"))
        v = b.get("velocity")
        if energy and "total_pressure" in b:
            # isentropic total-pressure inlet: OpenFOAM sets the velocity; estimate it for the turbulence values
            outs = [float(pp["bc"].get("pressure", 0.0)) for pp in patches if pp["kind"] in ("outlet", "opening")]
            dp = max(float(b["total_pressure"]) - (min(outs) if outs else 0.0), 0.0)
            vmean = float(np.sqrt(2 * dp / rho)) if dp > 0 else 1.0
            I = float(b.get("intensity", 0.05))
            L = float(b["length_scale"]) * SCALE if "length_scale" in b else 0.07 * dh
            k = float(1.5 * (vmean * I) ** 2)
            om = float(np.sqrt(k) / (CMU ** 0.25 * L))
            kin[p["name"]] = (k, om)
            plan["inlets"][p["name"]] = {"total_pressure_Pa": float(b["total_pressure"]),
                                         "total_temperature_C": float(b.get("total_temperature", th["T_op"] - 273.15)),
                                         "U_estimate": vmean, "area_m2": Am2, "hydraulic_diameter_m": dh, **desc,
                                         **({"k": k, "omega": om} if turb else {})}
            continue
        if energy and "mass_flow" in b:
            Tin = float(b.get("temperature", th["T_op"] - 273.15)) + 273.15
            rin = th["p_op"] / (th["R"] * Tin) if th["kind"] == "gas" else th["rho"]
            v = float(b["mass_flow"]) / (rin * Am2)
        if isinstance(v, (list, tuple)):
            Uf = np.tile(np.asarray(v, float), (len(f), 1))
            vmean = float(np.linalg.norm(v))
        else:
            if v is None:
                Q = float(b["flow_rate"]) if "flow_rate" in b else float(b["flow_rate_lpm"]) / 60000.0
                v = Q / Am2
            vmean = float(v)
            Uf = -vmean * rel[:, None] * n
        if b.get("swirl"):
            Uf = Uf + _tangential(swirl_velocity(g["Cf"][f], b["swirl"]), n)     # the flow rate is unchanged
            vmean = float((np.linalg.norm(Uf, axis=1) * A).sum() / A.sum())
        Qin = float(-(np.einsum("ij,ij->i", Uf, Sf)).sum() * SCALE ** 2)
        if Qin <= 0:
            raise MeshError(f"inlet {p['name']!r}: the velocity points out of the domain")
        I = float(b.get("intensity", 0.05))
        if "length_scale" in b:
            L = float(b["length_scale"]) * SCALE
        elif external:
            ptags = [t for t, i in mesh.tag_info.items() if i.get("on") == "part"]
            pf = mesh.faces[mesh.ninternal:][np.isin(mesh.btag, ptags)]
            pp = mesh.points[np.unique(pf[pf >= 0])]
            L = 0.07 * float((pp.max(axis=0) - pp.min(axis=0)).max()) * SCALE if len(pp) else 0.07 * dh
        else:
            L = 0.07 * dh
        k = float(1.5 * (vmean * I) ** 2)
        om = float(np.sqrt(k) / (CMU ** 0.25 * L))
        U[p["name"]] = Uf
        kin[p["name"]] = (k, om)
        plan["inlets"][p["name"]] = {"Q": Qin, "U_mean": Qin / Am2, "area_m2": Am2, "hydraulic_diameter_m": dh,
                                     "profile": b.get("profile", "uniform"), **desc,
                                     **({"k": k, "omega": om, "length_scale_m": L, "intensity": I} if turb else {})}
    if kin:
        k0 = max(v[0] for v in kin.values())
        om0 = max(v[1] for v in kin.values())
        uref = max(pl.get("U_mean", pl.get("U_estimate", 1.0)) for pl in plan["inlets"].values())
    else:
        uref = float(study.get("reference", {}).get("velocity", 1.0))
        k0 = 1.5 * (uref * 0.05) ** 2
        Lr = float(study.get("reference", {}).get("length", 0)) * SCALE or mesh.info.get("diag", 1.0) * SCALE * 0.07
        om0 = float(np.sqrt(k0) / (CMU ** 0.25 * Lr))
    plan["U_ref"] = uref
    if study.get("cavitation"):                 # liquid and vapour (cavitation.py: incompressibleVoF, LTS)
        from .cavitation import write_vof
        return write_vof(study, mesh, patches, case, plan, U, kin, k0, om0, turb, zones, g, q)
    if energy:
        return _write_energy(study, mesh, patches, case, plan, U, kin, k0, om0, th, turb, external)
    # fields
    Ub, pb, kb, wb, nb = {}, {}, {}, {}, {}
    has_fixed_p = False
    for p in patches:
        nm, kind, b = p["name"], p["kind"], p["bc"]
        if kind == "empty":
            for d in (Ub, pb, kb, wb, nb):
                d[nm] = {"type": "empty"}
            continue
        if kind == "symmetry":
            for d in (Ub, pb, kb, wb, nb):
                d[nm] = {"type": "symmetry"}
            continue
        if kind == "inlet":
            Uf = U[nm]
            same = np.allclose(Uf, Uf[0], rtol=0, atol=1e-14 * max(1.0, float(np.abs(Uf).max())))
            Ub[nm] = {"type": "fixedValue", "value": _uniform(Uf[0]) if same else _nonuniform(Uf)}
            pb[nm] = {"type": "zeroGradient"}
            k, om = kin[nm]
            kb[nm] = {"type": "fixedValue", "value": _uniform(k)}
            wb[nm] = {"type": "fixedValue", "value": _uniform(om)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind in ("outlet", "opening"):
            has_fixed_p = True
            pk = float(b.get("pressure", 0.0)) / rho
            pb[nm] = {"type": "fixedValue", "value": _uniform(pk)}
            if kind == "outlet":
                Ub[nm] = {"type": "inletOutlet", "inletValue": _uniform([0, 0, 0]), "value": _uniform([0, 0, 0])}
            else:
                Ub[nm] = {"type": "pressureInletOutletVelocity", "value": _uniform([0, 0, 0])}
            kb[nm] = {"type": "inletOutlet", "inletValue": _uniform(k0), "value": _uniform(k0)}
            wb[nm] = {"type": "inletOutlet", "inletValue": _uniform(om0), "value": _uniform(om0)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind == "wall":
            Ub[nm] = wall_velocity(p, mesh, g, zones, plan)
            pb[nm] = {"type": "zeroGradient"}
            kb[nm] = {"type": "kqRWallFunction", "value": _uniform(k0)}
            wb[nm] = {"type": "omegaWallFunction", "value": _uniform(om0)}
            nb[nm] = {"type": "nutkWallFunction", "value": _uniform(0)}
        elif kind == "slip":
            Ub[nm] = {"type": "slip"}
            pb[nm] = {"type": "zeroGradient"}
            kb[nm] = {"type": "zeroGradient"}
            wb[nm] = {"type": "zeroGradient"}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        else:
            raise MeshError(f"unknown bc kind {kind}")
    U0 = [0.0, 0.0, 0.0]
    if external and len(U) == 1:
        Uf = next(iter(U.values()))
        U0 = Uf.mean(axis=0).tolist()
    (case / "0" / "U").write_text(_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", _uniform(U0), Ub))
    (case / "0" / "p").write_text(_field("volScalarField", "p", "[0 2 -2 0 0 0 0]", _uniform(0), pb))
    if turb:
        (case / "0" / "k").write_text(_field("volScalarField", "k", "[0 2 -2 0 0 0 0]", _uniform(k0), kb))
        (case / "0" / "omega").write_text(_field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", _uniform(om0), wb))
        (case / "0" / "nut").write_text(_field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", _uniform(0), nb))
    # constant
    (case / "constant" / "physicalProperties").write_text(_dict("physicalProperties", "constant",
        f"viscosityModel  constant;\n\nnu              {nu:.12g};\n\n// rho = {rho:g} kg/m3 (results only; the solver is kinematic)"))
    if turb:
        mt = "simulationType  RAS;\n\nRAS\n{\n    model           kOmegaSST;\n\n    turbulence      on;\n\n    printCoeffs     on;\n}"
    else:
        mt = "simulationType  laminar;"
    (case / "constant" / "momentumTransport").write_text(_dict("momentumTransport", "constant", mt))
    write_mrf(case, zones, plan)
    mesh.write(case)
    # system
    s = study.get("solve", {})
    iters = int(s.get("iterations", 3000 if turb else 2000))
    from .transient import settings as _transient
    tr = _transient(study, mesh)
    if tr:
        plan["transient"] = tr
        iters = max(int(tr["init"]), 1)      # the steady start (its solution is the transient run's initial field)
    res = float(s.get("residual", 1e-5))
    unstructured = q["max_non_orthogonality"] > 5.0
    plan["schemes"] = "unstructured" if unstructured else "orthogonal"
    relax = s.get("relax", {})
    rf = nonortho_schemes(q)[3]                  # very skewed tetrahedra: gentler relaxation
    rU = float(relax.get("U", (0.7 if unstructured else 0.9) * rf))
    rO = float(relax.get("other", (0.7 if unstructured else 0.9) * rf))
    plan.update({"iterations": iters, "residual": res, "relax": {"U": rU, "other": rO}, "fixed_pressure": has_fixed_p})
    walls = [p["name"] for p in patches if p["type"] == "wall"]
    funcs = []
    if walls:
        funcs.append("    wallShearStress\n    {\n        type            wallShearStress;\n"
                     "        libs            (\"libfieldFunctionObjects.so\");\n"
                     f"        patches         ({' '.join(walls)});\n        writeControl    writeTime;\n    }}")
        if turb:
            funcs.append("    yPlus\n    {\n        type            yPlus;\n"
                         "        libs            (\"libfieldFunctionObjects.so\");\n        writeControl    writeTime;\n    }")
    # per-iteration monitors (torque, inlet/outlet total pressure) for window averages (average.py)
    from .average import monitor_functions
    # the body's walls (external: every wall patch of the part's faces; internal: all walls), for forces
    ptags = {int(t) for t, i in (mesh.tag_info or {}).items() if i.get("on") == "part"}
    plan["body_walls"] = [p["name"] for p in patches if p["type"] == "wall"
                          and (not external or (p["tags"] and all(int(t) in ptags for t in p["tags"])))]
    mf, mon = monitor_functions(study, plan, float(plan["rho"]))
    funcs += mf
    plan["monitors"] = mon
    sc = _scalar_parts(study, plan, patches, case, turb, nu, None)
    funcs += sc["funcs"]
    cd = (f"solver          incompressibleFluid;\n\nstartFrom       startTime;\n\nstartTime       0;\n\n"
          f"stopAt          endTime;\n\nendTime         {iters};\n\ndeltaT          1;\n\nwriteControl    timeStep;\n\n"
          f"writeInterval   {iters};\n\npurgeWrite      0;\n\nwriteFormat     ascii;\n\nwritePrecision  12;\n\n"
          "writeCompression off;\n\ntimeFormat      general;\n\ntimePrecision   6;\n\nrunTimeModifiable false;\n\n"
          "functions\n{\n" + "\n".join(funcs) + "\n}")
    (case / "system" / "controlDict").write_text(_dict("controlDict", "system", cd))
    if unstructured:
        grad = "default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n    grad(k)         cellLimited Gauss linear 1;\n    grad(omega)     cellLimited Gauss linear 1;"
        lap, sn, nonorth, _ = nonortho_schemes(q)
    else:
        grad, lap, sn, nonorth = "default         Gauss linear;", "Gauss linear corrected", "corrected", 0
    fs = (f"ddtSchemes\n{{\n    default         steadyState;\n}}\n\ngradSchemes\n{{\n    {grad}\n}}\n\n"
          "divSchemes\n{\n    default         Gauss linear;\n    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
          "    div(phi,k)      bounded Gauss limitedLinear 1;\n    div(phi,omega)  bounded Gauss limitedLinear 1;\n"
          "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n" + sc["div"] + "}\n\n"
          f"laplacianSchemes\n{{\n    default         {lap};\n}}\n\ninterpolationSchemes\n{{\n    default         linear;\n}}\n\n"
          f"snGradSchemes\n{{\n    default         {sn};\n}}\n\nwallDist\n{{\n    method          meshWave;\n}}")
    (case / "system" / "fvSchemes").write_text(_dict("fvSchemes", "system", fs))
    tol_p, tol_u = res * 1e-2, res * 1e-2
    fv = ("solvers\n{\n    p\n    {\n        solver          GAMG;\n"
          f"        tolerance       {tol_p:.3g};\n        relTol          0.05;\n        smoother        GaussSeidel;\n    }}\n\n"
          "    pcorr\n    {\n        solver          GAMG;\n"
          f"        tolerance       {tol_p:.3g};\n        relTol          0;\n        smoother        GaussSeidel;\n    }}\n\n"
          "    \"(U|k|omega)\"\n    {\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          f"        tolerance       {tol_u:.3g};\n        relTol          0.1;\n    }}\n" + sc["solvers"] + "}\n\n"
          f"SIMPLE\n{{\n    nNonOrthogonalCorrectors {nonorth};\n    consistent      yes;\n"
          "    pRefCell        0;\n    pRefValue       0;\n\n    residualControl\n    {\n"
          f"        p               {res:.3g};\n        U               {res:.3g};\n        \"(k|omega)\"       {res:.3g};\n    }}\n}}\n\n"
          f"relaxationFactors\n{{\n    equations\n    {{\n        U               {rU:g};\n" + sc["relax"] +
          f"        \".*\"            {rO:g};\n    }}\n}}")
    (case / "system" / "fvSolution").write_text(_dict("fvSolution", "system", fv))
    from .porous import fv_model_entries
    _write_fv_models(case, sc["fvModels"] + fv_model_entries(plan, None))
    if (case / "system" / "fvConstraints").exists():      # left by an energy run in the same folder
        (case / "system" / "fvConstraints").unlink()
    (case / "case.foam").write_text("")
    slide = [z for z in zones.values() if z.get("motion") == "sliding"]
    if slide:
        from .sliding import check as _sliding_check
        _sliding_check(mesh, g, slide[0], plan, patches)
        try:                               # the steady start runs with MRF: leaks there only spoil its start
            _mrf_wall_leaks(mesh, g, zones, plan, patches)
        except MeshError as exc:
            plan.setdefault("notes", []).append(f"steady start (MRF, before the mesh slides): {exc}")
    else:
        _mrf_wall_leaks(mesh, g, zones, plan, patches)
    plan["functions"] = funcs
    plan["laplacian"], plan["sngrad"], plan["nonorth"] = lap, sn, nonorth
    plan["grad"] = grad
    if tr and not tr["init"]:
        from .transient import write_transient
        write_transient(case, study, plan)
    return plan


def _scalar_parts(study, plan, patches, case, turb, nu, rho_src) -> dict:
    """Tracers and the mean age (scalars.py): 0/ files written, the rest returned for the system files."""
    from .scalars import case_parts
    for f in (case / "0").glob("*"):                 # fields of scalars from an earlier run in this folder
        if f.name not in ("U", "p", "p_rgh", "T", "k", "omega", "nut", "alphat") and f.is_file():
            f.unlink()
    plan.pop("scalars", None)
    return case_parts(study, plan, patches, case, turb, nu, rho_src)


def _write_fv_models(case, entries: list) -> None:
    """constant/fvModels from the given entries (scalar sources, porous zones), or none."""
    f = case / "constant" / "fvModels"
    if entries:
        f.write_text(_dict("fvModels", "constant", "\n\n".join(entries)))
    elif f.exists():
        f.unlink()


# ---- energy (OpenFOAM's fluid solver) -------------------------------------------------------------
def _thermal_wall(b: dict, T_op: float) -> dict:
    """The T boundary condition of a wall: fixed T, heat flux, power, convection, or adiabatic."""
    if "T" in b:
        return {"type": "fixedValue", "value": _uniform(float(b["T"]) + 273.15)}
    if "q" in b:
        return {"type": "externalTemperature", "q": f"constant {float(b['q']):.12g}", "value": _uniform(T_op)}
    if "power" in b:
        return {"type": "externalTemperature", "Q": f"constant {float(b['power']):.12g}", "value": _uniform(T_op)}
    if "h" in b:
        return {"type": "externalTemperature", "h": _uniform(float(b["h"])),
                "Ta": f"constant {float(b['T_inf']) + 273.15:.12g}", "value": _uniform(T_op)}
    return {"type": "zeroGradient"}


def _write_energy(study, mesh, patches, case, plan, U, kin, k0, om0, th, turb, external) -> dict:
    g = mesh.geometry()
    q = plan["quality"]
    p_op, T_op = th["p_op"], th["T_op"]
    gas = th["kind"] == "gas"
    # a closed domain (no inlet, outlet or opening) has no pressure level: use the p_rgh formulation of
    # OpenFOAM's closed-cavity cases (heRhoThermo, reference cell, strong relaxation), with zero gravity if
    # none is given; the psiThermo/consistent setting of open flows diverged there (mochi, conduction slab)
    closed = not any(p["kind"] in ("inlet", "outlet", "opening") for p in patches)
    gravity = study.get("gravity") or ([0.0, 0.0, 0.0] if closed else None)
    buoy = gravity is not None
    # closed and no gravity: nothing couples density to the flow, so take air's density as constant
    # (rhoConst). With perfectGas there, p_rgh's solver diverged in 3 iterations and enthalpy -> T failed
    # (mochi, conduction slab, 2026.10.18.5); the same case with rhoConst runs in seconds.
    rho_const = bool(closed and not study.get("gravity"))
    Ub, pb, prb, Tb, kb, wb, nb, ab = ({} for _ in range(8))
    mach_est = 0.0
    for p in patches:
        nm, kind, b = p["name"], p["kind"], p["bc"]
        if kind in ("empty", "symmetry"):
            for d in (Ub, pb, prb, Tb, kb, wb, nb, ab):
                d[nm] = {"type": kind}
            continue
        if kind == "inlet":
            if "total_pressure" in b:
                p0 = p_op + float(b["total_pressure"])
                T0 = float(b.get("total_temperature", T_op - 273.15)) + 273.15
                Ub[nm] = {"type": "pressureInletOutletVelocity", "value": _uniform([0, 0, 0])}
                # ramp (default 400 iterations, at most half the run) + transonic: the only combination that ran the
                # subsonic venturi from rest on mochi; without the ramp, or without transonic, it diverged
                sv = study.get("solve", {})
                ramp = int(sv["ramp"]) if "ramp" in sv else (min(400, int(sv.get("iterations", 3000)) // 2) if gas else 0)
                if ramp > 0:      # raise the total pressure from the outlet's over `ramp` iterations
                    p_start = p_op + min([float(pp["bc"].get("pressure", 0.0)) for pp in patches
                                          if pp["kind"] in ("outlet", "opening")] or [0.0])
                    pb[nm] = {"type": "uniformTotalPressure", "p0": f"table ((0 {p_start:.12g}) ({ramp} {p0:.12g}))",
                              "psi": "psi", "gamma": f"{th['gamma']:.12g}", "value": _uniform(p_start)}
                else:
                    pb[nm] = {"type": "totalPressure", "p0": _uniform(p0), "psi": "psi",
                              "gamma": f"{th['gamma']:.12g}", "value": _uniform(p0)}
                Tb[nm] = {"type": "totalTemperature", "T0": _uniform(T0), "gamma": f"{th['gamma']:.12g}",
                          "value": _uniform(T0)}
                outs = [float(pp["bc"].get("pressure", 0.0)) for pp in patches if pp["kind"] in ("outlet", "opening")]
                if gas and outs:
                    ratio = p0 / (p_op + min(outs))
                    g_ = th["gamma"]
                    mach_est = max(mach_est, float(np.sqrt(max(2 / (g_ - 1) * (ratio ** ((g_ - 1) / g_) - 1), 0))))
            else:
                Uf = U[nm]
                same = np.allclose(Uf, Uf[0], rtol=0, atol=1e-14 * max(1.0, float(np.abs(Uf).max())))
                Ub[nm] = {"type": "fixedValue", "value": _uniform(Uf[0]) if same else _nonuniform(Uf)}
                pb[nm] = {"type": "zeroGradient"}
                prb[nm] = {"type": "fixedFluxPressure", "value": _uniform(0)}
                Tin = float(b.get("temperature", T_op - 273.15)) + 273.15
                Tb[nm] = {"type": "fixedValue", "value": _uniform(Tin)}
                if gas:
                    c = np.sqrt(th["gamma"] * th["R"] * Tin)
                    mach_est = max(mach_est, float(np.linalg.norm(Uf, axis=1).max() / c))
                plan["inlets"][nm]["temperature_C"] = Tin - 273.15
            k, om = kin[nm]
            kb[nm] = {"type": "fixedValue", "value": _uniform(k)}
            wb[nm] = {"type": "fixedValue", "value": _uniform(om)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
            ab[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind in ("outlet", "opening"):
            pabs = p_op + float(b.get("pressure", 0.0))
            pb[nm] = {"type": "fixedValue", "value": _uniform(pabs)}
            prb[nm] = {"type": "prghPressure", "p": _uniform(pabs), "value": _uniform(0)}
            if kind == "outlet":
                Ub[nm] = {"type": "inletOutlet", "inletValue": _uniform([0, 0, 0]), "value": _uniform([0, 0, 0])}
            else:
                Ub[nm] = {"type": "pressureInletOutletVelocity", "value": _uniform([0, 0, 0])}
            Tback = float(b.get("temperature", T_op - 273.15)) + 273.15
            Tb[nm] = {"type": "inletOutlet", "inletValue": _uniform(Tback), "value": _uniform(Tback)}
            kb[nm] = {"type": "inletOutlet", "inletValue": _uniform(k0), "value": _uniform(k0)}
            wb[nm] = {"type": "inletOutlet", "inletValue": _uniform(om0), "value": _uniform(om0)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
            ab[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind == "wall":
            vw = b.get("velocity")
            if b.get("rotation") or (vw and np.linalg.norm(vw) > 0):
                f = mesh.patch_faces(nm)
                Sf = g["Sf"][f]
                n = Sf / np.linalg.norm(Sf, axis=1)[:, None]
                Ut = swirl_velocity(g["Cf"][f], b["rotation"]) if b.get("rotation") else \
                    _tangential(np.tile(np.asarray(vw, float), (len(f), 1)), n)
                same = np.allclose(Ut, Ut[0], rtol=0, atol=1e-12 * max(float(np.abs(Ut).max()), 1e-300))
                Ub[nm] = {"type": "fixedValue", "value": _uniform(Ut[0]) if same else _nonuniform(Ut)}
                plan["patches"][nm]["moving"] = True
            else:
                Ub[nm] = {"type": "noSlip"}
            pb[nm] = {"type": "zeroGradient"}
            prb[nm] = {"type": "fixedFluxPressure", "value": _uniform(0)}
            Tb[nm] = _thermal_wall(b, T_op)
            plan["patches"][nm]["thermal"] = {k: b[k] for k in ("T", "q", "power", "h", "T_inf") if k in b} or "adiabatic"
            kb[nm] = {"type": "kqRWallFunction", "value": _uniform(k0)}
            wb[nm] = {"type": "omegaWallFunction", "value": _uniform(om0)}
            nb[nm] = {"type": "nutkWallFunction", "value": _uniform(0)}
            ab[nm] = {"type": "compressible::alphatWallFunction", "Prt": "0.85", "value": _uniform(0)}
        elif kind == "slip":
            Ub[nm] = {"type": "slip"}
            pb[nm] = {"type": "zeroGradient"}
            prb[nm] = {"type": "fixedFluxPressure", "value": _uniform(0)}
            Tb[nm] = {"type": "zeroGradient"}
            kb[nm] = {"type": "zeroGradient"}
            wb[nm] = {"type": "zeroGradient"}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
            ab[nm] = {"type": "calculated", "value": _uniform(0)}
        else:
            raise MeshError(f"unknown bc kind {kind}")
    U0 = [0.0, 0.0, 0.0]
    if external and len(U) == 1:
        U0 = next(iter(U.values())).mean(axis=0).tolist()
    inl = [p for p in patches if p["kind"] == "inlet"]
    start = study.get("solve", {}).get("start", "rest")
    if not external and len(inl) == 1 and "total_pressure" in inl[0]["bc"] and start == "estimate":
        # solve.start 'estimate': start at half the isentropic inlet speed along the inlet's inward normal (on the
        # venturi it made the first pressure correction larger, not smaller: the default is 'rest' with a ramp)
        nm = inl[0]["name"]
        Sf = g["Sf"][mesh.patch_faces(nm)].sum(axis=0)
        u_est = float(plan["inlets"].get(nm, {}).get("U_estimate") or 0.0)
        if np.linalg.norm(Sf) > 0 and u_est > 0:
            U0 = (-0.5 * u_est * Sf / np.linalg.norm(Sf)).tolist()
            plan["initial_U"] = U0
    P = "[1 -1 -2 0 0 0 0]"
    (case / "0" / "U").write_text(_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", _uniform(U0), Ub))
    (case / "0" / "T").write_text(_field("volScalarField", "T", "[0 0 0 1 0 0 0]", _uniform(T_op), Tb))
    if buoy:
        (case / "0" / "p").write_text(_field("volScalarField", "p", P, _uniform(p_op),
                                             {k: ({"type": v["type"]} if v["type"] in ("empty", "symmetry")
                                                  else {"type": "calculated", "value": _uniform(p_op)})
                                              for k, v in prb.items()}))
        (case / "0" / "p_rgh").write_text(_field("volScalarField", "p_rgh", P, _uniform(0), prb))
    else:
        (case / "0" / "p").write_text(_field("volScalarField", "p", P, _uniform(p_op), pb))
        pr_file = case / "0" / "p_rgh"
        if pr_file.exists():
            pr_file.unlink()
    if turb:
        (case / "0" / "k").write_text(_field("volScalarField", "k", "[0 2 -2 0 0 0 0]", _uniform(k0), kb))
        (case / "0" / "omega").write_text(_field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", _uniform(om0), wb))
        (case / "0" / "nut").write_text(_field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", _uniform(0), nb))
        (case / "0" / "alphat").write_text(_field("volScalarField", "alphat", "[1 -1 -1 0 0 0 0]", _uniform(0), ab))
    # constant
    if gas and not rho_const:
        thermo_type, eos, eos_block = ("heRhoThermo" if buoy else "hePsiThermo"), "perfectGas", ""
        M = th["molar_mass"]
    elif gas:
        thermo_type, eos, eos_block = "heRhoThermo", "rhoConst", f"    equationOfState\n    {{\n        rho             {th['rho']:.12g};\n    }}\n"
        M = th["molar_mass"]
    else:
        thermo_type, eos, eos_block = "heRhoThermo", "rhoConst", f"    equationOfState\n    {{\n        rho             {th['rho']:.12g};\n    }}\n"
        M = 18.02
    pp = (f"thermoType\n{{\n    type            {thermo_type};\n    mixture         pureMixture;\n"
          f"    transport       const;\n    thermo          hConst;\n    equationOfState {eos};\n"
          "    specie          specie;\n    energy          sensibleEnthalpy;\n}\n\n"
          f"mixture\n{{\n    specie\n    {{\n        molWeight       {M:.12g};\n    }}\n{eos_block}"
          f"    thermodynamics\n    {{\n        Cp              {th['cp']:.12g};\n        hf              0;\n    }}\n"
          f"    transport\n    {{\n        mu              {th['mu']:.12g};\n        Pr              {th['pr']:.12g};\n    }}\n}}")
    (case / "constant" / "physicalProperties").write_text(_dict("physicalProperties", "constant", pp))
    if turb:
        mt = "simulationType  RAS;\n\nRAS\n{\n    model           kOmegaSST;\n\n    turbulence      on;\n\n    printCoeffs     on;\n}"
        (case / "constant" / "thermophysicalTransport").write_text(_dict("thermophysicalTransport", "constant",
            "RAS\n{\n    model           unityLewisEddyDiffusivity;\n    Prt             0.85;\n}"))
    else:
        mt = "simulationType  laminar;"
        tt = case / "constant" / "thermophysicalTransport"
        if tt.exists():
            tt.unlink()
    (case / "constant" / "momentumTransport").write_text(_dict("momentumTransport", "constant", mt))
    for f_ in ("g", "pRef", "MRFProperties"):
        if (case / "constant" / f_).exists():
            (case / "constant" / f_).unlink()
    if buoy:
        gx, gy, gz = (float(v) for v in gravity)
        (case / "constant" / "g").write_text(header("uniformDimensionedVectorField", "g", "constant") +
                                             f"\ndimensions      [0 1 -2 0 0 0 0];\nvalue           ({gx:.12g} {gy:.12g} {gz:.12g});\n")
        (case / "constant" / "pRef").write_text(header("uniformDimensionedScalarField", "pRef", "constant") +
                                                f"\ndimensions      {P};\nvalue           {p_op:.12g};\n")
    mesh.zones = {k: v for k, v in (mesh.zones or {}).items() if k in (plan.get("porous") or {})}   # no MRF here
    mesh.write(case)
    # system
    s = study.get("solve", {})
    iters = int(s.get("iterations", 3000))
    res = float(s.get("residual", 1e-5))
    unstructured = q["max_non_orthogonality"] > 5.0
    s_tr = s.get("transonic")
    # transonic (p as a convected variable) only near Mach 1: from rest it diverged on the subsonic venturi
    # (M 0.37 to 0.57) in two iterations on mochi; a study can still ask for it
    tp_in = any(p["kind"] == "inlet" and "total_pressure" in p["bc"] for p in patches)
    transonic = bool(gas and not buoy and (s_tr if s_tr is not None else (mach_est > 0.7 or tp_in)))
    relax = s.get("relax", {})
    if rho_const:          # no buoyancy coupling: enthalpy converges fast without strong relaxation
        rU, rO = float(relax.get("U", 0.7)), float(relax.get("other", 0.9))
    elif buoy:
        rU, rO = float(relax.get("U", 0.3)), float(relax.get("other", 0.3))
    else:
        fast = gas and mach_est > 0.1          # compressible: start more gently
        rU = float(relax.get("U", 0.7 if (unstructured or transonic or fast) else 0.9))
        rO = float(relax.get("other", 0.7 if (unstructured or transonic or fast) else 0.9))
    rP = float(relax.get("p", 0.3 if (gas and mach_est > 0.1 and not buoy and not transonic) else 1.0))
    # no heat put in anywhere and low Mach: temperature varies only by u^2/cp (kinetic energy, ~1e-4 K), so
    # h's normalised residual is round-off and stalls (pipe_air_low_mach on mochi: 1.5e-4 after 3000
    # iterations). OpenFOAM then stops on U and p alone, and 'converged' ignores h (runner.zero_fields).
    T_C = T_op - 273.15
    heat_in = any(any(k in p["bc"] for k in ("T", "q", "power", "h")) for p in patches if p["kind"] == "wall") or any(
        abs(float(p["bc"].get(k, T_C)) - T_C) > 1e-9 for p in patches for k in ("temperature", "total_temperature")
        if p["kind"] in ("inlet", "outlet", "opening")) or any("total_pressure" in p["bc"] for p in patches)
    isothermal = not heat_in and mach_est < 0.1
    plan["schemes"] = "unstructured" if unstructured else "orthogonal"
    plan.update({"iterations": iters, "residual": res, "relax": {"U": rU, "other": rO, **({"p": rP} if rP < 1 else {})},
                 "fixed_pressure": any(pp["kind"] in ("outlet", "opening") for pp in patches),
                 "energy": {**{k: v for k, v in th.items()}, "buoyancy": buoy, "gravity": gravity, "closed": closed, "isothermal": isothermal,
                            "density": "constant (closed, no gravity)" if rho_const else ("constant" if not gas else "perfect gas"),
                            "transonic": transonic, "mach_estimate": mach_est,
                            "solver": "fluid", "thermo": thermo_type}})
    walls = [p["name"] for p in patches if p["type"] == "wall"]
    lib = "        libs            (\"libfieldFunctionObjects.so\");\n"
    funcs = []
    if walls:
        funcs.append("    wallShearStress\n    {\n        type            wallShearStress;\n" + lib +
                     f"        patches         ({' '.join(walls)});\n        writeControl    writeTime;\n    }}")
        funcs.append("    wallHeatFlux\n    {\n        type            wallHeatFlux;\n" + lib +
                     f"        patches         ({' '.join(walls)});\n        writeControl    writeTime;\n    }}")
        if turb:
            funcs.append("    yPlus\n    {\n        type            yPlus;\n" + lib + "        writeControl    writeTime;\n    }")
    # no MachNo function object: our results compute Mach from U and T, and on the venturi it was MachNo's sqrt
    # that stopped the run (SIGFPE) on the first iterations' transient pressures (mochi, 2026.10.18.11)
    sc = _scalar_parts(study, plan, patches, case, turb, th["mu"] / th["rho"], th["rho"])
    funcs += sc["funcs"]
    cd = (f"solver          fluid;\n\nstartFrom       startTime;\n\nstartTime       0;\n\n"
          f"stopAt          endTime;\n\nendTime         {iters};\n\ndeltaT          1;\n\nwriteControl    timeStep;\n\n"
          f"writeInterval   {iters};\n\npurgeWrite      0;\n\nwriteFormat     ascii;\n\nwritePrecision  12;\n\n"
          "writeCompression off;\n\ntimeFormat      general;\n\ntimePrecision   6;\n\nrunTimeModifiable false;\n\n"
          "functions\n{\n" + "\n".join(funcs) + "\n}")
    (case / "system" / "controlDict").write_text(_dict("controlDict", "system", cd))
    if unstructured:
        grad = ("default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n"
                "    grad(h)         cellLimited Gauss linear 1;\n    grad(k)         cellLimited Gauss linear 1;\n"
                "    grad(omega)     cellLimited Gauss linear 1;")
        lap, sn, nonorth, _ = nonortho_schemes(q)
    else:
        grad, lap, sn, nonorth = "default         Gauss linear;", "Gauss linear corrected", "corrected", 0
    fs = (f"ddtSchemes\n{{\n    default         steadyState;\n}}\n\ngradSchemes\n{{\n    {grad}\n}}\n\n"
          "divSchemes\n{\n    default         Gauss linear;\n    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
          "    div(phi,h)      bounded Gauss limitedLinear 1;\n    div(phi,K)      bounded Gauss limitedLinear 1;\n"
          "    div(phid,p)     Gauss limitedLinear 1;\n"
          "    div(phi,k)      bounded Gauss limitedLinear 1;\n    div(phi,omega)  bounded Gauss limitedLinear 1;\n"
          "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n" + sc["div"] + "}\n\n"
          f"laplacianSchemes\n{{\n    default         {lap};\n}}\n\ninterpolationSchemes\n{{\n    default         linear;\n}}\n\n"
          f"snGradSchemes\n{{\n    default         {sn};\n}}\n\nwallDist\n{{\n    method          meshWave;\n}}")
    (case / "system" / "fvSchemes").write_text(_dict("fvSchemes", "system", fs))
    pname = "p_rgh" if buoy else "p"
    tol = res * 1e-2
    # maxIter: the pressure here is absolute (1e5 Pa) and GAMG stalls at a round-off floor just above relTol when
    # the pressure differences are tiny; uncapped it took 1000 sweeps every iteration (heated channel on mochi:
    # 0.26 s per iteration for 4000 cells, 17 min in all)
    solvers = (f"solvers\n{{\n    {pname}\n    {{\n        solver          GAMG;\n"
               f"        tolerance       {tol:.3g};\n        relTol          0.01;\n        smoother        GaussSeidel;\n"
               "        maxIter         50;\n    }\n\n"
               "    \"(U|h|k|omega)\"\n    {\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
               f"        tolerance       {tol:.3g};\n        relTol          0.1;\n        maxIter         100;\n    }}\n"
               + sc["solvers"] + "}\n\n")
    rc = (f"    residualControl\n    {{\n        {pname:<15} {res:.3g};\n        U               {res:.3g};\n"
          + ("" if isothermal else f"        h               {res:.3g};\n")
          + f"        \"(k|omega)\"       {res:.3g};\n    }}\n")
    if buoy:
        pim = (f"PIMPLE\n{{\n    momentumPredictor no;\n    nNonOrthogonalCorrectors {nonorth};\n"
               f"    pRefCell        0;\n    pRefValue       0;\n\n{rc}}}\n\n"
               "relaxationFactors\n{\n    fields\n    {\n        rho             1;\n        p_rgh           0.7;\n    }\n"
               f"    equations\n    {{\n        U               {rU:g};\n        h               {rO:g};\n" + sc["relax"] +
               "        \"(k|omega)\"       0.7;\n    }\n}")
    else:
        pim = (f"PIMPLE\n{{\n    nNonOrthogonalCorrectors {nonorth};\n    transonic       {'yes' if transonic else 'no'};\n"
               f"    consistent      {'no' if s.get('consistent') is False else 'yes'};\n    pRefCell        0;\n"
               f"    pRefValue       {p_op:.12g};\n\n{rc}}}\n\n"
               f"relaxationFactors\n{{\n    fields\n    {{\n        p               {rP:g};\n    }}\n"
               f"    equations\n    {{\n        p               1;\n        U               {rU:g};\n"
               f"        h               {rO:g};\n" + sc["relax"] + f"        \".*\"            {rO:g};\n    }}\n}}")
    (case / "system" / "fvSolution").write_text(_dict("fvSolution", "system", solvers + pim))
    from .porous import fv_model_entries
    _write_fv_models(case, sc["fvModels"] + fv_model_entries(plan, th))
    # pressure limits: OpenFOAM clips p to [0.5, 2] x its boundary range, so a bad start cannot make it negative
    # (OpenFOAM 14 wants them here, not as pMinFactor/pMaxFactor in PIMPLE: 2026.10.18.9 failed on mochi)
    fc = case / "system" / "fvConstraints"
    if buoy:
        if fc.exists():
            fc.unlink()
    else:
        fc.write_text(_dict("fvConstraints", "system",
                            "limitp\n{\n    type            limitPressure;\n    minFactor       0.5;\n    maxFactor       2;\n}"))
    (case / "case.foam").write_text("")
    return plan
