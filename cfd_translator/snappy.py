"""snappy.py — 3D meshes with snappyHexMesh: hexahedra refined towards the part, cut and snapped to its
surface and sharp edges, with prism layers on its walls.
Version: 2026.10.22.8

    mesh_snappy(study, folder) -> PolyMesh

1. Gmsh triangulates the part's surface (each CAD face keeps its tag). The bcs are chosen on that surface
   (and on the domain box's sides), exactly as on a Gmsh mesh, so the selectors work the same.
2. The surface goes to OpenFOAM as an OBJ with one region per bc; the domain box is the background
   (blockMesh) with one patch per bc. Part faces lying on a domain side (a half car's cut at y = 0, a tyre's
   flattened contact patch) are left out and the faces around them extended past the side, so the cut
   meshes cleanly. surfaceFeatures finds the sharp edges, snappyHexMesh refines (surface levels, edges,
   boxes / cylinders / spheres / distance from the part, narrow gaps), snaps and adds layers. Porous and
   rotating zones become cellZones whose faces are snapped to their box, cylinder or sphere.
3. The mesh is read back; every boundary face gets the tag of the CAD face it lies on (the nearest surface
   triangle of its bc), so per-face results (feat_loads) work as on a Gmsh mesh. checkMesh judges it; a mesh
   that fails is made again once with stricter quality settings.

Lengths: mm in the study, metres in the OpenFOAM files. Refinement level L means cells of size / 2^L.
"""
from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

import numpy as np

from .polymesh import BOX_SIDES, SCALE, MeshError, from_foam

PART = "part"
BOX_TAG0 = 1_000_000               # tags of the domain box's sides in the surface mesh
BOX_FACES = {"xmin": (0, 4, 7, 3), "xmax": (2, 6, 5, 1), "ymin": (1, 5, 4, 0),
             "ymax": (3, 7, 6, 2), "zmin": (0, 3, 2, 1), "zmax": (4, 5, 6, 7)}


# ---- sizes and levels -----------------------------------------------------------------------------
def level_for(h0: float, size: float) -> int:
    """The refinement level whose cells (h0 / 2^L) are no larger than size (7 % slack: background cells come out
    a little larger than mesh.size when the domain is not a whole number of them, and that must not cost a level)."""
    return max(0, int(math.ceil(math.log2(h0 / float(size)) - 0.1)))


def _nu(study: dict) -> float:
    from .format import fluid_props, thermo_props
    if study.get("energy"):
        th = thermo_props(study)
        return float(th["mu"]) / float(th["rho"])
    f = fluid_props(study["fluid"])
    if "rho" in f:
        return float(f["mu"]) / float(f["rho"])
    th = thermo_props(study)
    return float(th["mu"]) / float(th["rho"])


def _velocity(study: dict) -> float | None:
    ref = study.get("reference") or {}
    if ref.get("velocity"):
        return float(ref["velocity"])
    best = None
    for b in study.get("bcs", []):
        v = b.get("velocity") if b.get("kind") == "inlet" else None
        if isinstance(v, (int, float)):
            best = max(best or 0.0, abs(float(v)))
        elif isinstance(v, list) and len(v) == 3:
            best = max(best or 0.0, float(np.linalg.norm(v)))
    return best


def layer_plan(study: dict, surface_cell_mm: float | None, extent_mm: float | None) -> dict | None:
    """Prism layers: first height (given, or from a target y+ with a flat-plate friction estimate), growth,
    number (given, or the most whose whole stack fits in one of the finest surface cells: on the Ahmed body
    13 layers, 57 mm, on 10-20 mm cells collapsed to 1.5 on mochi), total; and the y+ the first cell should see
    (at its centre: half the first height, as results.y_plus measures it). surface_cell_mm: the finest."""
    L = (study.get("mesh") or {}).get("layers")
    if not L:
        return None
    nu = _nu(study)
    U = _velocity(study)
    Lref = (study.get("reference") or {}).get("length") or extent_mm
    u_tau = None
    out = {}
    if U and Lref:
        Re = U * float(Lref) * SCALE / nu
        cf = 0.0576 * Re ** -0.2 if Re > 5e5 else 0.664 / math.sqrt(max(Re, 1.0))
        u_tau = U * math.sqrt(cf / 2)
        out.update({"basis": {"velocity": U, "length_mm": float(Lref), "Re": float(f"{Re:.4g}"),
                              "cf_flat_plate": float(f"{cf:.4g}"), "u_tau": float(f"{u_tau:.4g}")}})
    if "first" in L:
        first = float(L["first"])
    elif "y_plus" in L:
        if not u_tau:
            raise MeshError("mesh.layers.y_plus needs a velocity (reference.velocity or an inlet velocity) and a "
                            "length (reference.length, else the part's size)")
        first = 2 * float(L["y_plus"]) * nu / u_tau / SCALE
        out["y_plus_target"] = float(L["y_plus"])
    else:
        raise MeshError("mesh.layers: give first (mm) or y_plus")
    r = float(L.get("growth", 1.2))
    def stack(k):
        return first * k if abs(r - 1) < 1e-12 else first * (r ** k - 1) / (r - 1)
    if "n" in L:
        n = int(L["n"])
    elif surface_cell_mm:
        n = 1
        while n < 30 and stack(n + 1) <= surface_cell_mm:
            n += 1
    else:
        n = 5
    total = stack(n)
    out.update({"n": n, "first_mm": float(f"{first:.4g}"), "growth": r, "total_mm": float(f"{total:.4g}"),
                "last_mm": float(f"{first * r ** (n - 1):.4g}")})
    if u_tau:
        out["y_plus_first_cell_estimate"] = float(f"{u_tau * first * SCALE / 2 / nu:.3g}")
    if surface_cell_mm:
        out["last_to_surface_cell"] = float(f"{first * r ** (n - 1) / surface_cell_mm:.3g}")
        out["stack_to_surface_cell"] = float(f"{total / surface_cell_mm:.3g}")
    return out


def settings(study: dict, external: bool, box: list, part_bb: tuple | None = None) -> dict:
    """Everything snappyHexMesh is told, in mm and levels (no OpenFOAM, no Gmsh): background cells, surface
    and edge levels, refinement regions, gap increment, layers, quality."""
    m = study.get("mesh") or {}
    h = float(m["size"])
    lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
    if external:
        n = np.maximum(1, np.round((hi - lo) / h)).astype(int)
    else:                                     # internal: the part's box plus a cell and a half, cells exactly h
        n = np.ceil((hi - lo) / h + 3).astype(int)
        c = 0.5 * (lo + hi)
        lo, hi = c - n * h / 2, c + n * h / 2
    h0 = float(((hi - lo) / n).max())
    sf = m.get("surface", h / 8)
    smin, smax = (float(sf[0]), float(sf[1])) if isinstance(sf, list) else (float(sf), float(sf))
    Ls = (level_for(h0, smax), level_for(h0, smin))            # (min level: coarse, max level: fine)
    feat = m.get("features", smin)
    Lf = level_for(h0, float(feat)) if feat else None
    gap = int(m.get("gap", 1))
    regions, dist = [], []
    for k, r in enumerate(m.get("refine") or []):
        L = level_for(h0, float(r["size"]))
        e = {"level": L, "cell_mm": h0 / 2 ** L}
        if "distance" in r:
            dist.append((float(r["distance"]), L))
            continue
        if "box" in r:
            e.update(shape="box", box=list(map(float, r["box"])))
        elif "cylinder" in r:
            c = r["cylinder"]
            a = np.asarray(c["axis"], float)
            a /= np.linalg.norm(a)
            s0, s1 = c.get("span", [-1e9, 1e9])
            ctr = np.asarray(c.get("center", [0, 0, 0]), float)
            e.update(shape="cylinder", p1=(ctr + max(s0, -1e7) * a).tolist(), p2=(ctr + min(s1, 1e7) * a).tolist(),
                     radius=float(c["radius"]))
        else:
            e.update(shape="sphere", center=list(map(float, r["at"])), radius=float(r["r"]))
        e["name"] = f"refine{k}"
        regions.append(e)
    dist.sort()
    dist = [(d, max(L for _, L in dist[i:])) for i, (d, _) in enumerate(dist)]     # nearer: at least as fine
    ext = float(np.max(np.asarray(part_bb[1]) - np.asarray(part_bb[0]))) if part_bb else None
    lay = layer_plan(study, h0 / 2 ** Ls[1], ext)
    q = m.get("quality") or {}
    quality = {"maxNonOrtho": float(q.get("max_non_ortho", 65)), "maxBoundarySkewness": 20.0,
               "maxInternalSkewness": float(q.get("max_skewness", 4)), "maxConcave": 80.0, "minVol": -1e30,
               "minTetQuality": 1e-15, "minTwist": 0.02, "minDeterminant": 0.001, "minFaceWeight": 0.05,
               "minVolRatio": 0.01, "nSmoothScale": 4, "errorReduction": 0.75}
    return {"h0": h0, "cells": n.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "surface": Ls,
            "surface_mm": [h0 / 2 ** Ls[0], h0 / 2 ** Ls[1]], "features": Lf, "gap": gap, "regions": regions,
            "distance": dist, "layers": lay, "quality": quality, "retry": bool(q.get("retry", True)),
            "max_cells": int(float(m.get("max_cells", 2e7))),
            "feature_angle": float(((m.get("layers") or {}).get("feature_angle", 130)))}


def stricter(qs: dict) -> dict:
    """Quality settings for the second attempt when the first mesh fails checkMesh."""
    q = dict(qs)
    q.update(maxNonOrtho=max(45.0, qs["maxNonOrtho"] - 10), maxInternalSkewness=min(3.0, qs["maxInternalSkewness"]),
             maxBoundarySkewness=15.0, maxConcave=60.0, minDeterminant=0.01, minFaceWeight=0.1,
             nSmoothScale=6, errorReduction=0.6)
    return q


# ---- the part's surface (Gmsh) ------------------------------------------------------------------------
def surface_premesh(study: dict, workdir: Path, h_max: float, h_min: float, box: list | None):
    """Triangles of the part's surface, oriented out of the solid: (points mm, [(tag, tris)], inside point
    for internal flow or None, part bounding box). Each CAD face keeps its Gmsh tag."""
    try:
        import gmsh
    except ImportError as exc:
        raise MeshError("snappy meshes need Gmsh for the part's surface (pip install gmsh)") from exc
    from .mesher import _gmsh_init, _part, _surface_mesh
    _gmsh_init(gmsh)
    try:
        gmsh.model.add("surface")
        part = _part(gmsh, study, workdir)
        vol = part[0][1]
        bb = np.array(gmsh.model.getBoundingBox(3, vol))
        diag = float(np.linalg.norm(bb[3:] - bb[:3]))
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 48)
        gmsh.model.mesh.generate(2)
        pts, _, surf = _surface_mesh(gmsh)

        def inside(x) -> bool:
            return bool(gmsh.model.isInside(3, vol, [float(v) for v in x]))
        out = []
        for tag, tri in surf:
            if tri.shape[1] != 3:
                raise MeshError("unexpected quadrilaterals in the surface mesh")
            P = pts[tri]
            nv = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            a = np.linalg.norm(nv, axis=1)
            k = int(np.argmax(a))
            c, nn = P[k].mean(axis=0), nv[k] / a[k]
            eps = max(1e-5 * diag, 0.05 * math.sqrt(a[k]))
            ip, im = inside(c + eps * nn), inside(c - eps * nn)
            if ip and not im:
                tri = tri[:, ::-1]
            elif ip == im:
                # thin or ambiguous: the orientation OpenCASCADE gives the face in the solid's boundary
                sgn = {abs(s): (1 if s > 0 else -1) for _, s in gmsh.model.getBoundary([(3, vol)], oriented=True)}
                if sgn.get(tag, 1) < 0:
                    tri = tri[:, ::-1]
            out.append((int(tag), np.ascontiguousarray(tri)))
        ipt = None
        if box is None:                                    # internal flow: a point well inside the part
            h = h_min
            cands = [np.asarray(gmsh.model.occ.getCenterOfMass(3, vol))]
            g = np.linspace(0.13, 0.87, 7)
            cands += [bb[:3] + (bb[3:] - bb[:3]) * np.array([x, y, z]) for x in g for y in g for z in g]
            for c in cands:
                c = c + h * np.array([0.0123, 0.0217, 0.0311])
                if inside(c) and all(inside(c + d) for d in h * np.vstack([np.eye(3), -np.eye(3)])):
                    ipt = c
                    break
            if ipt is None:
                raise MeshError("found no point inside the part for snappyHexMesh (is it thinner than the surface cells?)")
        else:
            lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
            for corner in ((0.0137, 0.0219, 0.0313), (0.9863, 0.0219, 0.0313), (0.0137, 0.9781, 0.9687),
                           (0.9863, 0.9781, 0.9687), (0.5137, 0.9781, 0.9687)):
                c = lo + (hi - lo) * np.array(corner)
                if not inside(c):
                    ipt = c
                    break
            if ipt is None:
                raise MeshError("found no point of the domain box outside the part")
        return pts, out, ipt, (bb[:3], bb[3:])
    finally:
        gmsh.finalize()


def cut_at_box(pts: np.ndarray, surf: list, box: list, h: float):
    """Part faces lying on a domain side (a half model's cut, a flattened contact patch) are not fluid
    boundaries: drop them, and extend the faces around them past the side by about a surface cell, so that
    snappyHexMesh cuts the part cleanly at the side. Returns (points, surface, {tag: side}, notes)."""
    lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
    tol = 1e-6 * float(np.linalg.norm(hi - lo)) + 1e-6
    planes = [(ax, lo[ax] if s == 0 else hi[ax], -1.0 if s == 0 else 1.0, BOX_SIDES[2 * ax + s])
              for ax in range(3) for s in (0, 1)]
    dropped, notes = {}, []
    keep = []
    for tag, tri in surf:
        P = pts[tri.ravel()]
        hit = [pl for pl in planes if np.all(np.abs(P[:, pl[0]] - pl[1]) < tol)]
        if hit:
            dropped[tag] = hit[0][3]
        else:
            keep.append((tag, tri))
    if not dropped:
        return pts, surf, {}, notes
    new_pts = [pts]
    npt = len(pts)
    out = []
    allE = np.concatenate([t[:, [0, 1]] for _, t in keep] + [t[:, [1, 2]] for _, t in keep] + [t[:, [2, 0]] for _, t in keep])
    u, cnt = np.unique(np.sort(allE, axis=1), axis=0, return_counts=True)
    single = {tuple(r) for r in u[cnt == 1].tolist()}
    for tag, tri in keep:
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        extra = []
        for ax, val, sgn, side in planes:
            if side not in dropped.values():
                continue
            on = np.abs(pts[e, ax] - val) < tol
            ee = e[on.all(axis=1)]
            if not len(ee):
                continue
            # only edges used once (the rim of the dropped face)
            ee = np.array([r for r in ee.tolist() if tuple(sorted(r)) in single], np.int64).reshape(-1, 2)
            if not len(ee):
                continue
            nodes = np.unique(ee)
            shift = np.zeros(3)
            shift[ax] = sgn * h
            moved = pts[nodes] + shift
            idx = {int(v): npt + i for i, v in enumerate(nodes)}
            new_pts.append(moved)
            npt += len(nodes)
            for a, b in ee:
                a2, b2 = idx[int(a)], idx[int(b)]
                extra += [(b, a, a2), (b, a2, b2)]
        out.append((tag, np.concatenate([tri, np.array(extra, np.int64).reshape(-1, 3)]) if extra else tri))
    notes.append(f"part faces on the domain sides left out (the part is cut there): "
                 f"{', '.join(f'{t} on {s}' for t, s in sorted(dropped.items()))}")
    return np.concatenate(new_pts), out, dropped, notes


def touching(pts: np.ndarray, surf: list, box: list, dropped: dict) -> list[str]:
    """Domain sides the part touches only at an edge or a point (tangent): snappyHexMesh would pinch cells
    there."""
    lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
    tol = 1e-6 * float(np.linalg.norm(hi - lo)) + 1e-6
    P = pts[np.unique(np.concatenate([t.ravel() for _, t in surf]))]
    out = []
    for ax in range(3):
        for s, val in ((0, lo[ax]), (1, hi[ax])):
            side = BOX_SIDES[2 * ax + s]
            if side in dropped.values():
                continue
            d = (P[:, ax] - val) * (1 if s == 0 else -1)
            if d.min() < -tol:
                continue                     # the part crosses the side: snappyHexMesh cuts it
            if d.min() < tol:
                out.append(side)
    return out


def box_triangles(box: list, start: int):
    """The domain box's sides as two triangles each, normals out of the box: (points, [(tag, tris)], info)."""
    x0, y0, z0, x1, y1, z1 = (float(v) for v in box)
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    surf, info = [], {}
    for i, side in enumerate(BOX_SIDES):
        a, b, c, d = BOX_FACES[side]
        tag = BOX_TAG0 + i
        surf.append((tag, np.array([[a, b, c], [a, c, d]], np.int64) + start))
        info[tag] = {"on": "box", "box": side}
    return V, surf, info


def narrow_gaps(pts: np.ndarray, surf: list, external: bool, width_mm: float, limit: int = 5) -> list[dict]:
    """Places where two facing walls of the part are closer than width_mm (fluid between them): what
    snappyHexMesh may leave with too few (or pinched) cells."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return []
    tri = np.concatenate([t for _, t in surf])
    P = pts[tri]
    c = P.mean(axis=1)
    nv = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    a = np.linalg.norm(nv, axis=1)
    ok = a > 0
    c, nv, a = c[ok], nv[ok] / a[ok, None], a[ok]
    if not external:
        nv = -nv                                       # into the fluid
    if len(c) > 400_000:
        sel = np.random.default_rng(0).choice(len(c), 400_000, replace=False)
        c, nv, a = c[sel], nv[sel], a[sel]
    edge = float(np.sqrt(2 * np.median(a)))
    r = max(width_mm, 1.5 * edge)
    pairs = cKDTree(c).query_pairs(r, output_type="ndarray")
    if not len(pairs):
        return []
    i, j = pairs[:, 0], pairs[:, 1]
    facing = np.einsum("ij,ij->i", nv[i], nv[j]) < -0.7
    d = c[j] - c[i]
    along = np.einsum("ij,ij->i", d, nv[i])
    lat = np.linalg.norm(d - along[:, None] * nv[i], axis=1)
    m = facing & (along > 1e-9) & (along < width_mm) & (lat < np.maximum(along, edge))
    if not m.any():
        return []
    i, j, w = i[m], j[m], along[m]
    order = np.argsort(w)
    found = []
    for k in order:
        x = 0.5 * (c[i[k]] + c[j[k]])
        if all(np.linalg.norm(x - np.asarray(f["at"])) > 5 * width_mm for f in found):
            found.append({"at": np.round(x, 1).tolist(), "width_mm": float(f"{w[k]:.3g}")})
            if len(found) >= limit:
                break
    return found


# ---- OpenFOAM files ---------------------------------------------------------------------------------
def _hdr(obj: str, cls: str = "dictionary", location: str = "system") -> str:
    return ("FoamFile\n{\n    format      ascii;\n    class       " + cls + ";\n    location    \"" + location +
            "\";\n    object      " + obj + ";\n}\n\n")


def _v(x) -> str:
    return "(" + " ".join(f"{float(v) * SCALE:.10g}" for v in x) + ")"


def write_obj(path: Path, pts: np.ndarray, regions: list) -> None:
    """regions: [(name, tris)]: one OBJ group per region (snappyHexMesh's surface regions); metres."""
    used = np.unique(np.concatenate([t.ravel() for _, t in regions]))
    remap = -np.ones(len(pts), np.int64)
    remap[used] = np.arange(len(used))
    with open(path, "w") as f:
        f.write("# cfd-translator: the part's surface, one group per boundary condition\n")
        f.write("".join(f"v {x * SCALE:.10g} {y * SCALE:.10g} {z * SCALE:.10g}\n" for x, y, z in pts[used]))
        for name, tri in regions:
            f.write(f"g {name}\n")
            t = remap[tri] + 1
            f.write("".join(f"f {a} {b} {c}\n" for a, b, c in t.tolist()))


def block_mesh_dict(st: dict, box_patches: list, internal: bool) -> str:
    """box_patches: [(patch name, type, [sides])]."""
    x0, y0, z0 = st["lo"]
    x1, y1, z1 = st["hi"]
    V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    nx, ny, nz = st["cells"]
    b = [_hdr("blockMeshDict"), "vertices\n(\n" + "".join(f"    {_v(v)}\n" for v in V) + ");\n\n",
         f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)\n);\n\n", "boundary\n(\n"]
    for name, typ, sides in box_patches:
        fs = " ".join("(" + " ".join(map(str, BOX_FACES[s])) + ")" for s in sides)
        b.append(f"    {name}\n    {{\n        type {typ};\n        faces ({fs});\n    }}\n")
    b.append(");\n\n")
    if internal:
        b.append("defaultPatch\n{\n    name background;\n    type patch;\n}\n")
    return "".join(b)


def _level_pair(a: int, b: int) -> str:
    return f"({a} {b})"


def snappy_dict(st: dict, regions: list, inside_point, layer_patches: list, zones: list, quality: dict,
                layers_on: bool = True) -> str:
    """regions: [(OBJ group, patch name, patch type)]; zones: [{name, shape, ..., level}]."""
    Lmin, Lmax = st["surface"]
    geo = [f"    {PART}\n    {{\n        type triSurface;\n        file \"{PART}.obj\";\n        regions\n        {{\n"]
    geo += [f"            {g} {{ name {n}; }}\n" for g, n, _ in regions]
    geo.append("        }\n    }\n")

    def shape(name, e):
        if e["shape"] == "box":
            bx = e["box"]
            return f"    {name}\n    {{\n        type box;\n        min {_v(bx[:3])};\n        max {_v(bx[3:])};\n    }}\n"
        if e["shape"] == "cylinder":
            return (f"    {name}\n    {{\n        type cylinder;\n        point1 {_v(e['p1'])};\n        point2 {_v(e['p2'])};\n"
                    f"        radius {e['radius'] * SCALE:.10g};\n    }}\n")
        return (f"    {name}\n    {{\n        type sphere;\n        centre {_v(e['center'])};\n"
                f"        radius {e['radius'] * SCALE:.10g};\n    }}\n")
    for e in st["regions"]:
        geo.append(shape(e["name"], e))
    for z in zones:
        geo.append(shape("zone_" + z["name"], z))
    rs = [f"        {PART}\n        {{\n            level {_level_pair(Lmin, Lmax)};\n"
          f"            gapLevelIncrement {st['gap']};\n            regions\n            {{\n"]
    rs += [f"                {g} {{ level {_level_pair(Lmin, Lmax)}; patchInfo {{ type {t}; }} }}\n" for g, _, t in regions]
    rs.append("            }\n        }\n")
    for z in zones:
        rs.append(f"        zone_{z['name']}\n        {{\n            level ({z['level']} {z['level']});\n"
                  f"            faceZone {z['name']};\n            cellZone {z['name']};\n            mode inside;\n        }}\n")
    rr = [f"        {e['name']} {{ mode inside; level {e['level']}; }}\n" for e in st["regions"]]
    if st["distance"]:
        lv = " ".join(f"({d * SCALE:.10g} {L})" for d, L in st["distance"])
        rr.append(f"        {PART} {{ mode distance; levels ({lv}); }}\n")
    feats = (f"( {{ file \"{PART}.extendedFeatureEdgeMesh\"; level {st['features']}; }} )"
             if st["features"] is not None else "()")
    lay = st["layers"]
    add = bool(lay and layer_patches and layers_on)
    q = quality
    out = [_hdr("snappyHexMeshDict"),
           f"castellatedMesh true;\nsnap            true;\naddLayers       {'true' if add else 'false'};\n\n",
           "geometry\n{\n", *geo, "}\n\n",
           "castellatedMeshControls\n{\n    maxLocalCells 2000000;\n"
           f"    maxGlobalCells {st['max_cells']};\n    minRefinementCells 10;\n    maxLoadUnbalance 0.10;\n"
           f"    nCellsBetweenLevels 3;\n    features {feats};\n    refinementSurfaces\n    {{\n", *rs, "    }\n",
           "    resolveFeatureAngle 30;\n    refinementRegions\n    {\n", *rr, "    }\n",
           f"    insidePoint {_v(inside_point)};\n    allowFreeStandingZoneFaces false;\n}}\n\n",
           "snapControls\n{\n    nSmoothPatch 3;\n    tolerance 2.0;\n    nSolveIter 50;\n    nRelaxIter 5;\n"
           "    nFeatureSnapIter 10;\n    explicitFeatureSnap true;\n    implicitFeatureSnap false;\n"
           "    multiRegionFeatureSnap true;\n}\n\n"]
    if add:
        names = "|".join(layer_patches)
        out.append("addLayersControls\n{\n    relativeSizes false;\n    layers\n    {\n"
                   f"        \"({names})\" {{ nSurfaceLayers {lay['n']}; }}\n    }}\n"
                   f"    firstLayerThickness {lay['first_mm'] * SCALE:.6g};\n    expansionRatio {lay['growth']:.6g};\n"
                   f"    minThickness {lay['first_mm'] * SCALE:.6g};\n    nGrow 0;\n    featureAngle {st['feature_angle']:g};\n"
                   "    slipFeatureAngle 30;\n    nRelaxIter 5;\n    nSmoothSurfaceNormals 1;\n    nSmoothNormals 3;\n"
                   "    nSmoothThickness 10;\n    maxFaceThicknessRatio 0.5;\n    maxThicknessToMedialRatio 0.3;\n"
                   "    minMedialAxisAngle 90;\n    nBufferCellsNoExtrude 0;\n    nLayerIter 50;\n    nRelaxedIter 20;\n}\n\n")
    out.append("meshQualityControls\n{\n" + "".join(f"    {k} {v:g};\n" for k, v in q.items()) +
               f"    minFaceFlatness 0.5;\n    relaxed\n    {{\n        maxNonOrtho {max(q['maxNonOrtho'], 75):g};\n    }}\n}}\n\n"
               "mergeTolerance 1e-6;\n")
    return "".join(out)


def split3(procs: int, lengths) -> tuple:
    """Ranks along x, y, z (product procs) giving the most cube-like pieces of a box of these lengths."""
    L = [float(v) for v in lengths]
    best, bcost = (procs, 1, 1), float("inf")
    for a in range(1, procs + 1):
        if procs % a:
            continue
        for b in range(1, procs // a + 1):
            if (procs // a) % b:
                continue
            c = procs // a // b
            piece = [L[0] / a, L[1] / b, L[2] / c]
            cost = max(piece) / max(min(piece), 1e-30)
            if cost < bcost - 1e-12:
                best, bcost = (a, b, c), cost
    return best


def decompose_dict(procs: int, lengths) -> str:
    """snappyHexMesh in parallel also needs a distributor (to rebalance while refining; OpenFOAM 14 stops
    without one). hierarchical is built in (ptscotch needs the third-party library)."""
    n = split3(procs, lengths)
    return (_hdr("decomposeParDict") + f"numberOfSubdomains {procs};\n\ndecomposer      scotch;\n"
            f"distributor     hierarchical;\n\nhierarchical\n{{\n    n               ({n[0]} {n[1]} {n[2]});\n"
            "    order           xyz;\n}\n")


def control_dict() -> str:
    return (_hdr("controlDict") + "startFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\n"
            "writeControl timeStep;\nwriteInterval 1;\npurgeWrite 0;\nwriteFormat binary;\nwritePrecision 10;\n"
            "writeCompression off;\ntimeFormat general;\ntimePrecision 6;\nrunTimeModifiable false;\n")


# ---- logs -------------------------------------------------------------------------------------------
_STAGE = re.compile(r"^(Refined|Snapped|Layer) mesh : cells:(\d+)", re.M)


def parse_log(text: str) -> dict:
    """Cells after each stage, the final layer table, illegal faces, cells refined for narrow gaps, time."""
    out = {"cells": {m.group(1).lower(): int(m.group(2)) for m in _STAGE.finditer(text)}}
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if re.match(r"^\s*patch\s+faces\s+layers\s+overall thickness", l):
            rows = []
            for r in lines[i + 3:]:
                f = r.split()
                if len(f) != 5:
                    break
                try:
                    rows.append({"patch": f[0], "faces": int(f[1]), "layers": float(f[2]),
                                 "thickness_mm": float(f[3]) / SCALE, "thickness_pct": float(f[4])})
                except ValueError:
                    break
            out["layers"] = rows
    m = re.search(r"Finished meshing with (\d+) illegal faces", text)
    out["illegal_faces"] = int(m.group(1)) if m else 0
    g = re.findall(r"close opposite surfaces\s*:\s*(\d+) cells", text)
    if g:
        out["gap_cells"] = sum(int(x) for x in g)
    m = re.search(r"Finished meshing in = ([\d.eE+-]+) s", text)
    if m:
        out["seconds"] = float(m.group(1))
    out["finished"] = "Finished meshing" in text
    k = text.find("FOAM FATAL")
    if k >= 0:
        out["fatal"] = "\n".join(l for l in text[k:k + 2000].splitlines()[:20] if l.strip())
    return out


def _read_cell_zones(d: Path) -> dict:
    """constant/polyMesh/cellZones (ASCII or binary) -> {name: cells}."""
    p = Path(d) / "cellZones"
    if not p.is_file():
        return {}
    raw = p.read_bytes()
    hdr = re.search(rb"FoamFile\s*\{(.*?)\}", raw, re.S)
    binary = bool(hdr and re.search(rb"format\s+binary", hdr.group(1)))
    pos = hdr.end() if hdr else 0
    out = {}
    for m in re.finditer(rb"(\w+)\s*\{\s*(?:type\s+\w+;\s*)?cellLabels\s+List<label>\s*", raw[pos:]):
        name = m.group(1).decode()
        at = pos + m.end()
        if raw[at:at + 1] == b"{" or re.match(rb"\d+\s*\{", raw[at:]):
            raise MeshError("uniform cellZone lists are not expected")
        mm = re.match(rb"(\d+)\s*\(", raw[at:])
        n = int(mm.group(1))
        op = at + mm.end() - 1
        if binary:
            size = 4 if raw[op + 1 + 4 * n:op + 2 + 4 * n] == b")" else 8
            a = np.frombuffer(raw, dtype="<i4" if size == 4 else "<i8", count=n, offset=op + 1)
        else:
            cl = raw.index(b")", op)
            a = np.array(raw[op + 1:cl].split(), np.int64)
        out[name] = np.asarray(a, np.int64)
    return out


def _read(path: Path) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


# ---- the whole path -----------------------------------------------------------------------------------
def mesh_snappy(study: dict, folder: Path):
    from . import foam
    from .executor import procs_of, timeout_s
    from .select import Groups, assign_bcs

    folder = Path(folder)
    m = study.get("mesh") or {}
    external = study.get("domain", {}).get("kind") == "external"
    box = list(map(float, study["domain"]["box"])) if external else None
    h = float(m["size"])
    sf = m.get("surface", h / 8)
    smin = float(min(sf) if isinstance(sf, list) else sf)
    smax = float(max(sf) if isinstance(sf, list) else sf)
    notes = []
    pts, surf, ipt, pbb = surface_premesh(study, folder, 2 * smax, smin / 4, box)
    if external:
        st = settings(study, True, box, pbb)
    else:
        st = settings(study, False, list(pbb[0]) + list(pbb[1]), pbb)
    dropped = {}
    if external:
        pts, surf, dropped, nn = cut_at_box(pts, surf, box, st["surface_mm"][0])
        notes += nn
        for side in touching(pts, surf, box, dropped):
            notes.append(f"the part touches the domain side {side} only at an edge or a point: snappyHexMesh pinches "
                         "cells there; let the part cross the side a little (1 mm) or end on it with a flat face")
    # bcs on the part's surface (and the box's sides), as on a Gmsh mesh
    info = {t: {"on": "part"} for t, _ in surf}
    allp, alls = pts, list(surf)
    if external:
        V, bs, binfo = box_triangles(box, len(pts))
        allp = np.concatenate([pts, V])
        alls += bs
        info.update(binfo)
    tags = np.concatenate([np.full(len(t), tg) for tg, t in alls])
    rings = np.concatenate([t[:, ::-1] if (external and info[tg]["on"] == "part") else t for tg, t in alls])
    groups = Groups(allp, rings, tags, info, external)
    patches, selection, warns = assign_bcs(groups, study, [])
    # regions of the OBJ (part faces) and patches of the background box
    tri_of = dict(surf)
    regions, box_patches, of_names = [], [], {}
    for k, p in enumerate(patches):
        pt = [t for t in p["tags"] if info[t]["on"] == "part"]
        bt = [t for t in p["tags"] if info[t]["on"] == "box"]
        if pt:
            regions.append((f"r{k}", p["name"], p["type"], pt))
            for alias in (p["name"], f"{PART}_r{k}", PART):
                of_names.setdefault(alias, (k, pt))
        if bt:
            nm = p["name"] if not pt else p["name"] + "_box"
            box_patches.append((nm, p["type"], [info[t]["box"] for t in bt]))
            of_names[nm] = (k, bt)
    if len(regions) > 1:
        of_names.pop(PART, None)
    # layers: the part's walls by default, or the bcs named in layers.on
    lay_on = (m.get("layers") or {}).get("on")
    layer_patches = []
    if st["layers"]:
        for k, p in enumerate(patches):
            pt = any(info[t]["on"] == "part" for t in p["tags"])
            bt = any(info[t]["on"] == "box" for t in p["tags"])
            if (lay_on is None and p["kind"] == "wall" and pt) or (lay_on and p["name"] in lay_on):
                if pt:
                    layer_patches += [p["name"], f"{PART}_r{k}"]
                if bt and lay_on:
                    layer_patches.append(p["name"] if not pt else p["name"] + "_box")
    # porous and rotating zones as snapped cellZones
    zones = []
    for z in list(study.get("porous") or []) + list(study.get("rotating") or []):
        reg = z.get("region") or {"cylinder": {k: z[k] for k in ("center", "axis", "radius", "span") if k in z}}
        e = {"name": z["name"]}
        dims = []
        if "box" in reg:
            e.update(shape="box", box=list(map(float, reg["box"])))
            dims = list(np.asarray(reg["box"][3:], float) - np.asarray(reg["box"][:3], float))
        elif "sphere" in reg:
            e.update(shape="sphere", center=list(map(float, reg["sphere"]["center"])), radius=float(reg["sphere"]["radius"]))
            dims = [2 * float(reg["sphere"]["radius"])]
        else:
            c = reg["cylinder"]
            a = np.asarray(c["axis"], float)
            a /= np.linalg.norm(a)
            lo_, hi_ = (np.asarray(st["lo"]), np.asarray(st["hi"]))
            big = float(np.linalg.norm(hi_ - lo_))
            s0, s1 = c.get("span", [-big, big])
            ctr = np.asarray(c.get("center", [0, 0, 0]), float)
            e.update(shape="cylinder", p1=(ctr + s0 * a).tolist(), p2=(ctr + s1 * a).tolist(), radius=float(c["radius"]))
            dims = [2 * float(c["radius"]), float(s1 - s0)]
        e["level"] = min(max(st["surface"][0], level_for(st["h0"], min(dims) / 6)), st["surface"][1] + 1)
        zones.append(e)
    # gaps narrower than three of the finest cells snappyHexMesh may make there
    h_gap = st["h0"] / 2 ** (st["surface"][1] + st["gap"])
    gaps = narrow_gaps(pts, surf, external, 3 * h_gap)
    # files
    sd = folder / "snappy"
    if sd.exists():
        shutil.rmtree(sd)
    for d in ("system", "constant/geometry", "0"):
        (sd / d).mkdir(parents=True, exist_ok=True)
    write_obj(sd / "constant" / "geometry" / f"{PART}.obj", pts, [(g, np.concatenate([tri_of[t] for t in tt]))
                                                                  for g, _, _, tt in regions])
    (sd / "system" / "controlDict").write_text(control_dict())
    (sd / "system" / "blockMeshDict").write_text(block_mesh_dict(st, box_patches, not external))
    (sd / "system" / "surfaceFeaturesDict").write_text(_hdr("surfaceFeaturesDict") +
                                                       f"surfaces (\"{PART}.obj\");\n\nincludedAngle   150;\n")
    procs = max(1, procs_of(study))
    import os
    if os.environ.get("SLURM_NTASKS", "").isdigit():
        procs = max(1, min(procs, int(os.environ["SLURM_NTASKS"])))
    if procs > 1:
        (sd / "system" / "decomposeParDict").write_text(
            decompose_dict(procs, np.asarray(st["hi"], float) - np.asarray(st["lo"], float)))
    fm = foam.find_foam()
    if not fm or "error" in (fm or {}):
        raise MeshError((fm or {}).get("error", "snappy meshes need OpenFOAM (not found)"))
    tmo = timeout_s(study)
    reg3 = [(g, n, t) for g, n, t, _ in regions]
    attempts = []
    qual = st["quality"]
    for attempt in range(2 if st["retry"] else 1):
        for p in sd.glob("processor*"):
            shutil.rmtree(p, ignore_errors=True)
        shutil.rmtree(sd / "constant" / "polyMesh", ignore_errors=True)
        (sd / "system" / "snappyHexMeshDict").write_text(snappy_dict(st, reg3, ipt, layer_patches, zones, qual))
        (sd / "system" / "meshQualityDict").write_text(_hdr("meshQualityDict") +
                                                       "".join(f"{k} {v:g};\n" for k, v in qual.items()))
        foam._step(fm, "blockMesh", "log.blockMesh", sd, tmo)
        if attempt == 0 and st["features"] is not None:
            foam._step(fm, "surfaceFeatures", "log.surfaceFeatures", sd, tmo)
        if procs > 1:
            foam._step(fm, "decomposePar -force", "log.decomposePar", sd, tmo)
            for pd in sd.glob("processor*"):            # the surface and its edges, wherever the ranks look for them
                for sub in ("geometry", "extendedFeatureEdgeMesh"):
                    if (sd / "constant" / sub).is_dir() and not (pd / "constant" / sub).exists():
                        shutil.copytree(sd / "constant" / sub, pd / "constant" / sub)
            rc = foam._step(fm, f"{foam.mpirun_command(procs)} -np {procs} snappyHexMesh -parallel",
                            "log.snappyHexMesh", sd, tmo, raise_on_error=False)
            log = parse_log(_read(sd / "log.snappyHexMesh"))
            if rc == 0:
                foam._step(fm, "reconstructPar -constant", "log.reconstructPar", sd, tmo)
        else:
            rc = foam._step(fm, "snappyHexMesh", "log.snappyHexMesh", sd, tmo, raise_on_error=False)
            log = parse_log(_read(sd / "log.snappyHexMesh"))
        if rc == 0 and not log["finished"]:
            raise MeshError("snappyHexMesh left no finished log (log.snappyHexMesh): " + (_read(sd / "log.snappyHexMesh")[-1500:]
                                                                                          or "the file is missing"))
        if rc != 0 or "fatal" in log:
            raise MeshError("snappyHexMesh stopped:\n" + (log.get("fatal") or foam._tail(sd / "log.snappyHexMesh")))
        foam._step(fm, "checkMesh", "log.checkMesh", sd, min(tmo, 1800), raise_on_error=False)
        from .foamio import parse_checkmesh
        cm = parse_checkmesh((sd / "log.checkMesh").read_text(errors="replace"))
        rec = {"attempt": attempt + 1, "quality": {"maxNonOrtho": qual["maxNonOrtho"],
                                                   "maxInternalSkewness": qual["maxInternalSkewness"]},
               "checkMesh": {k: cm[k] for k in ("ok", "failed_checks", "max_non_orthogonality", "max_skewness",
                                                "min_volume", "cells") if k in cm},
               "illegal_faces": log["illegal_faces"], "cells": log["cells"]}
        attempts.append(rec)
        if cm.get("ok") and not log["illegal_faces"]:
            break
        if attempt == 0 and st["retry"]:
            qual = stricter(qual)
            notes.append("the first mesh failed checkMesh or had illegal faces: made again with stricter quality settings")
    # read it back; tags per face from the surface it lies on
    d = sd / "constant" / "polyMesh"
    mesh = from_foam(d)
    fp = mesh.info["foam_patches"]
    ni = mesh.ninternal
    g = mesh.geometry()
    btag = np.zeros(mesh.nboundary, np.int64)
    out_patches = []
    seen = set()
    for P in fp:
        if P["n"] == 0:
            continue
        hit = of_names.get(P["name"])
        if hit is None:
            raise MeshError(f"snappyHexMesh left {P['n']} faces in an unexpected patch {P['name']!r} "
                            "(the part may not separate the domain as expected, or the inside point is wrong)")
        k, tg = hit
        f = np.arange(P["start"], P["start"] + P["n"])
        btag[f - ni] = _nearest_tags(g["Cf"][f], tg, tri_of, pts, info, box)
        p = patches[k]
        out_patches.append({"name": P["name"], "type": p["type"], "kind": p["kind"], "bc": p["bc"],
                            "tags": sorted({int(t) for t in np.unique(btag[f - ni])})})
        seen.add(k)
    for k, p in enumerate(patches):
        if k not in seen:
            raise MeshError(f"bc {p['name']!r} has no faces in the snappyHexMesh mesh (its faces are smaller than the "
                            "surface cells, or outside the fluid): refine mesh.surface or check the selector")
    mesh.btag = btag
    mesh.tag_info = {int(t): v for t, v in info.items()}
    mesh.zones = {}
    snapped = _read_cell_zones(d)
    report = parse_log(_read(sd / "log.snappyHexMesh"))
    lay = dict(st["layers"]) if st["layers"] else None
    if lay and lay.get("stack_to_surface_cell", 0) > 1.5:
        notes.append(f"mesh.layers: the stack ({lay['total_mm']} mm) is {lay['stack_to_surface_cell']} of the finest surface "
                     "cells; snappyHexMesh tends to collapse stacks thicker than about one cell (fewer layers, or leave n out)")
    if lay and report.get("layers"):
        cov = {r["patch"]: {k: r[k] for k in ("faces", "layers", "thickness_mm", "thickness_pct")} for r in report["layers"]}
        lay["coverage"] = cov
        part_rows = [r for r in report["layers"] if r["patch"] in layer_patches]
        if part_rows:
            fa = sum(r["faces"] for r in part_rows)
            lay["coverage_pct"] = float(f"{sum(r['thickness_pct'] * r['faces'] for r in part_rows) / max(fa, 1):.3g}")
            lay["layers_mean"] = float(f"{sum(r['layers'] * r['faces'] for r in part_rows) / max(fa, 1):.3g}")
            for r in part_rows:
                if r["thickness_pct"] < 60:
                    notes.append(f"layers on {r['patch']!r} reached only {r['thickness_pct']:.0f} % of their thickness "
                                 f"({r['layers']:.2g} of {lay['n']} layers on average): check its sharp edges and gaps, "
                                 "or lower mesh.layers.feature_angle / growth")
    if gaps:
        notes.append(f"narrow gaps: {len(gaps)} place(s) where facing walls are closer than 3 of the finest cells "
                     f"({3 * h_gap:.3g} mm), e.g. {gaps[0]['width_mm']} mm at {gaps[0]['at']}: refine there (mesh.refine) "
                     "or raise mesh.gap")
    for a in attempts[-1:]:
        cm = a["checkMesh"]
        if not cm.get("ok", True):
            notes.append(f"checkMesh still fails ({cm.get('failed_checks', '?')} checks; max non-orthogonality "
                         f"{cm.get('max_non_orthogonality', '?')}, max skewness {cm.get('max_skewness', '?')})")
    mesh.info.update({
        "kind": "snappy", "structured": False, "empty_tags": [], "size": st["h0"],
        "snappy": {"background_cells": st["cells"], "cell_mm": st["h0"], "surface_levels": list(st["surface"]),
                   "surface_mm": [float(f"{x:.4g}") for x in st["surface_mm"]],
                   "features_level": st["features"], "gap_level_increment": st["gap"],
                   "refine": [{k: v for k, v in e.items() if k in ("name", "shape", "level", "cell_mm")} for e in st["regions"]]
                   + [{"shape": "distance", "distance_mm": d_, "level": L} for d_, L in st["distance"]],
                   "zones": {z["name"]: {"level": z["level"], "cells": int(len(snapped.get(z["name"], [])))} for z in zones},
                   "cells_by_stage": report.get("cells", {}), "seconds": report.get("seconds"),
                   "illegal_faces": report.get("illegal_faces", 0), "gap_cells": report.get("gap_cells", 0),
                   "attempts": attempts, "narrow_gaps": gaps, "procs": procs},
        "assigned": (out_patches, selection, warns + notes),
        "snapped_zones": snapped})
    if lay:
        mesh.info["layers"] = lay
    return mesh


def _nearest_tags(Cf: np.ndarray, tags: list, tri_of: dict, pts: np.ndarray, info: dict, box) -> np.ndarray:
    """The tag of the surface face each mesh face lies on: box sides by their plane, part faces by the
    nearest of their triangles (exact distance to the 8 nearest by centre)."""
    tags = list(tags)
    if len(tags) == 1:
        return np.full(len(Cf), tags[0], np.int64)
    out = np.empty(len(Cf), np.int64)
    btags = [t for t in tags if info[t]["on"] == "box"]
    ptags = [t for t in tags if info[t]["on"] == "part"]
    if btags:
        lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
        dist = []
        for t in btags:
            i = BOX_SIDES.index(info[t]["box"])
            ax, val = i // 2, (lo if i % 2 == 0 else hi)[i // 2]
            dist.append(np.abs(Cf[:, ax] - val))
        dist = np.array(dist)
    if ptags:
        from scipy.spatial import cKDTree

        from .select import _point_triangle_distance
        T = np.concatenate([tri_of[t] for t in ptags])
        TT = np.concatenate([np.full(len(tri_of[t]), t) for t in ptags])
        C = pts[T].mean(axis=1)
        k = min(8, len(C))
        _, idx = cKDTree(C).query(Cf, k=k)
        idx = idx.reshape(len(Cf), -1)
        best = np.full(len(Cf), np.inf)
        bt = np.zeros(len(Cf), np.int64)
        for j in range(idx.shape[1]):
            tri = T[idx[:, j]]
            dd = _point_triangle_distance(Cf, pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]])
            better = dd < best
            best[better] = dd[better]
            bt[better] = TT[idx[better, j]]
        pdist = best
    if btags and ptags:
        bmin = dist.min(axis=0)
        out = np.where(bmin < pdist, np.array(btags)[dist.argmin(axis=0)], bt)
    elif btags:
        out = np.array(btags)[dist.argmin(axis=0)]
    else:
        out = bt
    return out
