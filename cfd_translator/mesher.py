"""mesher.py — the fluid domain as a PolyMesh: a structured box, or Gmsh (tetrahedra, or a swept 2D mesh).
Version: 2026.10.16.2

    mesh_study(study, workdir) -> PolyMesh

- box geometry, internal flow, mesh.cells: structured hexahedra (what blockMesh would make), no Gmsh.
- otherwise Gmsh (OpenCASCADE): the geometry (b3dt goal, STEP, box, cylinder), minus nothing (internal)
  or cut out of the domain box (external); then either 3D tetrahedra, or (2d / mesh.extrude) the bottom
  face meshed in 2D (triangles or quads, optional boundary layers on the walls) and swept along Z.
Every boundary face keeps the tag of the Gmsh surface it lies on; domain-box sides are named.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .polymesh import BOX_SIDES, MeshError, box_mesh, extrude, from_cells, tags_from_lists


def geometry_step(study: dict, workdir: Path) -> Path | None:
    """A STEP file of a goal or STEP geometry (None for primitives)."""
    g = study["geometry"]
    if "step" in g:
        p = Path(g["step"]).expanduser()
        if not p.is_file():
            raise MeshError(f"no STEP file {p}")
        return p
    if "goal" in g:
        try:
            from build123d_translator import translate as T
        except ImportError as exc:
            raise MeshError("geometry is a goal but build123d-translator is not installed "
                            "(pip install -e ../build123d-translator)") from exc
        solid, _ = T.build(g["goal"])
        return Path(T.export(solid, Path(workdir) / "part", ("step",))["step"])
    return None


def _gmsh_init(gmsh):
    try:
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


def _build_geometry(gmsh, study: dict, workdir: Path):
    """The fluid volume tag and {surface tag: {"on", "box"}}."""
    occ = gmsh.model.occ
    g = study["geometry"]
    if "box" in g:
        lx, ly, lz = (float(v) for v in g["box"])
        part = [(3, occ.addBox(-lx / 2, -ly / 2, 0, lx, ly, lz))]
    elif "cylinder" in g:
        r, h = (float(v) for v in g["cylinder"])
        part = [(3, occ.addCylinder(0, 0, 0, 0, 0, h, r))]
    else:
        part = [dt for dt in occ.importShapes(str(geometry_step(study, workdir))) if dt[0] == 3]
    occ.synchronize()
    if len(part) != 1:
        raise MeshError(f"the geometry has {len(part)} solids; a study is one solid")
    d = study.get("domain", {})
    info = {}
    if d.get("kind") == "external":
        x0, y0, z0, x1, y1, z1 = (float(v) for v in d["box"])
        box = occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
        out, _ = occ.cut([(3, box)], part)
        occ.synchronize()
        vols = [t for dim, t in out if dim == 3]
        if len(vols) != 1:
            raise MeshError(f"domain box minus the part gives {len(vols)} volumes (the part must lie inside the box "
                            "and not split it)")
        vol = vols[0]
        lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
        tol = 1e-6 * float(np.linalg.norm(hi - lo))
        for _, s in gmsh.model.getBoundary([(3, vol)], combined=False, oriented=False):
            bb = np.array(gmsh.model.getBoundingBox(2, abs(s)))
            side = None
            for ax in range(3):
                if bb[ax + 3] - bb[ax] < tol:
                    if abs(bb[ax] - lo[ax]) < tol:
                        side = BOX_SIDES[2 * ax]
                    elif abs(bb[ax] - hi[ax]) < tol:
                        side = BOX_SIDES[2 * ax + 1]
            info[abs(s)] = {"on": "box", "box": side} if side else {"on": "part"}
        if not any(v["on"] == "part" for v in info.values()):
            raise MeshError("no part faces in the fluid domain (is the part inside the domain box?)")
    else:
        vol = part[0][1]
        for _, s in gmsh.model.getBoundary([(3, vol)], combined=False, oriented=False):
            info[abs(s)] = {"on": "part"}
    return vol, info


def _sizes(gmsh, study: dict, h: float, fields_too: bool = True):
    """Size options, and (fields_too) one Ball field per refine sphere; returns the field tags."""
    m = study.get("mesh", {})
    refine = m.get("refine", [])
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min([h] + [float(r["size"]) for r in refine]) / 4)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
    fields = []
    if not fields_too:
        return fields
    for r in refine:
        f = gmsh.model.mesh.field.add("Ball")
        (x, y, z), rad = r["at"], float(r["r"])
        for k, v in (("XCenter", x), ("YCenter", y), ("ZCenter", z), ("Radius", rad), ("Thickness", rad),
                     ("VIn", float(r["size"])), ("VOut", h)):
            gmsh.model.mesh.field.setNumber(f, k, float(v))
        fields.append(f)
    return fields


def _surface_mesh(gmsh):
    """Nodes (N, 3) and [(surface tag, triangles (m, 3) as node indices)] of the current 2D mesh."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, np.int64)
    pts = np.asarray(coords, float).reshape(-1, 3)
    lut = -np.ones(int(tags.max()) + 1, np.int64)
    lut[tags] = np.arange(len(tags))
    out = []
    for _, s in gmsh.model.getEntities(2):
        et, _, en = gmsh.model.mesh.getElements(2, s)
        for t, n in zip(et, en):
            k = {2: 3, 3: 4}.get(int(t))
            if k:
                out.append((int(s), lut[np.asarray(n, np.int64).reshape(-1, k)]))
    return pts, lut, out


def mesh_study(study: dict, workdir: Path, size_factor: float = 1.0):
    """The PolyMesh of the fluid domain; mesh.info says how it was made; mesh.info['empty_tags'] are
    the 2D front/back faces."""
    m = study.get("mesh", {})
    g = study["geometry"]
    external = study.get("domain", {}).get("kind") == "external"
    two_d = bool(study.get("2d"))
    if "box" in g and "cells" in m and not external:
        lx, ly, lz = (float(v) for v in g["box"])
        nx, ny, nz = (max(1, round(c / size_factor)) if size_factor != 1 else c for c in m["cells"])
        if two_d:
            nz = 1
        mesh = box_mesh(lx, ly, lz, nx, ny, nz, grading=m.get("grading", (1, 1, 1)))
        mesh.info["empty_tags"] = [4, 5] if two_d else []
        return mesh
    try:
        import gmsh
    except ImportError as exc:
        raise MeshError("this geometry needs Gmsh (pip install gmsh); only box geometries with mesh.cells mesh without it") from exc
    _gmsh_init(gmsh)
    try:
        gmsh.model.add("fluid")
        vol, info = _build_geometry(gmsh, study, workdir)
        bb = np.array(gmsh.model.getBoundingBox(3, vol))
        diag = float(np.linalg.norm(bb[3:] - bb[:3]))
        h = float(m.get("size") or diag / 30) * size_factor
        if two_d or "extrude" in m:
            mesh = _swept(gmsh, study, vol, info, bb, h, size_factor)
        else:
            mesh = _tets(gmsh, study, vol, info, h)
        mesh.info.update({"size": h, "diag": diag})
        try:
            gmsh.write(str(Path(workdir) / "mesh2d.msh" if mesh.info.get("kind") == "extruded" else Path(workdir) / "mesh.msh"))
        except Exception:  # noqa: BLE001 (the .msh is a convenience)
            pass
    finally:
        gmsh.finalize()
    return mesh


def _tets(gmsh, study, vol, info, h):
    fields = _sizes(gmsh, study, h)
    if fields:
        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    pts, lut, surf = _surface_mesh(gmsh)
    et, _, en = gmsh.model.mesh.getElements(3, vol)
    tets = [lut[np.asarray(n, np.int64).reshape(-1, 4)] for t, n in zip(et, en) if int(t) == 4]
    if not tets:
        raise MeshError("Gmsh produced no tetrahedra")
    mesh = from_cells(pts, {"tet": np.concatenate(tets)}, tags_from_lists(surf), info)
    mesh.info.update({"kind": "tetrahedra", "structured": False, "empty_tags": []})
    return mesh


def _swept(gmsh, study, vol, info, bb, h, size_factor):
    """Bottom face meshed in 2D, swept along Z (2d: one layer)."""
    m = study.get("mesh", {})
    two_d = bool(study.get("2d"))
    zmin, zmax = float(bb[2]), float(bb[5])
    tol = max(1e-6 * float(np.linalg.norm(bb[3:] - bb[:3])), 1e-4)
    bottom, top, sides = [], [], []
    for s in info:
        sb = np.array(gmsh.model.getBoundingBox(2, s))
        if sb[5] - sb[2] < tol:
            if abs(sb[2] - zmin) < tol:
                bottom.append(s)
            elif abs(sb[2] - zmax) < tol:
                top.append(s)
            else:
                sides.append(None)          # a step in height: not a prism
        elif abs(sb[2] - zmin) < tol and abs(sb[5] - zmax) < tol:
            sides.append(s)
        else:
            sides.append(None)
    if len(bottom) != 1 or len(top) != 1 or None in sides:
        raise MeshError("2d / extrude needs a prism along Z: one face at z = min, one at z = max, and side "
                        f"faces spanning the full height (found {len(bottom)} bottom, {len(top)} top faces)")
    bottom, top = bottom[0], top[0]
    # bottom curves -> side surfaces (by adjacency), remembered by bounding box
    curves = {}
    for _, c in gmsh.model.getBoundary([(2, bottom)], combined=False, oriented=False):
        up, _ = gmsh.model.getAdjacencies(1, abs(c))
        side = [int(s) for s in up if int(s) != bottom and int(s) in info]
        if len(side) != 1:
            raise MeshError(f"bottom edge {abs(c)} touches {len(side)} side faces")
        curves[abs(c)] = (side[0], np.array(gmsh.model.getBoundingBox(1, abs(c))))
    # keep only the bottom face (and its edges and corners)
    gmsh.model.occ.remove([(3, vol)])
    gmsh.model.occ.remove([(2, s) for s in info if s != bottom])
    gmsh.model.occ.synchronize()
    now = [abs(c) for _, c in gmsh.model.getBoundary([(2, bottom)], combined=False, oriented=False)]
    corners = {abs(p) for _, p in gmsh.model.getBoundary([(2, bottom)], combined=False, oriented=False, recursive=True)}
    extra = [(1, c) for _, c in gmsh.model.getEntities(1) if c not in now]
    if extra:
        gmsh.model.occ.remove(extra)
        gmsh.model.occ.synchronize()
    extra = [(0, p) for _, p in gmsh.model.getEntities(0) if p not in corners]
    if extra:
        gmsh.model.occ.remove(extra)
        gmsh.model.occ.synchronize()
    remap = {}
    for c in now:
        if c in curves:                    # non-recursive removal keeps the tags
            remap[c] = c
            continue
        cb = np.array(gmsh.model.getBoundingBox(1, c))
        best = min(curves, key=lambda k: float(np.abs(curves[k][1] - cb).max()))
        if float(np.abs(curves[best][1] - cb).max()) > tol:
            raise MeshError("lost track of the bottom face's edges")
        remap[c] = best
    fields = _sizes(gmsh, study, h)
    if fields:
        fmin = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    quads = bool(m.get("quads", True))
    gmsh.option.setNumber("Mesh.RecombineAll", 1 if quads else 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8 if quads else 6)
    layers = 1 if two_d else int(m["extrude"])
    empty = [bottom, top] if two_d else []
    mesh = _sweep_bottom(gmsh, bottom, top, now, remap, curves, info, zmin, zmax, layers, tol)
    wall_curves = []
    if "layers" in m:
        # boundary layers on the walls (external flow: the part's walls, not domain-box sides); the walls
        # are found by selecting on the swept mesh without layers, then the bottom face is meshed again
        from .select import assign_bcs, mesh_groups
        external = study.get("domain", {}).get("kind") == "external"
        patches, _, _ = assign_bcs(mesh_groups(mesh, external), study, empty)
        walls = {t for p in patches if p["kind"] == "wall" for t in p["tags"] if info.get(t, {}).get("on") != "box"}
        wall_curves = [c for c in now if curves[remap[c]][0] in walls]
        if wall_curves:
            L = m["layers"]
            n, first, gr = int(L["n"]), float(L["first"]) * size_factor, float(L.get("growth", 1.2))
            thick = first * n if abs(gr - 1) < 1e-12 else first * (gr ** n - 1) / (gr - 1)
            bl = gmsh.model.mesh.field.add("BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(bl, "CurvesList", wall_curves)
            gmsh.model.mesh.field.setNumber(bl, "Size", first)
            gmsh.model.mesh.field.setNumber(bl, "Ratio", gr)
            gmsh.model.mesh.field.setNumber(bl, "Thickness", thick)
            gmsh.model.mesh.field.setNumber(bl, "Quads", 1)
            gmsh.model.mesh.field.setAsBoundaryLayer(bl)
            gmsh.model.mesh.clear()
            mesh = _sweep_bottom(gmsh, bottom, top, now, remap, curves, info, zmin, zmax, layers, tol)
    mesh.info.update({"empty_tags": empty, "bottom": bottom, "top": top,
                      "boundary_layers_on": sorted({curves[remap[c]][0] for c in wall_curves})})
    return mesh


def _sweep_bottom(gmsh, bottom, top, now, remap, curves, info, zmin, zmax, layers, tol):
    """Mesh the bottom face in 2D and sweep it along Z."""
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes(2, bottom, includeBoundary=True)
    tags = np.asarray(tags, np.int64)
    pts = np.asarray(coords, float).reshape(-1, 3)
    lut = -np.ones(int(tags.max()) + 1, np.int64)
    lut[tags] = np.arange(len(tags))
    tris, quads_ = [], []
    et, _, en = gmsh.model.mesh.getElements(2, bottom)
    for t, nn in zip(et, en):
        if int(t) == 2:
            tris.append(lut[np.asarray(nn, np.int64).reshape(-1, 3)])
        elif int(t) == 3:
            quads_.append(lut[np.asarray(nn, np.int64).reshape(-1, 4)])
        else:
            raise MeshError(f"unexpected 2D element type {int(t)}")
    edge_tags = {}
    for c in now:
        et, _, en = gmsh.model.mesh.getElements(1, c)
        for t, nn in zip(et, en):
            if int(t) == 1:
                for a, b in lut[np.asarray(nn, np.int64).reshape(-1, 2)]:
                    edge_tags[(int(a), int(b))] = curves[remap[c]][0]
    if np.any(np.abs(pts[:, 2] - zmin) > tol):
        raise MeshError("the bottom face is not flat at z = min")
    tri = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    qd = np.concatenate(quads_) if quads_ else np.zeros((0, 4), np.int64)
    mesh = extrude(pts[:, :2], tri, qd, zmin, zmax, layers, bottom, top, edge_tags, info)
    mesh.info.update({"cells_2d": int(len(tri) + len(qd)), "quads_2d": int(len(qd))})
    return mesh
