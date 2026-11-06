"""select.py — boundary faces grouped by CAD face, and the face selectors (same as fea-translator).
Version: 2026.11.06.1

A group is every boundary face of the mesh with one tag (one CAD face, or one side of the domain box).
Selectors:  {"nx", "ny", "nz" (+ "at")}  planar face by outward normal
            {"pts": [[x, y, z], ...]}    the face nearest to each point
            {"cyl_r": r (+ "side", "at")} cylindrical faces by radius
            {"all": true}                every face of the part
            {"box": "xmin" | [...]}      sides of the domain box (external flow only)
            {"group": name | [names]}    a named face group of a mesh geometry (STL solid, OBJ group)
            {"solid": k | [k, ...]}      every face of the k-th body (0-based) of a part made of several
            {"in_box": [x0,y0,z0,x1,y1,z1]}                 faces whose centroid lies in the box
            {"in_cyl": {"axis", "center", "r", "span" | "x"/"y"/"z": [lo, hi], "r_in"}}   ... in the cylinder
            [selector, ...]              union
in_box / in_cyl on a snappy mesh split the part's faces at the region's boundary first (triangle by triangle:
split_regions), so they take exactly the surface inside; on a Gmsh mesh they take the whole CAD faces whose area
lies mostly inside.
Normals are the part's: for internal flow the part is the fluid (outward = out of the fluid); for
external flow the part is the solid (outward = into the fluid), exactly as in build123d-translator
and fea-translator, so one selector names the same face in all three tools.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .polymesh import MeshError, face_geometry


def _point_triangle_distance(q, a, b, c):
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    nn = np.where(nn > 0, nn, 1e-300)
    w = q - a
    v = np.einsum("ij,ij->i", np.cross(w, ac), n) / nn
    u = np.einsum("ij,ij->i", np.cross(ab, w), n) / nn
    inside = (v >= 0) & (u >= 0) & (u + v <= 1)
    dplane = np.abs(np.einsum("ij,ij->i", w, n)) / np.sqrt(nn)

    def seg(p0, p1):
        d = p1 - p0
        t = np.clip(np.einsum("ij,ij->i", q - p0, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-300), 0, 1)
        return np.linalg.norm(q - (p0 + d * t[:, None]), axis=1)
    return np.where(inside, dplane, np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a)))


def fit_cylinder(pts: np.ndarray, normals: np.ndarray, rtol: float = 2e-3):
    """(radius, axis, point on axis) if the points lie on a cylinder, else None."""
    if len(pts) < 6:
        return None
    w, v = np.linalg.eigh(normals.T @ normals)
    # facet normals of a coarse, short cylinder tilt a lot (a triangle spans a chord and a vertex on the
    # other rim), so this only proposes the axis; the point fit below decides
    if w[0] > 0.25 * w[1]:
        return None
    axis = v[:, 0]
    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    u, s = pts @ e1, pts @ e2
    A = np.column_stack([2 * u, 2 * s, np.ones_like(u)])
    sol, *_ = np.linalg.lstsq(A, u * u + s * s, rcond=None)
    r0 = float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0)))
    if r0 <= 0:
        return None
    from scipy.optimize import least_squares

    def resid(x):
        a = axis + x[0] * e1 + x[1] * e2
        a = a / np.linalg.norm(a)
        c = x[2] * e1 + x[3] * e2
        d = pts - c
        return np.linalg.norm(d - np.outer(d @ a, a), axis=1) - x[4]
    fit = least_squares(resid, [0.0, 0.0, sol[0], sol[1], r0], x_scale="jac")
    r = abs(float(fit.x[4]))
    a = axis + fit.x[0] * e1 + fit.x[1] * e2
    a /= np.linalg.norm(a)
    return (r, a, fit.x[2] * e1 + fit.x[3] * e2) if np.abs(fit.fun).max() <= rtol * r else None


@dataclass
class Group:
    tag: int
    on: str                 # "part" or "box"
    box: str | None         # box side name
    rings: np.ndarray       # (T, 4) node rings, -1 padded
    Sf: np.ndarray          # (T, 3) area vectors, part-outward
    Cf: np.ndarray          # (T, 3)
    points: np.ndarray      # mesh points (shared)
    kind: str = "surface"
    group: str | None = None    # mesh geometry: the face group (STL solid, OBJ group) it belongs to
    solid: int | None = None    # the body of a part made of several
    radius: float | None = None
    concave: bool | None = None
    arc: float | None = None

    @property
    def area(self) -> np.ndarray:
        return np.linalg.norm(self.Sf, axis=1)

    @property
    def normal(self) -> np.ndarray:
        return self.Sf / np.maximum(self.area, 1e-300)[:, None]

    @property
    def total_area(self) -> float:
        return float(self.area.sum())

    @property
    def center(self) -> np.ndarray:
        return (self.Cf * self.area[:, None]).sum(axis=0) / max(self.total_area, 1e-300)

    @property
    def planar(self) -> bool:
        n = self.normal
        return bool(np.all(n @ n[0] > 1 - 1e-6))

    def describe(self) -> dict:
        d = {"tag": int(self.tag), "on": self.on, "kind": "plane" if self.planar else self.kind,
             "area_mm2": round(self.total_area, 4), "center": np.round(self.center, 4).tolist()}
        if self.box:
            d["box"] = self.box
        if self.group is not None:
            d["group"] = self.group
        if self.solid is not None:
            d["solid"] = self.solid
        if self.planar:
            d["normal"] = np.round(self.normal[0], 4).tolist()
        if self.radius:
            d["radius"] = round(float(self.radius), 4)
            d["side"] = "inner" if self.concave else "outer"
            d["arc_deg"] = round(float(self.arc), 1)
        return d

    def triangles(self):
        r = self.rings
        n = (r >= 0).sum(axis=1)
        tris = [r[:, [0, 1, 2]]]
        q = n == 4
        if np.any(q):
            tris.append(r[q][:, [0, 2, 3]])
        t = np.concatenate(tris)
        P = self.points
        return P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]

    def distance(self, q) -> float:
        a, b, c = self.triangles()
        return float(_point_triangle_distance(np.asarray(q, float), a, b, c).min())

    def point_on(self) -> list:
        """A point on the face, away from its edges: the centre of its largest mesh face."""
        k = int(np.argmax(self.area))
        return np.round(self.Cf[k], 6).tolist()


class Groups:
    """The boundary groups of a mesh (or of a surface mesh) and selection on them."""

    def __init__(self, points: np.ndarray, rings: np.ndarray, tags: np.ndarray, tag_info: dict, external: bool):
        self.external = external
        Sf, Cf = face_geometry(points, rings)
        self.groups: dict[int, Group] = {}
        tags = np.asarray(tags)
        order = np.argsort(tags, kind="stable")          # one pass, not a mask per tag (thousands of mesh faces)
        ut, start = np.unique(tags[order], return_index=True)
        bounds = list(start) + [len(order)]
        for k, t in enumerate(ut):
            t = int(t)
            m = order[bounds[k]:bounds[k + 1]]
            info = tag_info.get(t, {"on": "part"})
            on = info.get("on", "part") if external else "part"     # internal: every face is the part's
            s = Sf[m] * (-1.0 if (external and on == "part") else 1.0)
            g = Group(t, on, info.get("box") if on == "box" else None, rings[m], s, Cf[m], points,
                      group=info.get("group"), solid=info.get("solid"))
            if not g.planar and on == "part":
                nodes = np.unique(g.rings[g.rings >= 0])
                fit = fit_cylinder(points[nodes], g.normal)
                if fit is not None:
                    r, axis, c = fit
                    d = g.Cf - c
                    radial = d - np.outer(d @ axis, axis)
                    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-300)[:, None]
                    g.kind, g.radius = "cylinder", r
                    g.concave = bool(np.sum(g.area * np.einsum("ij,ij->i", radial, g.normal)) < 0)
                    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
                    e1 /= np.linalg.norm(e1)
                    ang = np.sort(np.degrees(np.arctan2(radial @ np.cross(axis, e1), radial @ e1)) % 360)
                    gaps = np.diff(np.concatenate([ang, [ang[0] + 360]]))
                    g.arc = 360.0 if gaps.max() < 30 else float(360 - gaps.max())
            self.groups[t] = g
        self.size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    def part(self) -> list[Group]:
        return [g for g in self.groups.values() if g.on == "part"]

    def select(self, sel) -> list[int]:
        if isinstance(sel, list):
            out = []
            for s in sel:
                out += [t for t in self.select(s) if t not in out]
            return out
        if "box" in sel:
            if not self.external:
                raise MeshError("'box' selects sides of the external-flow domain; this study is internal flow")
            names = sel["box"] if isinstance(sel["box"], list) else [sel["box"]]
            out = []
            for nm in names:
                hit = [g.tag for g in self.groups.values() if g.box == nm]
                if not hit:
                    raise MeshError(f"no domain-box side {nm!r} left (it may be covered by the part)")
                out += hit
            return out
        groups = self.part()
        if sel.get("all") is True:
            return [g.tag for g in groups]
        if "group" in sel:
            names = sel["group"] if isinstance(sel["group"], list) else [sel["group"]]
            known = sorted({g.group for g in groups if g.group is not None})
            out = []
            for nm in names:
                hit = [g.tag for g in groups if g.group == nm]
                if not hit:
                    raise MeshError(f"no face group {nm!r}" + (f" (groups: {known})" if known else
                                    " (only mesh geometries (stl) have groups)"))
                out += hit
            return out
        if "solid" in sel:
            ks = sel["solid"] if isinstance(sel["solid"], list) else [sel["solid"]]
            out = []
            n = len({g.solid for g in groups if g.solid is not None})
            for k in ks:
                hit = [g.tag for g in groups if g.solid == int(k)]
                if not hit:
                    raise MeshError(f"no solid {k} (the part has {max(n, 1)} solid(s), numbered from 0)")
                out += hit
            return out
        if "in_box" in sel or "in_cyl" in sel:
            out = []
            for g in groups:
                inside = region_mask(sel, g.Cf)
                if inside.any() and float(g.area[inside].sum()) >= 0.5 * g.total_area:
                    out.append(g.tag)
            if not out:
                raise MeshError(f"no part face has its centroid inside {sel}")
            return out
        if "pts" in sel:
            tol = max(0.05, 0.02 * self.size)
            out = []
            for q in sel["pts"]:
                dm, tag = min((g.distance(q), g.tag) for g in groups)
                if dm > tol:
                    raise MeshError(f"no face within {tol:.3g} mm of {list(q)}")
                if tag not in out:
                    out.append(tag)
            return out
        if "cyl_r" in sel:
            r = float(sel["cyl_r"])
            cands = [g for g in groups if g.radius and abs(g.radius - r) <= 0.01 * r]
            if sel.get("side") in ("inner", "outer"):
                cands = [g for g in cands if g.concave is (sel["side"] == "inner")]
            if not cands:
                radii = sorted({round(g.radius, 3) for g in groups if g.radius})
                raise MeshError(f"no cylindrical face of radius {r:g}; radii here: {radii}")
            if "at" in sel:
                return [min(cands, key=lambda g: g.distance(sel["at"])).tag]
            return [g.tag for g in cands]
        d = np.array([float(sel.get(k, 0.0)) for k in ("nx", "ny", "nz")])
        d /= np.linalg.norm(d)
        cands = [g for g in groups if g.planar and g.normal[0] @ d > 0.999]
        if not cands:
            raise MeshError(f"no planar face with outward normal {np.round(d, 4).tolist()}")
        if "at" in sel:
            g = min(cands, key=lambda g: (round(g.distance(sel["at"]), 6), -round(g.total_area, 6)))
        else:
            g = max(cands, key=lambda g: (round(g.total_area, 6), float(g.center @ d)))
        return [g.tag]


def region_mask(sel: dict, X: np.ndarray) -> np.ndarray:
    """Which points lie in the selector's region: in_box [x0, y0, z0, x1, y1, z1] or in_cyl {axis, center, r,
    r_in, span [s0, s1] along the axis from center | x/y/z [lo, hi] coordinate limits}."""
    X = np.atleast_2d(np.asarray(X, float))
    if "in_box" in sel:
        b = np.asarray(sel["in_box"], float)
        return np.all((X >= b[:3]) & (X <= b[3:]), axis=1)
    c = sel["in_cyl"]
    a = np.asarray(c["axis"], float)
    a /= np.linalg.norm(a)
    ctr = np.asarray(c.get("center", [0, 0, 0]), float)
    d = X - ctr
    s = d @ a
    r = np.linalg.norm(d - np.outer(s, a), axis=1)
    m = (r <= float(c["r"])) & (r >= float(c.get("r_in", 0.0)))
    if "span" in c:
        m &= (s >= float(c["span"][0])) & (s <= float(c["span"][1]))
    for k, ax in (("x", 0), ("y", 1), ("z", 2)):
        if k in c:
            m &= (X[:, ax] >= float(c[k][0])) & (X[:, ax] <= float(c[k][1]))
    return m


def region_selectors(sel) -> list[dict]:
    """The in_box / in_cyl selectors inside a bc's faces (a list or one)."""
    if isinstance(sel, list):
        return [r for x in sel for r in region_selectors(x)]
    return [sel] if isinstance(sel, dict) and ("in_box" in sel or "in_cyl" in sel) else []


def split_regions(pts: np.ndarray, surf: list, info: dict, study: dict) -> tuple[list, dict, list]:
    """Part faces (tag, triangles) that straddle an in_box / in_cyl region of a bc are split at its boundary by
    triangle centroid: the part inside becomes a face of its own (a new tag, same group and solid), so the
    selector takes exactly the triangles inside. Returns (surf, info, notes)."""
    regs = [r for b in study.get("bcs") or [] for r in region_selectors(b.get("faces"))]
    if not regs:
        return surf, info, []
    info = dict(info)
    nxt = max([int(t) for t, _ in surf] + [int(t) for t in info]) + 1
    split = 0
    for r in regs:
        out = []
        for t, tri in surf:
            if info.get(t, {}).get("on", "part") != "part" or not len(tri):
                out.append((t, tri))
                continue
            m = region_mask(r, pts[tri].mean(axis=1))
            if m.all() or not m.any():
                out.append((t, tri))
                continue
            out.append((t, tri[~m]))
            out.append((nxt, tri[m]))
            info[nxt] = dict(info.get(t, {"on": "part"}), split_from=int(t))
            nxt += 1
            split += 1
        surf = out
    notes = [f"in_box / in_cyl: {split} part face(s) split at the regions' boundaries"] if split else []
    return surf, info, notes


def mesh_groups(mesh, external: bool) -> Groups:
    b = mesh.boundary_slice()
    return Groups(mesh.points, mesh.faces[b], mesh.btag, mesh.tag_info, external)


PATCH_TYPE = {"inlet": "patch", "outlet": "patch", "opening": "patch", "wall": "wall", "slip": "patch",
              "symmetry": "symmetry"}


def _is_all(sel) -> bool:
    return isinstance(sel, dict) and sel.get("all") is True


def assign_bcs(groups: Groups, study: dict, empty_tags=()) -> tuple[list, dict, list]:
    """Patches from the study's bcs: [{"name", "type", "kind", "bc", "tags"}], what each bc took, warnings.
    A face taken by two bcs is an error, except that {'all': true} yields to the others. Faces no bc
    takes become the no-slip patch 'walls'; with 2d the bottom and top faces are 'frontAndBack' (empty)."""
    from .format import bc_names

    bcs = study["bcs"]
    names = bc_names(study)
    empty = set(int(t) for t in empty_tags)
    claimed: dict[int, int] = {}
    order = sorted(range(len(bcs)), key=lambda i: _is_all(bcs[i]["faces"]))
    warns = []
    taken = {i: [] for i in range(len(bcs))}
    for i in order:
        b = bcs[i]
        tags = groups.select(b["faces"])
        for t in tags:
            if t in empty:
                if _is_all(b["faces"]):
                    continue
                raise MeshError(f"bc {names[i]!r}: face {t} is a 2D front/back face (empty); select an edge-on face")
            if t in claimed:
                if _is_all(b["faces"]):
                    continue
                raise MeshError(f"face {t} {groups.groups[t].describe()} is taken by bc {names[claimed[t]]!r} and bc {names[i]!r}")
            claimed[t] = i
            taken[i].append(t)
    patches = []
    for i, b in enumerate(bcs):
        if not taken[i]:
            raise MeshError(f"bc {names[i]!r} has no faces left")
        patches.append({"name": names[i], "type": PATCH_TYPE[b["kind"]], "kind": b["kind"], "bc": b, "tags": taken[i]})
    rest = [t for t in sorted(groups.groups) if t not in claimed and t not in empty]
    if rest:
        patches.append({"name": "walls", "type": "wall", "kind": "wall", "bc": {"kind": "wall"}, "tags": rest})
        for t in rest:
            g = groups.groups[t]
            if g.on == "box":
                warns.append(f"domain-box side {g.box} has no bc: it is a no-slip wall ('walls')")
    if empty:
        patches.append({"name": "frontAndBack", "type": "empty", "kind": "empty", "bc": {"kind": "empty"},
                        "tags": sorted(empty)})
    report = {p["name"]: [groups.groups[t].describe() for t in p["tags"]] for p in patches if p["kind"] != "empty"}
    return patches, report, warns
