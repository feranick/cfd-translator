"""polymesh.py — the fluid mesh as OpenFOAM stores it: points, faces, owner, neighbour, patches.
Version: 2026.10.16.1

Pure NumPy. Cells (tetrahedra, hexahedra, prisms, pyramids; Gmsh node order) become polyMesh faces:

- every cell face is oriented outward from its cell (checked against the cell centroid, so the
  mesher's node order never matters);
- faces shared by two cells are internal: owner = the lower cell index, normal from owner to
  neighbour; internal faces sorted by (owner, neighbour) (OpenFOAM's upper-triangular order);
- the others are boundary faces, each with the tag of the CAD face (or box side) it came from;
  patches group tags, and their faces are written contiguously in patch order.

Geometry follows OpenFOAM (primitiveMesh): face area vectors and centres from triangles around the
face's mean point, cell centres and volumes from face pyramids, boundary delta coefficients
1/(n . (Cf - C)). Coordinates are mm here; OpenFOAM gets metres (points x 1e-3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CELL_FACES = {
    "tet": ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)),
    "hex": ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    "prism": ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    "pyramid": ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}
CELL_NODES = {"tet": 4, "hex": 8, "prism": 6, "pyramid": 5}
SCALE = 1e-3          # mm -> m


class MeshError(Exception):
    """A mesh OpenFOAM cannot use, or a selector that matches no face."""


def _rows(a: np.ndarray) -> np.ndarray:
    """Rows of an int array as hashable/sortable void scalars."""
    a = np.ascontiguousarray(a, dtype=np.int64)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def face_geometry(points: np.ndarray, faces: np.ndarray):
    """Area vectors (F, 3) and centres (F, 3) of faces given as (F, 4) node rings padded with -1,
    computed as OpenFOAM does (triangles around the mean point, centre weighted by triangle area)."""
    n = (faces >= 0).sum(axis=1)
    P = points[np.where(faces >= 0, faces, 0)]                   # (F, 4, 3)
    mask = (faces >= 0)[:, :, None]
    xc = (P * mask).sum(axis=1) / n[:, None]
    Sf = np.zeros_like(xc)
    sumA = np.zeros(len(faces))
    sumAc = np.zeros_like(xc)
    for k in range(4):
        valid = k < n
        a = P[:, k]
        b = P[np.arange(len(faces)), np.where(k + 1 < n, k + 1, 0)]
        tn = np.cross(b - a, xc - a)
        tn[~valid] = 0
        ta = np.linalg.norm(tn, axis=1)
        Sf += tn
        sumA += ta
        sumAc += ta[:, None] * (a + b + xc)
    Sf *= 0.5
    Cf = np.where(sumA[:, None] > 1e-300, sumAc / (3 * np.maximum(sumA, 1e-300))[:, None], xc)
    tri = n == 3                                                 # exact for triangles
    Cf[tri] = xc[tri]
    return Sf, Cf


@dataclass
class PolyMesh:
    points: np.ndarray            # (P, 3) mm
    faces: np.ndarray             # (F, 4) node rings, -1 padded; internal faces first
    owner: np.ndarray             # (F,)
    neighbour: np.ndarray         # (Fi,)
    btag: np.ndarray              # (F - Fi,) CAD-face tag of each boundary face
    ncells: int
    tag_info: dict = field(default_factory=dict)     # tag -> {"on": "part"|"box", "box": "xmin", ...}
    patches: list = field(default_factory=list)      # [{"name", "type", "start", "n", "tags"}]
    info: dict = field(default_factory=dict)

    # ---- sizes ----
    @property
    def nfaces(self) -> int:
        return len(self.faces)

    @property
    def ninternal(self) -> int:
        return len(self.neighbour)

    @property
    def nboundary(self) -> int:
        return self.nfaces - self.ninternal

    def tags(self) -> list[int]:
        return sorted(int(t) for t in np.unique(self.btag))

    # ---- geometry (cached) ----
    def geometry(self):
        if getattr(self, "_geo", None) is None:
            Sf, Cf = face_geometry(self.points, self.faces)
            nc = self.ncells
            own, nei, ni = self.owner, self.neighbour, self.ninternal
            cnt = np.bincount(own, minlength=nc) + np.bincount(nei, minlength=nc)
            cest = np.zeros((nc, 3))
            np.add.at(cest, own, Cf)
            np.add.at(cest, nei, Cf[:ni])
            cest /= cnt[:, None]
            v_own = np.einsum("ij,ij->i", Sf, Cf - cest[own])
            v_nei = -np.einsum("ij,ij->i", Sf[:ni], Cf[:ni] - cest[nei])
            pc_own = 0.75 * Cf + 0.25 * cest[own]
            pc_nei = 0.75 * Cf[:ni] + 0.25 * cest[nei]
            vol3 = np.bincount(own, v_own, nc) + np.bincount(nei, v_nei, nc)
            C = np.zeros((nc, 3))
            np.add.at(C, own, v_own[:, None] * pc_own)
            np.add.at(C, nei, v_nei[:, None] * pc_nei)
            ok = np.abs(vol3) > 1e-300
            C[ok] /= vol3[ok, None]
            C[~ok] = cest[~ok]
            self._geo = {"Sf": Sf, "Cf": Cf, "C": C, "V": vol3 / 3.0}
        return self._geo

    def boundary_slice(self) -> slice:
        return slice(self.ninternal, self.nfaces)

    def delta_coeffs_boundary(self) -> np.ndarray:
        """1/(n . (Cf - C_owner)) on boundary faces, mm^-1 (OpenFOAM's fvPatch::deltaCoeffs)."""
        g = self.geometry()
        b = self.boundary_slice()
        Sf, Cf = g["Sf"][b], g["Cf"][b]
        n = Sf / np.linalg.norm(Sf, axis=1)[:, None]
        d = np.einsum("ij,ij->i", n, Cf - g["C"][self.owner[b]])
        return 1.0 / np.maximum(d, 1e-300)

    # ---- quality ----
    def quality(self) -> dict:
        """Closed cells, positive volumes, orientation, non-orthogonality, face ordering."""
        g = self.geometry()
        Sf, V = g["Sf"], g["V"]
        nc, ni = self.ncells, self.ninternal
        close = np.zeros((nc, 3))
        np.add.at(close, self.owner, Sf)
        np.subtract.at(close, self.neighbour, Sf[:ni])
        amag = np.zeros(nc)
        np.add.at(amag, self.owner, np.linalg.norm(Sf, axis=1))
        np.add.at(amag, self.neighbour, np.linalg.norm(Sf[:ni], axis=1))
        open_rel = float((np.linalg.norm(close, axis=1) / np.maximum(amag, 1e-300)).max()) if nc else 0.0
        d = g["C"][self.neighbour] - g["C"][self.owner[:ni]]
        cosang = np.einsum("ij,ij->i", d, Sf[:ni]) / np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(Sf[:ni], axis=1), 1e-300)
        nonorth = np.degrees(np.arccos(np.clip(cosang, -1, 1))) if ni else np.zeros(0)
        order_ok = bool(np.all(self.owner[:ni] < self.neighbour)) and bool(
            np.all(np.diff(self.owner[:ni] * (nc + 1) + self.neighbour) > 0)) if ni else True
        bvec = g["Cf"][ni:] - g["C"][self.owner[ni:]]
        outward = bool(np.all(np.einsum("ij,ij->i", bvec, Sf[ni:]) > 0))
        return {"cells": int(nc), "faces": int(self.nfaces), "internal_faces": int(ni), "points": int(len(self.points)),
                "volume_mm3": float(V.sum()), "min_volume_mm3": float(V.min()) if nc else 0.0,
                "max_open_cell": open_rel, "max_non_orthogonality": float(nonorth.max()) if ni else 0.0,
                "mean_non_orthogonality": float(nonorth.mean()) if ni else 0.0,
                "upper_triangular": order_ok, "boundary_outward": outward,
                "ok": bool(nc and V.min() > 0 and open_rel < 1e-6 and order_ok and outward
                           and (nonorth.max() if ni else 0) < 89.0)}

    # ---- patches ----
    def set_patches(self, patches: list[dict]) -> None:
        """patches: [{"name", "type", "tags": [...]}] covering every boundary tag once. Reorders the
        boundary faces so each patch is contiguous (in this order), and records start/n."""
        ni = self.ninternal
        seen = {}
        for k, p in enumerate(patches):
            for t in p["tags"]:
                if t in seen:
                    raise MeshError(f"face {t} is in patches {patches[seen[t]]['name']} and {p['name']}")
                seen[t] = k
        missing = sorted(set(self.tags()) - set(seen))
        if missing:
            raise MeshError(f"boundary faces {missing} are in no patch")
        pidx = np.array([seen[int(t)] for t in self.btag]) if len(self.btag) else np.zeros(0, int)
        order = np.lexsort((self.owner[ni:], pidx))
        self.faces = np.concatenate([self.faces[:ni], self.faces[ni:][order]])
        self.owner = np.concatenate([self.owner[:ni], self.owner[ni:][order]])
        self.btag = self.btag[order]
        pidx = pidx[order]
        out, start = [], ni
        for k, p in enumerate(patches):
            n = int((pidx == k).sum())
            out.append({"name": p["name"], "type": p["type"], "start": start, "n": n, "tags": list(p["tags"])})
            start += n
        self.patches = out
        self._geo = None

    def patch(self, name: str) -> dict:
        for p in self.patches:
            if p["name"] == name:
                return p
        raise KeyError(name)

    def patch_faces(self, name: str) -> np.ndarray:
        p = self.patch(name)
        return np.arange(p["start"], p["start"] + p["n"])

    # ---- OpenFOAM files ----
    def write(self, case: Path) -> Path:
        d = Path(case) / "constant" / "polyMesh"
        d.mkdir(parents=True, exist_ok=True)
        if not self.patches:
            raise MeshError("set_patches before writing")
        note = f"nPoints:{len(self.points)} nCells:{self.ncells} nFaces:{self.nfaces} nInternalFaces:{self.ninternal}"
        pts = self.points * SCALE
        _write_list(d / "points", "vectorField", "points", (f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in pts), len(pts))
        n = (self.faces >= 0).sum(axis=1)
        fl = (f"{k}(" + " ".join(map(str, f[:k])) + ")" for f, k in zip(self.faces.tolist(), n.tolist()))
        _write_list(d / "faces", "faceList", "faces", fl, self.nfaces)
        _write_list(d / "owner", "labelList", "owner", map(str, self.owner.tolist()), self.nfaces, note)
        _write_list(d / "neighbour", "labelList", "neighbour", map(str, self.neighbour.tolist()), self.ninternal, note)
        lines = [header("polyBoundaryMesh", "boundary", "constant/polyMesh"), f"{len(self.patches)}", "("]
        for p in self.patches:
            lines += [f"    {p['name']}", "    {", f"        type            {p['type']};"]
            if p["type"] in ("wall", "empty", "symmetry", "symmetryPlane"):
                lines.append(f"        inGroups        List<word> 1({p['type']});")
            lines += [f"        nFaces          {p['n']};", f"        startFace       {p['start']};", "    }"]
        lines += [")", "", "// " + "*" * 73 + " //", ""]
        (d / "boundary").write_text("\n".join(lines))
        return d


def header(cls: str, obj: str, location: str, note: str | None = None) -> str:
    n = f'    note        "{note}";\n' if note else ""
    return ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "  Written by cfd-translator\n"
            "\\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n{\n    format      ascii;\n"
            f"    class       {cls};\n{n}    location    \"{location}\";\n    object      {obj};\n}}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n")


def _write_list(path: Path, cls: str, obj: str, items, n: int, note: str | None = None) -> None:
    with open(path, "w") as f:
        f.write(header(cls, obj, "constant/polyMesh", note))
        f.write(f"\n{n}\n(\n")
        for s in items:
            f.write(s)
            f.write("\n")
        f.write(")\n\n// " + "*" * 73 + " //\n")


# ---- building from cells ----------------------------------------------------------------------
def from_cells(points: np.ndarray, cells: dict, tag_faces, tag_info: dict | None = None) -> PolyMesh:
    """PolyMesh from cells {"tet"|"hex"|"prism"|"pyramid": (n, k) node indices} (cell numbers in
    this dict order). tag_faces(rings) -> tags gives the CAD-face tag of each boundary face ring
    ((B, 4), -1 padded)."""
    points = np.asarray(points, float)
    blocks, cid0 = [], 0
    for kind in ("tet", "hex", "prism", "pyramid"):
        if kind in cells and len(cells[kind]):
            c = np.asarray(cells[kind], np.int64)
            if c.shape[1] != CELL_NODES[kind]:
                raise MeshError(f"{kind} cells need {CELL_NODES[kind]} nodes, got {c.shape[1]}")
            blocks.append((kind, c, cid0))
            cid0 += len(c)
    nc = cid0
    if nc == 0:
        raise MeshError("no cells")
    cent = np.zeros((nc, 3))
    for kind, c, c0 in blocks:
        cent[c0:c0 + len(c)] = points[c].mean(axis=1)
    by_size = {3: ([], []), 4: ([], [])}
    for kind, c, c0 in blocks:
        ids = np.arange(c0, c0 + len(c))
        for lf in CELL_FACES[kind]:
            by_size[len(lf)][0].append(c[:, list(lf)])
            by_size[len(lf)][1].append(ids)
    int_faces, int_own, int_nei, b_faces, b_own = [], [], [], [], []
    for s, (fl, il) in by_size.items():
        if not fl:
            continue
        F = np.concatenate(fl)
        cid = np.concatenate(il)
        P = points[F]
        xc = P.mean(axis=1)
        area = 0.5 * np.cross(P, np.roll(P, -1, axis=1)).sum(axis=1)
        inward = np.einsum("ij,ij->i", area, xc - cent[cid]) < 0
        F[inward] = F[inward, ::-1]
        keys = _rows(np.sort(F, axis=1))
        _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        if np.any(cnt > 2):
            raise MeshError(f"{int((cnt > 2).sum())} faces shared by more than two cells (non-manifold mesh)")
        order = np.argsort(inv, kind="stable")
        si = inv[order]
        pair_start = np.nonzero((si[:-1] == si[1:]))[0] if len(si) > 1 else np.zeros(0, int)
        a, b = order[pair_start], order[pair_start + 1]
        own = np.minimum(cid[a], cid[b])
        nei = np.maximum(cid[a], cid[b])
        if np.any(own == nei):
            raise MeshError("a cell has the same face twice")
        ring = np.where((cid[a] == own)[:, None], F[a], F[b])
        int_faces.append(_pad(ring))
        int_own.append(own)
        int_nei.append(nei)
        single = cnt[inv] == 1
        b_faces.append(_pad(F[single]))
        b_own.append(cid[single])
    fi = np.concatenate(int_faces) if int_faces else np.zeros((0, 4), np.int64)
    oi = np.concatenate(int_own) if int_own else np.zeros(0, np.int64)
    ni_ = np.concatenate(int_nei) if int_nei else np.zeros(0, np.int64)
    o = np.lexsort((ni_, oi))
    fi, oi, ni_ = fi[o], oi[o], ni_[o]
    fb = np.concatenate(b_faces)
    ob = np.concatenate(b_own)
    tags = np.asarray(tag_faces(fb), np.int64)
    if len(tags) != len(fb):
        raise MeshError("tag_faces returned the wrong number of tags")
    if np.any(tags < 0):
        raise MeshError(f"{int((tags < 0).sum())} boundary faces belong to no CAD face (mesh and geometry disagree)")
    # keep only used points
    used = np.unique(np.concatenate([fi[fi >= 0], fb[fb >= 0]]))
    remap = -np.ones(len(points), np.int64)
    remap[used] = np.arange(len(used))
    fi = np.where(fi >= 0, remap[np.maximum(fi, 0)], -1)
    fb = np.where(fb >= 0, remap[np.maximum(fb, 0)], -1)
    m = PolyMesh(points[used], np.concatenate([fi, fb]), np.concatenate([oi, ob]), ni_, tags, nc,
                 dict(tag_info or {}))
    return m


def _pad(F: np.ndarray) -> np.ndarray:
    if F.shape[1] == 4:
        return F
    return np.concatenate([F, -np.ones((len(F), 4 - F.shape[1]), F.dtype)], axis=1)


def tags_from_lists(tagged: list) -> callable:
    """tag_faces for from_cells: tagged = [(tag, rings (m, 3|4)), ...]; a boundary face takes the tag
    of the listed face with the same nodes; unlisted faces get -1."""
    keys, tags = [], []
    for t, rings in tagged:
        r = np.asarray(rings, np.int64)
        if not len(r):
            continue
        keys.append(_rows(np.sort(_pad(r), axis=1)))
        tags.append(np.full(len(r), int(t)))
    if not keys:
        return lambda fb: -np.ones(len(fb), np.int64)
    K = np.concatenate(keys)
    T = np.concatenate(tags)
    o = np.argsort(K)
    K, T = K[o], T[o]

    def f(fb):
        q = _rows(np.sort(fb, axis=1))
        pos = np.searchsorted(K, q)
        pos = np.minimum(pos, len(K) - 1)
        return np.where(K[pos] == q, T[pos], -1)
    return f


# ---- structured box (what blockMesh would make) ------------------------------------------------
BOX_SIDES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _spacing(L: float, n: int, grading: float) -> np.ndarray:
    """n + 1 coordinates from 0 to L; grading = last cell / first cell (blockMesh simpleGrading)."""
    if n < 1:
        raise MeshError("a box needs at least one cell along each axis")
    if abs(grading - 1.0) < 1e-12 or n == 1:
        return np.linspace(0.0, L, n + 1)
    q = grading ** (1.0 / (n - 1))
    s = q ** np.arange(n)
    return np.concatenate([[0.0], np.cumsum(s)]) * (L / s.sum())


def box_mesh(lx, ly, lz, nx, ny, nz, origin=None, grading=(1, 1, 1)) -> PolyMesh:
    """Hexahedral box [origin, origin + (lx, ly, lz)] (default: centred in X/Y, base on z = 0), with
    tags 0..5 = xmin, xmax, ymin, ymax, zmin, zmax."""
    ox, oy, oz = origin if origin is not None else (-lx / 2, -ly / 2, 0.0)
    xs = ox + _spacing(lx, nx, grading[0])
    ys = oy + _spacing(ly, ny, grading[1])
    zs = oz + _spacing(lz, nz, grading[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])

    def nid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(order="F"), j.ravel(order="F"), k.ravel(order="F")        # cell = i + nx (j + ny k)
    hexes = np.column_stack([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                             nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])
    lo = np.array([xs[0], ys[0], zs[0]])
    hi = np.array([xs[-1], ys[-1], zs[-1]])
    tol = 1e-9 * max(lx, ly, lz)

    def tag(fb):
        c = pts[fb[:, :4]].mean(axis=1)
        t = -np.ones(len(fb), np.int64)
        for ax in range(3):
            t[np.abs(c[:, ax] - lo[ax]) < tol] = 2 * ax
            t[np.abs(c[:, ax] - hi[ax]) < tol] = 2 * ax + 1
        return t
    info = {t: {"on": "box", "box": s} for t, s in enumerate(BOX_SIDES)}
    m = from_cells(pts, {"hex": hexes}, tag, info)
    m.info.update({"kind": "box", "cells_xyz": [int(nx), int(ny), int(nz)], "grading": list(map(float, grading)),
                   "structured": True, "x": xs.tolist(), "y": ys.tolist(), "z": zs.tolist()})
    return m


# ---- extrusion of a 2D mesh along Z -----------------------------------------------------------
def extrude(points2d: np.ndarray, tris: np.ndarray, quads: np.ndarray, z0: float, z1: float, layers: int,
            bottom_tag: int, top_tag: int, edge_tags: dict, tag_info: dict | None = None) -> PolyMesh:
    """Prisms and hexahedra from triangles and quadrangles at z0 swept to z1 in `layers` layers.
    edge_tags: {(a, b): tag} for every boundary edge of the 2D mesh (node indices, any order)."""
    p2 = np.asarray(points2d, float)[:, :2]
    npl = len(p2)
    zs = np.linspace(z0, z1, layers + 1)
    pts = np.column_stack([np.tile(p2, (layers + 1, 1)), np.repeat(zs, npl)])
    cells = {}
    tris = np.asarray(tris, np.int64).reshape(-1, 3)
    quads = np.asarray(quads, np.int64).reshape(-1, 4)
    if len(tris):
        cells["prism"] = np.concatenate([np.column_stack([tris + l * npl, tris + (l + 1) * npl]) for l in range(layers)])
    if len(quads):
        cells["hex"] = np.concatenate([np.column_stack([quads + l * npl, quads + (l + 1) * npl]) for l in range(layers)])
    ek = np.array(sorted((min(a, b) * npl + max(a, b), t) for (a, b), t in edge_tags.items()), np.int64).reshape(-1, 2)

    def tag(fb):
        valid = fb >= 0
        lay = np.where(valid, fb // npl, -1)
        base = np.where(valid, fb % npl, -1)
        lmax = np.where(valid, lay, -1).max(axis=1)
        lmin = np.where(valid, lay, layers + 1).min(axis=1)
        t = -np.ones(len(fb), np.int64)
        t[(lmax == 0)] = bottom_tag
        t[(lmin == layers)] = top_tag
        side = (lmax != lmin)
        if np.any(side) and len(ek):
            b = base[side]
            lo = np.where(b >= 0, b, npl + 1).min(axis=1)
            hi = b.max(axis=1)
            key = lo * npl + hi
            pos = np.minimum(np.searchsorted(ek[:, 0], key), len(ek) - 1)
            t[np.nonzero(side)[0]] = np.where(ek[pos, 0] == key, ek[pos, 1], -1)
        return t
    m = from_cells(pts, cells, tag, tag_info)
    m.info.update({"kind": "extruded", "layers": int(layers), "structured": False})
    return m
