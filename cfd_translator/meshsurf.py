"""meshsurf.py — closed triangle meshes (STL, OBJ, PLY) as a study's geometry, with named face groups.
Version: 2026.11.06.1

    "geometry": {"stl": "body.stl"}                    one or several closed bodies (snappy meshes only)
    "geometry": {"stl": "car.obj", "scale": 1000}      model in metres: times 1000 to mm
    "geometry": {"stl": "car.stl", "groups": "groups.json"}   per-triangle groups {name: [triangle indices]}

Groups: the solids of an ASCII STL (solid wheel_front ... endsolid wheel_front), the g/o groups of an OBJ, an
integer face property of a PLY (group, label, region, segment: groups 'g<value>'), or a 'groups' file / dict
naming triangles. A bc takes a group with {"group": "wheel_front"}.

The surface must be closed: every edge shared by exactly two triangles (no holes, no edges shared by three or more),
so snappyHexMesh can tell inside from outside. Each closed body is oriented outward (its signed volume positive).

Faces: triangles of one group and one body joined across edges sharper than FEATURE_DEG apart are split into
'faces' (like CAD faces): planar ones answer nx/ny/nz selectors, cylindrical ones cyl_r, and a face lying on a domain
side (a half model's cut) is left out as for STEP parts.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

FEATURE_DEG = 35.0
MAX_FACES = 6000
SUFFIXES = (".stl", ".obj", ".ply")


class SurfaceError(ValueError):
    pass


# ---- readers --------------------------------------------------------------------------------------------------
def read(path, scale: float = 1.0, groups=None) -> dict:
    """{"points" (N, 3) mm, "tris" (M, 3), "group" (M,) int, "names" [group names]} with duplicate points merged."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise SurfaceError(f"no mesh file {p}")
    suf = p.suffix.lower()
    if suf == ".stl":
        pts, tris, gid, names = _read_stl(p)
    elif suf == ".obj":
        pts, tris, gid, names = _read_obj(p)
    elif suf == ".ply":
        pts, tris, gid, names = _read_ply(p)
    else:
        raise SurfaceError(f"{p.name}: a mesh geometry is .stl, .obj or .ply")
    if not len(tris):
        raise SurfaceError(f"{p.name}: no triangles")
    pts = np.asarray(pts, float) * float(scale)
    pts, tris = _merge_points(pts, np.asarray(tris, np.int64))
    tris, gid = _drop_degenerate(pts, tris, np.asarray(gid, np.int64))
    if groups is not None:
        gid, names = _apply_groups(groups, len(tris), gid, names, p.parent)
    return {"points": pts, "tris": tris, "group": gid, "names": names}


def _read_stl(p: Path):
    data = p.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):                      # binary (even when the header starts with 'solid')
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            v = rec["v"].astype(float).reshape(-1, 3)
            return v, np.arange(3 * n).reshape(n, 3), np.zeros(n, np.int64), ["part"]
    text = data.decode("ascii", errors="replace")
    if not text.lstrip().lower().startswith("solid"):
        raise SurfaceError(f"{p.name}: neither a binary STL (size does not match its triangle count) nor ASCII STL")
    num = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    vre = re.compile(r"vertex\s+" + num + r"\s+" + num + r"\s+" + num)
    pts, gids, names = [], [], []
    for m in re.finditer(r"^\s*solid[ \t]*([^\r\n]*)$(.*?)^\s*endsolid", text, re.S | re.M | re.I):
        nm = (m.group(1).strip().split() or [f"solid{len(names) + 1}"])[0]
        v = np.array(vre.findall(m.group(2)), float)
        if len(v) % 3:
            raise SurfaceError(f"{p.name}: solid {nm!r} has {len(v)} vertices, not a multiple of three")
        if not len(v):
            continue
        if nm in names:                                   # the same name twice: one group
            k = names.index(nm)
        else:
            k = len(names)
            names.append(nm)
        pts.append(v)
        gids.append(np.full(len(v) // 3, k))
    if not pts:
        raise SurfaceError(f"{p.name}: no 'solid ... endsolid' block with vertices")
    v = np.concatenate(pts)
    return v, np.arange(len(v)).reshape(-1, 3), np.concatenate(gids), names


def _read_obj(p: Path):
    pts, tris, gid, names = [], [], [], []
    cur = None
    for line in p.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s[0] == "#":
            continue
        tag, _, rest = s.partition(" ")
        if tag == "v":
            pts.append([float(x) for x in rest.split()[:3]])
        elif tag in ("g", "o"):
            nm = (rest.split() or ["default"])[0]
            if nm not in names:
                names.append(nm)
            cur = names.index(nm)
        elif tag == "f":
            idx = []
            for tok in rest.split():
                i = int(tok.split("/")[0])
                idx.append(i - 1 if i > 0 else len(pts) + i)
            if cur is None:
                names.append("part")
                cur = 0
            for k in range(1, len(idx) - 1):               # polygons as triangle fans
                tris.append([idx[0], idx[k], idx[k + 1]])
                gid.append(cur)
    return np.array(pts, float).reshape(-1, 3), np.array(tris, np.int64).reshape(-1, 3), np.array(gid, np.int64), \
        names or ["part"]


_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
_PLY_GROUP = ("group", "label", "region", "segment", "patch")


def _read_ply(p: Path):
    data = p.read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise SurfaceError(f"{p.name}: not a PLY file")
    head = data[:end].decode("ascii", errors="replace").splitlines()
    body_at = data.index(b"\n", end) + 1
    fmt, elements = "ascii", []
    for line in head:
        f = line.split()
        if not f:
            continue
        if f[0] == "format":
            fmt = f[1]
        elif f[0] == "element":
            elements.append({"name": f[1], "n": int(f[2]), "props": []})
        elif f[0] == "property" and elements:
            if f[1] == "list":
                elements[-1]["props"].append((f[4], "list", f[2], f[3]))
            else:
                elements[-1]["props"].append((f[2], f[1]))
    order = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(fmt)
    verts = faces = None
    gvals = None
    if fmt == "ascii":
        lines = data[body_at:].decode("ascii", errors="replace").split("\n")
        k = 0
        for el in elements:
            rows = [lines[k + i].split() for i in range(el["n"])]
            k += el["n"]
            if el["name"] == "vertex":
                names = [pr[0] for pr in el["props"]]
                ix = [names.index(c) for c in ("x", "y", "z")]
                verts = np.array([[float(r[i]) for i in ix] for r in rows], float)
            elif el["name"] == "face":
                faces, gl = [], []
                for r in rows:
                    j, vals = 0, {}
                    for pr in el["props"]:
                        if pr[1] == "list":
                            c = int(r[j])
                            vals[pr[0]] = [int(x) for x in r[j + 1:j + 1 + c]]
                            j += 1 + c
                        else:
                            vals[pr[0]] = float(r[j])
                            j += 1
                    faces.append(vals.get("vertex_indices", vals.get("vertex_index")))
                    gl.append(next((vals[g] for g in _PLY_GROUP if g in vals), None))
                if any(x is not None for x in gl):
                    gvals = np.array([x if x is not None else -1 for x in gl], float).astype(np.int64)
    elif order:
        pos = body_at
        for el in elements:
            if all(pr[1] != "list" for pr in el["props"]):
                dt = np.dtype([(pr[0], order + _PLY_TYPES[pr[1]]) for pr in el["props"]])
                arr = np.frombuffer(data, dt, count=el["n"], offset=pos)
                pos += dt.itemsize * el["n"]
                if el["name"] == "vertex":
                    verts = np.column_stack([arr[c].astype(float) for c in ("x", "y", "z")])
                continue
            faces, gl = [], []
            for _ in range(el["n"]):
                vals = {}
                for pr in el["props"]:
                    if pr[1] == "list":
                        ct = np.dtype(order + _PLY_TYPES[pr[2]])
                        c = int(np.frombuffer(data, ct, 1, pos)[0])
                        pos += ct.itemsize
                        it = np.dtype(order + _PLY_TYPES[pr[3]])
                        vals[pr[0]] = np.frombuffer(data, it, c, pos).astype(np.int64).tolist()
                        pos += it.itemsize * c
                    else:
                        t = np.dtype(order + _PLY_TYPES[pr[1]])
                        vals[pr[0]] = np.frombuffer(data, t, 1, pos)[0]
                        pos += t.itemsize
                if el["name"] == "face":
                    faces.append(vals.get("vertex_indices", vals.get("vertex_index")))
                    gl.append(next((vals[g] for g in _PLY_GROUP if g in vals), None))
            if el["name"] == "face" and any(x is not None for x in gl):
                gvals = np.array([x if x is not None else -1 for x in gl], float).astype(np.int64)
    else:
        raise SurfaceError(f"{p.name}: PLY format {fmt!r} not supported")
    if verts is None or faces is None:
        raise SurfaceError(f"{p.name}: no vertex or face element")
    tris, src = [], []
    for i, f in enumerate(faces):
        for k in range(1, len(f) - 1):
            tris.append([f[0], f[k], f[k + 1]])
            src.append(i)
    src = np.array(src, np.int64)
    if gvals is not None:
        u = np.unique(gvals)
        names = [f"g{int(v)}" for v in u]
        gid = np.searchsorted(u, gvals[src])
    else:
        names, gid = ["part"], np.zeros(len(tris), np.int64)
    return verts, np.array(tris, np.int64).reshape(-1, 3), gid, names


def _merge_points(pts: np.ndarray, tris: np.ndarray):
    """Points closer than 1e-9 of the model's size are one (STL repeats every vertex per triangle)."""
    size = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
    q = np.round((pts - pts.min(axis=0)) / (size * 1e-9)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return pts[first], inv.reshape(-1)[tris]


def _drop_degenerate(pts, tris, gid):
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    a = np.linalg.norm(np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]]), axis=1)
    ok &= a > 1e-14 * max(float(a.max()), 1e-300)
    return tris[ok], gid[ok]


def _apply_groups(groups, n: int, gid, names, base: Path):
    """groups: {name: [triangle indices]} or a JSON file of it; triangles not named keep their own group."""
    if isinstance(groups, str):
        gp = Path(groups).expanduser()
        if not gp.is_absolute():
            gp = base / gp
        try:
            groups = json.loads(gp.read_text())
        except (OSError, ValueError) as exc:
            raise SurfaceError(f"geometry.groups: cannot read {gp}: {exc}") from exc
    if isinstance(groups, list):                       # one name per triangle
        if len(groups) != n:
            raise SurfaceError(f"geometry.groups: a list needs one name per triangle ({n}), not {len(groups)}")
        u = sorted(set(map(str, groups)))
        return np.array([u.index(str(x)) for x in groups], np.int64), u
    if not isinstance(groups, dict):
        raise SurfaceError("geometry.groups: {name: [triangle indices]}, a list of names (one per triangle) or a file")
    gid = gid.copy()
    names = list(names)
    for nm, idx in groups.items():
        idx = np.asarray(idx, np.int64)
        if len(idx) and (idx.min() < 0 or idx.max() >= n):
            raise SurfaceError(f"geometry.groups[{nm!r}]: triangle indices 0..{n - 1}")
        if nm not in names:
            names.append(nm)
        gid[idx] = names.index(nm)
    return gid, names


# ---- topology ---------------------------------------------------------------------------------------------------
def edges(tris: np.ndarray):
    """(sorted edge keys per triangle side (3M,), unique keys, inverse, counts)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    key = e[:, 0] * (int(tris.max()) + 1) + e[:, 1]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return e, u, inv, cnt


def _adjacency(tris: np.ndarray):
    """Pairs of triangles sharing a manifold edge: (a, b) arrays, and the side index (0..2) in each."""
    m = len(tris)
    _, u, inv, cnt = edges(tris)
    order = np.argsort(inv, kind="stable")
    si = inv[order]
    two = cnt[si] == 2
    first = np.r_[True, si[1:] != si[:-1]]
    a_pos = order[first & two]
    b_pos = order[np.r_[False, (si[1:] == si[:-1])] & two]
    return a_pos % m, b_pos % m, a_pos // m, b_pos // m


def components(tris: np.ndarray) -> np.ndarray:
    """Body index per triangle: triangles connected through shared edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a, b, _, _ = _adjacency(tris)
    m = len(tris)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
    return connected_components(g, directed=False)[1]


def check(pts: np.ndarray, tris: np.ndarray, limit: int = 5) -> dict:
    """Is the surface closed? {open_edges, nonmanifold_edges, at: [mm], bodies, volume_mm3, area_mm2, bbox}."""
    e, u, inv, cnt = edges(tris)
    c = cnt[inv]
    out = {"triangles": int(len(tris)), "open_edges": int((cnt == 1).sum()), "nonmanifold_edges": int((cnt > 2).sum())}
    bad = (c == 1) | (c > 2)
    if bad.any():
        mid = 0.5 * (pts[e[bad, 0]] + pts[e[bad, 1]])
        pick = [0]
        for _ in range(min(limit, len(mid)) - 1):
            d = np.min(np.linalg.norm(mid[:, None, :] - mid[pick][None, :, :], axis=2), axis=1)
            pick.append(int(np.argmax(d)))
        out["at"] = [[round(float(v), 3) for v in mid[k]] for k in dict.fromkeys(pick)]
    P = pts[tris]
    out["area_mm2"] = float(0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum())
    out["bbox_min"] = [round(float(v), 4) for v in pts.min(axis=0)]
    out["bbox_max"] = [round(float(v), 4) for v in pts.max(axis=0)]
    return out


def orient(pts: np.ndarray, tris: np.ndarray, comp: np.ndarray) -> tuple[np.ndarray, int]:
    """Triangles ordered consistently within each body and pointing out of it (positive signed volume).
    Returns (tris, triangles flipped)."""
    from collections import deque
    tris = tris.copy()
    m = len(tris)
    a, b, sa, sb = _adjacency(tris)
    # the shared edge runs the same way in both triangles: one of them is flipped relative to the other
    def side(t, s):
        return tris[t, s], tris[t, (s + 1) % 3]
    nb = [[] for _ in range(m)]
    for x, y, s1, s2 in zip(a.tolist(), b.tolist(), sa.tolist(), sb.tolist()):
        nb[x].append((y, s1, s2))
        nb[y].append((x, s2, s1))
    flip = np.zeros(m, bool)
    seen = np.zeros(m, bool)
    for s0 in range(m):
        if seen[s0]:
            continue
        seen[s0] = True
        dq = deque([s0])
        while dq:
            t = dq.popleft()
            for y, st, sy in nb[t]:
                if seen[y]:
                    continue
                p0, _ = side(t, st)
                q0, _ = side(y, sy)
                # consistent neighbours run the shared edge in opposite directions: y flips when it runs the
                # edge the same way as t does after t's own flip
                flip[y] = bool((p0 == q0) != flip[t])
                seen[y] = True
                dq.append(y)
    tris[flip] = tris[flip][:, ::-1]
    nflip = int(flip.sum())
    P = pts[tris]
    vol = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6.0
    vb = np.bincount(comp, vol, comp.max() + 1)
    neg = vb[comp] < 0
    tris[neg] = tris[neg][:, ::-1]
    return tris, nflip + int(neg.sum())


def body_volumes(pts: np.ndarray, tris: np.ndarray, comp: np.ndarray) -> np.ndarray:
    P = pts[tris]
    vol = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6.0
    return np.bincount(comp, vol, comp.max() + 1)


def face_tags(pts: np.ndarray, tris: np.ndarray, gid: np.ndarray, comp: np.ndarray,
              angle_deg: float = FEATURE_DEG) -> np.ndarray:
    """Face index (0..) per triangle: one group and one body, joined across edges smoother than angle_deg. When
    that gives more than MAX_FACES faces (a noisy scan), the faces are the (group, body) pairs alone."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P = pts[tris]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    a, b, _, _ = _adjacency(tris)
    keep = (gid[a] == gid[b]) & (comp[a] == comp[b])
    smooth = keep & (np.einsum("ij,ij->i", n[a], n[b]) >= np.cos(np.radians(angle_deg)))
    m = len(tris)
    k, lab = connected_components(coo_matrix((np.ones(int(smooth.sum())), (a[smooth], b[smooth])), shape=(m, m)),
                                  directed=False)
    if k > MAX_FACES:
        k, lab = connected_components(coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(m, m)),
                                      directed=False)
    return lab


def inside(pts: np.ndarray, tris: np.ndarray, q) -> np.ndarray:
    """Points inside the closed surface: ray parity along three skew directions, majority vote."""
    q = np.atleast_2d(np.asarray(q, float))
    A, B, C = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    e1, e2 = B - A, C - A
    votes = np.zeros(len(q), int)
    for d in ((0.5773, 0.5774, 0.5775), (-0.3011, 0.8123, 0.4993), (0.7071, -0.1234, -0.6963)):
        d = np.asarray(d) / np.linalg.norm(d)
        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-300
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        for i, x in enumerate(q):
            s = x - A
            u = np.einsum("ij,ij->i", s, h) * inv
            qq = np.cross(s, e1)
            v = (qq @ d) * inv
            t = np.einsum("ij,ij->i", e2, qq) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            votes[i] += int(hit.sum()) % 2
    return votes >= 2


# ---- the part for snappyHexMesh -----------------------------------------------------------------------------------
def load(geometry: dict) -> dict:
    """The study's mesh geometry, checked and oriented: {"points", "tris", "group", "names", "body", "face",
    "bodies", "check", "flipped"}. Raises SurfaceError when it is not closed."""
    s = read(geometry["stl"], float(geometry.get("scale", 1.0)), geometry.get("groups"))
    pts, tris = s["points"], s["tris"]
    ck = check(pts, tris)
    if ck["open_edges"] or ck["nonmanifold_edges"]:
        where = "; ".join(map(str, ck.get("at", [])[:4]))
        raise SurfaceError(
            f"the mesh geometry is not a closed surface: {ck['open_edges']} open edges (holes) and "
            f"{ck['nonmanifold_edges']} edges shared by three or more triangles (near {where} mm). snappyHexMesh needs "
            "watertight bodies: close the holes and separate touching bodies (or merge them into one), e.g. with a "
            "shrink-wrap or 'make manifold' in the mesh tool")
    comp = components(tris)
    tris, flipped = orient(pts, tris, comp)
    vols = body_volumes(pts, tris, comp)
    face = face_tags(pts, tris, s["group"], comp)
    return {**s, "tris": tris, "body": comp, "face": face, "bodies": int(comp.max() + 1),
            "body_volumes_mm3": [float(v) for v in vols], "check": ck, "flipped": flipped}


_STATS: dict = {}


def surface_stats(path, scale: float = 1.0) -> dict | None:
    """Volume, area, bounding box, bodies and groups of a mesh file (cached by path, size and time), or None."""
    try:
        p = Path(path).expanduser()
        key = (str(p), p.stat().st_mtime, p.stat().st_size, float(scale))
    except OSError:
        return None
    if key not in _STATS:
        try:
            s = read(p, scale)
            ck = check(s["points"], s["tris"])
            comp = components(s["tris"])
            tris, _ = orient(s["points"], s["tris"], comp)
            vols = body_volumes(s["points"], tris, comp)
            _STATS[key] = {**ck, "volume_mm3": float(np.abs(vols).sum()), "bodies": int(comp.max() + 1),
                           "groups": list(s["names"])}
        except Exception:  # noqa: BLE001
            _STATS[key] = None
    return _STATS[key]
