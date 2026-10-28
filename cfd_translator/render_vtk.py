"""render_vtk.py — 3D pictures with VTK (through PyVista), offscreen: lit wall surfaces (an image's
"quality": "high") and isosurfaces ("iso": Q-criterion, |U| or total-pressure coefficient) of a solved study.
Matplotlib (render.py) stays the default for slices and surfaces, and the fallback when VTK cannot draw.
Version: 2026.10.28.1

    render(state_dir, specs, out_dir, name) -> ({name: {path, what, ...}}, warnings)    (in the caller's process)
    probe() -> {ok, window, prefix, why}: can this machine draw offscreen (cached per process)
    q_criterion(mesh, fields) -> Q per cell (1/s2), from Gauss gradients of the cell velocities

The drawing runs in a child process (python -m cfd_translator.render_vtk <state dir> <specs.json> <out dir> <name>):
an OpenGL crash cannot take down a server or a job, and its memory goes back when it ends. The child reads the
saved state (state.npz, written by every run). Offscreen OpenGL comes from EGL (a GPU driver), OSMesa (libosmesa6)
or an X display; without any, the child runs under xvfb-run when it is installed. CFDT_RENDER=matplotlib turns
VTK off; CFDT_RENDER_TIMEOUT (s, default 900) bounds the child; CFDT_ISO_VOXELS (default 6e6) the isosurface grid.

Isosurfaces: Q (or |U|, or Cp_total = (p + rho |U|^2 / 2 - p_ref) / q_ref) per cell is spread onto a uniform
grid over the image's window by inverse-distance weights from the nearest cell centres; grid points farther than
a cell from any cell centre (inside solids) are left below the value. Flying edges draws the surface, which is
coloured by |U|, p or Cp at its points the same way. Half models are mirrored on their symmetry side (mirror).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .polymesh import SCALE

_PROBE: dict = {}
SIZE = (1400, 900)
ISO_FIELDS = {"q": "Q-criterion (1/s²)", "u": "speed (m/s)", "cp_total": "total-pressure coefficient"}
COLORS = {"u": "speed (m/s)", "p": "p (Pa, gauge)", "cp": "Cp"}


# ---- can this machine draw? ---------------------------------------------------------------------
def available() -> bool:
    return importlib.util.find_spec("pyvista") is not None and importlib.util.find_spec("vtk") is not None


_PROBE_CODE = ("import pyvista as pv\n"
               "p = pv.Plotter(off_screen=True, window_size=(64, 64))\n"
               "p.add_mesh(pv.Sphere())\n"
               "img = p.screenshot(return_img=True)\n"
               "print('VTKOK', p.ren_win.GetClassName(), int(img.std() > 0))\n")


def _try(prefix: list) -> tuple[bool, str]:
    try:
        r = subprocess.run(prefix + [sys.executable, "-c", _PROBE_CODE], capture_output=True, text=True, timeout=90,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("VTKOK")), None)
    if r.returncode == 0 and line and line.endswith(" 1"):
        return True, line.split()[1]
    err = [ln for ln in (r.stderr or "").splitlines() if "WARN" in ln or "rror" in ln]
    return False, "; ".join(dict.fromkeys(e.split("|")[-1].strip() for e in err[-3:])) or f"exit {r.returncode}"


def probe() -> dict:
    """{ok, window (the VTK render window class), prefix (e.g. xvfb-run), why}. Cached per process."""
    if _PROBE:
        return _PROBE
    if os.environ.get("CFDT_RENDER", "").strip().lower() == "matplotlib":
        _PROBE.update(ok=False, why="CFDT_RENDER=matplotlib")
    elif not available():
        _PROBE.update(ok=False, why="PyVista/VTK not installed (pip install 'cfd-translator[render]')")
    else:
        ok, how = _try([])
        if ok:
            _PROBE.update(ok=True, window=how, prefix=[])
        elif shutil.which("xvfb-run"):
            pre = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"]
            ok2, how2 = _try(pre)
            _PROBE.update(ok=ok2, window=how2 if ok2 else None, prefix=pre if ok2 else None,
                          why=None if ok2 else f"no offscreen OpenGL ({how}); xvfb-run failed too ({how2})")
        else:
            _PROBE.update(ok=False, why=f"no offscreen OpenGL ({how}): install libosmesa6 or libegl1, or xvfb")
    return _PROBE


def status() -> dict:
    """For tool_status: the PyVista version, and how it draws once a picture has been asked for."""
    if not available():
        return {"vtk": None}
    from importlib.metadata import version
    try:
        v = version("pyvista")
    except Exception:  # noqa: BLE001
        v = "yes"
    return {"vtk": v, "offscreen": (_PROBE.get("window") or ("no: " + str(_PROBE.get("why")))) if _PROBE else "not checked"}


# ---- in the caller ------------------------------------------------------------------------------
def render(state_dir: Path, specs: list, out_dir: Path, name: str = "study") -> tuple[dict, list]:
    """Draws specs (iso, or surfaces with quality 'high') in a child process. Images it could not draw are
    missing from the answer, with a warning each."""
    state_dir, out_dir = Path(state_dir), Path(out_dir)
    names = [sp["name"] for sp in specs]
    pr = probe()
    if not pr.get("ok"):
        return {}, [f"image {n!r}: needs VTK: {pr.get('why')}" for n in names]
    if not (state_dir / "state.npz").is_file():
        return {}, [f"image {n!r}: needs VTK, which reads the saved state; none in {state_dir}" for n in names]
    out_dir.mkdir(parents=True, exist_ok=True)
    sp_file = out_dir / f".vtk_specs_{os.getpid()}.json"
    sp_file.write_text(json.dumps(specs))
    env = dict(os.environ)
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = pr["prefix"] + [sys.executable, "-m", "cfd_translator.render_vtk", str(state_dir), str(sp_file),
                          str(out_dir), name]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
                           timeout=float(os.environ.get("CFDT_RENDER_TIMEOUT", "900")))
    except subprocess.TimeoutExpired:
        return {}, [f"image {n!r}: VTK took longer than CFDT_RENDER_TIMEOUT" for n in names]
    finally:
        sp_file.unlink(missing_ok=True)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), None)
    if r.returncode != 0 or line is None:
        tail = " ".join((r.stderr or "").strip().splitlines()[-2:])[-300:]
        return {}, [f"image {n!r}: the VTK renderer failed (exit {r.returncode}): {tail}" for n in names]
    res = json.loads(line)
    return res["images"], res["warnings"]


# ---- numbers (no VTK needed) ----------------------------------------------------------------------
def cell_gradient(mesh, U: np.ndarray, patches: dict) -> np.ndarray:
    """Gauss gradient of the cell velocities, (n, 3, 3) with [c, i, j] = dU_i/dx_j in 1/s. Faces take the
    distance-weighted mean of their two cells; boundary faces their patch value (else the owner's)."""
    g = mesh.geometry()
    Sf, Cf, C, V = g["Sf"], g["Cf"], g["C"], g["V"]
    ni, own, nei, nc = mesh.ninternal, mesh.owner, mesh.neighbour, mesh.ncells
    U = np.asarray(U, float)
    do = np.linalg.norm(Cf[:ni] - C[own[:ni]], axis=1)
    dn = np.linalg.norm(Cf[:ni] - C[nei], axis=1)
    w = dn / np.maximum(do + dn, 1e-300)
    Uf = np.empty((mesh.nfaces, 3))
    Uf[:ni] = w[:, None] * U[own[:ni]] + (1 - w)[:, None] * U[nei]
    Uf[ni:] = U[own[ni:]]
    for p in mesh.patches:
        pu = (patches.get(p["name"]) or {}).get("U")
        if p["type"] != "empty" and pu is not None and np.size(pu) == 3 * p["n"]:
            Uf[p["start"]:p["start"] + p["n"]] = np.asarray(pu, float).reshape(-1, 3)
    G = np.empty((nc, 3, 3))
    for i in range(3):
        for j in range(3):
            fl = Uf[:, i] * Sf[:, j]
            G[:, i, j] = np.bincount(own, fl, nc) - np.bincount(nei, fl[:ni], nc)
    G /= np.where(np.abs(V) > 1e-300, V, 1e-300)[:, None, None]
    return G / SCALE                                             # 1/mm -> 1/m


def q_criterion(mesh, fields) -> np.ndarray:
    """Q = (|Omega|^2 - |S|^2) / 2 = -tr(G G) / 2 per cell, 1/s2: positive where rotation beats strain."""
    G = cell_gradient(mesh, fields["U"], fields["patches"])
    return -0.5 * np.einsum("nij,nji->n", G, G)


def iso_values(study, mesh, plan, fields, field: str) -> np.ndarray:
    from .render import _pa, _q_ref
    if field == "q":
        return q_criterion(mesh, fields)
    U = np.asarray(fields["U"], float)
    if field == "u":
        return np.linalg.norm(U, axis=1)
    qd, pr = _q_ref(study, plan)                                  # cp_total
    en = plan.get("energy")
    rho = np.asarray(fields["rho"], float) if en and fields.get("rho") is not None else float(plan["rho"])
    return (_pa(plan, fields["p"]) + 0.5 * rho * np.einsum("ij,ij->i", U, U) - pr) / qd


def color_values(study, plan, fields, color: str) -> np.ndarray | None:
    from .render import _pa, _q_ref
    if color in (None, "none"):
        return None
    if color == "u":
        return np.linalg.norm(np.asarray(fields["U"], float), axis=1)
    p = _pa(plan, fields["p"])
    if color == "cp":
        qd, pr = _q_ref(study, plan)
        return (p - pr) / qd
    return p


def part_box(mesh) -> tuple[np.ndarray, np.ndarray] | None:
    ptags = [t for t, i in mesh.tag_info.items() if i.get("on") == "part"]
    pf = mesh.faces[mesh.ninternal:][np.isin(mesh.btag, ptags)]
    if not len(pf):
        return None
    pp = mesh.points[np.unique(pf[pf >= 0])]
    return pp.min(axis=0), pp.max(axis=0)


def default_window(study, mesh) -> list[float]:
    """External flow: from a quarter of the part's length ahead of it to 1.5 lengths behind (downstream: the
    drag direction, else +x), 0.4 lengths to the sides; otherwise the whole mesh. Clipped to the mesh."""
    lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
    pb = part_box(mesh) if (study.get("domain") or {}).get("kind") == "external" else None
    if pb is None:
        return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])]
    plo, phi = pb
    L = float(np.max(phi - plo))
    d = np.asarray((study.get("reference") or {}).get("drag_dir") or [1, 0, 0], float)
    a = int(np.argmax(np.abs(d)))
    wlo, whi = plo - 0.4 * L, phi + 0.4 * L
    if d[a] >= 0:
        wlo[a], whi[a] = plo[a] - 0.25 * L, phi[a] + 1.5 * L
    else:
        wlo[a], whi[a] = plo[a] - 1.5 * L, phi[a] + 0.25 * L
    wlo, whi = np.maximum(wlo, lo), np.minimum(whi, hi)
    return [float(v) for pair in zip(wlo, whi) for v in pair]


def symmetry_planes(study) -> list[tuple[int, float]]:
    """(axis, coordinate) of each symmetry bc on a side of an external domain box: a half model."""
    dom = study.get("domain") or {}
    box = dom.get("box")
    if dom.get("kind") != "external" or not box:
        return []
    out = []
    for b in study.get("bcs", []):
        if b.get("kind") != "symmetry":
            continue
        sides = (b.get("faces") or {}).get("box")
        for s in ([sides] if isinstance(sides, str) else sides or []):
            if s in ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax"):
                a = "xyz".index(s[0])
                out.append((a, float(box[a] if s.endswith("min") else box[a + 3])))
    return out


def grid_values(C: np.ndarray, h: np.ndarray, vals: list, window, budget: float, below: list):
    """Cell values spread on a uniform grid over window [x0, x1, y0, y1, z0, z1] by inverse-distance weights
    from the 8 nearest cell centres. Grid points farther than a cell from every centre (solids) take
    below[k]. Returns (origin, spacing, dims, [arrays in x-fastest order])."""
    from scipy.spatial import cKDTree
    w = np.asarray(window, float).reshape(3, 2)
    span = np.maximum(w[:, 1] - w[:, 0], 1e-9)
    near = np.all((C >= w[:, 0] - 2 * h[:, None]) & (C <= w[:, 1] + 2 * h[:, None]), axis=1)
    if near.sum() < 8:
        raise ValueError("no cells in the window")
    Cn, hn = C[near], h[near]
    dx = max(float(np.cbrt(np.prod(span) / budget)), 0.5 * float(np.min(hn)))
    dims = np.maximum(np.round(span / dx).astype(int) + 1, 2)
    axes = [np.linspace(w[k, 0], w[k, 1], dims[k]) for k in range(3)]
    sp = [(axes[k][1] - axes[k][0]) for k in range(3)]
    tree = cKDTree(Cn)
    out = [np.empty(int(np.prod(dims))) for _ in vals]
    zz, yy = np.meshgrid(axes[2], axes[1], indexing="ij")         # x fastest: index = i + nx (j + ny k)
    yz = np.column_stack([yy.ravel(), zz.ravel()])
    nx = dims[0]
    step = max(1, int(1_000_000 // nx))
    for a in range(0, len(yz), step):
        b = min(a + step, len(yz))
        pts = np.column_stack([np.tile(axes[0], b - a), np.repeat(yz[a:b, 0], nx), np.repeat(yz[a:b, 1], nx)])
        dist, idx = tree.query(pts, k=min(8, len(Cn)), workers=-1)
        wt = 1.0 / np.maximum(dist, 1e-9 * dx) ** 2
        wt /= wt.sum(axis=1, keepdims=True)
        solid = dist[:, 0] > hn[idx[:, 0]]
        for k, v in enumerate(vals):
            vn = v[near]
            r = (wt * vn[idx]).sum(axis=1)
            r[solid] = below[k]
            out[k][a * nx:b * nx] = r
    return w[:, 0], sp, dims, out


def idw_at(C: np.ndarray, v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(C).query(pts, k=min(8, len(C)), workers=-1)
    wt = 1.0 / np.maximum(dist, 1e-12) ** 2
    return (wt * v[idx]).sum(axis=1) / wt.sum(axis=1)


# ---- in the child: drawing ----------------------------------------------------------------------
def _polydata(mesh, faces_idx: np.ndarray):
    import pyvista as pv
    rows = mesh.faces[faces_idx]
    cnt = (rows >= 0).sum(axis=1)
    order = np.argsort(cnt, kind="stable")
    rows, cnt = rows[order], cnt[order]
    used = np.unique(rows[rows >= 0])
    conn = []
    for c in np.unique(cnt):
        r = np.searchsorted(used, rows[cnt == c][:, :c])
        conn.append(np.column_stack([np.full(len(r), c), r]).ravel())
    return pv.PolyData(mesh.points[used], np.concatenate(conn)), order


def _mirrored(m, planes):
    for a, c in planes:
        n = [0.0, 0.0, 0.0]
        n[a] = 1.0
        pt = [0.0, 0.0, 0.0]
        pt[a] = c
        m = m.merge(m.reflect(n, point=pt))
    return m


def _camera(p, view, bounds):
    elev, azim = np.radians(view[0]), np.radians(view[1])
    b = np.asarray(bounds, float).reshape(3, 2)
    c = b.mean(axis=1)
    d = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    p.camera.focal_point = c
    p.camera.position = c + d * float(np.linalg.norm(b[:, 1] - b[:, 0])) * 2
    p.camera.up = (0, 0, 1) if abs(np.sin(elev)) < 0.99 else (0, 1, 0)
    p.reset_camera(bounds=b.ravel())
    p.camera.zoom(1.15)


def _plotter(title):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=SIZE)
    p.set_background("white")
    p.add_text(title, position="upper_left", font_size=11, color="black")
    try:
        p.enable_anti_aliasing("fxaa")
    except Exception:  # noqa: BLE001
        pass
    return p


def _bar(label):
    return {"title": label, "color": "black", "vertical": True, "position_x": 0.88, "position_y": 0.15,
            "height": 0.7, "width": 0.05, "title_font_size": 14, "label_font_size": 12, "fmt": "%.3g"}


def draw_surface(study, mesh, plan, fields, spec, title, path) -> dict:
    from .render import CMAPS, FIELDS, _limits, wall_values
    field = spec.get("field", "p")
    f, vals = wall_values(study, mesh, plan, fields, spec)
    m, order = _polydata(mesh, f)
    m.cell_data["v"] = vals[order]
    m = m.cell_data_to_point_data(pass_cell_data=False)
    planes = symmetry_planes(study) if spec.get("mirror", True) else []
    if planes:
        m = _mirrored(m, planes)
    lo, hi, _ = _limits(vals, spec, "robust")
    p = _plotter(title)
    p.add_mesh(m, scalars="v", cmap=CMAPS.get(field, "viridis"), clim=[lo, hi], smooth_shading=True,
               split_sharp_edges=True, specular=0.3, scalar_bar_args=_bar(FIELDS.get(field, field)))
    _camera(p, spec.get("view", [25, -60]), m.bounds)
    p.screenshot(str(path))
    p.close()
    return {"faces": int(len(f)), "min": float(vals.min()), "max": float(vals.max()), "clim": [lo, hi],
            "renderer": "vtk", **({"mirrored": ["xyz"[a] for a, _ in planes]} if planes else {})}


def draw_iso(study, mesh, plan, fields, spec, title, path) -> dict:
    import pyvista as pv

    from .render import CMAPS, _limits, _cell_size
    iso = spec["iso"]
    field = iso.get("field", "q")
    vals = iso_values(study, mesh, plan, fields, field)
    win = spec.get("window") or default_window(study, mesh)
    g = mesh.geometry()
    C, h, V = g["C"], _cell_size(mesh, False), g["V"]
    inw = np.all((C >= np.asarray(win[0::2])) & (C <= np.asarray(win[1::2])), axis=1)
    value, auto = iso.get("value", "auto"), False
    if value == "auto":
        auto = True
        if field == "q":                     # the volume-weighted 90th percentile of the positive Q in the window
            pos = inw & (vals > 0)
            if not pos.any():
                raise ValueError("no positive Q in the window")
            o = np.argsort(vals[pos])
            cw = np.cumsum(V[pos][o])
            value = float(vals[pos][o][np.searchsorted(cw, 0.9 * cw[-1])])
        elif field == "u":
            ref = (study.get("reference") or {}).get("velocity")
            value = 0.5 * float(ref) if ref else 0.5 * float(np.max(vals[inw]))
        else:
            value = 0.0
    value = float(value)
    below = value - 1.0 - abs(value)                 # solids: below the surface's value (Cp_total, |U|: inside)
    col = spec.get("color", "u")
    cv = color_values(study, plan, fields, col)
    budget = float(os.environ.get("CFDT_ISO_VOXELS", "6e6"))
    origin, sp, dims, arrs = grid_values(C, h, [vals], win, budget, [below])
    grid = pv.ImageData(dimensions=tuple(int(d) for d in dims), spacing=tuple(sp), origin=tuple(origin))
    grid.point_data["s"] = arrs[0]
    surf = grid.contour([value], scalars="s", method="flying_edges")
    if surf.n_points == 0:
        raise ValueError(f"no {field} = {value:g} surface in the window (values there: "
                         f"{float(np.min(vals[inw])):.4g} to {float(np.max(vals[inw])):.4g})")
    try:
        surf = surf.smooth_taubin(n_iter=15, pass_band=0.1)
    except Exception:  # noqa: BLE001
        pass
    clim = None
    if cv is not None:
        sel = np.all((C >= np.asarray(win[0::2]) - 2 * h[:, None]) & (C <= np.asarray(win[1::2]) + 2 * h[:, None]), axis=1)
        surf.point_data["c"] = idw_at(C[sel], cv[sel], np.asarray(surf.points))
        lo, hi, _ = _limits(np.asarray(surf.point_data["c"]), spec, "robust")
        clim = [lo, hi]
    planes = symmetry_planes(study) if spec.get("mirror", True) else []
    walls = None
    try:
        from .render import wall_values
        ext = (study.get("domain") or {}).get("kind") == "external"
        wf, _ = wall_values(study, mesh, plan, fields, {"surface": "part" if ext else "walls", "field": "p"})
        walls, _o = _polydata(mesh, wf)
    except ValueError:
        pass
    if planes:
        surf = _mirrored(surf, planes)
        walls = _mirrored(walls, planes) if walls is not None else None
    p = _plotter(title)
    if walls is not None:
        ext = (study.get("domain") or {}).get("kind") == "external"
        p.add_mesh(walls, color="#c8c8c8", smooth_shading=True, split_sharp_edges=True, specular=0.2,
                   opacity=1.0 if ext else 0.18)
    if clim:
        p.add_mesh(surf, scalars="c", cmap=CMAPS.get(col, "viridis"), clim=clim, smooth_shading=True, specular=0.3,
                   scalar_bar_args=_bar(COLORS[col]))
    else:
        p.add_mesh(surf, color="#4f8fc0", smooth_shading=True, specular=0.3)
    b = np.array(surf.bounds).reshape(3, 2)
    if walls is not None:
        wb = np.array(walls.bounds).reshape(3, 2)
        b = np.column_stack([np.minimum(b[:, 0], wb[:, 0]), np.maximum(b[:, 1], wb[:, 1])])
    _camera(p, spec.get("view", [25, -60]), b.ravel())
    p.screenshot(str(path))
    p.close()
    return {"field": field, "value": float(f"{value:.6g}"),
            "value_auto": auto, "window": [round(x, 3) for x in win], "grid": [int(d) for d in dims],
            "grid_mm": round(float(sp[0]), 4), "triangles": int(surf.n_cells), "renderer": "vtk",
            **({"color": col, "clim": clim} if clim else {}),
            **({"mirrored": ["xyz"[a] for a, _ in planes]} if planes else {})}


def _child(state_dir: str, spec_file: str, out_dir: str, name: str) -> dict:
    from .render import FIELDS, load_state
    mesh, plan, fields, study = load_state(Path(state_dir))
    specs = json.loads(Path(spec_file).read_text())
    images, warns = {}, []
    for sp in specs:
        nm = sp["name"]
        path = Path(out_dir) / f"{nm}.png"
        try:
            if "iso" in sp:
                f = sp["iso"].get("field", "q")
                what = f"{name}: {ISO_FIELDS[f]} isosurface" + (f", coloured by {COLORS[sp.get('color', 'u')]}"
                                                                 if sp.get("color", "u") != "none" else "")
                info = draw_iso(study, mesh, plan, fields, sp, what, path)
            else:
                f = sp.get("field", "p")
                what = f"{name}: {FIELDS[f]} on {sp.get('surface', 'walls')}"
                info = draw_surface(study, mesh, plan, fields, sp, what, path)
            images[nm] = {"path": str(path), "what": what, **info}
        except Exception as exc:  # noqa: BLE001 (a picture must not fail the study)
            warns.append(f"image {nm!r}: {type(exc).__name__}: {exc}")
    return {"images": images, "warnings": warns}


if __name__ == "__main__":
    print(json.dumps(_child(*sys.argv[1:5]), default=float))
