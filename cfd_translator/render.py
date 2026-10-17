"""render.py — pictures of a solved study: slices (velocity or pressure, with streamlines) and wall surfaces.
Version: 2026.10.17.1

    render(study, mesh, plan, fields, folder, specs) -> ({name: {path, what}}, warnings)
    default_images(study, mesh) -> specs (for "images": true)
    save_state(folder, mesh, plan, fields, study) / load_state(folder) -> (mesh, plan, fields, study): re-render without solving

A slice takes the cells whose centres lie within about half a cell of the plane (all cells in 2D), plus
the boundary faces there (walls at their wall velocity), triangulates them in the plane and drops the
triangles longer than about two local cells: solid parts show as holes. Streamlines follow the in-plane
velocity interpolated on that triangulation. Surfaces colour each wall face by pressure (Pa) or wall
shear (Pa). Units on the axes: mm. Matplotlib only (Agg), so it runs anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .polymesh import SCALE, PolyMesh

AXES = {"x": 0, "y": 1, "z": 2}
FIELDS = {"u": "|U| (m/s)", "ux": "Ux (m/s)", "uy": "Uy (m/s)", "uz": "Uz (m/s)", "p": "p (Pa)",
          "wall_shear": "wall shear (Pa)"}


def default_images(study: dict, mesh) -> list[dict]:
    """Two-dimensional: |U| with streamlines and p on the plane. 3D: |U| with streamlines on a plane through
    the middle (external: mid-height of the part, and a side view), and the pressure on the walls."""
    if study.get("2d"):
        zm = float(mesh.points[:, 2].mean())
        return [{"name": "velocity", "slice": {"axis": "z", "at": zm}, "field": "u", "streamlines": True},
                {"name": "pressure", "slice": {"axis": "z", "at": zm}, "field": "p"}]
    lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
    if study.get("domain", {}).get("kind") == "external":
        ptags = [t for t, i in mesh.tag_info.items() if i.get("on") == "part"]
        pf = mesh.faces[mesh.ninternal:][np.isin(mesh.btag, ptags)]
        pp = mesh.points[np.unique(pf[pf >= 0])] if len(pf) else mesh.points
        plo, phi_ = pp.min(axis=0), pp.max(axis=0)
        c = 0.5 * (plo + phi_)
        return [{"name": "velocity_top", "slice": {"axis": "z", "at": float(c[2])}, "field": "u", "streamlines": True},
                {"name": "velocity_side", "slice": {"axis": "y", "at": float(c[1])}, "field": "u", "streamlines": True},
                {"name": "pressure_surface", "surface": "part", "field": "p"}]
    c = 0.5 * (lo + hi)
    ax = "xyz"[int(np.argmin(hi - lo))]
    return [{"name": "velocity", "slice": {"axis": ax, "at": float(c[AXES[ax]])}, "field": "u", "streamlines": True},
            {"name": "pressure", "slice": {"axis": ax, "at": float(c[AXES[ax]])}, "field": "p"},
            {"name": "pressure_walls", "surface": "walls", "field": "p"}]


def _cell_size(mesh, two_d: bool) -> np.ndarray:
    V = mesh.geometry()["V"]
    if two_d:
        t = max(float(np.ptp(mesh.points[:, 2])), 1e-300)
        return np.sqrt(V / t)
    return np.cbrt(V)


def _slice(study, mesh, plan, fields, spec, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    two_d = bool(study.get("2d"))
    rho = plan["rho"]
    a = AXES[spec["slice"]["axis"]]
    if two_d and a != 2:
        raise ValueError("a 2D study has only the z slice")
    s = float(spec["slice"]["at"])
    i, j = [k for k in range(3) if k != a]
    g = mesh.geometry()
    h = _cell_size(mesh, two_d)
    C = g["C"]
    sel = np.ones(len(C), bool) if two_d else np.abs(C[:, a] - s) <= 0.6 * h
    if sel.sum() < 3:
        raise ValueError(f"no cells near {spec['slice']['axis']} = {s:g} mm")
    X = [C[sel][:, [i, j]]]
    U = [np.asarray(fields["U"], float)[sel]]
    P = [np.asarray(fields["p"], float)[sel] * rho]
    H = [h[sel]]
    for patch in mesh.patches:
        pf = fields["patches"].get(patch["name"])
        if patch["type"] == "empty" or pf is None or pf.get("U") is None:
            continue
        f = np.arange(patch["start"], patch["start"] + patch["n"])
        hf = h[mesh.owner[f]]
        m = np.ones(len(f), bool) if two_d else np.abs(g["Cf"][f, a] - s) <= 0.6 * hf
        if m.any():
            X.append(g["Cf"][f[m]][:, [i, j]])
            U.append(np.asarray(pf["U"], float).reshape(-1, 3)[m])
            P.append(np.asarray(pf["p"], float)[m] * rho)
            H.append(hf[m])
    X, U, P, H = (np.concatenate(v) for v in (X, U, P, H))
    X, idx = np.unique(np.round(X, 9), axis=0, return_index=True)
    U, P, H = U[idx], P[idx], H[idx]
    tri = mtri.Triangulation(X[:, 0], X[:, 1])
    e = X[tri.triangles]
    L = np.max(np.linalg.norm(e - np.roll(e, 1, axis=1), axis=2), axis=1)
    tri.set_mask(L > 2.0 * H[tri.triangles].max(axis=1))
    field = spec.get("field", "u")
    val = {"u": np.linalg.norm(U, axis=1), "ux": U[:, 0], "uy": U[:, 1], "uz": U[:, 2], "p": P}[field]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    win = spec.get("window")
    cmap = "coolwarm" if field == "p" else "viridis"
    tp = ax.tripcolor(tri, val, shading="gouraud", cmap=cmap)
    fig.colorbar(tp, ax=ax, label=FIELDS[field], shrink=0.85)
    if spec.get("streamlines"):
        x0, x1 = (win[0], win[1]) if win else (X[:, 0].min(), X[:, 0].max())
        y0, y1 = (win[2], win[3]) if win else (X[:, 1].min(), X[:, 1].max())
        n = 300
        nx, ny = (n, max(20, int(n * (y1 - y0) / (x1 - x0)))) if x1 - x0 >= y1 - y0 else (max(20, int(n * (x1 - x0) / (y1 - y0))), n)
        gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
        ui = mtri.LinearTriInterpolator(tri, U[:, i])(gx, gy)
        vi = mtri.LinearTriInterpolator(tri, U[:, j])(gx, gy)
        if np.ma.count(ui) > 10:
            ax.streamplot(gx, gy, ui.filled(np.nan), vi.filled(np.nan), density=1.6, color="white",
                          linewidth=0.6, arrowsize=0.7)
    if win:
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.set_xlabel(f"{'xyz'[i]} (mm)")
    ax.set_ylabel(f"{'xyz'[j]} (mm)")
    ax.set_title(title + ("" if two_d else f" — {'xyz'[a]} = {s:g} mm"))
    ax.set_facecolor("0.85")                       # solid parts (holes in the slice) show grey
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"cells": int(sel.sum()), "min": float(np.nanmin(val)), "max": float(np.nanmax(val))}


def _surface(study, mesh, plan, fields, spec, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rho = plan["rho"]
    which = spec.get("surface", "walls")
    ptags = {t for t, i in mesh.tag_info.items() if i.get("on") == "part"}
    polys, vals = [], []
    g = mesh.geometry()
    for patch in mesh.patches:
        if patch["type"] != "wall":
            continue
        if which not in ("walls", "part") and patch["name"] != which:
            continue
        f = np.arange(patch["start"], patch["start"] + patch["n"])
        if which == "part":
            f = f[np.isin(mesh.btag[f - mesh.ninternal], list(ptags))]
        if not len(f):
            continue
        pf = fields["patches"][patch["name"]]
        loc = f - patch["start"]
        if spec.get("field", "p") == "wall_shear":
            w = pf.get("wss")
            if w is not None:
                v = np.linalg.norm(np.asarray(w, float).reshape(-1, 3)[loc], axis=1) * rho
            else:                                      # our estimate: nu (U_f - U_P) deltaCoeff, tangential
                Sf = g["Sf"][f]
                n = Sf / np.linalg.norm(Sf, axis=1)[:, None]
                dU = np.asarray(pf["U"], float).reshape(-1, 3)[loc] - np.asarray(fields["U"], float)[mesh.owner[f]]
                dU -= np.einsum("ij,ij->i", dU, n)[:, None] * n
                v = rho * plan["nu"] * np.linalg.norm(dU, axis=1) * (mesh.delta_coeffs_boundary()[f - mesh.ninternal] / SCALE)
        else:
            v = np.asarray(pf["p"], float)[loc] * rho
        for r in mesh.faces[f]:
            polys.append(mesh.points[r[r >= 0]])
        vals.append(v)
    if not polys:
        raise ValueError(f"no wall faces for surface {which!r}")
    vals = np.concatenate(vals)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    hi_ = float(vals.max()) if vals.max() > vals.min() else float(vals.min()) + 1e-12
    norm = mcolors.Normalize(vmin=float(vals.min()), vmax=hi_)
    cmap = matplotlib.colormaps["coolwarm" if spec.get("field", "p") == "p" else "viridis"]
    fig = plt.figure(figsize=(9, 7), dpi=130)
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cmap(norm(vals)), edgecolors="none"))
    pts = np.concatenate(polys)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-9 * max(hi - lo)))
    elev, azim = spec.get("view", [25, -60])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=FIELDS[spec.get("field", "p")], shrink=0.7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"faces": len(polys), "min": float(vals.min()), "max": float(vals.max())}


def render(study, mesh, plan, fields, folder, specs, name="study") -> tuple[dict, list]:
    out, warns = {}, []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        return {}, ["images need matplotlib (pip install matplotlib)"]
    for k, sp in enumerate(specs):
        nm = sp.get("name") or f"image{k + 1}"
        path = folder / f"{nm}.png"
        field = sp.get("field", "u" if "slice" in sp else "p")
        sp = dict(sp, field=field)
        try:
            if "slice" in sp:
                what = f"{name}: {FIELDS[field]}" + (" with streamlines" if sp.get("streamlines") else "")
                info = _slice(study, mesh, plan, fields, sp, what, path)
            else:
                what = f"{name}: {FIELDS[field]} on {sp.get('surface', 'walls')}"
                info = _surface(study, mesh, plan, fields, sp, what, path)
            out[nm] = {"path": str(path), "what": what, **{k2: round(v, 6) if isinstance(v, float) else v for k2, v in info.items()}}
        except Exception as exc:  # noqa: BLE001 (a picture must not fail the study)
            warns.append(f"image {nm!r}: {type(exc).__name__}: {exc}")
    return out, warns


# ---- saved state (re-render without solving) ---------------------------------------------------
def save_state(folder: Path, mesh, plan: dict, fields: dict, study: dict | None = None) -> None:
    folder = Path(folder)
    arr = {"points": mesh.points, "faces": mesh.faces, "owner": mesh.owner, "neighbour": mesh.neighbour,
           "btag": mesh.btag, "U": np.asarray(fields["U"], float), "p": np.asarray(fields["p"], float)}
    for nm, pf in fields["patches"].items():
        for k in ("U", "p", "phi", "wss"):
            if pf.get(k) is not None:
                arr[f"patch/{nm}/{k}"] = np.asarray(pf[k], float)
    for nm, cells in (mesh.zones or {}).items():
        arr[f"zone/{nm}"] = np.asarray(cells)
    np.savez_compressed(folder / "state.npz", **arr)
    meta = {"ncells": int(mesh.ncells), "patches": mesh.patches, "tag_info": {str(k): v for k, v in mesh.tag_info.items()},
            "info": {k: v for k, v in mesh.info.items() if isinstance(v, (int, float, str, list, bool))}, "plan": plan,
            "study": study or {}}
    (folder / "state.json").write_text(json.dumps(meta, default=float))


def load_state(folder: Path):
    folder = Path(folder)
    meta = json.loads((folder / "state.json").read_text())
    z = np.load(folder / "state.npz")
    mesh = PolyMesh(z["points"], z["faces"], z["owner"], z["neighbour"], z["btag"], meta["ncells"],
                    {int(k): v for k, v in meta["tag_info"].items()}, meta["patches"], dict(meta["info"]))
    mesh.zones = {k[5:]: z[k] for k in z.files if k.startswith("zone/")}
    patches = {}
    for p in meta["patches"]:
        rec = {k: (z[f"patch/{p['name']}/{k}"] if f"patch/{p['name']}/{k}" in z.files else None)
               for k in ("U", "p", "phi", "wss")}
        patches[p["name"]] = rec
    fields = {"U": z["U"], "p": z["p"], "patches": patches}
    return mesh, meta["plan"], fields, meta.get("study", {})
