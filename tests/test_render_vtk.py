"""3D pictures with VTK (render_vtk.py): Q-criterion, the isosurface grid, half-model mirroring, validation, and
(where VTK can draw offscreen) the images themselves and the fallback to Matplotlib."""
from __future__ import annotations

import numpy as np
import pytest

from cfd_translator import render_vtk as RV
from cfd_translator.format import image_errors
from cfd_translator.polymesh import box_mesh
from cfd_translator.render import load_state, render, save_state


def _box(n=6, field=None):
    m = box_mesh(60, 30, 30, 2 * n, n, n)
    m.patches = [{"name": "walls", "type": "wall", "start": m.ninternal, "n": m.nboundary, "tags": m.tags()}]
    g = m.geometry()
    U = field(g["C"])
    fields = {"U": U, "p": np.zeros(m.ncells),
              "patches": {"walls": {"U": field(g["Cf"][m.ninternal:]), "p": np.linspace(0, 1, m.nboundary)}}}
    return m, fields


def test_q_criterion_is_exact_for_linear_fields():
    w = 40.0                                                     # 1/s; coordinates are mm, so r in m = x / 1000

    def rotation(X):
        U = np.zeros_like(X)
        U[:, 0], U[:, 1] = -w * X[:, 1] / 1000, w * X[:, 0] / 1000
        return U
    m, f = _box(field=rotation)
    assert RV.q_criterion(m, f) == pytest.approx(np.full(m.ncells, w ** 2), rel=1e-9)
    m, f = _box(field=lambda X: np.column_stack([w * X[:, 0], -w * X[:, 1], 0 * X[:, 2]]) / 1000)   # pure strain
    assert RV.q_criterion(m, f) == pytest.approx(np.full(m.ncells, -w ** 2), rel=1e-9)
    m, f = _box(field=lambda X: np.column_stack([w * X[:, 1], 0 * X[:, 1], 0 * X[:, 2]]) / 1000)    # simple shear
    assert np.abs(RV.q_criterion(m, f)).max() < 1e-6 * w ** 2


def test_grid_values_reproduce_cells_and_blank_solids():
    m, f = _box(n=4, field=lambda X: X / 1000)
    g = m.geometry()
    C, h = g["C"], np.cbrt(g["V"])
    v = C[:, 0] + 2 * C[:, 1]
    # a grid through the cell centres: every grid point is a centre, so the values come back exactly
    win = [C[:, 0].min(), C[:, 0].max(), C[:, 1].min(), C[:, 1].max(), C[:, 2].min(), C[:, 2].max()]
    origin, sp, dims, (a,) = RV.grid_values(C, h, [v], win, 52.5 * 22.5 * 22.5 / 7.5 ** 3 + 1e-9, [-99])
    assert list(dims) == [8, 4, 4] and sp[0] == pytest.approx(7.5)
    X = np.stack(np.meshgrid(*[origin[k] + sp[k] * np.arange(dims[k]) for k in range(3)], indexing="ij"), -1)
    assert a == pytest.approx((X[..., 0] + 2 * X[..., 1]).transpose(2, 1, 0).ravel(), abs=1e-9)
    # without the cells of one half (a solid there), its grid points take the 'below' value
    keep = C[:, 0] < 0                                          # box_mesh centres x and y on 0
    _, _, _, (b,) = RV.grid_values(C[keep], h[keep], [v[keep]], [-30, 30, -15, 15, 0, 30], 2000, [-99])
    b = b.reshape(-1)
    assert (b == -99).mean() == pytest.approx(0.4, abs=0.15) and (b != -99).any()


def test_half_models_windows_and_validation():
    st = {"domain": {"kind": "external", "box": [-200, 0, 0, 600, 150, 150]}, "reference": {"velocity": 30},
          "bcs": [{"kind": "symmetry", "faces": {"box": "ymin"}}, {"kind": "slip", "faces": {"box": ["zmax", "ymax"]}}]}
    assert RV.symmetry_planes(st) == [(1, 0.0)]
    assert RV.symmetry_planes(dict(st, bcs=[{"kind": "symmetry", "faces": {"box": ["ymin", "zmin"]}}])) == [(1, 0.0), (2, 0.0)]
    assert RV.symmetry_planes(dict(st, domain={"kind": "internal"})) == []
    m = box_mesh(800, 150, 150, 8, 3, 3, origin=(-200, 0, 0))
    for t, i in m.tag_info.items():                              # pretend the floor is the part
        if i.get("box") == "zmin":
            i["on"] = "part"
    w = RV.default_window(st, m)
    assert w[0] == -200 and w[1] == 600 and w[2:4] == [0, 150]    # clipped to the mesh
    ok = [{"name": "q", "iso": {"field": "q", "value": "auto"}, "color": "cp", "window": [0, 1, 0, 1, 0, 1], "view": [20, -120]},
          {"name": "w", "surface": "part", "field": "cp", "quality": "high", "mirror": False}]
    assert image_errors(ok) == []
    bad = image_errors([{"iso": {"field": "lambda2"}}, {"iso": {"field": "q"}, "window": [0, 1, 0, 1]},
                        {"slice": {"axis": "z", "at": 0}, "quality": "high"}, {"surface": "walls", "color": "u"},
                        {"iso": {"field": "q"}, "field": "u"}, {"iso": {"field": "cp_total"}}], ref_velocity=False)
    assert len(bad) == 6, bad
    assert image_errors([{"iso": {"field": "q"}}], two_d=True) == ["images[0].iso: isosurfaces are for 3D studies"]


def _vortex_case(tmp_path):
    def vel(X):
        U = np.zeros((len(X), 3))
        U[:, 0] = 30.0
        dy, dz = (X[:, 1] - 15) * 1e-3, (X[:, 2] - 15) * 1e-3
        r2 = dy ** 2 + dz ** 2
        fct = 0.5 / (2 * np.pi * np.maximum(r2, 1e-12)) * (1 - np.exp(-r2 / 0.004 ** 2))
        U[:, 1], U[:, 2] = -fct * dz, fct * dy
        return U
    m, f = _box(n=10, field=vel)
    study = {"domain": {"kind": "internal"}, "reference": {"velocity": 30}, "bcs": [], "fluid": "air"}
    plan = {"rho": 1.2, "mu": 1.8e-5}
    save_state(tmp_path, m, plan, f, study)
    return m, f, study, plan


def test_vtk_images_or_their_fallback(tmp_path, monkeypatch):
    m, f, study, plan = _vortex_case(tmp_path)
    q = RV.q_criterion(m, f)
    assert q.max() > 0 and q[np.argmax(q)] > 100 * abs(np.median(q))      # the vortex core stands out
    specs = [{"name": "q", "iso": {"field": "q", "value": "auto"}},
             {"name": "hq", "surface": "walls", "field": "p", "quality": "high"},
             {"name": "plain", "surface": "walls", "field": "p"}]
    RV._PROBE.clear()
    if RV.probe().get("ok"):
        imgs, w = render(study, m, plan, f, tmp_path / "images", specs, "v")
        assert w == [] and set(imgs) == {"q", "hq", "plain"}
        assert imgs["q"]["renderer"] == "vtk" and imgs["q"]["value_auto"] and imgs["q"]["triangles"] > 50
        assert imgs["hq"]["renderer"] == "vtk" and "renderer" not in imgs["plain"]
        import matplotlib.image as mpimg
        px = mpimg.imread(imgs["q"]["path"])
        assert px.shape[:2] == (RV.SIZE[1], RV.SIZE[0]) and (px[..., :3] < 0.9).any()     # not a blank picture
    # without VTK: the high-quality surface is drawn with Matplotlib, the isosurface is skipped, both with warnings
    monkeypatch.setenv("CFDT_RENDER", "matplotlib")
    RV._PROBE.clear()
    imgs, w = render(study, m, plan, f, tmp_path / "images2", specs, "v")
    RV._PROBE.clear()
    assert set(imgs) == {"hq", "plain"} and "renderer" not in imgs["hq"]
    assert any("'q'" in x and "CFDT_RENDER=matplotlib" in x for x in w) and any("'hq'" in x and "Matplotlib instead" in x for x in w)
    assert load_state(tmp_path)[0].ncells == m.ncells


def test_status_reports_the_renderer():
    s = RV.status()
    assert "vtk" in s and (s["vtk"] is None or "offscreen" in s)
