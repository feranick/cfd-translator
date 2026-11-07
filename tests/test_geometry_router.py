"""Tests of 2026.11.06.1: mesh geometry (STL/OBJ/PLY) with groups, region selectors, several solids, goal files and
get_case refs, window averages, init_from, per-node queues, request names, image NaN guard, local cancel state.
pytest -q"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from cfd_translator import api, meshsurf, select, snappy
from cfd_translator.format import _step_solids, validate_study


def _box(lo, hi, n=1):
    """Triangles (T, 3, 3) of a closed box, outward, each side split n x n."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    tris = []
    for ax in range(3):
        u, v = [k for k in range(3) if k != ax]
        for side, sgn in ((lo[ax], -1), (hi[ax], 1)):
            g = np.linspace(0, 1, n + 1)
            for i in range(n):
                for j in range(n):
                    def P(a, b):
                        p = np.zeros(3)
                        p[ax] = side
                        p[u] = lo[u] + (hi[u] - lo[u]) * a
                        p[v] = lo[v] + (hi[v] - lo[v]) * b
                        return p
                    q = [P(g[i], g[j]), P(g[i + 1], g[j]), P(g[i + 1], g[j + 1]), P(g[i], g[j + 1])]
                    t1, t2 = [q[0], q[1], q[2]], [q[0], q[2], q[3]]
                    nrm = np.cross(q[1] - q[0], q[2] - q[0])
                    if nrm[ax] * sgn < 0:
                        t1, t2 = t1[::-1], t2[::-1]
                    tris += [t1, t2]
    return np.array(tris)


def _cyl_y(c, r, h, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P0 = np.c_[c[0] + r * np.cos(a), np.full(n, c[1]), c[2] + r * np.sin(a)]
    P1 = P0 + [0, h, 0]
    C0, C1 = np.asarray(c, float), np.asarray(c, float) + [0, h, 0]
    T = []
    for i in range(n):
        j = (i + 1) % n
        T += [[P0[i], P0[j], P1[j]], [P0[i], P1[j], P1[i]], [C0, P0[j], P0[i]], [C1, P1[i], P1[j]]]
    return np.array(T)


def _ascii_stl(path, solids):
    with open(path, "w") as f:
        for nm, T in solids:
            f.write(f"solid {nm}\n")
            for t in T:
                f.write(" facet normal 0 0 0\n  outer loop\n")
                for v in t:
                    f.write(f"   vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {nm}\n")
    return str(path)


def _binary_stl(path, T):
    with open(path, "wb") as f:
        f.write(b"solid but binary".ljust(80, b" "))
        f.write(struct.pack("<I", len(T)))
        for t in T:
            f.write(struct.pack("<3f", 0, 0, 0) + struct.pack("<9f", *np.asarray(t, float).ravel()) + b"\0\0")
    return str(path)


@pytest.fixture
def car(tmp_path):
    body = _box([0, 0, 0], [100, 50, 40], n=4)
    body[5] = body[5][::-1]                        # one triangle the wrong way round
    return _ascii_stl(tmp_path / "car.stl", [("body", body), ("wheel_front", _cyl_y([20, -15, 10], 10, 12)),
                                              ("wheel_rear", _cyl_y([80, -15, 10], 10, 12))])


def _study(stl, bcs, **kw):
    return {"geometry": {"stl": stl}, "domain": {"kind": "external", "box": [-100, -60, -30, 300, 120, 120]},
            "fluid": "air", "model": "kOmegaSST", "mesh": {"mesher": "snappy", "size": 20},
            "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": [10, 0, 0]},
                    {"kind": "outlet", "faces": {"box": "xmax"}}] + bcs, **kw}


# ---- reading and checking mesh geometry ----------------------------------------------------------------------------
def test_ascii_stl_groups_bodies_and_orientation(car):
    L = meshsurf.load({"stl": car})
    assert L["names"] == ["body", "wheel_front", "wheel_rear"] and L["bodies"] == 3
    assert all(v > 0 for v in L["body_volumes_mm3"])                 # every body turned outward
    assert L["body_volumes_mm3"][0] == pytest.approx(100 * 50 * 40)
    assert L["flipped"] >= 1
    assert meshsurf.inside(L["points"], L["tris"], [[50, 25, 20], [150, 0, 0], [20, -10, 10]]).tolist() == \
        [True, False, True]
    st = meshsurf.surface_stats(car)
    assert st["bodies"] == 3 and st["open_edges"] == 0 and st["groups"][1] == "wheel_front"


def test_binary_stl_obj_and_ply_with_groups(tmp_path):
    T = _box([0, 0, 0], [10, 20, 30])
    b = meshsurf.load({"stl": _binary_stl(tmp_path / "b.stl", T)})
    assert b["names"] == ["part"] and b["body_volumes_mm3"][0] == pytest.approx(6000, rel=1e-5)
    # OBJ: shared vertices, two groups, a quad
    obj = tmp_path / "m.obj"
    V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    F = {"bottom": [[1, 4, 3, 2]], "rest": [[5, 6, 7, 8], [1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8]]}
    obj.write_text("\n".join(f"v {x} {y} {z}" for x, y, z in V) + "\n" +
                   "".join(f"g {g}\n" + "".join("f " + " ".join(map(str, f)) + "\n" for f in fs) for g, fs in F.items()))
    o = meshsurf.load({"stl": str(obj), "scale": 10})
    assert o["names"] == ["bottom", "rest"] and o["body_volumes_mm3"][0] == pytest.approx(1000)
    assert int((o["group"] == 0).sum()) == 2
    # PLY (ascii) with a per-face label
    ply = tmp_path / "m.ply"
    faces = [f for fs in F.values() for f in fs]
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\n"
                   "element face 6\nproperty list uchar int vertex_indices\nproperty int label\nend_header\n" +
                   "".join(f"{x} {y} {z}\n" for x, y, z in V) +
                   "".join(f"4 {' '.join(str(i - 1) for i in f)} {0 if k == 0 else 1}\n" for k, f in enumerate(faces)))
    p = meshsurf.load({"stl": str(ply)})
    assert p["names"] == ["g0", "g1"] and p["body_volumes_mm3"][0] == pytest.approx(1)
    # groups given per triangle
    g = meshsurf.load({"stl": str(ply), "groups": {"lid": [0, 1]}})
    assert "lid" in g["names"] and int((g["group"] == g["names"].index("lid")).sum()) == 2


def test_open_mesh_is_rejected(tmp_path):
    T = _box([0, 0, 0], [10, 10, 10])[:-1]
    f = _ascii_stl(tmp_path / "open.stl", [("x", T)])
    with pytest.raises(meshsurf.SurfaceError, match="not a closed surface"):
        meshsurf.load({"stl": f})
    e = validate_study(_study(f, []))["errors"]
    assert any("not a closed surface" in x for x in e)


def test_stl_validation(car):
    assert validate_study(_study(car, [{"kind": "wall", "faces": {"group": "wheel_front"}}]))["ok"]
    no_snappy = _study(car, [])
    no_snappy["mesh"] = {"size": 20}
    assert any("mesher 'snappy'" in x for x in validate_study(no_snappy)["errors"])
    internal = dict(_study(car, []), domain={"kind": "internal"}, bcs=[{"kind": "inlet", "faces": {"nx": -1},
                                                                         "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}])
    assert any("separate bodies" in x for x in validate_study(internal)["errors"])
    bad = [{"in_box": [0, 0, 0, -1, 1, 1]}, {"in_cyl": {"axis": [0, 0, 0], "r": 1}}, {"group": 3}, {"solid": -1},
           {"in_cyl": {"axis": [0, 1, 0], "r": 5, "y": [3, 1]}}, {"group": "a", "at": [0, 0, 0]}]
    for sel in bad:
        assert not validate_study(_study(car, [{"kind": "wall", "faces": sel}]))["ok"], sel


# ---- faces, selectors and splitting ---------------------------------------------------------------------------------
def _groups(st):
    pts, surf, ipt, _ = snappy.surface_premesh(st, "/tmp", 10, 1, st["domain"]["box"])
    meta = snappy.surface_premesh.meta
    surf, info, notes = select.split_regions(pts, surf, {t: {"on": "part", **meta.get(t, {})} for t, _ in surf}, st)
    tags = np.concatenate([np.full(len(t), tg) for tg, t in surf])
    rings = np.concatenate([t[:, ::-1] for _, t in surf])
    return select.Groups(pts, rings, tags, info, True), surf, info, notes, ipt


def test_group_solid_and_region_selectors(car):
    reg = {"in_box": [-1, -1, -1, 30.5, 51, 41]}               # the front of the body: splits its long sides
    st = _study(car, [{"kind": "wall", "name": "front", "faces": reg}])
    G, surf, info, notes, ipt = _groups(st)
    assert notes and "split" in notes[0]
    wf, wr = G.select({"group": "wheel_front"}), G.select({"group": "wheel_rear"})
    assert wf and wr and not set(wf) & set(wr)
    assert set(G.select({"solid": 1})) == set(wf)
    assert len(G.select({"cyl_r": 10})) == 2                   # both treads (faceted, 48 sides)
    front = G.select(reg)
    C = np.concatenate([G.groups[t].Cf for t in front])
    assert (C[:, 0] <= 30.5).all() and len(front) >= 5        # exactly the triangles inside: x0 face + 4 sides' parts
    wheel = G.select({"in_cyl": {"axis": [0, 1, 0], "center": [80, -15, 10], "r": 11, "y": [-16, -2]}})
    assert set(wheel) == set(wr)
    with pytest.raises(Exception, match="no face group"):
        G.select({"group": "spoiler"})
    # assign_bcs with them: no face taken twice
    st2 = _study(car, [{"kind": "wall", "name": "wf", "faces": {"group": "wheel_front"}},
                       {"kind": "wall", "name": "wr", "faces": {"in_cyl": {"axis": [0, 1, 0], "center": [80, -15, 10],
                                                                           "r": 11, "span": [-1, 13]}}},
                       {"kind": "wall", "name": "front", "faces": reg}])
    G2, *_ = _groups(st2)
    st2["bcs"] = st2["bcs"][2:]                                # the box sides are not in this surface-only test
    patches, sel, warns = select.assign_bcs(G2, st2)
    tags = {p["name"]: set(p["tags"]) for p in patches}
    assert list(tags)[:3] == ["wf", "wr", "front"] and "walls" in tags
    assert not (tags["wf"] & tags["front"]) and tags["wr"] == set(G2.select({"group": "wheel_rear"}))


def test_stl_inside_point_internal(tmp_path):
    f = _ascii_stl(tmp_path / "duct.stl", [("duct", _box([0, 0, 0], [200, 20, 20], n=2))])
    st = {"geometry": {"stl": f}, "fluid": "water", "mesh": {"mesher": "snappy", "size": 4},
          "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}]}
    pts, surf, ipt, bb = snappy.surface_premesh(st, tmp_path, 4, 1, None)
    assert ipt is not None and (ipt > 0).all() and ipt[0] < 200 and ipt[1] < 20
    assert snappy.surface_premesh.stl["faces"] == 6


def test_step_solid_count(tmp_path):
    p = tmp_path / "x.step"
    body = "ISO-10303-21;\n" + "#1=MANIFOLD_SOLID_BREP('a',#2);\n" + "x" * (1 << 22) + "#3=MANIFOLD_SOLID_BREP('b',#4);\n"
    p.write_text(body)
    assert _step_solids(str(p)) == 2
    st = {"geometry": {"step": str(p)}, "fluid": "water", "mesh": {"size": 2},
          "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}]}
    assert any("fuse" in e for e in validate_study(st)["errors"])


# ---- registration, refs, goal files -------------------------------------------------------------------------------
def test_add_geometry_stl_and_goal_file(tmp_path, monkeypatch, car):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    r = api.add_geometry("rogue", stl=car)
    assert r["ok"] and r["bodies"] == 3 and r["groups"] == ["body", "wheel_front", "wheel_rear"] and "note" in r
    from cfd_translator.variants import resolve
    st, err = resolve({"geometry": {"ref": "rogue"}})
    assert err is None and st["geometry"]["stl"].endswith("rogue.stl")
    gf = tmp_path / "goal.json"
    gf.write_text(json.dumps({"base": {"features": [{"type": "box", "size": [10, 10, 10]}]}}))
    from cfd_translator.variants import _goal_from_text
    g, e = _goal_from_text(str(gf))
    assert e is None and g["features"][0]["type"] == "box"
    assert _goal_from_text(str(tmp_path / "missing.json"))[1].startswith("goal: no file")
    assert "no build123d" in _goal_from_text('{"a": 1}')[1]
    bad = api.add_geometry("x", stl=str(tmp_path / "missing.stl"))
    assert not bad["ok"]


def test_get_case_resolves_a_ref(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    r = api.case({"geometry": {"ref": "nothing"}, "fluid": "water", "bcs": []})
    assert r["error"] == "invalid study" and "no geometry 'nothing'" in r["errors"][0]


# ---- run handling ---------------------------------------------------------------------------------------------------
COUETTE = {"2d": True, "geometry": {"box": [1.0, 0.1, 0.01]}, "fluid": {"rho": 1000.0, "mu": 1e-3},
           "bcs": [{"kind": "wall", "name": "lid", "faces": {"ny": 1}, "velocity": [0.01, 0, 0]},
                   {"kind": "wall", "name": "bottom", "faces": {"ny": -1}},
                   {"kind": "opening", "name": "left", "faces": {"nx": -1}},
                   {"kind": "opening", "name": "right", "faces": {"nx": 1}}],
           "mesh": {"cells": [20, 10, 1]}}


def test_request_name_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    monkeypatch.delenv("CFDT_EXECUTOR", raising=False)
    r = api.run({"c": COUETTE}, "ref", wait=30, request_name="run_router_abc123")
    assert r["request"].endswith("run_router_abc123") and r["done"]
    again = api.run({"c": COUETTE}, "ref", wait=1, request_name="run_router_abc123")
    assert "submitted before" in again["note"] and again["studies"]["c"]["ok"]
    assert "error" in api.run({"c": COUETTE}, "ref", request_name="../evil")


def test_local_cancel_flag_reads_cancelled(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    req = tmp_path / "out" / "run_x"
    (req / "c").mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"names": ["c"], "solver": "ref", "studies": {"c": COUETTE}}))
    (req / "c" / "cancel").write_text("1")
    s = api.status(str(req))
    assert s["studies"]["c"]["state"] == "cancelled"


def test_image_limits_ignore_nan():
    from cfd_translator.render import _limits
    lo, hi, _ = _limits(np.array([1.0, np.nan, 3.0, np.inf]), {}, "full")
    assert (lo, hi) == (1.0, 3.0)
    with pytest.raises(ValueError, match="no finite values"):
        _limits(np.array([np.nan, np.nan]), {}, "robust")


# ---- window averages -------------------------------------------------------------------------------------------------
def _write(path: Path, header: str, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Forces\n#\n" + header + "\n" + "\n".join(rows) + "\n")


def _window_run(tmp_path, porosity=True):
    """A finished DES-like run: adaptive time steps (dt grows 3x over the run), drag with a 0.65 Hz wobble and a step
    at 2 s, a porous part written kinematic (rhoRef 1) as OpenFOAM does in an incompressible case; the result's
    averages made by average_results itself, as the runner does."""
    from cfd_translator.average import average_results
    folder = tmp_path / "s"
    rng = np.random.default_rng(1)
    dt = np.linspace(4e-5, 1.2e-4, 60000)
    t = np.cumsum(dt)
    t = t[t <= 6.0]
    rho = 1.2
    Fx = 270 + 6 * np.sin(2 * np.pi * 0.65 * t) + 4 * (t > 2) + rng.normal(0, 2, len(t))
    Fz = 60 * np.sin(2 * np.pi * 0.65 * t + 1) + rng.normal(0, 5, len(t))
    por = 21.0 / rho                                          # N / rho: kinematic, as OpenFOAM writes it
    hdr = "# Time\tpressure_x\tpressure_y\tpressure_z\tviscous_x\tviscous_y\tviscous_z\tporous_x\tporous_y\tporous_z"
    _write(folder / "case" / "postProcessing" / "cfdtF_body" / "0" / "forces.dat", hdr,
           [f"{a:.9g}\t({0.9 * x:.9g} 0 {z:.9g})\t({0.1 * x:.9g} 0 0)\t({por:.9g} 0 0)" for a, x, z in zip(t, Fx, Fz)])
    _write(folder / "case" / "postProcessing" / "cfdtFp_wheel" / "0" / "forces.dat", "# Time\ttotal_x\ttotal_y\ttotal_z",
           [f"{a:.9g}\t({0.1 * x:.9g} 0 0)" for a, x in zip(t, Fx)])
    q, A = 0.5 * rho * 30 ** 2, 1.4
    plan = {"rho": rho, "patches": {"walls": {"kind": "wall"}},
            "monitors": {"pressure": {}, "torque": {},
                         "forces": {"fo": "cfdtF_body", "patches": ["walls"], "porosity": porosity},
                         "settings": {"transient": True, "from": 1.0, "min_samples": 20, "drift": 0.02,
                                      "stderr": 0.01}}}
    m = t >= 1.0
    Ftot = Fx + 21.0
    res = {"coefficients": {"cd": float(Ftot[m].mean() / (q * A)), "cl": 0.0, "drag_dir": [1, 0, 0],
                            "lift_dir": [0, 0, 1], "q_Pa": q, "area_m2": A},
           "forces": {"part": {"fx": float(Ftot[m].mean()), "porous_included": ["radiator"]},
                      "porous": {"radiator": {"fx": 21.0, "fy": 0.0, "fz": 0.0}}}}
    avg, w = average_results(folder / "case", plan, res)
    res["averaged"] = avg
    (folder / "s.json").write_text(json.dumps({"ok": True, "results": res}, default=float))
    (folder / "state.json").write_text(json.dumps({"plan": plan}))
    return folder, t, Fx + 21.0, q * A


def test_window_over_the_run_reproduces_its_averages(tmp_path):
    from cfd_translator.average import window_average
    folder, t, F, qa = _window_run(tmp_path)
    av = json.loads((folder / "s.json").read_text())["results"]["averaged"]
    w = window_average(folder, "s", {}, 1.0, None)
    for k in ("cd", "cl"):
        assert w[k] == av["quantities"][k], k                 # exactly: same series, weights, window and period
    m = t >= 1.0
    tw = np.gradient(t[m])
    assert w["force_N"]["fx"]["mean"] == pytest.approx((F[m] * tw).sum() / tw.sum(), rel=3e-6)   # porous in N
    assert w["frequency_Hz"] == av["frequency"]["frequency_Hz"]
    assert "time-weighted" in w["weighting"]


def test_window_average_sub_windows(tmp_path):
    from cfd_translator.average import window_average
    folder, t, F, qa = _window_run(tmp_path)
    w = window_average(folder, "s", {}, 4.0, 6.0)
    assert w["window_s"] == [4.0, 6.0] and w["samples_s"][0] == pytest.approx(4.0, abs=2e-4)
    m = (t >= 4.0) & (t <= 6.0)
    assert w["time_steps"] == int(m.sum()) and w["cd"]["n"] == int(m.sum())            # no trimming
    tw = np.gradient(t[m])
    assert w["cd"]["mean"] == pytest.approx((F[m] * tw).sum() / tw.sum() / qa, rel=3e-6)
    assert w["frequency_Hz"] == pytest.approx(0.65, rel=0.1) and "whole window" in w["note"]
    assert w["patches"]["wheel"]["force_N"]["fx"] == pytest.approx(0.1 * (F[m] - 21.0).mean(), rel=1e-2)
    w2 = window_average(folder, "s", {}, 1.0, 2.0)
    assert w2["cd"]["mean"] == pytest.approx(w2["force_N"]["fx"]["mean"] / qa, rel=3e-6)
    assert "error" in window_average(folder, "s", {}, 7.0, 8.0)


# ---- init_from -----------------------------------------------------------------------------------------------------------
def test_init_from_maps_fields(tmp_path, monkeypatch):
    from cfd_translator import transient
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    src = tmp_path / "run_a" / "base" / "case"
    (src / "constant" / "polyMesh").mkdir(parents=True)
    (src / "system").mkdir()
    (src / "system" / "controlDict").write_text("x")
    (src / "2.5").mkdir()
    for f in ("U", "p", "k", "omega", "UMean", "pMean"):
        (src / "2.5" / f).write_text(f"FoamFile\n{{\n    object      {f};\n}}\n")
    st = {"solve": {"transient": {"end_time": 2.5, "init_from": "run_a/base"}}}
    assert transient.init_problems(st) == []
    assert transient.init_problems({"solve": {"transient": {"end_time": 1, "init_from": "run_a/none"}}})
    dst = tmp_path / "run_b" / "v" / "case"
    (dst / "0").mkdir(parents=True)
    (dst / "system").mkdir()
    for f in ("U", "p", "k", "omega", "nut"):
        (dst / "0" / f).write_text("x")
    ran = []

    def step(cmd, log):
        ran.append(cmd)
        tmp = Path(cmd.split()[1])
        assert sorted(p.name for p in (tmp / "1").iterdir()) == ["U", "k", "omega", "p"]
        assert "object      U;" in (tmp / "1" / "U").read_text()
    notes, info = transient.map_from(dst, "run_a/base", "mean", step)
    assert info["same_mesh"] is False and "U" in info["fields"]
    assert ran and "-sourceTime 1" in ran[0] and "U (mean)" in notes[0]
    assert "patchMap" in (dst / "system" / "mapFieldsDict").read_text()
    assert transient.settings({"model": "kOmegaSST", "solve": st["solve"], "bcs": [],
                               "reference": {"velocity": 10}})["init"] == 0
    bad = {"geometry": {"box": [1, 1, 1]}, "fluid": "air", "model": "kOmegaSSTDES", "mesh": {"size": 0.2},
           "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}],
           "solve": {"transient": {"end_time": 1, "init_from": "x/y", "init": 100, "init_fields": "avg"}}}
    e = validate_study(bad)["errors"]
    assert any("init_fields" in x for x in e) and any("do not go together" in x for x in e)


# ---- per-node queue ---------------------------------------------------------------------------------------------------
def test_queue_counts_this_node_only(monkeypatch):
    from cfd_translator import executor
    out = {"squeue": "443|cfdt:r1:a|RUNNING|20|spark-0808|spark-0808|me\n"
                     "444|cfdt:r2:b|PENDING|20||spark-0808|me\n"
                     "450|cfdt:r3:c|RUNNING|4|spark-5da5|spark-5da5|me\n",
           "sinfo": "4/16/0/20\n"}
    monkeypatch.setenv("USER", "me")
    monkeypatch.setattr(executor, "_run", lambda cmd, timeout=20: (0, out[cmd[0]], ""))
    monkeypatch.setattr(executor, "effective_nodelist", lambda: "spark-5da5")
    q = executor.queue_summary()
    assert q["queue"] == {"running": 1, "queued": 0, "cpus_in_use": 4}
    assert q["cluster_queue"]["running"] == 2 and q["cluster_queue"]["queued"] == 1
    assert q["machine"]["node"] == "spark-5da5" and q["starts_now"] is False      # 4 CPUs in use
    out["squeue"] = "443|cfdt:r1:a|RUNNING|20|spark-0808|spark-0808|me\n444|cfdt:r2:b|PENDING|20||spark-0808|me\n"
    out["sinfo"] = "0/20/0/20\n"
    q = executor.queue_summary()
    assert q["queue"]["running"] == 0 and q["starts_now"] is True and q["pending_for_node"] == 0


def test_eta_start_for_a_later_segment(monkeypatch):
    from cfd_translator import executor
    monkeypatch.setattr(executor, "running_etas", lambda: {"500": {"node": "n1", "cpus": 20, "eta_min": 30.0,
                                                                     "name": "r/a"}})
    es = executor.eta_start({"segment": 2, "previous_jobs": ["500"]}, "Dependency", "2099-01-01T00:00:00")
    assert es and es["expected_start"] < "2099" and "previous segment" in es["expected_start_note"]
    assert executor.eta_start({"segment": 2, "previous_jobs": ["500"]}, "Dependency", "2000-01-01T00:00:00") is None


# ---- init_from on steady runs (MRF, cavitation), chained NPSH sweeps --------------------------------------------------
def _field(name, internal, bc="inlet { type fixedValue; value uniform 1; }"):
    return (f"FoamFile\n{{\n    format      ascii;\n    class       volScalarField;\n    object      {name};\n}}\n"
            f"dimensions [0 0 0 0 0 0 0];\n\ninternalField   {internal};\n\nboundaryField\n{{\n    {bc}\n}}\n")


def _case(root: Path, fields: dict, t="0", mesh="same"):
    for f in ("points", "faces", "owner", "neighbour"):
        (root / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
        (root / "constant" / "polyMesh" / f).write_text(f"{f} {mesh}")
    (root / "system").mkdir(exist_ok=True)
    (root / t).mkdir(parents=True, exist_ok=True)
    for f, txt in fields.items():
        (root / t / f).write_text(txt)


def test_steady_init_from_same_mesh_copies_internal_fields(tmp_path):
    from cfd_translator import transient
    src = tmp_path / "run_a" / "p20"
    _case(src / "case", {f: _field(f, "nonuniform List<scalar> 2(1 2)", "inlet { type src; }")
                         for f in ("U", "p_rgh", "alpha.water", "k", "omega", "nut", "p")}, t="4000")
    dst = tmp_path / "run_a" / "p10" / "case"
    _case(dst, {f: _field(f, "uniform 0") for f in ("U", "p_rgh", "alpha.water", "k", "omega", "nut")})
    notes, info = transient.map_from(dst, src, "last", lambda c, l: pytest.fail("no mapFields on the same mesh"))
    assert info["same_mesh"] and info["vof"] and set(info["fields"]) == {"U", "p_rgh", "alpha.water", "k", "omega", "nut"}
    t = (dst / "0" / "alpha.water").read_text()
    assert "List<scalar> 2(1 2)" in t and "type fixedValue" in t and "type src" not in t
    assert "copied (same mesh)" in notes[0]


def test_single_phase_source_leaves_cavitation_pressure(tmp_path):
    from cfd_translator import transient
    src = tmp_path / "run_a" / "mrf"
    _case(src / "case", {f: _field(f, "uniform 3") for f in ("U", "p", "k", "omega", "nut")}, t="3000")
    dst = tmp_path / "run_b" / "cav" / "case"
    _case(dst, {f: _field(f, "uniform 0") for f in ("U", "p_rgh", "alpha.water", "k", "omega", "nut")})
    notes, info = transient.map_from(dst, src, "last", lambda c, l: None)
    assert "p" not in info["fields"] and "p_rgh" not in info["fields"] and "U" in info["fields"]
    assert "pressure not carried over" in notes[0]
    assert "uniform 3" in (dst / "0" / "U").read_text() and "uniform 0" in (dst / "0" / "p_rgh").read_text()


def test_restart_tables_from_the_previous_npsh_point(tmp_path):
    from cfd_translator import cavitation
    req = tmp_path / "run_a"
    src = req / "s_npsh20"
    src.mkdir(parents=True)
    st = {"fluid": "water", "cavitation": {"npsh": 20.0}}
    (req / "request.json").write_text(json.dumps({"names": ["s_npsh20"], "studies": {"s_npsh20": st}}))
    p0_src = cavitation.source_p0(src)
    assert p0_src == pytest.approx(cavitation.water_vapour(20)["p"] + 998.2 * 9.80665 * 20, rel=2e-3)
    p0_tab = cavitation._table([(0, 5e5), (300, 1.2e5), (1e6, 1.2e5)])
    m_tab = cavitation._table([(0, 0.27), (100, 5.5), (1e6, 5.5)])
    case = tmp_path / "case"
    (case / "0").mkdir(parents=True)
    (case / "0" / "p_rgh").write_text(f"inlet {{ p0 {p0_tab}; }}")
    (case / "0" / "U").write_text(f"outlet {{ massFlowRate {m_tab}; }}")
    plan = {"vof": {"rho_l": 998.2, "Q": 5.5 / 998.2, "p0_Pa": 1.2e5, "ramp_iterations": 300, "iterations_total": 1000,
                    "p0_table": p0_tab, "m_table": m_tab}}
    notes = cavitation.restart_tables(case, plan, src)
    assert len(notes) == 2
    assert f"(0 {cavitation._fmt(p0_src)})" in (case / "0" / "p_rgh").read_text()
    assert "(0 5.5)" in (case / "0" / "U").read_text() and plan["vof"]["flow_ramp_iterations"] == 0


def test_chained_npsh_sweep_and_steady_init_from_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    from cfd_translator.variants import expand
    st = {"fluid": "water", "cavitation": {"npsh": [20, 10, 6], "chain": True},
          "solve": {"init_from": "run_x/base"}}
    out, groups, errs = expand({"s": st})
    assert not errs
    assert out["s_npsh20"]["solve"]["init_from"] == "run_x/base"
    assert out["s_npsh10"]["solve"]["init_from"] == "s_npsh20" and out["s_npsh6"]["solve"]["init_from"] == "s_npsh10"
    from cfd_translator.transient import init_problems
    assert init_problems(out["s_npsh10"], list(out), "s_npsh10") == []
    assert "itself" in init_problems({"solve": {"init_from": "a"}}, ["a"], "a")[0]
    assert "not a study of this request" in init_problems({"solve": {"init_from": "zz"}}, ["a"], "a")[0]
    assert "no case" in init_problems({"solve": {"init_from": "run_x/base"}})[0]
    base = {"geometry": {"box": [10, 1, 1]}, "fluid": "water", "mesh": {"size": 0.5},
            "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}]}
    assert validate_study(dict(base, solve={"init_from": "b"}))["ok"]
    assert not validate_study(dict(base, solve={"init_fields": "last"}))["ok"]


def test_chained_study_reuses_the_mesh(tmp_path, monkeypatch):
    from cfd_translator import mesher, runner, transient
    req = tmp_path / "run_c"
    a = dict(COUETTE)
    b = dict(COUETTE, solve={"init_from": "a"})
    c = dict(COUETTE, mesh={"cells": [10, 10, 1]}, solve={"init_from": "b"})       # other cells: meshes itself
    (req / "a").mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"names": ["a", "b", "c"], "studies": {"a": a, "b": b, "c": c}}))
    assert runner._mesh_dependents(a, req / "a") == ["b"] and runner._mesh_dependents(b, req / "b") == []
    runner.prepare(a, req / "a")
    assert (req / "a" / runner.MESHED).is_file()
    calls = []
    real = mesher.mesh_study
    monkeypatch.setattr(mesher, "mesh_study", lambda *x, **k: calls.append(1) or real(*x, **k))
    mesh, patches, sel, plan, warns = runner.prepare(b, req / "b")
    assert not calls and any("mesh reused from" in w for w in warns) and mesh.info["reused_from"].endswith("/a")
    assert not (req / "a" / runner.MESHED).exists()                  # one copy at a time
    assert transient._same_mesh(req / "a" / "case", req / "b" / "case")
    runner.prepare(c, req / "c")
    assert calls == [1]                                              # different mesh settings: meshed


def test_run_mem_checked_against_the_node(monkeypatch):
    """mochi, 2026.11.07: run.mem 90G passed validation and failed at sbatch ('Memory specification can not be
    satisfied'): a node's RealMemory is now a cap, reported in caps.mem / mem_mb for the router."""
    from cfd_translator import executor
    executor._NODES.clear()
    monkeypatch.setenv("CFDT_EXECUTOR", "slurm")
    monkeypatch.delenv("CFDT_MAX_MEM", raising=False)
    monkeypatch.setattr(executor, "effective_nodelist", lambda: None)
    monkeypatch.setattr(executor, "node_cores", lambda: 20)

    def run(cmd, timeout=20):
        return (0, "mochi 64000\n", "") if cmd[:2] == ["sinfo", "-h"] and "%N %m" in cmd else (1, "", "")
    monkeypatch.setattr(executor, "_run", run)
    c = executor.caps()
    assert c["mem"] == "62G" and c["mem_mb"] == 62 * 1024 and "RealMemory" in c["mem_from"]
    e = executor.run_errors({"run": {"mem": "90G"}})
    assert e and "at most 62G" in e[0] and "machines()" in e[0]
    assert executor.run_errors({"run": {"mem": "48G"}}) == []
    assert executor.mem_plan({"run": {"mem": "auto"}}, 30e6)[0] <= 62 * 1024          # the estimate is capped too
    monkeypatch.setenv("CFDT_MAX_MEM", "32G")                                          # a lower setting wins
    assert executor.caps()["mem"] == "32G" and executor.caps()["mem_mb"] == 32768
    executor._NODES.clear()
    monkeypatch.setattr(executor, "_run", lambda cmd, timeout=20: (0, "mochi 1\n", ""))   # memory not configured
    monkeypatch.delenv("CFDT_MAX_MEM", raising=False)
    assert "mem" not in executor.caps()
    executor._NODES.clear()
