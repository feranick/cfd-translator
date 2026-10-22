"""Geometry by reference, study variants and mesh-convergence comparison (variants.py), brief results."""
from __future__ import annotations

import json

import pytest

from cfd_translator import api, variants

BASE = {"geometry": {"box": [100, 20, 10]}, "domain": {"kind": "external", "box": [-200, -100, -50, 400, 100, 100]},
        "fluid": "air", "model": "kOmegaSST", "reference": {"velocity": 30, "area": 200},
        "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 30}, {"kind": "outlet", "faces": {"box": "xmax"}},
                {"kind": "slip", "name": "far", "faces": {"box": ["ymin", "ymax", "zmin", "zmax"]}},
                {"kind": "wall", "name": "body", "faces": {"all": True}}],
        "mesh": {"mesher": "snappy", "size": 40, "surface": [2.5, 5], "features": 2.5,
                 "refine": [{"box": [-50, -20, -20, 200, 20, 30], "size": 10}], "layers": {"y_plus": 50}}}
GOAL = {"features": [{"kind": "base_box", "params": {"w": 100, "d": 20, "h": 10}}]}


def test_mesh_scales_spell_out_levels():
    ex, groups, errs = variants.expand({"car": dict(BASE, mesh_scales=[1, 0.5])})
    assert not errs and list(ex) == ["car_L1", "car_L2"] and groups["car"]["scales"] == [1.0, 0.5]
    m2 = ex["car_L2"]["mesh"]
    assert m2["size"] == 20 and m2["surface"] == [1.25, 2.5] and m2["features"] == 1.25 and m2["refine"][0]["size"] == 5
    assert m2["layers"] == {"y_plus": 50} and "mesh_scales" not in ex["car_L1"]
    assert variants.scale_mesh({"cells": [40, 16, 1], "extrude": 10}, 0.5) == {"cells": [80, 32, 1], "extrude": 20}


def test_variants_merge_and_errors():
    ex, groups, errs = variants.expand({"car": dict(BASE, variants={"slip": {"bcs": BASE["bcs"][:3] + [
        {"kind": "slip", "name": "body", "faces": {"all": True}}]}, "lam": {"model": "laminar", "reference": None}})})
    assert set(ex) == {"car_slip", "car_lam"} and ex["car_lam"]["model"] == "laminar" and "reference" not in ex["car_lam"]
    assert ex["car_slip"]["bcs"][3]["kind"] == "slip" and ex["car_slip"]["mesh"] == BASE["mesh"]
    _, _, errs = variants.expand({"a": dict(BASE, mesh_scales=[1]), "b": dict(BASE, variants=[1]),
                                  "c": dict(BASE, mesh_scales=[1, 0.5], variants={"x": {}})})
    assert set(errs) == {"a", "b", "c"}


def test_geometry_by_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    r = variants.add_geometry("box1", goal=GOAL)
    assert r["ok"] and r["use"] == {"geometry": {"ref": "box1"}}
    assert [g["name"] for g in variants.list_geometry()["geometry"]] == ["box1"]
    st, err = variants.resolve(dict(BASE, geometry={"ref": "box1"}))
    assert err is None and st["geometry"] == {"goal": GOAL}
    _, err = variants.resolve(dict(BASE, geometry={"ref": "nope"}))
    assert "box1" in err
    # a STEP uploaded as base64
    step = b"ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n"
    import base64
    r = variants.add_geometry("p", step_base64=base64.b64encode(step).decode())
    assert r["ok"] and variants.resolve(dict(BASE, geometry={"ref": "p"}))[0]["geometry"]["step"].endswith("p.step")
    assert not variants.add_geometry("q", step_base64=base64.b64encode(b"hello").decode())["ok"]
    assert not variants.add_geometry("bad name", goal=GOAL)["ok"]
    # the geometry of an earlier study
    req = tmp_path / "run_x"
    req.mkdir()
    (req / "request.json").write_text(json.dumps({"studies": {"old": dict(BASE, geometry={"goal": GOAL})}}))
    st, err = variants.resolve(dict(BASE, geometry={"ref": "run_x/old"}))
    assert err is None and st["geometry"] == {"goal": GOAL}
    # through the API: validation sees the resolved geometry and the spelled-out levels
    v = api.validate({"car": dict(BASE, geometry={"ref": "box1"}, mesh_scales=[1, 0.707])})
    assert v["ok"], v and v["variants"] == {"car": ["car_L1", "car_L2"]}
    v = api.validate({"car": dict(BASE, geometry={"ref": "nope"})})
    assert not v["ok"] and "box1" in v["studies"]["car"]["errors"][0]


def test_mesh_convergence_of_the_ahmed_series():
    # 2026.10.22 on mochi: Cd 0.322, 0.293, 0.266 (25 deg), changes not shrinking fast enough to trust
    c = variants.convergence([0.322, 0.293, 0.266], [1, 0.7071, 0.5])
    assert c["change_last_pct"] == pytest.approx(-10.2, abs=0.1) and not c["converging"] and c["order"] < 0.3
    c2 = variants.convergence([0.310, 0.297, 0.273], [1, 0.7071, 0.5])
    assert not c2["converging"]
    # a second-order series: Richardson recovers the limit
    c3 = variants.convergence([1.04, 1.01, 1.0025], [1, 0.5, 0.25])
    assert c3["order"] == pytest.approx(2, abs=0.01) and c3["extrapolated"] == pytest.approx(1.0, abs=1e-4)
    res = {f"a_L{i + 1}": {"results": {"coefficients": {"cd": v}}, "mesh": {"cells": 1000 * (i + 1)}, "reliable": True}
           for i, v in enumerate([1.04, 1.01, 1.0025])}
    cmp = variants.compare(res, {"names": list(res), "scales": [1, 0.5, 0.25]})
    assert cmp["studies"]["a_L2"]["coefficients.cd"] == 1.01
    assert cmp["mesh_convergence"]["coefficients.cd"]["extrapolated"] == pytest.approx(1.0, abs=1e-4)
    part = variants.compare({"a_L1": res["a_L1"]}, {"names": list(res), "scales": [1, 0.5, 0.25]})
    assert part["studies"]["a_L3"] == {"state": "pending"} and "mesh_convergence" not in part


def test_brief_result_is_short():
    r = {"ok": True, "reliable": True, "warnings": ["w"], "selection": {"body": [{"area_mm2": 1}] * 50},
         "results": {"coefficients": {"cd": 0.3, "cl": 0.1, "drag_dir": [1, 0, 0], "cd_std": 1e-4},
                     "forces": {"body": {"fx": 1, "fy": 2, "fz": 3, "mx": 0, "pressure": [1, 2, 3]}},
                     "feat_loads": {"faces": list(range(1000))}, "checks": {"mass_balance": {"error": 1e-9}},
                     "y_plus_part_mean": 60},
         "mesh": {"cells": 5, "layers": {"n": 3, "coverage_pct": 98, "coverage": {"body": {}}}},
         "solve": {"iterations": 10, "settled": True, "residuals": {}}, "seconds": {"wall": 12}}
    b = api.brief(r)
    assert b["results"]["coefficients"] == {"cd": 0.3, "cl": 0.1, "cd_std": 1e-4}
    assert b["results"]["forces"] == {"body": {"fx": 1, "fy": 2, "fz": 3}} and "feat_loads" not in b["results"]
    assert b["mesh"]["layers"] == {"n": 3, "coverage_pct": 98} and b["seconds"] == 12 and b["checks"]["mass_balance"] == 1e-9
    assert len(json.dumps(b)) < 700
