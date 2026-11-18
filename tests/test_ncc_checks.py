"""Sliding-mesh (NCC) checks from the S50 v14 port and the NCC ladder (2026.11.14.1): snapped slivers at walls next
to the interface (thin_wall_cells), the flux balance through the coupling (interface_balance), and the locator
(a coded function object and its reader). Runs without OpenFOAM."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cfd_translator import locator, sliding
from cfd_translator.format import validate_study
from cfd_translator.polymesh import box_mesh


def test_thin_wall_cells_find_a_squashed_layer_at_a_wall():
    """A 10 mm box of 1 mm cells, the zone x < 0; the stator side's first layer at the y-min wall squashed to
    0.1 mm (a feature snapped onto): thin, and below the unreliable fraction."""
    m = box_mesh(10, 10, 10, 10, 10, 10, origin=(-5, -5, 0))
    P = m.points
    sel = (np.abs(P[:, 1] + 4.0) < 1e-9) & (P[:, 0] > -1e-9)
    P[sel, 1] = -4.9
    m.points = P
    m._geom = None if hasattr(m, "_geom") else None
    g = m.geometry()
    inz = g["C"][:, 0] < 0
    f, _ = sliding.interface(m, np.nonzero(inz)[0])
    ni = m.ninternal
    wall = ni + np.nonzero(np.isin(m.btag, [2]))[0]                  # y-min
    t = sliding.thin_wall_cells(m, g, f, inz, wall, np.array([0.0, 0.0, 1.0]), np.zeros(3))
    st, ro = t["stator_side"], t["rotor_side"]
    assert st["thin"] > 0 and st["h_min_mm"] == pytest.approx(0.1, rel=0.05) and st["h_ref_mm"] == pytest.approx(1.0, rel=0.05)
    assert ro["thin"] == 0 and t["unreliable"] and t["worst_fraction"] < 0.2
    assert all(abs(p_["at_mm"][0]) < 2 for p_ in st["places"])       # within two cells of the interface
    clean = box_mesh(10, 10, 10, 10, 10, 10, origin=(-5, -5, 0))
    g2 = clean.geometry()
    f2, _ = sliding.interface(clean, np.nonzero(g2["C"][:, 0] < 0)[0])
    t2 = sliding.thin_wall_cells(clean, g2, f2, g2["C"][:, 0] < 0, wall, np.array([0.0, 0.0, 1.0]), np.zeros(3))
    assert not t2["unreliable"] and t2["stator_side"]["thin"] == 0


def _phi(path, patches):
    bf = "".join(f"    {n}\n    {{\n        type            {t};\n        value           nonuniform List<scalar> "
                 f"{len(v)}({' '.join(map(str, v))});\n    }}\n" for n, (t, v) in patches.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("FoamFile\n{\n    format ascii;\n    class surfaceScalarField;\n    object phi;\n}\n"
                    "dimensions [0 3 -1 0 0 0 0];\ninternalField nonuniform List<scalar> 2(1 2);\n"
                    "boundaryField\n{\n" + bf + "    inlet\n    {\n        type calculated;\n        value uniform -1;\n    }\n}\n")


def test_interface_balance_sums_the_couple_on_every_processor(tmp_path):
    sl = {"rotor": "imp_rotor", "stator": "imp_stator"}
    q = 20 / 3600
    for k, (r, s_, e) in enumerate(((2e-3, -2e-3, 0.0), (-1e-3, 1.0e-3 - 3e-5, 0.0))):
        _phi(tmp_path / f"processor{k}" / "0.01" / "phi", {
            "nonConformalCyclic_on_imp_rotor": ("nonConformalCyclic", [r, 0.0]),
            "nonConformalCyclic_on_imp_stator": ("nonConformalCyclic", [s_, 0.0]),
            "nonConformalError_on_imp_rotor": ("nonConformalError", [e]),
            f"nonConformalProcessorCyclic_{k}to{1 - k}_through_nonConformalCyclic_on_imp_rotor":
                ("nonConformalProcessorCyclic", [1e-4]),
            f"nonConformalProcessorCyclic_{k}to{1 - k}_through_nonConformalCyclic_on_imp_stator":
                ("nonConformalProcessorCyclic", [-1e-4])})
    b = sliding.interface_balance(tmp_path, sl, q)
    assert b["time"] == 0.01 and b["rotor_side_m3s"] == pytest.approx(1.2e-3)
    assert b["imbalance_m3s"] == pytest.approx(-3e-5, abs=1e-12) and b["error"] == pytest.approx(3e-5 / q, rel=1e-3)
    assert b["crossing_one_way_m3s"] == pytest.approx(2e-3 + 2e-4)
    assert sliding.BALANCE_WARN < b["error"] < sliding.BALANCE_UNRELIABLE
    assert sliding.interface_balance(tmp_path / "none", sl, q) is None


def test_locator_function_reader_and_divergence_note(tmp_path):
    cfg = locator.settings({"locator": {"steps": 40, "watch": [[-12.69, 58.45, -76.31]]}})
    fo = locator.function(cfg)
    assert "steps           40;" in fo and "((-0.01269 0.05845 -0.07631))" in fo and "m.moving()" in fo
    assert fo.count("{") == fo.count("}") and "codeExecute" in fo
    assert locator.settings({"locator": True})["steps"] == 30 and locator.settings({}) is None
    d = tmp_path / "processor0"
    d.mkdir()
    (d / "cfdtWhere.dat").write_text(
        "0.001 pvf 0.11 0.0597 0 -0.0823 1e-12\n0.001 Co 22 0.0 0.05958 -0.067 1e-12\n"
        "0.001 cell p0c7 22 0.0 0.05958 -0.067 1e-12 -4 0 0 9 0 0 0.12\n"
        "0.001 face p0c7 internal -1.86e-7 1.03e-7 1e-8 0 0.05964 -0.067 0 1 0\n"
        "0.001 face p0c7 impeller 0 0 1e-8 0 0.0595 -0.067 0 -1 0 -18.07 0 0\n"
        "0.002 Co 698 0.0 0.0598 -0.076 1e-12\n")
    loc = locator.read(tmp_path, [0, 0, 1], [0, 0, 0])
    s1, s2 = loc["steps"]
    assert s1["Co"][0]["r_mm"] == pytest.approx(59.58) and s1["pvf"][0]["r_mm"] == pytest.approx(59.7)
    cd = s1["Co_detail"][0]
    assert cd["faces"][0]["v_n_m_s"] == pytest.approx(-18.6) and cd["pvf"] == 0.12
    # at (0, 59.58) about +z the rotation's direction is -x: U (-4, 0, 0) runs with it, the mesh's (9, 0, 0) against
    assert cd["U_r_theta_s"] == [0.0, 4.0, 0.0] and cd["Umesh_r_theta_s"] == [0.0, -9.0, 0.0]
    wall = [f_ for f_ in cd["faces"] if f_["patch"] == "impeller"][0]
    assert wall["U_other_r_theta_s"] == [0.0, 18.07, 0.0]
    assert s2["Co"][0]["value"] == 698 and s2["Co"][0]["s_mm"] == pytest.approx(-76.0)
    assert locator.diverged_note(tmp_path) == ""
    locator.write_frame(tmp_path, [0, 0, 1], [0, 0, 0])
    assert "step 2: Co 698" in locator.diverged_note(tmp_path)


def test_locator_validation():
    st = {"geometry": {"box": [100, 40, 40]}, "fluid": "water", "model": "kOmegaSST",
          "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}],
          "rotating": [{"name": "imp", "axis": [1, 0, 0], "center": [0, 0, 0], "rpm": 2900, "radius": 15,
                        "motion": "sliding"}],
          "mesh": {"mesher": "snappy", "size": 5},
          "solve": {"transient": {"end_time": 0.01, "locator": {"steps": 40, "watch": [[1, 2, 3]]}}}}
    assert validate_study(st)["ok"], validate_study(st)
    bad = json.loads(json.dumps(st))
    bad["solve"]["transient"]["locator"] = {"steps": -1, "where": 1, "watch": [[1, 2]]}
    e = validate_study(bad)["errors"]
    assert any("locator.steps" in x for x in e) and any("'where'" in x for x in e) and any("watch" in x for x in e)
    mrf = json.loads(json.dumps(st))
    mrf["rotating"][0]["motion"] = "mrf"
    assert any("only with a sliding zone" in x for x in validate_study(mrf)["errors"])


def test_end_faces_count_the_faces_exactly_on_each_end():
    """The zone's end face on a plane of the cells: every face on it; snapped (moved off it): few."""
    m = box_mesh(10, 10, 10, 10, 10, 10, origin=(-5, -5, 0))
    g = m.geometry()
    inz = g["C"][:, 2] < 5
    f, _ = sliding.interface(m, np.nonzero(inz)[0])
    a, o = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    h = np.ones(len(f))
    e = sliding.end_faces(g["Cf"][f], g["Sf"][f], a, o, [-1e12, 5.0], h)
    assert e == [{"s_mm": 5.0, "faces_near": 100, "faces_exact": 100, "share_exact": 1.0, "plane_mm": 5.0,
                  "plane_offset_mm": 0.0, "share_on_plane": 1.0}]
    # asked at 5.06: no face exactly on it, all of them flat on the cells' plane 0.06 below (conformal, shifted)
    e1 = sliding.end_faces(g["Cf"][f], g["Sf"][f], a, o, [-1e12, 5.06], h)[0]
    assert e1["faces_exact"] == 0 and e1["plane_mm"] == 5.0 and e1["plane_offset_mm"] == -0.06 and e1["share_on_plane"] == 1
    P = m.points
    P[(np.abs(P[:, 2] - 5) < 1e-9) & (P[:, 0] > 0.5), 2] += 0.03          # half of it snapped a little off
    m.points = P
    m2 = box_mesh(10, 10, 10, 10, 10, 10, origin=(-5, -5, 0))
    m2.points = P
    g2 = m2.geometry()
    e2 = sliding.end_faces(g2["Cf"][f], g2["Sf"][f], a, o, [-1e12, 5.0], h)
    assert e2[0]["faces_near"] == 100 and e2[0]["share_exact"] < sliding.END_SHARE_WARN
    assert e2[0]["share_on_plane"] < sliding.END_SHARE_WARN                 # scattered: snapped, not shifted


def test_end_gap_cells_count_whole_cells_to_the_wall_along_the_axis():
    """A 10 mm box of 1 mm cells, walls at z 0 and z 10, the zone z < 9: one cell between its top face and the wall
    above (the S50 long-cell mesh's top face and wear plate), more than the search below; z < 8: two cells, fine."""
    m = box_mesh(10, 10, 10, 10, 10, 10, origin=(-5, -5, 0))
    g = m.geometry()
    a, o = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    wall = m.ninternal + np.nonzero(np.isin(m.btag, [4, 5]))[0]
    for top, above, flagged in ((9.0, 1, 100), (8.0, 2, 0), (6.0, 4, 0), (5.0, sliding.END_GAP_SEARCH + 1, 0)):
        inz = g["C"][:, 2] < top
        f, _ = sliding.interface(m, np.nonzero(inz)[0])
        e = sliding.end_gap_cells(m, g, f, inz, wall, a, o)
        assert len(e) == 1 and e[0]["s_mm"] == top and e[0]["faces"] == 100
        assert e[0]["stator_side"]["min_cells"] == above and e[0]["stator_side"]["faces_under_min"] == flagged
        assert e[0]["rotor_side"]["min_cells"] == sliding.END_GAP_SEARCH + 1
        if flagged:
            assert e[0]["stator_side"]["places"][0]["s_mm"] == top
    # the zone at the bottom (z > 1): one cell between its lower face and the wall below, on the stator side;
    # the zone's own side (z 1..10) reaches the top wall beyond the search
    inz = g["C"][:, 2] > 1
    f, _ = sliding.interface(m, np.nonzero(inz)[0])
    e = sliding.end_gap_cells(m, g, f, inz, wall, a, o)[0]
    assert e["stator_side"]["min_cells"] == 1 and e["rotor_side"]["min_cells"] == sliding.END_GAP_SEARCH + 1
    # the zone in the middle with walls only at x: no wall along the axis on either side
    side_walls = m.ninternal + np.nonzero(np.isin(m.btag, [0, 1]))[0]
    inz = g["C"][:, 2] < 9
    f, _ = sliding.interface(m, np.nonzero(inz)[0])
    e = sliding.end_gap_cells(m, g, f, inz, side_walls, a, o)[0]
    assert e["stator_side"]["min_cells"] == sliding.END_GAP_SEARCH + 1      # the top is not a wall here


def test_stray_end_faces_and_end_gap_on_the_intended_planes():
    """The zone z < 5 in a 10 mm box of 1 mm cells: its end face on the plane z 5 has no stray faces; asked at 5.06,
    the faces on the grid plane 5.0 next to it count as its own (conformal, shifted). Points of the end face at x > 2
    pulled 0.4 mm down (an edge dragged by snapping): those faces are stray and leave the end-gap count."""
    a, o = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    m = box_mesh(10, 10, 10, 10, 10, 10, origin=(-5, -5, 0))
    wall = m.ninternal + np.nonzero(np.isin(m.btag, [4, 5]))[0]
    g = m.geometry()
    inz = g["C"][:, 2] < 5
    f, _ = sliding.interface(m, np.nonzero(inz)[0])
    s = sliding.stray_end_faces(m, g, f, inz, a, o, [5.0])
    assert s["faces"] == 0 and s["planes_used"] == [5.0]
    s = sliding.stray_end_faces(m, g, f, inz, a, o, [5.06])
    assert s["faces"] == 0 and s["planes_used"] == [5.06, 5.0]
    e = sliding.end_gap_cells(m, g, f, inz, wall, a, o, planes=[5.06])
    assert len(e) == 1 and e[0]["s_mm"] == 5.06 and e[0]["faces"] == 100
    P = m.points
    P[(np.abs(P[:, 2] - 5) < 1e-9) & (P[:, 0] > 2.5), 2] -= 0.4
    m2 = box_mesh(10, 10, 10, 10, 10, 10, origin=(-5, -5, 0))
    m2.points = P
    g2 = m2.geometry()
    s2 = sliding.stray_end_faces(m2, g2, f, inz, a, o, [5.0])
    assert s2["faces"] >= 20 and s2["max_offset_mm"] == pytest.approx(0.4, abs=1e-6)
    assert all(q["s_mm"] < 5 and q["r_mm"] > 2 for q in s2["places"])
    e2 = sliding.end_gap_cells(m2, g2, f, inz, wall, a, o, planes=[5.0])
    assert len(e2) == 1 and e2[0]["faces"] == 100 - s2["faces"]
    assert len(sliding.end_gap_cells(m2, g2, f, inz, wall, a, o)) > 1      # unplanned: every cluster is a 'plane'


def test_cells_per_step_is_reported_and_noted():
    """check() reports the interface cells the rotor passes per time step and notes more than 2 (pump_mesh: r 11.5,
    36 cells round about 2 mm each; 5 degrees is 1 mm of arc)."""
    from test_sliding import patches_for, pump_mesh, zone_of
    from cfd_translator.case import write_case  # noqa: F401  (the same imports test_sliding uses)
    m = pump_mesh(vane=None)
    pats = patches_for(m)
    m.zones = {"rotor": zone_of(m, 11.5)}
    g = m.geometry()
    zone = {"name": "rotor", "axis": [0, 0, 1], "center": [0, 0, 0], "radius": 11.5, "rpm": 1500, "motion": "sliding"}
    period = 60 / 1500
    for deg, fixed in ((1.0, True), (40.0, True), (40.0, False)):
        plan = {"zones": {"rotor": {"omega_rad_s": 157.0}},
                "transient": {"degrees_per_step": deg, "period": period, "fixed_step": fixed, "cells_per_step": 2.0,
                              "dt": deg / 360 * period, "max_dt": deg / 360 * period}}
        try:
            sliding.check(m, g, zone, plan, pats)
        except sliding.SlidingError:
            pytest.skip("this test mesh does not slide (see test_sliding)")
        cps = plan["sliding"]["interface_cells"]["cells_per_step"]
        tr = plan["transient"]
        if deg == 1.0:                                       # about 0.1 cells a step: left alone
            assert cps["value"] < 2 and "capped_from_degrees" not in cps and tr["degrees_per_step"] == 1.0
            continue
        # 40 degrees is several cells: capped to 2, the step with it
        assert cps["capped_from_degrees"] == 40.0 and cps["value"] == pytest.approx(2.0, rel=0.01)
        assert tr["degrees_per_step"] < 40 and tr["max_dt"] == pytest.approx(tr["degrees_per_step"] / 360 * period, rel=1e-3)
        assert tr["dt"] <= tr["max_dt"] * (1 + 1e-9)
        assert any("is capped to" in n for n in plan["notes"])
    assert any("started from rest at full speed" in n for n in plan["notes"])
    assert any("include the start from rest" in n for n in plan["notes"])          # average_from 0: within rev 1
    # null: reported, not capped; averages from the second revolution: no note about them
    plan = {"zones": {"rotor": {"omega_rad_s": 157.0}},
            "transient": {"degrees_per_step": 40.0, "period": period, "fixed_step": True, "cells_per_step": None,
                          "dt": 1.0, "max_dt": 1.0, "average_from": period}}
    sliding.check(m, g, zone, plan, pats)
    assert plan["transient"]["degrees_per_step"] == 40.0 and any("not capped" in n for n in plan["notes"])
    assert not any("include the start from rest" in n for n in plan["notes"])


def test_cells_per_step_settings_and_validation():
    from cfd_translator import transient
    st = {"geometry": {"box": [100, 40, 40]}, "fluid": "water", "model": "kOmegaSST",
          "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}],
          "rotating": [{"name": "imp", "axis": [1, 0, 0], "center": [0, 0, 0], "rpm": 2900, "radius": 15,
                        "motion": "sliding"}],
          "mesh": {"mesher": "snappy", "size": 5}, "solve": {"transient": {"end_time": 0.01}}}
    assert transient.settings(st)["cells_per_step"] == 2.0 and validate_study(st)["ok"]
    st["solve"]["transient"]["cells_per_step"] = None
    assert transient.settings(st)["cells_per_step"] is None and validate_study(st)["ok"]
    st["solve"]["transient"]["cells_per_step"] = 0.1
    assert any("cells_per_step" in e for e in validate_study(st)["errors"])


def test_estimate_plans_with_the_capped_step():
    """The router's estimate takes the step the run will use: rung 2 g 0.45 (0.15 mm cells at the interface, r
    50.225): 1 degree passes about 5.9 cells; capped to 2: about 0.34 degrees, so about 2.9x the steps."""
    from cfd_translator import executor
    st = {"geometry": {"box": [130, 130, 90]}, "fluid": "water", "model": "kOmegaSST",
          "bcs": [{"kind": "inlet", "faces": {"nz": 1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}],
          "rotating": [{"name": "imp", "axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 2900, "radius": 50.225,
                        "span": [-0.5, 15], "motion": "sliding"}],
          "mesh": {"mesher": "snappy", "size": 2.0, "surface": [1.0, 2.0],
                   "refine": [{"size": 0.15, "cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 50.95,
                                                         "inner_radius": 49.5, "span": [-3, 5]}}]},
          "solve": {"transient": {"end_time": 0.062069, "max_co": None, "degrees_per_step": 1.0}}}
    sl = st["rotating"][0]
    assert executor.interface_cell_mm(st, sl) == pytest.approx(0.15)
    deg = executor.sliding_degrees(st, sl, st["solve"]["transient"])
    assert deg == pytest.approx(np.degrees(2 * 0.15 / 50.225), rel=1e-6)
    n_capped = executor.transient_steps(st)
    st["solve"]["transient"]["cells_per_step"] = None
    assert executor.sliding_degrees(st, sl, st["solve"]["transient"]) == 1.0
    assert n_capped / executor.transient_steps(st) == pytest.approx(1.0 / deg, rel=1e-3)


def test_sliding_results_follow_the_normal_reliability_tests():
    """2026.11.14.8: the hold-back lifted (the S50's backward band was physical start-up flow): a settled, balanced
    sliding run is reliable; a sliding run's notes say how it started and whether the averages include the start."""
    from cfd_translator.runner import _verdict
    assert sliding.NCC_UNDER_INVESTIGATION == ""
    out = {"solve": {"transient": True, "settled": True, "reached_time": 1.0, "end_time": 1.0,
                     "sliding": {"zone": "imp"}}}
    ok, why = _verdict(out, {"averaged": {}, "checks": {}})
    assert ok and not why
    out["solve"]["sliding"]["interface_balance"] = {"error": 0.05}
    assert not _verdict(out, {"averaged": {}, "checks": {}})[0]


# ---- refine_coverage: a fine band over all of the sliding zone's cylinder (S50 noboth ESI, job 643) ----
def _s50_zone():
    return {"name": "impeller", "axis": [0, 0, 1], "center": [0, 0, -82.25], "radius": 59.7, "span": [0, 36.0],
            "motion": "sliding", "rpm": 2900}


def _cyl(c, r, span, size, inner=None):
    d = {"cylinder": {"center": [0, 0, c], "axis": [0, 0, 1], "radius": r, "span": span}, "size": size}
    if inner:
        d["cylinder"]["inner_radius"] = inner
    return d


def test_refine_coverage_sliding7_set_covers():
    from cfd_translator.sliding import refine_coverage
    s7 = [_cyl(-106, 64, [0, 64], 1.25), _cyl(-47.5, 61, [0, 2.5], 0.25), _cyl(-83.5, 61, [0, 2.5], 0.25),
          _cyl(-83.5, 61, [0, 38.5], 0.25, 58.5), _cyl(-83.5, 60.2, [0, 38.5], 0.125, 59.2)]
    assert refine_coverage(_s50_zone(), s7) is None
    assert refine_coverage(_s50_zone(), []) is None                  # no band at all: even at the surface level
    assert refine_coverage(_s50_zone(), [_cyl(-106, 64, [0, 64], 1.25)]) is None   # one band all over


def test_refine_coverage_mrf_set_leaves_the_floor_coarse():
    from cfd_translator.sliding import coverage_message, refine_coverage
    rc = [_cyl(-106, 64, [0, 64], 1.25), _cyl(-77.1, 61, [0, 30.8], 0.25, 58.5)]
    cov = refine_coverage(_s50_zone(), rc)
    assert cov["finest_mm"] == 0.25
    side = cov["side"][0]
    assert side["axial_mm"][0] == -82.25 and -77.3 < side["axial_mm"][1] < -77.0 and side["size_mm"] == 1.25
    assert {e["end"] for e in cov["ends"]} == {"start", "end"}
    msg = coverage_message("impeller", cov, "z")
    assert "the side from z -82.25 to" in msg and "1.25 mm cells" in msg
    # off the band and inside no region: 'no band'
    cov2 = refine_coverage(_s50_zone(), [_cyl(-77.1, 61, [0, 30.8], 0.25, 58.5)])
    assert cov2["side"][0]["size_mm"] is None and "no band" in coverage_message("impeller", cov2)
