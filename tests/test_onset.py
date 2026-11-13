"""onset (2026.11.12.17): p_eff = p_abs - c rho k below the vapour / air-release pressures, per region, with regions
derived from the rotating zone and its walls. Runs without OpenFOAM (fields written by hand)."""
from __future__ import annotations

import numpy as np
import pytest

from cfd_translator import onset
from cfd_translator.format import validate_study
from cfd_translator.runner import _onset_into

from test_sliding import patches_for, pump_mesh, zone_of

STUDY = {"geometry": {"box": [40, 40, 10]}, "fluid": "water", "model": "kOmegaSST",
         "bcs": [{"kind": "inlet", "name": "inlet", "faces": {"nz": -1}, "flow_rate": 0.001},
                 {"kind": "outlet", "name": "outlet", "faces": {"nz": 1}},
                 {"kind": "wall", "name": "blade", "faces": {"all": True}, "rotation": "rotor"}],
         "rotating": [{"name": "rotor", "axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 1500, "radius": 11.5}],
         "conditions": {"p": 50000},
         "onset": {"c": 0.3, "below": ["vapour", {"air_release": 40000}],
                   "regions": ["eye", "vane_leading_edges", "tip_clearance",
                               {"name": "outer", "cylinder": {"axis": [0, 0, 1], "radius": 20, "inner_radius": 14}}]}}


def test_validation():
    assert validate_study(STUDY)["ok"], validate_study(STUDY)
    e = lambda on: validate_study(dict(STUDY, onset=on))["errors"]   # noqa: E731
    assert any("onset.c" in x for x in e({"c": 5}))
    assert any("onset.below[0]" in x for x in e({"below": [{"air": 3}]}))
    assert any("none of" in x for x in e({"regions": ["hub"]}))
    assert any("not water" in x for x in validate_study(dict(STUDY, fluid={"rho": 800, "mu": 1e-3}))["errors"])
    assert validate_study(dict(STUDY, fluid={"rho": 800, "mu": 1e-3}, onset={"p_vapour": 500}))["ok"]
    assert validate_study(dict(STUDY, onset=True))["ok"]
    assert any("p_eff" in x for x in validate_study(dict(STUDY, images=[{"slice": {"axis": "z", "at": 5},
                                                                          "field": "q_nope"}]))["errors"])
    assert validate_study(dict(STUDY, images=[{"slice": {"axis": "z", "at": 5}, "field": "p_eff"}]))["ok"]


def test_p_eff_regions_and_thresholds(monkeypatch):
    monkeypatch.setitem(pump_mesh.__globals__, "RADII", [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20])
    m = pump_mesh(blade=(1, 5), vane=(7, 9))                    # a radial blade, r 6-10 mm; a vane at r 14-18
    patches_for(m)
    m.zones = {"rotor": zone_of(m, 11.5)}
    g = m.geometry()
    C = g["C"]
    rho = 998.2
    r = np.hypot(C[:, 0], C[:, 1])
    # gauge 0 Pa everywhere, a low-pressure spot (-49,000 Pa) at the blade's leading edge (r 6.5-8, first sector)
    p = np.zeros(m.ncells)
    th = np.degrees(np.arctan2(C[:, 1], C[:, 0])) % 360
    spot = (r < 7) & (r > 6) & ((th < 15) | (th > 345))
    p[spot] = -49000.0 / rho
    k = np.full(m.ncells, 2.0)
    fields = {"p": p, "U": np.zeros((m.ncells, 3)), "patches": {}, "scalars": {"k": {"internal": k, "patches": {}}}}
    plan = {"rho": rho}
    res = {"rotating": {"rotor": {"rotating_walls": ["hub", "blade"]}}}
    w = _onset_into(STUDY, m, plan, fields, res)
    on = res["onset"]
    assert on["thresholds_Pa"]["vapour"] == pytest.approx(2339, rel=0.01) and on["thresholds_Pa"]["air_release"] == 40000
    # p_eff = 50,000 - 49,000 - 0.3 rho 2 = 401 Pa in the spot: below the vapour pressure and the air release
    v = on["below"]["vapour"]
    assert v["domain"]["volume_mm3"] > 0 and v["domain"]["depth_max_Pa"] == pytest.approx(on["thresholds_Pa"]["vapour"] - 401.08, rel=1e-3)
    assert on["regions"]["domain"]["p_eff_min_Pa"] == pytest.approx(50000 - 49000 - 0.3 * rho * 2, rel=1e-6)
    assert v["vane_leading_edges"]["volume_mm3"] > 0 and "leading-edge blade faces" in on["regions"]["vane_leading_edges"]["how"]
    pv = on["thresholds_Pa"]["vapour"]
    assert v["outer"]["volume_mm3"] == 0 and v["outer"]["margin_Pa"] == pytest.approx(50000 - 0.3 * rho * 2 - pv, rel=1e-4)
    assert on["below"]["air_release"]["domain"]["volume_mm3"] >= v["domain"]["volume_mm3"]
    eye = on["regions"]["eye"]
    assert eye["cells"] > 0 and "within the leading-edge radius 6.5 mm" in eye["how"]
    assert on["regions"]["outer"]["how"] == "as given"
    assert "tip_clearance" in on["regions"]
    assert any("onset: p_eff falls below vapour" in x for x in w)
    assert "p_eff" in fields["scalars"] and float(fields["scalars"]["p_eff"]["internal"].min()) == pytest.approx(401, rel=0.01)


def test_no_rotating_zone_and_laminar():
    m = pump_mesh(vane=None)
    patches_for(m)
    st = dict(STUDY, rotating=[], model="laminar", onset={"regions": ["eye"]})
    fields = {"p": np.zeros(m.ncells), "U": np.zeros((m.ncells, 3)), "patches": {}}
    res = {}
    w = _onset_into(st, m, {"rho": 998.2}, fields, res)
    assert res["onset"]["regions"]["eye"]["cells"] == 0 and "no rotating zone" in res["onset"]["regions"]["eye"]["how"]
    assert res["onset"]["below"]["vapour"]["domain"]["volume_mm3"] == 0 and not w
    assert onset.settings({"onset": True})["c"] == 0.3


def test_npsh_sets_the_absolute_level():
    m = pump_mesh(vane=None)
    patches_for(m)
    rho = 998.2
    st = dict(STUDY, onset={"npsh": 9, "regions": []})
    fields = {"p": np.full(m.ncells, -2000.0 / rho), "U": np.zeros((m.ncells, 3)), "patches": {},
              "scalars": {"k": {"internal": np.zeros(m.ncells), "patches": {}}}}
    res = {"pressure": {"inlet": {"p_total": 1000.0}}}
    _onset_into(st, m, {"rho": rho}, fields, res)
    on = res["onset"]
    pv = on["thresholds_Pa"]["vapour"]
    shift = pv + rho * 9.80665 * 9 - 1000.0
    assert on["p_level"]["gauge_zero_Pa"] == pytest.approx(shift, rel=1e-5)
    assert on["regions"]["domain"]["p_eff_min_Pa"] == pytest.approx(shift - 2000.0, rel=1e-5)
    assert validate_study(st)["ok"] and any("onset.npsh" in e for e in validate_study(dict(STUDY, onset={"npsh": -1}))["errors"])


def test_robust_statistics_leave_out_one_bad_cell_and_inlet_saturation():
    """S50 variants (spark1, 2026.11.13): one cell at -822 kPa absolute in the disc-rim gap at the rotating zone's edge
    set 'domain' and 'tip_clearance'. The robust values (0.1 % / 1 % volume quantiles, minimum and volumes without
    cells of skewness > 4 or within 2 cells of the zone's boundary) do not see it. Air release relative to the
    inlet's own saturation, with ratios."""
    m = pump_mesh(vane=None)
    patches_for(m)
    m.zones = {"rotor": zone_of(m, 11.5)}
    C = m.geometry()["C"]
    r = np.hypot(C[:, 0], C[:, 1])
    rho = 998.2
    p = np.full(m.ncells, -10000.0 / rho)                    # 40 kPa absolute everywhere (conditions.p 50 kPa)
    edge = int(np.argmin(np.abs(r - 12.98)))                 # a cell just outside the zone (its boundary at r 12)
    p[edge] = -872000.0 / rho                                # -822 kPa absolute
    fields = {"p": p, "U": np.zeros((m.ncells, 3)), "patches": {}, "scalars": {"k": {"internal": np.zeros(m.ncells),
                                                                                     "patches": {}}}}
    st = dict(STUDY, onset={"below": ["vapour", {"air_release": "inlet", "ratios": [0.5]}]})
    res = {"pressure": {"inlet": {"p_mean": -2000.0, "p_total": 0.0}}}
    w = _onset_into(st, m, {"rho": rho}, fields, res)
    on = res["onset"]
    d = on["regions"]["domain"]
    assert d["p_eff_min_Pa"] == pytest.approx(-822000, rel=1e-6) and d["min_cell_flagged"]
    assert d["robust"]["p_eff_min_Pa"] == pytest.approx(40000, rel=1e-6)
    assert d["robust"]["p_eff_q0.1pct_Pa"] == pytest.approx(40000, rel=1e-6) and d["robust"]["cells_left_out"] > 0
    assert on["left_out"]["zone_boundary"] > 0 and any("flagged cell" in x for x in w)
    v = on["below"]["vapour"]["domain"]
    assert v["volume_mm3"] > 0 and v["robust"]["volume_mm3"] == 0 and v["robust"]["margin_q1pct_Pa"] > 0
    # the inlet's static 48 kPa: 40 kPa is below it (supersaturation 1.2), not below half of it
    assert on["thresholds_Pa"]["air_release"] == pytest.approx(48000) and on["thresholds_Pa"]["air_release_x0.5"] == 24000
    ar = on["below"]["air_release"]["domain"]["robust"]
    assert ar["region_fraction"] == pytest.approx(1.0) and ar["supersaturation_q1pct"] == pytest.approx(1.2)
    assert on["below"]["air_release_x0.5"]["domain"]["robust"]["volume_mm3"] == 0
    # exclude false: every cell counts, no robust block
    res2 = {"pressure": {"inlet": {"p_mean": -2000.0}}}
    _onset_into(dict(STUDY, onset={"exclude": False}), m, {"rho": rho}, dict(fields), res2)
    assert res2["onset"]["regions"]["domain"]["robust"]["cells_left_out"] == 0
    assert validate_study(st)["ok"] and not validate_study(dict(STUDY, onset={"below": [{"air_release": "x"}]}))["ok"]
