"""Tests of the per-iteration monitors, window averages and the 'settled' criterion (average.py). pytest -q"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cfd_translator.average import (_calibrate, average_results, monitor_functions, read_monitor, settings,
                                    window_stats)
from cfd_translator.format import validate_study
from cfd_translator.runner import _verdict, prepare

RHO, G0 = 1000.0, 9.80665
OMEGA = 1450 * 2 * math.pi / 60
Q = 0.05


def _write(path: Path, header: str, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Forces\n# CofR : (0 0 0)\n#\n" + header + "\n" + "\n".join(rows) + "\n")


def _plan():
    return {"rho": RHO,
            "patches": {"inlet": {"kind": "inlet"}, "outlet": {"kind": "outlet"}, "imp": {"kind": "wall"}},
            "zones": {"rot": {"axis": [-1.0, 0, 0], "omega_rad_s": OMEGA, "center_mm": [97.5, 0, 0],
                              "rotating_walls": ["imp"]}},
            "monitors": {"pressure": {"inlet": "cfdtP_inlet", "outlet": "cfdtP_outlet"},
                         "torque": {"rot": "cfdtM_rot"},
                         "settings": {"window": 0.25, "min_iterations": 500, "drift": 0.01, "scatter": 0.02}}}


def _case(tmp: Path, n=4000, ramp=0.0, t_amp=0.5, p_amp=1000.0):
    """Monitor files as OpenFOAM.org writes them: torque about -X oscillating around -50 N m, total pressure
    rising by 100 kPa (+ ramp per iteration) with an oscillation; static p kinematic, total(p) in Pa."""
    it = np.arange(1, n + 1)
    T = -50.0 + t_amp * np.sin(it / 7.0)                  # by the fluid on the impeller, about the axis (-X)
    Mx = -T                                               # moment about +X
    rise = 1e5 + p_amp * np.sin(it / 5.0) + ramp * it
    _write(tmp / "postProcessing" / "cfdtM_rot" / "0" / "moments.dat",
           "# Time\ttotal_x\ttotal_y\ttotal_z\tpressure_x\tpressure_y\tpressure_z\tviscous_x\tviscous_y\tviscous_z",
           [f"{i}\t({m:.9g} 1.5 -2)\t({m * 0.9:.9g} 1 -2)\t({m * 0.1:.9g} 0.5 0)" for i, m in zip(it, Mx)])
    _write(tmp / "postProcessing" / "cfdtM_rot" / "0" / "forces.dat",
           "# Time\ttotal_x\ttotal_y\ttotal_z", [f"{i}\t(1 2 3)" for i in it])
    _write(tmp / "postProcessing" / "cfdtP_inlet" / "0" / "surfaceFieldValue.dat",
           "# Time        \tweightedAverage(p)\tweightedAverage(total(p))",
           [f"{i}\t{-20.0:.9g}\t{-15000.0:.9g}" for i in it])
    _write(tmp / "postProcessing" / "cfdtP_outlet" / "0" / "surfaceFieldValue.dat",
           "# Time        \tweightedAverage(p)\tweightedAverage(total(p))",
           [f"{i}\t{(r - 15000.0 - 4500.0) / RHO:.9g}\t{r - 15000.0:.9g}" for i, r in zip(it, rise)])
    tp_last = -float(rise[-1])
    H = -tp_last / (RHO * G0)
    P = -float(T[-1]) * OMEGA
    res = {"rotating": {"rot": {"torque_Nm": float(T[-1]), "power_W": P}},
           "total_pressure_drop": tp_last, "head_loss_m": tp_last / (RHO * G0),
           "pressure_drop": -20.0 * RHO - (float(rise[-1]) - 19500.0),
           "flow": {"inlet": {"Q": Q}, "outlet": {"Q": -Q}},
           "pump": {"head_m": H, "flow_m3h": Q * 3600, "power_W": P, "efficiency": RHO * G0 * Q * H / P,
                    "note": "hydraulic efficiency"}}
    return res, it, T, rise


def test_monitor_entries_in_the_controldict(tmp_path):
    st = {"geometry": {"box": [100, 100, 10]}, "2d": True, "fluid": {"rho": 1000, "mu": 0.01},
          "rotating": [{"name": "rotor", "axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 30, "radius": 500}],
          "bcs": [{"kind": "inlet", "name": "inlet", "faces": {"nx": -1}, "velocity": 0.01},
                  {"kind": "outlet", "name": "outlet", "faces": {"nx": 1}},
                  {"kind": "wall", "name": "spin", "faces": {"ny": 1}, "rotation": "rotor"}],
          "mesh": {"cells": [10, 10, 1]}}
    assert validate_study(st)["ok"], validate_study(st)
    *_, plan, _ = prepare(st, tmp_path / "a")
    cd = (tmp_path / "a" / "case" / "system" / "controlDict").read_text()
    for name in ("cfdtTotalP", "cfdtP_inlet", "cfdtP_outlet", "cfdtM_rotor"):
        assert name in cd
    assert "patches         (spin);" in cd and "weightField     phi;" in cd and "fields          (p total(p));" in cd
    assert cd.count("{") == cd.count("}")
    assert plan["monitors"]["torque"] == {"rotor": "cfdtM_rotor"}
    assert set(plan["monitors"]["pressure"]) == {"inlet", "outlet"}
    off = dict(st, solve={"average": False})
    *_, plan, _ = prepare(off, tmp_path / "b")
    assert "cfdt" not in (tmp_path / "b" / "case" / "system" / "controlDict").read_text() and not plan["monitors"]
    # without a rotating zone: off unless asked
    plain = {k: v for k, v in st.items() if k != "rotating"}
    plain["bcs"] = st["bcs"][:2]
    assert settings(plain) is None and settings(dict(plain, solve={"average": True}))["window"] == 0.25
    fs, mon = monitor_functions(dict(plain, solve={"average": {"window": 0.5}}),
                                {"patches": {"inlet": {"kind": "inlet"}, "outlet": {"kind": "outlet"}}, "zones": {}}, 1.0)
    assert len(fs) == 3 and mon["settings"]["window"] == 0.5 and not mon["torque"]


def test_read_monitor_vectors_header_and_restart(tmp_path):
    _write(tmp_path / "postProcessing" / "m" / "0" / "surfaceFieldValue.dat",
           "# Time  weightedAverage(p)", [f"{i} {i * 0.5}" for i in range(1, 11)])
    _write(tmp_path / "postProcessing" / "m" / "8" / "surfaceFieldValue.dat",
           "# Time  weightedAverage(p)", [f"{i} {-1.0}" for i in range(9, 13)])
    its, header, vals = read_monitor(tmp_path, "m")
    assert list(its) == list(range(1, 13)) and header == ["Time", "weightedAverage(p)"]
    assert vals[7, 0] == 4.0 and vals[8, 0] == -1.0 and vals[-1, 0] == -1.0     # the restart replaces 9, 10
    assert read_monitor(tmp_path, "missing") is None


def test_window_stats_drift_and_scatter():
    it = np.arange(1, 2001)
    v = 10 + 0.1 * np.sin(it / 3.0)
    s = window_stats(it, v, {"window": 0.25, "min_iterations": 500})
    assert s["n"] == 500 and s["window"] == [1501, 2000]
    assert s["mean"] == pytest.approx(10, rel=1e-3) and s["scatter"] == pytest.approx(0.1 / math.sqrt(2) / 10, rel=0.05)
    assert s["drift"] < 1e-3
    r = window_stats(it, 10 + 0.001 * it, {"window": 0.25, "min_iterations": 500})
    assert r["drift"] == pytest.approx(0.25 / 11.75, rel=0.01)
    assert window_stats(it[:100], v[:100], {"window": 0.25, "min_iterations": 500}) is None
    assert _calibrate(2.0, 2000.0, 1000.0) == 1000.0 and _calibrate(2.0, 2.05, 1000.0) == 1.0
    assert _calibrate(2.0, 3.0, 1000.0) is None


def test_averages_replace_the_last_iteration_and_settle(tmp_path):
    res, it, T, rise = _case(tmp_path)
    avg, w = average_results(tmp_path, _plan(), res)
    assert not w, w
    assert avg["settled"] and avg["judged"] == ["head_m", "power_W"]
    q = avg["quantities"]
    win = it[-1000:]
    H = (1e5 + 1000 * np.sin(win / 5.0)) / (RHO * G0)
    P = -(-50 + 0.5 * np.sin(win / 7.0)) * OMEGA
    assert q["head_m"]["mean"] == pytest.approx(H.mean(), rel=1e-5) and q["head_m"]["window"] == [3001, 4000]
    assert q["power_W"]["mean"] == pytest.approx(P.mean(), rel=1e-5)
    assert q["torque_Nm.rot"]["mean"] == pytest.approx(T[-1000:].mean(), rel=1e-5)
    pump = res["pump"]
    assert pump["head_m"] == pytest.approx(H.mean(), rel=1e-5) and pump["head_m_last"] == pytest.approx(H[-1])
    assert pump["efficiency"] == pytest.approx(RHO * G0 * Q * H.mean() / P.mean(), rel=1e-5)
    assert pump["averaged_over"] == [3001, 4000] and "efficiency_last" in pump and "head_m_std" in pump
    assert res["rotating"]["rot"]["torque_Nm_last"] == pytest.approx(T[-1])
    assert res["total_pressure_drop"] == pytest.approx(-(1e5 + 1000 * np.sin(win / 5.0)).mean(), rel=1e-5)
    assert res["head_loss_m"] == pytest.approx(res["total_pressure_drop"] / (RHO * G0), rel=1e-5)


def test_drift_and_scatter_are_not_settled(tmp_path):
    res, *_ = _case(tmp_path / "d", ramp=5.0)            # head still rising: 5 Pa per iteration
    avg, _ = average_results(tmp_path / "d", _plan(), res)
    assert not avg["settled"] and any("head_m drifts" in x for x in avg["why"])
    res, *_ = _case(tmp_path / "s", t_amp=5.0)           # torque swings +-10 %
    avg, _ = average_results(tmp_path / "s", _plan(), res)
    assert not avg["settled"] and any("power_W scatters" in x for x in avg["why"])
    ok, why = _verdict({"solve": {"converged": False, "settled": False}}, {"averaged": avg, "checks": {}})
    assert not ok and "not settled" in why[0] and "power_W scatters" in why[0]


def test_a_monitor_that_disagrees_with_the_fields_is_not_used(tmp_path):
    res, *_ = _case(tmp_path)
    res["rotating"]["rot"]["torque_Nm"] = -80.0          # the fields say otherwise
    t_last = res["pump"]["power_W"]
    avg, w = average_results(tmp_path, _plan(), res)
    assert any("does not match" in x for x in w)
    assert "torque_Nm.rot" not in avg["quantities"] and "power_W" not in avg["quantities"]
    assert avg["judged"] == ["head_m"] and res["pump"]["power_W"] == t_last       # untouched
    # too short a run: nothing replaced
    res2, *_ = _case(tmp_path / "short", n=300)
    h = res2["pump"]["head_m"]
    avg2, w2 = average_results(tmp_path / "short", _plan(), res2)
    assert avg2["settled"] is False and res2["pump"]["head_m"] == h and any("min_iterations" in x for x in w2)


def test_static_pressure_fallback_without_total_pressure(tmp_path):
    res, it, T, rise = _case(tmp_path)
    for n in ("inlet", "outlet"):                         # drop the total(p) column
        f = tmp_path / "postProcessing" / f"cfdtP_{n}" / "0" / "surfaceFieldValue.dat"
        lines = [l.rsplit("\t", 1)[0] for l in f.read_text().splitlines()]
        f.write_text("\n".join(lines) + "\n")
    avg, w = average_results(tmp_path, _plan(), res)
    assert not w, w
    assert avg["quantities"]["head_m"]["mean"] == pytest.approx((1e5 + 1000 * np.sin(it[-1000:] / 5.0)).mean() / (RHO * G0),
                                                                rel=1e-5)


def test_verdict_accepts_settled_and_validation():
    assert _verdict({"solve": {"converged": False, "settled": True}}, {"checks": {}}) == (True, [])
    ok, why = _verdict({"solve": {"converged": False}}, {"checks": {}})
    assert not ok and "did not converge" in why[0] and "settled" not in why[0]
    base = {"geometry": {"box": [10, 10, 10]}, "fluid": "water", "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1},
                                                       {"kind": "outlet", "faces": {"nx": 1}}]}
    assert validate_study(dict(base, solve={"average": {"window": 0.3, "drift": 0.02}}))["ok"]
    assert validate_study(dict(base, solve={"average": True}))["ok"]
    for bad in ({"average": {"windows": 0.3}}, {"average": {"window": 1.5}}, {"average": "yes"},
                {"average": {"min_iterations": 5}}, {"average": {"scatter": 0}}):
        assert any("solve.average" in e for e in validate_study(dict(base, solve=bad))["errors"]), bad
