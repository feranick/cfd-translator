"""Two-fluid gas-liquid flow (twophase.py: OpenFOAM 14 multiphaseEuler): validation, the case it writes (from the
v14 tutorials aeratedStirredTankMRF and bubbleColumnLaminar), the analytic answers of the validation cases, the
monitors' reading (air mass balance, regions, head, torque, the single-phase reference) and the runner's path.
No OpenFOAM here: the dictionaries are checked against the tutorials' syntax, not run."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cfd_translator import foamio, twophase
from cfd_translator.case import write_case
from cfd_translator.format import validate_study

from test_sliding import patches_for, pump_mesh, zone_of

RHO_W, MU_W, RHO_A = 998.2, 1.002e-3, 1.2041


def _pump_study(**tp):
    return {"geometry": {"box": [40, 40, 10]}, "fluid": "water", "model": "kOmegaSST",
            "rotating": [{"name": "rotor", "axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 1500, "radius": 11.5}],
            "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": 1.0},
                    {"kind": "outlet", "faces": {"nz": 1}, "pressure": 0}],
            "mesh": {"mesher": "snappy", "size": 2},
            "two_phase": {"bubble_size": {"fixed": 1.0}, "inlet": {"gas_fraction": 0.05},
                          "regions": [{"name": "eye", "cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 6}}],
                          **tp}}


def test_validation():
    st = _pump_study()
    assert validate_study(st)["ok"], validate_study(st)["errors"]
    e = twophase.errors(dict(st, two_phase={"bubble_size": {"iate": {}}}))
    assert any("iate" in x and "not available yet" in x for x in e)
    sl = dict(st, rotating=[dict(st["rotating"][0], motion="sliding")])
    assert any("sliding" in x for x in twophase.errors(sl))
    assert any("solve.transient" in x for x in twophase.errors(dict(st, solve={"transient": {"end_time": 1}})))
    assert any("cavitation" in x for x in twophase.errors(dict(st, cavitation={"npsh": 3})))
    assert any("regions[0]" in x for x in twophase.errors(dict(st, two_phase={"regions": [{"name": "x"}]})))
    assert any("gas_fraction: 0-0.6" in x for x in twophase.errors(dict(st, two_phase={"inlet": {"gas_fraction": 0.9}})))
    assert any("fluid 'water'" in x for x in twophase.errors(dict(st, fluid="air")))
    assert any("unknown keys" in x for x in twophase.errors(dict(st, two_phase={"bubble": 1})))
    assert twophase.settings(dict(st, two_phase={"bubble_size": 0.3}))["d_mm"] == 0.3     # a number: fixed


def test_analytic_answers():
    # the Stokes limit, and Schiller-Naumann above it
    u, re = twophase.slip_velocity(1e-3, 1e-4, RHO_W, RHO_A, MU_W)
    assert u == pytest.approx((RHO_W - RHO_A) * 1e-3 * 1e-8 / (18 * MU_W), rel=1e-3) and re < 1e-3
    u1, re1 = twophase.slip_velocity(9.81, 1e-3, RHO_W, RHO_A, MU_W)
    cd = 24 / re1 * (1 + 0.15 * re1 ** 0.687)
    assert (RHO_W - RHO_A) * 9.81 * math.pi * 1e-9 / 6 == pytest.approx(cd * math.pi * 1e-6 / 8 * RHO_W * u1 ** 2, rel=1e-6)
    assert 0.10 < u1 < 0.12
    # the rotating cylinder: Stokes would give f = (a/R)^2 exp(2 k t); the drag law's correction lowers it a little
    k = (RHO_W - RHO_A) * 1e-8 * 100 / (18 * MU_W)
    f8 = twophase.core_fraction(8.0, 0.05, 0.05 / 3, 10.0, 1e-4, RHO_W, RHO_A, MU_W)
    assert f8 < math.exp(2 * k * 8) / 9 and f8 == pytest.approx(math.exp(2 * k * 8) / 9, rel=0.04)
    assert twophase.core_fraction(0.0, 0.05, 0.025, 10, 1e-4, RHO_W, RHO_A, MU_W) == 0.25
    assert twophase.core_fraction(1e4, 0.05, 0.025, 10, 1e-4, RHO_W, RHO_A, MU_W) == 1.0


def _mrf_case(tmp_path, **tp):
    m = pump_mesh(vane=None)
    st = _pump_study(**tp)
    st["rotating"][0]["radius"] = 11.5
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    pts = patches_for(m, hub_rot="rotor")
    for p in pts:
        if p["kind"] == "inlet":
            p["bc"] = {"kind": "inlet", "name": "inlet", "velocity": 1.0}
    case = tmp_path / "case"
    plan = write_case(st, m, pts, case)
    return m, st, plan, case


def test_the_mrf_pump_case(tmp_path):
    m, st, plan, case = _mrf_case(tmp_path)
    e = plan["euler"]
    assert e["inlet_gas_fraction"] == 0.05 and e["Q_gas_m3s"] == pytest.approx(e["Q_liquid_m3s"] * 0.05 / 0.95)
    c = case / "constant"
    pp = foamio.read_foam((c / "phaseProperties").read_text())
    assert pp["phases"] == ["air", "water"] if isinstance(pp["phases"], list) else "air" in str(pp["phases"])
    assert pp["air"]["type"] == "pureIsothermalPhaseModel" and float(pp["air"]["diameterModel"]["d"]) == pytest.approx(1e-3)
    mt = (c / "momentumTransfer").read_text()
    assert "air_dispersedIn_water" in mt and "SchillerNaumann" in mt and "segregated" in mt and "Burns" in mt
    assert "kOmegaSSTSato" in (c / "momentumTransport.water").read_text()
    assert "perfectGas" in (c / "physicalProperties.air").read_text()
    assert "rhoConst" in (c / "physicalProperties.water").read_text()
    assert "phaseTurbulenceStabilisation" in (c / "fvModels").read_text()
    assert (c / "MRFProperties").is_file()
    z = case / "0"
    Ua = foamio.read_foam((z / "U.air").read_text())["boundaryField"]
    Uw = foamio.read_foam((z / "U.water").read_text())["boundaryField"]
    assert Uw["blade"]["type"] == "MRFnoSlip" and Ua["blade"]["type"] == "MRFslip"
    assert Uw["casing"]["type"] == "noSlip" and Ua["casing"]["type"] == "slip"
    assert Ua["inlet"]["type"] == "flowRateInletVelocity" and Ua["inlet"]["alpha"] == "alpha.air"
    assert float(Ua["inlet"]["volumetricFlowRate"]) == pytest.approx(e["Q_gas_m3s"], rel=1e-6)
    al = foamio.read_foam((z / "alpha.air").read_text())["boundaryField"]
    assert float(np.asarray(al["inlet"]["value"])) == pytest.approx(0.05)
    assert al["outlet"]["type"] == "inletOutlet"
    for f in ("k.water", "omega.air", "nut.water", "alphat.air", "T.water", "p", "p_rgh"):
        assert (z / f).is_file(), f
    cd = (case / "system" / "controlDict").read_text()
    assert "solver          multiphaseEuler;" in cd and "tpM_rotor_water" in cd and "tpM_rotor_air" in cd
    assert "phase           water;" in cd and "cellZone        tp_eye;" in cd and "weightField     rho.air;" in cd
    fs = (case / "system" / "fvSchemes").read_text()
    assert "localEuler" in fs                                                  # steady by default (LTS)
    fv = (case / "system" / "fvSolution").read_text()
    assert "maxCo" in fv and "pRefCell" not in fv
    zones = (case / "constant" / "polyMesh" / "cellZones").read_text()
    assert "tp_eye" in zones and "rotor" in zones


def test_the_bubble_column_case(tmp_path):
    from cfd_translator import bench
    from cfd_translator.runner import prepare
    st = bench._cases()["bubble_column_euler"]["study"]
    mesh, patches, sel, plan, w = prepare(st, tmp_path / "bc")
    case = tmp_path / "bc" / "case"
    a = foamio.read_foam((case / "0" / "alpha.air").read_text())
    C = mesh.geometry()["C"]
    ai = np.asarray(a["internalField"], float)
    assert np.all(ai[C[:, 1] > 100] == 1.0) and np.allclose(ai[C[:, 1] < 100], 0.02)
    assert a["boundaryField"]["top"]["inletValue"] == "uniform 1" or "1" in str(a["boundaryField"]["top"]["inletValue"])
    pr = foamio.read_foam((case / "0" / "p_rgh").read_text())["boundaryField"]
    assert pr["top"]["type"] == "prghTotalPressure" and pr["top"]["rho"] == "rho.air"
    g = (case / "constant" / "g").read_text()
    assert "(0 -9.81 0)" in g
    assert "Euler;" in (case / "system" / "fvSchemes").read_text()
    assert plan["euler"]["air_dispersed0_kg"] == pytest.approx(0.02 * 20e-3 * 300e-3 * 2e-3 * plan["euler"]["rho_g_ref"], rel=0.02)


def _dat(case, fo, rows):
    d = Path(case) / "postProcessing" / fo / "0"
    d.mkdir(parents=True, exist_ok=True)
    txt = "# Time value\n" + "\n".join(" ".join(f"({' '.join(map(str, x))})" if isinstance(x, (list, tuple)) else str(x)
                                               for x in r) for r in rows)
    (d / "fo.dat").write_text(txt + "\n")


def test_air_balance_regions_and_reference(tmp_path, monkeypatch):
    case = tmp_path / "c"
    case.mkdir()
    mon = {"air_mass": "tpAirMass", "air_volume": "tpAirVolume", "patch_air": {"inlet": "ai", "outlet": "ao"},
           "patch_water": {"inlet": "wi", "outlet": "wo"}, "patch_p": {"inlet": "pi", "outlet": "po"},
           "regions": {"eye": {"alpha": "ra", "U_air": "rua", "U_water": "ruw"}}, "moment": {}}
    plan = {"euler": {"monitors": mon, "steady": False, "air_dispersed0_kg": 0.0, "d_mm": 1, "drag": "SchillerNaumann",
                      "inlet_gas_fraction": 0.0, "regions": {"eye": {"volume_m3": 1e-6}}, "gravity": [0, 0, 0],
                      "rho_l": 998.2},
            "patches": {"inlet": {"kind": "inlet"}, "outlet": {"kind": "outlet"}},
            "inlets": {"inlet": {"U_mean": 1.0}}, "zones": {}}
    t = np.linspace(0, 1, 101)
    m_in = 1e-3                                         # kg/s in, half of it leaving: the held air grows by 0.5e-3 t
    _dat(case, "tpAirMass", [(x, 2e-3 + 0.5e-3 * x) for x in t])
    _dat(case, "tpAirVolume", [(x, (2e-3 + 0.5e-3 * x) / 1.2) for x in t])
    _dat(case, "ai", [(x, -m_in) for x in t])
    _dat(case, "ao", [(x, 0.5e-3) for x in t])
    _dat(case, "wi", [(x, -1e-3) for x in t])
    _dat(case, "wo", [(x, 1e-3) for x in t])
    _dat(case, "pi", [(x, 1e5) for x in t])
    _dat(case, "po", [(x, 1e5 + 998.2 * 9.80665 * 2) for x in t])
    _dat(case, "ra", [(x, 0.1) for x in t])
    _dat(case, "rua", [(x, (0.0, 0.2, 0.0)) for x in t])
    _dat(case, "ruw", [(x, (0.0, -0.01, 0.0)) for x in t])
    bal = twophase.air_balance(case, plan)
    assert bal["error"] < 1e-6 and bal["air_in_play_kg"] == pytest.approx(1e-3, rel=1e-6)
    # a leak: the held air does not grow as the flows say
    _dat(case, "tpAirMass", [(x, 2e-3 + 0.4e-3 * x) for x in t])
    assert twophase.air_balance(case, plan)["error"] == pytest.approx(0.1, rel=1e-3)
    # the reference single-phase run
    ref = tmp_path / "out" / "run_x" / "sp"
    ref.mkdir(parents=True)
    (ref / "sp.json").write_text(json.dumps({"results": {"pump": {"head_m": 2.0}, "rotating": {"r": {"torque_Nm": -1.0}}}}))
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    res, w = twophase.results(case, {"fluid": "water", "two_phase": {"reference": "run_x/sp"}}, plan)
    tp = res["two_phase"]
    assert tp["regions"]["eye"]["gas_fraction"] == pytest.approx(0.1) and tp["regions"]["eye"]["slip"][1] == pytest.approx(0.21)
    assert tp["pump"]["head_m"] == pytest.approx(2.0 - 1 / (2 * 9.80665), rel=1e-4)
    assert tp["reference"]["head_ratio"] == pytest.approx((2.0 - 1 / (2 * 9.80665)) / 2.0, rel=1e-3)
    assert tp["flows"]["outlet"]["water_m3s"] == pytest.approx(1e-3)
    assert res["checks"]["air_mass_balance"]["error"] > 0.01 and any("air mass balance" in x for x in w)


def test_the_runner_takes_the_two_fluid_path(tmp_path, monkeypatch):
    from cfd_translator import bench, foam, runner
    st = bench._cases()["bubble_column_euler"]["study"]

    def fake_run(case, timeout, procs, resume=False, deadline=None, **k):
        t = np.linspace(0, 1.0, 51)
        _dat(case, "tpAirMass", [(x, 3e-7) for x in t])
        _dat(case, "tpAirVolume", [(x, 2.5e-7) for x in t])
        _dat(case, "tpAirFlow_top", [(x, 0.0) for x in t])
        _dat(case, "tpWaterFlow_top", [(x, 0.0) for x in t])
        _dat(case, "tpP_top", [(x, 101325.0) for x in t])
        _dat(case, "tpR_mid_alpha", [(x, 0.02) for x in t])
        _dat(case, "tpR_mid_U_air", [(x, (0.0, 0.112, 0.0)) for x in t])
        _dat(case, "tpR_mid_U_water", [(x, (0.0, -0.0023, 0.0)) for x in t])
        return {"log": {"iterations": 400, "residuals": {}, "converged": True}, "checkMesh": {"ok": True},
                "openfoam": "14", "procs": procs, "time": 1.0, "reached_end": True}
    monkeypatch.setattr(foam, "run", fake_run)
    monkeypatch.setattr(foam, "find_foam", lambda: {"version": "14"})
    r = runner.run_study("bc", st, tmp_path / "bc", "foam", 600)
    assert r["ok"] and r["solve"]["solver"] == "multiphaseEuler" and r["reliable"], r.get("unreliable_because")
    ev = bench.evaluate("bubble_column_euler", r)
    assert ev["passed"], ev


def test_air_balance_takes_each_steps_own_flux(tmp_path):
    """bubble_column_euler on spark1 (2026-11-20, job 766): the top's air flux changes sign between steps; the
    trapezoid mixed two steps' fluxes (1.19 % of the bubbles' air). Each written flux belongs to the step ending at its
    time: M_n - M_(n-1) = -F_n dt_n holds exactly."""
    case = tmp_path / "c"
    case.mkdir()
    mon = {"air_mass": "m", "patch_air": {"top": "f"}}
    plan = {"euler": {"monitors": mon, "steady": False, "air_dispersed0_kg": 1e-6}}
    t = np.cumsum(np.r_[0.0, np.full(200, 0.004) * (1 + 0.3 * np.sin(np.arange(200)))])
    F = 1e-6 * np.r_[0.0, np.where(np.arange(200) % 2 == 0, 1.0, -0.8)]
    M = 5e-6 - np.cumsum(np.r_[0.0, F[1:] * np.diff(t)])
    _dat(case, "m", list(zip(t, M)))
    _dat(case, "f", list(zip(t, F)))
    bal = twophase.air_balance(case, plan)
    assert bal["error"] < 1e-9 and bal["error_trapezoid"] > 1e-3 and len(bal["series"]) <= 40


def test_reprocess_reads_the_monitors_again(tmp_path, monkeypatch):
    from cfd_translator import runner
    req = tmp_path / "req"
    folder = req / "s"
    case = folder / "case"
    case.mkdir(parents=True)
    st = {"fluid": "water", "two_phase": {}}
    (req / "request.json").write_text(json.dumps({"names": ["s"], "studies": {"s": st}}))
    plan = {"euler": {"monitors": {"air_mass": "m", "air_volume": "v", "patch_air": {}, "patch_water": {}, "patch_p": {},
                                   "regions": {}, "moment": {}},
                      "steady": False, "air_dispersed0_kg": 1e-6, "d_mm": 1, "drag": "SchillerNaumann",
                      "inlet_gas_fraction": 0.0, "regions": {}, "gravity": [0, 0, 0], "rho_l": 998.2},
            "patches": {}, "inlets": {}, "zones": {}}
    twophase.save_plan(case, plan)
    t = np.linspace(0, 1, 11)
    _dat(case, "m", [(x, 5e-6) for x in t])
    _dat(case, "v", [(x, 4e-6) for x in t])
    (folder / "s.json").write_text(json.dumps({"ok": True, "translator": "2026.11.20.3", "results": {},
                                               "warnings": ["air mass balance off by 1.19% of the air in play (> 1%)"],
                                               "reliable": False, "unreliable_because": ["air mass balance off"]}))
    r = runner.reprocess_study(folder, "s")
    assert r["reliable"] and r["results"]["two_phase"]["air_mass_balance"]["error"] == 0
    assert not any("air mass balance" in w for w in r["warnings"]) and r["reprocessed"]["from"] == "the monitors"
    assert (folder / "s.json.2026.11.20.3.bak").is_file()
