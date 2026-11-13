"""free_flow (2026.11.13.5): a transient pump run holds the flow while it settles, then the inlet holds the total
pressure that flow needed (the held phase's last revolutions, or the run it continues) and the flow follows it.
Runs without OpenFOAM (fields, monitors and the solver stood in by hand)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cfd_translator import freeflow

PERIOD = 60.0 / 2900.0
RHO = 998.2
STUDY = {"geometry": {"box": [100, 40, 40]}, "fluid": "water", "model": "kOmegaSST",
         "bcs": [{"kind": "inlet", "name": "inlet", "faces": {"nx": -1}, "flow_rate": 20 / 3600},
                 {"kind": "outlet", "name": "outlet", "faces": {"nx": 1}}],
         "rotating": [{"name": "imp", "axis": [1, 0, 0], "center": [0, 0, 0], "rpm": 2900, "radius": 15,
                       "motion": "sliding"}],
         "mesh": {"mesher": "snappy", "size": 5},
         "solve": {"transient": {"end_time": 12 * PERIOD, "average_from": 8 * PERIOD, "free_flow": True}}}

U_FILE = """FoamFile
{
    format      ascii;
    class       volVectorField;
    object      U;
}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet
    {
        type            flowRateInletVelocity;
        volumetricFlowRate 0.0055556;
        value           nonuniform List<vector> 2((1 0 0) (1.1 0 0));
    }
    outlet
    {
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }
}
"""
P_FILE = """FoamFile
{
    format      ascii;
    class       volScalarField;
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
}
"""


def _monitors(case: Path, t: np.ndarray, p_tot_pa: np.ndarray, q_m3s: np.ndarray):
    d = case / "postProcessing" / "cfdtP_inlet" / "0"
    d.mkdir(parents=True)
    stat = (p_tot_pa - 2000.0) / RHO                                  # kinematic static, 2 kPa of dynamic head
    (d / "surfaceFieldValue.dat").write_text("# Time average(p) average(total(p))\n" + "".join(
        f"{a:.8g} {b:.8g} {c:.8g}\n" for a, b, c in zip(t, stat, p_tot_pa)))
    d = case / "postProcessing" / "cfdtQ_inlet" / "0"
    d.mkdir(parents=True)
    (d / "surfaceFieldValue.dat").write_text("# Time sum(phi)\n" + "".join(f"{a:.8g} {-b:.8g}\n" for a, b in zip(t, q_m3s)))


def test_validation():
    from cfd_translator.format import validate_study
    assert validate_study(STUDY)["ok"], validate_study(STUDY)
    st = json.loads(json.dumps(STUDY))
    st["solve"]["transient"]["free_flow"] = {"after": {"revolutions": 2}, "window": {"revolutions": 3}}
    assert any("window <= after" in e for e in validate_study(st)["errors"])
    st["solve"]["transient"]["free_flow"] = {"after": {"turns": 2}}
    assert any("free_flow" in e for e in validate_study(st)["errors"])
    st = json.loads(json.dumps(STUDY))
    st["bcs"][0] = {"kind": "inlet", "name": "inlet", "faces": {"nx": -1}, "pressure": 0}
    assert any("must hold a flow" in e for e in validate_study(st)["errors"])
    st = json.loads(json.dumps(STUDY))
    st["solve"]["transient"]["average_from"] = 2 * PERIOD
    assert any("before the flow is freed" in e for e in validate_study(st)["errors"])


def test_switch_inlet_and_p0_from_the_held_phase(tmp_path):
    case = tmp_path / "case"
    for d in (case / "0.04", case / "processor0" / "0.04"):
        d.mkdir(parents=True)
        (d / "U").write_text(U_FILE)
        (d / "p").write_text(P_FILE)
    files = freeflow.switch_inlet(case, "inlet", -148.0)
    assert len(files) == 4
    u = (case / "0.04" / "U").read_text()
    assert "pressureInletOutletVelocity" in u.split("outlet")[0] and "nonuniform List<vector> 2((1 0 0) (1.1 0 0));" in u
    assert "flowRateInletVelocity" not in u and "volumetricFlowRate" not in u
    p = (case / "processor0" / "0.04" / "p").read_text()
    assert "totalPressure;" in p and "p0              uniform -148;" in p and "fixedValue" in p
    from cfd_translator.foamio import read_foam
    assert read_foam(p)["boundaryField"]["inlet"]["type"] == "totalPressure"
    # the held phase: total pressure -147.8 kPa with a +-3 kPa pulsation; p0 is the mean over the last two revolutions
    t = np.linspace(0, 4 * PERIOD, 4 * 360 + 1)
    p_tot = -147800.0 + 3000.0 * np.sin(2 * np.pi * 6 * t / PERIOD) + np.where(t < PERIOD, -20000.0, 0.0)
    _monitors(case, t, p_tot, np.full_like(t, 20 / 3600))
    p0, how = freeflow.p0_from_monitor(case, "inlet", 2 * PERIOD, 4 * PERIOD, RHO)
    assert p0 == pytest.approx(-147800.0, abs=50) and how["units"] == "Pa"


def test_run_holds_then_frees_the_flow(tmp_path, monkeypatch):
    """_solve_transient stops at the end of the held phase, sets p0, switches the inlet and goes on to end_time."""
    from cfd_translator import foam, runner
    case = tmp_path / "s" / "case"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text("FoamFile {}\nendTime         0.2068966;\ndeltaT 1e-5;\n")
    (case / "0").mkdir()
    (case / "0" / "U").write_text(U_FILE)
    (case / "0" / "p").write_text(P_FILE)
    plan = {"rho": RHO, "inlets": {"inlet": {"Q": 20 / 3600}}, "patches": {"inlet": {"kind": "inlet"},
                                                                          "outlet": {"kind": "outlet"}},
            "transient": {"end_time": 12 * PERIOD, "init": 0, "max_jobs": 3, "dt": PERIOD / 360, "period": PERIOD}}
    ends, runs = [], []

    def run(case_, timeout, procs, resume=False, deadline=None):
        import re
        end = float(re.search(r"endTime\s+([^;]+);", (case_ / "system" / "controlDict").read_text()).group(1))
        ends.append(end)
        if not runs:                                  # the held phase: monitors up to its end, then the fields there
            t = np.linspace(0, end, 4 * 360 + 1)
            _monitors(case_, t, np.full_like(t, -147800.0), np.full_like(t, 20 / 3600))
            d = case_ / f"{end:.8g}"
            d.mkdir()
            (d / "U").write_text(U_FILE)
            (d / "p").write_text(P_FILE)
        runs.append(resume)
        return {"log": {"iterations": 100}, "time": end, "reached_end": True}
    monkeypatch.setattr(foam, "run", run)
    state = {}
    warns = []
    runner._solve_transient(case, STUDY, plan, 10000.0, 1, case.parent, state, __import__("time").time(), warns)
    assert ends == [pytest.approx(4 * PERIOD), pytest.approx(12 * PERIOD)] and runs == [False, True]
    ff = state["free_flow"]
    assert ff["p0_Pa"] == pytest.approx(-147800.0) and ff["switched_at"] == pytest.approx(4 * PERIOD, rel=1e-6)
    p = (case / f"{4 * PERIOD:.8g}" / "p").read_text()
    assert f"uniform {-147800.0 / RHO:.10g}" in p and any("holds the total pressure" in w for w in warns)
    assert state["time_steps"] == 200


def test_p0_from_the_transient_run_it_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    src = tmp_path / "run_a" / "base"
    src.mkdir(parents=True)
    (src / "base.json").write_text(json.dumps({"ok": True, "solve": {"transient": True},
                                               "results": {"pressure": {"inlet": {"p_total": -147800.0}}}}))
    st = json.loads(json.dumps(STUDY))
    st["solve"]["transient"]["continue"] = "run_a/base"
    cfg = freeflow.settings(st, {"rho": RHO, "inlets": {"inlet": {"Q": 20 / 3600}}, "transient": {"period": PERIOD}},
                            tmp_path / "run_b" / "v")
    assert cfg["p0_Pa"] == -147800.0 and "base" in cfg["p0_from"] and cfg["hold_s"] == 0
    # a steady (frozen-rotor) source: its head is not the sliding run's, so the run holds the flow itself
    (src / "base.json").write_text(json.dumps({"ok": True, "solve": {"converged": True},
                                               "results": {"pressure": {"inlet": {"p_total": -150000.0}}}}))
    cfg = freeflow.settings(st, {"rho": RHO, "inlets": {}, "transient": {"period": PERIOD}}, tmp_path / "run_b" / "v")
    assert cfg["p0_Pa"] is None and cfg["hold_s"] == pytest.approx(4 * PERIOD)


def test_report_per_revolution(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    t = np.linspace(4 * PERIOD, 9 * PERIOD, 5 * 360 + 1)
    creep = np.interp(t, [4 * PERIOD, 9 * PERIOD], [19.6, 20.2])
    q = (creep + 0.4 * np.sin(2 * np.pi * 6 * t / PERIOD)) / 3600
    _monitors(case, t, np.full_like(t, -147800.0), q)
    d = case / "postProcessing" / "cfdtProbes" / "0"
    d.mkdir(parents=True)
    (d / "p").write_text("# Probe 0 (0 0 -0.04)\n# Time p0\n" + "".join(
        f"{a:.8g} {-150 + 19 * np.sin(2 * np.pi * 6 * a / PERIOD):.6g}\n" for a in t))
    plan = {"patches": {"inlet": {"kind": "inlet"}, "outlet": {"kind": "outlet"}}, "monitors": {"probes": ["eye"]}}
    cfg = {"inlet": "inlet", "rho": RHO, "period": PERIOD, "Q_target": 20 / 3600}
    state = {"free_flow": {"switched_at": 4 * PERIOD, "p0_Pa": -147800.0, "p0_from": "held"}}
    block, w = freeflow.report(case, plan, cfg, state, STUDY)
    revs = block["flow_m3h_by_revolution"]
    assert len(revs) == 5 and revs[0]["error_pct"] < -1 and abs(revs[-1]["error_pct"]) < 1
    assert revs[0]["pulsation_pct"] == pytest.approx(2.0, rel=0.2)          # +-2 % (the creep adds a little)
    eye = block["probes_p_Pa_by_revolution"]["eye"][0]
    assert eye["max"] - eye["min"] == pytest.approx(2 * 19 * RHO, rel=0.02)
    assert "total_Pa" in block["pressure_by_revolution"]["inlet"] and not w
