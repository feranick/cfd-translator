"""Transient runs (transient.py): settings, the case files, the steady start becoming time 0, time averages and
frequencies (average.py), continuing over SLURM jobs (runner, executor). OpenFOAM is not needed; the benchmarks
cylinder_2d2, cylinder_2d2_restart and ahmed*_des run it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cfd_translator import average, executor, transient
from cfd_translator.format import validate_study
from cfd_translator.runner import prepare

CH = {"geometry": {"box": [1000, 100, 10]}, "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
      "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.01}, {"kind": "outlet", "faces": {"nx": 1}}],
      "mesh": {"cells": [40, 16, 1]}, "reference": {"velocity": 0.01, "area": 1000, "length": 100},
      "solve": {"transient": {"end_time": 20.0}}}


def test_validation_and_defaults(tmp_path):
    assert validate_study(CH)["ok"], validate_study(CH)["errors"]
    bad = dict(CH, solve={"transient": {"end_time": 1, "average_from": 2}})
    assert any("average_from" in e for e in validate_study(bad)["errors"])
    des = dict(CH, model="kOmegaSSTDES", solve={})
    assert any("transient model" in e for e in validate_study(des)["errors"])
    assert any("not with energy" in e for e in validate_study(dict(CH, energy=True, fluid="air"))["errors"])
    mesh, patches, sel, plan, w = prepare(CH, tmp_path)
    tr = plan["transient"]
    assert tr["init"] == 0 and tr["average_from"] == pytest.approx(20 / 3) and tr["write_interval"] == 2.0
    assert tr["max_co"] == 1 and tr["velocity"] == 0.01 and tr["length_m"] == 0.1 and tr["max_jobs"] == 10
    assert 0 < tr["dt"] <= 0.1


def test_case_files_for_laminar_and_des(tmp_path):
    mesh, patches, sel, plan, w = prepare(CH, tmp_path / "lam")
    case = tmp_path / "lam" / "case"
    cd = (case / "system" / "controlDict").read_text()
    for k in ("startFrom       latestTime", "adjustTimeStep  yes", "maxCo           1", "runTimeModifiable true",
              "writeControl    adjustableRunTime", "endTime         20", "type            fieldAverage",
              "startTime       6.6666667", "U { mean on; prime2Mean on; }"):
        assert k in cd, k
    fv = (case / "system" / "fvSolution").read_text()
    assert "PIMPLE" in fv and "residualControl" not in fv and '"(U)Final"' in fv and "pFinal" in fv
    assert "backward" in (case / "system" / "fvSchemes").read_text()
    assert "laminar" in (case / "constant" / "momentumTransport").read_text()
    # DES: a RANS (kOmegaSST) steady start is written first; write_transient then switches to LES kOmegaSSTDES
    des = dict(CH, model="kOmegaSSTDES", solve={"transient": {"end_time": 1.0, "init": 50}})
    mesh, patches, sel, plan, w = prepare(des, tmp_path / "des")
    case = tmp_path / "des" / "case"
    assert "SIMPLE" in (case / "system" / "fvSolution").read_text()
    assert "RAS" in (case / "constant" / "momentumTransport").read_text()
    transient.write_transient(case, des, plan)
    mt = (case / "constant" / "momentumTransport").read_text()
    assert "simulationType  LES" in mt and "kOmegaSSTDES" in mt and "cubeRootVol" in mt
    assert "Gauss LUST grad(U)" in (case / "system" / "fvSchemes").read_text()
    assert '"(U|k|omega)Final"' in (case / "system" / "fvSolution").read_text()
    cd = (case / "system" / "controlDict").read_text()
    assert "executeControl  writeTime" in cd                  # yPlus, wallShearStress only when writing


def test_steady_start_becomes_time_zero(tmp_path):
    c = tmp_path / "case"
    for d in ("0", "500", "500/uniform", "processor0/0", "processor0/500/uniform", "processor1/500", "postProcessing/x/0"):
        (c / d).mkdir(parents=True)
    (c / "500" / "U").write_text("steady")
    (c / "log.foamRun").write_text("x")
    assert transient.after_init(c) == 500
    assert (c / "0" / "U").read_text() == "steady" and not (c / "500").exists()
    assert (c / "processor0" / "0").is_dir() and not (c / "processor0" / "0" / "uniform").exists()
    assert (c / "processor1" / "0").is_dir() and not (c / "postProcessing").exists()
    assert (c / "log.foamRun.init").exists() and not (c / "log.foamRun").exists()
    assert transient.latest_time(c) == 0
    # stopping a run: stopAt writeNow and back
    (c / "system").mkdir()
    (c / "system" / "controlDict").write_text("stopAt          endTime;\nendTime 1;\n")
    assert transient.request_stop(c) and "writeNow" in (c / "system" / "controlDict").read_text()
    transient.reset_stop(c)
    assert "stopAt          endTime;" in (c / "system" / "controlDict").read_text()


def test_progress_of_a_transient_log():
    log = "".join(f"Courant Number mean: 0.1 max: 0.9{i}\ndeltaT = 0.001\nTime = {0.001 * i:g}s\n\n" for i in range(1, 51))
    p = transient.progress(log, 1.0)
    assert p["time_steps"] == 50 and p["time"] == pytest.approx(0.05) and p["courant_max"] == 0.95 and p["dt"] == 0.001


def test_time_averages_and_frequency():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.5e-3, 1.5e-3, 6000))          # uneven steps, ~6 s
    v = 1.0 + 0.2 * np.sin(2 * np.pi * 3.0 * t)
    st = average.time_window_stats(t, v, {"from": 2.0, "min_samples": 20}, period=1 / 3.0)
    assert st["mean"] == pytest.approx(1.0, abs=2e-3) and st["max"] == pytest.approx(1.2, abs=1e-3)
    assert st["drift"] < 5e-3 and st["stderr"] < 2e-3 and st["window"][0] >= 2.0
    fq = average.dominant_frequency(t, v, 2.0)
    assert fq["frequency_Hz"] == pytest.approx(3.0, rel=5e-3) and fq["periods"] == pytest.approx(3 * (t[-1] - 2), rel=0.02)
    assert average.time_window_stats(t, v, {"from": t[-1] - 1e-3, "min_samples": 20}) is None


def _forces_file(path: Path, t: np.ndarray, fx: np.ndarray, fz: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{a:.8g}\t(({x:.8g} 0 {z:.8g}) (0 0 0)) ((0 0 0) (0 0 0))\n" for a, x, z in zip(t, fx, fz))
    path.write_text("# Forces\n# CofR : (0 0 0)\n# Time  forces(pressure viscous) moments(pressure viscous)\n" + rows)


def test_transient_force_history_over_a_restart(tmp_path):
    case = tmp_path / "case"
    t = np.linspace(0.0, 8.0, 8001)
    qa = 0.5 * 1.0 * 1.0 ** 2 * 1e-3
    cd = 3.2 + 0.03 * np.sin(2 * np.pi * 6.0 * t)
    cl = 1.0 * np.sin(2 * np.pi * 3.0 * t)
    a, b = t <= 4.0, t >= 4.0                                  # the second job starts again at 4.0 (repeated row)
    _forces_file(case / "postProcessing" / "cfdtF_body" / "0" / "forces.dat", t[a], cd[a] * qa, cl[a] * qa)
    _forces_file(case / "postProcessing" / "cfdtF_body" / "4" / "forces.dat", t[b], cd[b] * qa, cl[b] * qa)
    plan = {"rho": 1.0, "patches": {}, "zones": {},
            "transient": {"velocity": 1.0, "length_m": 0.1},
            "monitors": {"pressure": {}, "torque": {}, "forces": {"fo": "cfdtF_body"},
                         "settings": {"transient": True, "from": 5.0, "drift": 0.02, "stderr": 0.01, "min_samples": 20}}}
    res = {"coefficients": {"cd": 3.21, "cl": 0.0, "q_Pa": 0.5, "area_m2": 1e-3, "drag_dir": [1, 0, 0],
                            "lift_dir": [0, 0, 1]}}
    blk, w = average.average_results(case, plan, res)
    co = res["coefficients"]
    assert not w and blk["settled"], (w, blk.get("why"))
    assert co["cd"] == pytest.approx(3.2, abs=2e-3) and co["cd_max"] == pytest.approx(3.23, abs=1e-3)
    assert co["cl_max"] == pytest.approx(1.0, abs=1e-3) and co["frequency_Hz"] == pytest.approx(3.0, rel=5e-3)
    assert co["strouhal"] == pytest.approx(0.3, rel=5e-3) and blk["frequency"]["of"] == "cl"
    assert "time" in blk["history"] and blk["quantities"]["cd"]["window"][0] >= 5.0


def test_runs_continue_in_the_next_job(tmp_path, monkeypatch):
    """The steady start once, a transient segment that stops at the job's budget, the next job submitted after
    this one; the next job resumes and finishes."""
    from cfd_translator import foam, runner
    des = dict(CH, model="kOmegaSSTDES", solve={"transient": {"end_time": 1.0, "init": 10, "max_jobs": 3}})
    folder = tmp_path / "req" / "s"
    mesh, patches, sel, plan, w, state = runner._prepare_transient(des, folder)
    case = folder / "case"
    calls, subs = [], []

    def fake_run(c, timeout, procs, resume=False, deadline=None):
        calls.append({"resume": resume, "deadline": deadline})
        if len(calls) == 1:                              # the steady start
            (c / "10").mkdir()
            return {"log": {"iterations": 10}, "reached_end": True}
        if len(calls) == 2:                              # first transient part: stops at 0.4 s
            return {"log": {"iterations": 400}, "reached_end": False, "time": 0.4}
        return {"log": {"iterations": 600}, "reached_end": True, "time": 1.0}
    monkeypatch.setattr(foam, "run", fake_run)
    monkeypatch.setattr(executor, "submit", lambda req, name, st, after=None, segment=None:
                        subs.append((str(req), name, after, segment)) or {"job_id": "777"})
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    warns = []
    info = runner._solve_transient(case, des, plan, 3600, 1, folder, state, __import__("time").time(), warns)
    assert info["continued"] and info["next_job"] == "777" and subs == [(str(folder.parent), "s", "555", 2)]
    assert calls[0]["deadline"] is None and calls[1]["resume"] and calls[1]["deadline"] is not None
    assert "LES" in (case / "constant" / "momentumTransport").read_text() and (case / "0").is_dir()
    # the next job: reloads the prepared mesh and plan, resumes, reaches the end
    mesh2, patches2, sel2, plan2, w2, state2 = runner._prepare_transient(des, folder)
    assert state2["init_done"] and state2["segments"] == 1 and mesh2.ncells == mesh.ncells
    info2 = runner._solve_transient(case, des, plan2, 3600, 1, folder, state2, __import__("time").time(), warns)
    assert info2["reached_end"] and calls[2]["resume"] and len(calls) == 3
    st = json.loads((folder / "transient.json").read_text())
    assert st["time_steps"] == 1000 and st["init_iterations"] == 10


def test_continuation_job_waits_for_the_running_one(tmp_path, monkeypatch):
    req = tmp_path / "req"
    (req / "s").mkdir(parents=True)
    (req / "s" / "job.json").write_text(json.dumps({"job_id": "555", "previous_jobs": ["444"]}))
    seen = {}

    class R:
        returncode, stdout, stderr = 0, "556\n", ""

    def fake(cmd, **k):
        seen["cmd"] = cmd
        return R()
    monkeypatch.setattr(executor.subprocess, "run", fake)
    monkeypatch.setenv("CFDT_EXECUTOR", "slurm")
    info = executor.submit(req, "s", CH, after="555", segment=2)
    assert "--dependency=afterany:555" in seen["cmd"] and info["job_id"] == "556"
    j = json.loads((req / "s" / "job.json").read_text())
    assert j["segment"] == 2 and j["previous_jobs"] == ["444", "555"]
