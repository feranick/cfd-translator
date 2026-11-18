"""A multi-job transient run's jobs (solve.transient.chain): the next job submitted as the current one starts, the
hand-over at its end, the stop at a job's end (cancel at='job_end'), job_ends, and times with their UTC offset."""
from __future__ import annotations

import json
import os
import time

import pytest

from cfd_translator import api, executor, runner
from cfd_translator.case import write_case
from cfd_translator.format import validate_study

from test_sliding import _study, patches_for, pump_mesh, zone_of


@pytest.fixture
def slurm_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    return "100"


def fake_submit(calls):
    def submit(request, name, study, after=None, segment=None):
        calls.append({"after": after, "segment": segment, "open_mode": os.environ.get("SBATCH_OPEN_MODE")})
        rec = {"executor": "slurm", "job_id": str(100 + len(calls)), "segment": segment, "after": [after]}
        (request / name / "job.json").write_text(json.dumps(rec))        # as llmt-runtime's submit does
        return rec
    return submit


def test_presubmit_hand_over_and_drop(tmp_path, monkeypatch, slurm_job):
    d = tmp_path / "req" / "s"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"job_id": "100", "segment": 1}))
    calls, cancelled = [], []
    monkeypatch.setattr(executor, "submit", fake_submit(calls))
    monkeypatch.setattr(executor.J, "scancel", lambda ids: cancelled.extend(ids))
    monkeypatch.delenv("SBATCH_OPEN_MODE", raising=False)
    nxt = executor.presubmit_next(d, {}, 2)
    assert nxt["job_id"] == "101" and calls == [{"after": "100", "segment": 2, "open_mode": "append"}]
    assert "SBATCH_OPEN_MODE" not in os.environ                      # only for that sbatch
    j = json.loads((d / "job.json").read_text())
    assert j["job_id"] == "100" and j["next_job"]["job_id"] == "101"  # still the running job's record
    assert executor.presubmit_next(d, {}, 2)["job_id"] == "101" and len(calls) == 1    # once
    # the waiting job, started without the hand-over (job 100 killed, cancelled or done): exits at once
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    assert executor.not_handed_over(d)
    (tmp_path / "req" / "request.json").write_text("{}")              # never read: it returns first
    assert executor.run_study_job(str(tmp_path / "req"), "s") == 0
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    assert not executor.not_handed_over(d)
    assert executor.hand_over(d)["job_id"] == "101"
    assert json.loads((d / "job.json").read_text())["job_id"] == "101"
    monkeypatch.setenv("SLURM_JOB_ID", "101")
    assert not executor.not_handed_over(d)
    # the run ends in job 101 with a waiting 102: dropped
    (d / "job.json").write_text(json.dumps({"job_id": "101", "next_job": {"job_id": "102"}}))
    assert executor.drop_next(d) == "102" and cancelled == ["102"]
    assert json.loads((d / "job.json").read_text())["next_job"]["dropped"]


def test_presubmit_failure_keeps_the_record(tmp_path, monkeypatch, slurm_job):
    d = tmp_path / "s"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"job_id": "100"}))

    def refuse(request, name, study, after=None, segment=None):
        (request / name / "job.json").write_text(json.dumps({"broken": True}))
        return {"ok": False, "why": "could not submit: down"}
    monkeypatch.setattr(executor, "submit", refuse)
    assert executor.presubmit_next(d, {}, 2) is None
    assert json.loads((d / "job.json").read_text()) == {"job_id": "100"}


def _transient_case(tmp_path, **tr):
    m = pump_mesh()
    st = _study(**tr)
    st["model"] = "laminar"
    st["solve"]["transient"].update(init=0, max_jobs=4)
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    folder = tmp_path / "req" / "s"
    case = folder / "case"
    plan = write_case(st, m, patches_for(m), case)
    (folder / "job.json").write_text(json.dumps({"job_id": "100", "segment": 1}))
    return m, st, folder, case, plan


def _stub_solver(monkeypatch, reached=False):
    from cfd_translator import foam, sliding
    monkeypatch.setattr(foam, "run", lambda c, *a, **k: {"log": {"iterations": 3}, "checkMesh": {"ok": True},
                                                          "time": 0.2, "reached_end": reached})
    monkeypatch.setattr(foam, "_step", lambda *a, **k: 0)
    monkeypatch.setattr(foam, "find_foam", lambda: {"version": "14"})
    monkeypatch.setattr(sliding, "to_sliding", lambda *a, **k: [])


def test_together_presubmits_at_the_start_and_hands_over(tmp_path, monkeypatch, slurm_job):
    m, st, folder, case, plan = _transient_case(tmp_path)
    _stub_solver(monkeypatch)
    calls = []
    monkeypatch.setattr(executor, "submit", fake_submit(calls))
    monkeypatch.setattr(executor, "estimate_minutes", lambda *a, **k: 1e4)
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    t0 = time.time()
    out = runner._solve_transient(case, st, plan, 3600, 1, folder, state, t0, [], m)
    assert out["continued"] and out["next_job"] == "101" and len(calls) == 1      # submitted once, at the start
    assert calls[0]["segment"] == 2 and calls[0]["open_mode"] == "append"
    assert json.loads((folder / "job.json").read_text())["job_id"] == "101"       # handed over at the end
    saved = json.loads((folder / "transient.json").read_text())
    assert saved["job"]["next_job"] == "101" and saved["job"]["stops_at"] == pytest.approx(t0 + 3600 - 360, abs=1)


def test_share_submits_at_the_end(tmp_path, monkeypatch, slurm_job):
    m, st, folder, case, plan = _transient_case(tmp_path, chain="share")
    _stub_solver(monkeypatch)
    calls = []
    monkeypatch.setattr(executor, "submit", fake_submit(calls))
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    out = runner._solve_transient(case, st, plan, 3600, 1, folder, state, time.time(), [], m)
    assert out["continued"] and len(calls) == 1 and calls[0]["open_mode"] is None


def test_short_first_job_and_last_job_do_not_presubmit(tmp_path, monkeypatch, slurm_job):
    m, st, folder, case, plan = _transient_case(tmp_path)
    _stub_solver(monkeypatch, reached=True)
    calls = []
    monkeypatch.setattr(executor, "submit", fake_submit(calls))
    monkeypatch.setattr(executor, "estimate_minutes", lambda *a, **k: 5.0)          # fits in this job
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    runner._solve_transient(case, st, plan, 3600, 1, folder, state, time.time(), [], m)
    assert calls == []
    state = {"prepared": True, "segments": 3, "init_done": True, "started": True, "time_steps": 0}
    runner._solve_transient(case, st, plan, 3600, 1, folder, state, time.time(), [], m)   # job 4 of max_jobs 4
    assert calls == []


def test_stop_after_job(tmp_path, monkeypatch, slurm_job):
    m, st, folder, case, plan = _transient_case(tmp_path)
    _stub_solver(monkeypatch)
    calls = []
    monkeypatch.setattr(executor, "submit", fake_submit(calls))
    monkeypatch.setattr(executor, "estimate_minutes", lambda *a, **k: 1e4)
    (folder / executor.STOP_FILE).write_text("now")
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    warns = []
    out = runner._solve_transient(case, st, plan, 3600, 1, folder, state, time.time(), warns, m)
    assert not out.get("continued") and calls == []
    assert "after job 1" in state["stopped"] and any("cancel at job_end" in w for w in warns)


def test_cancel_at_job_end_through_the_runtime(tmp_path, monkeypatch):
    """cancel(at='job_end') is llmt-runtime's cancel_request: the stop_after_job flag (cfdt's name), a waiting next job
    cancelled."""
    req = tmp_path / "req"
    for n in ("des", "done"):
        (req / n).mkdir(parents=True)
    (req / "des" / "job.json").write_text(json.dumps({"job_id": "1", "next_job": {"job_id": "2"}}))
    (req / "done" / "job.json").write_text(json.dumps({"job_id": "3"}))
    (req / "done" / "done.json").write_text("{}")
    meta = {"names": ["des", "done"], "studies": {"des": {"solve": {"transient": {"end_time": 3}}},
                                                  "done": {"solve": {"transient": {"end_time": 3}}}}}
    (req / "request.json").write_text(json.dumps(meta))
    cancelled = []
    monkeypatch.setattr(executor.J, "scancel", lambda ids: cancelled.extend(ids))
    out = api.cancel(str(req), None, "job_end")
    assert out["stop_after_job"] == ["des"] and (req / "des" / executor.STOP_FILE).is_file()
    assert executor.STOP_FILE == "stop_after_job" and executor.stop_requested(req / "des")
    assert cancelled == ["2"] and out["not_flagged"] == {"done": "finished"}
    assert "error" in api.cancel(str(req), None, "later")


def test_run_after_keys_are_accepted_and_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    a = _study(chain="together")
    b = dict(_study(), solve={}, run={"after": "a", "after_mode": "ok"})
    v = api.validate({"a": a, "b": b})
    assert not any("unknown keys" in e for e in v["studies"]["b"]["errors"]), v["studies"]["b"]
    self_ = dict(_study(), solve={}, run={"after": "b"})
    assert any("cannot wait for itself" in e for e in api.validate({"b": self_})["studies"]["b"]["errors"])
    gone = dict(_study(), solve={}, run={"after": "run_nothing/x"})
    assert any("no study x" in e for e in api.validate({"g": gone})["studies"]["g"]["errors"])
    assert any("after_mode" in e for e in api.validate({"m": dict(_study(), solve={}, run={"after": "a", "after_mode":
                                                                                         "maybe"}),
                                                    "a": a})["studies"]["m"]["errors"])


def test_job_end_info(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    t = time.time()
    (d / "transient.json").write_text(json.dumps({"job": {"id": "100", "segment": 2, "stops_at": t + 600,
                                                          "limit_at": t + 900, "next_job": "101"}}))
    (d / "job.json").write_text(json.dumps({"job_id": "100"}))
    info = api.job_end_info(d)
    assert info["job_segment"] == 2 and info["next_job_waiting"] == "101"
    assert info["job_ends"][-6] in "+-" and info["job_ends"][-3] == ":"            # with its UTC offset
    (d / "job.json").write_text(json.dumps({"job_id": "101"}))                      # handed over, 101 not started
    assert api.job_end_info(d) == {}


def test_chain_validation():
    st = _study(chain="together")
    assert validate_study(st)["ok"]
    bad = _study(chain="alternate")
    assert any("solve.transient.chain" in e for e in validate_study(bad)["errors"])


def test_a_fixed_step_run_reaches_its_end_within_half_a_step():
    """carbonio, 2026-11-18: 60 steps of 1 degree ended at 0.0034482759 s against endTime 0.00344828; with a 1e-6
    relative tolerance the run was 'not finished' and its next jobs started again at the end time."""
    from cfd_translator.foam import end_tolerance
    fixed = "endTime 0.00344828;\ndeltaT 5.747126e-05;\nadjustTimeStep  no;\n"
    assert 0.0034482759 >= end_tolerance(0.00344828, fixed)
    assert 0.0034 < end_tolerance(0.00344828, fixed)                       # a step short is not the end
    adj = "endTime 0.00344828;\ndeltaT 5.747126e-05;\nadjustTimeStep  yes;\n"
    assert 0.0034482759 < end_tolerance(0.00344828, adj)                   # adjustable: it lands on endTime


def test_end_time_is_a_write_time_after_a_resume(tmp_path):
    """carbonio, 2026-11-18: a run resumed at 0.0021264381 s wrote 0.0028160937 and 0.0031609215 (its own write grid
    from its start) and reached endTime 0.00344828 unwritten: every next job started again from 0.0031609215."""
    from cfd_translator.foam import align_writes
    case = tmp_path / "c"
    (case / "system").mkdir(parents=True)
    (case / "processor0" / "0.0021264381").mkdir(parents=True)
    (case / "system" / "controlDict").write_text(
        "startFrom       latestTime;\nendTime         0.00344828;\ndeltaT          5.74713e-05;\nadjustTimeStep  no;\n"
        "writeControl    runTime;\n\nwriteInterval   0.000344828;\n\nfunctions\n{\n    f\n    {\n        writeInterval   1;\n"
        "    }\n}\n")
    w = align_writes(case)
    t0, end = 0.0021264381, 0.00344828
    k = round((end - t0) / w)
    assert k == 4 and abs(t0 + k * w - end) < 1e-12 and w <= 0.000344828 * 1.01
    cd = (case / "system" / "controlDict").read_text()
    assert "        writeInterval   1;" in cd and "// cfdt writeInterval 0.000344828" in cd
    (case / "processor0" / "0.0031609215").mkdir()                       # the next job: from the study's interval
    w2 = align_writes(case)
    assert round((end - 0.0031609215) / w2) == 1
    assert (case / "system" / "controlDict").read_text().count("// cfdt writeInterval") == 1
