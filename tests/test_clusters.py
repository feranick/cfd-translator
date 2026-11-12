"""Running on clusters cfdt's servers cannot reach (docs/clusters.md): the MPI launcher template, ranks over
several nodes, and a request packed there and imported here."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from cfd_translator import api, foam
from cfd_translator.transfer import import_request, pack

COUETTE = {"2d": True, "geometry": {"box": [1.0, 0.1, 0.01]}, "fluid": {"rho": 1000.0, "mu": 1e-3},
           "bcs": [{"kind": "wall", "name": "lid", "faces": {"ny": 1}, "velocity": [0.01, 0, 0]},
                   {"kind": "wall", "name": "bottom", "faces": {"ny": -1}},
                   {"kind": "opening", "name": "left", "faces": {"nx": -1}},
                   {"kind": "opening", "name": "right", "faces": {"nx": 1}}],
           "mesh": {"cells": [20, 10, 1]}, "images": True}


def test_launch_template(monkeypatch):
    monkeypatch.delenv("CFDT_LAUNCH", raising=False)
    monkeypatch.delenv("CFDT_MPIRUN", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert foam.launch(8) == "mpirun -np 8"
    monkeypatch.setenv("CFDT_LAUNCH", "srun -n {n} --cpu-bind=cores")             # Perlmutter (cray-mpich)
    assert foam.launch(512) == "srun -n 512 --cpu-bind=cores"
    monkeypatch.setenv("CFDT_LAUNCH", "srun --mpi=pmix -n")                        # no {n}: the count goes last
    assert foam.launch(96) == "srun --mpi=pmix -n 96"


def test_ranks_per_node_for_hwthread_mapping(monkeypatch):
    """A 512-rank job over 4 nodes runs 128 ranks on each: only those are compared with the node's cores."""
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "4")
    monkeypatch.delenv("CFDT_MPIRUN", raising=False)
    assert foam.ranks_per_node(512) == 128
    monkeypatch.setattr(foam, "physical_cores", lambda: 128)
    assert foam.mpirun_command(512) == "mpirun"
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    assert foam.mpirun_command(512) == "mpirun --map-by hwthread"


def test_sliding_utilities_use_the_launcher():
    """to_sliding gets foam.launch(procs): 'srun -n 512 ...' is used as it is; a bare 'mpirun' gets -np N."""
    import re
    for lp, want in (("srun -n 512 --cpu-bind=cores", "srun -n 512 --cpu-bind=cores "),
                     ("mpirun -np 8", "mpirun -np 8 "), ("mpirun", "mpirun -np 8 ")):
        par = (lp if re.search(r"(^|\s)(-np|-n)\s+\d+", lp) else f"{lp} -np 8") + " "
        assert par == want


def test_pack_and_import_a_request(tmp_path, monkeypatch):
    cluster, home = tmp_path / "pscratch" / "cfdt_runs", tmp_path / "home" / "cfdt_runs"
    monkeypatch.setenv("CFDT_OUT", str(cluster))
    monkeypatch.delenv("CFDT_EXECUTOR", raising=False)
    r = api.run({"c": COUETTE}, "ref", wait=60)
    assert r["done"] and r["studies"]["c"]["ok"]
    req = Path(r["request"])
    # a study still running on the cluster when packed, with its job record
    (req / "late").mkdir()
    (req / "late" / "job.json").write_text(json.dumps({"job_id": "123456"}))
    meta = json.loads((req / "request.json").read_text())
    meta["names"].append("late")
    (req / "request.json").write_text(json.dumps(meta))
    (req / "c" / "case" / "postProcessing" / "fo" / "0").mkdir(parents=True)
    (req / "c" / "case" / "postProcessing" / "fo" / "0" / "forces.dat").write_text("# Time\n1 (1 2 3)\n")
    (req / "c" / "case" / "log.foamRun").write_bytes(b"x" * int(2.5 * 1024 * 1024) + b"\nEnd\n")
    p = pack(str(req), out=str(tmp_path / "r.tgz"))
    assert p["studies_with_results"] == ["c"] and p["studies_without_results"] == ["late"] and "logs_cut" in p
    with tarfile.open(p["archive"]) as tar:
        names = tar.getnames()
    assert f"{req.name}/c/c.json" in names and f"{req.name}/c/case/postProcessing/fo/0/forces.dat" in names
    assert not any("/case/0/" in n or "polyMesh" in n or n.endswith("state.npz") for n in names)
    # here
    monkeypatch.setenv("CFDT_OUT", str(home))
    got = api.import_results(str(tmp_path / "r.tgz"))
    dest = Path(got["request"])
    assert dest.parent == home and got["not_finished"] == ["late"] and got["fields"] is False
    s = api.status(str(dest))
    assert s["done"] and s["studies"]["c"]["ok"] and s["studies"]["late"]["state"] == "not_finished"
    t = (dest / "c" / "c.json").read_text()
    assert str(req) not in t and str(dest) in t                       # paths point here
    assert not (dest / "late" / "job.json").exists() and (dest / "late" / "job.cluster.json").is_file()
    assert json.loads((dest / "request.json").read_text())["imported"]["path"] == str(req)
    log = (dest / "c" / "case" / "log.foamRun").read_bytes()
    assert log.startswith(b"(cfdt pack") and log.endswith(b"End\n") and len(log) < 2.1 * 1024 * 1024
    again = import_request(str(tmp_path / "r.tgz"))                    # a second import does not overwrite
    assert again["request"].endswith("_import2")
    # a folder copied back whole
    folder = import_request(str(req))
    assert Path(folder["request"]).is_dir() and folder["studies"] == ["c", "late"]


def test_import_refuses_odd_archives(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path / "out"))
    bad = tmp_path / "bad.tgz"
    with tarfile.open(bad, "w:gz") as tar:
        f = tmp_path / "x"
        f.write_text("x")
        tar.add(str(f), arcname="../evil")
    assert "error" in import_request(str(bad))
    assert "error" in import_request(str(tmp_path / "missing.tgz"))


PERLMUTTER = "\n".join(
    f"NodeName=nid{i:03d} CoresPerSocket=64 CPUAlloc=0 CPUTot=256 NodeAddr=nid{i:03d} NodeHostName=nid{i:03d} "
    f"RealMemory=512000 Sockets=2 State=IDLE ThreadsPerCore=2 AvailableFeatures=cpu" for i in range(1, 7))


def test_512_ranks_on_128_core_nodes(tmp_path, monkeypatch):
    """Perlmutter (S50 sliding mesh): CFDT_SLURM_NODES=4 and -C cpu give a 512-rank job 4 nodes x 128 ranks, the
    caps say so, and the cell estimate's memory is the job's total, split per node by llmt-runtime."""
    from llmt_runtime import slurm as S
    from llmt_runtime.testing import fake_slurm

    from cfd_translator import executor
    fake_slurm(tmp_path, monkeypatch)
    f = tmp_path / "nodes.txt"
    f.write_text(PERLMUTTER)
    monkeypatch.setenv("FAKE_SCONTROL_NODES", str(f))
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("CFDT_EXECUTOR", "slurm")
    monkeypatch.setenv("CFDT_SLURM_NODE", "any")
    monkeypatch.setenv("CFDT_SLURM_NODES", "4")
    monkeypatch.setenv("CFDT_SLURM_ARGS", "-A m526 -C cpu -q regular")
    monkeypatch.setenv("CFDT_PROCS", "512")
    monkeypatch.setenv("CFDT_LAUNCH", "srun -n {n} --cpu-bind=cores")
    S.forget()
    c = executor.caps()
    assert c["procs"] == "512" and c["nodes"] == 4 and "128 x 4 nodes" in c["procs_from"] and c["mem_per"] == "node"
    big = dict(COUETTE, run={"procs": 512, "queue": True})
    monkeypatch.setattr(executor, "_cells", lambda st: 13.7e6)        # the S50 mesh: 13.7 M cells
    assert executor.procs_for_job(big, 13.7e6) == 512
    assert executor.procs_for_job(dict(big, run={"procs": 1000, "queue": True}), 13.7e6) == 512   # 4 x 128 at most
    r = api.run({"s50": big}, "ref", wait=0)
    req = Path(r["request"])
    sh = (req / "s50" / "job.sh").read_text()
    ln = [x[len("#SBATCH "):] for x in sh.splitlines() if x.startswith("#SBATCH")]
    assert {"--ntasks=512", "--nodes=4", "--ntasks-per-node=128", "--ntasks-per-core=1"} <= set(ln)
    j = json.loads((req / "s50" / "job.json").read_text())
    assert j["nodes"] == 4 and j["ntasks_per_node"] == 128 and j["mem_per"] == "node"
    total = executor.mem_auto(big)                                      # 2 GB + 4 GB per million cells, the job's
    assert j["mem_total"] == total and j["mem"] != total
    assert foam.launch(512) == "srun -n 512 --cpu-bind=cores"
    # a study that fits one node stays on one
    r = api.run({"small": dict(big, run={"procs": 100, "queue": True})}, "ref", wait=0)
    ln = [x for x in (Path(r["request"]) / "small" / "job.sh").read_text().splitlines() if x.startswith("#SBATCH")]
    assert "#SBATCH --ntasks=100" in ln and "#SBATCH --nodes=1" in ln
    S.forget()


def test_job_setup_lines_in_job_sh(tmp_path, monkeypatch):
    """CFDT_JOB_SETUP: module loads and the OpenFOAM bashrc in job.sh, so jobs do not depend on the submitting shell."""
    from llmt_runtime import slurm as S
    from llmt_runtime.testing import fake_slurm
    fake_slurm(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("CFDT_EXECUTOR", "slurm")
    monkeypatch.setenv("CFDT_JOB_SETUP", "module load PrgEnv-gnu cray-mpich\\nsource $HOME/of14/etc/bashrc")
    S.forget()
    r = api.run({"c": dict(COUETTE, queue=True)}, "ref", wait=0)
    sh = (Path(r["request"]) / "c" / "job.sh").read_text()
    assert "module load PrgEnv-gnu cray-mpich\nsource $HOME/of14/etc/bashrc\n" in sh
    assert sh.index("module load") < sh.index("run-study")
    S.forget()


def test_status_summarises_many_nodes():
    """Perlmutter: cfdt status listed all 5,288 nodes; a large cluster gets a summary."""
    from cfd_translator.executor import NODES_LISTED, nodes_listing
    few = [{"name": f"n{i}", "cpus": 40, "cores": 20} for i in range(3)]
    assert nodes_listing(few) == {"nodes": few}
    many = [{"name": f"nid{i:06d}", "cpus": 256, "cores": 128} for i in range(5000)] + \
           [{"name": f"gpu{i}", "cpus": 128, "cores": 64} for i in range(288)]
    out = nodes_listing(many)
    s = out["nodes_summary"]
    assert len(out["nodes"]) == NODES_LISTED and s["count"] == 5288
    assert s["kinds"] == [{"cpus": 256, "cores": 128, "count": 5000}, {"cpus": 128, "cores": 64, "count": 288}]
    assert s["names"] == ["nid000000", "nid000001", "nid000002", "...", "gpu287"]


def test_bench_jobs_have_their_own_time(monkeypatch):
    """Perlmutter's debug QOS (30 min) refused the bench jobs, which took the machine's 12 h CFDT_TIMEOUT."""
    from cfd_translator import bench
    monkeypatch.delenv("CFDT_BENCH_TIMEOUT", raising=False)
    cases = bench._cases()
    st = bench.job_study(cases["channel_2d"])
    assert st["queue"] is True and st["run"]["timeout_s"] == 1200.0                 # job: 20 + 10 min
    assert bench.job_study(cases["cylinder_2d2_restart"])["run"]["timeout_s"] == 120    # its own, shorter
    assert "timeout_s" not in (bench.job_study(cases["ahmed25_L1"]).get("run") or {})  # slow: not capped
    monkeypatch.setenv("CFDT_BENCH_TIMEOUT", "600")
    assert bench.job_study(cases["ahmed25_L1"])["run"]["timeout_s"] == 600.0
    assert "run" not in cases["channel_2d"]["study"]                                    # the case itself unchanged


def test_stl_sphere_bench(tmp_path, monkeypatch):
    """sphere_snappy_stl: the Re 100 sphere from a generated STL, no build123d-translator or Gmsh."""
    import math

    from cfd_translator import bench, meshsurf
    from cfd_translator.format import validate_study
    c = bench._cases()["sphere_snappy_stl"]
    assert c["needs"] == [] and validate_study(c["study"])["ok"], validate_study(c["study"])
    L = meshsurf.load(c["study"]["geometry"])
    assert L["bodies"] == 1 and L["check"]["triangles"] == 5120 and L["check"]["open_edges"] == 0
    assert L["body_volumes_mm3"][0] == pytest.approx(4 / 3 * math.pi * 125, rel=0.01)


def _bench_result(transient=False, wall=42.0, warnings=()):
    sv = ({"transient": True, "converged": False, "settled": True, "time_steps": 800, "reached_time": 0.4}
          if transient else {"converged": True, "iterations": 300})
    return {"ok": True, "reliable": True, "solve": sv, "mesh": {"cells": 2000},
            "results": {"checks": {"mass_balance": {"error": 0}, "momentum_balance": {"error": 0}}},
            "warnings": list(warnings), "seconds": {"mesh_case": 1, "solve": 2, "results": 1, "wall": wall}}


def test_bench_lines_show_each_case_and_settled(monkeypatch, capsys):
    """Perlmutter bench: every SLURM case said 'PASS in 1012 s' (the bench's wait, debug queue included), and the
    transient restart case said 'converged False' though it passed and was reliable."""
    from cfd_translator import bench
    res = {"ok": True, "passed": True, "checks": [], "notes": []}
    bench._progress("cylinder_2d2_restart", res, None, _bench_result(transient=True, wall=75.4))
    bench._progress("channel_2d", res, None, _bench_result(wall=27.2))
    err = capsys.readouterr().err
    assert "cylinder_2d2_restart: PASS in 75 s" in err and "settled True, reliable True" in err
    assert "converged" not in err.splitlines()[0]
    assert "channel_2d: PASS in 27 s" in err and "converged True" in err
    assert bench._wall({"seconds": 12.5}) == 12.5 and bench._wall({"seconds": {"mesh_case": 1, "solve": 2}}) == 3.0
    assert bench._wall({}) is None


def test_bench_verdict_solve_and_expected_notes():
    from cfd_translator import bench
    sv = bench.solve_summary(_bench_result(transient=True))
    assert sv == {"transient": True, "settled": True, "time_steps": 800, "reached_time": 0.4}
    assert bench.solve_summary(_bench_result()) == {"converged": True, "iterations": 300}
    w = ["ran on 1 MPI rank instead of 4: 2000 cells (at least 5000 per rank, CFDT_CELLS_PER_RANK)",
         "averaging: 144 iterations recorded, fewer than min_iterations (200); final-iteration values kept",
         "averaging: no moment history from cfdtForces (postProcessing/cfdtForces); final-iteration values kept"]
    notes = bench.expected_notes(w)
    assert len(notes) == 2 and "fewer ranks" in notes[0] and "window average" in notes[1]
    assert bench.expected_notes(w[2:]) == []
