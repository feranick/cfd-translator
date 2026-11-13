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


def _perlmutter(tmp_path, monkeypatch):
    from llmt_runtime import slurm as S
    from llmt_runtime.testing import fake_slurm
    fake_slurm(tmp_path, monkeypatch)
    f = tmp_path / "nodes.txt"
    f.write_text(PERLMUTTER)
    monkeypatch.setenv("FAKE_SCONTROL_NODES", str(f))
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    for k, v in (("CFDT_EXECUTOR", "slurm"), ("CFDT_SLURM_NODE", "any"), ("CFDT_SLURM_NODES", "4"),
                 ("CFDT_SLURM_ARGS", "-A m526 -C cpu -q regular"), ("CFDT_PROCS", "512")):
        monkeypatch.setenv(k, v)
    S.forget()
    return S


def test_run_mem_is_per_node_on_multi_node_jobs(tmp_path, monkeypatch):
    """Perlmutter: run.mem is SLURM's --mem, per node (llmt-runtime passes it as given), so its cap is a node's
    memory; validation now says so, the 'auto' run.mem is split over the nodes (it asked for the whole estimate on
    each), and the auto total may reach a node's memory on each node."""
    S = _perlmutter(tmp_path, monkeypatch)
    from cfd_translator import executor
    c = executor.caps()
    assert c["mem_per"] == "node" and "per node" in c["mem_note"]
    big = dict(COUETTE, run={"procs": 512, "queue": True, "mem": "600G"})
    errs = executor.run_errors(big)
    assert any("per node" in e and "node's share" in e for e in errs), errs
    assert not executor.run_errors(dict(big, run={"procs": 512, "queue": True, "mem": "500G"}))
    monkeypatch.setattr(executor, "_cells", lambda st: 200e6)           # 2 + 800 GB: more than one node has
    auto = dict(big, run={"procs": 512, "queue": True, "mem": "auto"})
    assert executor.job_nodes(auto, 200e6) == 4
    total = executor.mem_mb(executor.mem_auto(auto, 200e6))
    assert total > 512000                                               # not cut to one node's memory
    per = executor.mem_mb(executor.mem_auto_per_node(auto, 200e6))
    assert per <= 512000 and abs(per - total / 4) < 1100
    assert executor.mem_mb(executor._for_job(auto)["run"]["mem"]) == per
    assert executor.mem_needed_mb(auto, 200e6) == per                   # the router compares it with caps.mem_mb
    r = api.run({"s": auto}, "ref", wait=0)
    j = json.loads((Path(r["request"]) / "s" / "job.json").read_text())
    assert j["nodes"] == 4 and executor.mem_mb(j["mem"]) == per
    # one rule: each node gets what one node would (2 GB + its share), and 8G a node when the size is unknown
    # (CFDT_SLURM_MEM=auto: the runtime divides mem_auto's total; it asked for 2G a node on 4 nodes)
    monkeypatch.setattr(executor, "cell_estimate", lambda st: None)
    monkeypatch.setattr(executor, "_cells", lambda st: None)
    plain = dict(COUETTE, run={"procs": 512, "queue": True})
    assert executor.job_nodes(plain, None) == 4
    assert executor.mem_mb(executor.mem_auto(plain, None)) == 4 * 8192
    assert executor.mem_mb(S.mem_plan(executor.RT, plain)["mem"]) == 8192
    assert executor.mem_needed_mb(plain, None) == 8192
    S.forget()


def test_cli_validate_exit_code(tmp_path, capsys):
    """`cfdt validate s.json && cfdt run s.json` ran the run although the study was invalid (exit 0)."""
    from cfd_translator import cli
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"s": dict(COUETTE, fluid="lava")}))
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"s": COUETTE}))
    with pytest.raises(SystemExit) as e:
        cli.main(["validate", str(bad)])
    assert e.value.code == 1 and '"ok": false' in capsys.readouterr().out
    cli.main(["validate", str(good)])                                   # valid: returns, exit 0
    assert cli.exit_code("pack", {"error": "no request"}) == 1 and cli.exit_code("bench", {"a": {"passed": False}}) == 0


def test_caps_say_what_capped_the_memory(tmp_path, monkeypatch):
    """Perlmutter refused --mem=480G with RealMemory 515100M: the cap is what a job can get (llmt-runtime 2026.11.12.1),
    and cfdt's caps() names the runtime's reason instead of 'a node's memory (SLURM RealMemory)'."""
    S = _perlmutter(tmp_path, monkeypatch)
    from cfd_translator import executor
    nodes, parts = tmp_path / "nodes.txt", tmp_path / "parts.txt"
    monkeypatch.setenv("FAKE_SCONTROL_PARTITIONS", str(parts))
    nodes.write_text(PERLMUTTER.replace("RealMemory=512000", "RealMemory=515100 MemSpecLimit=30000"))
    parts.write_text("PartitionName=regular_milan_ss11 Default=YES MaxMemPerNode=UNLIMITED")
    S.forget()
    c = executor.caps()
    assert c["mem_mb"] == 485100 and "MemSpecLimit" in c["mem_from"], c
    big = dict(COUETTE, run={"procs": 512, "queue": True, "mem": "480G"})
    assert any("485100M" in e for e in executor.run_errors(big))
    parts.write_text("PartitionName=regular_milan_ss11 Default=YES MaxMemPerNode=460000")
    S.forget()
    c = executor.caps()
    assert c["mem"] == "460000M" and "MaxMemPerNode" in c["mem_from"], c
    S.forget()


def test_init_pack_carries_a_run_to_another_machine(tmp_path, monkeypatch):
    """init_from a run of another machine (2026.11.13.6): `cfdt pack REQUEST --studies p8 --init` there, `cfdt import`
    here; the imported study answers init_from '<imported>/<study>' and the same mesh copies its fields."""
    from test_geometry_router import _case, _field

    from cfd_translator import transient
    from cfd_translator.transfer import init_check
    there, here = tmp_path / "spark2" / "cfdt_runs", tmp_path / "spark1" / "cfdt_runs"
    monkeypatch.setenv("CFDT_OUT", str(there))
    req = there / "run_sat"
    fields = {f: _field(f, "nonuniform List<scalar> 2(1 2)", "inlet { type src; }")
              for f in ("U", "p_rgh", "alpha.water", "k", "omega", "nut")}
    for n, t in (("p8", "4000"), ("p6", None)):
        _case(req / n / "case", fields if t else {}, t=t or "0")
        (req / n / "case" / "constant" / "triSurface").mkdir()
        (req / n / "case" / "constant" / "triSurface" / "pump.stl").write_bytes(b"x" * 1000)
        (req / n / "case" / "constant" / "g").write_text("g")
        (req / n / "case" / "processor0" / "4000").mkdir(parents=True)
        (req / n / "case" / "processor0" / "4000" / "U").write_text("decomposed")
        (req / n / "case" / "system" / "controlDict").write_text("cd")
    (req / "p8" / "p8.json").write_text(json.dumps({"name": "p8", "ok": True, "reliable": True,
                                                    "path": str(req / "p8")}))
    (req / "request.json").write_text(json.dumps({"names": ["p8", "p6"], "studies": {"p8": {}, "p6": {}}}))
    c = init_check("run_sat/p8")
    assert c["ok"] and c["finished"] and c["time"] == 4000 and c["init_bytes"] > 0
    c6 = init_check("run_sat/p6")
    assert not c6["ok"] and any("no result yet" in p for p in c6["problems"])
    assert "error" in pack(str(req), studies=["p6"], init=True)                 # not finished: nothing packed
    p = pack(str(req), out=str(tmp_path / "p8.tgz"), studies=["p8"], init=True)
    assert p["init"]["p8"]["time"] == 4000 and p["studies_with_results"] == ["p8"]
    with tarfile.open(p["archive"]) as tar:
        names = tar.getnames()
        meta = json.loads(tar.extractfile("run_sat/request.json").read())
    assert "run_sat/p8/case/4000/alpha.water" in names and "run_sat/p8/case/constant/polyMesh/faces" in names
    assert "run_sat/p8/case/system/controlDict" in names and "run_sat/p8/case/constant/g" in names
    assert not any("triSurface" in n or "processor0" in n or "/p6/" in n for n in names)
    assert meta["names"] == ["p8"] and list(meta["studies"]) == ["p8"]
    # here: imported under its own name (the router picks it), then a run starts from it
    monkeypatch.setenv("CFDT_OUT", str(here))
    got = import_request(p["archive"], "run_sat@spark2")
    assert got["init_from"] == {"p8": "run_sat@spark2/p8"} and got["not_finished"] == []
    assert transient.init_problems({"solve": {"init_from": got["init_from"]["p8"]}}) == []
    assert "no case" in transient.init_problems({"solve": {"init_from": "run_sat/p8"}})[0]
    dst = here / "run_low" / "p4" / "case"
    _case(dst, {f: _field(f, "uniform 0") for f in ("U", "p_rgh", "alpha.water", "k", "omega", "nut")})
    notes, info = transient.map_from(dst, "run_sat@spark2/p8", "last", lambda c, l: pytest.fail("same mesh"))
    assert info["same_mesh"] and info["copied_from"] and "copied here" in notes[0]
    assert "List<scalar> 2(1 2)" in (dst / "0" / "alpha.water").read_text()
    from cfd_translator.cli import exit_code
    assert exit_code("init-check", {"ok": False}) == 1 and exit_code("init-check", {"ok": True}) == 0
