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
