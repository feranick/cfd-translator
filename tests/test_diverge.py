"""diverge.py: the worst-cell report of a diverged run, and the field write before the kill (foam._wait)."""
import json
import subprocess
import time

import numpy as np
import pytest

from cfd_translator import diverge, foam, foamio
from cfd_translator.locator import write_frame
from cfd_translator.polymesh import from_cells


def box_mesh(n=6, h=2.0, x0=-6.0):
    """n^3 hex cells of h mm from x0 in x, y, z."""
    g = np.arange(n + 1) * h + x0
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    hexes = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                hexes.append([idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
                              idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1]])
    m = from_cells(pts, {"hex": np.array(hexes)}, lambda c: [1] * len(c))
    m.set_patches([{"name": "walls", "type": "wall", "tags": [1]}])
    return m


def write_U(path, U):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"({float(a):.10g} {float(b):.10g} {float(c):.10g})" for a, b, c in U)
    path.write_text("FoamFile\n{\n    format ascii;\n    class volVectorField;\n    object U;\n}\n"
                    "dimensions [0 1 -1 0 0 0 0];\n"
                    f"internalField nonuniform List<vector> {len(U)}\n(\n{body}\n);\n"
                    "boundaryField\n{\n    walls { type noSlip; }\n}\n")


def make_case(tmp_path, hot_at=(3.0, 3.0, 3.0), speed=1e5, nan=False, decomposed=False):
    case = tmp_path / "case"
    m = box_mesh()
    root = case / "processor0" if decomposed else case
    m.write(root)
    g = m.geometry()
    U = np.zeros((m.ncells, 3))
    U[:, 0] = 1.0
    c = int(np.argmin(np.linalg.norm(g["C"] - np.asarray(hot_at), axis=1)))
    U[c] = [speed, 0, 0]
    if nan:
        U[0] = [np.nan, np.nan, np.nan]
    write_U(root / "0.000115" / "U", U)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text("deltaT          5.7e-05;\nstopAt          endTime;\n")
    (case / "log.foamRun").write_text("Time = 5.7e-05\n\ndeltaT = 5.7e-05\nTime = 0.000115\n")
    return case, c, g


def test_report_finds_the_hot_cell_in_the_rotor_frame(tmp_path):
    case, c, g = make_case(tmp_path)
    write_frame(case, [0, 0, 1], [0, 0, 0], radius=4.0, span=[-6, 6], omega=0.0)
    rep = diverge.report(case)
    w = rep["worst_courant"][0]
    assert w["cell"] == c and w["at_mm"] == [3.0, 3.0, 3.0]
    # Co = 0.5 sum |U.S| dt / V: the hot cell's faces (x faces at the face average) carry the flux
    assert w["Co"] == pytest.approx(0.5 * 2 * (1e5 + 1) / 2 * 4e-6 * 5.7e-5 / 8e-9, rel=1e-3)
    assert w["r_mm"] == pytest.approx(np.hypot(3, 3), abs=1e-3) and w["s_mm"] == 3.0 and w["theta_deg"] == 45.0
    assert w["dr_cylinder_mm"] == pytest.approx(np.hypot(3, 3) - 4, abs=1e-3) and w["in_zone"] is False
    assert w["size_mm"] == 2.0 and w["thickness_mm"] == 2.0
    assert rep["fastest"][0]["speed_m_s"] == 1e5
    assert rep["time"] == 0.000115 and rep["dt"] == 5.7e-5 and rep["cells"] == 216
    assert rep["hot"]["cells"] >= 1 and rep["hot"]["near_interface"] >= 1
    s = diverge.summary(rep)
    assert "worst cell: Courant" in s and "r 4.243 s 3.0 theta 45.0" in s and "from the zone's cylinder" in s


def test_report_decomposed_nan_and_rotor_relative(tmp_path):
    case, c, g = make_case(tmp_path, speed=50.0, nan=True, decomposed=True)
    write_frame(case, [0, 0, 1], [0, 0, 0], radius=100.0, span=[-100, 100], omega=1000.0)
    rep = diverge.report(case)
    assert rep["nonfinite_cells"] == 1
    assert rep["worst_courant"][0]["Co"] is None and rep["worst_courant"][0]["proc"] == 0
    assert "processor0" in rep["fields"] and "relative to the rotor" in rep["note"]
    assert "1 cells have a NaN" in diverge.summary(rep)


def test_report_without_fields(tmp_path):
    assert diverge.report(tmp_path) is None
    assert diverge.summary(None) == ""


def test_attach_keeps_the_report(tmp_path):
    case, c, g = make_case(tmp_path)
    exc = diverge.attach(foam.Diverged("the solution diverged at time 0.000115"), case)
    assert isinstance(exc, foam.Diverged) and exc.report["worst_courant"][0]["cell"] == c
    assert str(exc).startswith("the solution diverged") and "Fields written at time 0.000115" in str(exc)
    assert json.loads((case / diverge.FILE).read_text())["cells"] == 216


def test_read_points_ascii(tmp_path):
    m = box_mesh(n=2)
    d = m.write(tmp_path)
    assert np.allclose(foamio.read_points(d / "points") * 1e3, m.points)


def test_wait_asks_for_the_fields_before_the_kill(tmp_path, monkeypatch):
    """On divergence a transient run is asked to write (stopAt writeNow) and given time to do it."""
    monkeypatch.setattr(foam, "WATCH_S", 0.2)
    log = tmp_path / "log.foamRun"
    log.write_text("Time = 1\nCourant Number mean: 1 max: 5e7\n")
    asked = []

    def stop():
        asked.append(time.time())
        (tmp_path / "stop").write_text("")
    p = subprocess.Popen(["bash", "-c", f"while [ ! -f '{tmp_path}/stop' ]; do sleep 0.1; done; echo wrote > "
                                        f"'{tmp_path}/wrote'"], start_new_session=True)
    with pytest.raises(foam.Diverged):
        foam._wait(p, "x", 60, None, None, stop, log)
    assert asked and (tmp_path / "wrote").read_text().strip() == "wrote"


def test_wait_kills_when_the_write_takes_too_long(tmp_path, monkeypatch):
    monkeypatch.setattr(foam, "WATCH_S", 0.2)
    monkeypatch.setenv("CFDT_DIVERGE_WRITE_S", "0.5")
    log = tmp_path / "log.foamRun"
    log.write_text("Time = 1\nCourant Number mean: 1 max: 5e7\n")
    p = subprocess.Popen(["bash", "-c", "sleep 30"], start_new_session=True)
    t0 = time.time()
    with pytest.raises(foam.Diverged):
        foam._wait(p, "x", 60, None, None, lambda: None, log)
    assert time.time() - t0 < 10 and p.poll() is not None
