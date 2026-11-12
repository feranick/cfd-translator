"""Pump cavitation work: k and nut as image fields; a transient run's pressure statistics over its averaging window
(p rms from pPrime2Mean; the fraction of time below absolute pressure thresholds, a coded function object's
pBelowTime_<Pa> fields) as image fields and per-region summaries; kOmegaSSTDES with a sliding zone.
OpenFOAM is not needed (the coded function object compiles on the machines)."""
from __future__ import annotations

import numpy as np
import pytest

from cfd_translator import foamio, transient
from cfd_translator.case import _field, _nonuniform, _uniform
from cfd_translator.foam import read_fields
from cfd_translator.format import validate_study
from cfd_translator.runner import prepare

CH = {"geometry": {"box": [1000, 100, 10]}, "2d": True, "fluid": {"rho": 1000.0, "mu": 1e-3},
      "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.01}, {"kind": "outlet", "faces": {"nx": 1}}],
      "mesh": {"cells": [40, 16, 1]}, "reference": {"velocity": 0.01, "area": 1000, "length": 100},
      "conditions": {"p": 30000},
      "solve": {"transient": {"end_time": 20.0, "average_from": 10.0, "p_below": [2339, 20000],
                              "p_below_regions": [{"name": "front", "box": [-600, -60, -10, 0, 60, 20]},
                                                  {"name": "core", "cylinder": {"center": [0, 0, 5], "axis": [0, 0, 1],
                                                                                "radius": 30, "span": [-10, 10]}}]}}}


def test_validation():
    assert validate_study(CH)["ok"], validate_study(CH)["errors"]
    for bad in ([], [-1], [2339, 2339], "x"):
        st = dict(CH, solve={"transient": {"end_time": 20.0, "p_below": bad}})
        assert any("p_below" in e for e in validate_study(st)["errors"]), bad
    st = dict(CH, solve={"transient": {"end_time": 20.0, "p_below_regions": [{"name": "a", "box": [0, 0, 0, 1, 1, 1]}]}})
    assert any("goes with p_below" in e for e in validate_study(st)["errors"])
    st = dict(CH, solve={"transient": {"end_time": 20.0, "p_below": [2339], "p_below_regions": [{"name": "domain"}]}})
    assert any("p_below_regions[0]" in e for e in validate_study(st)["errors"])
    # image fields
    ok = dict(CH, model="kOmegaSST", images=[{"slice": {"axis": "z", "at": 5}, "field": f}
                                              for f in ("k", "nut", "p_rms", "p_below_2339", "p_below_20000")])
    ok["2d"] = True
    assert validate_study(ok)["ok"], validate_study(ok)["errors"]
    lam = dict(CH, images=[{"slice": {"axis": "z", "at": 5}, "field": "k"}])
    assert any(".field" in e for e in validate_study(lam)["errors"])                # laminar: no k


def test_case_has_the_statistics(tmp_path):
    mesh, patches, sel, plan, w = prepare(CH, tmp_path)
    tr = plan["transient"]
    assert tr["p_below"] == [2339.0, 20000.0] and tr["p_operating"] == 30000.0
    cd = (tmp_path / "case" / "system" / "controlDict").read_text()
    assert "p { mean on; prime2Mean on; }" in cd
    assert "type            coded;" in cd and "libutilityFunctionObjects.so" in cd and "cfdtPBelow" in cd
    assert "thresholds      (2339 20000);" in cd and "names           (pBelowTime_2339 pBelowTime_20000);" in cd
    assert "averageFrom     10;" in cd and "pOperating      30000;" in cd and "rho             1000;" in cd
    assert "pos0(thr_[i] - pa)" in cd and "IOobject::AUTO_WRITE" in cd and "dict.lookup<scalarList>" in cd
    # balanced braces and code blocks
    i = cd.index("cfdtPBelow")
    blk = cd[i:]
    assert blk.count("#{") == blk.count("#};") == 4


def _write(case, t, mesh, patches, name, cls, dims, internal):
    bf = {p["name"]: {"type": "calculated", "value": _uniform(0 if cls != "volVectorField" else [0, 0, 0])}
          for p in patches if p["type"] != "empty"}
    bf.update({p["name"]: {"type": "empty"} for p in patches if p["type"] == "empty"})
    (case / t / name).write_text(_field(cls, name, dims, internal, bf))


def test_fields_and_summaries(tmp_path):
    mesh, patches, sel, plan, w = prepare(dict(CH, model="kOmegaSST", solve={"transient": dict(
        CH["solve"]["transient"], init=0)}), tmp_path)
    case = tmp_path / "case"
    (case / "20").mkdir()
    nc = mesh.ncells
    C = mesh.geometry()["C"]
    _write(case, "20", mesh, patches, "U", "volVectorField", "[0 1 -1 0 0 0 0]", _uniform([0.01, 0, 0]))
    _write(case, "20", mesh, patches, "p", "volScalarField", "[0 2 -2 0 0 0 0]", _uniform(0))
    _write(case, "20", mesh, patches, "phi", "surfaceScalarField", "[0 3 -1 0 0 0 0]", _uniform(0))
    _write(case, "20", mesh, patches, "k", "volScalarField", "[0 2 -2 0 0 0 0]", _nonuniform(np.linspace(0.1, 1, nc)))
    _write(case, "20", mesh, patches, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", _uniform(1e-4))
    _write(case, "20", mesh, patches, "pPrime2Mean", "volScalarField", "[0 4 -4 0 0 0 0]", _uniform(4.0))
    front = C[:, 0] < 0
    _write(case, "20", mesh, patches, "pBelowTime_2339", "volScalarField", "[0 0 1 0 0 0 0]",
           _nonuniform(np.where(front, 5.0, 0.0)))                       # half of the 10 s window, in the front half
    _write(case, "20", mesh, patches, "pBelowTime_20000", "volScalarField", "[0 0 1 0 0 0 0]", _uniform(10.0))
    f = read_fields(case, mesh, plan, mean=True)
    sc = f["scalars"]
    assert np.allclose(sc["k"]["internal"], np.linspace(0.1, 1, nc)) and np.allclose(sc["nut"]["internal"], 1e-4)
    assert np.allclose(sc["p_rms"]["internal"], 2000.0)                 # rho sqrt(p'2): 1000 x 2 Pa
    assert np.allclose(sc["p_below_2339"]["internal"], np.where(front, 0.5, 0.0))
    assert np.allclose(sc["p_below_20000"]["internal"], 1.0) and f["p_stats_window_s"] == 10.0
    ps = transient.pressure_statistics(CH, mesh, plan, f)
    assert ps["p_rms_Pa"]["max"] == pytest.approx(2000) and ps["window_s"] == 10.0
    d = ps["below"]["2339"]
    V = mesh.geometry()["V"] * 1e-9
    assert d["domain"]["mean_fraction"] == pytest.approx(0.25, rel=1e-3)                 # half the time, half the volume
    assert d["front"]["mean_fraction"] == pytest.approx(1 * 0.5, rel=1e-3) and d["front"]["max_fraction"] == 0.5
    assert d["domain"]["volume_time_m3s"] == pytest.approx(float((V[front] * 0.5).sum() * 10), rel=1e-3)
    assert d["core"]["mean_fraction"] > 0 and ps["below"]["20000"]["domain"]["volume_over_10pct_mm3"] > 0
    # pictures of them
    from cfd_translator.render import render
    imgs, warns = render(CH, mesh, plan, f, tmp_path / "img",
                         [{"name": n, "slice": {"axis": "z", "at": 5}, "field": n} for n in ("k", "nut", "p_rms",
                                                                                              "p_below_2339")])
    assert not warns, warns
    assert imgs["k"]["max"] == pytest.approx(1.0, rel=0.05) and "fraction of time with p below 2339" in imgs["p_below_2339"]["what"]


def test_des_with_a_sliding_zone(tmp_path):
    """kOmegaSSTDES with a sliding zone: validated, the transient files LES with the sliding controls."""
    from test_sliding import ZONE, _study, patches_for, pump_mesh, zone_of
    from cfd_translator.case import write_case
    st = _study(max_co=None)
    st["model"] = "kOmegaSSTDES"
    st["solve"]["transient"]["init"] = 0
    st["solve"]["transient"]["p_below"] = [2339]
    assert validate_study(st)["ok"], validate_study(st)["errors"]
    m = pump_mesh()
    st["rotating"] = [dict(ZONE)]
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    plan = write_case(st, m, patches_for(m), tmp_path / "case")
    transient.write_transient(tmp_path / "case", st, plan)
    mt = (tmp_path / "case" / "constant" / "momentumTransport").read_text()
    assert "kOmegaSSTDES" in mt and "LES" in mt
    fv = (tmp_path / "case" / "system" / "fvSolution").read_text()
    assert "correctPhi      yes;" in fv and "MeshPhi" in fv
    cd = (tmp_path / "case" / "system" / "controlDict").read_text()
    assert "adjustTimeStep  no;" in cd and "cfdtPBelow" in cd and foamio is not None
