"""Tracers and the mean age (scalars.py): validation, case files, and the results from synthetic fields."""
from __future__ import annotations

import numpy as np
import pytest

from cfd_translator import foamio
from cfd_translator.format import validate_study
from cfd_translator.runner import prepare
from cfd_translator.scalars import results

CH = {"geometry": {"box": [100, 10, 1]}, "2d": True, "fluid": "water", "model": "laminar",
      "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.01}, {"kind": "outlet", "faces": {"nx": 1}},
              {"kind": "slip", "faces": [{"ny": 1}, {"ny": -1}]}],
      "mesh": {"cells": [50, 5, 1]}}


def test_validation():
    ok = dict(CH, scalars=[{"name": "dye", "inlets": {"inlet": 1}, "Sc": 2}], age=True)
    assert validate_study(ok)["ok"], validate_study(ok)
    bad = dict(CH, scalars=[{"name": "U", "inlets": {"outlet": 1}, "Sc": 1, "diffusivity": 1e-9}], age={"Sc": -1})
    e = " ".join(validate_study(bad)["errors"])
    for w in ("scalars[0].name", "not an inlet bc", "Sc or diffusivity", "age.Sc"):
        assert w in e, w
    assert not validate_study(dict(CH, images=[{"slice": {"axis": "z", "at": 0.5}, "field": "dye"}]))["ok"]
    assert validate_study(dict(ok, images=[{"slice": {"axis": "z", "at": 0.5}, "field": "dye"},
                                           {"slice": {"axis": "z", "at": 0.5}, "field": "age"}]))["ok"]


def test_case_files(tmp_path):
    st = dict(CH, scalars=[{"name": "dye", "inlets": {"inlet": 1}, "diffusivity": 1e-9}], age=True)
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    c = tmp_path / "case"
    dye = foamio.read_foam((c / "0" / "dye").read_text())["boundaryField"]
    assert dye["inlet"]["type"] == "fixedValue" and float(dye["inlet"]["value"]) == 1
    assert dye["outlet"]["type"] == "inletOutlet" and dye["slip"]["type"] == "zeroGradient"
    age = foamio.read_foam((c / "0" / "age").read_text())
    assert age["dimensions"].replace(" ", "") in ("[0010000]",) or "0 0 1" in str(age["dimensions"])
    cd = (c / "system" / "controlDict").read_text()
    assert "type            scalarTransport;" in cd and "libsolverFunctionObjects.so" in cd
    assert "diffusivity     constant;" in cd and "D               1e-09;" in cd       # laminar with a diffusivity
    assert "diffusivity     viscosity;" in cd and "alphal          1;" in cd         # age: Sc 1
    assert cd.count("writeControl    writeTime;") >= 2
    fs = (c / "system" / "fvSchemes").read_text()
    assert "div(phi,dye)  bounded Gauss limitedLinear01 1;" in fs and "div(phi,age)  bounded Gauss limitedLinear 1;" in fs
    fv = foamio.read_foam((c / "system" / "fvSolution").read_text())
    assert "(dye|age)" in fv["solvers"] and float(fv["relaxationFactors"]["equations"]["dye"]) == 1
    fm = foamio.read_foam((c / "constant" / "fvModels").read_text())["cfdtAgeSource"]
    assert fm["type"] == "semiImplicitSource" and fm["cellZone"] == "all" and fm["volumeMode"] == "specific"
    assert float(fm["sources"]["age"]["explicit"]) == 1
    # without scalars: no fields, no fvModels left behind
    prepare(CH, tmp_path)
    assert not (c / "0" / "dye").exists() and not (c / "constant" / "fvModels").exists()


def _plug(tmp_path, st, cfun):
    """Plug flow at U = 0.01 m/s along x through the channel, with scalar fields from cfun(x [m])."""
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    g = mesh.geometry()
    U = 0.01
    pf = {}
    sc = {n: {"internal": cfun(n, g["C"][:, 0] * 1e-3), "patches": {}} for n in plan["scalars"]}
    for P in mesh.patches:
        f = np.arange(P["start"], P["start"] + P["n"])
        Sf = g["Sf"][f] * 1e-6
        phi = Sf @ np.array([U, 0, 0])
        pf[P["name"]] = {"phi": phi, "nut": None, "U": np.tile([U, 0, 0], (len(f), 1)), "p": np.zeros(len(f))}
        for n in plan["scalars"]:
            if P["name"] == "inlet":
                sc[n]["patches"]["inlet"] = np.full(len(f), plan["scalars"][n]["inlets"].get("inlet", 0.0))
            elif P["name"] == "outlet":
                sc[n]["patches"]["outlet"] = cfun(n, g["Cf"][f, 0] * 1e-3)
    fields = {"U": np.tile([U, 0, 0], (mesh.ncells, 1)), "p": np.zeros(mesh.ncells), "patches": pf, "scalars": sc}
    out, warns = {"checks": {}}, []
    results(st, mesh, plan, fields, out, warns)
    return out, warns, plan


def test_results_from_plug_flow(tmp_path):
    st = dict(CH, scalars=[{"name": "dye", "inlets": {"inlet": 1}, "diffusivity": 1e-15}], age={"Sc": 1e12})
    out, warns, plan = _plug(tmp_path, st, lambda n, x: np.ones_like(x) if n == "dye" else (x + 0.05) / 0.01)
    d = out["scalars"]["dye"]
    assert d["outlets"]["outlet"]["mean"] == pytest.approx(1) and d["outlets"]["outlet"]["mixing_index"] == pytest.approx(1)
    assert out["checks"]["scalar_balance"]["dye"]["error"] < 1e-9
    a = out["age"]
    assert a["V_over_Q_s"] == pytest.approx(10.0) and a["mean_residence_time_s"] == pytest.approx(10.0)
    assert out["checks"]["age_balance"]["error"] < 1e-6 and warns == []
    assert a["volume_fraction_older_than_2tau"] == 0


def test_imbalance_is_reported(tmp_path):
    st = dict(CH, scalars=[{"name": "dye", "inlets": {"inlet": 1}, "diffusivity": 1e-15}])
    out, warns, _ = _plug(tmp_path, st, lambda n, x: np.full_like(x, 0.9))          # 10 % lost
    assert out["checks"]["scalar_balance"]["dye"]["error"] == pytest.approx(0.1, rel=1e-6)
    assert any("balance off" in w for w in warns)
