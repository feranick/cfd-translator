"""sections (2026.11.13.7): flow through internal surfaces from the face fluxes, liquid and gas apart. The S50 air
sweep's question: does the flow (and the gas) through the wear-plate holes change with suction pressure?
Runs without OpenFOAM (a box mesh and a flux field written by hand)."""
from __future__ import annotations

import numpy as np
import pytest

from cfd_translator.format import validate_study
from cfd_translator.polymesh import box_mesh
from cfd_translator.runner import _sections_into
from cfd_translator.sections import errors

HOLES = [{"name": "h1", "center": [5, 5, 0], "radius": 3.2}, {"name": "h2", "center": [14, 14, 0], "radius": 3.2}]


def _flow():
    """A 20 mm box, 1 mm cells; U = (0, 0, w(x, y)) (divergence-free): w 2 m/s in the first hole's columns,
    -1 in the second's, 0 elsewhere; the liquid fraction 0.8 in the first hole, 1 elsewhere."""
    m = box_mesh(20, 20, 20, 20, 20, 20)
    g = m.geometry()
    C = g["C"]
    ni = len(m.neighbour)

    def w(xy):
        r1 = np.hypot(xy[:, 0] - 5, xy[:, 1] - 5)
        r2 = np.hypot(xy[:, 0] - 14, xy[:, 1] - 14)
        return np.where(r1 <= 3, 2.0, np.where(r2 <= 3, -1.0, 0.0))
    phi = w(g["Cf"][:ni, :2]) * g["Sf"][:ni, 2] * 1e-6
    alpha = np.where(np.hypot(C[:, 0] - 5, C[:, 1] - 5) <= 3, 0.8, 1.0)
    cols = C[(C[:, 2] > 9) & (C[:, 2] < 10)]
    q1 = 2.0 * 1e-6 * (np.hypot(cols[:, 0] - 5, cols[:, 1] - 5) <= 3).sum()
    q2 = -1.0 * 1e-6 * (np.hypot(cols[:, 0] - 14, cols[:, 1] - 14) <= 3).sum()
    return m, phi, alpha, q1, q2


def test_flow_through_two_holes_liquid_and_gas():
    m, phi, alpha, q1, q2 = _flow()
    plan = {"vof": {"rho_l": 998.2, "gas": {"R": 287.0, "T_K": 293.15}}}
    fields = {"phi_internal": phi, "alpha": alpha, "p_abs": np.full(m.ncells, 50000.0), "time": "3000"}
    st = {"sections": [{"name": "holes", "plane": {"axis": "z", "at": 10.3}, "within": HOLES},
                       {"name": "whole", "plane": {"point": [0, 0, 4.2], "normal": [0, 0, -2]}}]}
    res = {}
    w = _sections_into(st, m, plan, fields, res)
    s = res["sections"]["holes"]
    assert s["flow_m3s"] == pytest.approx(q1 + q2) and s["parts"]["h1"]["flow_m3s"] == pytest.approx(q1)
    assert s["parts"]["h2"]["backward_m3s"] == pytest.approx(q2) and s["parts"]["h2"]["forward_m3s"] == 0
    h1 = s["parts"]["h1"]
    assert h1["liquid_m3s"] == pytest.approx(0.8 * q1) and h1["gas_m3s"] == pytest.approx(0.2 * q1, rel=1e-4)
    assert h1["gas_kg_s"] == pytest.approx(0.2 * q1 * 50000 / (287 * 293.15), rel=1e-3)
    assert h1["gas_volume_fraction"] == pytest.approx(0.2, rel=1e-3) and h1["mean_velocity_m_s"] > 0
    assert "upwind" in h1["alpha_from"]
    # the plane across the whole box, normal -z: the same net flow, the other sign; no ring to check
    assert res["sections"]["whole"]["flow_m3s"] == pytest.approx(-(q1 + q2))
    assert res["sections"]["whole"]["area_mm2"] == pytest.approx(400)
    # the discs (3.2 mm) leave no fluid crossing just outside them? here the box is fluid everywhere: warned
    assert any("cuts fluid" in x for x in w) and res["sections"]["holes"]["ring_outside"]["faces"] > 0
    # the solver's liquid flux, when written, is used as it is
    fields["alphaPhi_internal"] = 0.5 * phi
    _sections_into(st, m, plan, fields, res)
    assert res["sections"]["holes"]["parts"]["h1"]["liquid_m3s"] == pytest.approx(0.5 * q1)


def test_no_internal_flux_and_validation():
    m, phi, alpha, q1, q2 = _flow()
    res = {}
    w = _sections_into({"sections": [{"name": "a", "plane": {"axis": "z", "at": 5.5}}]}, m, {}, {}, res)
    assert "sections" not in res and "could not be read" in w[0]
    _sections_into({"sections": [{"name": "a", "plane": {"axis": "z", "at": 5.5}}]}, m, {}, {"phi_internal": phi}, res)
    assert res["sections"]["a"]["flow_m3s"] == pytest.approx(q1 + q2) and "liquid_m3s" not in res["sections"]["a"]
    assert errors({}) == [] and errors({"sections": [{"name": "a", "plane": {"axis": "z", "at": 1},
                                                      "within": {"box": [0, 0, 0, 1, 1, 1]}}]}) == []
    bad = errors({"sections": [{"name": "a", "plane": {"axis": "w", "at": 1}},
                               {"name": "a", "plane": {"point": [0, 0, 0], "normal": [0, 0, 0]}},
                               {"name": "b", "plane": {"axis": "z", "at": 0}, "within": [{"center": [0, 0], "radius": 1}]},
                               {"name": "c", "plane": {"axis": "z", "at": 0}, "within": {"box": [1, 0, 0, 0, 1, 1]}},
                               {"name": "d", "plane": {"axis": "z", "at": 0}, "where": 1}]})
    assert len(bad) == 6 and any("twice" in e for e in bad) and any("unknown key 'where'" in e for e in bad)
    st = {"geometry": {"box": [20, 20, 20]}, "fluid": "water",
          "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nz": 1}}],
          "sections": [{"name": "holes", "plane": {"axis": "z", "at": 10}, "within": HOLES}]}
    assert validate_study(st)["ok"], validate_study(st)
    assert not validate_study(dict(st, sections={"name": "x"}))["ok"]


def test_read_fields_takes_the_internal_fluxes(tmp_path):
    from cfd_translator.foam import read_fields
    m = box_mesh(3, 1, 1, 3, 1, 1)
    ni = len(m.neighbour)
    t = tmp_path / "case" / "100"
    t.mkdir(parents=True)

    def vol(name, cls, internal):
        return (f"FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {name};\n}}\ndimensions [0 0 0 0 0 0 0];\n"
                f"internalField {internal};\nboundaryField\n{{\n}}\n")
    (t / "U").write_text(vol("U", "volVectorField", "uniform (1 0 0)"))
    (t / "p").write_text(vol("p", "volScalarField", "uniform 0"))
    (t / "phi").write_text(vol("phi", "surfaceScalarField", f"nonuniform List<scalar> {ni}(" +
                               " ".join(f"{1e-6 * (k + 1):g}" for k in range(ni)) + ")"))
    f = read_fields(tmp_path / "case", m, {}, internal_flux=True)
    assert f["phi_internal"] == pytest.approx([1e-6 * (k + 1) for k in range(ni)])
    assert "phi_internal" not in read_fields(tmp_path / "case", m, {})


S50 = [{"name": "wear_plate_holes", "plane": {"point": [0, 0, -40], "normal": [0, 0, -1]},
        "within": [{"name": "h1", "center": [-15.45, -47.55, -40], "radius": 5},
                   {"name": "h2", "center": [0, -53, -40], "radius": 5},
                   {"name": "h3", "center": [15.45, -47.55, -40], "radius": 5}]},
       {"name": "bottom_opening", "plane": {"point": [0, -59.25, -27.75], "normal": [0, -1, 0]},
        "within": [{"name": "opening", "box": [-14, -62, -34, 14, -56.5, -21.5]}]}]


def test_named_boxes_in_the_within_list():
    """2026.11.13.8: a within list takes boxes beside discs (the S50 bottom opening: the 'no opening' plug's box)."""
    assert errors({"sections": S50}) == []
    assert errors({"sections": [{"name": "a", "plane": {"axis": "y", "at": 0},
                                 "within": [{"name": "o", "box": [0, 0, 0, 1, 1, 1], "radius": 2}]}]})
    m, phi, alpha, q1, q2 = _flow()
    res = {}
    st = {"sections": [{"name": "holes", "plane": {"axis": "z", "at": 10.3},
                        "within": [HOLES[0], {"name": "b2", "box": [10.5, 10.5, 9, 17.5, 17.5, 11]}]}]}
    _sections_into(st, m, {}, {"phi_internal": phi}, res)
    s = res["sections"]["holes"]
    assert s["parts"]["h1"]["flow_m3s"] == pytest.approx(q1) and s["parts"]["b2"]["flow_m3s"] == pytest.approx(q2)


def test_init_check_fields_mtime_ignores_a_reprocess(tmp_path, monkeypatch):
    import json
    import os
    import time

    from test_geometry_router import _case, _field

    from cfd_translator.transfer import init_check
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    d = tmp_path / "run_a" / "p12"
    _case(d / "case", {"U": _field("U", "uniform 0")}, t="3000")
    old = time.time() - 3600
    for f in [d / "case" / "3000", *(d / "case" / "3000").iterdir()]:
        os.utime(f, (old, old))
    (d / "p12.json").write_text(json.dumps({"ok": True}))           # written (reprocessed) now
    c = init_check("run_a/p12")
    assert c["ok"] and c["fields_mtime"] == pytest.approx(old, abs=1) and c["fields_mtime"] < (d / "p12.json").stat().st_mtime
