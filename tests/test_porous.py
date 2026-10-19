"""Porous zones (porous.py): validation, coefficients, case files, momentum source and results."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cfd_translator import bench, foamio
from cfd_translator.format import validate_study
from cfd_translator.porous import coefficients, momentum_source, results
from cfd_translator.runner import prepare


def _v(x):
    import re
    if isinstance(x, str):
        return [float(t) for t in re.findall(r"[-+0-9.eE]+", x)]
    return [float(t) for t in x]


def _study(zone):
    return bench._cases()["porous_channel_1d"]["study"] | {"porous": [dict({"name": "bed", "region": {"box": [-25, -10, -1, 25, 10, 2]}}, **zone)]}


def test_validation():
    assert validate_study(_study({"darcy": 1e8}))["ok"]
    bad = _study({"darcy": 1e8, "ergun": {"particle_diameter": 1, "porosity": 1.5}})
    e = " ".join(validate_study(bad)["errors"])
    assert "give one of" in e and "porosity: 0..1" in e
    e = " ".join(validate_study(_study({"monolith": {"channel": 1, "open_fraction": 0.7}}))["errors"])
    assert "monolith needs axes" in e
    e = " ".join(validate_study(_study({"darcy": 1e8, "axes": [[1, 0, 0], [1, 1, 0]]}))["errors"])
    assert "perpendicular" in e


def test_coefficients():
    d, f, how = coefficients({"ergun": {"particle_diameter": 1.0, "porosity": 0.4}})
    assert d[0] == pytest.approx(150 * 0.36 / (0.064 * 1e-6)) and f[0] == pytest.approx(3.5 * 0.6 / (0.064 * 1e-3))
    d, f, how = coefficients({"monolith": {"channel": 1.0, "open_fraction": 0.5, "shape": "round"}, "axes": [[1, 0, 0], [0, 1, 0]]})
    assert d[0] == pytest.approx(32 / (0.5 * 1e-6)) and d[1] == pytest.approx(1000 * d[0]) and not f.any()


def test_case_files_and_zone(tmp_path):
    s2 = math.sqrt(0.5)
    mesh, patches, sel, plan, w = prepare(_study({"darcy": [1e8, 1e7, 1e7], "forchheimer": 100,
                                                  "axes": [[s2, s2, 0], [-s2, s2, 0]]}), tmp_path)
    c = tmp_path / "case"
    zp = plan["porous"]["bed"]
    assert zp["cells"] == 100 * 4 and zp["volume_mm3"] == pytest.approx(50 * 10 * 1)        # x in [-25, 25]
    zones = (c / "constant" / "polyMesh" / "cellZones").read_text()
    assert "bed" in zones
    fm = foamio.read_foam((c / "constant" / "fvModels").read_text())["porous_bed"]
    assert fm["type"] == "porosityForce"
    pf = fm["porosityForce"]
    assert pf["type"] == "fixedCoeff" and pf["cellZone"] == "bed" and float(pf["rhoRef"]) == 1   # kinematic equation
    assert _v(pf["alpha"]) == pytest.approx([1e-6 * 1e8, 1e-6 * 1e7, 1e-6 * 1e7])  # nu d
    assert _v(pf["beta"]) == pytest.approx([50, 50, 50])                        # f / 2
    rot = pf["coordinateSystem"]["coordinateRotation"]
    assert rot["type"] == "axesRotation" and _v(rot["e3"]) == pytest.approx([0, 0, 1])


def test_momentum_source_and_results_from_a_linear_pressure_drop(tmp_path):
    st = _study({"darcy": 1e8, "forchheimer": 1e3})
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    g = mesh.geometry()
    U = np.tile([0.01, 0, 0], (mesh.ncells, 1))
    F = momentum_source(plan, mesh, U)
    dp = 0.05 * (1e-3 * 1e8 * 0.01 + 0.5 * 1000 * 1e3 * 1e-4)
    assert F[:, 0].sum() == pytest.approx(-dp * 1e-5) and abs(F[:, 1:]).max() == 0
    x = g["C"][:, 0]
    p = np.where(x < -25, dp, np.where(x > 25, 0.0, dp * (25 - x) / 50)) / 1000.0       # kinematic
    fields = {"U": U, "p": p, "patches": {}}
    out, warns = {}, []
    results(st, mesh, plan, fields, out, warns)
    r = out["porous"]["bed"]
    assert r["pressure_drop_Pa"] == pytest.approx(dp) and r["uniformity_index_out"] == pytest.approx(1)
    assert r["superficial_velocity"]["mean"] == pytest.approx(0.01) and r["flow_in_m3s"] == pytest.approx(0.01 * 1e-5)


def test_overlapping_zones_are_refused(tmp_path):
    st = _study({"darcy": 1e8})
    st["porous"].append({"name": "bed2", "region": {"box": [0, -10, -1, 50, 10, 2]}, "darcy": 1e8})
    with pytest.raises(Exception, match="overlaps"):
        prepare(st, tmp_path)
