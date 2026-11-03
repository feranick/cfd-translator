"""Cavitation (2026.11.03.1): incompressibleVoF + VoFCavitation cases, the reader's unit conversion, results, the
NPSH sweep and NPSH3. Runs without OpenFOAM (the solver's output is written by hand)."""
from __future__ import annotations

import numpy as np
import pytest

from cfd_translator import foamio
from cfd_translator.cavitation import G0, npsh_curve, results, water_vapour
from cfd_translator.format import validate_study
from cfd_translator.runner import prepare

DUCT = {"geometry": {"box": [200, 20, 20]}, "fluid": "water", "model": "kOmegaSST",
        "bcs": [{"kind": "inlet", "name": "inlet", "faces": {"nx": -1}, "flow_rate": 0.002},
                {"kind": "outlet", "name": "outlet", "faces": {"nx": 1}, "pressure": 0}],
        "mesh": {"cells": [20, 4, 4]}, "cavitation": {"npsh": 5}}


def test_water_vapour():
    w = water_vapour(20)
    assert w["p"] == pytest.approx(2339, rel=0.01) and w["rho"] == pytest.approx(0.0173, rel=0.02)
    assert water_vapour(100)["p"] == pytest.approx(101325, rel=0.01)


def test_validation():
    assert validate_study(DUCT)["ok"], validate_study(DUCT)
    e = lambda st: validate_study(st)["errors"]   # noqa: E731
    assert any("a list makes a sweep" in x for x in e(dict(DUCT, cavitation={"npsh": [5, 3]})))
    assert any("pseudo-steady" in x for x in e(dict(DUCT, solve={"transient": {"end_time": 1}})))
    two = dict(DUCT, bcs=DUCT["bcs"] + [{"kind": "inlet", "faces": {"ny": 1}, "flow_rate": 0.001}])
    assert any("one inlet" in x for x in e(two))
    assert any("other than water" in x for x in e(dict(DUCT, fluid={"rho": 800, "mu": 1e-3})))
    oil = dict(DUCT, fluid={"rho": 800, "mu": 1e-3}, cavitation={"npsh": 5, "p_vapour": 500, "vapour": {"rho": 0.01, "mu": 1e-5}})
    assert validate_study(oil)["ok"], validate_study(oil)
    assert any("model" in x for x in e(dict(DUCT, cavitation={"npsh": 5, "model": "X"})))
    assert any("not with energy" in x for x in e(dict(DUCT, energy=True)))


def test_npsh_sweep_expands():
    from cfd_translator.variants import expand
    ex, groups, errs = expand({"p": dict(DUCT, cavitation={"npsh": [8, 4, 2.5], "temperature": 25})})
    assert not errs and sorted(ex) == ["p_npsh2.5", "p_npsh4", "p_npsh8"]
    assert ex["p_npsh2.5"]["cavitation"] == {"npsh": 2.5, "temperature": 25}
    assert groups["p"]["npsh"] == [8.0, 4.0, 2.5] and groups["p"]["head_drop"] == 0.03
    _, _, errs = expand({"q": dict(DUCT, cavitation={"npsh": [4]})})
    assert "q" in errs
    for n, st in ex.items():
        assert validate_study(st)["ok"], (n, validate_study(st))


def test_vof_case_files(tmp_path):
    mesh, patches, sel, plan, w = prepare(DUCT, tmp_path)
    c = tmp_path / "case"
    vf = plan["vof"]
    pv = water_vapour(20)["p"]
    assert vf["p0_Pa"] == pytest.approx(pv + 998.2 * G0 * 5, rel=1e-6) and vf["p_start_Pa"] > vf["p0_Pa"]
    cd = (c / "system" / "controlDict").read_text()
    assert "solver          incompressibleVoF;" in cd and "deltaT          1;" in cd and "endTime         4000;" in cd
    fs = (c / "system" / "fvSchemes").read_text()
    assert "localEuler" in fs and "noInterfaceCompression" in fs and "div(rhoPhi,U)" in fs
    fv = (c / "system" / "fvSolution").read_text()
    assert '"alpha.water.*"' in fv and "p_rghFinal" in fv and "maxAlphaCo" in fv and "rDeltaTSmoothingCoeff" in fv
    m = (c / "constant" / "fvModels").read_text()
    assert "type    VoFCavitation;" in m and "model   SchnerrSauer;" in m and f"pSat    {pv:.10g};" in m
    ph = (c / "constant" / "phaseProperties").read_text()
    assert "phases          (water vapour);" in ph
    assert not (c / "constant" / "physicalProperties").exists() and (c / "constant" / "g").exists()
    assert "rho             998.2;" in (c / "constant" / "physicalProperties.water").read_text()
    prgh = (c / "0" / "p_rgh").read_text()
    assert "uniformTotalPressure" in prgh and f"(1200 {vf['p0_Pa']:.10g})" in prgh and "fixedFluxPressure" in prgh
    u = (c / "0" / "U").read_text()
    assert "flowRateOutletVelocity" in u and "(400 0.002)" in u and "pressureInletOutletVelocity" in u
    assert (c / "0" / "alpha.water").exists() and not (c / "0" / "p").exists()
    ctl = foamio.read_foam(cd.split("functions")[0] + "\n")
    assert ctl["solver"] == "incompressibleVoF"
    # the pressure monitors use the mixture's density field (p in Pa)
    assert "rho             rho;" in cd and "rhoInf" not in cd


def test_monitors_for_vof_with_a_zone():
    from cfd_translator.average import monitor_functions
    plan = {"patches": {"inlet": {"kind": "inlet"}, "outlet": {"kind": "outlet"}},
            "zones": {"imp": {"rotating_walls": ["blade"], "center_mm": [0, 0, 0], "axis": [0, 0, 1], "omega_rad_s": 300}},
            "vof": {"p_ref_Pa": 51234.5}}
    funcs, mon = monitor_functions({"cavitation": {"npsh": 5}}, plan, 998.2)
    txt = "\n".join(funcs)
    assert "pRef            51234.5;" in txt and "rho             rho;" in txt and mon["torque"]["imp"] == "cfdtM_imp"


def test_reader_and_results(tmp_path):
    from cfd_translator.case import _field, _uniform, _nonuniform
    from cfd_translator.foam import read_fields
    from cfd_translator.post import results as post_results
    mesh, patches, sel, plan, w = prepare(DUCT, tmp_path)
    c = tmp_path / "case"
    vf = plan["vof"]
    t = c / "4000"
    t.mkdir()
    nc = mesh.ncells
    C = mesh.geometry()["C"]
    alpha = np.where(np.abs(C[:, 0]) < 20, 0.2, 1.0)            # a vapour pocket in the middle
    pabs = vf["p0_Pa"] - 100.0 * (C[:, 0] + 100)                 # falls along the duct
    U = np.tile([1.0, 0, 0], (nc, 1))
    bf = {p["name"]: {"type": "zeroGradient"} for p in patches}
    for p in patches:
        f = mesh.patch_faces(p["name"])
        x = mesh.geometry()["Cf"][f][:, 0]
        bf[p["name"]] = {"type": "calculated", "value": _nonuniform(vf["p0_Pa"] - 100.0 * (x + 100))}
    (t / "p").write_text(_field("volScalarField", "p", "[1 -1 -2 0 0 0 0]", _nonuniform(pabs), bf))
    Ub = {p["name"]: {"type": "calculated", "value": _uniform([1.0, 0, 0])} for p in patches}
    (t / "U").write_text(_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", _nonuniform(U), Ub))
    phib = {}
    for p in patches:
        f = mesh.patch_faces(p["name"])
        Sf = mesh.geometry()["Sf"][f] * 1e-6
        phib[p["name"]] = {"type": "calculated", "value": _nonuniform(Sf[:, 0] * 1.0)}
    (t / "phi").write_text(_field("surfaceScalarField", "phi", "[0 3 -1 0 0 0 0]", _uniform(0), phib))
    ab = {p["name"]: {"type": "zeroGradient"} for p in patches}
    (t / "alpha.water").write_text(_field("volScalarField", "alpha.water", "[0 0 0 0 0 0 0]", _nonuniform(alpha), ab))
    fields = read_fields(c, mesh, plan)
    assert np.allclose(fields["p"], (pabs - vf["p0_Pa"]) / 998.2)          # kinematic, gauge to the target
    assert np.allclose(fields["p_abs"], pabs) and np.allclose(fields["alpha"], alpha)
    assert np.allclose(fields["scalars"]["vapour"]["internal"], 1 - alpha)
    res, _w = post_results(DUCT, mesh, plan, fields)
    cav, warns = results(DUCT, mesh, plan, fields, res)
    V = mesh.geometry()["V"]
    assert cav["vapour_volume_mm3"] == pytest.approx(float((0.8 * V[np.abs(C[:, 0]) < 20]).sum()), rel=1e-4)
    assert cav["cavitating"] and abs(cav["vapour_region"]["centroid_mm"][0]) < 1
    assert cav["min_pressure_Pa"] == pytest.approx(float(pabs.min()), rel=1e-4)
    # the inlet's total pressure (the face values at x = -100: the target) gives back the NPSH asked for
    assert cav["npsh_inlet_m"] == pytest.approx(5 + 0.5 * 998.2 * 1.0 / (998.2 * G0), rel=0.02)
    assert res["checks"]["alpha_bounds"]["ok"] and "note" in res["checks"]["momentum_balance"]


def test_npsh_curve_and_npsh3():
    pts = [{"name": f"n{v}", "npsh": v, "head": h, "reliable": True, "vapour_mm3": 0.0 if v > 5 else 10.0}
           for v, h in ((10, 20.0), (6, 20.0), (4, 19.9), (3, 19.2), (2, 15.0))]
    out = npsh_curve(pts)
    assert out["npsh3_m"] == pytest.approx(4 - (19.9 - 19.4) / (19.9 - 19.2), rel=1e-3)
    assert out["between"] == ["n4", "n3"] and out["reference"] == {"npsh": 10, "head": 20.0}
    out = npsh_curve(pts[:3])
    assert out["npsh3_m"] is None and "add lower NPSH" in out["note"]
    out = npsh_curve([dict(p, vapour_mm3=5.0) for p in pts])
    assert any("higher NPSH" in w for w in out["warnings"])


def test_compare_gives_the_npsh_curve():
    from cfd_translator.variants import compare
    group = {"names": ["s_npsh8", "s_npsh4", "s_npsh2"], "scales": None, "npsh": [8.0, 4.0, 2.0], "head_drop": 0.03}
    res = {n: {"mesh": {"cells": 100}, "reliable": True,
               "results": {"pump": {"head_m": h}, "cavitation": {"vapour_volume_mm3": v, "npsh_inlet_m": s}}}
           for n, h, v, s in (("s_npsh8", 30.0, 0.0, 8.0), ("s_npsh4", 29.8, 3.0, 4.0), ("s_npsh2", 25.0, 900.0, 2.0))}
    out = compare(res, group)
    cur = out["npsh_curve"]
    assert cur["npsh3_m"] == pytest.approx(4 - (29.8 - 29.1) / (29.8 - 25.0) * 2, rel=1e-3)
    assert out["studies"]["s_npsh2"]["cavitation.vapour_volume_mm3"] == 900.0


def test_estimate_counts_the_vof_work():
    from cfd_translator.executor import estimate_minutes
    a = estimate_minutes(DUCT, "foam", cells=1e6, procs=10)
    b = estimate_minutes({k: v for k, v in DUCT.items() if k != "cavitation"}, "foam", cells=1e6, procs=10)
    assert a > b
