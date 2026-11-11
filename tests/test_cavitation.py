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
    # the liquid flow held as a mass flow (S50 at NPSH 3 m: a held volume flow let the inlet's fall 12 %)
    assert "flowRateOutletVelocity" in u and "massFlowRate" in u and "(400 1.9964)" in u and "rhoOutlet       998.2;" in u
    assert "pressureInletOutletVelocity" in u and "volumetricFlowRate" not in u
    assert "table ((0 0.09982)" in u            # never 0: flowRateOutletVelocity divides by it (SIGFPE, carbonio)
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


def test_pseudo_time_step_is_limited_by_the_rotation(tmp_path):
    """S50 at NPSH 15 m (spark2, 2026.11.03): with maxDeltaT 1 s the local time step in slow corners of the MRF zone
    let the frame's rotation turn the flow by ~300 rad per step: torque -47..+2 N m around -7.6, never settled."""
    st = dict(DUCT, rotating=[{"name": "rot", "axis": [1, 0, 0], "center": [0, 0, 10], "rpm": 2900, "radius": 5,
                               "span": [-20, 20]}])
    assert validate_study(st)["ok"], validate_study(st)
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    fv = (tmp_path / "case" / "system" / "fvSolution").read_text()
    w_ = 2900 * 2 * np.pi / 60
    assert f"maxDeltaT       {0.2 / w_:.6g};" in fv and "maxCo           2;" in fv
    assert plan["vof"]["max_delta_t"] == pytest.approx(0.2 / w_)
    st2 = dict(DUCT, cavitation={"npsh": 5, "max_co": 0.8})
    mesh, patches, sel, plan, w = prepare(st2, tmp_path / "b")
    fv = (tmp_path / "b" / "case" / "system" / "fvSolution").read_text()
    assert "maxCo           0.8;" in fv and "maxDeltaT       1;" in fv
    assert any("max_co" in e for e in validate_study(dict(DUCT, cavitation={"npsh": 5, "max_co": 50}))["errors"])


def test_mass_balance_and_vapour_formed(tmp_path):
    """S50 at NPSH 3 m (2026.11.05): outlet 5.5556e-3 m3/s, inlet 4.885e-3 (-12 %); counted as volume at the
    liquid's density this looked like a 12 % mass error. It is vapour forming: ~11 mg/s makes 0.67 L/s."""
    from cfd_translator.cavitation import npsh_curve
    mesh, patches, sel, plan, w = prepare(DUCT, tmp_path)
    vf = plan["vof"]
    n_in, n_out = vf["inlet"], vf["outlet"]
    nf = {P["name"]: P["n"] for P in mesh.patches}
    q_in, q_out = 4.885e-3, 5.5556e-3
    fields = {"patches": {n_in: {"phi": np.full(nf[n_in], -q_in / nf[n_in]), "alpha": np.ones(nf[n_in])},
                          n_out: {"phi": np.full(nf[n_out], q_out / nf[n_out]), "alpha": np.ones(nf[n_out])}},
              "alpha": np.ones(mesh.ncells), "p_abs": np.full(mesh.ncells, 1e5)}
    res = {"pressure": {}, "checks": {}}
    cav, warns = results(DUCT, mesh, plan, fields, res)
    fl = cav["flows"]
    assert fl["liquid_flow_error"] == pytest.approx(q_in / 0.002 - 1, rel=1e-3)
    evap = (q_out - q_in) / (1 / vf["rho_v"] - 1 / 998.2)
    assert fl["net_evaporation_kg_s"] == pytest.approx(evap, rel=1e-3) and 5e-6 < evap < 2e-5
    assert res["checks"]["mass_balance"]["error"] == pytest.approx(abs(q_out - q_in) / q_in, rel=1e-3)
    assert any("vapour volume is still changing" in x for x in warns)
    assert any("at another flow" in x for x in warns)
    # the NPSH curve leaves out points off the flow
    pts = [{"name": "a", "npsh": 15, "head": 16.73, "reliable": True, "flow_error": -0.016},
           {"name": "b", "npsh": 10, "head": 16.75, "reliable": True, "flow_error": 0.002},
           {"name": "c", "npsh": 3, "head": 17.42, "reliable": True, "flow_error": -0.121}]
    out = npsh_curve(pts)
    assert [p["name"] for p in out["left_out"]] == ["a", "c"] and any("left out of NPSH3" in x for x in out["warnings"])


GAS = dict(DUCT, cavitation={"npsh": 7, "inlet_gas_fraction": 0.02})


def test_gas_validation():
    assert validate_study(GAS)["ok"], validate_study(GAS)
    for bad in (-0.01, 0.5, "x"):                       # 0 is allowed (a point where no air comes out)
        e = validate_study(dict(GAS, cavitation={"npsh": 7, "inlet_gas_fraction": bad}))["errors"]
        assert any("inlet_gas_fraction" in x for x in e), bad
    e = validate_study(dict(GAS, cavitation={"npsh": 7, "inlet_gas_fraction": 0.02, "model": "Kunz"}))["errors"]
    assert any("liquid + air only" in x for x in e)


def test_gas_case_files(tmp_path):
    """inlet_gas_fraction: OpenFOAM 14 compressibleVoF, water (rhoConst) + perfect-gas air, no phase change; the
    energy equation off (isothermal), p bounded at the vapour pressure, LTS and the mass-flow outlet as before."""
    mesh, patches, sel, plan, w = prepare(GAS, tmp_path)
    c = tmp_path / "case"
    vf = plan["vof"]
    pv = water_vapour(20)["p"]
    gas = vf["gas"]
    assert vf["solver"] == "compressibleVoF" and vf["phase2"] == "air"
    assert gas["rho_inlet"] == pytest.approx(vf["p0_Pa"] / (8.314462618 / 0.0289 * 293.15), rel=1e-6)
    cd = (c / "system" / "controlDict").read_text()
    assert "solver          compressibleVoF;" in cd and "maxAlphaCo      1;" in cd and "maxDeltaT" in cd
    ph = (c / "constant" / "phaseProperties").read_text()
    assert "phases          (water air);" in ph and "pSat" not in ph
    air = (c / "constant" / "physicalProperties.air").read_text()
    assert "equationOfState perfectGas;" in air and "molWeight   28.9;" in air and "heRhoThermo" in air
    wat = (c / "constant" / "physicalProperties.water").read_text()
    assert "equationOfState rhoConst;" in wat and "rho         998.2;" in wat and "thermo          hConst;" in wat
    assert not (c / "constant" / "fvModels").exists() and not (c / "constant" / "physicalProperties.vapour").exists()
    fc = (c / "system" / "fvConstraints").read_text()
    assert "type            bound;" in fc and "field           p;" in fc and f"min             {pv:.10g};" in fc
    fv = (c / "system" / "fvSolution").read_text()
    assert "thermophysics   no;" in fv and '".*(rho|rhoFinal)"' in fv and "|T).*" in fv and "localEuler" not in fv
    fs = (c / "system" / "fvSchemes").read_text()
    assert "localEuler" in fs and "div(rhoPhi,k)" in fs and "div(alphaRhoPhi,e)" in fs and "div(phi,p)" in fs
    a = foamio.read_foam((c / "0" / "alpha.water").read_text())["boundaryField"]
    assert a["inlet"]["type"] == "fixedValue" and "0.98" in str(a["inlet"]["value"])
    assert (c / "0" / "p").exists() and (c / "0" / "T").exists()
    T = foamio.read_foam((c / "0" / "T").read_text())["boundaryField"]
    assert T["inlet"]["type"] == "fixedValue" and T["outlet"]["type"] == "inletOutlet"
    # the outlet draws the liquid's mass flow plus the air's entering with it
    m = 0.002 * 998.2 + 0.002 * 0.02 / 0.98 * gas["rho_inlet"]
    assert gas["mass_flow_kg_s"] == pytest.approx(m, rel=1e-9) and f"(400 {m:.10g})" in (c / "0" / "U").read_text()


def test_gas_results(tmp_path):
    mesh, patches, sel, plan, w = prepare(GAS, tmp_path)
    vf = plan["vof"]
    gas = vf["gas"]
    n_in, n_out = vf["inlet"], vf["outlet"]
    nf = {P["name"]: P["n"] for P in mesh.patches}
    q = 0.002 / 0.98                                         # liquid 0.002 + 2 % air at the inlet
    p_in, p_out = vf["p0_Pa"], vf["p0_Pa"] + 5e4              # the air is compressed through the pump
    kin = lambda pa: (pa - vf["p_ref_Pa"]) / 998.2            # noqa: E731 (post's kinematic gauge layout)
    q_air_out = q * 0.02 * p_in / p_out
    a_out = 1 - q_air_out / (0.002 + q_air_out)
    fields = {"patches": {n_in: {"phi": np.full(nf[n_in], -q / nf[n_in]), "alpha": np.full(nf[n_in], 0.98),
                                 "p": np.full(nf[n_in], kin(p_in))},
                          n_out: {"phi": np.full(nf[n_out], (0.002 + q_air_out) / nf[n_out]),
                                  "alpha": np.full(nf[n_out], a_out), "p": np.full(nf[n_out], kin(p_out))}},
              "alpha": np.full(mesh.ncells, 0.97), "p_abs": np.full(mesh.ncells, p_in)}
    fields["p_abs"][0] = vf["p_min_Pa"]                      # one cell at the bound: vapour would form there
    res = {"pressure": {}, "checks": {"momentum_balance": {"error": 0.0}}}
    cav, warns = results(GAS, mesh, plan, fields, res)
    fl = cav["flows"]
    assert cav["phases"] == ["water", "air"] and "vapour_volume_mm3" not in cav and cav["gas_volume_mm3"] > 0
    assert cav["inlet_gas_fraction"] == 0.02 and cav["gas_holdup"] == pytest.approx(0.03 / 0.02, rel=1e-3)
    assert fl["gas_in_kg_s"] == pytest.approx(fl["gas_out_kg_s"], rel=1e-6)      # air neither forms nor condenses
    assert abs(fl["liquid_flow_error"]) < 1e-6 and res["checks"]["mass_balance"]["error"] < 1e-6
    assert fl["volume_source_m3s"] < 0                                           # compressed: less volume leaves
    assert cav["at_vapour_pressure"]["cells"] == 1 and any("vapour would form" in x for x in warns)
    assert "not judged" in res["checks"]["momentum_balance"]["note"]


def test_gas_fraction_from_saturation():
    """{saturated_at}: the air a liquid saturated at 1 atm cannot hold at the inlet (Henry's law), about
    0.02 (p_s / p_inlet - 1) at 20 C: none at 1 atm, about 1 % at NPSH 7 m, 3 % at 4 m."""
    from cfd_translator.cavitation import gas_fraction, henry_air
    pv = water_vapour(20)["p"]
    f0, info = gas_fraction({"saturated_at": 101325}, 101325, pv, 20)
    assert f0 == 0 and 22 < info["dissolved_at_saturation_mg_per_L"] < 26            # ~24 mg/L of air at 20 C
    assert henry_air(20) > henry_air(40)                                               # less soluble when warmer
    for npsh in (7, 4, 2):
        p_in = pv + 998.2 * G0 * npsh
        f, info = gas_fraction({"saturated_at": 101325}, p_in, pv, 20)
        rough = 0.02 * (101325 / p_in - 1)
        assert f == pytest.approx(rough / (1 + rough), rel=0.15), npsh
    f7 = gas_fraction({"saturated_at": 101325}, pv + 998.2 * G0 * 7, pv, 20)[0]
    assert 0.007 < f7 < 0.011
    half = gas_fraction({"saturated_at": 101325, "release": 0.5}, pv + 998.2 * G0 * 7, pv, 20)[0]
    assert half == pytest.approx(f7 / 2, rel=0.02)
    assert gas_fraction(0.03, 5e4, pv, 20) == (0.03, {"from": "given"})


def test_gas_fraction_per_point_and_validation(tmp_path):
    from cfd_translator.variants import expand
    st = dict(DUCT, cavitation={"npsh": [9, 7, 5], "inlet_gas_fraction": [0.005, 0.01, 0.02]})
    out, groups, errs = expand({"s": st})
    assert not errs and [out[f"s_npsh{n}"]["cavitation"]["inlet_gas_fraction"] for n in (9, 7, 5)] == [0.005, 0.01, 0.02]
    assert all(validate_study(v)["ok"] for v in out.values())
    _, _, errs = expand({"s": dict(DUCT, cavitation={"npsh": [9, 7], "inlet_gas_fraction": [0.01]})})
    assert "1 values for 2 NPSH points" in errs["s"][0]
    assert any("goes with an npsh list" in e for e in validate_study(
        dict(DUCT, cavitation={"npsh": 7, "inlet_gas_fraction": [0.01, 0.02]}))["errors"])
    sat = dict(DUCT, cavitation={"npsh": 7, "inlet_gas_fraction": {"saturated_at": 101325}})
    assert validate_study(sat)["ok"], validate_study(sat)
    deep = dict(DUCT, cavitation={"npsh": 0.5, "inlet_gas_fraction": {"saturated_at": 101325}})
    assert any("more than 0.2" in e for e in validate_study(deep)["errors"])
    assert not validate_study(dict(DUCT, cavitation={"npsh": 7, "inlet_gas_fraction": {"saturated": 1}}))["ok"]
    # each point's own fraction in its case; none comes out above saturation, and the run is still liquid + air
    mesh, patches, sel, plan, w = prepare(sat, tmp_path / "a")
    g = plan["vof"]["gas"]
    assert 0.007 < g["inlet_fraction"] < 0.011 and g["from"].startswith("saturation at 101325 Pa")
    high = dict(DUCT, cavitation={"npsh": 12, "inlet_gas_fraction": {"saturated_at": 101325}})
    mesh, patches, sel, plan, w = prepare(high, tmp_path / "b")
    assert plan["vof"]["solver"] == "compressibleVoF" and plan["vof"]["gas"]["inlet_fraction"] == 0
    a = foamio.read_foam((tmp_path / "b" / "case" / "0" / "alpha.water").read_text())["boundaryField"]
    assert "1" in str(a["inlet"]["value"])
