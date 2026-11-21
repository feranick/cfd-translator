"""swirl.py: the regions' mean tangential velocity and angular momentum about a rotating zone's axis (the S50 zero-air
gate against single phase, pump chat 20 Nov 2026), and the reference comparison that judges it."""
import json

import numpy as np
import pytest

from cfd_translator import swirl, twophase
from cfd_translator.format import validate_study

from test_diverge import box_mesh

OM = 10.0
FR = {"axis": np.array([0.0, 0.0, 1.0]), "center_mm": np.zeros(3), "omega": OM, "zone": "rot"}
REGS = [{"name": "core", "cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 3.5}},
        {"name": "ring", "cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 6.5, "inner_radius": 3.5}},
        {"name": "corner", "box": [3, 3, -6, 6, 6, 6]}]


def solid(C_mm, k=1.0):
    C = np.asarray(C_mm) * 1e-3
    return k * np.cross([0, 0, OM], C)


def test_solid_body_rotation_is_one_and_scales():
    m = box_mesh()
    g = m.geometry()
    s = swirl.from_fields(m, solid(g["C"]), REGS, FR, 1000.0, "U")
    for name in ("core", "ring", "corner"):
        r = s["regions"][name]
        assert r["swirl_ratio"] == pytest.approx(1.0, abs=1e-9) and r["L_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert r["u_theta_m_s"] > 0 and r["angular_momentum_kg_m2_s"] > 0
    s2 = swirl.from_fields(m, solid(g["C"], 1.07), REGS, FR, 1000.0, "U")
    assert s2["regions"]["ring"]["swirl_ratio"] == pytest.approx(1.07, abs=1e-6)
    # the zone turning the other way: the same flow counts against it
    s3 = swirl.from_fields(m, solid(g["C"]), REGS, dict(FR, omega=-OM), 1000.0, "U")
    assert s3["regions"]["ring"]["swirl_ratio"] == pytest.approx(-1.0, abs=1e-9)
    # a uniform translation has no mean swirl in an axisymmetric region
    s4 = swirl.from_fields(m, np.tile([0.3, 0.0, 0.0], (m.ncells, 1)), REGS[:2], FR, 1000.0, "U")
    assert abs(s4["regions"]["ring"]["u_theta_m_s"]) < 1e-12
    assert s["about"]["zone"] == "rot" and s["about"]["omega_rad_s"] == OM


def test_liquid_fraction_weights_the_water():
    m = box_mesh()
    g = m.geometry()
    U = solid(g["C"])
    w = np.where(np.linalg.norm(g["C"][:, :2], axis=1) < 3, 0.0, 1.0)      # the core full of air
    U[w == 0] *= 5.0                                                       # whatever the air cells hold
    s = swirl.from_fields(m, U, REGS, FR, 1000.0, "U.water", w=w)
    assert s["regions"]["ring"]["swirl_ratio"] == pytest.approx(1.0, abs=1e-9)


def write_field(path, cls, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    if cls == "vector":
        body = "\n".join(f"({a:.10g} {b:.10g} {c:.10g})" for a, b, c in values)
        lst = f"nonuniform List<vector> {len(values)}\n(\n{body}\n)"
        klass = "volVectorField"
    else:
        body = "\n".join(f"{v:.10g}" for v in values)
        lst = f"nonuniform List<scalar> {len(values)}\n(\n{body}\n)"
        klass = "volScalarField"
    path.write_text(f"FoamFile\n{{\n    format ascii;\n    class {klass};\n    object {path.name};\n}}\n"
                    f"dimensions [0 1 -1 0 0 0 0];\ninternalField {lst};\nboundaryField\n{{\n}}\n")


def test_from_case_steady_latest_and_transient_mean(tmp_path):
    case = tmp_path / "case"
    m = box_mesh()
    m.write(case)
    g = m.geometry()
    for t, k in (("1", 1.0), ("2", 1.1), ("3", 1.2)):
        write_field(case / t / "U.water", "vector", solid(g["C"], k))
        write_field(case / t / "alpha.water", "scalar", np.ones(m.ncells))
    s = swirl.from_case(case, REGS, FR, 1000.0, latest_only=True)
    assert s["regions"]["ring"]["swirl_ratio"] == pytest.approx(1.2, abs=1e-6) and "time 3" in s["basis"]
    s = swirl.from_case(case, REGS, FR, 1000.0, t_from=2.0)
    assert s["regions"]["ring"]["swirl_ratio"] == pytest.approx(1.15, abs=1e-6) and "mean of 2" in s["basis"]
    assert s["regions"]["ring"]["cells"] == int((np.linalg.norm(g["C"][:, :2], axis=1) >= 3.5).sum() -
                                                (np.linalg.norm(g["C"][:, :2], axis=1) > 6.5).sum())


def test_from_case_decomposed(tmp_path):
    """Two processor directories: the sums add up to the whole."""
    case = tmp_path / "case"
    m = box_mesh()
    g = m.geometry()
    whole = swirl.from_fields(m, solid(g["C"], 1.05), REGS, FR, 1000.0, "U")
    for k in (0, 1):
        d = case / f"processor{k}"
        m.write(d)
        write_field(d / "5" / "U.water", "vector", solid(g["C"], 1.05))
        write_field(d / "5" / "alpha.water", "scalar", np.ones(m.ncells))
    s = swirl.from_case(case, REGS, FR, 1000.0, latest_only=True)
    assert s["regions"]["ring"]["swirl_ratio"] == pytest.approx(1.05, abs=1e-6)
    assert s["regions"]["ring"]["cells"] == 2 * whole["regions"]["ring"]["cells"]
    assert s["regions"]["ring"]["angular_momentum_kg_m2_s"] == pytest.approx(
        2 * whole["regions"]["ring"]["angular_momentum_kg_m2_s"], rel=1e-4)


def test_reference_comparison_judges_swirl(tmp_path, monkeypatch):
    ref = tmp_path / "req" / "single"
    ref.mkdir(parents=True)
    (ref / "single.json").write_text(json.dumps({"results": {
        "pump": {"head_m": 10.0}, "rotating": {"rot": {"torque_Nm": 2.0}},
        "swirl": {"regions": {"eye": {"u_theta_m_s": 1.0}, "pocket": {"u_theta_m_s": 2.0}}}}}))
    sw = {"regions": {"eye": {"u_theta_m_s": 1.02}, "pocket": {"u_theta_m_s": 2.1}, "reservoir": {"u_theta_m_s": 0.1}}}
    c = twophase.reference_comparison(str(ref), {"head_m": 10.1, "torque_Nm": 2.02}, sw)
    assert c["head_ratio"] == 1.01 and c["swirl_ratio_by_region"] == {"eye": 1.02, "pocket": 1.05}
    assert c["within_tolerance"] is False and c["swirl_missing_in_reference"] == ["reservoir"]
    assert c["swirl_gate"]["pocket"]["ok"] is False and c["swirl_gate"]["eye"]["gate"] == "relative 0.03"
    sw["regions"]["pocket"]["u_theta_m_s"] = 2.05
    # a region the single-phase run lacks cannot be judged: the gate fails until both runs have it
    assert twophase.reference_comparison(str(ref), {"head_m": 10.1, "torque_Nm": 2.02}, sw)["within_tolerance"] is False
    sw["regions"].pop("reservoir")
    assert twophase.reference_comparison(str(ref), {"head_m": 10.1, "torque_Nm": 2.02}, sw)["within_tolerance"] is True
    # head and torque alone, as before
    c = twophase.reference_comparison(str(ref), {"head_m": 10.5, "torque_Nm": 2.0})
    assert c["within_tolerance"] is False and "swirl_ratio_by_region" not in c


def _study(**kw):
    st = {"geometry": {"cylinder": [50, 10]}, "fluid": "water", "model": "laminar",
          "bcs": [{"kind": "wall", "name": "container", "faces": {"all": True}, "rotation": "rot"}],
          "rotating": [{"name": "rot", "axis": [0, 0, 1], "center": [0, 0, -1], "rpm": 95.5, "radius": 60,
                        "span": [0, 12]}],
          "mesh": {"size": 2, "extrude": 4}}
    st.update(kw)
    return st


def test_regions_validate():
    assert validate_study(_study(regions=REGS))["ok"]
    r = validate_study(_study(regions=[{"name": "a", "box": [1, 0, 0, 0, 1, 1]}]))
    assert not r["ok"] and any(e.startswith("regions[0]") for e in r["errors"])
    r = validate_study(_study(regions=[dict(REGS[0], zone="nope")]))
    assert any("regions[0].zone" in e for e in r["errors"])
    st = _study(regions=REGS)
    st.pop("rotating")
    st["bcs"][0].pop("rotation")
    r = validate_study(st)
    assert any("rotating zone's axis" in e for e in r["errors"])
    r = validate_study(_study(regions=[REGS[0], REGS[0]]))
    assert any("unique" in e for e in r["errors"])


def test_two_phase_takes_the_studys_regions():
    st = _study(regions=REGS[:2], two_phase={"bubble_size": {"fixed": 1}, "gravity": [0, 0, 0],
                                             "regions": [{"name": "ring", "box": [0, 0, 0, 1, 1, 1]}]})
    regs = twophase.settings(st)["regions"]
    assert [r["name"] for r in regs] == ["core", "ring"] and "box" in regs[1]


def test_absolute_gate_for_stationary_pockets(tmp_path):
    """Pump chat, 20 Nov: the pocket and the reservoir are stationary with little swirl; a ratio of tangential
    velocities is ill-conditioned there, so they are judged by the swirl ratios' difference (0.03 of the solid body)."""
    ref = tmp_path / "req" / "single"
    ref.mkdir(parents=True)
    (ref / "single.json").write_text(json.dumps({"results": {"swirl": {"regions": {
        "eye": {"u_theta_m_s": 5.0, "swirl_ratio": 0.6},
        "pocket": {"u_theta_m_s": 0.10, "swirl_ratio": 0.020},
        "reservoir": {"u_theta_m_s": 0.05, "swirl_ratio": 0.010}}}}}))
    regs = [{"name": "eye", "cylinder": {"radius": 22}, "gate": "relative"},
            {"name": "pocket", "box": {"min": [0, 0, 0], "max": [1, 1, 1]}, "gate": "absolute"},
            {"name": "reservoir", "box": [0, 0, 0, 1, 1, 1]}]                      # gate chosen: |0.01| < 0.25
    sw = {"regions": {"eye": {"u_theta_m_s": 5.1, "swirl_ratio": 0.612},
                      "pocket": {"u_theta_m_s": 0.20, "swirl_ratio": 0.040},      # twice the swirl, 0.02 apart
                      "reservoir": {"u_theta_m_s": 0.02, "swirl_ratio": 0.004}}}
    c = twophase.reference_comparison(str(ref), {}, sw, regs)
    g = c["swirl_gate"]
    assert g["eye"]["ok"] and g["pocket"]["ok"] and g["reservoir"]["ok"] and c["within_tolerance"] is True
    assert g["pocket"]["gate"] == "absolute 0.03" and g["reservoir"]["gate"] == "absolute 0.03"
    sw["regions"]["pocket"]["swirl_ratio"] = 0.06
    assert twophase.reference_comparison(str(ref), {}, sw, regs)["within_tolerance"] is False
    regs[1]["gate"] = {"absolute": 0.05}
    assert twophase.reference_comparison(str(ref), {}, sw, regs)["within_tolerance"] is True


def test_box_as_min_max():
    m = box_mesh()
    g = m.geometry()
    a = twophase.region_cells({"box": [0, 0, -6, 6, 6, 6]}, g["C"])
    b = twophase.region_cells({"box": {"min": [0, 0, -6], "max": [6, 6, 6]}}, g["C"])
    assert len(a) and list(a) == list(b)
    assert validate_study(_study(regions=[{"name": "p", "box": {"min": [-35, -66, -33], "max": [35, -38, -20]},
                                           "gate": "absolute"}]))["ok"]
    r = validate_study(_study(regions=[{"name": "p", "box": {"min": [1, 0, 0], "max": [0, 1, 1]}}]))
    assert not r["ok"]
    r = validate_study(_study(regions=[{"name": "p", "box": [0, 0, 0, 1, 1, 1], "gate": "loose"}]))
    assert not r["ok"]
