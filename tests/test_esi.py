"""The ESI OpenFOAM (v2506) backend for sliding meshes (esi.py): validation, settings, the dialect's dictionaries,
the function objects, the zone headers, the AMI log, and the run sequence with OpenFOAM stand-ins (no OpenFOAM here:
what the dictionaries do in v2506 is checked against the hand-built S50 v2506 case, not run)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfd_translator import esi, foamio
from cfd_translator.case import write_case
from cfd_translator.format import validate_study

from test_sliding import _study, patches_for, pump_mesh, zone_of

LOG = Path(__file__).parent / "data" / "esi_v2506_pimpleFoam_s50.log"


def _esi_study(**tr):
    st = _study(**tr)
    st["solve"]["backend"] = "esi"
    return st


def test_validation_and_refusals():
    st = _esi_study()
    assert validate_study(st)["ok"], validate_study(st)
    bad = dict(st, solve=dict(st["solve"], backend="foundation"))
    assert any("solve.backend: one of" in e for e in validate_study(bad)["errors"])
    mrf = dict(st, rotating=[{k: v for k, v in st["rotating"][0].items() if k != "motion"}])
    assert any("only with a sliding zone" in e for e in validate_study(mrf)["errors"])
    for key, val in (("energy", True), ("scalars", [{"name": "c", "inlets": {}}])):
        errs = esi.errors(dict(st, **{key: val}))
        assert errs and "OpenFOAM 14" in errs[0]
    for k, v in (("locator", True), ("cells_per_step", 3), ("p_below", [90000]), ("correct_mesh_phi", False)):
        s2 = _esi_study(**{k: v})
        assert any(k in e for e in esi.errors(s2)), k
    steady = dict(st, solve={"backend": "esi"})
    assert any("only for transient runs" in e for e in esi.errors(steady))
    assert esi.errors(_study()) == []                                     # no backend: OpenFOAM 14, nothing to say


def test_settings_fixed_step_without_the_ncc_cap():
    from cfd_translator.executor import sliding_degrees
    from cfd_translator.transient import settings
    s = settings(_esi_study(degrees_per_step=1))
    assert s["fixed_step"] and s["max_co"] is None and s["cells_per_step"] is None and s["backend"] == "esi"
    assert s["dt"] == pytest.approx(0.04 / 360, rel=1e-4)                # 1500 rpm, 1 degree
    s2 = settings(_esi_study(max_co=5))                                   # asked: adjustable
    assert not s2["fixed_step"] and s2["max_co"] == 5
    st = _esi_study(degrees_per_step=1)
    assert sliding_degrees(st, st["rotating"][0], st["solve"]["transient"]) == 1.0
    v14 = settings(_study())
    assert v14["cells_per_step"] == 2.0 and "backend" not in v14


def test_dictionaries_follow_the_hand_built_case():
    sl = {"zone": "rotor", "group": "rotor_slide", "face_zone": "rotor_interface", "rotor": "rotor_rotor",
          "stator": "rotor_stator", "center_mm": [0, 0, 0], "axis": [0, 0, 1], "omega_rad_s": 303.6873}
    d = foamio.read_foam(esi.dynamic_mesh_dict(sl))
    assert d["dynamicFvMesh"] == "dynamicMotionSolverFvMesh" and d["motionSolver"] == "solidBody"
    assert d["solidBodyMotionFunction"] == "rotatingMotion" and float(d["omega"]) == pytest.approx(303.6873)
    assert d["cellZone"] == "rotor"
    cb = foamio.read_foam(esi.create_baffles_dict(sl))
    p = cb["baffles"]["rotor_slide"]["patches"]
    assert p["master"]["name"] == "rotor_rotor" and p["master"]["neighbourPatch"] == "rotor_stator"
    assert p["slave"]["name"] == "rotor_stator" and p["slave"]["neighbourPatch"] == "rotor_rotor"
    assert p["master"]["type"] == p["slave"]["type"] == "cyclicAMI" and p["master"]["transform"] == "noOrdering"
    assert float(p["master"]["lowWeightCorrection"]) == 0.2 and cb["baffles"]["rotor_slide"]["zoneName"] == "rotor_interface"
    tp = foamio.read_foam(esi.transport_properties(1.003e-6))
    assert tp["transportModel"] == "Newtonian" and float(tp["nu"]) == pytest.approx(1.003e-6)
    tu = foamio.read_foam(esi.turbulence_properties("kOmegaSST"))
    assert tu["simulationType"] == "RAS" and tu["RAS"]["RASModel"] == "kOmegaSST"
    des = foamio.read_foam(esi.turbulence_properties("kOmegaSSTDES"))
    assert des["simulationType"] == "LES" and des["LES"]["delta"] == "cubeRootVol"
    assert foamio.read_foam(esi.turbulence_properties("laminar"))["simulationType"] == "laminar"
    fs = esi.fv_schemes("kOmegaSST")
    assert "default         Euler;" in fs and "correctWalls    false;" in fs and "cellLimited Gauss linear 1" in fs
    assert "default         backward;" in esi.fv_schemes("kOmegaSSTDES") and "LUST" in esi.fv_schemes("kOmegaSSTDES")
    fv = foamio.read_foam(esi.fv_solution("kOmegaSST", 1, fixed_pressure=True))
    pim = fv["PIMPLE"]
    assert int(pim["nOuterCorrectors"]) == 3 and int(pim["nCorrectors"]) == 2 and pim["correctPhi"] == "yes"
    assert pim["checkMeshCourantNo"] == "yes" and "outerCorrectorResidualControl" in pim and "pRefCell" not in pim
    assert 'pcorr.*' in fv["solvers"] and '(U|k|omega)Final' in fv["solvers"]
    assert "pRefCell" in foamio.read_foam(esi.fv_solution("laminar", 2, fixed_pressure=False))["PIMPLE"]


def test_function_objects_in_esi_form():
    sfv = ("    cfdtP_inlet\n    {\n        type            surfaceFieldValue;\n"
           "        libs            (\"libfieldFunctionObjects.so\");\n        patch           inlet;\n"
           "        operation       average;\n        weightField     phi;\n        fields          (p total(p));\n    }")
    out, why = esi.function(sfv)
    assert why is None and "regionType      patch;" in out and "name           inlet;" in out.replace("name ", "name ")
    assert "operation       weightedAverage;" in out and "(fieldFunctionObjects)" in out and "libfield" not in out
    q = sfv.replace("operation       average;\n        weightField     phi;\n", "operation       sum;\n")
    assert "operation       sum;" in esi.function(q)[0]
    pr = ("    cfdtTotalP\n    {\n        type            pressure;\n        libs            (\"libfieldFunctionObjects.so\");\n"
          "        calcTotal       yes;\n        calcCoeff       no;\n        rho             rhoInf;\n        result          total(p);\n    }")
    out, _ = esi.function(pr)
    assert "mode            total;" in out and "calcTotal" not in out and "calcCoeff" not in out and "result          total(p);" in out
    fa = "    cfdtAverage\n    {\n        type            fieldAverage;\n        startTime       0.12;\n    }"
    assert "timeStart       0.12;" in esi.function(fa)[0]
    out, why = esi.function("    cfdtPBelow\n    {\n        type            coded;\n    }")
    assert out is None and "coded" in why
    fo = "    cfdtM_rotor\n    {\n        type            forces;\n        libs            (\"libforces.so\");\n    }"
    assert "(forces)" in esi.function(fo)[0]
    blocks, notes = esi.functions([sfv, "    x\n    {\n        type            coded;\n    }"])
    assert len(blocks) == 1 and len(notes) == 1


def test_write_transient_from_a_sliding_plan(tmp_path):
    m = pump_mesh()
    st = _esi_study(degrees_per_step=1)
    st["model"] = "kOmegaSST"
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    case = tmp_path / "case"
    plan = write_case(st, m, patches_for(m), case)
    assert plan["backend"] == "esi" and plan["transient"]["cells_per_step"] is None
    plan["sliding"] = dict(plan["sliding"])
    notes = esi.write_transient(case, st, plan)
    cd = foamio.read_foam((case / "system" / "controlDict").read_text())
    assert cd["application"] == "pimpleFoam" and cd["adjustTimeStep"] == "no" and "solver" not in cd
    assert float(cd["deltaT"]) == pytest.approx(plan["transient"]["dt"])
    fns = cd["functions"]
    assert fns["cfdtAverage"]["timeStart"] and "startTime" not in fns["cfdtAverage"]
    for n, f in fns.items():
        assert "libs" not in f or ".so" not in str(f["libs"]), n
        if f.get("type") == "surfaceFieldValue":
            assert f["regionType"] == "patch" and "patch" not in f
    assert (case / "constant" / "turbulenceProperties").is_file() and (case / "constant" / "transportProperties").is_file()
    assert "dynamicMotionSolverFvMesh" in (case / "constant" / "dynamicMeshDict").read_text()
    assert isinstance(notes, list)


def test_zone_headers_and_ami_fields(tmp_path):
    poly = tmp_path / "polyMesh"
    poly.mkdir()
    (poly / "cellZones").write_text('FoamFile\n{\n    format      ascii;\n    class       cellZoneList;\n'
                                    '    object      cellZones;\n}\n1\n(\nrotor\n{\n    type cellZone;\n'
                                    '    cellLabels List<label> 2(0 1);\n}\n)\n')
    raw = (b'FoamFile\n{\n    format      binary;\n    class       faceZoneList;\n    object      faceZones;\n}\n'
           b'1\n(\nz\n{\n    type faceZone;\n    faceLabels List<label> 2(\x00\x01\x02\x03\x04\x05\x06\x07);\n}\n)\n')
    (poly / "faceZones").write_bytes(raw)
    assert sorted(esi.zone_headers(poly)) == ["cellZones", "faceZones"]
    assert "class       regIOobject;" in (poly / "cellZones").read_text()
    fz = (poly / "faceZones").read_bytes()
    assert b"class       regIOobject;" in fz and fz.endswith(raw[raw.index(b"}\n") + 2:])     # body untouched
    assert esi.zone_headers(poly) == []                                                      # once
    zero = tmp_path / "0"
    zero.mkdir()
    (zero / "U").write_text("FoamFile\n{\n    class volVectorField;\n}\ninternalField uniform (0 0 0);\n"
                            "boundaryField\n{\n    inlet\n    {\n        type fixedValue;\n        value uniform (1 0 0);\n    }\n}\n")
    assert esi.add_ami_fields(zero, ["rotor_rotor", "rotor_stator"]) == 1
    assert esi.add_ami_fields(zero, ["rotor_rotor", "rotor_stator"]) == 0                    # idempotent
    bf = foamio.read_foam((zero / "U").read_text())["boundaryField"]
    assert bf["rotor_rotor"]["type"] == bf["rotor_stator"]["type"] == "cyclicAMI" and bf["inlet"]["type"] == "fixedValue"


def test_the_v2506_log():
    """A captured v2506 pimpleFoam log (the S50 on Perlmutter): the AMI's weights; the flow's Courant number, not
    the 'Mesh Courant Number' line; the time steps."""
    from cfd_translator import foam
    from cfd_translator.transient import courant_stats, progress
    text = LOG.read_text()
    st = esi.ami_stats(text)
    assert st["at_rest"]["source"] == {"min": 1.0, "max": 1.1025322, "average": 1.0001945}
    assert st["min"] == pytest.approx(0.0979, abs=1e-4) and st["max"] == pytest.approx(1.3681, abs=1e-4)
    assert st["low_weight_faces_max"] == 4 and st["low_weight_threshold"] == 0.2
    assert esi.ami_notes(st) == []                                      # sound: averages about 1, max below 1.5
    assert esi.ami_stats_file(LOG)["max"] == st["max"]
    co = courant_stats(LOG)
    assert co["max"] == pytest.approx(68.392862)                        # the mesh Courant (38.0) is not counted
    assert foam.divergence(LOG) is None
    pr = progress(text, 0.1655172)
    assert pr["time"] == pytest.approx(0.000172413) and pr["job_clock_s"] == 15448.0
    lg = foamio.parse_log(text)
    assert lg["iterations"] == 3 and lg["ended"] and "fatal" not in lg
    bad = {"source": "x", "min": 0.0, "max": 1.9, "average_min": 0.95, "average_max": 1.0, "at_rest": {}, "updates": 1}
    assert esi.ami_notes(bad)


def test_ami_balance_reads_both_sides(tmp_path):
    sl = {"rotor": "rotor_rotor", "stator": "rotor_stator"}
    for k, (r, s) in enumerate(((["1e-3", "-2e-4"], ["-1e-3", "2e-4"]), (["5e-4"], ["-4.9e-4"]))):
        d = tmp_path / f"processor{k}" / "0.01"
        d.mkdir(parents=True)
        (d / "phi").write_text("FoamFile\n{\n    class surfaceScalarField;\n}\ninternalField nonuniform List<scalar> 0();\n"
                               "boundaryField\n{\n"
                               f"    rotor_rotor\n    {{\n        type cyclicAMI;\n        value nonuniform List<scalar> {len(r)}({' '.join(r)});\n    }}\n"
                               f"    rotor_stator\n    {{\n        type cyclicAMI;\n        value nonuniform List<scalar> {len(s)}({' '.join(s)});\n    }}\n"
                               "}\n")
    b = esi.ami_balance(tmp_path, sl, 0.01)
    assert b["rotor_side_m3s"] == pytest.approx(1.3e-3) and b["stator_side_m3s"] == pytest.approx(-1.29e-3)
    assert b["imbalance_m3s"] == pytest.approx(1e-5) and b["error"] == pytest.approx(1e-3) and b["coupling"] == "cyclicAMI"


def test_the_transient_run_goes_to_esi_after_the_steady_start(tmp_path, monkeypatch):
    """runner._solve_transient with solve.backend 'esi': the steady MRF start on OpenFOAM 14 (the conformal mesh), then
    the ESI utilities (createBaffles to a cyclicAMI pair, mergeOrSplitBaffles), ESI's dictionaries, and the sliding run
    with backend 'esi'; once."""
    from cfd_translator import foam, runner
    from cfd_translator.sliding import interface
    m = pump_mesh()
    st = _esi_study(degrees_per_step=1)
    st["model"] = "kOmegaSST"
    st["solve"]["transient"]["init"] = 5
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    folder = tmp_path / "s"
    case = folder / "case"
    plan = write_case(st, m, patches_for(m), case)
    seen = []

    def fake_run(c, timeout, procs, resume=False, deadline=None, **k):
        seen.append(("run", resume, k.get("backend", "openfoam.org")))
        if not resume:
            import shutil
            shutil.copytree(c / "0", c / "5")
            return {"log": {"iterations": 5}, "checkMesh": {"ok": True}}
        (c / "log.foamRun").write_text(LOG.read_text())
        return {"log": {"iterations": 3}, "checkMesh": {"ok": True}, "time": 0.4, "reached_end": True}

    def fake_step(fm, cmd, log, c, timeout, raise_on_error=True, **k):
        seen.append((cmd.split()[0], fm.get("version")))
        if fm.get("version") == "v2506":                 # ESI's utilities see a controlDict without function objects
            cd = foamio.read_foam((c / "system" / "controlDict").read_text())
            assert cd["application"] == "pimpleFoam" and not cd.get("functions")
        if cmd.startswith("createBaffles"):
            f, _ = interface(m, m.zones["rotor"])
            b = (c / "constant" / "polyMesh" / "boundary").read_text()
            extra = "".join(f"    {n}\n    {{\n        type            cyclicAMI;\n        nFaces          {len(f)};\n"
                            f"        startFace       {m.nfaces};\n    }}\n" for n in ("rotor_rotor", "rotor_stator"))
            i = b.rindex(")")
            (c / "constant" / "polyMesh" / "boundary").write_text(b[:i] + extra + b[i:])
        return 0
    monkeypatch.setattr(foam, "run", fake_run)
    monkeypatch.setattr(foam, "_step", fake_step)
    monkeypatch.setattr(foam, "find_foam", lambda: {"version": "14"})
    monkeypatch.setattr(esi, "find_esi", lambda refresh=False: {"version": "v2506", "bashrc": "x", "pimpleFoam": "y"})
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    warns = []
    runner._solve_transient(case, st, plan, 3600, 1, folder, state, 0.0, warns, m)
    assert seen[0] == ("run", False, "openfoam.org")                                 # MRF start on OpenFOAM 14
    assert seen[1:4] == [("checkMesh", "14"), ("createBaffles", "v2506"), ("mergeOrSplitBaffles", "v2506")]
    assert seen[4] == ("run", True, "esi")                                           # the sliding run on ESI
    assert not (case / "constant" / "MRFProperties").exists()
    cd = foamio.read_foam((case / "system" / "controlDict").read_text())
    assert cd["application"] == "pimpleFoam" and "cfdtAverage" in cd["functions"]      # the run's, back
    U = foamio.read_foam((case / "0" / "U").read_text())["boundaryField"]
    assert U["blade"]["type"] == "movingWallVelocity" and U["rotor_rotor"]["type"] == "cyclicAMI"
    assert "regIOobject" in (case / "constant" / "polyMesh" / "faceZones").read_text()
    assert plan["sliding"]["backend"] == "esi" and plan["sliding"]["ami_patches"] == {"rotor_rotor": "cyclicAMI",
                                                                                      "rotor_stator": "cyclicAMI"}
    saved = json.loads((folder / "transient.json").read_text())
    assert saved["sliding_done"] and any("cyclicAMI" in w for w in warns)
    assert saved["courant"][0]["max"] == pytest.approx(68.392862)
    seen.clear()
    runner._solve_transient(case, st, plan, 3600, 1, folder, saved, 0.0, [], m)
    assert seen == [("run", True, "esi")]                                            # not split again


def test_ncc_notes_adapted_for_esi():
    notes = ["sliding interface: estimated projected volume fraction 0.571 with the stator side owning the couple",
             "the sliding interface has 300,000 faces: every time step re-intersects them and rebuilds the mesh",
             "sliding zone 'imp': 12 interface faces facing along the axis lie on none of its end planes (-1.5 mm, at r 52): "
             "an end face's edge dragged by snapping, usually where it meets the cylinder next to a wall (S50 v14 port, "
             "mesh 621: the top rim corner, coverage 0 to 1.95 at rest, diverged within 17 steps). Finer cells there",
             "sliding interface, stator side: 3 cells next to walls (thinnest 0.01 mm): a wall feature there is snapped "
             "onto. Such cells seeded the S50's NCC divergence (v14 port, 2026.11.14); keep the interface away",
             "sliding interface cells: worst non-orthogonality 40 deg"]
    out = esi.adapt_notes(notes)
    assert len(out) == 3 and not any("projected volume fraction" in n or "1.95" in n or "NCC" in n for n in out)
    assert "lowWeightCorrection" in out[0] and "Finer cells there" in out[0] and out[2] == notes[4]


def test_sliding_check_notes_on_esi(tmp_path):
    m = pump_mesh()
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    v14 = write_case(_study(), m, patches_for(m), tmp_path / "a")
    e = write_case(_esi_study(), m, patches_for(m), tmp_path / "b")
    assert not any("projected volume fraction" in n for n in e.get("notes", []))
    assert len(e.get("notes", [])) <= len(v14.get("notes", []))


def test_brief_shows_the_coupling():
    from cfd_translator.api import brief
    r = {"ok": True, "results": {"pump": {"head_m": 1.0}}, "mesh": {"cells": 10},
         "solve": {"openfoam": "v2506", "backend": "esi", "transient": True, "time_steps": 1080,
                   "step_profile": {"s_per_step": 2.0},
                   "sliding": {"coupling": "cyclicAMI (ESI OpenFOAM)", "degrees_per_step": 1.0, "revolutions": 3.0,
                               "ami": {"min": 0.93, "max": 1.41, "average_min": 1.0009, "average_max": 1.0048,
                                       "at_rest": {}, "updates": 1081},
                               "interface_balance": {"error": 1.4e-6},
                               "interface_cells": {"cells_per_step": {"value": 1.74}}}}}
    b = brief(r)["solve"]
    assert b["backend"] == "esi" and b["s_per_step"] == 2.0 and b["sliding"]["ami"]["max"] == 1.41
    assert b["sliding"]["interface_balance_error"] == 1.4e-6 and b["sliding"]["cells_per_step"] == 1.74


def test_torque_split_by_part_and_cad_face(tmp_path):
    """post.results(): each rotating wall's torque as pressure + viscous and per CAD face (2026.11.17.6: to set
    cfdt's S50 power gap against another code's moment.dat face by face). The pump mesh with the rotor as an MRF zone,
    a still fluid with a uniform pressure: the friction of the turning blade gives the torque, the pressure none."""
    import numpy as np
    from cfd_translator.post import results
    m = pump_mesh()
    st = _study()
    st["rotating"] = [{k: v for k, v in st["rotating"][0].items() if k != "motion"}]
    st["solve"] = {}
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    plan = write_case(st, m, patches_for(m), tmp_path / "c")
    g = m.geometry()
    zp = plan["zones"]["rotor"]
    pat = {}
    for P in m.patches:
        f = np.arange(P["start"], P["start"] + P["n"])
        U = np.zeros((len(f), 3))
        if P["name"] in zp["rotating_walls"]:
            U = np.cross(np.asarray(zp["axis"]) * zp["omega_rad_s"], g["Cf"][f] * 1e-3)
        pat[P["name"]] = {"U": U, "p": np.full(len(f), 0.3), "phi": np.zeros(len(f)), "wss": None, "nut": None}
    res, _ = results(st, m, plan, {"U": np.zeros((m.ncells, 3)), "p": np.full(m.ncells, 0.3), "patches": pat})
    r = res["rotating"]["rotor"]
    assert r["torque_by_wall_Nm"] and abs(r["torque_Nm"]) > 0 and abs(r["torque_parts_Nm"][r["rotating_walls"][0]]["viscous"]) > 0
    for wn, t in r["torque_by_wall_Nm"].items():
        parts = r["torque_parts_Nm"][wn]
        assert parts["pressure"] + parts["viscous"] == pytest.approx(t, rel=1e-5, abs=1e-15)
    faces = r["torque_by_cad_face_Nm"]
    assert faces and sum(f_["torque"] for f_ in faces) == pytest.approx(r["torque_Nm"], rel=1e-5, abs=1e-15)
    assert all(f_["area_mm2"] > 0 and f_["wall"] in r["rotating_walls"] and "Nm_per_kPa" in f_ for f_ in faces)
    assert "last iteration" in r["torque_by_wall_basis"]
    # a uniform pressure on the blade (a closed set of faces round it) gives no torque: the faces' sensitivities
    # to the pressure level add up to about nothing over the wall
    assert abs(sum(f_["Nm_per_kPa"] for f_ in faces)) < 1e-6 + 1e-3 * sum(abs(f_["Nm_per_kPa"]) for f_ in faces)


def test_whole_revolutions_with_times_given_to_seven_digits():
    """640: end_time 0.1241379 and average_from 0.062069 at 2900 rpm are 3 revolutions, not 2."""
    from cfd_translator.transient import settings
    st = _esi_study()
    st["rotating"][0]["rpm"] = 2900
    st["solve"]["transient"].update(end_time=0.1241379, average_from=0.062069)
    s = settings(st)
    assert s["average_revolutions"] == 3 and s["average_from"] == pytest.approx(0.0620690, abs=1e-6)


def test_zone_holes_found_inside_the_cylinder(tmp_path):
    """post.zone_holes: cells inside the zone's cylinder but left out of the zone are counted, with places."""
    import numpy as np
    from cfd_translator.post import zone_holes
    m = pump_mesh()
    st = _study()
    st["rotating"] = [{k: v for k, v in st["rotating"][0].items() if k != "motion"}]
    st["solve"] = {}
    full = zone_of(m, 11.5)
    m.info["snapped_zones"] = {"rotor": full}
    plan = write_case(st, m, patches_for(m), tmp_path / "c")
    g = m.geometry()
    assert zone_holes(st, m, g, plan)["rotor"]["cells"] == 0
    z = st["rotating"][0]
    d = g["C"] - np.asarray(z.get("center", [0, 0, 0]), float)
    r = np.linalg.norm(d[:, :2], axis=1)
    keep = np.asarray([c for c in m.zones["rotor"] if not (5.0 < r[c] < 7.0)], np.int64)
    m.zones["rotor"] = keep
    zh = zone_holes(st, m, g, plan)["rotor"]
    assert zh["cells"] > 0 and zh["places"] and all(4.5 < p_["r_mm"] < 7.5 for p_ in zh["places"])
