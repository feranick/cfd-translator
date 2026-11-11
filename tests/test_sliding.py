"""Sliding mesh (rotating[].motion 'sliding', 2026.11.02.1): checks, case files, the topology steps (stand-ins
for OpenFOAM's createBaffles, splitBaffles and createNonConformalCouples), time steps and averaging."""
from __future__ import annotations

import numpy as np
import pytest

from cfd_translator import foamio
from cfd_translator.case import rotation_key, write_case
from cfd_translator.format import validate_study
from cfd_translator.polymesh import from_cells
from cfd_translator.sliding import SlidingError, check, interface, set_patch_type, to_sliding

R_IN, R_OUT, L = 5.0, 20.0, 10.0
RADII = [5, 6.5, 8, 10, 12, 14, 16, 18, 20]
TAGS = {"hub": 1, "casing": 2, "inlet": 3, "outlet": 4, "blade": 5, "vane": 6}


def pump_mesh(nt=36, nz=4, blade=(1, 3), vane=(5, 7)):
    """A polar hex mesh about z between a hub (r 5) and a casing (r 20), inlet z = 0, outlet z = 10; a radial
    blade (radial cells blade[0]..blade[1], one sector) and a stationary vane (radial cells vane[0]..vane[1])
    cut out of it."""
    th = np.linspace(0, 2 * np.pi, nt + 1)[:-1]
    zs = np.linspace(0, L, nz + 1)
    nr = len(RADII)
    pts = np.array([[r * np.cos(t), r * np.sin(t), z] for z in zs for t in th for r in RADII])

    def nid(i, j, k):
        return (k * nt + j % nt) * nr + i
    hexes = []
    for k in range(nz):
        for j in range(nt):
            for i in range(nr - 1):
                if (blade and blade[0] <= i < blade[1] and j == 0 and 0 < k < nz - 1) or \
                        (vane and vane[0] <= i < vane[1] and j == nt // 2 and 0 < k < nz - 1):
                    continue
                hexes.append([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                              nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])

    def tag(fb):
        out = []
        for ring in fb:
            P = pts[ring[ring >= 0]]
            c = P.mean(axis=0)
            r = np.hypot(P[:, 0], P[:, 1])
            if np.allclose(P[:, 2], 0):
                out.append(TAGS["inlet"])
            elif np.allclose(P[:, 2], L):
                out.append(TAGS["outlet"])
            elif np.allclose(r, R_IN):
                out.append(TAGS["hub"])
            elif np.allclose(r, R_OUT):
                out.append(TAGS["casing"])
            else:
                out.append(TAGS["blade"] if np.hypot(c[0], c[1]) < 11 else TAGS["vane"])
        return out
    m = from_cells(pts, {"hex": np.array(hexes)}, tag)
    key = rotation_key([0, 0, 1], [0, 0, 0])
    m.info["revolution_tags"] = {key: [TAGS["hub"], TAGS["casing"]]}
    return m


def patches_for(m, blade_rot="rotor", vane_rot=None, hub_rot="rotor"):
    kinds = {"hub": ("wall", {"rotation": hub_rot} if hub_rot else {}),
             "casing": ("wall", {}),
             "inlet": ("inlet", {"velocity": 1.0}),
             "outlet": ("outlet", {"pressure": 0}),
             "blade": ("wall", {"rotation": blade_rot} if blade_rot else {}),
             "vane": ("wall", {"rotation": vane_rot} if vane_rot else {})}
    present = set(m.tags())
    out = []
    for nm, t in TAGS.items():
        if t in present:
            k, bc = kinds[nm]
            out.append({"name": nm, "kind": k, "type": "wall" if k == "wall" else "patch", "tags": [t],
                        "bc": {"kind": k, "name": nm, **bc}})
    m.set_patches([{"name": p["name"], "type": p["type"], "tags": p["tags"]} for p in out])
    return out


def zone_of(m, radius):
    C = m.geometry()["C"]
    return np.nonzero(np.hypot(C[:, 0], C[:, 1]) < radius)[0]


def plan_for(radius, rpm=1500):
    return {"zones": {"rotor": {"axis": [0, 0, 1], "center_mm": [0, 0, 0], "omega_rad_s": rpm * 2 * np.pi / 60}}}


ZONE = {"name": "rotor", "axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 1500, "radius": 11.5, "motion": "sliding"}


def test_a_zone_that_can_slide_and_its_interface():
    m = pump_mesh()
    pts = patches_for(m)
    m.zones = {"rotor": zone_of(m, 11.5)}
    plan = plan_for(11.5)
    check(m, m.geometry(), ZONE, plan, pts)
    sl = plan["sliding"]
    assert sl["moving_walls"] == ["hub", "blade"] and sl["interface_lean_deg"] < 0.1
    f, flip = interface(m, m.zones["rotor"])
    assert sl["interface_faces"] == len(f) == 36 * 4                     # the r = 12 cylinder, every sector and layer
    inz = np.zeros(m.ncells, bool)
    inz[m.zones["rotor"]] = True
    side = np.where(flip, m.neighbour[f], m.owner[f])                    # what createBaffles gives the rotor patch
    assert inz[side].all()


def test_walls_on_the_wrong_side_of_a_sliding_zone():
    m = pump_mesh()
    pts = patches_for(m)
    # the zone reaches the stationary vane (r 14-18): it would turn with the rotor
    m.zones = {"rotor": zone_of(m, 17.5)}
    with pytest.raises(SlidingError, match="'vane' does not turn"):
        check(m, m.geometry(), dict(ZONE, radius=17.5), plan_for(17.5), pts)
    # the zone stops short of the blade's tip (blade r 6.5-10, zone r < 8): its tip would stand still
    m.zones = {"rotor": zone_of(m, 8.5)}
    with pytest.raises(SlidingError, match="'blade' turns with sliding zone"):
        check(m, m.geometry(), dict(ZONE, radius=8.5), plan_for(8.5), pts)
    # a stationary hub (a surface of revolution) may lie inside the zone: it slides along itself
    m2 = pump_mesh(vane=None)
    pts2 = patches_for(m2, hub_rot=None)
    m2.zones = {"rotor": zone_of(m2, 11.5)}
    plan = plan_for(11.5)
    check(m2, m2.geometry(), ZONE, plan, pts2)
    assert plan["sliding"]["moving_walls"] == ["blade"]


def test_a_jagged_zone_boundary_cannot_slide():
    """Gmsh zones are chosen by cell centres: their boundary is a staircase that cannot slide along itself."""
    m = pump_mesh(vane=None)
    pts = patches_for(m)
    C = m.geometry()["C"]
    m.zones = {"rotor": np.nonzero((np.abs(C[:, 0]) < 11.5) & (np.abs(C[:, 1]) < 11.5))[0]}     # a square
    with pytest.raises(SlidingError, match="not a surface of revolution"):
        check(m, m.geometry(), ZONE, plan_for(11.5), pts)


def _study(**tr):
    return {"geometry": {"box": [40, 40, 10]}, "fluid": "water", "model": "laminar",
            "rotating": [dict(ZONE)], "mesh": {"mesher": "snappy", "size": 2},
            "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": 1.0},
                    {"kind": "outlet", "faces": {"nz": 1}, "pressure": 0}],
            "solve": {"transient": {"end_time": 0.4, "average_from": 0.15, **tr}}}


def test_format_and_time_steps():
    from cfd_translator.transient import settings
    st = _study()
    assert validate_study(st)["ok"], validate_study(st)
    e = validate_study(dict(st, solve={}))["errors"]
    assert any("sliding mesh is transient" in x for x in e)
    e = validate_study(dict(st, mesh={"size": 2}))["errors"]
    assert any("mesher snappy" in x for x in e)
    two = dict(st, rotating=[dict(ZONE), dict(ZONE, name="other")])
    assert any("one sliding zone" in x for x in validate_study(two)["errors"])
    assert any("motion" in x for x in validate_study(dict(st, rotating=[dict(ZONE, motion="spin")]))["errors"])
    mrf = dict(st, rotating=[{k: v for k, v in ZONE.items() if k != "motion"}],
               solve={"transient": {"end_time": 0.4, "degrees_per_step": 2}})
    assert any("degrees_per_step" in x for x in validate_study(mrf)["errors"])
    # 1500 rpm: a revolution takes 0.04 s; 1 degree per step at most; the average over whole revolutions
    s = settings(_study())
    assert s["period"] == pytest.approx(0.04) and s["max_dt"] == pytest.approx(0.04 / 360, rel=1e-3)
    assert s["max_co"] == 2.0 and s["revolutions"] == pytest.approx(10)
    assert s["average_revolutions"] == 6 and s["average_from"] == pytest.approx(0.16)
    s = settings(_study(degrees_per_step=2, max_co=0.8))
    assert s["max_dt"] == pytest.approx(0.08 / 360, rel=1e-3) and s["max_co"] == 0.8


def test_case_files_and_the_topology_steps(tmp_path):
    from cfd_translator.average import settings as avg_settings
    from cfd_translator.transient import write_transient
    m = pump_mesh()
    pts = patches_for(m)
    st = _study()
    st["model"] = "kOmegaSST"
    st["solve"]["transient"]["init"] = 0
    st["rotating"] = [dict(ZONE)]
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    case = tmp_path / "case"
    plan = write_case(st, m, pts, case)
    assert plan["sliding"]["moving_walls"] == ["hub", "blade"] and plan["zones"]["rotor"]["motion"] == "sliding"
    assert (case / "constant" / "MRFProperties").exists()                 # the steady start's frame (if any)
    fv = (case / "system" / "fvSolution").read_text()
    assert "correctMeshPhi  yes;" in fv and "MeshPhi" in fv and '"pcorr.*"' in fv and "nOuterCorrectors 2;" in fv
    assert "maxCo           2;" in (case / "system" / "controlDict").read_text()
    cfg = avg_settings(st, plan)
    assert cfg["period"] == pytest.approx(0.04) and cfg["from"] == pytest.approx(0.16)
    # the stand-ins: createBaffles adds the two patches (the original ones keep their faces)
    calls = []
    f, flip = interface(m, m.zones["rotor"])

    def run_step(cmd, log):
        calls.append(cmd)
        if cmd == "createBaffles":
            b = (case / "constant" / "polyMesh" / "boundary").read_text()
            extra = "".join(f"    {n}\n    {{\n        type            patch;\n        nFaces          {len(f)};\n"
                            f"        startFace       {m.nfaces};\n    }}\n" for n in ("rotor_rotor", "rotor_stator"))
            k = b.rindex(")")
            (case / "constant" / "polyMesh" / "boundary").write_text(b[:k] + extra + b[k:])
    # what the steady start leaves beside the solved fields (s180 on spark2: createBaffles stopped on total(p))
    for extra_field in ("total(p)", "phi", "wallShearStress", "yPlus"):
        (case / "0" / extra_field).write_text("FoamFile {}\n")
    notes = to_sliding(case, plan, m, run_step)
    assert sorted(p.name for p in (case / "0").iterdir() if p.is_file()) == ["U", "k", "nut", "omega", "p"]
    assert (case / "0.steady" / "total(p)").exists() and (case / "0.steady" / "phi").exists()
    assert "class       faceZoneList;" in (case / "constant" / "polyMesh" / "faceZones").read_text()
    assert "class       cellZoneList;" in (case / "constant" / "polyMesh" / "cellZones").read_text()
    # the zone's cells (r 5-12 mm, 1.5-2 mm apart) are smaller than the stator's: the stator side is the owner
    mc = plan["sliding"]["interface_cells"]["mean_cell_mm"]
    assert mc["zone_side"] < mc["other_side"] and plan["sliding"]["ncc_owner"] == "rotor_stator"
    assert calls == ["createBaffles", "splitBaffles", "createNonConformalCouples -fields rotor_stator rotor_rotor"]
    ic = plan["sliding"]["interface_cells"]
    assert ic["faces"] == len(f) and ic["max_skewness"] < 1 and ic["size_ratio_max"] >= 1
    assert any(n.startswith("sliding interface cells:") for n in plan["notes"])
    assert not (case / "constant" / "MRFProperties").exists()
    dyn = (case / "constant" / "dynamicMeshDict").read_text()
    assert "solidBody" in dyn and "cellZone        rotor;" in dyn and "1500 [rpm]" in dyn and "axis            (0 0 1)" in dyn
    U = foamio.read_foam((case / "0" / "U").read_text())["boundaryField"]
    assert U["blade"]["type"] == "movingWallVelocity" and U["hub"]["type"] == "movingWallVelocity"
    assert U["casing"]["type"] == "noSlip"
    fz = (case / "constant" / "polyMesh" / "faceZones").read_text()
    assert "rotor_interface" in fz and f"faceLabels List<label> {len(f)}" in fz and f"flipMap List<bool> {len(f)}" in fz
    cb = (case / "system" / "createBafflesDict").read_text()
    assert "zoneName        rotor_interface;" in cb and "movingWallSlipVelocity" in cb and "omega" in cb
    assert "fields          true;" in cb and plan["sliding"]["patches_added"] == ["rotor_rotor", "rotor_stator"]
    assert notes and "1500 rpm" in notes[0]
    # rewriting the transient files keeps the sliding controls
    write_transient(case, st, plan)
    assert "correctPhi      yes;" in (case / "system" / "fvSolution").read_text()


def test_set_patch_type_keeps_the_value():
    txt = ("boundaryField\n{\n    casing\n    {\n        type            noSlip;\n    }\n    blade\n    {\n"
           "        type            MRFnoSlip;\n        value           nonuniform List<vector> 2((1 0 0) (0 1 0));\n    }\n}\n")
    out, ok = set_patch_type(txt, "blade", "movingWallVelocity")
    assert ok and "type            movingWallVelocity;" in out and "(1 0 0) (0 1 0)" in out and "noSlip" in out
    assert not set_patch_type(txt, "nope", "x")[1]


def test_sliding_time_step_estimate():
    from cfd_translator.executor import transient_steps
    st = _study()
    n = transient_steps(st)
    assert n >= 0.4 / (0.04 / 360)               # at least one step per degree over 10 revolutions


def test_the_transient_run_turns_the_mesh_once(tmp_path, monkeypatch):
    """runner._solve_transient: the steady MRF start, then the topology steps once (not again in the next SLURM job's
    segment), then the transient run from the new time 0."""
    import json

    from cfd_translator import foam, runner
    m = pump_mesh()
    pts = patches_for(m)
    st = _study()
    st["model"] = "kOmegaSST"
    st["solve"]["transient"]["init"] = 5
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    folder = tmp_path / "s"
    case = folder / "case"
    plan = write_case(st, m, pts, case)
    seen = []

    def fake_run(c, timeout, procs, resume=False, deadline=None):
        seen.append(("run", resume, (c / "constant" / "dynamicMeshDict").exists()))
        if not resume:                                  # the steady start writes its last iteration
            import shutil
            shutil.copytree(c / "0", c / "5")
            return {"log": {"iterations": 5}, "checkMesh": {"ok": True}}
        (c / "log.foamRun").write_text("Courant Number mean: 0.12 max: 25\nTime = 0.4\n")
        return {"log": {"iterations": 40}, "checkMesh": {"ok": True}, "time": 0.4, "reached_end": True}

    def fake_step(fm, cmd, log, c, timeout, raise_on_error=True, **k):
        seen.append(cmd.split()[0])
        if cmd == "createBaffles":
            f, _ = interface(m, m.zones["rotor"])
            b = (c / "constant" / "polyMesh" / "boundary").read_text()
            extra = "".join(f"    {n}\n    {{\n        type            patch;\n        nFaces          {len(f)};\n"
                            f"        startFace       {m.nfaces};\n    }}\n" for n in ("rotor_rotor", "rotor_stator"))
            i = b.rindex(")")
            (c / "constant" / "polyMesh" / "boundary").write_text(b[:i] + extra + b[i:])
        return 0
    monkeypatch.setattr(foam, "run", fake_run)
    monkeypatch.setattr(foam, "_step", fake_step)
    monkeypatch.setattr(foam, "find_foam", lambda: {"version": "14"})
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    warns = []
    runner._solve_transient(case, st, plan, 3600, 1, folder, state, 0.0, warns, m)
    assert seen[0] == ("run", False, False)                               # MRF start on the conformal mesh
    assert seen[1:5] == ["checkMesh", "createBaffles", "splitBaffles", "createNonConformalCouples"]
    assert seen[5] == ("run", True, True)                                 # then the sliding run
    saved = json.loads((folder / "transient.json").read_text())
    assert saved["sliding_done"] and any("sliding mesh" in w for w in warns)
    assert saved["courant"] == [{"segment": 1, "time": 0.4, "mean_last": 0.12, "max_last": 25.0, "max": 25.0,
                                 "max_median": 25.0, "steps_seen": 1}]                  # this segment's Courant numbers
    U = foamio.read_foam((case / "0" / "U").read_text())["boundaryField"]
    assert U["blade"]["type"] == "movingWallVelocity"
    # a later segment does not split the mesh again
    seen.clear()
    runner._solve_transient(case, st, plan, 3600, 1, folder, saved, 0.0, [], m)
    assert seen == [("run", True, True)]


def _labels(path, a):
    from cfd_translator.polymesh import header
    path.write_text(header("labelList", path.name, "constant/polyMesh") + f"\n{len(a)}\n(\n" +
                    "\n".join(str(int(x)) for x in a) + "\n)\n")


def test_parallel_topology_steps_and_the_decomposed_reader(tmp_path, monkeypatch):
    """s180 (13.3 M cells, spark2, 2026.11.03): decomposePar of the coupled mesh stopped after 'Distributing faces
    to processors' (processInterCyclics loops over every pair of the couple's faces). Now the conformal case is
    decomposed first, the three utilities run in parallel, and the results are read from processor*."""
    from cfd_translator import foam
    from cfd_translator.case import _field, _nonuniform
    from cfd_translator.polymesh import header
    m = pump_mesh()
    pts = patches_for(m)
    st = _study()
    st["model"] = "kOmegaSST"
    st["solve"]["transient"]["init"] = 0
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    case = tmp_path / "case"
    plan = write_case(st, m, pts, case)
    f, _ = interface(m, m.zones["rotor"])
    C = m.geometry()["C"]
    proc_of = (C[:, 0] > 0).astype(int)                       # two processors: x <= 0, x > 0
    calls = []

    def run_step(cmd, log):
        calls.append(cmd)
        if cmd == "decomposePar -force":
            for k in range(2):
                d = case / f"processor{k}" / "constant" / "polyMesh"
                d.mkdir(parents=True)
                _labels(d / "cellProcAddressing", np.nonzero(proc_of == k)[0])
                fa, lines, start = [], [], 0
                for P in m.patches:
                    gf = np.arange(P["start"], P["start"] + P["n"])
                    mine = gf[proc_of[m.owner[gf]] == k]
                    fa += list(mine + 1)
                    lines += [f"    {P['name']}", "    {", "        type            patch;",
                              f"        nFaces          {len(mine)};", f"        startFace       {start};", "    }"]
                    start += len(mine)
                _labels(d / "faceProcAddressing", fa)
                (d / "boundary").write_text(header("polyBoundaryMesh", "boundary", "constant/polyMesh") +
                                            f"{len(m.patches)}\n(\n" + "\n".join(lines) + "\n)\n")
        elif "createBaffles" in cmd:
            for k, n in ((0, len(f) // 2), (1, len(f) - len(f) // 2)):
                b = case / f"processor{k}" / "constant" / "polyMesh" / "boundary"
                t = b.read_text()
                extra = "".join(f"    {nm}\n    {{\n        type            patch;\n        nFaces          {n};\n"
                                f"        startFace       0;\n    }}\n" for nm in ("rotor_rotor", "rotor_stator"))
                i = t.rindex(")")
                b.write_text(t[:i] + extra + t[i:])
    notes = to_sliding(case, plan, m, run_step, 2, "mpirun")
    assert calls == ["decomposePar -force", "mpirun -np 2 createBaffles -parallel", "mpirun -np 2 splitBaffles -parallel",
                     "mpirun -np 2 createNonConformalCouples -parallel -fields rotor_stator rotor_rotor"]
    assert foam.decomposed(case)["procs"] == 2 and "stays decomposed" in notes[0]
    assert (case / "processor1" / "cfdt_addressing" / "faceProcAddressing").is_file()
    assert "numberOfSubdomains 2;" in (case / "system" / "decomposeParDict").read_text()
    # fields written per processor by the 'solver': read back in the complete mesh's layout
    p_true = C[:, 0] * 0.1 + C[:, 1]
    U_true = np.column_stack([C[:, 0], C[:, 1], C[:, 2]])
    for k in range(2):
        t = case / f"processor{k}" / "0.4"
        t.mkdir()
        cells = np.nonzero(proc_of == k)[0]
        bp, bu, bphi = {}, {}, {}
        for P in m.patches:
            gf = np.arange(P["start"], P["start"] + P["n"])
            mine = gf[proc_of[m.owner[gf]] == k]
            Cf = m.geometry()["Cf"][mine]
            bp[P["name"]] = {"type": "calculated", "value": _nonuniform(Cf[:, 0] * 0.1 + Cf[:, 1])}
            bu[P["name"]] = {"type": "calculated", "value": _nonuniform(Cf)} if len(mine) else {"type": "noSlip"}
            bphi[P["name"]] = {"type": "calculated", "value": _nonuniform(np.full(len(mine), float(k + 1)))}
        (t / "p").write_text(_field("volScalarField", "p", "[0 2 -2 0 0 0 0]", _nonuniform(p_true[cells]), bp))
        (t / "U").write_text(_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", _nonuniform(U_true[cells]), bu))
        (t / "phi").write_text(_field("surfaceScalarField", "phi", "[0 3 -1 0 0 0 0]", "uniform 0", bphi))
    fields = foam.read_fields(case, m, plan)
    assert fields["time"] == "0.4"
    assert np.allclose(fields["p"], p_true) and np.allclose(fields["U"], U_true)
    for P in m.patches:
        gf = np.arange(P["start"], P["start"] + P["n"])
        Cf = m.geometry()["Cf"][gf]
        assert np.allclose(fields["patches"][P["name"]]["p"], Cf[:, 0] * 0.1 + Cf[:, 1])
        assert np.allclose(fields["patches"][P["name"]]["phi"], proc_of[m.owner[gf]] + 1)
    # the solver must keep the decomposition: another rank count is refused, reconstructPar is not run
    monkeypatch.setattr(foam, "find_foam", lambda: {"version": "14"})
    with pytest.raises(foam.FoamError, match="stays decomposed for 2 ranks"):
        foam.run(case, 100, 4, resume=True)


def test_round_rotor_walls_outside_the_zone_keep_turning(tmp_path):
    """S50 (spark2, 2026.11.03): 68 % of 'impeller_back' (hub, shaft, back shroud: surfaces of revolution) lay
    outside the sliding zone and stood still, so its disk friction was missing from the torque."""
    m = pump_mesh(vane=None)
    pts = patches_for(m)
    m.zones = {"rotor": zone_of(m, 11.5)}
    # the casing (r 20) as a rotor wall with its own rotation at the zone's speed: all of it outside the zone
    for p in pts:
        if p["name"] == "casing":
            p["bc"]["rotation"] = {"axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 1500}
    plan = plan_for(11.5)
    check(m, m.geometry(), ZONE, plan, pts)
    assert plan["sliding"]["spinning_walls"] == ["casing"] and "casing" not in plan["sliding"]["moving_walls"]
    assert any("rotatingWallVelocity" in n for n in plan["notes"])
    assert plan["sliding"]["interface_cells"]["max_non_ortho"] < 5


def test_own_rotation_walls_count_in_the_zone_torque(tmp_path):
    m = pump_mesh(vane=None)
    pts = patches_for(m)
    for p in pts:
        if p["name"] == "casing":
            p["bc"]["rotation"] = {"axis": [0, 0, 1], "center": [0, 0, 5], "rpm": 1500}     # same axis line and speed
    st = dict(_study(), model="kOmegaSST", rotating=[dict(ZONE, motion="mrf")])
    st["solve"] = {"iterations": 10}
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    plan = write_case(st, m, pts, tmp_path / "case")
    assert plan["zones"]["rotor"]["rotating_walls"] == ["hub", "blade", "casing"]


def test_spinning_wall_bc_and_mesh_phi_retry(tmp_path, monkeypatch):
    from cfd_translator import foam, runner
    from cfd_translator.runner import mesh_phi_crash
    m = pump_mesh(vane=None)
    pts = patches_for(m)
    for p in pts:
        if p["name"] == "casing":
            p["bc"]["rotation"] = {"axis": [0, 0, 1], "rpm": 1500}
    st = _study()
    st["model"] = "kOmegaSST"
    st["solve"]["transient"]["init"] = 0
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    folder = tmp_path / "s"
    case = folder / "case"
    plan = write_case(st, m, pts, case)
    f, _ = interface(m, m.zones["rotor"])
    runs = []

    def fake_step(fm, cmd, log, c, timeout, raise_on_error=True, **k):
        if cmd == "createBaffles":
            b = (c / "constant" / "polyMesh" / "boundary").read_text()
            extra = "".join(f"    {n}\n    {{\n        type            patch;\n        nFaces          {len(f)};\n"
                            f"        startFace       {m.nfaces};\n    }}\n" for n in ("rotor_rotor", "rotor_stator"))
            i = b.rindex(")")
            (c / "constant" / "polyMesh" / "boundary").write_text(b[:i] + extra + b[i:])
        return 0

    def fake_run(c, timeout, procs, resume=False, deadline=None):
        runs.append("correctMeshPhi  yes" in (c / "system" / "fvSolution").read_text())
        if len(runs) == 1:
            (c / "log.foamRun").write_text("#3  Foam::fvMeshStitchers::moving::unconformInternalFaceCorrectMeshPhi(...)\n"
                                           "#4  free()\nSegmentation fault\n")
            raise foam.FoamError("foamRun exited with code 139")
        return {"log": {"iterations": 10}, "checkMesh": {"ok": True}, "time": 0.4, "reached_end": True}
    monkeypatch.setattr(foam, "run", fake_run)
    monkeypatch.setattr(foam, "_step", fake_step)
    monkeypatch.setattr(foam, "find_foam", lambda: {"version": "14"})
    (case / "log.checkMesh").write_text("Mesh OK.\n")
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    warns = []
    runner._solve_transient(case, st, plan, 3600, 1, folder, state, 0.0, warns, m)
    assert runs == [True, False] and state["correct_mesh_phi"] is False
    assert any("correctMeshPhi off" in w for w in warns)
    U = (case / "0" / "U").read_text()
    blk = U[U.index("    casing"):]
    blk = blk[:blk.index("}")]
    assert "rotatingWallVelocity" in blk and "omega           157.0796" in blk and "axis            (0 0 1)" in blk
    assert mesh_phi_crash(case)
    from cfd_translator.format import validate_study
    assert validate_study(_study(correct_mesh_phi=False))["ok"]
    assert any("correct_mesh_phi" in e for e in validate_study(_study(correct_mesh_phi="no"))["errors"])


def test_cell_size_mismatch_across_the_interface_is_reported(monkeypatch):
    """S50 (spark1, 2026.11.04): 0.2 mm cells on one side of the zone's end, 0.5 mm on the other; OpenFOAM's
    projected volume fraction grew to 1.06 in 0.26 degrees and the run gave NaN."""
    monkeypatch.setitem(pump_mesh.__globals__, "RADII", [5, 6.5, 8, 10, 11, 11.5, 11.8, 12, 18, 20])
    m = pump_mesh(vane=None)
    pts = patches_for(m)
    m.zones = {"rotor": zone_of(m, 11.95)}
    plan = plan_for(11.95)
    check(m, m.geometry(), dict(ZONE, radius=11.95), plan, pts)
    ic = plan["sliding"]["interface_cells"]
    assert ic["size_ratio_max"] > 3 and ic["size_ratio_over_2"] == 1.0
    assert any("differ in size" in n and "Refine both sides" in n for n in plan["notes"])


def test_skewness_like_checkmesh():
    from cfd_translator.polymesh import box_mesh
    m = box_mesh(10, 10, 10, 2, 1, 1)
    q = m.quality()
    assert q["max_skewness"] < 1e-9 and q["skewness_over_4"] == 0
    # move the middle face's points off centre along y: its centre leaves the line between the cell centres
    mid = np.isclose(m.points[:, 0], 0.0)
    m.points = m.points.copy()
    m.points[mid & (m.points[:, 1] > 0), 0] += 3.0
    m._geo = None
    q = m.quality()
    assert q["max_skewness"] > 0.05


def test_rings_and_refinement_that_is_not_round():
    """S50 (2026.11.04): 18 boxes of 0.25 mm cells at the vane tips, at fixed angles, inside 1.25 mm cells: both
    sides match facing each other at t = 0, but as the rotor turns its fine cells pass coarse stator cells."""
    from cfd_translator.sliding import not_round_regions, ring_ratio
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    R = 59.7
    Cf = np.column_stack([R * np.cos(th), R * np.sin(th), np.full(360, -60.0)])
    tips = (np.round(np.degrees(th)).astype(int) % 20) < 2                  # 18 boxes, 2 degrees each
    L = np.where(tips, 0.25, 1.25)
    a, o = np.array([0, 0, 1.0]), np.zeros(3)
    facing = np.maximum(L, L) / np.minimum(L, L)
    assert facing.max() == 1.0
    rg = ring_ratio(Cf, L, L, a, o, 0.25)
    assert rg["max"] == pytest.approx(5.0) and rg["faces_over_2"] == 1.0 and rg["radius_mm"] == pytest.approx(R, abs=0.1)
    assert ring_ratio(Cf, np.full(360, 0.5), np.full(360, 0.25), a, o, 0.25)["max"] == pytest.approx(2.0)
    refine = [{"box": [57, -1, -70, 62, 1, -50], "size": 0.25},
              {"cylinder": {"center": [0, 0, -77.1], "axis": [0, 0, 1], "radius": 61, "span": [0, 30]}, "size": 0.25},
              {"cylinder": {"center": [5, 0, -77.1], "axis": [0, 0, 1], "radius": 61, "span": [0, 30]}, "size": 0.25},
              {"at": [59.7, 0, -60], "r": 2, "size": 0.25}, {"distance": 2, "size": 0.5},
              {"box": [-5, -5, 0, 5, 5, 10], "size": 1}]
    bad = not_round_regions(refine, Cf, a, o)
    assert [(w["index"], w["shape"]) for w in bad] == [(0, "box"), (2, "cylinder"), (3, "sphere")]


def test_interface_off_its_cylinder_is_found_and_stops_the_run():
    """S50 (2026.11.04): the snapped zone boundary was stepped by the vane tips (faces 87 degrees off the cylinder,
    r 59.66 against the zone's 59.7, tips 59.45, bore 60.0); the 6th try diverged at the 28th step."""
    from cfd_translator.sliding import off_surface
    m = pump_mesh(vane=None)
    pts = patches_for(m)
    m.zones = {"rotor": zone_of(m, 11.5)}
    plan = plan_for(11.5)
    check(m, m.geometry(), dict(ZONE, radius=12.0), plan, pts)
    off = plan["sliding"]["interface_cells"]["off_surface"]
    assert off["max_cells"] < 0.05 and off["max_deg"] < 6 and off["places"] == []
    # a rim of 0.25 mm faces round r 59.7, z -82.25..-46.25, with one step: a face turned along the axis, 0.2 mm out
    a, o = np.array([0, 0, 1.0]), np.zeros(3)
    th = np.linspace(0, 2 * np.pi, 1500, endpoint=False)
    Cf = np.column_stack([59.7 * np.cos(th), 59.7 * np.sin(th), np.full(1500, -60.0)])
    Sf = np.column_stack([np.cos(th), np.sin(th), np.zeros(1500)]) * 0.0625
    Cf[700, :2] *= 59.9 / 59.7
    Sf[700] = [0, 0, 0.0625]
    off = off_surface(Cf, Sf, np.full(1500, 0.25), a, o, 59.7, -82.25, -46.25)
    assert off["max_cells"] == pytest.approx(0.8, abs=0.01) and off["max_deg"] == pytest.approx(90.0)
    assert off["places"][0]["part"] == "rim" and off["places"][0]["at"][2] == -60.0
    # end faces are judged against their own plane
    Ce = np.column_stack([np.linspace(0, 50, 50), np.zeros(50), np.full(50, -46.25)])
    off = off_surface(Ce, np.tile([0, 0, 0.0625], (50, 1)), np.full(50, 0.25), a, o, 59.7, -82.25, -46.25)
    assert off["max_cells"] < 1e-9 and off["max_deg"] < 1e-6
    # a stepped boundary stops before the solve (here a stand-in report: the test mesh is too coarse for a step
    # that passes the surface-of-revolution gate)
    import cfd_translator.sliding as sl_mod
    rep = {"max_cells": 400.0, "max_deg": 87.2, "share_over_half_cell": 0.02, "share_over_30_deg": 0.01,
           "places": [{"at": [-11.2, 58.6, -71.1], "cells": 0.9, "deg": 87.2, "part": "rim", "radius_mm": 59.66,
                       "axial_mm": 11.2}]}
    orig = sl_mod.off_surface
    sl_mod.off_surface = lambda *a_, **k_: rep
    try:
        with pytest.raises(SlidingError, match="did not snap the zone's boundary") as ei:
            check(m, m.geometry(), dict(ZONE, radius=12.0), plan_for(11.5), pts)
        assert "[-11.2, 58.6, -71.1] mm" in str(ei.value) and "4 or more cells" in str(ei.value)
    finally:
        sl_mod.off_surface = orig


def test_fixed_time_step(tmp_path):
    """max_co null: exactly degrees_per_step a step, no Courant limit (snappy's micrometre cells in the S50's 0.5 mm
    gaps took max_co 2 to 0.02 deg a step; the v2506 run went at 1 deg, Courant ~25 in a few cells, mean 0.12)."""
    from cfd_translator.executor import transient_steps
    from cfd_translator.transient import courant_stats, settings
    st = _study(degrees_per_step=1, max_co=None)
    assert validate_study(st)["ok"], validate_study(st)
    s = settings(st)
    assert s["fixed_step"] and s["max_co"] is None and s["dt"] == pytest.approx(0.04 / 360, rel=1e-5)
    assert transient_steps(st) == pytest.approx(0.4 / (0.04 / 360) * 1.6, rel=1e-4)        # 3600 steps (x 1.6)
    m = pump_mesh()
    st2 = dict(st, model="kOmegaSST", rotating=[dict(ZONE)])
    st2["solve"] = {"transient": dict(st["solve"]["transient"], init=0)}
    m.info["snapped_zones"] = {"rotor": zone_of(m, 11.5)}
    case = tmp_path / "case"
    plan = write_case(st2, m, patches_for(m), case)
    from cfd_translator.transient import write_transient
    write_transient(case, st2, plan)
    cd = (case / "system" / "controlDict").read_text()
    assert "adjustTimeStep  no;" in cd and "maxCo" not in cd and "writeControl    runTime;" in cd
    assert f"deltaT          {0.04 / 360:.6g};" in cd
    # without a sliding zone a fixed step needs dt
    plain = {k: v for k, v in _study().items() if k != "rotating"}
    plain["solve"] = {"transient": {"end_time": 0.4, "max_co": None}}
    assert any("fixed time step" in e for e in validate_study(plain)["errors"])
    plain["solve"]["transient"]["dt"] = 1e-4
    assert not any("fixed time step" in e for e in validate_study(plain)["errors"])
    assert settings(plain)["dt"] == 1e-4 and settings(plain)["fixed_step"]
    # the Courant numbers OpenFOAM printed
    log = tmp_path / "log.foamRun"
    log.write_text("".join(f"Courant Number mean: 0.12 max: {mx}\nTime = {k}\n" for k, mx in enumerate((20, 25, 31, 24))))
    c = courant_stats(log)
    assert c == {"mean_last": 0.12, "max_last": 24.0, "max": 31.0, "max_median": 25.0, "steps_seen": 4}
