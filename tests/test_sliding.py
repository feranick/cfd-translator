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
    assert calls == ["createBaffles", "splitBaffles", "createNonConformalCouples -fields rotor_rotor rotor_stator"]
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
                     "mpirun -np 2 createNonConformalCouples -parallel -fields rotor_rotor rotor_stator"]
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
