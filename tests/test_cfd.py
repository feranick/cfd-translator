"""Tests of cfd-translator. Run: pytest -q
Tests that need Gmsh, OpenFOAM, build123d-translator or fea-translator are skipped when missing."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cfd_translator import api, bench, foamio
from cfd_translator.case import inlet_shape
from cfd_translator.format import validate_study
from cfd_translator.polymesh import box_mesh, extrude, from_cells
from cfd_translator.post import results
from cfd_translator.refsolver import solve_case
from cfd_translator.runner import prepare, run_study
from cfd_translator.select import Groups, assign_bcs, mesh_groups

try:
    import gmsh  # noqa: F401
    HAVE_GMSH = True
except ImportError:
    HAVE_GMSH = False
try:
    from cfd_translator.foam import find_foam
    _f = find_foam()
    HAVE_FOAM = bool(_f) and "error" not in _f
except Exception:  # noqa: BLE001
    HAVE_FOAM = False
gmsh_only = pytest.mark.skipif(not HAVE_GMSH, reason="needs Gmsh")
foam_only = pytest.mark.skipif(not HAVE_FOAM, reason="needs OpenFOAM")

CHANNEL = {"geometry": {"box": [1000, 100, 10]}, "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
           "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.01, "profile": "parabolic"},
                   {"kind": "outlet", "faces": {"nx": 1}}],
           "mesh": {"cells": [40, 16, 1]}, "probes": [{"name": "a", "at": [-300, 0, 5]}, {"name": "b", "at": [300, 0, 5]}]}


# ---- mesh ---------------------------------------------------------------------------------------
def test_box_mesh_is_a_valid_polymesh():
    m = box_mesh(100, 50, 20, 10, 5, 4, grading=(2, 1, 0.5))
    q = m.quality()
    assert q["ok"] and q["upper_triangular"] and q["boundary_outward"]
    assert q["volume_mm3"] == pytest.approx(100 * 50 * 20)
    assert m.tags() == [0, 1, 2, 3, 4, 5]
    assert m.ncells == 200 and m.ninternal == 3 * 200 - (5 * 4 + 10 * 4 + 10 * 5)


def test_tets_prisms_pyramids_are_closed_and_matched():
    # a unit cube: one hex split into a pyramid + tets, and a prism stacked on top
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                  [0, 0, 2], [1, 0, 2], [1, 1, 2]], float)
    c = np.array([0.5, 0.5, 0.5])
    p = np.vstack([p, c])
    C = 11
    pyr = [[0, 1, 2, 3, C], [4, 7, 6, 5, C]]
    tets = [[0, 1, 5, C], [0, 5, 4, C], [1, 2, 6, C], [1, 6, 5, C], [2, 3, 7, C], [2, 7, 6, C], [3, 0, 4, C], [3, 4, 7, C]]
    prism = [[4, 5, 6, 8, 9, 10]]
    m = from_cells(p, {"tet": tets, "pyramid": pyr, "prism": prism}, lambda fb: np.zeros(len(fb), np.int64))
    q = m.quality()
    assert q["ok"], q
    assert q["volume_mm3"] == pytest.approx(1.0 + 0.5)


def test_extrude_keeps_side_tags_and_volume():
    p2 = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [1, 0], [1, 1]], float)
    m = extrude(p2, [[0, 4, 5]], [[4, 1, 2, 5]], 0, 3, 4, 10, 11,
                {(0, 4): 1, (4, 1): 1, (1, 2): 2, (2, 5): 3, (5, 3): 3, (3, 0): 4, (0, 5): 99} | {}, {})
    # (0, 5) is not a boundary edge of this mesh: unused
    m2 = extrude(p2, [[0, 4, 5], [0, 5, 3]], [[4, 1, 2, 5]], 0, 3, 4, 10, 11,
                 {(0, 4): 1, (4, 1): 1, (1, 2): 2, (2, 5): 3, (5, 3): 3, (3, 0): 4})
    assert m2.quality()["ok"]
    assert m2.quality()["volume_mm3"] == pytest.approx(2 * 1 * 3)
    assert m2.tags() == [1, 2, 3, 4, 10, 11]
    with pytest.raises(Exception):
        extrude(p2, [[0, 4, 5], [0, 5, 3]], [[4, 1, 2, 5]], 0, 3, 4, 10, 11, {(0, 4): 1})   # missing edges
    assert m.ncells == 8


def test_polymesh_files_read_back(tmp_path):
    m = box_mesh(10, 10, 1, 3, 2, 1)
    m.set_patches([{"name": "a", "type": "patch", "tags": [0, 1]}, {"name": "w", "type": "wall", "tags": [2, 3]},
                   {"name": "e", "type": "empty", "tags": [4, 5]}])
    d = m.write(tmp_path)
    pts = foamio.read_list((d / "points").read_text())
    faces = foamio.read_faces((d / "faces").read_text())
    own = foamio.read_list((d / "owner").read_text())
    nei = foamio.read_list((d / "neighbour").read_text())
    bnd = foamio.read_boundary((d / "boundary").read_text())
    assert np.allclose(pts, m.points * 1e-3)
    assert [f for f in faces] == [list(r[r >= 0]) for r in m.faces]
    assert np.array_equal(own.astype(int), m.owner) and np.array_equal(nei.astype(int), m.neighbour)
    assert [(b["name"], b["type"], b["n"], b["start"]) for b in bnd] == [(p["name"], p["type"], p["n"], p["start"]) for p in m.patches]


# ---- selection --------------------------------------------------------------------------------
def test_selectors_internal_and_external_normals():
    m = box_mesh(100, 50, 20, 4, 2, 2)
    g = mesh_groups(m, external=False)
    assert g.select({"nx": 1}) == [1] and g.select({"nz": -1}) == [4]
    assert g.select({"pts": [[0, 25, 10]]}) == [3]
    assert sorted(g.select({"all": True})) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(Exception):
        g.select({"box": "xmin"})
    # external: pretend xmin/xmax are faces of the part (seen from the fluid, normals flip)
    info = {0: {"on": "part"}, 1: {"on": "part"}, 2: {"on": "box", "box": "ymin"}, 3: {"on": "box", "box": "ymax"},
            4: {"on": "box", "box": "zmin"}, 5: {"on": "box", "box": "zmax"}}
    b = m.boundary_slice()
    ge = Groups(m.points, m.faces[b], m.btag, info, external=True)
    assert ge.select({"nx": 1}) == [0]                 # the part's +x face is the fluid's -x boundary
    assert ge.select({"box": ["ymin", "zmax"]}) == [2, 5]
    assert sorted(ge.select({"all": True})) == [0, 1]


def test_assign_defaults_and_conflicts():
    m = box_mesh(100, 50, 20, 4, 2, 1)
    g = mesh_groups(m, external=False)
    st = {"bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}]}
    patches, sel, warns = assign_bcs(g, st, [4, 5])
    names = [p["name"] for p in patches]
    assert names == ["inlet", "outlet", "walls", "frontAndBack"]
    assert patches[2]["tags"] == [2, 3] and patches[3]["type"] == "empty"
    st2 = {"bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "wall", "faces": [{"nx": -1}, {"ny": 1}]}]}
    with pytest.raises(Exception, match="taken by"):
        assign_bcs(g, st2)
    st3 = {"bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "slip", "faces": {"all": True}}]}
    p3, _, _ = assign_bcs(g, st3)
    assert p3[0]["tags"] == [0] and sorted(p3[1]["tags"]) == [1, 2, 3, 4, 5]     # 'all' yields


# ---- format -----------------------------------------------------------------------------------
def test_validation_messages():
    assert validate_study(CHANNEL)["ok"]
    bad = dict(CHANNEL, bcs=[{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": [1, 0, 0], "profile": "parabolic"}])
    e = " ".join(validate_study(bad)["errors"])
    assert "box" in e and "parabolic" in e and "outlet" in e
    assert not validate_study(dict(CHANNEL, mesh={"cells": [10, 10, 2]}))["ok"]          # 2d needs nz = 1
    assert not validate_study({**CHANNEL, "bcs": [{"kind": "wall", "faces": {"all": True}}]})["ok"]   # no driver
    f = api.study_format()
    assert len(json.dumps(f)) < 12000 and api.validate(f["example"])["ok"]


# ---- case -------------------------------------------------------------------------------------
def test_case_files_and_exact_inlet_flow(tmp_path):
    st = dict(CHANNEL, model="kOmegaSST")
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    c = tmp_path / "case"
    U = foamio.read_foam((c / "0" / "U").read_text())
    inl = U["boundaryField"]["inlet"]["value"]
    assert inl.shape == (16, 3)
    f = mesh.patch_faces("inlet")
    Sf = mesh.geometry()["Sf"][f] * 1e-6
    Q = -np.einsum("ij,ij->i", inl, Sf).sum()
    assert Q == pytest.approx(0.01 * 0.1 * 0.01, rel=1e-12)
    assert plan["inlets"]["inlet"]["Q"] == pytest.approx(Q, rel=1e-12)
    assert inl[:, 0].max() == pytest.approx(1.5 * 0.01, rel=0.01)                     # parabola: max = 1.5 mean
    p = foamio.read_foam((c / "0" / "p").read_text())
    assert p["boundaryField"]["outlet"]["type"] == "fixedValue" and p["boundaryField"]["frontAndBack"]["type"] == "empty"
    k = foamio.read_foam((c / "0" / "k").read_text())
    assert k["boundaryField"]["walls"]["type"] == "kqRWallFunction"
    cd = foamio.read_foam((c / "system" / "controlDict").read_text())
    assert cd["solver"] == "incompressibleFluid" and "wallShearStress" in cd["functions"] and "yPlus" in cd["functions"]
    fv = foamio.read_foam((c / "system" / "fvSolution").read_text())
    assert fv["SIMPLE"]["consistent"] == "yes"
    pp = foamio.read_foam((c / "constant" / "physicalProperties").read_text())
    assert pp["nu"] == pytest.approx(1e-3)
    mt = foamio.read_foam((c / "constant" / "momentumTransport").read_text())
    assert mt["RAS"]["model"] == "kOmegaSST"


def _disk_mesh(R=5.0, nr=8, nt=32, h=20.0, layers=4):
    pts = [[0.0, 0.0]]
    for i in range(1, nr + 1):
        for j in range(nt):
            a = 2 * math.pi * j / nt
            pts.append([R * i / nr * math.cos(a), R * i / nr * math.sin(a)])
    pts = np.array(pts)
    tris, quads = [], []
    ring = lambda i, j: 1 + (i - 1) * nt + (j % nt)        # noqa: E731
    for j in range(nt):
        tris.append([0, ring(1, j), ring(1, j + 1)])
    for i in range(1, nr):
        for j in range(nt):
            quads.append([ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1)])
    edges = {(ring(nr, j), ring(nr, j + 1)): 1 for j in range(nt)}
    return extrude(pts, tris, quads, 0, h, layers, 2, 3, edges)


def test_parabolic_profiles_circle_and_square():
    m = _disk_mesh()
    m.set_patches([{"name": "in", "type": "patch", "tags": [2]}, {"name": "rest", "type": "wall", "tags": [1, 3]}])
    rel, dh, d = inlet_shape(m, m.patch_faces("in"), False, "parabolic")
    assert d["shape"] == "circle"
    A = np.linalg.norm(m.geometry()["Sf"][m.patch_faces("in")], axis=1)
    assert (rel * A).sum() / A.sum() == pytest.approx(1.0)
    assert rel.max() == pytest.approx(2.0, rel=0.05)                                    # pipe: max = 2 mean
    sq = box_mesh(10, 10, 30, 20, 20, 2)
    sq.set_patches([{"name": "in", "type": "patch", "tags": [4]}, {"name": "w", "type": "wall", "tags": [0, 1, 2, 3, 5]}])
    rel, dh, d = inlet_shape(sq, sq.patch_faces("in"), False, "parabolic")
    assert d["shape"] == "rectangle" and dh == pytest.approx(0.01)
    assert rel.max() == pytest.approx(2.0962, rel=0.01)                                 # square duct: 2.0962


# ---- reference solver and results ---------------------------------------------------------------
def test_reference_couette_exact(tmp_path):
    r = bench.run("ref", ["couette_2d"], tmp_path)["couette_2d"]
    assert r["passed"], r
    assert r["invariants"]["mass_balance"] < 1e-12 and r["invariants"]["momentum_balance"] < 1e-9


def test_reference_channel_and_cavity(tmp_path):
    r = bench.run("ref", ["channel_2d", "cavity_re100"], tmp_path)
    assert r["channel_2d"]["passed"], r["channel_2d"]
    assert r["cavity_re100"]["passed"], [c for c in r["cavity_re100"]["checks"] if not c["passed"]]


def test_reference_solver_reads_the_case_not_the_study(tmp_path):
    """Change the viscosity in the written case: the reference solver must follow the case."""
    mesh, patches, sel, plan, w = prepare(CHANNEL, tmp_path)
    c = tmp_path / "case"
    f1 = solve_case(c)
    pp = c / "constant" / "physicalProperties"
    pp.write_text(pp.read_text().replace("nu              0.001;", "nu              0.002;"))
    f2 = solve_case(c)
    r1, _ = results(CHANNEL, mesh, plan, f1)
    r2, _ = results(CHANNEL, mesh, plan, f2)
    assert r2["pressure_drop"] == pytest.approx(2 * r1["pressure_drop"], rel=1e-4)


# ---- OpenFOAM readers (fixtures in OpenFOAM's own output style) --------------------------------
def _write_foam_time(case: Path, mesh, fields, t="57"):
    """Write fields as OpenFOAM writes them: long lists over lines, short ones inline, no value for
    zeroGradient / noSlip / empty."""
    d = case / t
    d.mkdir()

    def lst(a, typ):
        a = np.asarray(a, float)
        if typ == "vector":
            items = [f"({x:.10g} {y:.10g} {z:.10g})" for x, y, z in a]
        else:
            items = [f"{x:.10g}" for x in a]
        if len(items) <= 10:
            return f"nonuniform List<{typ}> {len(items)}(" + " ".join(items) + ")"
        return f"nonuniform List<{typ}> \n{len(items)}\n(\n" + "\n".join(items) + "\n)\n"

    def write(name, cls, typ, internal, per_patch):
        s = ["FoamFile\n{\n    format      ascii;\n    class       %s;\n    location    \"%s\";\n    object      %s;\n}\n"
             % (cls, t, name), "dimensions      [m/s];\n", f"internalField   {lst(internal, typ)};\n", "boundaryField\n{"]
        for pn, (ptype, val) in per_patch.items():
            s.append(f"    {pn}\n    {{\n        type            {ptype};")
            if val is not None:
                s.append(f"        value           {lst(val, typ)};")
            s.append("    }")
        s.append("}\n\n// ****** //\n")
        (d / name).write_text("\n".join(s))
    U, P, PHI = {}, {}, {}
    for p in mesh.patches:
        pf = fields["patches"][p["name"]]
        if p["type"] == "empty":
            U[p["name"]] = P[p["name"]] = PHI[p["name"]] = ("empty", None)
            continue
        kind = {"inlet": "fixedValue", "outlet": "inletOutlet"}.get(p["name"], "noSlip")
        U[p["name"]] = (kind, None if kind == "noSlip" else pf["U"])
        P[p["name"]] = ("fixedValue", pf["p"]) if p["name"] == "outlet" else ("zeroGradient", None)
        PHI[p["name"]] = ("calculated", pf["phi"])
    write("U", "volVectorField", "vector", fields["U"], U)
    write("p", "volScalarField", "scalar", fields["p"], P)
    write("phi", "surfaceScalarField", "scalar", np.zeros(mesh.ninternal), PHI)


def test_foam_reader_reproduces_the_results(tmp_path):
    from cfd_translator.foam import read_fields
    mesh, patches, sel, plan, w = prepare(CHANNEL, tmp_path)
    c = tmp_path / "case"
    f_ref = solve_case(c)
    _write_foam_time(c, mesh, f_ref)
    f_read = read_fields(c, mesh, plan)
    assert f_read["time"] == "57"
    r1, _ = results(CHANNEL, mesh, plan, f_ref)
    r2, _ = results(CHANNEL, mesh, plan, f_read)
    for q in ("pressure_drop",):
        assert r2[q] == pytest.approx(r1[q], rel=1e-8)
    assert r2["forces"]["walls"]["fx"] == pytest.approx(r1["forces"]["walls"]["fx"], rel=1e-8)
    assert r2["flow"]["inlet"]["Q"] == pytest.approx(r1["flow"]["inlet"]["Q"], rel=1e-8)


def test_parse_log_and_checkmesh():
    log = """Starting time loop

Time = 1s

smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.05, No Iterations 2
smoothSolver:  Solving for Uy, Initial residual = 1, Final residual = 0.04, No Iterations 2
GAMG:  Solving for p, Initial residual = 1, Final residual = 0.03, No Iterations 5
time step continuity errors : sum local = 0.01, global = 1e-05, cumulative = 1e-05
ExecutionTime = 0.1 s  ClockTime = 0 s

Time = 2s

smoothSolver:  Solving for Ux, Initial residual = 0.5, Final residual = 0.05, No Iterations 2
smoothSolver:  Solving for Uy, Initial residual = 0.4, Final residual = 0.04, No Iterations 2
GAMG:  Solving for p, Initial residual = 1e-6, Final residual = 1e-8, No Iterations 5
GAMG:  Solving for p, Initial residual = 1e-7, Final residual = 1e-9, No Iterations 5

SIMPLE solution converged in 2 iterations

End
"""
    r = foamio.parse_log(log)
    assert r["iterations"] == 2 and r["converged"] and r["ended"]
    assert r["residuals"] == {"Ux": 0.5, "Uy": 0.4, "p": 1e-6}
    assert r["continuity"]["global"] == 1e-5
    bad = foamio.parse_log("Time = 1\n\n--> FOAM FATAL ERROR: (openfoam-14)\nUnknown patchField type nope\n\n    From x\n")
    assert "fatal" in bad and "nope" in bad["fatal"]
    cm = foamio.parse_checkmesh("    cells:            200\n    Mesh non-orthogonality Max: 12.5 average: 3.1\n"
                                "    Max skewness = 0.42 OK.\n\nMesh OK.\n")
    assert cm["ok"] and cm["max_non_orthogonality"] == 12.5 and cm["cells"] == 200


def test_run_study_json_and_expect(tmp_path):
    st = dict(CHANNEL, expect=[{"quantity": "flow.inlet.Q", "value": 1e-5, "tol": 1e-9},
                               {"quantity": "forces.walls.fx", "value": 1.2e-5, "tol": 0.02},
                               {"quantity": "nonsense.path", "value": 1}])
    r = run_study("ch", st, tmp_path / "ch", "ref")
    assert r["ok"]
    assert [e["passed"] for e in r["expect"]] == [True, True, False]
    assert json.loads((tmp_path / "ch" / "ch.json").read_text())["results"]["flow"]["inlet"]["Q"] == pytest.approx(1e-5)
    assert Path(r["files"]["archive"]).is_file()


# ---- need Gmsh / OpenFOAM ---------------------------------------------------------------------
@gmsh_only
def test_gmsh_swept_pipe_and_external_cylinder(tmp_path):
    from cfd_translator.mesher import mesh_study
    pipe = {"geometry": {"cylinder": [5, 50]}, "fluid": "water",
            "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": 0.1}, {"kind": "outlet", "faces": {"nz": 1}}],
            "mesh": {"size": 1.0, "extrude": 10}}
    m = mesh_study(pipe, tmp_path)
    q = m.quality()
    assert q["ok"] and q["volume_mm3"] == pytest.approx(math.pi * 25 * 50, rel=0.01)
    g = mesh_groups(m, False)
    assert g.select({"cyl_r": 5}) and g.select({"nz": 1})
    ext = {"geometry": {"cylinder": [50, 10]}, "domain": {"kind": "external", "box": [-200, -200, 0, 600, 210, 10]},
           "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
           "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 0.2, "profile": "parabolic"},
                   {"kind": "outlet", "faces": {"box": "xmax"}}, {"kind": "wall", "name": "cyl", "faces": {"cyl_r": 50}}],
           "mesh": {"size": 20, "layers": {"n": 4, "first": 1.0, "growth": 1.2}}}
    m = mesh_study(ext, tmp_path)
    q = m.quality()
    assert q["ok"] and q["volume_mm3"] == pytest.approx((800 * 410 - math.pi * 2500) * 10, rel=0.01)
    patches, sel, w = assign_bcs(mesh_groups(m, True), ext, m.info["empty_tags"])
    assert [p["name"] for p in patches] == ["inlet", "outlet", "cyl", "walls", "frontAndBack"]
    assert sel["cyl"][0]["side"] == "outer"


@gmsh_only
def test_gmsh_tets_box(tmp_path):
    from cfd_translator.mesher import mesh_study
    st = {"geometry": {"box": [20, 10, 10]}, "fluid": "air",
          "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 1}, {"kind": "outlet", "faces": {"nx": 1}}],
          "mesh": {"size": 2}}
    m = mesh_study(st, tmp_path)
    assert m.quality()["ok"] and m.quality()["volume_mm3"] == pytest.approx(2000)


@foam_only
def test_openfoam_couette_and_channel(tmp_path):
    r = bench.run("foam", ["couette_2d", "channel_2d"], tmp_path)
    assert r["couette_2d"]["passed"], r["couette_2d"]
    assert r["channel_2d"]["passed"], r["channel_2d"]


def test_cylinder_fit_on_coarse_short_triangulation():
    """A Gmsh-like surface mesh of a short cylinder side (R 50, h 10, 12 segments, staggered rims):
    facet normals tilt ~20 degrees, which the first fit rejected (the carbonio failure)."""
    R, h, n = 50.0, 10.0, 12
    a0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a1 = a0 + np.pi / n
    pts = np.vstack([np.column_stack([R * np.cos(a0), R * np.sin(a0), np.zeros(n)]),
                     np.column_stack([R * np.cos(a1), R * np.sin(a1), np.full(n, h)])])
    tris = []
    for j in range(n):
        tris.append([j, (j + 1) % n, n + j])
        tris.append([(j + 1) % n, n + (j + 1) % n, n + j])
    rings = np.pad(np.array(tris), ((0, 0), (0, 1)), constant_values=-1)
    g = Groups(pts, rings, np.zeros(len(rings), int), {0: {"on": "part"}}, external=False)
    assert g.groups[0].radius == pytest.approx(R, rel=1e-6)
    assert g.select({"cyl_r": 50, "side": "outer"}) == [0]


# ---- swirl, rotation, rotating zones, verdict, progress (2026.10.16.6) --------------------------
def test_swirl_inlet_keeps_flow_and_adds_tangential(tmp_path):
    st = {"geometry": {"box": [40, 40, 100]}, "fluid": "water",
          "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "velocity": 0.5,
                   "swirl": {"axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 60}},
                  {"kind": "outlet", "faces": {"nz": 1}}],
          "mesh": {"cells": [8, 8, 4]}}
    assert validate_study(st)["ok"]
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    U = foamio.read_foam((tmp_path / "case" / "0" / "U").read_text())["boundaryField"]["inlet"]["value"]
    f = mesh.patch_faces("inlet")
    C = mesh.geometry()["Cf"][f] * 1e-3
    assert np.allclose(U[:, 2], 0.5)                                       # normal part untouched
    assert np.allclose(U[:, 0], -2 * np.pi * C[:, 1]) and np.allclose(U[:, 1], 2 * np.pi * C[:, 0])   # omega x r
    assert plan["inlets"]["inlet"]["Q"] == pytest.approx(0.5 * 0.04 * 0.04)


def test_rotating_zone_files_and_rotating_wall(tmp_path):
    st = {"geometry": {"box": [100, 100, 10]}, "2d": True, "fluid": {"rho": 1000, "mu": 0.01},
          "rotating": [{"name": "rotor", "axis": [0, 0, 1], "center": [0, 0, 0], "rpm": 30, "radius": 30}],
          "bcs": [{"kind": "wall", "name": "spin", "faces": {"ny": 1}, "rotation": {"axis": [0, 0, 1], "rpm": 30}}],
          "mesh": {"cells": [10, 10, 1]}}
    assert validate_study(st)["ok"], validate_study(st)
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    c = tmp_path / "case"
    mrf = foamio.read_foam((c / "constant" / "MRFProperties").read_text())
    assert mrf["rotor"]["cellZone"] == "rotor" and mrf["rotor"]["omega"] == pytest.approx(np.pi)
    assert set(mrf["rotor"]) == {"cellZone", "origin", "axis", "omega"}
    zones = (c / "constant" / "polyMesh" / "cellZones").read_text()
    n = plan["zones"]["rotor"]["cells"]
    assert f"cellLabels List<label> {n}" in zones and 0 < n < 100
    U = np.atleast_2d(foamio.read_foam((c / "0" / "U").read_text())["boundaryField"]["spin"]["value"])
    # a rotating wall moves rigidly: omega x r = (-omega y, omega x, 0) on the lid (y = +50 mm)
    C = mesh.geometry()["Cf"][mesh.patch_faces("spin")] * 1e-3
    assert np.allclose(U[:, 0], -np.pi * C[:, 1]) and np.allclose(U[:, 1], np.pi * C[:, 0])
    with pytest.raises(Exception, match="MRF"):
        solve_case(c)
    bad = dict(st, bcs=[{"kind": "wall", "faces": {"ny": 1}, "rotation": "nope"}])
    assert any("not a rotating zone" in e for e in validate_study(bad)["errors"])
    # a wall rotating with the zone must lie in it
    out = dict(st, bcs=[{"kind": "wall", "faces": {"ny": 1}, "rotation": "rotor"}])
    with pytest.raises(Exception, match="outside the zone"):
        prepare(out, tmp_path / "o")
    inside = dict(st, rotating=[dict(st["rotating"][0], radius=500)],
                  bcs=[{"kind": "wall", "name": "spin", "faces": {"ny": 1}, "rotation": "rotor"}])
    prepare(inside, tmp_path / "i")
    Ui = foamio.read_foam((tmp_path / "i" / "case" / "0" / "U").read_text())["boundaryField"]["spin"]
    assert Ui["type"] == "MRFnoSlip"


def test_verdict_and_head_loss(tmp_path):
    from cfd_translator.runner import _verdict
    ok, why = _verdict({"solve": {"converged": False}}, {"checks": {"momentum_balance": {"error": 0.08}}})
    assert not ok and any("converge" in w for w in why) and any("momentum" in w for w in why)
    r = run_study("ch", CHANNEL, tmp_path / "ch", "ref")
    assert r["reliable"] is True and "unreliable_because" not in r
    R = r["results"]
    assert R["head_loss_m"] == pytest.approx(R["total_pressure_drop"] / (1.0 * 9.80665), rel=1e-5)


def test_progress_eta(tmp_path):
    from cfd_translator.api import _progress
    d = tmp_path / "s"
    (d / "case").mkdir(parents=True)
    assert _progress(d, {})["stage"].startswith("meshing")
    (d / "case" / "log.checkMesh").write_text("")
    (d / "case" / "log.foamRun").write_text("".join(f"Time = {i}s\n\n" for i in range(1, 101)))
    p = _progress(d, {"solve": {"iterations": 1000}})
    assert p["iteration"] == 100 and p["max_iterations"] == 1000 and p["eta_min_max"] >= 0


# ---- images (2026.10.17.1) ------------------------------------------------------------------------
def test_images_in_a_run_and_rerendered_from_the_saved_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    st = dict(CHANNEL, images=True)
    r = run_study("ch", st, tmp_path / "req" / "ch", "ref")
    assert r["ok"] and set(r["images"]) == {"velocity", "pressure"}, r.get("warnings")
    for v in r["images"].values():
        assert Path(v["path"]).stat().st_size > 10000
    more = api.render_images(str(tmp_path / "req" / "ch"),
                             [{"name": "zoom", "slice": {"axis": "z", "at": 5}, "field": "ux", "streamlines": True,
                               "window": [-500, -400, -50, 50]}])
    assert more["warnings"] == [] and Path(more["images"]["zoom"]["path"]).is_file()
    assert api.image_path(more["images"]["zoom"]["path"]).suffix == ".png"
    assert "errors" in api.render_images(str(tmp_path / "req" / "ch"), [{"slice": {"axis": "x", "at": 0}}])   # 2d: z only
    with pytest.raises(ValueError):
        api.image_path(str(tmp_path / "req" / "ch" / "ch.json"))


def test_slice_masks_solid_parts(tmp_path):
    """A ring mesh around a 5 mm cylinder: the slice's triangulation must leave the cylinder empty."""
    import matplotlib.tri as mtri
    from cfd_translator.render import _cell_size
    nr, nt, a, R = 20, 64, 5.0, 40.0
    pts = np.array([[a * (R / a) ** (i / nr) * math.cos(2 * math.pi * j / nt), a * (R / a) ** (i / nr) * math.sin(2 * math.pi * j / nt)]
                    for i in range(nr + 1) for j in range(nt)])
    q = [[i * nt + j, (i + 1) * nt + j, (i + 1) * nt + (j + 1) % nt, i * nt + (j + 1) % nt] for i in range(nr) for j in range(nt)]
    e = {(j, (j + 1) % nt): 1 for j in range(nt)} | {(nr * nt + j, nr * nt + (j + 1) % nt): 2 for j in range(nt)}
    m = extrude(pts, [], q, 0, 10, 1, 3, 4, e)
    C = m.geometry()["C"][:, :2]
    h = _cell_size(m, True)
    tri = mtri.Triangulation(C[:, 0], C[:, 1])
    E = C[tri.triangles]
    L = np.max(np.linalg.norm(E - np.roll(E, 1, axis=1), axis=2), axis=1)
    tri.set_mask(L > 2.0 * h[tri.triangles].max(axis=1))
    finder = tri.get_trifinder()
    assert finder(0.0, 0.0) == -1 and finder(1.0, 1.0) == -1          # inside the cylinder: no triangle
    assert finder(20.0, 0.0) >= 0                                     # in the fluid: covered


# ---- energy: compressible gas, heat transfer, buoyancy (2026.10.18.1) -----------------------------
SLAB = {"geometry": {"box": [20, 10, 1]}, "2d": True, "fluid": "air", "energy": True,
        "bcs": [{"kind": "wall", "name": "hot", "faces": {"ny": -1}, "T": 60},
                {"kind": "wall", "name": "cold", "faces": {"ny": 1}, "T": 20}],
        "mesh": {"cells": [4, 20, 1]}, "probes": [{"name": "mid", "at": [0, 0, 0.5]}]}


def test_energy_validation():
    from cfd_translator.format import thermo_props
    assert validate_study(SLAB)["ok"]
    e = validate_study(dict(SLAB, energy=False))["errors"]
    assert any("need energy" in x for x in e)
    assert not validate_study(dict(SLAB, gravity=[0, -9.81]))["ok"]
    bad = dict(SLAB, bcs=[{"kind": "wall", "faces": {"ny": -1}, "T": 60, "q": 5}])
    assert any("one of T, q, power" in x for x in validate_study(bad)["errors"])
    tp = dict(SLAB, fluid="water", bcs=[{"kind": "inlet", "faces": {"nx": -1}, "total_pressure": 100}, {"kind": "outlet", "faces": {"nx": 1}}])
    assert any("for gases" in x for x in validate_study(tp)["errors"])
    a = thermo_props(SLAB)
    assert a["kind"] == "gas" and a["rho"] == pytest.approx(1.204, rel=2e-3) and a["gamma"] == pytest.approx(1.4, rel=1e-3)
    assert thermo_props({"fluid": {"name": "water", "mu": 0.01}})["kind"] == "liquid"


def test_energy_case_files(tmp_path):
    cav = {"geometry": {"box": [21, 21, 2]}, "2d": True, "fluid": "air", "energy": True, "gravity": [0, -9.81, 0],
           "bcs": [{"kind": "wall", "name": "hot", "faces": {"nx": -1}, "T": 25}, {"kind": "wall", "name": "cold", "faces": {"nx": 1}, "q": -5}],
           "mesh": {"cells": [8, 8, 1]}}
    mesh, patches, sel, plan, w = prepare(cav, tmp_path / "cav")
    c = tmp_path / "cav" / "case"
    pr = foamio.read_foam((c / "0" / "p_rgh").read_text())["boundaryField"]
    assert pr["hot"]["type"] == "fixedFluxPressure"
    assert foamio.read_foam((c / "0" / "p").read_text())["boundaryField"]["hot"]["type"] == "calculated"
    T = foamio.read_foam((c / "0" / "T").read_text())["boundaryField"]
    assert T["hot"]["value"] == pytest.approx(298.15) and T["cold"]["type"] == "externalTemperature" and T["cold"]["q"] == "constant -5"
    assert (c / "constant" / "g").exists() and (c / "constant" / "pRef").exists()
    pp = foamio.read_foam((c / "constant" / "physicalProperties").read_text())
    assert pp["thermoType"]["type"] == "heRhoThermo" and pp["thermoType"]["equationOfState"] == "perfectGas"
    fv = foamio.read_foam((c / "system" / "fvSolution").read_text())
    assert "p_rgh" in fv["PIMPLE"]["residualControl"]
    cd = foamio.read_foam((c / "system" / "controlDict").read_text())
    assert cd["solver"] == "fluid" and "wallHeatFlux" in cd["functions"]
    ven = {"geometry": {"box": [300, 20, 2]}, "2d": True, "fluid": "air", "energy": True,
           "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "total_pressure": 10000}, {"kind": "outlet", "faces": {"nx": 1}},
                   {"kind": "slip", "faces": {"all": True}}], "mesh": {"cells": [30, 4, 1]}}
    mesh, patches, sel, plan, w = prepare(ven, tmp_path / "ven")
    c = tmp_path / "ven" / "case"
    p = foamio.read_foam((c / "0" / "p").read_text())["boundaryField"]
    assert p["inlet"]["type"] == "uniformTotalPressure" and "400 111325" in p["inlet"]["p0"]
    fv = foamio.read_foam((c / "system" / "fvSolution").read_text())
    assert fv["PIMPLE"]["transonic"] == "yes" and float(fv["relaxationFactors"]["fields"]["p"]) == 1   # total-pressure inlet
    pin = foamio.read_foam((c / "0" / "p").read_text())["boundaryField"]["inlet"]
    assert pin["type"] == "uniformTotalPressure" and "400 111325" in pin["p0"]                # ramp by default
    assert "pMinFactor" not in fv["PIMPLE"]                       # OpenFOAM 14: an error there
    assert "initial_U" not in plan                                 # from rest by default
    assert fv["PIMPLE"]["consistent"] == "yes"
    lim = foamio.read_foam((c / "system" / "fvConstraints").read_text())["limitp"]
    assert lim["type"] == "limitPressure" and float(lim["minFactor"]) == 0.5 and float(lim["maxFactor"]) == 2
    mesh, patches, sel, plan, w = prepare(dict(ven, solve={"transonic": False, "start": "estimate", "ramp": 0}), tmp_path / "ven2")
    assert foamio.read_foam((tmp_path / "ven2" / "case" / "system" / "fvSolution").read_text())["PIMPLE"]["transonic"] == "no"
    assert plan["initial_U"][0] == pytest.approx(0.5 * plan["inlets"]["inlet"]["U_estimate"])
    assert foamio.read_foam((tmp_path / "ven2" / "case" / "0" / "p").read_text())["boundaryField"]["inlet"]["type"] == "totalPressure"
    assert plan["energy"]["mach_estimate"] == pytest.approx(0.369, rel=0.01)
    assert not (c / "0" / "p_rgh").exists()
    with pytest.raises(Exception, match="energy"):
        solve_case(c)


def _foam_energy_fixture(case: Path, mesh, plan):
    """The exact conduction-slab solution (linear T, no flow) written as OpenFOAM writes fields."""
    def lst(a, typ):
        a = np.asarray(a, float)
        items = [f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in a] if typ == "vector" else [f"{x:.12g}" for x in a]
        return f"nonuniform List<{typ}> \n{len(items)}\n(\n" + "\n".join(items) + "\n)\n"

    def write(name, typ, internal, per):
        s = ["FoamFile\n{\n    format ascii;\n    object %s;\n}\n" % name, "dimensions [0 0 0 1 0 0 0];\n",
             f"internalField   {lst(internal, typ)};\n", "boundaryField\n{"]
        for pn, (t, v) in per.items():
            s.append(f"    {pn}\n    {{\n        type {t};" + (f"\n        value {lst(v, typ)};" if v is not None else "") + "\n    }")
        (d / name).write_text("\n".join(s) + "\n}\n")
    d = case / "100"
    d.mkdir()
    en = plan["energy"]
    g = mesh.geometry()
    Tc = 333.15 - 40 * (g["C"][:, 1] * 1e-3 + 0.005) / 0.01
    per = {k: {} for k in ("U", "p", "phi", "T", "rho", "wallHeatFlux")}
    for p in mesh.patches:
        f = np.arange(p["start"], p["start"] + p["n"])
        if p["type"] == "empty":
            for k in per:
                per[k][p["name"]] = ("empty", None)
            continue
        Tf = 333.15 - 40 * (g["Cf"][f, 1] * 1e-3 + 0.005) / 0.01
        q = {"hot": en["k"] * 4000, "cold": -en["k"] * 4000}.get(p["name"], 0.0)
        per["U"][p["name"]] = ("noSlip", None)
        per["p"][p["name"]] = ("zeroGradient", None)
        per["phi"][p["name"]] = ("calculated", np.zeros(len(f)))
        per["T"][p["name"]] = ("calculated", Tf)
        per["rho"][p["name"]] = ("calculated", en["p_op"] / (en["R"] * Tf))
        per["wallHeatFlux"][p["name"]] = ("calculated", np.full(len(f), q))
    nc = mesh.ncells
    write("U", "vector", np.zeros((nc, 3)), per["U"])
    write("p", "scalar", np.full(nc, en["p_op"]), per["p"])
    write("phi", "scalar", np.zeros(mesh.ninternal), per["phi"])
    write("T", "scalar", Tc, per["T"])
    write("rho", "scalar", en["p_op"] / (en["R"] * Tc), per["rho"])
    write("wallHeatFlux", "scalar", np.zeros(nc), per["wallHeatFlux"])


def test_energy_results_from_openfoam_files(tmp_path):
    from cfd_translator.foam import read_fields
    from cfd_translator.render import render
    mesh, patches, sel, plan, w = prepare(SLAB, tmp_path)
    _foam_energy_fixture(tmp_path / "case", mesh, plan)
    fields = read_fields(tmp_path / "case", mesh, plan)
    res, warns = results(SLAB, mesh, plan, fields)
    k = plan["energy"]["k"]
    assert res["heat"]["hot"]["Q_W"] == pytest.approx(k * 4000 * 0.02 * 0.001, rel=1e-5)
    assert res["heat"]["cold"]["Q_W"] == pytest.approx(-k * 4000 * 0.02 * 0.001, rel=1e-5)
    assert res["heat"]["hot"]["T_mean_C"] == pytest.approx(60) and res["probes"]["mid"]["T"] == pytest.approx(40, abs=1e-6)
    assert res["checks"]["energy_balance"]["error"] < 1e-9 and warns == []
    imgs, w2 = render(SLAB, mesh, plan, fields, tmp_path / "img", [{"name": "T", "slice": {"axis": "z", "at": 0.5}, "field": "T"},
                                                                    {"name": "q", "surface": "walls", "field": "q"}])
    assert w2 == [] and imgs["T"]["min"] == pytest.approx(20) and imgs["T"]["max"] == pytest.approx(60)
    assert imgs["q"]["max"] == pytest.approx(k * 4000, rel=1e-6)


def test_energy_benchmarks_are_valid_studies():
    C = bench._cases()
    for n in ("conduction_slab_2d", "heated_channel_2d", "natural_cavity_ra1e4", "venturi_subsonic", "pipe_air_low_mach"):
        assert validate_study(C[n]["study"])["ok"], n


def test_zero_fields_cover_exact_couette_and_isothermal_runs():
    from cfd_translator.runner import zero_fields
    U = np.zeros((10, 3))
    U[:, 0] = np.linspace(0, 0.01, 10)
    U[3, 1] = 1e-14                                                  # round-off
    f = {"U": U, "p": np.full(10, 1e-19)}
    assert zero_fields(f, {"rho": 1.0}) == ["Uy", "Uz", "p", "p_rgh"]          # mochi's Couette: Uy and p
    f["p"] = np.linspace(0, 1e-3, 10)                                             # a real pressure drop
    assert zero_fields(f, {"rho": 1.0}) == ["Uy", "Uz"]
    en = {"rho": 1.2, "T_op": 293.15}
    assert "h" in zero_fields({**f, "T": np.full(10, 293.15)}, {"energy": en})
    assert "h" not in zero_fields({**f, "T": np.linspace(293, 300, 10)}, {"energy": en})


def test_closed_energy_domain_uses_the_p_rgh_formulation(tmp_path):
    mesh, patches, sel, plan, w = prepare(SLAB, tmp_path)
    c = tmp_path / "case"
    assert plan["energy"]["closed"] and plan["energy"]["buoyancy"] and plan["energy"]["gravity"] == [0.0, 0.0, 0.0]
    assert (c / "0" / "p_rgh").exists() and (c / "constant" / "g").exists()
    assert foamio.read_foam((c / "constant" / "physicalProperties").read_text())["thermoType"]["type"] == "heRhoThermo"
    assert "p_rgh" in foamio.read_foam((c / "system" / "fvSolution").read_text())["PIMPLE"]["residualControl"]
    pp = foamio.read_foam((c / "constant" / "physicalProperties").read_text())
    # no gravity given: constant density (perfectGas diverged there on mochi, 2026.10.18.5)
    assert pp["thermoType"]["equationOfState"] == "rhoConst"
    assert float(pp["mixture"]["equationOfState"]["rho"]) == pytest.approx(plan["energy"]["rho"])
    assert plan["relax"]["other"] == 0.9 and plan["energy"]["density"].startswith("constant")


def test_venturi_outline_is_closed_and_small_velocity_components_count_as_zero():
    from cfd_translator.runner import zero_fields
    o = bench._cases()["venturi_subsonic"]["study"]["geometry"]["goal"]["features"][0]["params"]["outline"]
    assert o["segs"][-1][1:] == o["start"]                      # b3dt: the last point is the start
    U = np.zeros((10, 3))
    U[:, 0] = 1.0
    U[2, 1] = 5e-5                                              # heated channel: Uy round-off, residual stalls
    assert zero_fields({"U": U, "p": np.linspace(0, 1, 10)}, {"rho": 1.0}) == ["Uy", "Uz"]
    U[2, 1] = 1e-3                                              # a real cross flow
    assert zero_fields({"U": U, "p": np.linspace(0, 1, 10)}, {"rho": 1.0}) == ["Uz"]


def test_isothermal_energy_run_ignores_h_and_caps_gamg(tmp_path):
    from cfd_translator.runner import zero_fields
    pipe = {"geometry": {"box": [100, 10, 1]}, "2d": True, "fluid": "air", "energy": True,
            "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.5}, {"kind": "outlet", "faces": {"nx": 1}}],
            "mesh": {"cells": [20, 4, 1]}}
    mesh, patches, sel, plan, w = prepare(pipe, tmp_path / "a")
    fv = foamio.read_foam((tmp_path / "a" / "case" / "system" / "fvSolution").read_text())
    assert plan["energy"]["isothermal"] and "h" not in fv["PIMPLE"]["residualControl"]
    assert int(fv["solvers"]["p"]["maxIter"]) == 50
    U = np.zeros((10, 3))
    U[:, 0] = 1.0
    T = np.full(10, 293.15)
    T[0] += 5e-4                                    # u^2 / cp = 1e-3 K: kinetic energy, not heat
    assert "h" in zero_fields({"U": U, "p": np.linspace(1e5, 1e5 + 1, 10), "T": T}, plan)
    heated = dict(pipe, bcs=pipe["bcs"] + [{"kind": "wall", "faces": {"ny": 1}, "q": 100}])
    mesh, patches, sel, plan, w = prepare(heated, tmp_path / "b")
    fv = foamio.read_foam((tmp_path / "b" / "case" / "system" / "fvSolution").read_text())
    assert not plan["energy"]["isothermal"] and "h" in fv["PIMPLE"]["residualControl"]
    assert "h" not in zero_fields({"U": U, "p": np.linspace(1e5, 1e5 + 1, 10), "T": T}, plan)


def test_small_velocity_components_are_judged_against_the_peak_speed():
    from cfd_translator.runner import velocity_scaled
    U = np.zeros((10, 3))
    U[:, 0] = 7.5e-4
    U[3, 1] = 1e-6                                   # heated channel: |Uy| / |U| = 1.3e-3, residual 7e-3
    r = velocity_scaled({"Ux": 4e-7, "Uy": 7e-3, "h": 2e-6, "p": 1e-6}, {"U": U})
    assert r["Ux"] == pytest.approx(4e-7) and r["Uy"] == pytest.approx(7e-3 * 1e-7 / 7.5e-4, rel=1e-3) and r["h"] == 2e-6
    assert max(r.values()) < 1e-5


def test_total_pressure_ramp_and_start_from_rest(tmp_path):
    ven = {"geometry": {"box": [300, 20, 2]}, "2d": True, "fluid": "air", "energy": True,
           "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "total_pressure": 10000}, {"kind": "outlet", "faces": {"nx": 1}},
                   {"kind": "slip", "faces": {"all": True}}], "mesh": {"cells": [30, 4, 1]},
           "solve": {"ramp": 200, "start": "rest"}}
    assert validate_study(ven)["ok"]
    assert not validate_study(dict(ven, solve={"start": "moving"}))["ok"]
    mesh, patches, sel, plan, w = prepare(ven, tmp_path)
    c = tmp_path / "case"
    p = foamio.read_foam((c / "0" / "p").read_text())["boundaryField"]["inlet"]
    assert p["type"] == "uniformTotalPressure" and "200 111325" in p["p0"] and "0 101325" in p["p0"]
    assert "initial_U" not in plan
    cd = foamio.read_foam((c / "system" / "controlDict").read_text())
    assert "MachNo" not in cd["functions"]


# ---- 2026.10.21.1: y+ from the wall shear, summary mode, clim and Cp ----------------------------
def test_y_plus_from_wall_shear_on_moving_and_fixed_walls(tmp_path):
    """Couette: tau = mu U / H on both walls, u_tau = sqrt(tau / rho), y = half a cell: y+ = 0.05 on the
    moving lid as on the fixed bottom (the velocity relative to the wall, not the lid's own speed)."""
    st = bench._cases()["couette_2d"]["study"]
    r = run_study("c", st, tmp_path / "c", "ref")
    yp = r["results"]["y_plus"]
    u_tau = math.sqrt(1e-3 * 0.01 / 0.1 / 1.0)
    expect = u_tau * 0.005 / 1e-3
    for n in ("lid", "bottom"):
        assert yp[n]["mean"] == pytest.approx(expect, rel=1e-6) and yp[n]["part"]
    assert yp["lid"].get("moving") and not yp["bottom"].get("moving")
    assert "y_plus_part_mean" not in r["results"]                            # laminar: not judged


@gmsh_only
def test_external_y_plus_marks_the_part_walls(tmp_path):
    """The domain's sides (walls) are not the part: the y+ warning (turbulent runs) uses only part walls. Results
    from made-up fields on the Gmsh mesh (the reference solver runs structured boxes only)."""
    ext = {"geometry": {"cylinder": [50, 10]}, "domain": {"kind": "external", "box": [-200, -200, 0, 600, 210, 10]},
           "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3}, "model": "kOmegaSST",
           "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 0.2},
                   {"kind": "outlet", "faces": {"box": "xmax"}}, {"kind": "wall", "name": "cyl", "faces": {"cyl_r": 50}}],
           "mesh": {"size": 20}}
    mesh, patches, sel, plan, w = prepare(ext, tmp_path)
    g = mesh.geometry()
    U = np.array([0.2, 0.0, 0.0])
    fields = {"U": np.tile(U, (mesh.ncells, 1)), "p": np.zeros(mesh.ncells), "patches": {}, "source": "test"}
    for P in mesh.patches:
        f = np.arange(P["start"], P["start"] + P["n"])
        wall = P["type"] == "wall"
        Ub = np.zeros((P["n"], 3)) if wall or P["type"] == "empty" else np.tile(U, (P["n"], 1))
        fields["patches"][P["name"]] = {
            "U": Ub, "p": np.zeros(P["n"]), "phi": (Ub @ np.zeros(3)) if wall else g["Sf"][f] @ U * 1e-6,
            "wss": np.tile([-1e-3, 0.0, 0.0], (P["n"], 1)) if wall else None, "nut": None, "yPlus": None}
    res, warns = results(ext, mesh, plan, fields)
    yp = res["y_plus"]
    assert yp["cyl"]["part"] and not yp["walls"]["part"]
    assert res["y_plus_part_mean"] == pytest.approx(yp["cyl"]["mean"])


def test_summary_mode_drops_the_selection_dump(tmp_path):
    r = run_study("ch", CHANNEL, tmp_path / "ch", "ref")
    s = api.summarize(r)
    assert s["detail"] == "summary" and set(s["selection"]) == set(r["selection"])
    assert s["selection"]["walls"] == {"faces": 2, "area_mm2": pytest.approx(20000)}
    assert s["results"]["flow"] == r["results"]["flow"] and s["ok"] == r["ok"]
    assert len(json.dumps(s)) < len(json.dumps(r))
    assert api.summarize({"error": "x"}) == {"error": "x"}
    res = {"done": True, "studies": {"ch": r}}
    assert api._detail(res, "summary")["studies"]["ch"]["detail"] == "summary"
    assert api._detail(res, "full") is res


def test_image_clim_and_cp(tmp_path, monkeypatch):
    from cfd_translator.format import image_errors
    assert image_errors([{"slice": {"axis": "z", "at": 0}, "field": "cp"}], True, (), False)
    assert not image_errors([{"slice": {"axis": "z", "at": 0}, "field": "cp", "clim": [-1, 1]}], True, (), True)
    assert image_errors([{"surface": "part", "clim": [1, 0]}]) and image_errors([{"surface": "part", "clim": "wide"}])
    assert not image_errors([{"surface": "part", "field": "cp", "clim": "full"}])
    monkeypatch.setenv("CFDT_OUT", str(tmp_path))
    st = dict(CHANNEL, reference={"velocity": 0.01})
    r = run_study("ch", st, tmp_path / "req" / "ch", "ref")
    assert r["ok"]
    out = api.render_images(str(tmp_path / "req" / "ch"),
                            [{"name": "cp", "slice": {"axis": "z", "at": 5}, "field": "cp"},
                             {"name": "cpc", "slice": {"axis": "z", "at": 5}, "field": "cp", "clim": [-1, 1]},
                             {"name": "rob", "slice": {"axis": "z", "at": 5}, "field": "p", "clim": "robust"}])
    assert out["warnings"] == [], out
    q = 0.5 * 1.0 * 0.01 ** 2
    p = r["results"]["pressure"]
    assert out["images"]["cp"]["max"] == pytest.approx(p["inlet"]["p_mean"] / q, rel=0.2)
    assert out["images"]["cpc"]["clim"] == [-1, 1] and out["images"]["cp"]["clim"][0] == out["images"]["cp"]["min"]
    lo, hi = out["images"]["rob"]["clim"]
    assert out["images"]["rob"]["min"] <= lo < hi <= out["images"]["rob"]["max"]
