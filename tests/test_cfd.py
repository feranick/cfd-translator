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
