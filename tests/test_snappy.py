"""snappyHexMesh path (snappy.py), polyMesh reading (ASCII and binary), refinement shapes, layer plan, quality
schemes. OpenFOAM and Gmsh are not needed except where marked; the benchmarks sphere_snappy_re100,
sphere_half_snappy and pipe_snappy run the whole path on a machine with both."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cfd_translator import snappy
from cfd_translator.case import nonortho_schemes
from cfd_translator.executor import cell_estimate
from cfd_translator.foamio import parse_checkmesh, read_mesh_arrays
from cfd_translator.format import validate_study
from cfd_translator.polymesh import SCALE, box_mesh, face_geometry, from_foam

try:
    import gmsh  # noqa: F401
    HAVE_GMSH = True
except ImportError:
    HAVE_GMSH = False
gmsh_only = pytest.mark.skipif(not HAVE_GMSH, reason="needs Gmsh")

EXT = {"geometry": {"box": [100, 40, 30]}, "domain": {"kind": "external", "box": [-200, -100, 0, 500, 100, 150]},
       "fluid": "air", "model": "kOmegaSST", "reference": {"velocity": 30, "area": 1200, "length": 100},
       "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 30},
               {"kind": "outlet", "faces": {"box": "xmax"}},
               {"kind": "slip", "name": "far", "faces": {"box": ["ymin", "ymax", "zmax"]}},
               {"kind": "wall", "name": "ground", "faces": {"box": "zmin"}, "velocity": [30, 0, 0]},
               {"kind": "wall", "name": "body", "faces": {"all": True}}],
       "mesh": {"mesher": "snappy", "size": 40, "surface": [2.5, 5],
                "refine": [{"box": [-50, -40, 0, 300, 40, 60], "size": 10}, {"distance": 10, "size": 5},
                           {"distance": 30, "size": 10}],
                "layers": {"y_plus": 50, "growth": 1.2, "on": ["body", "ground"]}}}


# ---- polyMesh files ---------------------------------------------------------------------------------
def _patched_box():
    m = box_mesh(30, 20, 10, 3, 2, 2)
    m.set_patches([{"name": "a", "type": "patch", "tags": [0, 1]}, {"name": "w", "type": "wall", "tags": [2, 3, 4, 5]}])
    return m


def test_ascii_polymesh_round_trip(tmp_path):
    m = _patched_box()
    m.write(tmp_path)
    a = read_mesh_arrays(tmp_path / "constant" / "polyMesh")
    assert np.allclose(a["points"], m.points * SCALE) and a["ncells"] == m.ncells
    assert np.array_equal(a["owner"], m.owner) and np.array_equal(a["neighbour"], m.neighbour)
    assert np.array_equal(a["faces"][:, :4], m.faces)
    assert [(b["name"], b["n"], b["start"]) for b in a["boundary"]] == [(p["name"], p["n"], p["start"]) for p in m.patches]


def _write_binary(d: Path, m, label="<i4"):
    """polyMesh files as OpenFOAM writes them in binary: faces as faceCompactList (offsets, then labels)."""
    d.mkdir(parents=True, exist_ok=True)

    def hdr(cls, obj, note=""):
        n = f'    note        "{note}";\n' if note else ""
        return (f"FoamFile\n{{\n    format      binary;\n    class       {cls};\n{n}    location    \"constant/polyMesh\";\n"
                f"    object      {obj};\n}}\n// * * * //\n\n").encode()

    def blist(a):
        a = np.ascontiguousarray(a)
        return f"\n{len(a) if a.ndim == 1 else a.shape[0]}\n(".encode() + a.tobytes() + b")\n"
    (d / "points").write_bytes(hdr("vectorField", "points") + blist((m.points * SCALE).astype("<f8")))
    rows = [r[r >= 0] for r in m.faces]
    off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(label)
    (d / "faces").write_bytes(hdr("faceCompactList", "faces") + blist(off) + blist(np.concatenate(rows).astype(label)))
    note = f"nPoints:{len(m.points)} nCells:{m.ncells} nFaces:{m.nfaces} nInternalFaces:{m.ninternal}"
    (d / "owner").write_bytes(hdr("labelList", "owner", note) + blist(m.owner.astype(label)))
    (d / "neighbour").write_bytes(hdr("labelList", "neighbour", note) + blist(m.neighbour.astype(label)))
    m._write_boundary_and_zones(d)


@pytest.mark.parametrize("label", ["<i4", "<i8"])
def test_binary_polymesh_and_from_foam(tmp_path, label):
    m = _patched_box()
    d = tmp_path / "snappy" / "constant" / "polyMesh"
    _write_binary(d, m, label)
    f = from_foam(d)
    assert f.ncells == m.ncells and np.allclose(f.points, m.points)
    assert np.allclose(f.geometry()["V"], m.geometry()["V"])
    assert [p["name"] for p in f.info["foam_patches"]] == ["a", "w"]
    # patches in the file's order keep the faces where they are; write copies the files, rewrites boundary
    f.set_patches([{"name": "a", "type": "patch", "tags": [1]}, {"name": "w", "type": "wall", "tags": [2]}])
    assert f.patches[0]["start"] == m.patches[0]["start"] and f.patches[1]["n"] == m.patches[1]["n"]
    f.write(tmp_path / "case")
    out = tmp_path / "case" / "constant" / "polyMesh"
    assert (out / "faces").read_bytes() == (d / "faces").read_bytes()
    assert "type            wall;" in (out / "boundary").read_text()
    with pytest.raises(Exception):
        f.set_patches([{"name": "w", "type": "wall", "tags": [2]}, {"name": "a", "type": "patch", "tags": [1]}])


def test_face_geometry_of_polygons():
    n = 7
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(n)]) + [5.0, 1.0, 2.0]
    faces = np.array([list(range(n)) + [-1], [0, 1, 2, -1, -1, -1, -1, -1]])
    Sf, Cf = face_geometry(pts, faces)
    assert Sf[0] == pytest.approx([0, 0, 0.5 * n * math.sin(2 * math.pi / n)])
    assert Cf[0] == pytest.approx([5.0, 1.0, 2.0])
    S2, C2 = face_geometry(pts, faces, chunk=1)                          # chunked: the same
    assert np.allclose(S2, Sf) and np.allclose(C2, Cf)


def test_checkmesh_quality_numbers():
    log = ("    Mesh non-orthogonality Max: 63.2 average: 9.1\n"
           " ***Max skewness = 5.3, 12 highly skew faces detected which may impair the quality of the results\n"
           "    Min volume = 1.2e-09. Max volume = 0.004.  Total volume = 1.3.  Cell volumes OK.\n\nFailed 1 mesh checks.\n")
    c = parse_checkmesh(log)
    assert c["max_non_orthogonality"] == 63.2 and c["max_skewness"] == 5.3 and c["min_volume"] == 1.2e-09
    assert c["failed_checks"] == 1 and not c["ok"]


# ---- settings, dictionaries, logs -------------------------------------------------------------------
def test_levels_and_layers_from_a_target_y_plus():
    assert snappy.level_for(40, 5) == 3 and snappy.level_for(40, 6) == 3 and snappy.level_for(40, 40) == 0
    st = snappy.settings(EXT, True, EXT["domain"]["box"], ([-50, -20, 0], [50, 20, 30]))
    assert st["surface"] == (3, 4) and st["regions"][0]["level"] == 2 and st["cells"] == [18, 5, 4]
    assert st["distance"] == [(10.0, 3), (30.0, 2)]
    L = st["layers"]
    assert L["y_plus_target"] == 50 and L["y_plus_first_cell_estimate"] == pytest.approx(50, rel=1e-3)
    # n: the most layers whose stack fits in one of the finest surface cells
    assert L["total_mm"] <= st["surface_mm"][1] < L["total_mm"] + L["last_mm"] * L["growth"]
    assert L["stack_to_surface_cell"] <= 1
    # nearer distance bands are at least as fine as farther ones
    st2 = snappy.settings(dict(EXT, mesh=dict(EXT["mesh"], refine=[{"distance": 5, "size": 20}, {"distance": 50, "size": 5}])),
                          True, EXT["domain"]["box"])
    assert st2["distance"] == [(5.0, 3), (50.0, 3)]


def test_snappy_dictionaries_have_every_mandatory_keyword():
    st = snappy.settings(EXT, True, EXT["domain"]["box"], ([-50, -20, 0], [50, 20, 30]))
    zones = [{"name": "rad", "shape": "box", "box": [-60, -15, 5, -55, 15, 25], "level": 4}]
    txt = snappy.snappy_dict(st, [("r4", "body", "wall")], [-190.0, -95.0, 5.0], ["body", "part_r4", "ground"], zones,
                             st["quality"])
    for k in ("castellatedMesh", "snap", "addLayers", "geometry", "castellatedMeshControls", "snapControls",
              "addLayersControls", "meshQualityControls", "mergeTolerance", "maxLocalCells", "maxGlobalCells",
              "minRefinementCells", "nCellsBetweenLevels", "resolveFeatureAngle", "allowFreeStandingZoneFaces",
              "insidePoint", "nSmoothPatch", "tolerance", "nSolveIter", "nRelaxIter", "relativeSizes", "minThickness",
              "featureAngle", "nGrow", "maxFaceThicknessRatio", "nBufferCellsNoExtrude", "nLayerIter", "nSmoothSurfaceNormals",
              "minMedialAxisAngle", "nSmoothNormals", "maxThicknessToMedialRatio", "nSmoothThickness", "firstLayerThickness",
              "expansionRatio", "maxNonOrtho", "minVol", "minTetQuality", "maxConcave", "maxInternalSkewness",
              "maxBoundarySkewness", "minFaceWeight", "minVolRatio", "minTwist", "minDeterminant", "errorReduction",
              "nSmoothScale", "gapLevelIncrement", "faceZone rad", "cellZone rad", "mode inside", "mode distance",
              "r4 { name body; }", "patchInfo { type wall; }", "\"(body|part_r4|ground)\""):
        assert k in txt, k
    assert txt.count("{") == txt.count("}")
    assert "levels ((0.01 3) (0.03 2))" in txt and "type box;" in txt
    bm = snappy.block_mesh_dict(st, [("inlet", "patch", ["xmin"]), ("far", "patch", ["ymin", "ymax", "zmax"])], False)
    assert "(18 5 4)" in bm and "(0 4 7 3)" in bm and "defaultPatch" not in bm
    assert "defaultPatch" in snappy.block_mesh_dict(st, [], True)
    q2 = snappy.stricter(st["quality"])
    assert q2["maxNonOrtho"] == 55 and q2["maxInternalSkewness"] == 3


def test_snappy_log_layers_and_stages():
    log = """Refined mesh : cells:120000  faces:400000  points:150000
Marked for refinement due to close opposite surfaces       : 840 cells.
Snapped mesh : cells:118000  faces:390000  points:149000

patch      faces    layers avg thickness[m]
                           near-wall overall
-----      -----    ------ --------- -------
body       5000     5      0.0002    0.003

patch      faces    layers   overall thickness
                             [m]       [%]
-----      -----    ------   ---       ---
body       5000     4.6      0.0027    88.5
ground     9000     3.1      0.0019    61

Layer mesh : cells:160000  faces:520000  points:190000
Finished meshing with 3 illegal faces (concave, zero area or negative cell pyramid volume)
Finished meshing in = 81.5 s.
"""
    r = snappy.parse_log(log)
    assert r["cells"] == {"refined": 120000, "snapped": 118000, "layer": 160000}
    assert r["layers"][0] == {"patch": "body", "faces": 5000, "layers": 4.6, "thickness_mm": pytest.approx(2.7),
                              "thickness_pct": 88.5}
    assert len(r["layers"]) == 2 and r["illegal_faces"] == 3 and r["gap_cells"] == 840 and r["seconds"] == 81.5


def test_cell_zones_ascii_and_binary(tmp_path):
    d = tmp_path
    (d / "cellZones").write_text('FoamFile\n{\n    format ascii;\n    class cellZoneList;\n}\n\n2\n(\nrad\n{\n'
                                 '    cellLabels List<label> 3(4 5 9);\n}\nfan\n{\n    cellLabels List<label> \n12\n(\n'
                                 + "\n".join(map(str, range(12))) + "\n)\n;\n}\n)\n")
    z = snappy._read_cell_zones(d)
    assert z["rad"].tolist() == [4, 5, 9] and z["fan"].tolist() == list(range(12))
    raw = (b"FoamFile\n{\n    format binary;\n    class cellZoneList;\n}\n\n1\n(\nrad\n{\n    cellLabels List<label> \n3\n("
           + np.array([7, 8, 41], "<i4").tobytes() + b");\n}\n)\n")
    (d / "cellZones").write_bytes(raw)
    assert snappy._read_cell_zones(d)["rad"].tolist() == [7, 8, 41]


# ---- the part's surface ---------------------------------------------------------------------------------
def _cube(lo, hi, tag0=1):
    """A closed box surface, one tag per side, triangles oriented outward."""
    V, surf, _ = snappy.box_triangles(list(lo) + list(hi), 0)
    return V, [(tag0 + i, t) for i, (_, t) in enumerate(surf)]


def test_half_model_cut_at_the_symmetry_side():
    pts, surf = _cube([0, 0, 10], [40, 20, 30])            # its ymin face lies on the domain side y = 0
    box = [-100, 0, 0, 200, 100, 100]
    p2, s2, dropped, notes = snappy.cut_at_box(pts, surf, box, 2.0)
    assert dropped == {3: "ymin"} and notes
    assert 3 not in dict(s2)
    allp = p2[np.unique(np.concatenate([t.ravel() for _, t in s2]))]
    assert allp[:, 1].min() == pytest.approx(-2.0)          # the sides go on past the cut
    assert snappy.touching(p2, s2, box, dropped) == []
    # a body resting on the ground at an edge only (no face on it) is flagged
    assert snappy.touching(pts, surf, [-100, -50, 10, 200, 100, 100], {}) == ["zmin"]


def test_narrow_gaps_between_facing_walls():
    # two plates 3 mm apart (external flow between them)
    ps1, s1 = _cube([0, 0, 0], [50, 50, 2])
    ps2, s2 = _cube([0, 0, 5], [50, 50, 7], tag0=11)
    pts = np.concatenate([ps1, ps2])
    surf = s1 + [(t, tri + len(ps1)) for t, tri in s2]
    # refine the triangles so the plates have more than one centre each
    g = snappy.narrow_gaps(pts, surf, True, 4.0)
    assert g and g[0]["width_mm"] == pytest.approx(3.0)
    assert snappy.narrow_gaps(pts, surf, True, 2.5) == []


def test_faces_get_the_tag_of_the_surface_they_lie_on():
    pts, surf = _cube([0, 0, 0], [10, 10, 10])
    tri_of = dict(surf)
    info = {t: {"on": "part"} for t in tri_of}
    Cf = np.array([[5, 5, 10.01], [0.02, 5, 5], [5, -0.03, 4]])
    tags = snappy._nearest_tags(Cf, list(tri_of), tri_of, pts, info, None)
    assert tags.tolist() == [6, 1, 3]                       # zmax, xmin, ymin
    binfo = {snappy.BOX_TAG0 + 2: {"on": "box", "box": "ymin"}, snappy.BOX_TAG0 + 4: {"on": "box", "box": "zmin"}}
    t2 = snappy._nearest_tags(np.array([[50, -100, 30], [20, 0, -50]]), list(binfo), {}, pts, binfo,
                              [-200, -100, -50, 200, 100, 100])
    assert t2.tolist() == [snappy.BOX_TAG0 + 2, snappy.BOX_TAG0 + 4]


def test_obj_has_one_group_per_region(tmp_path):
    pts, surf = _cube([0, 0, 0], [10, 10, 10])
    snappy.write_obj(tmp_path / "p.obj", pts, [("r0", np.concatenate([t for _, t in surf[:2]])), ("r1", surf[2][1])])
    txt = (tmp_path / "p.obj").read_text()
    assert txt.count("\nv ") == 8 and txt.count("\nf ") == 6 and "g r0" in txt and "g r1" in txt
    assert "v 0.01 0.01 0.01" in txt                        # metres


# ---- validation, schemes, estimates -----------------------------------------------------------------------
def test_snappy_studies_validate():
    assert validate_study(EXT)["ok"], validate_study(EXT)["errors"]

    def errs(**mesh):
        return " ".join(validate_study(dict(EXT, mesh=dict(EXT["mesh"], **mesh)))["errors"])
    assert "only with mesher snappy" in errs(mesher="gmsh")
    assert "mesh.layers.on" in errs(layers={"first": 0.1, "on": ["far"]})            # a slip bc, not a wall
    assert "mesh.layers" in errs(layers={"first": 0.1, "y_plus": 30})
    assert "mesh.surface" in errs(surface=80)
    assert "mesh.refine" in errs(refine=[{"box": [0, 0, 0, -1, 1, 1], "size": 5}])
    assert "3D meshes only" in validate_study(dict(EXT, **{"2d": True}))["errors"][0] or \
        any("3D meshes only" in e for e in validate_study(dict(EXT, **{"2d": True}))["errors"])
    gm = dict(EXT, mesh={"size": 20, "layers": {"first": 0.1, "n": 3}})
    assert any("snappy" in e for e in validate_study(gm)["errors"])                   # 3D layers need snappy
    shapes = dict(EXT, mesh={"size": 20, "refine": [{"cylinder": {"center": [0, 0, 0], "axis": [1, 0, 0], "radius": 30,
                                                                   "span": [0, 300]}, "size": 8},
                                                     {"distance": 10, "size": 5}, {"at": [0, 0, 0], "r": 9, "size": 4}]})
    assert validate_study(shapes)["ok"], validate_study(shapes)["errors"]


def test_non_orthogonal_meshes_get_more_correctors():
    assert nonortho_schemes({"max_non_orthogonality": 40})[2] == 1
    assert nonortho_schemes({"max_non_orthogonality": 75})[:3] == ("Gauss linear limited corrected 0.33",
                                                                    "limited corrected 0.33", 2)
    assert nonortho_schemes({"max_non_orthogonality": 88})[2:] == (3, 0.7)


def test_snappy_cell_estimate():
    n = cell_estimate(EXT)
    assert 1e4 < n < 5e6


# ---- with Gmsh: the part's surface, oriented, with the bcs chosen on it ------------------------------------
@gmsh_only
def test_surface_premesh_is_oriented_out_of_the_part(tmp_path):
    pts, surf, ipt, bb = snappy.surface_premesh(EXT, tmp_path, 10, 2, EXT["domain"]["box"])
    vol = 0.0
    for _, t in surf:
        P = pts[t]
        vol += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6
    assert vol == pytest.approx(100 * 40 * 30, rel=1e-6)          # outward everywhere: the solid's volume
    assert ipt[0] < -50                                           # in the fluid, upstream
    pts2, surf2, ipt2, _ = snappy.surface_premesh({"geometry": {"cylinder": [5, 50]}}, tmp_path, 1, 0.25, None)
    assert np.hypot(ipt2[0], ipt2[1]) < 5 and 0 < ipt2[2] < 50


# ---- the whole path with OpenFOAM and Gmsh stood in for ----------------------------------------------------
def test_mesh_snappy_end_to_end_with_stand_ins(tmp_path, monkeypatch):
    """Internal box: the bcs are chosen on the surface, the 'snappyHexMesh' mesh (a box mesh written as binary
    polyMesh files with the region names) is read back, each face gets its CAD face's tag, the case is written
    around the copied files."""
    from cfd_translator import foam
    from cfd_translator.runner import prepare
    st = {"geometry": {"box": [100, 20, 10]}, "fluid": "water",
          "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.01}, {"kind": "outlet", "faces": {"nx": 1}}],
          "mesh": {"mesher": "snappy", "size": 5, "surface": 2.5, "layers": {"first": 0.5, "n": 2}}}
    assert validate_study(st)["ok"]
    pts, surf = _cube([-50, -10, 0], [50, 10, 10])
    monkeypatch.setattr(snappy, "surface_premesh", lambda *a, **k: (pts, surf, np.array([1.0, 1.0, 5.0]),
                                                                       (np.array([-50, -10, 0.0]), np.array([50, 10, 10.0]))))
    monkeypatch.setattr(foam, "find_foam", lambda: {"bashrc": None})
    # one rank: a machine whose config sets CFDT_PROCS would take the parallel path (decomposePar, mpirun)
    monkeypatch.setenv("CFDT_PROCS", "1")
    for k in ("SLURM_NTASKS", "SLURM_JOB_ID"):
        monkeypatch.delenv(k, raising=False)
    ran = []

    def step(fm, cmd, logname, case, timeout, raise_on_error=True):
        ran.append(cmd.split()[0])
        case = Path(case)
        if cmd.startswith("snappyHexMesh"):
            txt = (case / "system" / "snappyHexMeshDict").read_text()
            assert "r0 { name inlet; }" in txt and '"(walls|part_r2)"' in txt and "addLayers       true" in txt
            m = box_mesh(100, 20, 10, 10, 2, 2)
            m.set_patches([{"name": "inlet", "type": "patch", "tags": [0]}, {"name": "outlet", "type": "patch", "tags": [1]},
                           {"name": "walls", "type": "wall", "tags": [2, 3, 4, 5]}])
            _write_binary(case / "constant" / "polyMesh", m)
            (case / logname).write_text("Layer mesh : cells:400  faces:1300  points:500\n\npatch faces layers overall thickness\n"
                                        "  [m] [%]\n----- ----- ------ --- ---\nwalls 120 1.9 0.0011 95\n\n"
                                        "Finished meshing without any errors\nFinished meshing in = 1.5 s.\n")
        elif cmd.startswith("checkMesh"):
            (case / logname).write_text("    Mesh non-orthogonality Max: 0 average: 0\n\nMesh OK.\n")
        return 0
    monkeypatch.setattr(foam, "_step", step)
    mesh, patches, sel, plan, warns = prepare(st, tmp_path)
    assert ran == ["blockMesh", "surfaceFeatures", "snappyHexMesh", "checkMesh"]
    assert [p["name"] for p in patches] == ["inlet", "outlet", "walls"]
    assert patches[2]["tags"] == [3, 4, 5, 6] and patches[0]["tags"] == [1]
    walls = mesh.patch_faces("walls")
    Cf = mesh.geometry()["Cf"][walls]
    t = mesh.btag[walls - mesh.ninternal]
    assert np.all(np.abs(Cf[t == 3][:, 1] + 10) < 1e-9) and np.all(np.abs(Cf[t == 6][:, 2] - 10) < 1e-9)
    assert mesh.info["layers"]["coverage_pct"] == 95 and mesh.info["snappy"]["attempts"][0]["checkMesh"]["ok"]
    assert sel["inlet"][0]["normal"] == [-1.0, 0.0, 0.0]
    case = tmp_path / "case" / "constant" / "polyMesh"
    assert (case / "faces").read_bytes() == (tmp_path / "snappy" / "constant" / "polyMesh" / "faces").read_bytes()


def test_parallel_snappy_has_a_distributor():
    assert snappy.split3(8, [200, 100, 100]) == (2, 2, 2) and snappy.split3(6, [600, 100, 100]) == (6, 1, 1)
    assert snappy.split3(19, [100, 100, 100]) in ((19, 1, 1), (1, 19, 1), (1, 1, 19))
    t = snappy.decompose_dict(12, [400, 200, 100])
    assert "distributor     hierarchical;" in t and "n               (4 3 1);" in t and "decomposer      scotch;" in t


def test_surface_health_finds_holes():
    pts, surf = _cube([-50, -10, 0], [50, 10, 10])
    h = snappy.surface_health(pts, surf)
    assert h == {"open_edges": 0, "nonmanifold_edges": 0, "at": [], "faces": []}
    holed = [(t, tri[1:]) if k == 0 else (t, tri) for k, (t, tri) in enumerate(surf)]    # one triangle missing
    h = snappy.surface_health(pts, holed)
    assert h["open_edges"] == 3 and h["faces"] and len(h["at"]) >= 1


def _stand_in(monkeypatch, tmp_path, snappy_runs):
    """The end-to-end stand-ins (no Gmsh, no OpenFOAM); snappy_runs(k, dict_text, case) writes the k-th run's
    polyMesh and log, and returns its exit code."""
    from cfd_translator import foam
    pts, surf = _cube([-50, -10, 0], [50, 10, 10])
    monkeypatch.setattr(snappy, "surface_premesh", lambda *a, **k: (pts, surf, np.array([1.0, 1.0, 5.0]),
                                                                       (np.array([-50, -10, 0.0]), np.array([50, 10, 10.0]))))
    monkeypatch.setattr(foam, "find_foam", lambda: {"bashrc": None})
    monkeypatch.setenv("CFDT_PROCS", "1")
    for k in ("SLURM_NTASKS", "SLURM_JOB_ID"):
        monkeypatch.delenv(k, raising=False)
    calls = []

    def step(fm, cmd, logname, case, timeout, raise_on_error=True):
        case = Path(case)
        if cmd.startswith("snappyHexMesh"):
            calls.append((case / "system" / "snappyHexMeshDict").read_text())
            return snappy_runs(len(calls) - 1, calls[-1], case, logname)
        if cmd.startswith("checkMesh"):
            (case / logname).write_text("    Mesh non-orthogonality Max: 0 average: 0\n\nMesh OK.\n")
        return 0
    monkeypatch.setattr(foam, "_step", step)
    return calls


ST_BOX = {"geometry": {"box": [100, 20, 10]}, "fluid": "water",
          "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.01}, {"kind": "outlet", "faces": {"nx": 1}}],
          "mesh": {"mesher": "snappy", "size": 5, "surface": 2.5, "layers": {"first": 0.5, "n": 2}}}


def _good_mesh(case, logname, extra_patch=None):
    m = box_mesh(100, 20, 10, 10, 2, 2)
    pats = [{"name": "inlet", "type": "patch", "tags": [0]}, {"name": "outlet", "type": "patch", "tags": [1]},
            {"name": "walls", "type": "wall", "tags": [2, 3, 4, 5]}]
    if extra_patch:                                   # the zmax side left in snappy's background patch
        pats = pats[:2] + [{"name": "walls", "type": "wall", "tags": [2, 3, 4]}, {"name": extra_patch, "type": "patch", "tags": [5]}]
    m.set_patches(pats)
    _write_binary(case / "constant" / "polyMesh", m)
    (case / logname).write_text("Layer mesh : cells:400  faces:1300  points:500\n\nFinished meshing without any errors\n"
                                "Finished meshing in = 1.5 s.\n")
    return 0


def test_multiple_outside_loops_is_meshed_again_without_merging_faces(tmp_path, monkeypatch):
    from cfd_translator.runner import prepare

    def runs(k, txt, case, logname):
        if k == 0:
            (case / logname).write_text("--> FOAM FATAL ERROR: \nMultiple outside loops:0()\n\n    From function "
                                        "static Foam::face Foam::combineFaces::getOutsideFace\n")
            return 1
        return _good_mesh(case, logname)
    calls = _stand_in(monkeypatch, tmp_path, runs)
    mesh, patches, sel, plan, warns = prepare(ST_BOX, tmp_path)
    assert len(calls) == 2 and "mergeFaces" not in calls[0] and "mergeFaces false;" in calls[1]
    assert any("mergeFaces false" in w for w in warns + mesh.info.get("notes", []) + [str(mesh.info)])
    # no outsidePoint: OpenFOAM 14 would then keep closed pockets (inside blades) and stop in zonify
    assert "outsidePoint" not in calls[0] and "insidePoint (" in calls[0]


def test_a_leak_to_the_background_box_is_explained(tmp_path, monkeypatch):
    from cfd_translator.polymesh import MeshError
    from cfd_translator.runner import prepare
    _stand_in(monkeypatch, tmp_path, lambda k, txt, case, logname: _good_mesh(case, logname, "background"))
    with pytest.raises(MeshError, match="the fluid leaks out of the part"):
        prepare(ST_BOX, tmp_path)


def test_zone_edges_in_narrow_gaps_are_found():
    cyl = {"name": "impeller", "shape": "cylinder", "p1": [36.5, 0, 0], "p2": [78.5, 0, 0], "radius": 176}
    near = snappy._near_zone_edge
    assert near(cyl, np.array([36.8, 100.0, 0.0]), 2.0)          # on the end plane x = 36.5, inside the radius
    assert near(cyl, np.array([60.0, 0.0, 175.2]), 2.0)          # on the cylinder's side
    assert not near(cyl, np.array([60.0, 0.0, 100.0]), 2.0)      # inside, away from the boundary
    assert not near(cyl, np.array([36.8, 0.0, 200.0]), 2.0)      # beyond the radius, off the end plane
    box = {"shape": "box", "box": [0, 0, 0, 10, 10, 10]}
    assert near(box, np.array([5.0, 5.0, 9.5]), 1.0) and not near(box, np.array([5.0, 5.0, 5.0]), 1.0)


def test_quality_says_where_the_worst_cells_are():
    m = box_mesh(10, 10, 10, 4, 4, 4)
    q = m.quality()
    assert q["non_ortho_over_70"] == 0 and "worst_non_ortho" not in q
    m.points = m.points.copy()
    m.points[:, 0] += 4 * m.points[:, 1]                     # sheared: faces across y meet their cells at atan 4 = 76 deg
    m._geo = None
    q = m.quality()
    assert q["non_ortho_over_70"] > 0 and q["worst_non_ortho"][0]["deg"] == pytest.approx(76.0, abs=0.1)


def test_rotating_wall_faces_just_outside_the_zone_are_allowed():
    """s180 (2026.10.31): a semi-open impeller's vane tips sit 0.5 mm inside the zone's ends, across a 1 mm side
    gap, so 2 % of its faces touch cells just outside the zone. MRFnoSlip moves every face of the patch at
    Omega x r, so that is fine; faces far outside mean the zone is misplaced."""
    from cfd_translator.case import _check_rotating_wall
    from cfd_translator.polymesh import MeshError
    m = box_mesh(200, 200, 20, 20, 20, 2)                       # cells of 10 mm; x, y centred on 0, z from 0
    g = m.geometry()
    C = g["C"]
    z = {"name": "imp", "axis": [0, 0, 1], "center": [0, 0, 0], "radius": 40, "span": [0, 20], "rpm": 1000}
    m.zones = {"imp": np.nonzero(np.hypot(C[:, 0], C[:, 1]) <= 40)[0]}
    bf = np.arange(m.ninternal, m.nfaces)
    top = bf[np.abs(g["Cf"][bf, 2] - 20) < 1e-9]
    r = np.hypot(g["Cf"][top, 0], g["Cf"][top, 1])
    plan = {}
    _check_rotating_wall("impeller", "imp", top[r < 44], m, g, {"imp": z}, plan)       # 3 mm over the rim
    assert len(plan["notes"]) == 1 and "MRFnoSlip" in plan["notes"][0]
    with pytest.raises(MeshError, match="must cover the rotating walls"):
        _check_rotating_wall("impeller", "imp", top[r < 80], m, g, {"imp": z}, {})


def test_flatness_exception_is_meshed_again_without_the_check(tmp_path, monkeypatch):
    from cfd_translator.runner import prepare

    def runs(k, txt, case, logname):
        if k == 0:
            (case / logname).write_text("#0  Foam::error::printStack(Foam::Ostream&)\n#1  Foam::sigFpe::sigHandler(int)\n"
                                        "#3  Foam::meshCheck::checkFaceFlatness(bool, double, ...)\n"
                                        "#4  Foam::motionSmootherAlgo::scaleMesh(...)\nFloating point exception\n")
            return 136
        return _good_mesh(case, logname)
    calls = _stand_in(monkeypatch, tmp_path, runs)
    mesh, patches, sel, plan, warns = prepare(ST_BOX, tmp_path)
    assert len(calls) == 2 and "minFaceFlatness 0.5;" in calls[0] and "minFaceFlatness -1;" in calls[1]


def test_a_failed_snappy_mesh_keeps_its_findings(tmp_path, monkeypatch):
    from cfd_translator.polymesh import MeshError
    from cfd_translator.runner import prepare
    pts, surf = _cube([-50, -10, 0], [50, 10, 10])
    holed = [(t, tri[1:]) if k == 0 else (t, tri) for k, (t, tri) in enumerate(surf)]
    _stand_in(monkeypatch, tmp_path, lambda k, txt, case, logname: _good_mesh(case, logname, "background"))
    monkeypatch.setattr(snappy, "surface_premesh", lambda *a, **k: (pts, holed, np.array([1.0, 1.0, 5.0]),
                                                                       (np.array([-50, -10, 0.0]), np.array([50, 10, 10.0]))))
    with pytest.raises(MeshError) as ei:
        prepare(ST_BOX, tmp_path)
    assert "open edges" in str(ei.value) and any("open edges" in n for n in ei.value.notes)


def test_pockets_cut_off_from_the_flow_are_found():
    """s180 (2026.10.31): snappyHexMesh left 75 regions; the 74 closed pockets made the first pressure solve NaN."""
    class M:                                   # 6 cells in a row; cells 4 and 5 joined to each other only
        ncells, ninternal = 6, 4
        owner = np.array([0, 1, 2, 4, 0, 3, 4], np.int64)
        neighbour = np.array([1, 2, 3, 5], np.int64)
        info = {"foam_patches": [{"name": "inlet", "start": 4, "n": 1}, {"name": "outlet", "start": 5, "n": 1},
                                 {"name": "walls", "start": 6, "n": 1}]}

        def geometry(self):
            return {"C": np.column_stack([np.arange(6.0), np.zeros(6), np.zeros(6)]), "V": np.ones(6)}
    r = snappy.cut_off_regions(M(), {"inlet", "outlet"})
    assert r["keep"].tolist() == [0, 1, 2, 3] and r["n_pockets"] == 1 and r["cells"] == 2 and r["at"] == [4.5, 0.0, 0.0]
    M.ninternal = 5                                                  # all joined: nothing to remove
    M.neighbour = np.array([1, 2, 3, 4, 5], np.int64)
    M.owner = np.array([0, 1, 2, 3, 4, 0, 5], np.int64)
    M.info = {"foam_patches": [{"name": "inlet", "start": 5, "n": 1}, {"name": "outlet", "start": 6, "n": 1}]}
    assert snappy.cut_off_regions(M(), {"inlet", "outlet"}) is None


def test_a_nan_run_is_stopped_and_reported(tmp_path, monkeypatch):
    from cfd_translator import api, foam
    log = tmp_path / "log.foamRun"
    log.write_text("Time = 1s\n\nGAMG:  Solving for p, Initial residual = 1, Final residual = 0.01, No Iterations 12\n"
                   "Time = 2s\n\nGAMG:  Solving for p, Initial residual = 1, Final residual = nan, No Iterations 1000\n")
    assert foam.diverged(log) == "2" and "diverged at iteration 2" in foam.divergence_message("2")
    log.write_text("Time = 7s\n\nGAMG:  Solving for p, Initial residual = 0.3, Final residual = 0.2, No Iterations 1000\n"
                   "GAMG:  Solving for p, Initial residual = 0.2, Final residual = 0.1, No Iterations 1000\n")
    assert foam.diverged(log) is None and "1000-iteration limit" in api._solver_limit(log)
    # a running solver that writes NaN is killed at the next look
    monkeypatch.setattr(foam, "WATCH_S", 0.3)
    t0 = __import__("time").time()
    with pytest.raises(foam.Diverged, match="diverged at iteration 3"):
        foam._step({}, "sh -c \"printf 'Time = 3s\\nGAMG:  Solving for p, Initial residual = nan, Final residual = nan, "
                       "No Iterations 1000\\n'; sleep 30\" foamRun", "log.foamRun", tmp_path, 60)
    assert __import__("time").time() - t0 < 10


def _mesh_named(case, logname, names=("inlet", "outlet", "walls"), fold=False):
    m = box_mesh(100, 20, 10, 10, 2, 2)
    if fold:
        m.points = m.points.copy()
        m.points[:, 0] += 60 * m.points[:, 1]
        m._geo = None
    m.set_patches([{"name": names[0], "type": "patch", "tags": [0]}, {"name": names[1], "type": "patch", "tags": [1]},
                   {"name": names[2], "type": "wall", "tags": [2, 3, 4, 5]}])
    _write_binary(case / "constant" / "polyMesh", m)
    (case / logname).write_text("Layer mesh : cells:400  faces:1300  points:500\n\nFinished meshing without any errors\n"
                                "Finished meshing in = 1.5 s.\n")
    return 0


def test_a_layered_mesh_that_folds_is_made_again_without_layers(tmp_path, monkeypatch):
    """S50 (2026.11.01): 92 degrees 0.3 mm inside the outlet plane, one rejection per 20-minute attempt."""
    from cfd_translator.runner import prepare
    calls = _stand_in(monkeypatch, tmp_path, lambda k, txt, case, logname: _mesh_named(case, logname, fold=(k == 0)))
    mesh, patches, sel, plan, warns = prepare(ST_BOX, tmp_path)
    assert len(calls) == 2 and "addLayers       true" in calls[0] and "addLayers       false" in calls[1]
    assert any("made again without layers" in w for w in warns) and plan["quality"]["ok"]


def test_no_layers_on_walls_touching_openings(tmp_path, monkeypatch):
    from cfd_translator.runner import prepare
    st = dict(ST_BOX, mesh=dict(ST_BOX["mesh"], layers=dict(ST_BOX["mesh"]["layers"], at_openings=False)))
    from cfd_translator.format import validate_study
    assert validate_study(st)["ok"]
    calls = _stand_in(monkeypatch, tmp_path,
                      lambda k, txt, case, logname: _mesh_named(case, logname, ("inlet", "outlet", "walls_nolayers")))
    mesh, patches, sel, plan, warns = prepare(st, tmp_path)
    assert "r2o { name walls_nolayers; }" in calls[0]                       # the cube's side walls all touch an opening
    assert [p["name"] for p in patches] == ["inlet", "outlet", "walls"]      # merged back into their bc
    assert any("at_openings false" in w for w in warns)
    assert snappy.touching_tags([(1, np.array([[0, 1, 2]])), (2, np.array([[2, 3, 4]])), (3, np.array([[5, 6, 7]]))],
                                {1}) == {2}


def _check_by_layers(monkeypatch, fail_layered=True, fail_plain=False):
    """checkMesh stand-in failing (3 faces incorrectly oriented) on the meshes made with or without layers."""
    from cfd_translator import foam
    inner = foam._step
    state = {"layers": None}

    def step(fm, cmd, logname, case, timeout, raise_on_error=True):
        case = Path(case)
        if cmd.startswith("snappyHexMesh"):
            state["layers"] = "addLayers       true" in (case / "system" / "snappyHexMeshDict").read_text()
        rc = inner(fm, cmd, logname, case, timeout, raise_on_error)
        if cmd.startswith("checkMesh") and (fail_layered if state["layers"] else fail_plain):
            (case / logname).write_text("    Mesh non-orthogonality Max: 84 average: 10\n"
                                        " ***Number of faces incorrectly oriented: 3\n\nFailed 1 mesh checks.\n")
        return rc
    monkeypatch.setattr(foam, "_step", step)


def _mesh_by_layers(k, txt, case, logname):
    m = box_mesh(100, 20, 10, 10 if "addLayers       true" in txt else 20, 2, 2)     # tell them apart by cells
    m.set_patches([{"name": "inlet", "type": "patch", "tags": [0]}, {"name": "outlet", "type": "patch", "tags": [1]},
                   {"name": "walls", "type": "wall", "tags": [2, 3, 4, 5]}])
    _write_binary(case / "constant" / "polyMesh", m)
    (case / logname).write_text("Layer mesh : cells:400  faces:1300  points:500\n\nFinished meshing without any errors\n"
                                "Finished meshing in = 1.5 s.\n")
    return 0


def test_a_layered_mesh_that_fails_checkmesh_is_made_again_without_layers(tmp_path, monkeypatch):
    """s50_Q40_vol_n35 (2026.11.03): checkMesh failed (3 faces incorrectly oriented, worst 84 degrees) and the
    layers were kept, because only folds past 89 degrees triggered the rebuild."""
    from cfd_translator.runner import prepare
    calls = _stand_in(monkeypatch, tmp_path, _mesh_by_layers)
    _check_by_layers(monkeypatch, fail_layered=True, fail_plain=False)
    mesh, patches, sel, plan, warns = prepare(ST_BOX, tmp_path)
    assert ["addLayers       true" in c for c in calls] == [True, True, False]      # stricter quality, then no layers
    assert mesh.ncells == 80 and any("fails checkMesh (Number of faces incorrectly oriented: 3" in w for w in warns)
    assert any("without prism layers the mesh passes checkMesh" in w for w in warns)


def test_layers_kept_when_the_mesh_fails_checkmesh_without_them_too(tmp_path, monkeypatch):
    from cfd_translator.runner import prepare
    calls = _stand_in(monkeypatch, tmp_path, _mesh_by_layers)
    _check_by_layers(monkeypatch, fail_layered=True, fail_plain=True)
    mesh, patches, sel, plan, warns = prepare(ST_BOX, tmp_path)
    assert len(calls) == 4 and mesh.ncells == 40                                     # the layered mesh is back
    assert any("the layers were not the cause" in w for w in warns)
    assert mesh.info.get("layers") is not None


def test_snappy_killed_for_memory_says_so(tmp_path, monkeypatch):
    """carbonio (8 GB, 2026.11.03): snappyHexMesh was killed while adding layers; the result said 'MeshError' with an
    MPI 'proc-aborted-strsignal' trace."""
    from cfd_translator import foam
    from cfd_translator.runner import prepare

    def killed(k, txt, case, logname):
        (case / logname).write_text("Refinement phase\n----------------\nRefined mesh : cells:2100000\n"
                                    "Snapped mesh : cells:2250000\nShrinking and layer addition phase\n"
                                    "Layer addition iteration 0\n--------------------------------------------------\n"
                                    "prterun noticed that process rank 3 with PID 1234 on node carbonio exited on\n"
                                    "signal 9 (Killed).\n[proc-aborted-strsignal]\n")
        return 137
    _stand_in(monkeypatch, tmp_path, killed)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8192")
    with pytest.raises(foam.Killed) as ei:
        prepare(ST_BOX, tmp_path)
    msg = str(ei.value)
    assert "snappyHexMesh was killed while adding prism layers" in msg and "this job had 8 GB" in msg
    assert '"mem": "16G"' in msg and "2,250,000 cells" in msg
    assert type(ei.value).__name__ == "Killed" and hasattr(ei.value, "notes")


def test_a_killed_step_raises_killed(tmp_path):
    from cfd_translator import foam
    with pytest.raises(foam.Killed, match="sh was killed .signal 9., almost certainly because it ran out of memory"):
        foam._step({}, "sh -c 'echo working; exit 137'", "log.x", tmp_path, 30)
    assert foam.was_killed("x\nKilled\n") and not foam.was_killed("all fine", 1)


def test_hollow_cylinder_refinement(tmp_path, monkeypatch):
    """S50 sliding (2026.11.04): a 0.25 mm band r 58.5-61 mm round the rim instead of 18 boxes at the vane tips,
    without the ~14 M cells of a solid 0.25 mm cylinder. OpenFOAM 14 has no annulus surface: a closed
    triangulated one is written to constant/geometry."""
    from cfd_translator.executor import _refine_extents
    from cfd_translator.format import validate_study
    from cfd_translator.runner import prepare
    from cfd_translator.sliding import not_round_regions
    pts, T = snappy.annulus_tris([0, 0, -77.1], [0, 0, -45.0], 58.5, 61.0, 0.25)
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert (cnt == 2).all()                                              # closed
    P = pts[T]
    vol = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6
    assert vol == pytest.approx(np.pi * (61.0 ** 2 - 58.5 ** 2) * 32.1, rel=2e-3)   # outward: positive volume
    ann = {"cylinder": {"center": [0, 0, -77.1], "axis": [0, 0, 1], "radius": 61, "inner_radius": 58.5,
                        "span": [0, 32.1]}, "size": 0.25}
    st = dict(ST_BOX, mesh=dict(ST_BOX["mesh"], refine=[ann]))
    assert validate_study(st)["ok"], validate_study(st)
    bad = dict(ST_BOX, mesh=dict(ST_BOX["mesh"], refine=[{**ann, "cylinder": dict(ann["cylinder"], inner_radius=61)}]))
    assert any("inner_radius" in x for x in validate_study(bad)["errors"])
    (v_ann, _), = _refine_extents(st, [ann], 1e6, 10)
    (v_full, _), = _refine_extents(st, [{**ann, "cylinder": {k: v for k, v in ann["cylinder"].items() if k != "inner_radius"}}], 1e6, 10)
    assert v_ann == pytest.approx(v_full * (61 ** 2 - 58.5 ** 2) / 61 ** 2)
    calls = _stand_in(monkeypatch, tmp_path, lambda k, txt, case, logname: _good_mesh(case, logname))
    prepare(st, tmp_path)
    assert 'refine0\n    {\n        type triSurface;\n        file "refine0.obj";' in calls[0]
    assert "refine0 { mode inside; level" in calls[0]
    assert (tmp_path / "snappy" / "constant" / "geometry" / "refine0.obj").is_file()
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    Cf = np.column_stack([59.75 * np.cos(th), 59.75 * np.sin(th), np.full(36, -60.0)])
    assert not_round_regions([ann], Cf, np.array([0, 0, 1.0]), np.zeros(3)) == []


def test_local_ranks_are_capped_to_the_machine(monkeypatch):
    """get_case of an 18-rank S50 study on carbonio (6 cores): OpenMPI 5 stopped snappyHexMesh with
    'prte-rmaps-base:alloc-error'."""
    from cfd_translator import foam
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setattr(foam, "physical_cores", lambda: 6)
    n, note = foam.local_procs(18)
    assert n == 6 and "18 MPI ranks asked" in note and "allow 6" in note
    assert foam.local_procs(4) == (4, None)
    monkeypatch.setenv("SLURM_NTASKS", "10")
    n, note = foam.local_procs(18)
    assert n == 10 and "SLURM job's tasks" in note


def test_a_short_seam_is_told_apart_from_a_hole():
    """fluid_s50_df2.step: the B-rep is closed, but 3 triangle edges between faces 96 and 101 are open (the two
    faces' triangulations do not share their nodes): shorter than a surface cell."""
    pts, surf = _cube([0, 0, 0], [10, 10, 10])
    h = snappy.surface_health(pts, surf)
    assert h["open_edges"] == 0 and "open_length_mm" not in h
    cut = [(t, tri[1:]) if k == 0 else (t, tri) for k, (t, tri) in enumerate(surf)]
    h = snappy.surface_health(pts, cut)
    assert h["open_edges"] > 0 and h["open_length_mm"] > 0


def test_sliding_caps_on_background_grid_planes():
    """The hand-built ESI S50 case put snappy's background planes on the zone's caps (z -82.25 and -46.25 mm): no
    snapping there, AMI weights 0.98-1.16. sliding7's caps were mid-cell and the lower one leaned 64.7 degrees."""
    st = {"mesh": {"mesher": "snappy", "size": 4, "surface": [0.7, 2]},
          "rotating": [{"name": "impeller", "axis": [0, 0, 1], "center": [0, 0, -82.25], "rpm": 2900, "radius": 59.7,
                        "span": [0, 36.0], "motion": "sliding"}]}
    pbb = ([-120.3, -130.1, -131.7], [140.2, 400.9, 30.4])
    s = snappy.settings(st, False, list(pbb[0]) + list(pbb[1]), pbb)
    lo, hi, n = np.array(s["lo"]), np.array(s["hi"]), np.array(s["cells"])
    dz = (hi[2] - lo[2]) / n[2]
    for cap in (-82.25, -46.25):
        k = (cap - lo[2]) / dz
        assert abs(k - round(k)) < 1e-9, (cap, k)
        for L in range(5):                                    # a face plane at every refinement level
            kk = (cap - lo[2]) / (dz / 2 ** L)
            assert abs(kk - round(kk)) < 1e-9
    assert dz == pytest.approx(4.0) and s["h0"] == pytest.approx(4.0)
    assert lo[2] <= pbb[0][2] - 1.5 * dz + 1e-9 and hi[2] >= pbb[1][2] + 1.5 * dz - 1e-9
    assert s["aligned"]["done"] and s["aligned"]["cells_between_caps"] == 9
    # a span that is not a whole number of cells: the axial cells shrink a little so both caps are planes
    st["rotating"][0]["span"] = [0, 37.0]
    s = snappy.settings(st, False, list(pbb[0]) + list(pbb[1]), pbb)
    lo, hi, n = np.array(s["lo"]), np.array(s["hi"]), np.array(s["cells"])
    dz = (hi[2] - lo[2]) / n[2]
    assert dz == pytest.approx(3.7) and s["h0"] == pytest.approx(4.0)
    for cap in (-82.25, -45.25):
        assert abs((cap - lo[2]) / dz - round((cap - lo[2]) / dz)) < 1e-9
    # x and y untouched; an oblique axis or an external box: left as they were, and said so
    assert (hi[0] - lo[0]) / n[0] == pytest.approx(4.0)
    st["rotating"][0]["axis"] = [0, 0.6, 0.8]
    assert snappy.settings(st, False, list(pbb[0]) + list(pbb[1]), pbb)["aligned"]["done"] is False


def test_meshing_ranks_fit_the_memory():
    """S50 on mochi (58 GB, 18 ranks): snappyHexMesh killed while snapping at 8.3 M and at 3.7-4.2 M cells: every rank
    holds the whole surface. Without mesh.procs the meshing ranks are fitted to the job's memory."""
    # mochi, 6 ranks, 756,633 triangles, 3.9 M cells: 8.65 GB measured; the estimate within 20 %
    assert snappy.memory_estimate(6, 3.9e6, 756_633)["total_mb"] == pytest.approx(8.65 * 1024, rel=0.2)
    e = snappy.memory_estimate(18, 4e6, 3_000_000)
    assert e["per_rank_mb"] == pytest.approx(300 + 3_000_000 / 1024, abs=1) and e["total_mb"] > 58 * 1024 * 0.85
    p, mem, note = snappy.meshing_procs(18, 4e6, 3_000_000, 58 * 1024, False)
    assert 1 <= p < 18 and mem["total_mb"] <= 0.85 * 58 * 1024 and "instead of 18" in note and mem["procs"] == p
    assert snappy.memory_estimate(p + 1, 4e6, 3_000_000)["total_mb"] > 0.85 * 58 * 1024
    p2, _, note2 = snappy.meshing_procs(18, 4e6, 3_000_000, 58 * 1024, True)          # mesh.procs given: kept
    assert p2 == 18 and "may run out of memory" in note2
    assert snappy.meshing_procs(4, 1e5, 10_000, 58 * 1024, False)[2] is None


def test_mesh_procs_validation_and_presubmit_warning(monkeypatch):
    from cfd_translator import api
    from cfd_translator.format import validate_study
    st = {"geometry": {"box": [100, 50, 20]}, "domain": {"kind": "external", "box": [-200, -100, -100, 400, 100, 100]},
          "fluid": "air", "bcs": [{"kind": "inlet", "faces": {"box": "xmin"}, "velocity": 10},
                                  {"kind": "outlet", "faces": {"box": "xmax"}}],
          "mesh": {"mesher": "snappy", "size": 10, "procs": 4}}
    assert validate_study(st)["ok"], validate_study(st)
    assert any("mesh.procs" in e for e in validate_study(dict(st, mesh=dict(st["mesh"], procs=0)))["errors"])
    assert any("mesh.procs" in e for e in validate_study(dict(st, mesh={"size": 10, "procs": 2}))["errors"])
    w = api._snappy_mem_warning(dict(st, mesh=dict(st["mesh"], procs=18)), {"mem_mb": 8192, "cells": 4e6, "procs": 18})
    assert w and "every rank holds the whole surface" in w and "lower mesh.procs" in w
    assert api._snappy_mem_warning(st, {"mem_mb": 64000, "cells": 1e5, "procs": 4}) is None


def test_inverted_cells_at_a_wall_are_found(tmp_path):
    """Coarse S50 noholes / rot0 on mochi: checkMesh still failed after the stricter retry on one inverted boundary
    face. The cells around a few such faces (checkMesh's faceSet) are removed, their faces joining the wall."""
    from cfd_translator.polymesh import box_mesh
    m = box_mesh(10, 10, 10, 2, 2, 2)
    m.info["foam_patches"] = [{"name": "walls", "type": "wall", "start": m.ninternal, "n": m.nfaces - m.ninternal}]
    d = tmp_path / "constant" / "polyMesh"
    (d / "sets").mkdir(parents=True)
    f = m.nfaces - 1
    (d / "sets" / "wrongOrientedFaces").write_text(
        "FoamFile\n{\n    format      ascii;\n    class       faceSet;\n    object      wrongOrientedFaces;\n}\n\n"
        f"1\n(\n{f}\n)\n")
    (tmp_path / "log.checkMesh").write_text(
        "  <<Writing 1 faces with incorrect orientation to set wrongOrientedFaces\n")
    cm = {"errors": ["***Error in face pyramids: 1 faces are incorrectly oriented."]}
    bad = snappy.wrong_oriented_cells(tmp_path, d, m, cm)
    assert bad and list(bad["cells"]) == [int(m.owner[f])] and bad["patch"] == "walls" and bad["faces"] == 1
    assert snappy.wrong_oriented_cells(tmp_path, d, m, {"errors": ["***Max skewness = 4.6"]}) is None


def test_step_peak_memory(tmp_path):
    """mesh.snappy.memory.peak_mb / execution.memory_peak_mb: the largest resident memory of a step's processes."""
    import os

    from cfd_translator import foam
    if not os.path.isdir("/proc"):
        pytest.skip("Linux /proc")
    assert foam.session_rss_mb(os.getsid(0)) > 1
    foam.PEAK_MB.clear()
    foam._step({}, "sleep 1.6", "log.sleep", tmp_path, 30)
    assert foam.PEAK_MB.get("log.sleep", 0) > 0


MOCHI_CRASH = (
    "Snapping phase\n---------------\n"
    "[mochi:53815] *** Process received signal ***\n[mochi:53815] Signal: Segmentation fault (11)\n"
    "[mochi:53815] Signal code: Address not mapped (1)\n"
    "[mochi:53815] [ 9] /opt/openfoam14/platforms/linux64GccDPInt32Opt/lib/libsnappyHexMesh.so(_ZN4Foam16snappySnapDriver6"
    "doSnapERKNS_10dictionaryES3_ddRKNS_14snapParametersE+0x1a5b) [0x6c7b1516fdb]\n"
    "[mochi:53815] [10] snappyHexMesh(+0x19c02) [0x64f4a9635c02]\n[mochi:53815] *** End of error message ***\n"
    "Sorry!  You were supposed to get help about:\n    prun:proc-aborted-strsignal\n")


def test_crash_is_not_a_memory_kill():
    """mochi, S50 coarse base (job 3240): 'killed ... out of memory' while snapping at 8.7 GB of 58 GB; the log had
    OpenMPI's backtrace of a SIGSEGV in snappySnapDriver::doSnap."""
    from cfd_translator import foam
    cr = foam.crash_of(MOCHI_CRASH, 1)
    assert cr["name"] == "SIGSEGV" and cr["in"] == "Foam::snappySnapDriver::doSnap"
    assert not foam.was_killed(MOCHI_CRASH, 1)
    assert foam.crash_of("prterun noticed ... exited on\nsignal 9 (Killed).\n[proc-aborted-strsignal]\n", 137) is None
    of = "#0  Foam::error::printStack(Foam::Ostream&)\n#1  Foam::sigFpe::sigHandler(int)\n#2  ?\n" \
         "#3  Foam::GAMGSolver::solve(...)\n"
    assert foam.crash_of(of, 136) == {"signal": 8, "name": "SIGFPE", "in": "Foam::GAMGSolver::solve",
                                      "frames": ["Foam::GAMGSolver::solve"]}
    assert foam.build_of is not None


def test_parallel_snappy_crash_is_meshed_again_in_serial(tmp_path, monkeypatch):
    from cfd_translator import foam
    from cfd_translator.runner import prepare
    _stand_in(monkeypatch, tmp_path, lambda *a: 0)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    cmds = []

    def step(fm, cmd, logname, case, timeout, raise_on_error=True):
        case = Path(case)
        cmds.append(cmd)
        if "snappyHexMesh" in cmd:
            if "-parallel" in cmd:
                (case / logname).write_text("Build  : 14-mochi0000\n" + MOCHI_CRASH)
                return 1
            return _good_mesh(case, logname)
        if cmd.startswith("checkMesh"):
            (case / logname).write_text("    Mesh non-orthogonality Max: 0 average: 0\n\nMesh OK.\n")
        return 0
    monkeypatch.setattr(foam, "_step", step)
    st = dict(ST_BOX, mesh=dict(ST_BOX["mesh"], procs=2))
    mesh, patches, sel, plan, warns = prepare(st, tmp_path)
    assert any("snappyHexMesh crashed (SIGSEGV) while snapping to the surface in Foam::snappySnapDriver::doSnap" in w
               and "made again in serial" in w and "14-mochi0000" in w for w in warns), warns
    assert any(c.startswith("snappyHexMesh") and "-parallel" not in c for c in cmds)
    # in serial too: an error that says crash, not memory
    def step2(fm, cmd, logname, case, timeout, raise_on_error=True):
        if "snappyHexMesh" in cmd:
            (Path(case) / logname).write_text("Build  : 14-mochi0000\n" + MOCHI_CRASH)
            return 1
        return step(fm, cmd, logname, case, timeout, raise_on_error)
    monkeypatch.setattr(foam, "_step", step2)
    with pytest.raises(snappy.MeshError, match="crashed .SIGSEGV.*it crashed on 2 ranks too.*not in the study's size"):
        prepare(st, tmp_path / "again")
