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
    assert L["last_mm"] >= 0.5 * st["surface_mm"][0] > L["last_mm"] / L["growth"]       # n: last ~ half a cell
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
