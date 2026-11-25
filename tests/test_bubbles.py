"""One-way bubble tracking (bubbles.py) on analytic fields: still column, solid-body rotation in the rotating and the
absolute frame, a bubble on the axis; reading a case's fields and the study validation."""
import math

import numpy as np
import pytest

from cfd_translator import bubbles, twophase
from cfd_translator.format import validate_study

RHO, MU = 998.2, 1.002e-3
NU = MU / RHO


def _grid(x, y, z, h):
    g = np.stack(np.meshgrid(np.arange(x[0] + h / 2, x[1], h), np.arange(y[0] + h / 2, y[1], h),
                             np.arange(z[0] + h / 2, z[1], h), indexing="ij"), -1).reshape(-1, 3)
    return g, np.full(len(g), h ** 3)


def test_still_column_terminal_velocity():
    C, V = _grid((0, 0.01), (0, 0.01), (0, 0.1), 0.001)
    f = bubbles.Field(C, V, np.zeros_like(C), np.zeros(len(C)))
    for d in (0.1e-3, 1e-3):
        ut, _ = twophase.slip_velocity(9.80665, d, RHO, 0.0, MU)
        T = min(0.3, 0.05 / ut)
        z = [bubbles.track(f, [[0.005, 0.005, 0.01]], d, NU, tt, g=(0, 0, -9.80665), dt_max=1e-4)["X"][0, 2]
             for tt in (T / 2, T)]
        v = (z[1] - z[0]) / (T / 2)                 # the second half: at terminal speed after a few response times
        assert v == pytest.approx(ut, rel=0.02), (d, v, ut)


def _cylinder(R=0.05, H=0.004, h=0.001, omega=10.0, mrf=True):
    C, V = _grid((-R, R), (-R, R), (0, H), h)
    keep = np.hypot(C[:, 0], C[:, 1]) < R
    C, V = C[keep], V[keep]
    U = np.column_stack([-omega * C[:, 1], omega * C[:, 0], np.zeros(len(C))])
    p = 0.5 * omega ** 2 * (C[:, 0] ** 2 + C[:, 1] ** 2)
    zone = np.zeros(len(C), np.int32) if mrf else None
    m = [{"name": "rot", "cellZone": "rot", "origin": np.zeros(3), "axis": np.array([0, 0, 1.0]), "omega": omega}]
    return bubbles.Field(C, V, U, p, zone, m if mrf else ())


def _drag_law_r(r0, t, d, omega):
    r, n = r0, 2000
    dt = t / n
    for _ in range(n):
        r -= dt * twophase.slip_velocity(omega ** 2 * r, d, RHO, 0.0, MU)[0]
    return r


@pytest.mark.parametrize("mrf", [True, False])
def test_solid_body_migration_against_the_drag_law(mrf):
    f = _cylinder(mrf=mrf)
    d, T = 0.3e-3, 3.0
    X0 = [[0.04, 0.0, 0.002], [0.0, -0.03, 0.002]]
    r = bubbles.track(f, X0, d, NU, T, g=(0, 0, 0), dt_max=2e-3)
    rr = np.hypot(r["X"][:, 0], r["X"][:, 1])
    for i, r0 in enumerate((0.04, 0.03)):
        assert rr[i] == pytest.approx(_drag_law_r(r0, T, d, 10.0), rel=0.03), (mrf, i, rr[i])
    assert (r["fate"] == "flow").all()


def test_bubble_on_the_axis_stays():
    f = _cylinder(mrf=True)
    r = bubbles.track(f, [[0.0, 0.0, 0.002]], 1e-3, NU, 2.0, g=(0, 0, 0), dt_max=2e-3)
    assert np.hypot(*r["X"][0, :2]) < 2e-4


def test_target_residence_and_summary():
    f = _cylinder(mrf=True)
    tgt = {"name": "core", "cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 20, "span": [-1, 10]}}
    r = bubbles.track(f, [[0.03, 0.0, 0.002], [0.045, 0.0, 0.002]], 2e-3, NU, 1.0, g=(0, 0, 0), dt_max=2e-3,
                      target=tgt)
    s = bubbles.summarize(r, 10.0)
    assert s["n"] == 2 and s["reached_target"] > 0 and "residence_in_target_rev" in s


def test_validation():
    ok = {"bubbles": {"source": "run_x/p1", "releases": [{"name": "slot", "box": {"min": [0, 0, 0], "max": [1, 1, 1]},
                                                           "n": 50}],
                      "diameters_mm": [0.1, 1], "gravity": [0, -9.81, 0], "revolutions": 20, "target": "eye"},
          "regions": [{"name": "eye", "cylinder": {"center": [0, 0, -56], "axis": [0, 0, 1], "radius": 22,
                                                   "span": [0, 23]}}],
          "run": {"host": "spark1"}}
    assert validate_study(ok)["ok"], validate_study(ok)
    assert validate_study(dict(ok, solve={"procs": 1}))["ok"]          # the SLURM job adds solve.procs
    bad = {"bubbles": {"releases": [{"name": "a"}], "diameters_mm": [50]}, "geometry": {"box": [1, 1, 1]}}
    e = validate_study(bad)["errors"]
    assert any("source" in x for x in e) and any("diameters" in x for x in e) and any("only bubbles" in x for x in e)


def _write_field(path, cls, n, body, boundary=""):
    path.write_text("FoamFile\n{\n    format      ascii;\n    class       " + cls + ";\n    object      "
                    + path.name + ";\n}\n\ndimensions      [0 0 0 0 0 0 0];\n\ninternalField   nonuniform List<"
                    + ("vector" if "Vector" in cls else "scalar") + f">\n{n}\n(\n{body}\n)\n;\n\nboundaryField\n{{\n"
                    + boundary + "}\n")


def test_reads_a_case(tmp_path):
    case = tmp_path / "req" / "p1" / "case"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "system").mkdir()
    C, V = _grid((0, 0.004), (0, 0.004), (0, 0.004), 0.001)
    n = len(C)
    t = case / "500"
    t.mkdir()
    vec = lambda A: "\n".join(f"({a[0]:.6g} {a[1]:.6g} {a[2]:.6g})" for a in A)
    bnd = ("    outlet\n    {\n        type            calculated;\n        value           nonuniform List<vector> \n2\n(\n"
           "(0.004 0.001 0.001)\n(0.004 0.003 0.003)\n)\n;\n    }\n")
    _write_field(t / "C", "volVectorField", n, vec(C), bnd)
    _write_field(t / "V", "volScalarField", n, "\n".join(f"{v:.6g}" for v in V))
    _write_field(t / "U", "volVectorField", n, vec(np.zeros_like(C)))
    _write_field(t / "p", "volScalarField", n, "\n".join("0" for _ in V))
    (case / "constant" / "polyMesh" / "cellZones").write_text(
        "FoamFile\n{\n    format ascii;\n    class regIOobject;\n    object cellZones;\n}\n\n1\n(\nrot\n{\n"
        f"    type cellZone;\ncellLabels      List<label> 3\n(\n0\n1\n2\n)\n;\n}}\n)\n")
    (case / "constant" / "MRFProperties").write_text(
        "FoamFile\n{\n    format ascii;\n    class dictionary;\n    object MRFProperties;\n}\n\nrot\n{\n"
        "    cellZone        rot;\n    origin          (0 0 0);\n    axis            (0 0 1);\n    omega           5;\n}\n")
    d = bubbles.cell_data(case)
    assert d["time"] == 500 and d["C"].shape == (n, 3) and d["mrf"][0]["omega"] == 5
    assert (d["zone_of"][:3] == 0).all() and (d["zone_of"][3:] == -1).all()
    assert d["faces"]["outlet"].shape == (2, 3)


def test_study_end_to_end(tmp_path):
    """A bubbles study through the runner: a still column whose top is an outlet; the bubbles rise and leave."""
    import json

    from cfd_translator.runner import run_study
    req = tmp_path / "req"
    case = req / "col" / "case"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "system").mkdir()
    C, V = _grid((0, 0.01), (0, 0.01), (0, 0.04), 0.001)
    n = len(C)
    t = case / "100"
    t.mkdir()
    vec = lambda A: "\n".join(f"({a[0]:.6g} {a[1]:.6g} {a[2]:.6g})" for a in A)
    top, _ = _grid((0, 0.01), (0, 0.01), (0, 0.001), 0.001)
    top[:, 2] = 0.04
    bnd = ("    outlet\n    {\n        type            calculated;\n        value           nonuniform List<vector> \n"
           f"{len(top)}\n(\n{vec(top)}\n)\n;\n    }}\n")
    _write_field(t / "C", "volVectorField", n, vec(C), bnd)
    _write_field(t / "V", "volScalarField", n, "\n".join(f"{v:.6g}" for v in V))
    _write_field(t / "U", "volVectorField", n, vec(np.zeros_like(C)))
    _write_field(t / "p", "volScalarField", n, "\n".join("0" for _ in V))
    (req / "request.json").write_text(json.dumps({"studies": {"col": {
        "fluid": "water", "bcs": [{"kind": "outlet", "name": "outlet", "faces": {"nz": 1}, "pressure": 0}]}}}))
    st = {"bubbles": {"source": str(req / "col"), "diameters_mm": [1.0], "t_max_s": 1.0, "gravity": [0, 0, -9.81],
                      "releases": [{"name": "bottom", "box": {"min": [2, 2, 2], "max": [8, 8, 5]}, "n": 20}],
                      "target": {"name": "mid", "box": {"min": [0, 0, 15], "max": [10, 10, 25]}}, "keep": 5}}
    r = run_study("trk", st, tmp_path / "out" / "trk")
    assert r["ok"], r
    b = r["results"]["bubbles"]["releases"]["bottom"]["1"]
    assert b["outlet"] == 1.0 and b["reached_target"] == 1.0
    ut = twophase.slip_velocity(9.81, 1e-3, RHO, 0.0, MU)[0]
    assert b["residence_in_target_s"]["median"] == pytest.approx(0.01 / ut, rel=0.1)
    assert (tmp_path / "out" / "trk" / "bubbles_bottom_1mm.png").is_file()


def test_bubble_slides_up_an_inclined_ceiling():
    """At a wall the bubble keeps the part of its move along the wall (a sloping ceiling: it slides up the slope)
    instead of stopping (the first S50 run: 80-90 % of the pocket's bubbles 'stuck')."""
    C, V = _grid((0, 0.06), (0, 0.006), (0, 0.06), 0.001)
    keep = C[:, 2] < 0.02 + 0.3 * C[:, 0]                      # a ceiling rising along +x
    C, V = C[keep], V[keep]
    f = bubbles.Field(C, V, np.zeros_like(C), np.zeros(len(C)))
    r = bubbles.track(f, [[0.01, 0.003, 0.015]], 1e-3, NU, 1.0, g=(0, 0, -9.80665), dt_max=1e-3)
    assert r["fate"][0] == "flow" and r["touched_wall"][0]
    assert r["X"][0, 0] > 0.03                                   # it moved up the slope


def test_outside_is_judged_against_the_larger_neighbour():
    """A point inside a large cell next to small ones is in the fluid (nearest to a small cell's centre)."""
    big, Vb = _grid((0, 0.01), (0, 0.01), (0, 0.01), 0.002)
    small, Vs = _grid((0.01, 0.014), (0, 0.01), (0, 0.01), 0.0005)
    f = bubbles.Field(np.vstack([big, small]), np.concatenate([Vb, Vs]), np.zeros((len(big) + len(small), 3)),
                      np.zeros(len(big) + len(small)))
    out = f.at([[0.0097, 0.005, 0.005], [0.0200, 0.005, 0.005]])[4]
    assert not out[0] and out[1]


def test_held_under_a_flat_ceiling_is_reported_with_its_place():
    C, V = _grid((0, 0.01), (0, 0.01), (0, 0.02), 0.001)
    f = bubbles.Field(C, V, np.zeros_like(C), np.zeros(len(C)))
    r = bubbles.track(f, [[0.005, 0.005, 0.005]], 1e-3, NU, 1.0, g=(0, 0, -9.80665), dt_max=1e-3)
    assert r["fate"][0] == "wall" and r["X"][0, 2] > 0.018
    s = bubbles.summarize(r, None)
    assert s["stuck_at_wall"] == 1.0 and s["held_at_mm"]["centroid"][2] > 18 and s["finished"] == 1.0
