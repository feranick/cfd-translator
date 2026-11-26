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
    assert validate_study({"bubbles": {"source": "r/s", "slices": [{"name": "a", "z": -52}]}})["ok"]   # no releases
    assert not validate_study({"bubbles": {"source": "r/s", "slices": [{"name": "a"}]}})["ok"]
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
    # two targets side by side, and a dry run that counts each region's cells
    st2 = {"bubbles": dict(st["bubbles"], targets=["mid", "top"]),
           "regions": [st["bubbles"]["target"], {"name": "top", "box": {"min": [0, 0, 30], "max": [10, 10, 40]}}]}
    del st2["bubbles"]["target"]
    r2 = run_study("trk2", st2, tmp_path / "out" / "trk2")
    t2 = r2["results"]["bubbles"]["releases"]["bottom"]["1"]["targets"]
    assert t2["mid"]["reached"] == 1.0 and t2["top"]["reached"] == 1.0
    assert t2["top"]["time_to_s_median"] > t2["mid"]["time_to_s_median"]
    st3 = {"bubbles": dict(st2["bubbles"], dry_run=True,
                           releases=st["bubbles"]["releases"] + [{"name": "nowhere", "box": {"min": [50, 50, 50],
                                                                                              "max": [60, 60, 60]}}]),
           "regions": st2["regions"]}
    r3 = run_study("dry", st3, tmp_path / "out" / "dry")
    d = r3["results"]["bubbles"]["dry_run"]
    assert d["bottom"]["cells"] == 6 * 6 * 3 and d["nowhere"]["cells"] == 0 and d["target:top"]["cells"] > 0
    assert any("nowhere" in w for w in r3["warnings"])
    st4 = {"bubbles": dict(st2["bubbles"], tracer="backward", t_max_s=0.1, bins=["mid", "top"]),
           "regions": st2["regions"]}
    r4 = run_study("trc", st4, tmp_path / "out" / "trc")
    assert r4["ok"], r4
    e4 = r4["results"]["bubbles"]["releases"]["bottom"]["0.1"]["ends"]
    assert e4["in_bins"]["elsewhere"] == 1.0 and e4["by_fate"] == {"flow": 1.0}     # still liquid: tracers stay put


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


def test_backward_tracer_and_stayers_at_a_ceiling():
    # a liquid tracer traced back in a uniform stream ends upstream
    C, V = _grid((0, 0.1), (0, 0.01), (0, 0.01), 0.001)
    U = np.tile([0.1, 0.0, 0.0], (len(C), 1))
    f = bubbles.Field(C, V, U, np.zeros(len(C)))
    r = bubbles.track(f, [[0.06, 0.005, 0.005]], 1e-4, NU, 0.2, g=(0, 0, 0), tracer=-1)
    assert r["X"][0, 0] == pytest.approx(0.04, abs=2e-4)
    # bubbles held under a closed top inside a target count as staying there, at the wall named
    C, V = _grid((0, 0.01), (0, 0.01), (0, 0.02), 0.001)
    top, _ = _grid((0, 0.01), (0, 0.01), (0, 0.001), 0.001)
    top[:, 2] = 0.02
    f = bubbles.Field(C, V, np.zeros_like(C), np.zeros(len(C)), faces={"walls": top})
    tgt = {"name": "top", "box": {"min": [0, 0, 15], "max": [10, 10, 21]}}
    r = bubbles.track(f, [[0.005, 0.005, 0.005]] * 3, 1e-3, NU, 1.0, g=(0, 0, -9.80665), target=[tgt])
    assert (r["fate"] == "wall").all() and r["targets"]["top"]["in_at_end"].all()
    assert r["targets"]["top"]["t_in"][0] > 0.5                  # counted to the end, not to the moment it was held
    data = {"mrf": []}
    st = bubbles.stayers(f, data, {"X": r["X"], "fate": r["fate"], "targets": r["targets"]}, [tgt])
    assert st["top"]["n"] == 3 and st["top"]["held"] == 3 and st["top"]["held_by_wall"] == {"walls": 3}
    assert st["top"]["points"][0]["wall_mm"] < 1.0


def test_region_mask_annulus():
    r = {"cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 10, "inner_radius": 5, "span": [0, 1]}}
    X = np.array([[0.003, 0, 0.0005], [0.007, 0, 0.0005], [0.012, 0, 0.0005]])
    assert bubbles.region_mask(r, X).tolist() == [False, True, False]


def test_slices_and_path_speeds(tmp_path):
    f = _cylinder(mrf=True)                             # solid body: p = omega^2 r^2 / 2, no axial flow
    f.U[:, 2] = np.where(np.hypot(f.C[:, 0], f.C[:, 1]) < 0.01, 0.3, -0.1)     # a core flowing back along +z
    f.F[:, 1:4] = f.U
    data = {"mrf": f.mrf}
    out = bubbles.slices(f, data, [{"name": "mid", "z": 2.0, "r_max": 40, "mark": [[0, 0, 2]]},
                                   {"name": "mer", "meridional_deg": -60, "r_max": 40}], 998.2, tmp_path)
    assert out["mid"]["backflow_share"] > 0 and out["mid"]["p_min_r_mm"] < 2.0
    assert (tmp_path / "slice_mid.png").is_file() and out["mer"]["cells"] > 10
    # liquid speed along kept paths against rise velocities
    src = tmp_path / "req" / "trc"
    src.mkdir(parents=True)
    np.savez(src / "trajectories.npz", a=np.array([[0.0, 0.03, 0.002], [0.0, 0.031, 0.002]]))
    st = {"regions": [{"name": "zone", "box": {"min": [-50, -50, 0], "max": [50, 50, 4]}}]}
    r = bubbles.path_speeds(f, {"paths": str(src), "region": "zone", "direction": [0, 0, -1]}, st,
                            {"diameters_mm": [0.1, 1]}, NU, [])
    assert r["points"] == 2 and r["liquid_speed_m_s"]["median"] == pytest.approx(0.1, rel=1e-3)
    assert r["share_of_path_points_where_liquid_beats_rise"] == {"0.1": 1.0, "1": 0.0}


def test_region_sector():
    r = {"cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 30, "inner_radius": 20, "span": [0, 1],
                      "angles": [170, -170]}}                     # across +-180
    X = np.array([[-0.025, 0.001, 0.0005], [-0.025, -0.001, 0.0005], [0.025, 0, 0.0005], [0, -0.025, 0.0005]])
    assert bubbles.region_mask(r, X).tolist() == [True, True, False, False]
    r2 = {"cylinder": {"center": [0, 0, 0], "axis": [0, 0, 1], "radius": 30, "angles": [-72, -62]}}
    X2 = np.array([[0.025 * np.cos(np.radians(-66)), 0.025 * np.sin(np.radians(-66)), 0]])
    assert bubbles.region_mask(r2, X2).tolist() == [True]
    assert validate_study({"bubbles": {"source": "r/s", "releases": [{"name": "s", "cylinder": dict(
        r2["cylinder"], angles=[1])}]}})["errors"]


# ---- turbulence: Hinze's size, dispersion, escape, zone crossings (2026.11.26.6) --------------------------------
def test_hinze_dmax_formula():
    k, om = np.array([2.0]), np.array([5000.0])
    eps = 0.09 * 2.0 * 5000.0
    want = 0.725 * (0.072 / 998.0) ** 0.6 * eps ** -0.4
    assert bubbles.hinze_dmax(k, om, 998.0)[0] == pytest.approx(want, rel=1e-12)
    assert bubbles.epsilon(k, om)[0] == pytest.approx(eps)
    # stronger turbulence: smaller stable bubbles
    assert bubbles.hinze_dmax(k, 10 * om, 998.0)[0] < bubbles.hinze_dmax(k, om, 998.0)[0]


def test_random_walk_spreads_like_the_eddy_diffusivity():
    """Small bubbles in still liquid with uniform k, omega and no gravity: the mean-square displacement grows as
    2 k tau t (three components of variance 2k/3, one independent jump u' tau per eddy lifetime tau)."""
    C, V = _grid((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), 0.01)
    k0, om0 = 0.01, 100.0
    f = bubbles.Field(C, V, np.zeros_like(C), np.zeros(len(C)), tke=np.full(len(C), k0), omega=np.full(len(C), om0))
    T, nb = 1.0, 400
    r = bubbles.track(f, np.zeros((nb, 3)), 0.05e-3, NU, T, g=(0, 0, 0), dt_max=1e-3, dispersion=True,
                      rng=np.random.default_rng(3))
    tau = 2 * bubbles.C_L / (bubbles.C_MU * om0)
    msd = float((r["X"] ** 2).sum(axis=1).mean())
    assert msd == pytest.approx(2 * k0 * tau * T, rel=0.2), (msd, 2 * k0 * tau * T)
    assert r["dispersion"]
    # no k and omega: no dispersion, the bubbles stay put
    g = bubbles.Field(C, V, np.zeros_like(C), np.zeros(len(C)))
    r0 = bubbles.track(g, np.zeros((5, 3)), 0.05e-3, NU, 0.1, g=(0, 0, 0), dispersion=True)
    assert not r0["dispersion"] and np.abs(r0["X"]).max() < 1e-9


def test_zone_crossing_counted_once_for_a_bubble_rising_through_the_face():
    R, H, h, om = 0.01, 0.02, 0.001, 10.0
    C, V = _grid((-R, R), (-R, R), (0, H), h)
    keep = np.hypot(C[:, 0], C[:, 1]) < R
    C, V = C[keep], V[keep]
    U = np.column_stack([-om * C[:, 1], om * C[:, 0], np.zeros(len(C))])
    p = 0.5 * om ** 2 * (C[:, 0] ** 2 + C[:, 1] ** 2)
    zone = np.where(C[:, 2] < 0.01, 0, -1).astype(np.int32)
    m = [{"name": "rot", "cellZone": "rot", "origin": np.zeros(3), "axis": np.array([0, 0, 1.0]), "omega": om}]
    f = bubbles.Field(C, V, U, p, zone, m)
    r = bubbles.track(f, [[0.002, 0.0, 0.004]], 0.3e-3, NU, 0.25, g=(0, 0, -9.81), dt_max=1e-3)
    assert r["X"][0, 2] > 0.011 and r["crossings"][0] == 1 and r["fate"][0] == "flow"


def test_escape_summary():
    om = 2 * math.pi                               # one revolution per second
    n = 10
    fate = np.array(["outlet"] * 4 + ["flow"] * 6, dtype=object)
    t_end = np.array([0.5, 1.5, 3.0, 7.0] + [np.nan] * 6)
    r = {"fate": fate, "reached": np.zeros(n, bool), "t_first": np.full(n, np.nan), "t_in": np.zeros(n),
         "in_at_end": np.zeros(n, bool), "exited": np.zeros(n, bool), "t": np.full(n, 10.0), "t_max": 10.0,
         "t_end": t_end, "crossings_late": np.array([0] * 4 + [0, 1, 12, 30, 2, 0])}
    s = bubbles.summarize(r, om)
    assert s["outlet_time_s"]["median"] == pytest.approx(2.25)
    assert s["escape"]["per_rev_mean"] == pytest.approx(0.04)
    assert s["escape"]["per_rev_hazard"] == pytest.approx(-math.log(0.6) / 10, rel=1e-3)
    assert s["escape"]["out_by_rev"] == {"1": 0.1, "2": 0.2, "5": 0.3, "10": 0.4}
    assert s["zone_crossings_late_free"]["share_ge_10"] == pytest.approx(2 / 6, abs=1e-3)


def test_turbulent_study_hinze_probes_and_escape(tmp_path):
    """The column of test_study_end_to_end with k and omega: Hinze's size where the bubbles end, the time they take to
    leave, probes alone, and dispersion through the runner."""
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
    _write_field(t / "k", "volScalarField", n, "\n".join("1e-4" for _ in V))
    _write_field(t / "omega", "volScalarField", n, "\n".join("50" for _ in V))
    (req / "request.json").write_text(json.dumps({"studies": {"col": {
        "fluid": "water", "bcs": [{"kind": "outlet", "name": "outlet", "faces": {"nz": 1}, "pressure": 0}]}}}))
    want = 0.725 * (0.072 / RHO) ** 0.6 * (0.09 * 1e-4 * 50) ** -0.4 * 1e3
    st = {"bubbles": {"source": str(req / "col"), "diameters_mm": [1.0], "t_max_s": 0.1, "gravity": [0, 0, -9.81],
                      "releases": [{"name": "bottom", "box": {"min": [2, 2, 2], "max": [8, 8, 5]}, "n": 10}],
                      "target": {"name": "mid", "box": {"min": [0, 0, 15], "max": [10, 10, 25]}}, "keep": 2}}
    assert validate_study(st)["ok"]
    r = run_study("trk", st, tmp_path / "out" / "trk")
    assert r["ok"], r
    b = r["results"]["bubbles"]["releases"]["bottom"]["1"]
    assert b["hinze_d_max_mm"]["free_at_end"]["median"] == pytest.approx(want, rel=0.01)
    assert b["hinze_d_max_mm"]["free_at_end"]["share_below_d"] == (1.0 if want < 1.0 else 0.0)
    st_out = dict(st["bubbles"], t_max_s=1.0)
    r1 = run_study("out", {"bubbles": st_out}, tmp_path / "out" / "out")
    b1 = r1["results"]["bubbles"]["releases"]["bottom"]["1"]
    ut = twophase.slip_velocity(9.81, 1e-3, RHO, 0.0, MU)[0]
    assert b1["outlet"] == 1.0 and 0.035 / ut < b1["outlet_time_s"]["median"] < 0.039 / ut
    sp = {"bubbles": {"source": str(req / "col"), "probes": [{"name": "c", "at": [5, 5, 20]}]}}
    assert validate_study(sp)["ok"], validate_study(sp)
    rp = run_study("prb", sp, tmp_path / "out" / "prb")
    pc = rp["results"]["bubbles"]["probes"]["c"]
    assert pc["in_fluid"] and pc["hinze_d_max_mm"] == pytest.approx(want, rel=0.01) and pc["k_m2_s2"] == 1e-4
    sd = {"bubbles": dict(st["bubbles"], dispersion=True)}
    rd = run_study("dsp", sd, tmp_path / "out" / "dsp")
    assert rd["ok"], rd
    assert rd["results"]["bubbles"]["dispersion"]["model"] == "discrete random walk"
    errs = validate_study({"bubbles": dict(st["bubbles"], dispersion="yes", probes=[{"at": [1, 2]}])})["errors"]
    assert any("dispersion" in e for e in errs) and any("probes" in e for e in errs)
