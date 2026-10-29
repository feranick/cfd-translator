"""Porous-zone reactions as forces on the part (2026.10.28.2): forces.porous, forces.part and the coefficients
with them, cd_walls_only, on_part, feat_loads, the per-step force history (OpenFOAM's porous columns, or a
constant from the fields), and reprocessing a finished study."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cfd_translator import api, bench
from cfd_translator.average import DEFAULTS, _force_series, average_results, monitor_functions
from cfd_translator.post import results
from cfd_translator.render import save_state
from cfd_translator.runner import prepare

UP, RHO, MU, D, F = 0.01, 1000.0, 1e-3, 1e8, 1e3
DP = 0.05 * (MU * D * UP + 0.5 * RHO * F * UP ** 2)      # 52.5 Pa over the 50 mm block
A = 10e-3 * 1e-3                                           # the channel's section, m2


def _setup(tmp_path, **zone):
    """The porous channel benchmark's block, prepared (no solve), with the exact uniform flow and linear pressure
    drop across the block; the study is then called external, so the block alone is 'the part' (no walls)."""
    st = bench._cases()["porous_channel_1d"]["study"] | {
        "porous": [dict({"name": "bed", "region": {"box": [-25, -10, -1, 25, 10, 2]}, "darcy": D, "forchheimer": F}, **zone)]}
    mesh, patches, sel, plan, w = prepare(st, tmp_path)
    g = mesh.geometry()
    x = g["C"][:, 0]
    U = np.tile([UP, 0, 0], (mesh.ncells, 1))
    p = np.where(x < -25, DP, np.where(x > 25, 0.0, DP * (25 - x) / 50)) / RHO          # kinematic
    pat = {}
    for P in mesh.patches:
        f = np.arange(P["start"], P["start"] + P["n"])
        kind = plan["patches"][P["name"]]["kind"]
        pb = {"inlet": np.full(len(f), DP / RHO), "outlet": np.zeros(len(f))}.get(kind, p[mesh.owner[f]])
        pat[P["name"]] = {"U": np.tile([UP, 0, 0], (len(f), 1)), "p": pb, "phi": g["Sf"][f] @ [UP, 0, 0] * 1e-6,
                          "wss": None, "nut": None}
    ext = dict(st, domain={"kind": "external", "box": [-100, -5, 0, 100, 5, 1]},
               reference={"velocity": UP, "area": 10.0, "length": 50.0})
    return ext, mesh, plan, {"U": U, "p": p, "patches": pat}


def test_the_reaction_is_the_part_drag_and_equals_dp_times_area(tmp_path):
    st, mesh, plan, fields = _setup(tmp_path)
    res, warns = results(st, mesh, plan, fields)
    fp = res["forces"]["porous"]["bed"]
    assert fp["fx"] == pytest.approx(DP * A, rel=1e-9) and fp["on_part"] and abs(fp["fy"]) < 1e-15
    part = res["forces"]["part"]
    assert part["fx"] == pytest.approx(DP * A, rel=1e-9) and part["porous_included"] == ["bed"]
    assert part["walls"]["fx"] == 0                                    # no walls: the block alone
    q = 0.5 * RHO * UP ** 2
    co = res["coefficients"]
    assert co["cd"] == pytest.approx(DP * A / (q * 1e-5), rel=1e-6) and co["cd_walls_only"] == 0
    assert res["porous"]["bed"]["pressure_drop_Pa"] * A == pytest.approx(part["fx"], rel=0.02)   # the prompt's test
    # the momentum balance holds the porous source: in - out = boundary forces + reactions
    mb = res["checks"]["momentum_balance"]
    assert mb["error"] < 1e-6 and mb["porous_reaction_N"][0] == pytest.approx(DP * A)
    assert res["checks"]["porous_on_part"] == {"included": ["bed"], "excluded": []}
    assert res["feat_loads"]["total"][0] == pytest.approx(DP * A)       # no part face to carry it here
    # the reaction two other ways: per frontal area, and from the pressure and momentum through the zone's faces
    pz = res["porous"]["bed"]
    assert pz["frontal_area_m2"] == pytest.approx(A) and pz["pressure_drop_from_force_Pa"] == pytest.approx(DP)
    assert pz["face_balance"]["vs_force"] < 0.01 and pz["face_balance"]["reaction_N"][0] == pytest.approx(DP * A, rel=0.01)
    assert not any("differ" in w for w in warns)


def test_a_pressure_drop_that_does_not_match_the_force_is_flagged(tmp_path):
    st, mesh, plan, fields = _setup(tmp_path)
    flat = dict(fields, p=np.zeros(mesh.ncells),
                patches={n: dict(v, p=np.zeros(len(v["p"]))) for n, v in fields["patches"].items()})
    res, warns = results(st, mesh, plan, flat)
    pz = res["porous"]["bed"]
    assert pz["pressure_drop_Pa"] == 0 and pz["pressure_drop_from_force_Pa"] == pytest.approx(DP)
    assert pz["face_balance"]["vs_force"] == pytest.approx(1, abs=1e-6)
    assert any("'bed'" in w and "differ" in w for w in warns)
    assert res["forces"]["part"]["fx"] == pytest.approx(DP * A)        # the drag keeps the applied force


def test_on_part_false_leaves_it_out_with_a_warning(tmp_path):
    st, mesh, plan, fields = _setup(tmp_path, on_part=False)
    assert plan["porous"]["bed"]["on_part"] is False
    res, warns = results(st, mesh, plan, fields)
    assert "part" not in res["forces"] and res["forces"]["porous"]["bed"]["on_part"] is False
    assert res["checks"]["porous_on_part"]["excluded"] == ["bed"] and any("on_part: false" in w for w in warns)
    # internal flow: reported, never added to the walls
    st_int = dict(st, domain={"kind": "internal"})
    res, _ = results(st_int, mesh, plan | {"porous": {"bed": dict(plan["porous"]["bed"], on_part=True)}}, fields)
    assert res["forces"]["porous"]["bed"]["on_part"] is False and "porous_on_part" not in res["checks"]


def test_validation_of_on_part():
    from cfd_translator.format import validate_study
    st = bench._cases()["porous_channel_1d"]["study"]
    z = st["porous"][0]
    assert validate_study(dict(st, porous=[dict(z, on_part=False)]))["ok"]
    assert "on_part" in " ".join(validate_study(dict(st, porous=[dict(z, on_part="no")]))["errors"])


def test_monitor_asks_openfoam_for_the_porous_force(tmp_path):
    st, mesh, plan, fields = _setup(tmp_path)
    p2 = dict(plan, body_walls=["body"])
    st2 = dict(st, solve={"average": True})
    funcs, mon = monitor_functions(st2, p2, RHO)
    body = next(f for f in funcs if "cfdtF_body" in f)
    assert "porosity        yes;" in body and mon["forces"]["porosity"] is True
    p3 = dict(p2, porous={"bed": dict(plan["porous"]["bed"], on_part=False)})
    funcs, mon = monitor_functions(st2, p3, RHO)
    assert "porosity" not in next(f for f in funcs if "cfdtF_body" in f) and mon["forces"]["porosity"] is False


def _forces_dat(case, rows):
    d = case / "postProcessing" / "cfdtF_body" / "0"
    d.mkdir(parents=True)
    (d / "forces.dat").write_text("# Time forces(pressure viscous porous) moments(pressure viscous porous)\n" +
                                  "\n".join(rows) + "\n")


def test_force_history_with_porous_columns_and_with_a_constant(tmp_path):
    # OpenFOAM 14 with porosity: one file, 18 numbers; the porous part is kinematic (rhoRef 1)
    case = tmp_path / "c1"
    _forces_dat(case, [f"{i} ((1 0 0) (0.5 0 0) (0.002 0 0)) ((0 0 0) (0 0 0) (0 0 0))" for i in range(1, 4)])
    its, Fh = _force_series(case, "cfdtF_body", RHO)
    assert Fh[0] == pytest.approx([1 + 0.5 + 0.002 * RHO, 0, 0])
    # a run whose monitor held the walls only: the porous reaction from the fields is added as a constant
    case = tmp_path / "c2"
    d = case / "postProcessing" / "cfdtF_body" / "0"
    d.mkdir(parents=True)
    n = 2000
    rng = np.random.default_rng(1)
    fx = 1.0 + 1e-4 * rng.standard_normal(n)
    (d / "forces.dat").write_text("# Time forces(pressure viscous) moments(pressure viscous)\n" + "\n".join(
        f"{i + 1} (({fx[i]:.9g} 0 0) (0 0 0)) ((0 0 0) (0 0 0))" for i in range(n)) + "\n")
    q, Aref = 0.5 * RHO * UP ** 2, 1e-5
    res = {"forces": {"part": {"fx": 1.5, "fy": 0, "fz": 0, "porous_included": ["bed"]},
                      "porous": {"bed": {"fx": 0.5, "fy": 0, "fz": 0}}},
           "coefficients": {"cd": 1.5 / (q * Aref), "cl": 0.0, "drag_dir": [1, 0, 0], "lift_dir": [0, 1, 0],
                            "q_Pa": q, "area_m2": Aref, "cd_walls_only": 1.0 / (q * Aref)}}
    plan = {"rho": RHO, "patches": {}, "monitors": {"settings": dict(DEFAULTS), "pressure": {}, "torque": {},
                                                    "forces": {"fo": "cfdtF_body", "patches": ["body"]}}}
    avg, w = average_results(case, plan, res)
    assert avg["quantities"]["cd"]["mean"] == pytest.approx(1.5 / (q * Aref), rel=1e-4)
    assert "constant" in avg["porous_zones"] and res["coefficients"]["cd"] == pytest.approx(1.5 / (q * Aref), rel=1e-4)


def test_reprocess_puts_the_reaction_into_an_old_result(tmp_path, monkeypatch):
    root = tmp_path / "out"
    monkeypatch.setenv("CFDT_OUT", str(root))
    req = root / "run_20261025_000000_abcdef"
    sd = req / "car"
    sd.mkdir(parents=True)
    st, mesh, plan, fields = _setup(tmp_path / "prep")
    save_state(sd, mesh, plan, fields, st)
    (req / "request.json").write_text(json.dumps({"names": ["car", "other"]}))
    old = {"name": "car", "ok": True, "translator": "2026.10.25.6", "solver": "foam",
           "solve": {"converged": True}, "warnings": ["averaging: something old", "kept"],
           "results": {"coefficients": {"cd": 0.0, "cl": 0.0}, "forces": {}}}
    (sd / "car.json").write_text(json.dumps(old))
    r = api.reprocess(str(req))
    car = r["studies"]["car"]
    assert car["ok"] and car["fields"] == "the saved state" and r["studies"]["other"]["ok"] is False
    new = json.loads((sd / "car.json").read_text())
    q = 0.5 * RHO * UP ** 2
    assert new["results"]["coefficients"]["cd"] == pytest.approx(DP * A / (q * 1e-5), rel=1e-6)
    assert new["reprocessed"]["from"] == "2026.10.25.6" and "cd" in new["reprocessed"]["changes"]
    assert new["translator"] == "2026.10.25.6" and "kept" in new["warnings"]
    assert not any(x.startswith("averaging") for x in new["warnings"])
    assert (sd / "car.json.2026.10.25.6.bak").is_file()
    assert "error" in api.reprocess(str(tmp_path))                                  # not a request under CFDT_OUT
    assert "error" in api.reprocess(str(req), ["nope"])
