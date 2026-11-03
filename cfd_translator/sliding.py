"""sliding.py — a rotating zone whose cells turn: OpenFOAM 14 non-conformal coupling (NCC), transient.
Version: 2026.11.03.2

An MRF zone freezes the rotor: its results depend on where the blades stand against the cutwater or the
stator vanes, and every rotor-stator interaction is lost. With rotating[].motion 'sliding' the zone's cells turn
with the rotor (a solidBody mover) and slide along the zone's cylinder, where the two sides are coupled
non-conformally. The steps (OpenFOAM's propeller tutorial; cfd.direct, 'Using NCC in OpenFOAM'):

    check(mesh, g, zone, plan, patches)    before the case is written: the zone's boundary is a surface of
                                           revolution in fluid, the rotor's walls are inside it, stationary
                                           ones outside (unless they are surfaces of revolution)
    to_sliding(case, plan, mesh, foam)     once, before the transient run (after the steady MRF start):
                                           faceZone at the zone's boundary -> createBaffles (rotor and stator
                                           patches, slip boundary conditions) -> splitBaffles ->
                                           createNonConformalCouples; MRF replaced by constant/dynamicMeshDict;
                                           the rotor's walls movingWallVelocity

The steady start (solve.transient.init) runs on the conformal mesh with MRF, as before; its solution becomes
time 0 of the sliding run. createBaffles and createNonConformalCouples run in serial before decomposePar (the
mesh is decomposed again for the transient run).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

SCALE = 1e-3


from .polymesh import MeshError


class SlidingError(MeshError):
    pass


def names(zone: str) -> dict:
    """Patch and zone names made for a sliding zone."""
    return {"group": f"{zone}_slide", "face_zone": f"{zone}_interface", "rotor": f"{zone}_rotor",
            "stator": f"{zone}_stator"}


def interface(mesh, cells) -> tuple[np.ndarray, np.ndarray]:
    """The internal faces between the zone's cells and the rest, and their flip (True where the owner is outside
    the zone): createBaffles then gives the zone's side to the 'owner' (rotor) patch."""
    inz = np.zeros(mesh.ncells, bool)
    inz[np.asarray(cells, np.int64)] = True
    ni = mesh.ninternal
    a, b = inz[mesh.owner[:ni]], inz[mesh.neighbour]
    f = np.nonzero(a != b)[0]
    return f, ~a[f]


def _frame(z: dict):
    a = np.asarray(z["axis"], float)
    a = a / np.linalg.norm(a)
    return a, np.asarray(z.get("center", [0, 0, 0]), float)


def _crossing(Cf, Sf, a, o) -> np.ndarray:
    """|sin| of the angle between each face's normal and the plane through the axis (0: the face is part of a
    surface of revolution, which the rotation moves along itself)."""
    t = np.cross(a, Cf - o)
    tn = np.linalg.norm(t, axis=1)
    sn = np.linalg.norm(Sf, axis=1)
    ok = (tn > 1e-9 * max(float(tn.max()) if len(tn) else 0.0, 1e-30)) & (sn > 0)
    out = np.zeros(len(Cf))
    out[ok] = np.abs(np.einsum("ij,ij->i", Sf[ok], t[ok])) / (sn[ok] * tn[ok])
    return out


def _places(Cf, w, n=3) -> str:
    k = np.argsort(-w)[:n]
    return ", ".join(str(np.round(Cf[i], 1).tolist()) for i in k)


def check(mesh, g, zone: dict, plan: dict, patches: list) -> None:
    """Can this zone slide? Raises SlidingError with what to change; small deviations become plan notes.
    Records plan['sliding'] (names, interface size, moving walls)."""
    from .case import _same_rotation, rotation_key
    zn = zone["name"]
    cells = mesh.zones[zn]
    inz = np.zeros(mesh.ncells, bool)
    inz[cells] = True
    a, o = _frame(zone)
    notes = plan.setdefault("notes", [])
    # 1. the boundary between the zone and the rest: a surface of revolution about the axis
    f, _flip = interface(mesh, cells)
    if not len(f):
        raise SlidingError(f"sliding zone {zn!r} has no internal boundary with the rest of the fluid (it fills the "
                           "domain, or touches the rest only through walls): it must be a cylinder in the fluid")
    Sf, Cf = g["Sf"][f], g["Cf"][f]
    A = np.linalg.norm(Sf, axis=1)
    sn = _crossing(Cf, Sf, a, o)
    mean_deg = float(np.degrees(np.arcsin(min((sn * A).sum() / A.sum(), 1.0))))
    steep = sn > np.sin(np.radians(20.0))
    steep_frac = float(A[steep].sum() / A.sum())
    if mean_deg > 5.0 or steep_frac > 0.01:
        raise SlidingError(
            f"sliding zone {zn!r}: its boundary is not a surface of revolution about the axis (faces lean "
            f"{mean_deg:.1f} degrees on average from it, {steep_frac:.1%} of its area more than 20 degrees, e.g. at "
            f"{_places(Cf, sn * A)} mm): the cells could not slide along it. The zone's cylinder must lie in open "
            "fluid, clear of walls (snappyHexMesh snaps its faces to the cylinder only there); check center, "
            "radius and span against the casing, and refine where it passes through narrow gaps")
    plan.setdefault("zones", {}).setdefault(zn, {})
    # 2. the walls: rotating ones inside, stationary ones outside (or surfaces of revolution)
    rev = set((mesh.info.get("revolution_tags") or {}).get(rotation_key(zone["axis"], zone.get("center")), []))
    zp = plan["zones"][zn]
    moving = []
    for p in patches:
        if p["kind"] != "wall":
            continue
        pf = mesh.patch_faces(p["name"])
        if not len(pf):
            continue
        r = p["bc"].get("rotation")
        Aw = np.linalg.norm(g["Sf"][pf], axis=1)
        tot = float(Aw.sum()) or 1.0
        inside = inz[mesh.owner[pf]]
        tags = mesh.btag[pf - mesh.ninternal]
        on_rev = np.isin(tags, list(rev)) if rev else np.zeros(len(pf), bool)
        # faces moving along themselves count as surfaces of revolution too (snapped facets)
        on_rev |= _crossing(g["Cf"][pf], g["Sf"][pf], a, o) < np.sin(np.radians(6.0))
        turns = r == zn or (isinstance(r, dict) and zp.get("omega_rad_s") is not None and _same_rotation(r, zp))
        if turns:
            out = ~inside
            bad = out & ~on_rev
            fb = float(Aw[bad].sum() / tot)
            if fb > 0.01:
                raise SlidingError(
                    f"wall {p['name']!r} turns with sliding zone {zn!r} but {fb:.1%} of its area (not a surface of "
                    f"revolution) bounds cells outside the zone (e.g. at {_places(g['Cf'][pf[bad]], Aw[bad])} mm): "
                    "those faces would stand still while the rest turns. Enlarge the zone (radius, span) so that it "
                    "holds the whole rotor with at least one cell between the rotor's tips and the zone's ends; in a "
                    "narrow side gap, put the zone's end mid-gap and refine the gap (mesh.refine) to 3 cells or more")
            if bad.any():
                notes.append(f"{fb:.2%} of wall {p['name']!r} (rotor) bounds cells just outside sliding zone {zn!r}: "
                             "those faces stand still")
            fo = float(Aw[out & on_rev].sum() / tot)
            if fo > 0.001:
                notes.append(f"{fo:.1%} of wall {p['name']!r} (rotor; surfaces of revolution: hub, shaft, shrouds) lies "
                             f"outside sliding zone {zn!r}: it is treated as standing still there (no disk friction "
                             "from it in the torque)")
            if float(Aw[inside].sum() / tot) > 0.5:
                moving.append(p["name"])
            elif isinstance(r, str):
                raise SlidingError(f"wall {p['name']!r} turns with sliding zone {zn!r} but lies mostly outside it")
        else:
            bad = inside & ~on_rev
            fb = float(Aw[bad].sum() / tot)
            if fb > 0.01:
                raise SlidingError(
                    f"wall {p['name']!r} does not turn with sliding zone {zn!r}, but {fb:.1%} of its area (not a "
                    f"surface of revolution about the axis) bounds cells inside the zone (e.g. at "
                    f"{_places(g['Cf'][pf[bad]], Aw[bad])} mm): it would turn with the zone. The zone (radius, span) "
                    "must hold only the rotor; stationary walls (casing, cutwater, vanes) stay outside it")
            if bad.any():
                notes.append(f"{fb:.2%} of stationary wall {p['name']!r} bounds cells inside sliding zone {zn!r}: "
                             "those faces turn with it")
    nm = names(zn)
    plan["sliding"] = {"zone": zn, **nm, "interface_faces": int(len(f)), "interface_area_mm2": float(A.sum()),
                       "interface_lean_deg": round(mean_deg, 2), "moving_walls": moving,
                       "omega_rad_s": zp.get("omega_rad_s"), "axis": a.tolist(), "center_mm": o.tolist()}
    tr = plan.get("transient") or {}
    if tr.get("average_revolutions") is not None and tr["average_revolutions"] < 3:
        notes.append(f"the averages cover {tr['average_revolutions']} whole revolution(s) "
                     f"({tr['end_time'] - tr['average_from']:.4g} s of {tr['period']:.4g} s each): 3 or more give "
                     "a standard error (raise end_time or lower average_from)")


# ---- the case ------------------------------------------------------------------------------------
def dynamic_mesh_dict(sl: dict) -> str:
    from .case import _dict
    o = np.asarray(sl["center_mm"], float) * SCALE
    ax = sl["axis"]
    rpm = float(sl["omega_rad_s"]) * 60.0 / (2 * np.pi)
    body = (f"mover\n{{\n    type            solidBody;\n\n    libs            (\"libpointMeshMovers.so\");\n\n"
            f"    cellZone        {sl['zone']};\n\n    solidBodyMotionFunction\n    {{\n        type            rotatingMotion;\n\n"
            f"        origin          ({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});\n"
            f"        axis            ({ax[0]:.12g} {ax[1]:.12g} {ax[2]:.12g});\n\n"
            f"        omega           {rpm:.12g} [rpm];\n    }}\n}}")
    return _dict("dynamicMeshDict", "constant", body)


SOLVED = ("U", "p")
SOLVED_TURB = ("U", "p", "k", "omega", "nut")


def _turbulent(plan: dict) -> bool:
    return (plan.get("transient") or {}).get("model", plan.get("model", "laminar")) != "laminar"


def create_baffles_dict(sl: dict, turb: bool) -> str:
    from .case import _dict
    pf = ("        U\n        {\n            type            movingWallSlipVelocity;\n            value           uniform (0 0 0);\n        }\n"
          "        p\n        {\n            type            zeroGradient;\n        }\n")
    if turb:
        pf += ("        k\n        {\n            type            zeroGradient;\n        }\n"
               "        omega\n        {\n            type            zeroGradient;\n        }\n"
               "        nut\n        {\n            type            zeroGradient;\n        }\n")
    pf = "\n".join("        " + ln if ln else ln for ln in pf.splitlines())

    def side(key, name):
        return (f"            {key}\n            {{\n                name            {name};\n                type            patch;\n\n"
                f"                patchFields\n                {{\n{pf}\n                }}\n            }}\n")
    body = (f"internalFacesOnly true;\n\nfields          true;\n\nbaffles\n{{\n    {sl['group']}\n    {{\n"
            f"        type            faceZone;\n        zoneName        {sl['face_zone']};\n\n        patches\n        {{\n"
            + side("owner", sl["rotor"]) + "\n" + side("neighbour", sl["stator"]) + "        }\n    }\n}")
    return _dict("createBafflesDict", "system", body)


def write_face_zone(poly: Path, name: str, faces: np.ndarray, flip: np.ndarray) -> None:
    from .polymesh import header
    lines = [header("faceZoneList", "faceZones", "constant/polyMesh"), "1", "(", name, "{", "    type faceZone;",
             f"    faceLabels List<label> {len(faces)}", "("]
    lines += [str(int(i)) for i in faces]
    lines += [")", ";", f"    flipMap List<bool> {len(flip)}", "("]
    lines += ["1" if b else "0" for b in flip]
    lines += [")", ";", "}", ")", "", "// " + "*" * 73 + " //", ""]
    (Path(poly) / "faceZones").write_text("\n".join(lines))


def set_patch_type(text: str, patch: str, new_type: str) -> tuple[str, bool]:
    """A field file with patch's boundary condition type replaced (its value kept)."""
    m = re.search(r"(?m)^\s*" + re.escape(patch) + r"\s*\n\s*\{", text)
    if not m:
        return text, False
    t = re.compile(r"type\s+[^;]+;").search(text, m.end())
    if not t:
        return text, False
    return text[:t.start()] + f"type            {new_type};" + text[t.end():], True


def _boundary(poly: Path) -> list[dict]:
    from .foamio import read_boundary
    return read_boundary((Path(poly) / "boundary").read_text(errors="replace"))


DECOMPOSED = "cfdt_decomposed.json"


def to_sliding(case: Path, plan: dict, mesh, run_step, procs: int = 1, mpirun: str | None = None) -> list[str]:
    """Turns the case into a sliding-mesh case (see the module's notes). run_step(command, log) runs an OpenFOAM
    utility in the case (raising on failure). With procs > 1 the conformal case is decomposed first and the three
    utilities run in parallel (as OpenFOAM's propeller tutorial does): decomposePar on a mesh that already has the
    couple loops over every pair of faces of the two sides (domainDecompositionDecompose.C, processInterCyclics),
    which for s180 (1.24 M faces a side) never finished. The case then stays decomposed; foam.read_fields reads the
    processor directories (cfdt_decomposed.json). Returns notes."""
    import json
    case = Path(case)
    sl = plan["sliding"]
    zero = case / "0"
    # the solution of the steady start, without what does not survive the topology change
    for d in sorted(case.glob("processor*")):
        shutil.rmtree(d, ignore_errors=True)                # decomposed again for the transient run
    (case / DECOMPOSED).unlink(missing_ok=True)
    # createBaffles (fields true) reads every field in 0/ and needs a boundary condition for each on the new
    # patches: only the solved fields stay; the rest (phi, wallShearStress, yPlus, the monitors' total(p): s180 on
    # spark2, 2026.11.02, "keyword total(p) is undefined") go to 0.steady/, and the transient run writes its own
    keep = set(SOLVED_TURB if _turbulent(plan) else SOLVED)
    aside = case / "0.steady"
    for f in sorted(zero.iterdir()) if zero.is_dir() else []:
        if f.is_file() and f.name not in keep:
            aside.mkdir(exist_ok=True)
            shutil.move(str(f), str(aside / f.name))
    # the rotor's walls move with the mesh
    uf = zero / "U"
    txt = uf.read_text()
    for w in sl["moving_walls"]:
        txt, ok = set_patch_type(txt, w, "movingWallVelocity")
        if not ok:
            raise SlidingError(f"0/U has no entry for wall {w!r}")
    uf.write_text(txt)
    # MRF off, the mover on
    (case / "constant" / "MRFProperties").unlink(missing_ok=True)
    (case / "constant" / "dynamicMeshDict").write_text(dynamic_mesh_dict(sl))
    # the interface: faceZone -> baffles -> split -> non-conformal couple
    poly = case / "constant" / "polyMesh"
    f, flip = interface(mesh, mesh.zones[sl["zone"]])
    write_face_zone(poly, sl["face_zone"], f, flip)
    (case / "system" / "createBafflesDict").write_text(create_baffles_dict(sl, _turbulent(plan)))
    before = {p["name"]: p for p in mesh.patches}
    par = ""
    if procs > 1:
        (case / "system" / "decomposeParDict").write_text(
            "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      decomposeParDict;\n}\n\n"
            f"numberOfSubdomains {procs};\n\ndecomposer      scotch;\n")
        run_step("decomposePar -force", "log.decomposePar")
        # the conformal decomposition's addressing: cells keep it through the topology change, and each original
        # patch keeps its faces in order, so the results can be put together without reconstructPar
        for k in range(procs):
            d = case / f"processor{k}"
            src = d / "constant" / "polyMesh"
            dst = d / "cfdt_addressing"
            dst.mkdir(exist_ok=True)
            for nm in ("cellProcAddressing", "faceProcAddressing", "boundary"):
                if not (src / nm).is_file():
                    raise SlidingError(f"decomposePar wrote no {src / nm}")
                shutil.copyfile(src / nm, dst / nm)
        par = f"{mpirun or 'mpirun'} -np {procs} "
    run_step(f"{par}createBaffles" + (" -parallel" if par else ""), "log.createBaffles")
    run_step(f"{par}splitBaffles" + (" -parallel" if par else ""), "log.splitBaffles")
    run_step(f"{par}createNonConformalCouples" + (" -parallel" if par else "") +
             f" -fields {sl['rotor']} {sl['stator']}", "log.createNonConformalCouples")
    # the patches the results read must be the same faces in the same order (createBaffles keeps their order)
    if par:
        after = {}
        for k in range(procs):
            for p in _boundary(case / f"processor{k}" / "constant" / "polyMesh"):
                after.setdefault(p["name"], {"name": p["name"], "n": 0})["n"] += p["n"]
    else:
        after = {p["name"]: p for p in _boundary(poly)}
    for nm, p in before.items():
        q = after.get(nm)
        if q is None or q["n"] != p["n"]:
            raise SlidingError(f"after createBaffles patch {nm!r} has {q['n'] if q else 'no'} faces instead of {p['n']}")
    for nm in (sl["rotor"], sl["stator"]):
        if after.get(nm, {}).get("n", 0) != len(f):
            raise SlidingError(f"createBaffles made {after.get(nm, {}).get('n', 0)} faces on {nm!r}, expected {len(f)}")
    sl["patches_added"] = sorted(n for n in set(after) - set(before) if not n.startswith(("procBoundary", "nonConformalProcessorCyclic")))
    if par:
        (case / DECOMPOSED).write_text(json.dumps({"procs": procs, "why": "sliding mesh: the couple is built on the "
                                                  "decomposed case; results are read from processor*"}))
        sl["decomposed"] = procs
    return [f"sliding mesh: zone {sl['zone']!r} turns at {sl['omega_rad_s'] * 60 / (2 * np.pi):g} rpm; {len(f)} faces "
            f"on its cylinder coupled non-conformally ({sl['rotor']} | {sl['stator']})" +
            (f", built on the case decomposed for {procs} ranks (it stays decomposed)" if par else "")]
