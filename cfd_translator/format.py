"""format.py — the study format: geometry, domain, fluid, model, boundary conditions, validation.
Version: 2026.11.12.1

A study file is {name: study}; a study is one steady incompressible flow:

    {"geometry": {"goal": <build123d-translator goal>} | {"step": path} | {"box": [lx, ly, lz]} | {"cylinder": [r, h]},
     "domain": {"kind": "internal"} | {"kind": "external", "box": [x0, y0, z0, x1, y1, z1]},
     "2d": false,
     "fluid": "air" | "water" | {"rho": kg/m3, "mu": Pa s},
     "model": "laminar" | "kOmegaSST",
     "bcs": [{"kind": "inlet", "faces": {...}, "velocity": 2.0}, {"kind": "outlet", "faces": {...}}, ...],
     "mesh": {"size": mm, "refine": [...], "cells": [nx, ny, nz], "extrude": n, "layers": {...}},
     "solve": {"iterations": 2000, "residual": 1e-5},
     "reference": {"velocity": m/s, "area": mm2, ...},
     "probes": [{"name": "p1", "at": [x, y, z]}],
     "expect": [{"quantity": "pressure_drop", "value": 12.0, "tol": 0.02}]}

Geometry in mm (Z up, base on z = 0, centred in X/Y, as build123d-translator); everything else SI.
"""
from __future__ import annotations

import re

FLUIDS = {
    "air": {"rho": 1.204, "mu": 1.825e-5, "cp": 1005.0, "pr": 0.71, "molar_mass": 28.96,
            "note": "20 degrees C, 1 atm; with energy: perfect gas (density from p and T)"},
    "water": {"rho": 998.2, "mu": 1.002e-3, "cp": 4182.0, "pr": 7.0,
              "note": "20 degrees C; with energy: constant density, temperature solved"},
}
R_UNIVERSAL = 8314.462618          # J/(kmol K)
MODELS = ("laminar", "kOmegaSST", "kOmegaSSTDES")
TRANSIENT_KEYS = {"end_time", "average_from", "max_co", "write_interval", "init", "max_jobs", "dt", "continue",
                  "degrees_per_step", "correct_mesh_phi", "init_from", "init_fields"}
PROFILES = ("uniform", "parabolic")
BOX_SIDES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
RESERVED = ("walls", "frontAndBack")

# kind: (required, optional, description)
BC_KINDS = {
    "inlet": ({"faces": ""},
              {"velocity": "m/s: a number (normal into the domain) or [vx, vy, vz]",
               "flow_rate": "m3/s (instead of velocity)", "flow_rate_lpm": "L/min (instead of velocity)",
               "profile": "uniform (default) | parabolic (fully developed laminar)",
               "intensity": "turbulence intensity (kOmegaSST; default 0.05)",
               "length_scale": "mm (default 0.07 x hydraulic diameter)",
               "swirl": "{axis, center mm, rpm (solid body) or tangential m/s}: added tangential velocity (swirlers)",
               "mass_flow": "kg/s (energy only; instead of velocity)",
               "temperature": "degrees C (energy only; default conditions.T)",
               "total_pressure": "Pa gauge (energy, gases: instead of velocity; isentropic)",
               "total_temperature": "degrees C (with total_pressure; default conditions.T)"},
              "Fixed velocity in (pressure zero-gradient)."),
    "outlet": ({"faces": ""}, {"pressure": "Pa (static, default 0; gauge to conditions.p with energy)",
                               "temperature": "degrees C of any backflow (energy; default conditions.T)"},
               "Fixed static pressure; velocity zero-gradient, no backflow."),
    "opening": ({"faces": ""}, {"pressure": "Pa (static, default 0; gauge to conditions.p with energy)",
                                "temperature": "degrees C of inflow (energy; default conditions.T)"},
                "Fixed static pressure; flow may go in or out (open ends, far-field)."),
    "wall": ({"faces": ""}, {"velocity": "m/s [vx, vy, vz]: moving wall (tangential part)",
                             "rotation": "a rotating zone's name or {axis, center mm, rpm}: rotating wall (u = omega x r)",
                             "T": "energy: wall temperature, degrees C", "q": "energy: heat flux into the fluid, W/m2",
                             "power": "energy: total heat into the fluid, W",
                             "h": "energy: film coefficient to T_inf, W/(m2 K)",
                             "T_inf": "energy: outside temperature for h, degrees C"},
             "No-slip wall (every face no bc takes is a wall)."),
    "slip": ({"faces": ""}, {}, "Free-slip wall: no friction, no flow through (far-field sides)."),
    "symmetry": ({"faces": ""}, {}, "Symmetry plane."),
}

SELECTOR_DOC = {
    "nx, ny, nz": "a planar face with this outward normal (others default to 0): the largest, then the farthest "
                  "along the normal; with at = [x, y, z] the one nearest to that point",
    "pts": "[[x, y, z], ...]: the face nearest to each point (a point on the face, away from its edges)",
    "cyl_r": "every cylindrical face of this radius; side: 'inner' (holes, bores) or 'outer' (shafts, bosses); "
             "with at, only the nearest",
    "all": "true: every face of the part",
    "box": "external flow only: sides of the domain box, 'xmin' ... 'zmax' or a list",
    "group": "a face group of an stl geometry (STL solid, OBJ group, geometry.groups): name or list",
    "solid": "every face of body k (0-based) of a part of several bodies",
    "in_box": "[x0, y0, z0, x1, y1, z1]: part faces with centroid inside (snappy splits faces at the box)",
    "in_cyl": "{axis, center, r, span [s0, s1] or x/y/z [lo, hi], r_in}: as in_box, a cylinder (wheels)",
}

CONVENTIONS = [
    "Units: geometry, coordinates, mesh sizes, areas in mm (mm2); velocity m/s; pressure Pa; density kg/m3; "
    "viscosity Pa s; flow rate m3/s (or L/min with flow_rate_lpm); forces N; moments N m.",
    "Geometry: a build123d-translator goal, STEP (mm), a closed triangle mesh (stl, snappy only), a box [lx, ly, lz], "
    "cylinder [r, h] or annulus [r_in, r_out, h] (axis Z, base z = 0, centred in X/Y). Several bodies: external flow.",
    "domain internal (default): the geometry is the fluid (a channel, a manifold). external: the fluid is the "
    "domain box [x0, y0, z0, x1, y1, z1] minus the geometry (flow around a part).",
    "Faces: the part's selectors (as in build123d-translator and fea-translator; external flow: the solid's "
    "normals), {'box': 'xmin'} for external domain sides; a list is a union. Faces no bc takes: no-slip 'walls'.",
    "2d: true makes a one-cell-thick mesh along Z with 'empty' front and back (OpenFOAM 2D): the domain must be "
    "a prism along Z (one bottom face, one top face, vertical sides). Results are for the modelled thickness.",
    "The flow needs a driver (inlet, opening or moving wall); an inlet needs an outlet or opening. inlet "
    "velocity as a number is normal into the domain; 'parabolic' gives the fully developed laminar profile "
    "with that mean velocity (or flow rate).",
    "Mesh: box + cells (+ grading): structured hexes. Gmsh: size (mm, default diagonal / 30); tetrahedra, or extrude "
    "n prisms along Z (2d: 1) with quads and layers. mesher snappy (3D, aero): hexes of size, surface mm on the part "
    "([finest, coarsest], size/8), features, gap (extra levels in narrow gaps, 1), layers on the part's walls (or "
    "layers.on) with coverage and y+ in mesh.layers; zones snapped; part faces on a domain side (half model) cut. "
    "refine: [{size, and at + r | box | cylinder {center, axis, radius, inner_radius, span} | distance (mm)}].",
    "kOmegaSST uses wall functions (y+ 30-300 at the first cell is ideal; y_plus is reported).",
    "Results: flow.<bc>.Q (m3/s, + in), Q_lpm, U_mean; pressure.<bc>.p_mean, p_total; pressure_drop (static), "
    "total_pressure_drop, head_loss_m (losses); forces.<wall bc>.fx/fy/fz (N) and mx/my/mz about reference.cofr; "
    "forces.part (external; porous reactions on it unless on_part false); forces_by_solid; coefficients.cd/cl; "
    "probes; max_velocity; wall_shear; y_plus; feat_loads (external: fea-translator bcs); checks.",
    "energy: true solves temperature (OpenFOAM fluid, to transonic): air a perfect gas, water constant density; "
    "degrees C; pressures gauge to conditions.p (101325). Walls adiabatic unless T, q (W/m2 into the fluid), power (W) "
    "or h + T_inf; gravity adds buoyancy. Results add heat.<wall bc>.Q_W/q_mean/T_mean_C (h, Nu with reference.T, "
    "length), flow.<bc>.mdot/T_mean_C/mach, max_mach, T_min_C/T_max_C, checks.energy_balance, feat_thermal (external).",
    "Vehicles, bluff bodies: read guide('vehicle_aero') (MCP tool) first: domain, snappy sizes, layers, DES windows.",
    "Pumps, fans: a rotating zone (MRF) around the impeller, its faces a wall with rotation: the zone's name. "
    "Results: rotating.<zone>.torque_Nm, power_W; with an inlet and an outlet pump.head_m, efficiency.",
    "Sliding mesh (no frozen rotor): rotating[].motion 'sliding', mesher snappy, solve.transient; cavitation and "
    "NPSH curves: cavitation {npsh}; read guide('pumps') first.",
    "Averaging (solve.average; on with a rotating zone, external turbulent flow with a reference, or snappy): "
    "results.averaged: mean, std, drift, scatter of torque, total pressures, cd, cl over the last 25 % (>= 500 "
    "its); the means replace the results (*_last: last iteration). settled (drift < 1 %, scatter < 2 %; "
    "transient: drift < 2 %, standard error < 1 %) counts as converged. Other windows: study_status(window).",
    "reliable: false (with reasons) when the solver neither converged nor settled, or a balance is off: do not "
    "report the numbers; rerun with more iterations, lower relax or a better mesh.",
    "expect: [{quantity, value, tol}] turns the run into a check: quantity is a dotted path into the results "
    "(pressure_drop, flow.inlet.Q, coefficients.cd, probes.line.ux.4); tol is "
    "relative (default 0.02), or give abs (absolute).",
]

EXAMPLE = {"pipe_laminar": {
    "geometry": {"cylinder": [5, 200]},
    "fluid": "water",
    "model": "laminar",
    "bcs": [{"kind": "inlet", "name": "inlet", "faces": {"nz": -1}, "flow_rate_lpm": 0.5, "profile": "parabolic"},
            {"kind": "outlet", "name": "outlet", "faces": {"nz": 1}, "pressure": 0}],
    "mesh": {"size": 0.8, "extrude": 80},
    "expect": [{"quantity": "pressure_drop", "value": 6.80, "tol": 0.03}]}}

STUDY_KEYS = {
    "geometry": "{'goal': b3dt goal} | {'step': path} | {'box': [lx, ly, lz]} | {'cylinder': [r, h]} | {'annulus': "
                "[r_in, r_out, h]} | {'stl': path, 'scale', 'groups'} | "
                "{'ref': name (add_geometry) or '<request>/<study>'}",
    "domain": "{'kind': 'internal'} (default) | {'kind': 'external', 'box': [x0, y0, z0, x1, y1, z1]}",
    "2d": "true: one cell along Z, empty front/back",
    "fluid": f"one of {sorted(FLUIDS)} or {{rho, mu}}; with energy {{mu, cp, pr (or k), molar_mass (gas) or rho "
             "(liquid)}} (or a name plus overrides)",
    "model": "|".join(MODELS),
    "bcs": "list of boundary conditions",
    "mesh": "{size, refine, cells: [nx, ny, nz], grading, extrude, quads, layers: {first | y_plus, n, growth, on, "
            "feature_angle}, mesher: gmsh | snappy, surface, features, gap, quality: {max_non_ortho, max_skewness}, "
            "max_cells, optimize}",
    "solve": "{iterations (2000; 3000 kOmegaSST), residual (1e-5), relax: {U, other, p}, procs, energy: transonic, "
             "consistent, start, ramp; average (true | false | {window, min_iterations, drift, scatter}); "
             "transient: {end_time s, average_from, max_co (null: fixed dt), write_interval, init (steady its first), max_jobs, "
             "continue, degrees_per_step (sliding mesh)}: time averages; needed by kOmegaSSTDES; init_from "
             "('<request>/<study>' or '<study>' of this request, run first: its fields are the start, steady or "
             "transient), init_fields (last | mean)}",
    "reference": "{velocity (m/s), area (mm2), length (mm), cofr [x, y, z] mm, drag_dir, lift_dir, p (Pa, gauge zero)}",
    "probes": "[{name, at: [x, y, z]} | {name, line: {from, to, n}}]",
    "expect": "[{quantity, value, tol | abs}]",
    "rotating": "[{name, axis, center (mm), rpm, radius (mm), span: [s0, s1] mm along the axis, motion: mrf | "
                "sliding}]: MRF zones (default) or a sliding mesh (transient, snappy)",
    "images": "true or [{name, slice: {axis, at mm}, field: u|ux|uy|uz|p|cp, streamlines, window: [a0,a1,b0,b1]} "
              "| {name, surface: part|walls|<bc>, field: p|cp|wall_shear, view: [elev, azim], quality: high} | "
              "{name, iso: {field: q|u|cp_total, value|auto}, color: u|p|cp, window: [x0,x1,..,z1], view}], "
              "clim: [lo, hi]|full|robust: PNGs (get_image; render_images)",
    "energy": "true: solve temperature (see conventions); default false",
    "conditions": "{p (Pa absolute, 101325), T (C, 20)}: operating pressure, initial/ambient temperature (energy)",
    "gravity": "[gx, gy, gz] m/s2 (energy): buoyancy, natural convection (e.g. [0, 0, -9.81])",
    "queue": "true: always a SLURM job; false: the server's own workers (default: by time estimate)",
    "cavitation": "{npsh: m | [m, ...] (a sweep: NPSH3; chain: each point from the previous), temperature C | p_vapour Pa, model, iterations}: liquid + "
                  "vapour (VoF); guide('pumps')",
    "run": "{timeout_s, mem ('8G'), procs, queue}: overrides of the machine's defaults, within its caps (status)",
    "mesh_scales": "[1, 0.707, 0.5]: a study per factor on all mesh sizes (<name>_L1..; mesh convergence in "
                   "'variants'); or variants: {suffix: {changes}}",
    "analysis": "steady",
    "scalars": "[{name, inlets: {<inlet bc>: value} (others 0), Sc (1) | diffusivity m2/s, Sct (0.7)}]: passive "
               "tracers (mixing); results.scalars.<name>.outlets.<bc>.mean/mixing_index, volume_mean",
    "age": "true | {Sc, Sct}: mean age of fluid (s); results.age: mean_residence_time_s, V_over_Q_s, dead zones",
    "porous": "[{name, region: {box: [x0,y0,z0,x1,y1,z1]} | {cylinder: {center, axis, radius, span}} | {sphere: "
              "{center, radius}} (mm), darcy 1/m2, forchheimer 1/m (number | [e1,e2,e3]) | ergun: {particle_diameter mm, "
              "porosity} | monolith: {channel mm, open_fraction, shape}, axes: [[e1],[e2]], on_part}]; results.porous",
}
MESH_KEYS = {"size", "refine", "cells", "grading", "extrude", "quads", "layers", "mesher", "surface", "features",
             "gap", "quality", "max_cells", "optimize"}
LAYER_KEYS = {"n", "first", "y_plus", "growth", "on", "feature_angle", "at_openings"}
SOLVE_KEYS = {"iterations", "residual", "relax", "procs", "transonic", "consistent", "start", "ramp", "average",
              "transient", "init_from", "init_fields"}
AVERAGE_KEYS = {"window", "min_iterations", "drift", "scatter"}
REF_KEYS = {"velocity", "area", "length", "cofr", "drag_dir", "lift_dir", "p", "T"}
THERMAL_KEYS = ("T", "q", "power", "h", "T_inf")


def study_format() -> dict:
    bcs = {k: {"required": list(r), "optional": list(o) + ["name"], "what": w,
               "descriptions": {p: d for p, d in {**r, **o}.items() if d}}
           for k, (r, o, w) in BC_KINDS.items()}
    return {"format": "CFD study (cfd-translator)", "analyses": ["steady"], "models": list(MODELS),
            "conventions": CONVENTIONS, "study_keys": STUDY_KEYS, "fluids": FLUIDS, "bc_kinds": bcs,
            "selectors": SELECTOR_DOC, "example": EXAMPLE}


# ---- validation --------------------------------------------------------------------------------
def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vec(v, n=3) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(_num(x) for x in v)


def _stl_errors(g: dict, st: dict) -> list[str]:
    """A mesh geometry: a readable closed surface, meshed by snappy; several bodies only for external flow."""
    errs = []
    if not isinstance(g["stl"], str) or not g["stl"].lower().endswith((".stl", ".obj", ".ply")):
        return ["geometry.stl: the path of a closed triangle mesh (.stl, .obj or .ply, mm unless scale)"]
    if "scale" in g and (not _num(g["scale"]) or g["scale"] <= 0):
        errs.append("geometry.scale: a positive factor to mm (1000 for a model in metres)")
    if "groups" in g and not isinstance(g["groups"], (str, dict, list)):
        errs.append("geometry.groups: {name: [triangle indices]}, a list of names (one per triangle) or a JSON file")
    if ((st.get("mesh") or {}).get("mesher")) != "snappy":
        errs.append("geometry.stl: a triangle mesh is meshed with snappyHexMesh: set mesh.mesher 'snappy'")
    from .meshsurf import surface_stats
    ss = surface_stats(g["stl"], float(g.get("scale", 1.0)) if _num(g.get("scale", 1.0)) else 1.0)
    if ss is None:
        from pathlib import Path
        p = Path(g["stl"]).expanduser()
        errs.append(f"geometry.stl: no file {p} on this machine" if not p.is_file() else
                    f"geometry.stl: {p.name} cannot be read as STL/OBJ/PLY")
        return errs
    if ss["open_edges"] or ss["nonmanifold_edges"]:
        errs.append(f"geometry.stl: not a closed surface ({ss['open_edges']} open edges, {ss['nonmanifold_edges']} edges "
                    f"shared by 3+ triangles, near {ss.get('at', [])[:3]} mm): snappyHexMesh needs watertight bodies")
    external = (st.get("domain") or {}).get("kind") == "external"
    if ss["bodies"] > 1 and not external:
        errs.append(f"geometry.stl: {ss['bodies']} separate bodies; internal flow needs one fluid body (join them)")
    return errs


def _step_solids(path: str) -> int | None:
    """Solids in a STEP file by its entities (MANIFOLD_SOLID_BREP, BREP_WITH_VOIDS), read in chunks; None when
    the file is not there."""
    from pathlib import Path
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    n, tail = 0, b""
    keys = (b"MANIFOLD_SOLID_BREP", b"BREP_WITH_VOIDS")
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf = tail + chunk
            # occurrences wholly inside the tail were counted with the previous chunk
            n += sum(buf.count(k) - tail.count(k) for k in keys)
            tail = buf[-32:]
    return n


def _selector_errors(sel, where: str, external: bool) -> list[str]:
    if isinstance(sel, list):
        if not sel:
            return [f"{where}: empty selector list"]
        return [e for i, s in enumerate(sel) for e in _selector_errors(s, f"{where}[{i}]", external)]
    if not isinstance(sel, dict):
        return [f"{where}: a selector is a dict like {{'nz': 1}}, {{'pts': [[x, y, z]]}}, {{'cyl_r': 5}}, {{'all': true}}"]
    keys = set(sel)
    known = {"nx", "ny", "nz", "at", "pts", "cyl_r", "side", "all", "box", "group", "solid", "in_box", "in_cyl"}
    errs = [f"{where}: unknown selector key {k!r} (use {sorted(known)})" for k in sorted(keys - known)]
    kinds = [k for k in ("pts", "cyl_r", "all", "box", "group", "solid", "in_box", "in_cyl") if k in keys] + \
        (["normal"] if keys & {"nx", "ny", "nz"} else [])
    if len(kinds) != 1:
        return errs + [f"{where}: give exactly one of nx/ny/nz, pts, cyl_r, all, box, group, solid, in_box, in_cyl "
                       f"(got {kinds or 'none'})"]
    k = kinds[0]
    if k == "group":
        g = sel["group"]
        if not (isinstance(g, str) and g) and not (isinstance(g, list) and g and all(isinstance(x, str) for x in g)):
            errs.append(f"{where}: group is a face group's name (or a list of names)")
    elif k == "solid":
        v = sel["solid"] if isinstance(sel["solid"], list) else [sel["solid"]]
        if not v or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in v):
            errs.append(f"{where}: solid is a body's index from 0 (or a list of them)")
    elif k == "in_box":
        b = sel["in_box"]
        if not _vec(b, 6) or not all(b[i] < b[i + 3] for i in range(3)):
            errs.append(f"{where}: in_box is [x0, y0, z0, x1, y1, z1] with x0 < x1, y0 < y1, z0 < z1 (mm)")
    elif k == "in_cyl":
        c = sel["in_cyl"]
        ok_keys = {"axis", "center", "r", "r_in", "span", "x", "y", "z"}
        if not isinstance(c, dict):
            errs.append(f"{where}: in_cyl is {{axis: [ax, ay, az], center: [x, y, z], r: mm, span: [s0, s1] or "
                        "x/y/z: [lo, hi], r_in: mm}")
        else:
            errs += [f"{where}.in_cyl: unknown key {x!r} (use {sorted(ok_keys)})" for x in sorted(set(c) - ok_keys)]
            if not _vec(c.get("axis")) or all(float(x) == 0 for x in c.get("axis") or [0]):
                errs.append(f"{where}.in_cyl.axis: [ax, ay, az], not zero")
            if "center" in c and not _vec(c["center"]):
                errs.append(f"{where}.in_cyl.center: [x, y, z] (mm)")
            if not _num(c.get("r")) or c.get("r") <= 0:
                errs.append(f"{where}.in_cyl.r: a positive radius (mm)")
            if "r_in" in c and (not _num(c["r_in"]) or c["r_in"] < 0 or (_num(c.get("r")) and c["r_in"] >= c["r"])):
                errs.append(f"{where}.in_cyl.r_in: 0 <= r_in < r (mm)")
            for kk in ("span", "x", "y", "z"):
                if kk in c and (not _vec(c[kk], 2) or c[kk][0] >= c[kk][1]):
                    errs.append(f"{where}.in_cyl.{kk}: [lo, hi] with lo < hi (mm)")
    if k == "normal":
        if not all(_num(sel.get(c, 0)) for c in ("nx", "ny", "nz")) or all(float(sel.get(c, 0)) == 0 for c in ("nx", "ny", "nz")):
            errs.append(f"{where}: nx, ny, nz are numbers, not all zero")
    elif k == "pts":
        if not isinstance(sel["pts"], list) or not sel["pts"] or not all(_vec(p) for p in sel["pts"]):
            errs.append(f"{where}: pts is a list of [x, y, z] points (mm)")
    elif k == "cyl_r":
        if not _num(sel["cyl_r"]) or sel["cyl_r"] <= 0:
            errs.append(f"{where}: cyl_r is a positive radius (mm)")
        if "side" in sel and sel["side"] not in ("inner", "outer"):
            errs.append(f"{where}: side is 'inner' or 'outer'")
    elif k == "all":
        if sel["all"] is not True:
            errs.append(f"{where}: all must be true")
    elif k == "box":
        names = sel["box"] if isinstance(sel["box"], list) else [sel["box"]]
        bad = [n for n in names if n not in BOX_SIDES]
        if bad:
            errs.append(f"{where}: box sides are {list(BOX_SIDES)}, not {bad}")
        if not external:
            errs.append(f"{where}: 'box' selects sides of the external domain box; use nx/ny/nz for internal flow")
    if "at" in sel and k in ("group", "solid", "in_box", "in_cyl"):
        errs.append(f"{where}: at goes with nx/ny/nz or cyl_r")
    if "at" in sel and not _vec(sel["at"]):
        errs.append(f"{where}: at is [x, y, z] (mm)")
    if "side" in sel and k != "cyl_r":
        errs.append(f"{where}: side goes with cyl_r")
    return errs


def _scalar_names(st: dict) -> tuple:
    names = tuple(s.get("name") for s in (st.get("scalars") or []) if isinstance(s, dict) and isinstance(s.get("name"), str))
    return names + (("age",) if st.get("age") else ()) + (("vapour",) if st.get("cavitation") else ())


def image_errors(im, two_d: bool = False, scalars: tuple = (), ref_velocity: bool = True) -> list[str]:
    if im is True or im is False:
        return []
    if not isinstance(im, list):
        return ["images: true or a list of {name, slice | surface | iso, field, ...}"]
    errs, names = [], []
    for i, sp in enumerate(im):
        w = f"images[{i}]"
        if not isinstance(sp, dict) or sum(k in sp for k in ("slice", "surface", "iso")) != 1:
            errs.append(f"{w}: give slice: {{axis, at}}, surface: part | walls | <wall bc>, or iso: {{field, value}}")
            continue
        for k in sp:
            if k not in ("name", "slice", "surface", "iso", "field", "streamlines", "window", "view", "clim",
                         "quality", "mirror", "color"):
                errs.append(f"{w}: unknown key {k!r}")
        if "quality" in sp and (sp["quality"] not in ("high", "basic") or "slice" in sp):
            errs.append(f"{w}.quality: high (VTK: lit, smooth) or basic (Matplotlib), for surfaces")
        if "mirror" in sp and not isinstance(sp["mirror"], bool):
            errs.append(f"{w}.mirror: true (default: half models mirrored on their symmetry side) or false")
        if "color" in sp and "iso" not in sp:
            errs.append(f"{w}.color: for iso images")
        if "name" in sp:
            if not isinstance(sp["name"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", sp["name"]):
                errs.append(f"{w}.name: a word (letters, digits, _ . -)")
            names.append(sp.get("name"))
        if "slice" in sp:
            sl = sp["slice"]
            if not isinstance(sl, dict) or sl.get("axis") not in ("x", "y", "z") or not _num(sl.get("at")):
                errs.append(f"{w}.slice: {{axis: x | y | z, at: coordinate in mm}}")
            elif two_d and sl["axis"] != "z":
                errs.append(f"{w}.slice: a 2d study has only the z slice")
            if sp.get("field", "u") not in ("u", "ux", "uy", "uz", "p", "cp", "T", "mach") + tuple(scalars):
                errs.append(f"{w}.field: u, ux, uy, uz, p, cp, T or mach (T and mach with energy)"
                            + (f", or a scalar: {', '.join(scalars)}" if scalars else ""))
            if "window" in sp and (not _vec(sp["window"], 4) or sp["window"][0] >= sp["window"][1]
                                   or sp["window"][2] >= sp["window"][3]):
                errs.append(f"{w}.window: [a0, a1, b0, b1] in mm along the slice's two in-plane axes")
        elif "iso" in sp:
            iso = sp["iso"]
            if two_d:
                errs.append(f"{w}.iso: isosurfaces are for 3D studies")
            if not isinstance(iso, dict) or iso.get("field", "q") not in ("q", "u", "cp_total") \
                    or not (_num(iso.get("value", 0)) or iso.get("value") == "auto") \
                    or set(iso) - {"field", "value"}:
                errs.append(f"{w}.iso: {{field: q (Q-criterion, 1/s2) | u (m/s) | cp_total, value: number | auto}}")
            elif iso.get("field") == "cp_total" and not ref_velocity:
                errs.append(f"{w}.iso: cp_total needs reference.velocity")
            if sp.get("color", "u") not in ("u", "p", "cp", "none"):
                errs.append(f"{w}.color: u, p, cp or none")
            elif sp.get("color") == "cp" and not ref_velocity:
                errs.append(f"{w}.color: cp needs reference.velocity")
            if "window" in sp and (not _vec(sp["window"], 6) or any(sp["window"][2 * k] >= sp["window"][2 * k + 1]
                                                                    for k in range(3))):
                errs.append(f"{w}.window: [x0, x1, y0, y1, z0, z1] in mm")
            if "view" in sp and not _vec(sp["view"], 2):
                errs.append(f"{w}.view: [elevation, azimuth] in degrees")
            if "field" in sp or "streamlines" in sp:
                errs.append(f"{w}: an iso image takes iso.field (and color), not field or streamlines")
        else:
            if not isinstance(sp["surface"], str):
                errs.append(f"{w}.surface: part, walls or a wall bc's name")
            if sp.get("field", "p") not in ("p", "cp", "wall_shear", "T", "q"):
                errs.append(f"{w}.field: p, cp, wall_shear, T or q (T and q with energy)")
            if "view" in sp and not _vec(sp["view"], 2):
                errs.append(f"{w}.view: [elevation, azimuth] in degrees")
        if sp.get("field") == "cp" and not ref_velocity:
            errs.append(f"{w}.field: cp needs reference.velocity (and reference.p if the far-field gauge is not 0)")
        if "clim" in sp and not (sp["clim"] in ("full", "robust")
                                 or (_vec(sp["clim"], 2) and sp["clim"][0] < sp["clim"][1])):
            errs.append(f"{w}.clim: [lo, hi] in the field's units, 'full' (min to max) or 'robust' (2nd-98th percentile)")
        if "streamlines" in sp and not isinstance(sp["streamlines"], bool):
            errs.append(f"{w}.streamlines: true or false")
    if len([n for n in names if n]) != len({n for n in names if n}):
        errs.append("images: names must be unique")
    return errs


def _energy_bc_errors(b: dict, w: str, st: dict) -> list[str]:
    errs = []
    energy = bool(st.get("energy"))
    kind = b.get("kind")
    ek = {"inlet": ("mass_flow", "temperature", "total_pressure", "total_temperature"),
          "outlet": ("temperature",), "opening": ("temperature",), "wall": THERMAL_KEYS}.get(kind, ())
    used = [k for k in ek if k in b]
    if used and not energy:
        return [f"{w}: {used} need energy: true"]
    if not energy:
        return errs
    for k in ("temperature", "total_temperature", "T", "T_inf"):
        if k in b and (not _num(b[k]) or b[k] <= -273.15):
            errs.append(f"{w}: {k} in degrees C")
    if kind == "inlet":
        if "mass_flow" in b and (not _num(b["mass_flow"]) or b["mass_flow"] <= 0):
            errs.append(f"{w}: mass_flow is a positive number (kg/s)")
        if "total_pressure" in b:
            if not _num(b["total_pressure"]):
                errs.append(f"{w}: total_pressure is a number (Pa gauge)")
            if "gravity" in st:
                errs.append(f"{w}: total_pressure inlets are not available with gravity; give a velocity or mass_flow")
            try:
                gas = "molar_mass" in fluid_props(st["fluid"])
            except (KeyError, TypeError, AttributeError):
                gas = True                        # the fluid's own error is reported elsewhere
            if not gas:
                errs.append(f"{w}: total_pressure inlets are for gases")
            for k in ("profile", "swirl", "temperature"):
                if k in b:
                    errs.append(f"{w}: {k} does not go with total_pressure (use total_temperature)")
        elif "total_temperature" in b:
            errs.append(f"{w}: total_temperature goes with total_pressure (else use temperature)")
    if kind == "wall":
        th = [k for k in ("T", "q", "power", "h") if k in b]
        if len(th) > 1 and not (set(th) <= {"h"}):
            errs.append(f"{w}: give one of T, q, power, or h with T_inf")
        if ("h" in b) != ("T_inf" in b):
            errs.append(f"{w}: h and T_inf go together")
        for k in ("q", "power"):
            if k in b and not _num(b[k]):
                errs.append(f"{w}: {k} is a number")
        if "h" in b and (not _num(b["h"]) or b["h"] < 0):
            errs.append(f"{w}: h is a film coefficient (W/(m2 K), >= 0)")
    return errs


def _rotation_errors(r, where: str, speeds: tuple) -> list[str]:
    if not isinstance(r, dict):
        return [f"{where}: {{axis: [x, y, z], center: [x, y, z], {' or '.join(speeds)}}}"]
    errs = [f"{where}: unknown key {k!r}" for k in r if k not in ("axis", "center") + speeds]
    if not _vec(r.get("axis")) or not any(r["axis"]):
        errs.append(f"{where}.axis: [x, y, z], not zero")
    if "center" in r and not _vec(r["center"]):
        errs.append(f"{where}.center: [x, y, z] (mm)")
    given = [k for k in speeds if k in r]
    if len(given) != 1 or not _num(r[given[0]]):
        errs.append(f"{where}: give one of {list(speeds)} (a number; negative turns the other way)")
    return errs


def _goal_errors(goal: dict) -> list[str]:
    """build123d-translator's own validation of the goal (an open outline used to fail only when meshing:
    venturi_subsonic on mochi), when that package is installed; it is only needed to build the part."""
    try:
        from build123d_translator.goalformat import validate_goal
    except ImportError:
        return []
    try:
        v = validate_goal(goal)
    except Exception as e:  # noqa: BLE001
        return [f"geometry.goal: {type(e).__name__}: {e}"]
    return [f"geometry.goal: {e}" for e in v.get("errors", [])]


def fluid_props(f) -> dict:
    if isinstance(f, str):
        return dict(FLUIDS[f])
    base = dict(FLUIDS.get(f.get("name"), {})) if f.get("name") else {}
    over = {k: v for k, v in f.items() if k != "name"}
    if "rho" in over and "molar_mass" not in over:
        base.pop("molar_mass", None)          # a given density makes it a liquid
    if "molar_mass" in over:
        base.pop("rho", None)
    base.update(over)
    return base


def thermo_props(study: dict) -> dict:
    """Thermal properties of the study's fluid at its conditions: kind (gas/liquid), cp, mu, k, pr, R or rho,
    and the reference density (gas: p / (R T) at conditions)."""
    f = fluid_props(study["fluid"])
    c = study.get("conditions", {})
    p0, T0 = float(c.get("p", 101325.0)), float(c.get("T", 20.0)) + 273.15
    mu, cp = float(f["mu"]), float(f["cp"])
    pr = float(f["pr"]) if "pr" in f else mu * cp / float(f["k"])
    out = {"mu": mu, "cp": cp, "pr": pr, "k": mu * cp / pr, "p_op": p0, "T_op": T0}
    if "molar_mass" in f:
        R = R_UNIVERSAL / float(f["molar_mass"])
        out.update(kind="gas", molar_mass=float(f["molar_mass"]), R=R, rho=p0 / (R * T0),
                   gamma=cp / (cp - R))
    else:
        out.update(kind="liquid", rho=float(f["rho"]))
    return out


def bc_names(study: dict) -> list[str]:
    """The patch name of each bc: its name, or its kind (kind_2, kind_3 ... for further unnamed ones)."""
    out, count = [], {}
    for b in study.get("bcs", []):
        if not isinstance(b, dict):
            out.append("bc")
            continue
        if b.get("name"):
            out.append(b["name"])
            continue
        k = b.get("kind", "bc")
        count[k] = count.get(k, 0) + 1
        out.append(k if count[k] == 1 else f"{k}_{count[k]}")
    return out


def validate_study(st: dict) -> dict:
    errs, warns = [], []
    if not isinstance(st, dict):
        return {"ok": False, "errors": ["a study is a dict"], "warnings": []}
    for k in st:
        if k not in STUDY_KEYS:
            errs.append(f"unknown key {k!r} (keys: {sorted(STUDY_KEYS)})")
    if st.get("analysis", "steady") != "steady":
        errs.append("analysis: only 'steady' for now")
    g = st.get("geometry")
    geo_kind = None
    if not isinstance(g, dict):
        errs.append("geometry: {'goal': ...} | {'step': path} | {'box': [lx, ly, lz]} | {'cylinder': [r, h]} | {'annulus': [r_in, r_out, h]}")
    else:
        kinds = [k for k in ("goal", "step", "box", "cylinder", "annulus", "stl") if k in g]
        if len(kinds) != 1:
            errs.append(f"geometry: give exactly one of goal, step, stl, box, cylinder, annulus (got {kinds or list(g)})")
        else:
            geo_kind = kinds[0]
            if geo_kind == "box" and (not _vec(g["box"]) or min(g["box"]) <= 0):
                errs.append("geometry.box: [lx, ly, lz] positive (mm)")
            if geo_kind == "cylinder" and (not _vec(g["cylinder"], 2) or min(g["cylinder"]) <= 0):
                errs.append("geometry.cylinder: [r, h] positive (mm), axis Z, base on z = 0")
            if geo_kind == "annulus" and (not _vec(g["annulus"], 3) or min(g["annulus"]) <= 0
                                          or g["annulus"][0] >= g["annulus"][1]):
                errs.append("geometry.annulus: [r_in, r_out, h] positive with r_in < r_out (mm), axis Z, base on z = 0")
            if geo_kind == "step" and not isinstance(g["step"], str):
                errs.append("geometry.step: a path to a STEP file (mm)")
            elif geo_kind == "step" and (st.get("domain") or {}).get("kind") != "external":
                ns = _step_solids(g["step"])
                if ns and ns > 1:
                    errs.append(f"geometry.step: the file has {ns} solids; internal flow meshes one fluid body: fuse "
                                "them into one solid (build123d-translator 'combine' / your CAD's union) and export again")
            extra = set(g) - {"stl", "scale", "groups"} if geo_kind == "stl" else set()
            if extra:
                errs.append(f"geometry: unknown key(s) {sorted(extra)} with {geo_kind}" +
                            (" (stl takes scale and groups)" if geo_kind == "stl" else ""))
            if geo_kind == "stl":
                errs += _stl_errors(g, st)
            if geo_kind == "goal" and not isinstance(g["goal"], dict):
                errs.append("geometry.goal: a build123d-translator goal {'features': [...]}")
            elif geo_kind == "goal":
                errs += _goal_errors(g["goal"])
    d = st.get("domain", {"kind": "internal"})
    external = False
    if not isinstance(d, dict) or d.get("kind", "internal") not in ("internal", "external"):
        errs.append("domain: {'kind': 'internal'} or {'kind': 'external', 'box': [x0, y0, z0, x1, y1, z1]}")
    elif d.get("kind") == "external":
        external = True
        b = d.get("box")
        if not _vec(b, 6) or not all(b[i] < b[i + 3] for i in range(3)):
            errs.append("domain.box: [x0, y0, z0, x1, y1, z1] with x0 < x1, y0 < y1, z0 < z1 (mm)")
    two_d = st.get("2d", False)
    if not isinstance(two_d, bool):
        errs.append("2d: true or false")
    f = st.get("fluid")
    if f is None:
        errs.append(f"fluid: one of {sorted(FLUIDS)} or {{'rho': kg/m3, 'mu': Pa s}}")
    elif isinstance(f, str):
        if f not in FLUIDS:
            errs.append(f"fluid: {f!r} unknown; use {sorted(FLUIDS)} or {{'rho', 'mu'}}")
    elif isinstance(f, dict):
        if f.get("name") and f["name"] not in FLUIDS:
            errs.append(f"fluid.name: {f['name']!r} unknown")
        p = fluid_props(f)
        energy = bool(st.get("energy"))
        need = ("mu", "cp") + (("molar_mass",) if "molar_mass" in p else ("rho",)) if energy else ("rho", "mu")
        for k in need:
            if not _num(p.get(k)) or p[k] <= 0:
                errs.append(f"fluid.{k}: a positive number")
        if energy and not (_num(p.get("pr")) or _num(p.get("k"))):
            errs.append("fluid: give pr (Prandtl number) or k (W/(m K)) with energy")
        allowed = {"name", "rho", "mu"} | ({"cp", "pr", "k", "molar_mass"} if energy else set())
        for k in set(f) - allowed:
            errs.append(f"fluid: unknown key {k!r} (allowed: {sorted(allowed)})")
    else:
        errs.append("fluid: a name or {'rho', 'mu'}")
    model = st.get("model", "laminar")
    if model not in MODELS:
        errs.append(f"model: one of {list(MODELS)}")
    # bcs
    bcs = st.get("bcs")
    if not isinstance(bcs, list) or not bcs:
        errs.append("bcs: a non-empty list of boundary conditions")
        bcs = []
    names = bc_names(st) if bcs else []
    for i, b in enumerate(bcs):
        w = f"bcs[{i}]"
        if not isinstance(b, dict) or b.get("kind") not in BC_KINDS:
            errs.append(f"{w}: kind is one of {list(BC_KINDS)}")
            continue
        req, opt, _ = BC_KINDS[b["kind"]]
        for k in b:
            if k not in req and k not in opt and k not in ("kind", "name"):
                errs.append(f"{w} ({b['kind']}): unknown key {k!r} (allowed: {sorted(set(req) | set(opt) | {'name'})})")
        if "faces" not in b:
            errs.append(f"{w}: faces is required")
        else:
            errs += _selector_errors(b["faces"], f"{w}.faces", external)
        if "name" in b and (not isinstance(b["name"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", b["name"])):
            errs.append(f"{w}: name is a word (letters, digits, _; starting with a letter)")
        if names and names[i] in RESERVED:
            errs.append(f"{w}: name {names[i]!r} is reserved")
        if b["kind"] == "inlet":
            given = [k for k in ("velocity", "flow_rate", "flow_rate_lpm", "mass_flow", "total_pressure") if k in b]
            if len(given) != 1:
                errs.append(f"{w}: give exactly one of velocity, flow_rate, flow_rate_lpm, mass_flow, total_pressure")
            v = b.get("velocity")
            if v is not None and not (_num(v) or _vec(v)):
                errs.append(f"{w}: velocity is a number (m/s, normal into the domain) or [vx, vy, vz]")
            if _num(v) and v <= 0:
                errs.append(f"{w}: velocity as a number must be positive (into the domain)")
            for k in ("flow_rate", "flow_rate_lpm"):
                if k in b and (not _num(b[k]) or b[k] <= 0):
                    errs.append(f"{w}: {k} is a positive number")
            if b.get("profile", "uniform") not in PROFILES:
                errs.append(f"{w}: profile is one of {list(PROFILES)}")
            if b.get("profile") == "parabolic" and _vec(v):
                errs.append(f"{w}: a parabolic profile takes velocity as a number (mean, normal into the domain)")
            if "intensity" in b and (not _num(b["intensity"]) or not 0 < b["intensity"] < 0.5):
                errs.append(f"{w}: intensity between 0 and 0.5 (e.g. 0.05)")
            if "length_scale" in b and (not _num(b["length_scale"]) or b["length_scale"] <= 0):
                errs.append(f"{w}: length_scale is a positive length (mm)")
            if "swirl" in b:
                errs += _rotation_errors(b["swirl"], f"{w}.swirl", ("rpm", "tangential"))
        if b["kind"] in ("outlet", "opening") and "pressure" in b and not _num(b["pressure"]):
            errs.append(f"{w}: pressure is a number (Pa)")
        if b["kind"] == "wall" and "velocity" in b and not _vec(b["velocity"]):
            errs.append(f"{w}: velocity is [vx, vy, vz] (m/s)")
        if b["kind"] == "wall" and "rotation" in b and not isinstance(b["rotation"], str):
            errs += _rotation_errors(b["rotation"], f"{w}.rotation", ("rpm",))
            if "velocity" in b:
                errs.append(f"{w}: give velocity or rotation, not both")
        errs += _energy_bc_errors(b, w, st)
    if len(set(names)) != len(names):
        errs.append(f"bcs: names must be unique (got {names})")
    kinds = [b.get("kind") for b in bcs if isinstance(b, dict)]
    moving = any(b.get("kind") == "wall" and (b.get("velocity") or b.get("rotation")) for b in bcs if isinstance(b, dict))
    heated = bool(st.get("energy")) and any(b.get("kind") == "wall" and any(k in b for k in THERMAL_KEYS)
                                             for b in bcs if isinstance(b, dict))
    if bcs and not ({"inlet", "opening"} & set(kinds)) and not moving and not heated:
        errs.append("bcs: nothing drives the flow (add an inlet, an opening, a moving wall, or with energy a heated wall)")
    if "inlet" in kinds and not ({"outlet", "opening"} & set(kinds)):
        errs.append("bcs: an inlet needs an outlet or an opening for the flow to leave")
    if model != "laminar" and "inlet" not in kinds:
        warns.append("kOmegaSST without an inlet: turbulence initial values use reference.velocity (or 1 m/s)")
    # mesh
    m = st.get("mesh", {})
    if not isinstance(m, dict):
        errs.append("mesh: a dict")
        m = {}
    for k in m:
        if k not in MESH_KEYS:
            errs.append(f"mesh: unknown key {k!r} (keys: {sorted(MESH_KEYS)})")
    if "size" in m and (not _num(m["size"]) or m["size"] <= 0):
        errs.append("mesh.size: a positive element size (mm)")
    if "cells" in m:
        c = m["cells"]
        if not (isinstance(c, list) and len(c) == 3 and all(isinstance(x, int) and x >= 1 for x in c)):
            errs.append("mesh.cells: [nx, ny, nz] positive integers")
        if geo_kind != "box" or external:
            errs.append("mesh.cells (structured hex mesh) needs a box geometry and internal flow")
        if two_d and isinstance(c, list) and len(c) == 3 and c[2] != 1:
            errs.append("mesh.cells: 2d needs nz = 1")
        if "grading" in m and (not _vec(m["grading"]) or min(m["grading"]) <= 0):
            errs.append("mesh.grading: [gx, gy, gz] positive (last cell / first cell)")
    elif "grading" in m:
        errs.append("mesh.grading goes with mesh.cells")
    if "extrude" in m and (not isinstance(m["extrude"], int) or m["extrude"] < 1):
        errs.append("mesh.extrude: number of layers along Z (integer >= 1)")
    if "extrude" in m and two_d:
        errs.append("mesh.extrude: 2d already uses one layer; remove extrude")
    if "quads" in m and not isinstance(m["quads"], bool):
        errs.append("mesh.quads: true or false")
    snappy = m.get("mesher") == "snappy"
    if "mesher" in m and m["mesher"] not in ("gmsh", "snappy"):
        errs.append("mesh.mesher: gmsh (default) or snappy")
    if snappy:
        if two_d or "extrude" in m or "cells" in m or "quads" in m:
            errs.append("mesh.mesher snappy: 3D meshes only (no 2d, extrude, cells or quads)")
        if not _num(m.get("size")):
            errs.append("mesh.mesher snappy needs mesh.size: the background cell size (mm)")
        sf = m.get("surface")
        if sf is not None and not ((_num(sf) and sf > 0) or (isinstance(sf, list) and len(sf) == 2
                                                             and all(_num(x) and x > 0 for x in sf))):
            errs.append("mesh.surface: cell size on the part (mm), or [finest, coarsest]")
        elif _num(m.get("size")) and sf is not None and max(sf if isinstance(sf, list) else [sf]) > m["size"]:
            errs.append("mesh.surface: no larger than mesh.size")
        if "features" in m and not (isinstance(m["features"], bool) or (_num(m["features"]) and m["features"] > 0)):
            errs.append("mesh.features: cell size along sharp edges (mm), true (the finest surface size) or false")
        if "gap" in m and not (isinstance(m["gap"], int) and 0 <= m["gap"] <= 4):
            errs.append("mesh.gap: extra refinement levels in narrow gaps, 0-4 (default 1)")
        q = m.get("quality", {})
        if not isinstance(q, dict) or set(q) - {"max_non_ortho", "max_skewness", "retry"}:
            errs.append("mesh.quality: {max_non_ortho (65), max_skewness (4), retry (true)}")
        elif ("max_non_ortho" in q and not (_num(q["max_non_ortho"]) and 30 <= q["max_non_ortho"] <= 85)) or \
                ("max_skewness" in q and not (_num(q["max_skewness"]) and 1 <= q["max_skewness"] <= 20)):
            errs.append("mesh.quality: max_non_ortho 30-85 degrees, max_skewness 1-20")
        if "max_cells" in m and not (_num(m["max_cells"]) and m["max_cells"] >= 1e4):
            errs.append("mesh.max_cells: a cell budget (>= 10000)")
        if geo_kind == "box" and not external:
            warns.append("mesh.mesher snappy on an internal box: mesh.cells gives an exact structured mesh")
    else:
        for k in ("surface", "features", "gap", "quality", "max_cells"):
            if k in m:
                errs.append(f"mesh.{k}: only with mesher snappy")
    if "optimize" in m and not isinstance(m["optimize"], bool):
        errs.append("mesh.optimize: true or false (Gmsh tetrahedra: Netgen's optimiser, default true)")
    if "layers" in m:
        L = m["layers"]
        ok = isinstance(L, dict) and not (set(L) - LAYER_KEYS) and (("first" in L) != ("y_plus" in L))
        if ok:
            ok = (("first" not in L or (_num(L["first"]) and L["first"] > 0))
                  and ("y_plus" not in L or (_num(L["y_plus"]) and L["y_plus"] > 0))
                  and ("n" not in L or (isinstance(L["n"], int) and 1 <= L["n"] <= 30))
                  and _num(L.get("growth", 1.2)) and 1 <= L.get("growth", 1.2) <= 2
                  and ("feature_angle" not in L or (_num(L["feature_angle"]) and 30 <= L["feature_angle"] <= 180))
                  and ("at_openings" not in L or isinstance(L["at_openings"], bool)))
        if not ok:
            errs.append("mesh.layers: {first (mm) | y_plus, n (1-30; snappy: default so the last layer is half a "
                        "surface cell), growth (1-2, default 1.2), on: [bc names] (snappy), feature_angle (snappy, 130), "
                        "at_openings (snappy, true; false: none on walls touching inlets and outlets)}")
        elif not snappy:
            if not (two_d or "extrude" in m):
                errs.append("mesh.layers: with Gmsh, boundary layers need a swept mesh (2d or extrude); "
                            "3D layers: mesher snappy")
            if "n" not in L:
                errs.append("mesh.layers.n: needed with Gmsh")
            for k in ("on", "feature_angle", "at_openings"):
                if k in L:
                    errs.append(f"mesh.layers.{k}: only with mesher snappy")
        if isinstance(L, dict) and "on" in L:
            names = bc_names(st)
            if not (isinstance(L["on"], list) and all(isinstance(x, str) for x in L["on"])):
                errs.append("mesh.layers.on: [bc names]")
            else:
                for x in L["on"]:
                    if x not in names:
                        errs.append(f"mesh.layers.on: no bc named {x!r} (bcs: {names})")
                    elif st["bcs"][names.index(x)].get("kind") != "wall":
                        errs.append(f"mesh.layers.on: {x!r} is not a wall")
        if isinstance(L, dict) and "y_plus" in L and not ((st.get("reference") or {}).get("velocity") or any(
                b.get("kind") == "inlet" and "velocity" in b for b in st.get("bcs", []) if isinstance(b, dict))):
            errs.append("mesh.layers.y_plus needs reference.velocity or an inlet velocity")
    if "refine" in m:
        r = m["refine"]
        good = isinstance(r, list)
        for x in (r if good else []):
            if not (isinstance(x, dict) and _num(x.get("size")) and x["size"] > 0):
                good = False
            elif "box" in x:
                good &= _vec(x["box"], 6) and all(x["box"][i] < x["box"][i + 3] for i in range(3))
            elif "cylinder" in x:
                c = x["cylinder"]
                good &= isinstance(c, dict) and _vec(c.get("axis")) and _num(c.get("radius")) and c["radius"] > 0 \
                    and ("center" not in c or _vec(c["center"])) and ("span" not in c or (_vec(c["span"], 2) and c["span"][0] < c["span"][1])) \
                    and ("inner_radius" not in c or (_num(c["inner_radius"]) and 0 <= c["inner_radius"] < c["radius"])) \
                    and not (set(c) - {"center", "axis", "radius", "inner_radius", "span"})
            elif "distance" in x:
                good &= _num(x["distance"]) and x["distance"] > 0
            else:
                good &= _vec(x.get("at")) and _num(x.get("r")) and x["r"] > 0
        if not good:
            errs.append("mesh.refine: [{size (mm), and at: [x, y, z] + r (sphere) | box: [x0, y0, z0, x1, y1, z1] | "
                        "cylinder: {center, axis, radius, inner_radius (a hollow cylinder; 0 <= inner < radius), span: [s0, s1]} | "
                        "distance: mm from the part}]")
    # solve
    s = st.get("solve", {})
    if not isinstance(s, dict):
        errs.append("solve: a dict")
        s = {}
    for k in s:
        if k not in SOLVE_KEYS:
            errs.append(f"solve: unknown key {k!r} (keys: {sorted(SOLVE_KEYS)})")
    if "iterations" in s and (not isinstance(s["iterations"], int) or s["iterations"] < 1):
        errs.append("solve.iterations: a positive integer")
    if "init_from" in s and (not isinstance(s["init_from"], str) or not s["init_from"].strip()):
        errs.append("solve.init_from: '<request>/<study>' of a finished run on this machine, or '<study>' of this "
                    "request (it runs first): its fields are this run's start")
    if "init_fields" in s and s["init_fields"] not in ("last", "mean"):
        errs.append("solve.init_fields: 'last' (default) or 'mean' (a transient source's mean U and p)")
    if "init_fields" in s and "init_from" not in s:
        errs.append("solve.init_fields goes with solve.init_from")
    if s.get("init_from") and isinstance(s.get("transient"), dict) and s["transient"].get("init_from"):
        errs.append("solve.init_from and solve.transient.init_from: give one")
    if "residual" in s and (not _num(s["residual"]) or not 0 < s["residual"] < 1):
        errs.append("solve.residual: between 0 and 1 (e.g. 1e-5)")
    if "procs" in s and (not isinstance(s["procs"], int) or s["procs"] < 1):
        errs.append("solve.procs: number of MPI ranks (integer >= 1)")
    if "relax" in s:
        rl = s["relax"]
        if not isinstance(rl, dict) or not all(k in ("U", "other", "p") and _num(v) and 0 < v <= 1 for k, v in rl.items()):
            errs.append("solve.relax: {U: 0..1, other: 0..1, p: 0..1 (energy runs)}")
    if "start" in s and s["start"] not in ("rest", "estimate"):
        errs.append("solve.start: 'rest' (default) or 'estimate' (a total-pressure inlet starts at half its isentropic speed)")
    if "ramp" in s and (not isinstance(s["ramp"], int) or s["ramp"] < 0):
        errs.append("solve.ramp: iterations over which a total-pressure inlet rises from the outlet pressure (energy, gas; default 400, 0: none)")
    if "consistent" in s and not isinstance(s["consistent"], bool):
        errs.append("solve.consistent: true (SIMPLEC, default) or false (SIMPLE; energy runs)")
    if "transonic" in s and not isinstance(s["transonic"], bool):
        errs.append("solve.transonic: true or false (energy, gas; default: true above an estimated Mach 0.7 or with a total-pressure inlet)")
    if "average" in s:
        av = s["average"]
        if isinstance(av, dict):
            bad = [k for k in av if k not in AVERAGE_KEYS]
            if bad:
                errs.append(f"solve.average: unknown key(s) {bad} (keys: {sorted(AVERAGE_KEYS)})")
            if "window" in av and (not _num(av["window"]) or not 0 < av["window"] <= 0.9):
                errs.append("solve.average.window: fraction of the iterations at the end of the run (0 < w <= 0.9, default 0.25)")
            if "min_iterations" in av and (not isinstance(av["min_iterations"], int) or av["min_iterations"] < 20):
                errs.append("solve.average.min_iterations: integer >= 20 (default 500)")
            for k in ("drift", "scatter"):
                if k in av and (not _num(av[k]) or not 0 < av[k] < 1):
                    errs.append(f"solve.average.{k}: relative tolerance between 0 and 1 (default {0.01 if k == 'drift' else 0.02})")
        elif not isinstance(av, bool):
            errs.append("solve.average: true, false or {window, min_iterations, drift, scatter}")
        if av and st.get("energy"):
            warns.append("solve.average: not available with energy (the fluid solver) yet; final-iteration values are reported")
    tr = s.get("transient")
    if tr is not None:
        if not isinstance(tr, dict) or set(tr) - TRANSIENT_KEYS or not _num(tr.get("end_time")) or tr["end_time"] <= 0:
            errs.append("solve.transient: {end_time (s, required), average_from (s; end_time / 3), max_co (1), "
                        "write_interval (s; end_time / 10), init (steady iterations first; 500 turbulent, 0 laminar), "
                        "max_jobs (SLURM jobs in a row, 10), dt (first time step, s), continue, init_from, "
                        "init_fields, degrees_per_step, correct_mesh_phi}")
        else:
            T = float(tr["end_time"])
            if "average_from" in tr and (not _num(tr["average_from"]) or not 0 <= tr["average_from"] < T):
                errs.append("solve.transient.average_from: a time (s) from 0 to before end_time")
            if "max_co" in tr and tr["max_co"] is None:
                slz = any(isinstance(z, dict) and z.get("motion") == "sliding" for z in st.get("rotating") or [])
                if not slz and not _num(tr.get("dt")):
                    errs.append("solve.transient.max_co null (a fixed time step) needs dt (s), or a sliding zone "
                                "(degrees_per_step)")
            elif "max_co" in tr and (not _num(tr["max_co"]) or not 0.05 <= tr["max_co"] <= 20):
                errs.append("solve.transient.max_co: Courant number limit, 0.05-20 (default 1), or null for a fixed "
                            "time step (dt, or a sliding mesh's degrees_per_step)")
            for k in ("write_interval", "dt"):
                if k in tr and (not _num(tr[k]) or not 0 < tr[k] <= T):
                    errs.append(f"solve.transient.{k}: seconds, > 0 and <= end_time")
            if "init" in tr and (not isinstance(tr["init"], int) or tr["init"] < 0):
                errs.append("solve.transient.init: steady iterations before the transient run (integer >= 0)")
            if "max_jobs" in tr and (not isinstance(tr["max_jobs"], int) or not 1 <= tr["max_jobs"] <= 50):
                errs.append("solve.transient.max_jobs: 1-50 SLURM jobs in a row (default 10)")
            if "init_from" in tr and (not isinstance(tr["init_from"], str) or not tr["init_from"].strip()):
                errs.append("solve.transient.init_from: '<request>/<study>' of a finished run on this machine or "
                            "'<study>' of this request (its U, p, k, omega are mapped onto this mesh as the start; no "
                            "steady init)")
            if "init_fields" in tr and tr["init_fields"] not in ("last", "mean"):
                errs.append("solve.transient.init_fields: 'last' (the source's latest fields, default) or 'mean'")
            if "init_fields" in tr and "init_from" not in tr:
                errs.append("solve.transient.init_fields goes with init_from")
            if (tr.get("init_from") or s.get("init_from")) and tr.get("init"):
                errs.append("solve.transient: init (steady iterations) and init_from do not go together")
            if "continue" in tr and (not isinstance(tr["continue"], str) or "/" not in tr["continue"]):
                errs.append("solve.transient.continue: '<request>/<study>' of a finished transient run on this machine "
                            "(its mesh and latest time are reused; end_time must be later)")
        if st.get("energy") or st.get("scalars") or st.get("age"):
            errs.append("solve.transient: not with energy, scalars or age yet")
        for k in ("iterations", "residual", "average"):
            if k in s:
                warns.append(f"solve.{k}: a transient run uses solve.transient (end_time, average_from); "
                             f"{k} applies only to its steady start (init)" if k == "iterations" else
                             f"solve.{k}: not used by a transient run")
    elif model == "kOmegaSSTDES":
        errs.append("model kOmegaSSTDES is a transient model: add solve.transient {end_time, ...}")
    if "queue" in st and not isinstance(st["queue"], bool):
        errs.append("queue: true (always a SLURM job) or false (always local)")
    # energy, conditions, gravity
    energy = st.get("energy", False)
    if not isinstance(energy, bool):
        errs.append("energy: true or false")
    c = st.get("conditions")
    if c is not None:
        if not isinstance(c, dict) or set(c) - {"p", "T"}:
            errs.append("conditions: {p: Pa absolute, T: degrees C}")
        else:
            if "p" in c and (not _num(c["p"]) or c["p"] <= 0):
                errs.append("conditions.p: absolute pressure in Pa (> 0)")
            if "T" in c and (not _num(c["T"]) or c["T"] <= -273.15):
                errs.append("conditions.T: degrees C (> -273.15)")
        if not energy and c and "T" in c:
            warns.append("conditions.T is used only with energy: true")
    if "gravity" in st:
        if not _vec(st["gravity"]):
            errs.append("gravity: [gx, gy, gz] in m/s2")
        if not energy:
            errs.append("gravity (buoyancy) needs energy: true")
    if energy and st.get("rotating"):
        errs.append("rotating zones (MRF) are incompressible only for now; remove energy or the zone")
    # rotating zones
    rot = st.get("rotating", [])
    znames = []
    if not isinstance(rot, list):
        errs.append("rotating: a list of zones")
        rot = []
    for i, z in enumerate(rot):
        w = f"rotating[{i}]"
        if not isinstance(z, dict) or not isinstance(z.get("name"), str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", z.get("name", "")):
            errs.append(f"{w}: {{name (a word), axis, center, rpm, radius, span}}")
            continue
        znames.append(z["name"])
        errs += _rotation_errors({k: v for k, v in z.items() if k in ("axis", "center", "rpm")}, w, ("rpm",))
        for k in z:
            if k not in ("name", "axis", "center", "rpm", "radius", "span", "motion"):
                errs.append(f"{w}: unknown key {k!r}")
        if z.get("motion", "mrf") not in ("mrf", "sliding"):
            errs.append(f"{w}.motion: 'mrf' (default: a frozen rotor, steady) or 'sliding' (the mesh turns; transient)")
        if not _num(z.get("radius")) or z["radius"] <= 0:
            errs.append(f"{w}.radius: a positive radius (mm)")
        if "span" in z and (not _vec(z["span"], 2) or z["span"][0] >= z["span"][1]):
            errs.append(f"{w}.span: [s0, s1] (mm along the axis from center, s0 < s1)")
    if len(set(znames)) != len(znames):
        errs.append("rotating: zone names must be unique")
    sliding = [z["name"] for z in rot if isinstance(z, dict) and z.get("motion") == "sliding" and "name" in z]
    if sliding:
        if len(sliding) > 1 or len(sliding) != len(rot):
            errs.append("rotating: one sliding zone per study, and no MRF zones beside it")
        if not tr:
            errs.append(f"rotating {sliding[0]!r}: a sliding mesh is transient: add solve.transient {{end_time "
                        "(about 10 revolutions: 10 x 60 / rpm s), average_from (after 4-5 revolutions)}")
        if (st.get("mesh") or {}).get("mesher") != "snappy":
            errs.append(f"rotating {sliding[0]!r}: a sliding mesh needs mesh.mesher snappy (the zone's cylinder must be "
                        "a meshed surface for its cells to slide along it)")
        if two_d:
            errs.append(f"rotating {sliding[0]!r}: no sliding mesh in 2d")
    if isinstance(tr, dict) and "correct_mesh_phi" in tr and (not sliding or not isinstance(tr["correct_mesh_phi"], bool)):
        errs.append("solve.transient.correct_mesh_phi: true | false, with a sliding zone only (OpenFOAM's mesh-flux "
                    "correction at the interface; default true)")
    if isinstance(tr, dict) and "degrees_per_step" in tr:
        if not sliding:
            errs.append("solve.transient.degrees_per_step: only with a sliding zone (rotating[].motion 'sliding')")
        elif not _num(tr["degrees_per_step"]) or not 0.05 <= tr["degrees_per_step"] <= 10:
            errs.append("solve.transient.degrees_per_step: the largest rotation per time step, 0.05-10 degrees (default 1)")
    for i, b in enumerate(bcs):
        r = b.get("rotation") if isinstance(b, dict) else None
        if isinstance(r, str) and r not in znames:
            errs.append(f"bcs[{i}].rotation: {r!r} is not a rotating zone ({znames or 'none defined'})")
    # cavitation (cavitation.py)
    from .cavitation import errors as _cavitation_errors
    errs += _cavitation_errors(st)
    # images
    if "images" in st:
        errs += image_errors(st["images"], two_d, _scalar_names(st), bool((st.get("reference") or {}).get("velocity")))
    # reference, probes, expect
    ref = st.get("reference", {})
    if not isinstance(ref, dict):
        errs.append("reference: a dict")
        ref = {}
    for k, v in ref.items():
        if k not in REF_KEYS:
            errs.append(f"reference: unknown key {k!r} (keys: {sorted(REF_KEYS)})")
        elif k in ("velocity", "area", "length") and (not _num(v) or v <= 0):
            errs.append(f"reference.{k}: a positive number")
        elif k in ("cofr", "drag_dir", "lift_dir") and not _vec(v):
            errs.append(f"reference.{k}: [x, y, z]")
        elif k == "p" and not _num(v):
            errs.append("reference.p: a number (Pa)")
    pr = st.get("probes", [])
    pnames = []
    if not isinstance(pr, list):
        errs.append("probes: a list")
        pr = []
    for i, p in enumerate(pr):
        if not isinstance(p, dict) or not isinstance(p.get("name"), str):
            errs.append(f"probes[{i}]: {{name, at: [x, y, z]}} or {{name, line: {{from, to, n}}}}")
            continue
        pnames.append(p["name"])
        if ("at" in p) == ("line" in p):
            errs.append(f"probes[{i}]: give at or line")
        elif "at" in p and not _vec(p["at"]):
            errs.append(f"probes[{i}].at: [x, y, z] (mm)")
        elif "line" in p:
            ln = p["line"]
            if not (isinstance(ln, dict) and _vec(ln.get("from")) and _vec(ln.get("to"))
                    and isinstance(ln.get("n", 11), int) and ln.get("n", 11) >= 2):
                errs.append(f"probes[{i}].line: {{from: [x, y, z], to: [x, y, z], n (>= 2)}}")
    if len(set(pnames)) != len(pnames):
        errs.append("probes: names must be unique")
    ex = st.get("expect", [])
    if not isinstance(ex, list):
        errs.append("expect: a list")
        ex = []
    for i, e in enumerate(ex):
        if not (isinstance(e, dict) and isinstance(e.get("quantity"), str) and _num(e.get("value"))):
            errs.append(f"expect[{i}]: {{quantity: dotted path, value: number, tol (relative) or abs}}")
        elif ("tol" in e and (not _num(e["tol"]) or e["tol"] <= 0)) or ("abs" in e and (not _num(e["abs"]) or e["abs"] <= 0)):
            errs.append(f"expect[{i}]: tol and abs are positive")
    # combinations
    if two_d and geo_kind == "cylinder" and not external:
        errs.append("2d: an internal cylinder (axis Z) is not a 2D section; use a box, a goal, or external flow")
    if geo_kind != "box" and "cells" not in m and not m.get("size"):
        warns.append("mesh.size not given: the default (1/30 of the domain diagonal) is coarse; give one")
    from .executor import run_errors
    errs += run_errors(st)
    from .porous import validate as _porous
    _porous(st, errs)
    from .scalars import validate as _scalars
    _scalars(st, {n: b.get("kind") for n, b in zip(names, bcs) if isinstance(b, dict)}, errs)
    return {"ok": not errs, "errors": errs, "warnings": warns}


def validate(studies: dict) -> dict:
    if not isinstance(studies, dict) or not studies:
        return {"ok": False, "studies": {}, "errors": ["studies: a non-empty dict {name: study}"]}
    out = {n: validate_study(s) for n, s in studies.items()}
    bad = [n for n in studies if not re.fullmatch(r"[A-Za-z0-9_.-]+", n)]
    errs = [f"study names are words (letters, digits, _ . -): {bad}"] if bad else []
    return {"ok": all(v["ok"] for v in out.values()) and not errs, "studies": out, "errors": errs}
