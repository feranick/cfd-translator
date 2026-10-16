"""format.py — the study format: geometry, domain, fluid, model, boundary conditions, validation.
Version: 2026.10.16.1

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
    "air": {"rho": 1.204, "mu": 1.825e-5, "note": "20 degrees C, 1 atm"},
    "water": {"rho": 998.2, "mu": 1.002e-3, "note": "20 degrees C"},
}
MODELS = ("laminar", "kOmegaSST")
PROFILES = ("uniform", "parabolic")
BOX_SIDES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
RESERVED = ("walls", "frontAndBack")

# kind: (required, optional, description)
BC_KINDS = {
    "inlet": ({"faces": ""},
              {"velocity": "m/s: a number (normal into the domain) or [vx, vy, vz]",
               "flow_rate": "m3/s (instead of velocity)", "flow_rate_lpm": "L/min (instead of velocity)",
               "profile": "uniform (default) | parabolic (fully developed laminar: circular, rectangular or 2D inlets)",
               "intensity": "turbulence intensity (kOmegaSST; default 0.05)",
               "length_scale": "mm, turbulence length scale (default 0.07 x hydraulic diameter)"},
              "Fixed velocity in (pressure zero-gradient)."),
    "outlet": ({"faces": ""}, {"pressure": "Pa (static, default 0)"},
               "Fixed static pressure; velocity zero-gradient, no backflow."),
    "opening": ({"faces": ""}, {"pressure": "Pa (static, default 0)"},
                "Fixed static pressure; flow may go in or out (open ends, far-field)."),
    "wall": ({"faces": ""}, {"velocity": "m/s [vx, vy, vz]: moving wall (tangential part used; default at rest)"},
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
}

CONVENTIONS = [
    "Units: geometry, coordinates, mesh sizes, areas in mm (mm2); velocity m/s; pressure Pa; density kg/m3; "
    "viscosity Pa s; flow rate m3/s (or L/min with flow_rate_lpm); forces N; moments N m.",
    "One study is one steady incompressible flow (OpenFOAM incompressibleFluid, SIMPLE). The geometry is a "
    "build123d-translator goal, a STEP file (mm), a box [lx, ly, lz] or a cylinder [r, h] (axis Z), all with "
    "the base on z = 0 and centred in X/Y.",
    "domain internal (default): the geometry is the fluid (a channel, a manifold). external: the fluid is the "
    "domain box [x0, y0, z0, x1, y1, z1] minus the geometry (flow around a part).",
    "Faces are chosen with the part's own selectors (as in build123d-translator and fea-translator; for external "
    "flow the normals are the solid part's), plus {'box': 'xmin'} for the sides of the external domain box. "
    "'faces' may be a list (union). Every face no bc takes becomes a no-slip wall named 'walls'.",
    "2d: true makes a one-cell-thick mesh along Z with 'empty' front and back (OpenFOAM 2D): the domain must be "
    "a prism along Z (one bottom face, one top face, vertical sides). Results are for the modelled thickness.",
    "The flow needs a driver (inlet, opening or moving wall); an inlet needs an outlet or opening. inlet "
    "velocity as a number is normal into the domain; 'parabolic' gives the fully developed laminar profile "
    "with that mean velocity (or flow rate).",
    "Mesh: box geometry (internal) with cells [nx, ny, nz] (+ grading [gx, gy, gz] = last/first cell) gives a "
    "structured hex mesh; otherwise Gmsh: size (mm, default 1/30 of the domain diagonal), refine [{at, r, "
    "size}], and for prisms along Z extrude: n layers (2d uses 1) with quads (default true) and boundary "
    "layers {n, first (mm), growth} on the walls. 3D tetrahedral meshes have no boundary layers.",
    "kOmegaSST uses wall functions (y+ 30-300 at the first cell is ideal; y_plus is reported).",
    "Results: flow.<bc>.Q (m3/s, positive into the domain), Q_lpm, U_mean; pressure.<bc>.p_mean (area mean, Pa), "
    "p_total (flow-weighted total pressure); pressure_drop (mean inlet minus mean outlet static pressure), "
    "total_pressure_drop; forces.<wall bc>.fx/fy/fz (N, by the fluid on the wall, pressure + viscous) and "
    ".mx/my/mz about reference.cofr; forces.part (external: every part face); coefficients.cd/cl (with "
    "reference.velocity and area); probes.<name>.ux/uy/uz/p/u; max_velocity; wall_shear; y_plus; "
    "feat_loads (external: per part face, as fea-translator force bcs); checks (mass and momentum balance, "
    "convergence, mesh).",
    "expect: [{quantity, value, tol}] turns the run into a check: quantity is a dotted path into the results "
    "(pressure_drop, flow.inlet.Q, forces.part.fx, coefficients.cd, probes.p1.ux, probes.line.ux.4); tol is "
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
    "geometry": "{'goal': b3dt goal} | {'step': path} | {'box': [lx, ly, lz]} | {'cylinder': [r, h]}",
    "domain": "{'kind': 'internal'} (default) | {'kind': 'external', 'box': [x0, y0, z0, x1, y1, z1]}",
    "2d": "true: one cell along Z, empty front/back",
    "fluid": f"one of {sorted(FLUIDS)} or {{rho, mu}} (or a name plus overrides)",
    "model": "|".join(MODELS),
    "bcs": "list of boundary conditions",
    "mesh": "{size, refine: [{at, r, size}], cells: [nx, ny, nz], grading, extrude, quads, layers: {n, first, growth}}",
    "solve": "{iterations (2000), residual (1e-5), relax: {U, other}}",
    "reference": "{velocity (m/s), area (mm2), length (mm), cofr [x, y, z] mm, drag_dir, lift_dir, p (Pa, gauge zero)}",
    "probes": "[{name, at: [x, y, z]} | {name, line: {from, to, n}}]",
    "expect": "[{quantity, value, tol | abs}]",
    "analysis": "steady (the only one for now)",
}
MESH_KEYS = {"size", "refine", "cells", "grading", "extrude", "quads", "layers"}
SOLVE_KEYS = {"iterations", "residual", "relax"}
REF_KEYS = {"velocity", "area", "length", "cofr", "drag_dir", "lift_dir", "p"}


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


def _selector_errors(sel, where: str, external: bool) -> list[str]:
    if isinstance(sel, list):
        if not sel:
            return [f"{where}: empty selector list"]
        return [e for i, s in enumerate(sel) for e in _selector_errors(s, f"{where}[{i}]", external)]
    if not isinstance(sel, dict):
        return [f"{where}: a selector is a dict like {{'nz': 1}}, {{'pts': [[x, y, z]]}}, {{'cyl_r': 5}}, {{'all': true}}"]
    keys = set(sel)
    known = {"nx", "ny", "nz", "at", "pts", "cyl_r", "side", "all", "box"}
    errs = [f"{where}: unknown selector key {k!r} (use {sorted(known)})" for k in sorted(keys - known)]
    kinds = [k for k in ("pts", "cyl_r", "all", "box") if k in keys] + (["normal"] if keys & {"nx", "ny", "nz"} else [])
    if len(kinds) != 1:
        return errs + [f"{where}: give exactly one of nx/ny/nz, pts, cyl_r, all, box (got {kinds or 'none'})"]
    k = kinds[0]
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
    if "at" in sel and not _vec(sel["at"]):
        errs.append(f"{where}: at is [x, y, z] (mm)")
    if "side" in sel and k != "cyl_r":
        errs.append(f"{where}: side goes with cyl_r")
    return errs


def fluid_props(f) -> dict:
    if isinstance(f, str):
        return dict(FLUIDS[f])
    base = dict(FLUIDS.get(f.get("name"), {})) if f.get("name") else {}
    base.update({k: v for k, v in f.items() if k != "name"})
    return base


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
        errs.append("geometry: {'goal': ...} | {'step': path} | {'box': [lx, ly, lz]} | {'cylinder': [r, h]}")
    else:
        kinds = [k for k in ("goal", "step", "box", "cylinder") if k in g]
        if len(kinds) != 1:
            errs.append(f"geometry: give exactly one of goal, step, box, cylinder (got {kinds or list(g)})")
        else:
            geo_kind = kinds[0]
            if geo_kind == "box" and (not _vec(g["box"]) or min(g["box"]) <= 0):
                errs.append("geometry.box: [lx, ly, lz] positive (mm)")
            if geo_kind == "cylinder" and (not _vec(g["cylinder"], 2) or min(g["cylinder"]) <= 0):
                errs.append("geometry.cylinder: [r, h] positive (mm), axis Z, base on z = 0")
            if geo_kind == "step" and not isinstance(g["step"], str):
                errs.append("geometry.step: a path to a STEP file (mm)")
            if geo_kind == "goal" and not isinstance(g["goal"], dict):
                errs.append("geometry.goal: a build123d-translator goal {'features': [...]}")
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
        for k in ("rho", "mu"):
            if not _num(p.get(k)) or p[k] <= 0:
                errs.append(f"fluid.{k}: a positive number ({'kg/m3' if k == 'rho' else 'Pa s'})")
        for k in set(f) - {"name", "rho", "mu"}:
            errs.append(f"fluid: unknown key {k!r} (rho, mu, name)")
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
            given = [k for k in ("velocity", "flow_rate", "flow_rate_lpm") if k in b]
            if len(given) != 1:
                errs.append(f"{w}: give exactly one of velocity, flow_rate, flow_rate_lpm")
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
        if b["kind"] in ("outlet", "opening") and "pressure" in b and not _num(b["pressure"]):
            errs.append(f"{w}: pressure is a number (Pa)")
        if b["kind"] == "wall" and "velocity" in b and not _vec(b["velocity"]):
            errs.append(f"{w}: velocity is [vx, vy, vz] (m/s)")
    if len(set(names)) != len(names):
        errs.append(f"bcs: names must be unique (got {names})")
    kinds = [b.get("kind") for b in bcs if isinstance(b, dict)]
    moving = any(b.get("kind") == "wall" and b.get("velocity") for b in bcs if isinstance(b, dict))
    if bcs and not ({"inlet", "opening"} & set(kinds)) and not moving:
        errs.append("bcs: nothing drives the flow (add an inlet, an opening or a moving wall)")
    if "inlet" in kinds and not ({"outlet", "opening"} & set(kinds)):
        errs.append("bcs: an inlet needs an outlet or an opening for the flow to leave")
    if model == "kOmegaSST" and "inlet" not in kinds:
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
    if "layers" in m:
        L = m["layers"]
        if not (isinstance(L, dict) and isinstance(L.get("n"), int) and L["n"] >= 1 and _num(L.get("first"))
                and L["first"] > 0 and _num(L.get("growth", 1.2)) and L.get("growth", 1.2) >= 1):
            errs.append("mesh.layers: {n (>= 1), first (mm, > 0), growth (>= 1, default 1.2)}")
        if not (two_d or "extrude" in m):
            errs.append("mesh.layers: boundary layers need a swept mesh (2d or extrude)")
    if "refine" in m:
        r = m["refine"]
        if not isinstance(r, list) or not all(isinstance(x, dict) and _vec(x.get("at")) and _num(x.get("r"))
                                              and x["r"] > 0 and _num(x.get("size")) and x["size"] > 0 for x in r):
            errs.append("mesh.refine: [{at: [x, y, z], r (mm), size (mm)}]")
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
    if "residual" in s and (not _num(s["residual"]) or not 0 < s["residual"] < 1):
        errs.append("solve.residual: between 0 and 1 (e.g. 1e-5)")
    if "relax" in s:
        rl = s["relax"]
        if not isinstance(rl, dict) or not all(k in ("U", "other") and _num(v) and 0 < v <= 1 for k, v in rl.items()):
            errs.append("solve.relax: {U: 0..1, other: 0..1}")
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
    return {"ok": not errs, "errors": errs, "warnings": warns}


def validate(studies: dict) -> dict:
    if not isinstance(studies, dict) or not studies:
        return {"ok": False, "studies": {}, "errors": ["studies: a non-empty dict {name: study}"]}
    out = {n: validate_study(s) for n, s in studies.items()}
    bad = [n for n in studies if not re.fullmatch(r"[A-Za-z0-9_.-]+", n)]
    errs = [f"study names are words (letters, digits, _ . -): {bad}"] if bad else []
    return {"ok": all(v["ok"] for v in out.values()) and not errs, "studies": out, "errors": errs}
