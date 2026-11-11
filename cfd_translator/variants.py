"""variants.py — geometry by reference, and variants of one study (mesh levels, settings).
Version: 2026.11.12.2

A geometry is registered once on the machine that runs the studies and then named in any number of
studies, so a 10-15 KB goal is sent once instead of once per study:

    add_geometry(name, goal=... (or a goal JSON file) | step=<path on this machine> | step_base64=... |
                 stl=<STL/OBJ/PLY path> | stl_base64=...)
    {"geometry": {"ref": "ahmed25"}}                    a registered geometry
    {"geometry": {"ref": "<request>/<study>"}}          the geometry of an earlier study

One study can stand for several:

    "mesh_scales": [1, 0.707, 0.5]      every mesh length times each factor: studies <name>_L1, _L2, _L3
    "variants": {"a": {...}, "b": {...}}  each a patch merged into the study: <name>_a, <name>_b
    "cavitation": {"npsh": [8, 5, 3, 2]}  a study per NPSH: <name>_npsh8, ...; compare() adds the NPSH curve and
                                          NPSH3 (cavitation.npsh_curve)

When they have run, compare() puts their main results side by side; with mesh_scales it also gives the
change between the two finest levels and, when the changes shrink, Richardson's extrapolation.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import re
import time
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def geometry_dir() -> Path:
    from .api import out_root
    d = out_root() / "geometry"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- registered geometry --------------------------------------------------------------------------
def _goal_from_text(s: str) -> tuple[dict | None, str | None]:
    """A goal from a JSON file path on this machine or from JSON text: the goal itself, {"goal": goal}, or one
    {name: goal} (as build123d-translator's validate_goals takes them)."""
    t = s.strip()
    if not t.startswith("{"):
        p = Path(t).expanduser()
        if not p.is_file():
            return None, (f"goal: no file {p} on this machine (a local path is copied to the machine by the router; "
                          "otherwise give the goal as a dict)")
        try:
            t = p.read_text()
        except OSError as exc:
            return None, f"goal: cannot read {p}: {exc}"
    try:
        g = json.loads(t)
    except json.JSONDecodeError as exc:
        return None, f"goal: not JSON ({exc})"
    if isinstance(g, dict) and "features" not in g:
        if isinstance(g.get("goal"), dict):
            g = g["goal"]
        elif len(g) == 1 and isinstance(next(iter(g.values())), dict):
            g = next(iter(g.values()))
    if not isinstance(g, dict) or not isinstance(g.get("features"), list):
        return None, "goal: the file holds no build123d-translator goal {features: [...]}"
    return g, None


def add_geometry(name: str, goal: dict | str | None = None, step: str | None = None, step_base64: str | None = None,
                 stl: str | None = None, stl_base64: str | None = None, scale: float = 1.0, groups=None) -> dict:
    """Registers a geometry under name (replacing one of the same name). A goal is checked by building it
    (volume, area and bounding box are returned); a STEP file is copied here; a triangle mesh (STL/OBJ/PLY) is
    copied here and checked closed (bodies, groups, volume)."""
    if not isinstance(name, str) or not NAME.match(name):
        return {"ok": False, "error": "name: letters, digits, _ . - (up to 64)"}
    given = [k for k, v in (("goal", goal), ("step", step), ("step_base64", step_base64), ("stl", stl),
                            ("stl_base64", stl_base64)) if v is not None]
    if len(given) != 1:
        return {"ok": False, "error": "give exactly one of goal, step (a path on this machine), step_base64, stl (a "
                                      "path of an STL, OBJ or PLY file on this machine) or stl_base64"}
    if stl is not None or stl_base64 is not None:
        return _add_mesh(name, stl, stl_base64, scale, groups)
    d = geometry_dir()
    rec = {"name": name, "added": time.strftime("%Y-%m-%d %H:%M:%S")}
    if isinstance(goal, str):             # a goal JSON file on this machine (the router stages local paths), or its text
        goal, err = _goal_from_text(goal)
        if err:
            return {"ok": False, "error": err}
    if goal is not None:
        if not isinstance(goal, dict) or not isinstance(goal.get("features"), list):
            return {"ok": False, "error": "goal: a build123d-translator goal {features: [...]}"}
        rec.update(kind="goal", goal=goal)
        try:
            from build123d_translator import translate as T
            solid, _ = T.build(goal)
            bb = solid.bounding_box()
            rec.update(volume_mm3=round(float(solid.volume), 3), area_mm2=round(float(solid.area), 3),
                       bbox_min=[round(float(v), 4) for v in (bb.min.X, bb.min.Y, bb.min.Z)],
                       bbox_max=[round(float(v), 4) for v in (bb.max.X, bb.max.Y, bb.max.Z)])
        except ImportError:
            rec["note"] = "build123d-translator is not installed here: the goal was stored without building it"
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"the goal does not build: {type(exc).__name__}: {exc}"}
    else:
        dst = d / f"{name}.step"
        if step is not None:
            src = Path(step).expanduser()
            if not src.is_file():
                return {"ok": False, "error": f"no STEP file {src} on this machine (upload it with step_base64)"}
            dst.write_bytes(src.read_bytes())
        else:
            try:
                dst.write_bytes(base64.b64decode(step_base64, validate=True))
            except Exception:  # noqa: BLE001
                return {"ok": False, "error": "step_base64: not valid base64"}
        if b"ISO-10303" not in dst.read_bytes()[:200]:
            dst.unlink()
            return {"ok": False, "error": "not a STEP file (no ISO-10303 header)"}
        rec.update(kind="step", step=str(dst), bytes=dst.stat().st_size)
    (d / f"{name}.json").write_text(json.dumps(rec, indent=1))
    out = {k: v for k, v in rec.items() if k != "goal"}
    out.update(ok=True, use={"geometry": {"ref": name}})
    return out


def _add_mesh(name: str, stl, stl_base64, scale, groups) -> dict:
    from .meshsurf import SUFFIXES, SurfaceError, load
    d = geometry_dir()
    try:
        scale = float(scale)
        assert scale > 0
    except (TypeError, ValueError, AssertionError):
        return {"ok": False, "error": "scale: a positive factor to mm (1000 for a model in metres)"}
    if stl is not None:
        src = Path(stl).expanduser()
        if not src.is_file():
            return {"ok": False, "error": f"no mesh file {src} on this machine (upload it with stl_base64)"}
        if src.suffix.lower() not in SUFFIXES:
            return {"ok": False, "error": "stl: an .stl, .obj or .ply file"}
        dst = d / f"{name}{src.suffix.lower()}"
        dst.write_bytes(src.read_bytes())
    else:
        dst = d / f"{name}.stl"
        try:
            dst.write_bytes(base64.b64decode(stl_base64, validate=True))
        except Exception:  # noqa: BLE001
            return {"ok": False, "error": "stl_base64: not valid base64"}
    for old in d.glob(f"{name}.*"):                     # a previous registration in another format
        if old.suffix.lower() in SUFFIXES and old != dst:
            old.unlink()
    geo = {"stl": str(dst), **({"scale": scale} if scale != 1.0 else {})}
    if groups is not None:
        if isinstance(groups, str) and Path(groups).expanduser().is_file():
            gdst = d / f"{name}.groups.json"
            gdst.write_text(Path(groups).expanduser().read_text())
            groups = str(gdst)
        geo["groups"] = groups
    try:
        L = load(geo)
    except SurfaceError as exc:
        dst.unlink(missing_ok=True)
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        return {"ok": False, "error": f"cannot read {dst.name}: {type(exc).__name__}: {exc}"}
    ck = L["check"]
    rec = {"name": name, "added": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "stl", "geometry": geo,
           "bytes": dst.stat().st_size, "triangles": ck["triangles"], "bodies": L["bodies"],
           "body_volumes_mm3": [round(v, 3) for v in L["body_volumes_mm3"]],
           "volume_mm3": round(float(sum(L["body_volumes_mm3"])), 3), "area_mm2": round(ck["area_mm2"], 3),
           "bbox_min": ck["bbox_min"], "bbox_max": ck["bbox_max"], "groups": list(L["names"]),
           "faces": int(L["face"].max() + 1)}
    (d / f"{name}.json").write_text(json.dumps(rec, indent=1))
    out = dict(rec, ok=True, use={"geometry": {"ref": name}})
    if L["bodies"] > 1:
        out["note"] = (f"{L['bodies']} separate bodies: external flow only; each is a 'solid' (selector {{'solid': k}}) "
                       "and results give forces_by_solid")
    return out


def list_geometry() -> dict:
    out = []
    for p in sorted(geometry_dir().glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append({k: v for k, v in r.items() if k not in ("goal",)})
    return {"geometry": out, "use": {"geometry": {"ref": "<name>"}}}


def resolve(study: dict) -> tuple[dict, str | None]:
    """The study with geometry {"ref": ...} replaced by the geometry it names; (study, error)."""
    g = study.get("geometry")
    if not (isinstance(g, dict) and "ref" in g):
        return study, None
    if len(g) != 1:
        return study, "geometry: ref goes alone ({'ref': name})"
    ref = g["ref"]
    if not isinstance(ref, str) or not ref:
        return study, "geometry.ref: a registered geometry's name, or '<request>/<study>'"
    st = dict(study)
    if "/" not in ref:
        f = geometry_dir() / f"{ref}.json"
        if not f.is_file():
            names = sorted(p.stem for p in geometry_dir().glob("*.json"))
            return study, f"geometry.ref: no geometry {ref!r} (registered: {names}; add_geometry registers one)"
        rec = json.loads(f.read_text())
        kind = rec.get("kind")
        st["geometry"] = ({"goal": rec["goal"]} if kind == "goal" else copy.deepcopy(rec["geometry"]) if kind == "stl"
                          else {"step": rec["step"]})
        return st, None
    from .api import out_root
    p = Path(ref).expanduser()
    if not p.is_absolute():
        p = out_root() / p
    req, name = p.parent, p.name
    try:
        meta = json.loads((req / "request.json").read_text())
        st["geometry"] = copy.deepcopy(meta["studies"][name]["geometry"])
    except (OSError, KeyError, json.JSONDecodeError):
        return study, f"geometry.ref: no study {name!r} in request {req}"
    return st, None


# ---- variants -----------------------------------------------------------------------------------------
def merge(base, patch):
    """Deep merge: dicts merge key by key, anything else (lists too) replaces; None removes a key."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return copy.deepcopy(patch)
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = merge(out.get(k), v)
    return out


def scale_mesh(mesh: dict, f: float) -> dict:
    """Every cell size times f (size, surface, features, refine sizes); structured cell counts and extrusion
    layers divided by it. Prism layers keep their first height (the y+ the study asks for)."""
    m = copy.deepcopy(mesh or {})

    def r(v):
        return float(f"{float(v) * f:.6g}")
    if "size" in m:
        m["size"] = r(m["size"])
    if "surface" in m:
        m["surface"] = [r(v) for v in m["surface"]] if isinstance(m["surface"], list) else r(m["surface"])
    if isinstance(m.get("features"), (int, float)) and not isinstance(m["features"], bool):
        m["features"] = r(m["features"])
    for x in m.get("refine") or []:
        if "size" in x:
            x["size"] = r(x["size"])
    if "cells" in m and isinstance(m["cells"], list):
        m["cells"] = [max(1, int(round(c / f))) if i < 2 or c != 1 else 1 for i, c in enumerate(m["cells"])]
    if isinstance(m.get("extrude"), int):
        m["extrude"] = max(1, int(round(m["extrude"] / f)))
    return m


def expand(studies: dict) -> tuple[dict, dict, dict]:
    """(studies with every variant spelled out, groups {base: {names, scales}}, errors {name: [...]})."""
    out, groups, errs = {}, {}, {}
    for name, st in studies.items():
        cv = st.get("cavitation") if isinstance(st, dict) else None
        if isinstance(cv, dict) and isinstance(cv.get("npsh"), list):
            vals = cv["npsh"]
            if {"variants", "mesh_scales"} & set(st):
                errs[name] = ["an NPSH sweep (cavitation.npsh list) goes without variants or mesh_scales"]
                continue
            if not (2 <= len(vals) <= 12 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0
                                                 for x in vals) and len(set(vals)) == len(vals)):
                errs[name] = ["cavitation.npsh: 2-12 different NPSH values (m, > 0), e.g. [10, 6, 4, 3, 2.5, 2]; the "
                              "highest should be free of cavitation (the reference head)"]
                continue
            names = []
            for x in vals:
                n = f"{name}_npsh{float(x):g}"
                out[n] = merge(st, {"cavitation": {"npsh": float(x)}})
                if cv.get("chain") and names:
                    # each point starts from the previous one's fields (solve.init_from in the same request: it runs
                    # after it); the first keeps the study's own init_from, if any
                    out[n] = merge(out[n], {"solve": {"init_from": names[-1]}})
                    out[n]["solve"].pop("init_fields", None)
                names.append(n)
            groups[name] = {"names": names, "scales": None, "npsh": [float(x) for x in vals],
                            "head_drop": float(cv.get("head_drop", 0.03))}
            continue
        if not isinstance(st, dict) or not ({"variants", "mesh_scales"} & set(st)):
            out[name] = st
            continue
        if "variants" in st and "mesh_scales" in st:
            errs[name] = ["give variants or mesh_scales, not both"]
            continue
        base = {k: v for k, v in st.items() if k not in ("variants", "mesh_scales")}
        if "mesh_scales" in st:
            sc = st["mesh_scales"]
            if not (isinstance(sc, list) and 2 <= len(sc) <= 6 and all(isinstance(x, (int, float)) and 0 < x <= 4 for x in sc)):
                errs[name] = ["mesh_scales: 2-6 factors in (0, 4], coarse first, e.g. [1, 0.707, 0.5]"]
                continue
            names = []
            for i, f in enumerate(sc):
                n = f"{name}_L{i + 1}"
                out[n] = dict(base, mesh=scale_mesh(base.get("mesh") or {}, float(f)))
                names.append(n)
            groups[name] = {"names": names, "scales": [float(x) for x in sc]}
        else:
            vs = st["variants"]
            if not (isinstance(vs, dict) and vs and all(isinstance(v, dict) for v in vs.values())
                    and all(isinstance(k, str) and NAME.match(k) for k in vs)):
                errs[name] = ["variants: {suffix: {keys to change}}, e.g. {'coarse': {'mesh': {'size': 20}}}"]
                continue
            names = []
            for suf, patch in vs.items():
                n = f"{name}_{suf}"
                out[n] = merge(base, patch)
                names.append(n)
            groups[name] = {"names": names, "scales": None}
    return out, groups, errs


# ---- comparison --------------------------------------------------------------------------------------
QUANTITIES = ("coefficients.cd", "coefficients.cl", "pressure_drop", "total_pressure_drop", "head_loss_m",
              "y_plus_part_mean", "pump.head_m", "pump.efficiency", "age.mean_residence_time_s",
              "cavitation.npsh_inlet_m", "cavitation.vapour_volume_mm3", "cavitation.gas_volume_mm3",
              "cavitation.gas_holdup")


def _get(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def convergence(values: list, scales: list) -> dict:
    """Coarse to fine. Change between the two finest; with three or more levels whose changes shrink and keep
    their sign, the observed order and Richardson's extrapolation to zero cell size."""
    v = [float(x) for x in values]
    out = {"change_last_pct": float(f"{100 * (v[-1] - v[-2]) / max(abs(v[-1]), 1e-300):.3g}")}
    if len(v) >= 3:
        e1, e2 = v[-2] - v[-3], v[-1] - v[-2]
        r = scales[-2] / scales[-1]
        p = math.log(abs(e1) / abs(e2)) / math.log(r) if e1 * e2 > 0 and abs(e2) < abs(e1) and r > 1 else None
        if p is not None and p >= 0.5:
            ext = v[-1] + e2 / (r ** p - 1)
            out.update(converging=True, order=float(f"{p:.3g}"), extrapolated=float(f"{ext:.5g}"),
                       finest_vs_extrapolated_pct=float(f"{100 * (v[-1] - ext) / max(abs(ext), 1e-300):.3g}"))
        elif p is not None:
            out.update(converging=False, order=float(f"{p:.3g}"),
                       note="the changes shrink too slowly (observed order below 0.5): not in the asymptotic range "
                            "yet, no extrapolation (a finer level)")
        else:
            out.update(converging=False, note="the changes do not shrink between levels: not in the asymptotic range "
                                              "yet (a finer level, or the flow is not steady)")
    return out


def compare(results: dict, group: dict) -> dict:
    """The group's studies side by side (those finished): cells, reliable, and every main quantity present."""
    rows = {}
    for n in group["names"]:
        r = results.get(n)
        if not isinstance(r, dict) or "results" not in r:
            rows[n] = {"state": "pending" if r is None else "failed"}
            continue
        row = {"cells": (r.get("mesh") or {}).get("cells"), "reliable": r.get("reliable")}
        for q in QUANTITIES:
            x = _get(r["results"], q)
            if x is not None:
                row[q] = x
        rows[n] = row
    out = {"studies": rows}
    done = [n for n in group["names"] if "cells" in rows[n]]
    if group.get("scales") and len(done) >= 2 and len(done) == len(group["names"]):
        conv = {}
        for q in QUANTITIES:
            vals = [rows[n].get(q) for n in done]
            if all(x is not None for x in vals) and q != "y_plus_part_mean":
                conv[q] = convergence(vals, group["scales"])
        out["mesh_convergence"] = conv
    if group.get("npsh"):
        from .cavitation import npsh_curve
        pts = []
        for n, v in zip(group["names"], group["npsh"]):
            r = rows.get(n) or {}
            if "cells" in r:
                fl = ((results.get(n) or {}).get("results") or {}).get("cavitation", {}).get("flows") or {}
                pts.append({"name": n, "npsh": v, "head": r.get("pump.head_m"), "reliable": r.get("reliable"),
                            "vapour_mm3": r.get("cavitation.vapour_volume_mm3"),
                            "npsh_inlet": r.get("cavitation.npsh_inlet_m"), "flow_error": fl.get("liquid_flow_error")})
        out["npsh_curve"] = npsh_curve(pts, group.get("head_drop", 0.03))
        if len(pts) < len(group["names"]):
            out["npsh_curve"]["pending"] = [n for n in group["names"] if "cells" not in (rows.get(n) or {})]
    return out
