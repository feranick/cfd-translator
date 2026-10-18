"""runner.py — one study end to end: mesh, patches, case, solve, results, checks.
Version: 2026.10.18.1

    run_study(name, study, folder, solver="foam" | "ref", timeout) -> result dict (also <folder>/<name>.json)
    prepare(study, folder) -> (mesh, patches, selection, plan, warnings)   (no solve)
"""
from __future__ import annotations

import json
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from .format import validate_study


def prepare(study: dict, folder: Path):
    from .case import write_case
    from .mesher import mesh_study
    from .select import assign_bcs, mesh_groups

    external = study.get("domain", {}).get("kind") == "external"
    mesh = mesh_study(study, folder)
    groups = mesh_groups(mesh, external)
    patches, selection, warns = assign_bcs(groups, study, mesh.info.get("empty_tags", []))
    plan = write_case(study, mesh, patches, Path(folder) / "case")
    return mesh, patches, selection, plan, warns


def run_study(name: str, study: dict, folder: Path, solver: str = "foam", timeout: float = 3600) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    out = {"name": name, "translator": __version__, "solver": solver}
    v = validate_study(study)
    if not v["ok"]:
        out.update(ok=False, why="invalid study", errors=v["errors"])
        return _save(out, folder, name)
    warns = list(v["warnings"])
    try:
        mesh, patches, selection, plan, w = prepare(study, folder)
        warns += w
        t1 = time.perf_counter()
        case = folder / "case"
        out["selection"] = selection
        out["mesh"] = {k: v for k, v in plan["quality"].items()}
        out["mesh"].update({k: mesh.info[k] for k in ("kind", "size", "layers", "cells_2d", "quads_2d",
                                                       "boundary_layers_on", "cells_xyz") if k in mesh.info})
        out["case"] = {"inlets": plan["inlets"], "schemes": plan["schemes"], "iterations": plan["iterations"],
                       "residual": plan["residual"], "relax": plan["relax"]}
        if solver == "ref":
            from .refsolver import solve_case
            fields = solve_case(case)
            out["solve"] = {"solver": "reference (2D MAC, Picard)", "iterations": fields["iterations"],
                            "converged": fields["converged"], "change": fields["change"]}
            if not fields["converged"]:
                warns.append("reference solver did not converge")
        else:
            from . import foam
            info = foam.run(case, timeout, study.get("solve", {}).get("procs"))
            fields = foam.read_fields(case, mesh, plan)
            lg = info["log"]
            out["solve"] = {"openfoam": info.get("openfoam"), "procs": info.get("procs"), "iterations": lg["iterations"],
                            "converged": lg["converged"], "residuals": lg["residuals"],
                            "first_residuals": lg["first_residuals"], "time": fields["time"],
                            "checkMesh": info["checkMesh"]}
            if "continuity" in lg:
                out["solve"]["continuity"] = lg["continuity"]
            if not lg["converged"]:
                # a velocity component that is zero everywhere (exact 1D/2D solutions) has a residual of
                # round-off over a near-zero normalisation that never meets the target: leave it out
                Uc = np.asarray(fields["U"], float)
                umax = float(np.abs(Uc).max()) if Uc.size else 0.0
                tiny = [f"U{c}" for k, c in enumerate("xyz") if Uc.size and float(np.abs(Uc[:, k]).max()) <= 1e-9 * umax]
                rest = {f: r for f, r in lg["residuals"].items() if f not in tiny}
                if tiny and rest and max(rest.values()) <= plan["residual"]:
                    lg["converged"] = True
                    out["solve"]["converged"] = True
                    out["solve"]["converged_by"] = f"residuals, ignoring {', '.join(t for t in tiny if t in lg['residuals'])} (zero everywhere)"
            if not lg["converged"]:
                worst = max(lg["residuals"].items(), key=lambda kv: kv[1]) if lg["residuals"] else ("?", float("nan"))
                warns.append(f"not converged in {lg['iterations']} iterations (largest last residual: {worst[0]} "
                             f"{worst[1]:.3g}); raise solve.iterations, lower relax, or refine the mesh")
            if not info["checkMesh"].get("ok"):
                warns.append(f"checkMesh: {info['checkMesh'].get('errors') or 'failed checks'}")
            warns += fields.get("notes", [])
        t2 = time.perf_counter()
        from .post import check_expect, results
        res, w = results(study, mesh, plan, fields)
        warns += w
        out["results"] = res
        from .render import default_images, render, save_state
        try:
            save_state(folder, mesh, plan, fields, study)
        except Exception as exc:  # noqa: BLE001
            warns.append(f"could not save the state for re-rendering: {exc}")
        if study.get("images"):
            specs = default_images(study, mesh) if study["images"] is True else study["images"]
            imgs, w = render(study, mesh, plan, fields, folder / "images", specs, name)
            warns += w
            out["images"] = imgs
        if study.get("expect"):
            out["expect"] = check_expect(res, study["expect"])
            out["passed"] = all(e["passed"] for e in out["expect"])
        out["warnings"] = warns
        out["reliable"], why = _verdict(out, res)
        if why:
            out["unreliable_because"] = why
        out["ok"] = True
        out["files"] = {"case": str(case), "archive": _archive(case, folder / f"{name}_case.tgz")}
        out["seconds"] = {"mesh_case": round(t1 - t0, 3), "solve": round(t2 - t1, 3),
                          "results": round(time.perf_counter() - t2, 3)}
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, why=type(exc).__name__, error=str(exc)[:3000], warnings=warns)
        out["trace"] = traceback.format_exc()[-2000:]
    return _save(out, folder, name)


def _verdict(out: dict, res: dict) -> tuple[bool, list]:
    """Whether the numbers may be reported as results: converged, mass and momentum balanced."""
    why = []
    if not out.get("solve", {}).get("converged"):
        why.append("the solver did not converge (see solve.residuals)")
    ch = res.get("checks", {})
    if ch.get("mass_balance", {}).get("error", 0) > 1e-3:
        why.append(f"mass balance off by {ch['mass_balance']['error']:.2%}")
    mb = ch.get("momentum_balance", {})
    if "note" not in mb and mb.get("error", 0) > 0.03:
        why.append(f"momentum balance off by {mb['error']:.2%}")
    if ch.get("energy_balance", {}).get("error", 0) > 0.02:
        why.append(f"energy balance off by {ch['energy_balance']['error']:.2%}")
    if why:
        why.append("do not report these numbers as results: rerun with more solve.iterations, lower solve.relax "
                   "(e.g. 0.5) or a better mesh, and check that the quantities of interest stop changing")
    return not why, why


def _archive(case: Path, path: Path) -> str | None:
    try:
        with tarfile.open(path, "w:gz") as tf:
            for p in sorted(case.iterdir()):
                if not p.name.startswith("processor"):
                    tf.add(p, arcname=f"case/{p.name}")
        return str(path)
    except OSError:
        return None


def _save(out: dict, folder: Path, name: str) -> dict:
    (folder / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
    return out
