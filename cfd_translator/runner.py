"""runner.py — one study end to end: mesh, patches, case, solve, results, checks.
Version: 2026.10.18.9

    run_study(name, study, folder, solver="foam" | "ref", timeout) -> result dict (also <folder>/<name>.json)
    prepare(study, folder) -> (mesh, patches, selection, plan, warnings)   (no solve)
"""
from __future__ import annotations

import json
import os
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
    procs_req = int((study.get("solve") or {}).get("procs") or os.environ.get("CFDT_PROCS", "1") or 1)
    out["execution"] = execution_record(procs_req)
    if (folder / "cancel").exists():
        out.update(ok=False, state="cancelled", why="cancelled before it started")
        return _save(out, folder, name)
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
            out["execution"] = execution_record(procs_req, 1)
        else:
            from . import foam
            procs = int(study.get("solve", {}).get("procs") or os.environ.get("CFDT_PROCS", "1") or 1)
            per = int(os.environ.get("CFDT_CELLS_PER_RANK", "5000"))
            use = max(1, min(procs, mesh.ncells // max(per, 1)))
            if use < procs:
                warns.append(f"ran on {use} MPI rank{'s' if use > 1 else ''} instead of {procs}: {mesh.ncells} cells "
                             f"(at least {per} per rank, CFDT_CELLS_PER_RANK)")
            info = foam.run(case, timeout, use)
            out["execution"] = execution_record(procs_req, use, info.get("launch"), info.get("binding", "default"))
            fields = foam.read_fields(case, mesh, plan)
            lg = info["log"]
            out["solve"] = {"openfoam": info.get("openfoam"), "procs": info.get("procs"), "iterations": lg["iterations"],
                            "converged": lg["converged"], "residuals": lg["residuals"],
                            "first_residuals": lg["first_residuals"], "time": fields["time"],
                            "checkMesh": info["checkMesh"]}
            if "continuity" in lg:
                out["solve"]["continuity"] = lg["continuity"]
            if not lg["converged"]:
                zero = zero_fields(fields, plan)
                rest = {f: r for f, r in velocity_scaled(lg["residuals"], fields).items() if f not in zero}
                if rest and max(rest.values()) <= plan["residual"] and (zero or rest != lg["residuals"]):
                    lg["converged"] = True
                    out["solve"]["converged"] = True
                    why = []
                    if any(z in lg["residuals"] for z in zero):
                        why.append(f"ignoring {', '.join(z for z in zero if z in lg['residuals'])} (zero or uniform "
                                   "everywhere: their residuals are round-off)")
                    small = [f for f in rest if rest[f] != lg["residuals"][f]]
                    if small:
                        why.append(f"{', '.join(small)} scaled by their size relative to the peak speed")
                    out["solve"]["converged_by"] = "residuals, " + "; ".join(why)
                    out["solve"]["residuals_scaled"] = {f: float(f"{r:.3g}") for f, r in rest.items()}
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
        from .foam import Cancelled
        if isinstance(exc, Cancelled):
            out.update(ok=False, state="cancelled", why="cancelled while running", warnings=warns)
            return _save(out, folder, name)
        out.update(ok=False, why=type(exc).__name__, error=str(exc)[:3000], warnings=warns)
        out["trace"] = traceback.format_exc()[-2000:]
    return _save(out, folder, name)


def execution_record(procs_requested: int, procs_used: int | None = None, launch: str | None = None,
                     binding: str = "default") -> dict:
    """The run record of llm-translators' docs/EXECUTION.md: where the study ran and on what."""
    from .foam import hwthreads, physical_cores
    job = os.environ.get("SLURM_JOB_ID")
    return {"executor": "slurm" if job else os.environ.get("CFDT_EXECUTION", "local"), "slurm_job": job,
            "procs_requested": int(procs_requested), "procs_used": procs_used,
            "mpi": {"launch": launch, "binding": binding, "cores": physical_cores(), "hwthreads": hwthreads()}
            if launch else None}


def velocity_scaled(residuals: dict, fields: dict) -> dict:
    """OpenFOAM normalises each velocity component's residual by that component's own variation; for a small
    component (heated channel: |Uy| <= 1.3e-3 of the peak speed, residual stuck at 7e-3) that is round-off.
    Relative to the flow, the residual is smaller by max|U_i| / max|U|; other fields are left as they are."""
    Uc = np.asarray(fields.get("U", []), float)
    if not Uc.size:
        return dict(residuals)
    umax = float(np.abs(Uc).max())
    out = dict(residuals)
    for k, c in enumerate("xyz"):
        f = f"U{c}"
        if f in out and umax > 0:
            out[f] = residuals[f] * min(1.0, float(np.abs(Uc[:, k]).max()) / umax)
    return out


def zero_fields(fields: dict, plan: dict) -> list[str]:
    """Solved fields that are zero (velocity components, within 1e-4 of the peak speed) or uniform (pressure,
    enthalpy, within 1e-6 of their scale) everywhere. Their normalised residuals are round-off over a near-zero
    normalisation and never meet the target (exact solutions such as Couette flow: Uy = 0 and p = 0; isothermal
    energy runs: h)."""
    Uc = np.asarray(fields["U"], float)
    if not Uc.size:
        return []
    umax = float(np.abs(Uc).max())
    # a velocity component below 1e-4 of the peak speed cannot affect the results; its normalised residual can
    # stall far above the target (heated channel on mochi: Uy residual 7e-3 while Nu was within 1.5 %)
    out = [f"U{c}" for k, c in enumerate("xyz") if float(np.abs(Uc[:, k]).max()) <= 1e-4 * umax]
    en = plan.get("energy")
    rho = en["rho"] if en else 1.0                      # incompressible p is kinematic
    pc = np.asarray(fields["p"], float)
    if pc.size and float(np.ptp(pc)) <= 1e-6 * max(0.5 * rho * umax ** 2, 1e-300):
        out += ["p", "p_rgh"]
    T = fields.get("T")
    # no heat put in and low Mach (case.py): T varies only by the kinetic energy, up to u^2 / (2 cp)
    dT = 1e-6 * en["T_op"] + (umax ** 2 / en["cp"] if en and en.get("isothermal") else 0.0) if en else 0.0
    if en and T is not None and float(np.ptp(np.asarray(T, float))) <= dT:
        out.append("h")
    return out


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
    eb = ch.get("energy_balance", {})
    if "note" not in eb and eb.get("error", 0) > 0.02:
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
    """Atomically (temporary file, then rename): study_status never reads half a result."""
    from .executor import write_atomic
    write_atomic(folder / f"{name}.json", json.dumps(out, indent=1, default=float))
    return out
