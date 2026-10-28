"""runner.py — one study end to end: mesh, patches, case, solve, results, checks.
Version: 2026.10.28.1

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
    if "assigned" in mesh.info:            # snappyHexMesh: the bcs were chosen on the part's surface before meshing
        patches, selection, warns = mesh.info.pop("assigned")
    else:
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
    from .executor import procs_of
    procs_req = procs_of(study)
    # what bounded this run, and which of it the study set (study.run) rather than config/env
    mem = os.environ.get("SLURM_MEM_PER_NODE")
    run_over = study.get("run") or {}
    limits = {"timeout_s": float(timeout), "mem": f"{int(mem)}M" if mem and mem.isdigit() else None,
              "from_study": sorted(k for k in run_over if k in ("timeout_s", "mem", "procs", "queue"))}

    def record(*a, **k):
        return {**execution_record(*a, **k), "limits": limits}
    out["execution"] = record(procs_req)
    if (folder / "cancel").exists():
        out.update(ok=False, state="cancelled", why="cancelled before it started")
        return _save(out, folder, name)
    transient = solver != "ref" and bool((study.get("solve") or {}).get("transient"))
    t_job = time.time()
    try:
        if transient:
            mesh, patches, selection, plan, w, tstate = _prepare_transient(study, folder)
        else:
            mesh, patches, selection, plan, w = prepare(study, folder)
        warns += w
        t1 = time.perf_counter()
        case = folder / "case"
        out["selection"] = selection
        out["mesh"] = {k: v for k, v in plan["quality"].items()}
        out["mesh"].update({k: mesh.info[k] for k in ("kind", "size", "layers", "cells_2d", "quads_2d",
                                                       "boundary_layers_on", "cells_xyz", "snappy") if k in mesh.info})
        out["case"] = {"inlets": plan["inlets"], "schemes": plan["schemes"], "iterations": plan["iterations"],
                       "residual": plan["residual"], "relax": plan["relax"]}
        if solver == "ref":
            from .refsolver import solve_case
            fields = solve_case(case)
            out["solve"] = {"solver": "reference (2D MAC, Picard)", "iterations": fields["iterations"],
                            "converged": fields["converged"], "change": fields["change"]}
            if not fields["converged"]:
                warns.append("reference solver did not converge")
            out["execution"] = record(procs_req, 1)
        else:
            from . import foam
            procs = procs_of(study)
            per = int(os.environ.get("CFDT_CELLS_PER_RANK", "5000"))
            use = max(1, min(procs, mesh.ncells // max(per, 1)))
            if use < procs:
                warns.append(f"ran on {use} MPI rank{'s' if use > 1 else ''} instead of {procs}: {mesh.ncells} cells "
                             f"(at least {per} per rank, CFDT_CELLS_PER_RANK)")
            if transient:
                info = _solve_transient(case, study, plan, timeout, use, folder, tstate, t_job, warns)
                if info.get("continued"):
                    return {"name": name, "continued": True, **info}      # the next SLURM job carries on
            else:
                info = foam.run(case, timeout, use)
            out["execution"] = record(procs_req, use, info.get("launch"), info.get("binding", "default"))
            fields = foam.read_fields(case, mesh, plan, mean=transient)
            lg = info["log"]
            out["solve"] = {"openfoam": info.get("openfoam"), "procs": info.get("procs"), "iterations": lg["iterations"],
                            "converged": lg["converged"], "residuals": lg["residuals"],
                            "first_residuals": lg["first_residuals"], "time": fields["time"],
                            "checkMesh": info["checkMesh"]}
            if transient:
                tr = plan["transient"]
                out["solve"].update(converged=False, transient=True, model=tr["model"], end_time=tr["end_time"],
                                    reached_time=info.get("time"), time_steps=tstate.get("time_steps"),
                                    jobs=tstate.get("segments", 0) + 1, init_iterations=tstate.get("init_iterations"),
                                    averaged_fields=bool(fields.get("averaged")))
                out["solve"].pop("first_residuals", None)
                lg = dict(lg, converged=True)            # residual convergence is not the question in time
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
        if solver != "ref" and plan.get("monitors"):
            from .average import average_results
            try:
                avg, w = average_results(case, plan, res)
            except Exception as exc:  # noqa: BLE001
                avg, w = None, [f"averaging failed ({type(exc).__name__}: {exc}); final-iteration values kept"]
            warns += w
            if avg:
                res["averaged"] = avg
                out["solve"]["settled"] = bool(avg.get("settled"))
                if avg.get("settled"):
                    q = avg["quantities"]
                    if transient:
                        out["solve"]["settled_by"] = "; ".join(
                            f"{k}: drift {q[k]['drift']:.2%}, standard error {q[k]['stderr']:.2%}" for k in avg["judged"]) + \
                            f" over t = {q[avg['judged'][0]]['window'][0]}..{q[avg['judged'][0]]['window'][1]} s"
                    else:
                        out["solve"]["settled_by"] = "; ".join(
                            f"{k}: drift {q[k]['drift']:.2%}, scatter {q[k]['scatter']:.2%}" for k in avg["judged"]) + \
                            f" over iterations {q[avg['judged'][0]]['window'][0]}..{q[avg['judged'][0]]['window'][1]}"
        if transient:
            for k in ("momentum_balance", "angular_momentum_balance"):
                if k in res.get("checks", {}):
                    res["checks"][k]["note"] = ("from the time-averaged fields: the Reynolds stresses and the time "
                                                "derivative are not in it, so it is not judged")
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
        try:                                   # ParaView opens a case through an empty <name>.foam file
            (case / "case.foam").touch()
        except OSError:
            pass
        out["files"] = {"case": str(case), "archive": _archive(case, folder / f"{name}_case.tgz")}
        if (case / "case.foam").is_file():
            out["files"]["paraview"] = str(case / "case.foam")
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
    component (heated channel: |Uy| <= 1e-6 m/s against 7.5e-4, residual stuck at 7e-3) that is round-off.
    Relative to the flow, the residual is smaller by mean|U_i| / mean|U|; other fields are left as they are."""
    Uc = np.asarray(fields.get("U", []), float)
    if not Uc.size:
        return dict(residuals)
    umean = float(np.linalg.norm(Uc, axis=1).mean())
    out = dict(residuals)
    for k, c in enumerate("xyz"):
        f = f"U{c}"
        if f in out and umean > 0:      # mean, like OpenFOAM's normalisation (a sum over the cells)
            out[f] = residuals[f] * min(1.0, float(np.abs(Uc[:, k]).mean()) / umean)
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
    for n, sf in (fields.get("scalars") or {}).items():    # a tracer that is uniform (one value at every inlet)
        c = np.asarray(sf["internal"], float)
        if c.size and float(np.ptp(c)) <= 1e-6 * max(float(np.abs(c).max()), 1e-300):
            out.append(n)
    return out


def _verdict(out: dict, res: dict) -> tuple[bool, list]:
    """Whether the numbers may be reported as results: converged (residuals) or settled (the monitored
    quantities neither drift nor scatter over the averaging window), mass and momentum balanced."""
    why = []
    sv = out.get("solve", {})
    if sv.get("transient"):
        av = res.get("averaged", {})
        if not sv.get("settled"):
            why.append("the time averages have not converged "
                       f"({'; '.join(av.get('why', [])) or 'no averaged history: see results.averaged'})")
        if sv.get("reached_time") is not None and sv["reached_time"] < sv.get("end_time", 0) * (1 - 1e-6):
            why.append(f"the run stopped at t = {sv['reached_time']} s of {sv['end_time']} s")
        ch = res.get("checks", {})
        if ch.get("mass_balance", {}).get("error", 0) > 1e-3:
            why.append(f"mass balance off by {ch['mass_balance']['error']:.2%}")
        if why:
            why.append("do not report these numbers as results: raise solve.transient.end_time (a longer average), "
                       "or check the flow with images")
        return not why, why
    if not sv.get("converged") and not sv.get("settled"):
        if "settled" in sv:
            av = res.get("averaged", {})
            why.append("the solver did not converge (see solve.residuals) and the monitored quantities have not "
                       f"settled ({'; '.join(av.get('why', [])) or 'see results.averaged'})")
        else:
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
    for n, sb in (ch.get("scalar_balance") or {}).items():
        if sb.get("error", 0) > 1e-2:
            why.append(f"tracer {n}: balance off by {sb['error']:.2%}")
    if ch.get("age_balance", {}).get("error", 0) > 1e-2:
        why.append(f"age: balance off by {ch['age_balance']['error']:.2%}")
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


def _prepare_transient(study: dict, folder: Path):
    """prepare(), once: a later SLURM job of the same transient study reloads the mesh, patches and plan instead
    of meshing and writing the case again. Returns (..., state) with state from <folder>/transient.json."""
    import pickle
    sf, pf = folder / "transient.json", folder / "prepared.pkl"
    state = json.loads(sf.read_text()) if sf.is_file() else {}
    if state.get("prepared") and pf.is_file():
        with open(pf, "rb") as f:
            mesh, patches, selection, plan, warns = pickle.load(f)
        return mesh, patches, selection, plan, list(warns), state
    tr = (study.get("solve") or {}).get("transient") or {}
    if tr.get("continue"):
        return _continue_transient(study, folder)
    mesh, patches, selection, plan, warns = prepare(study, folder)
    with open(pf, "wb") as f:
        pickle.dump((mesh, patches, selection, plan, warns), f)
    state = {"prepared": True, "segments": 0, "init_done": False, "started": False, "time_steps": 0}
    sf.write_text(json.dumps(state))
    return mesh, patches, selection, plan, warns, state


def _continue_transient(study: dict, folder: Path):
    """solve.transient.continue: a finished transient run, carried on to a later end_time in a new study folder.
    Its prepared mesh and plan, the case (latest time only, decomposed or not) and the monitors' history are copied;
    the averages go on, or start again from average_from when that is later than the old run's."""
    import pickle

    from .transient import continue_problems, copy_case, restart_averages, settings, source_folder, write_transient
    bad = continue_problems(study)
    if bad:
        raise ValueError(bad[0])
    tr = study["solve"]["transient"]
    src = source_folder(tr["continue"])
    with open(src / "prepared.pkl", "rb") as f:
        mesh, patches, selection, plan, warns = pickle.load(f)
    old = json.loads((src / "transient.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    t = copy_case(src / "case", folder / "case")
    old_from = float((plan.get("transient") or {}).get("average_from", 0.0))
    new = settings(study, mesh)
    if "average_from" not in tr:
        new["average_from"] = old_from
    new["init"] = 0
    plan["transient"] = new
    notes = [f"continued from {src} at t = {t:g} s"]
    if new["average_from"] > old_from * (1 + 1e-9) and new["average_from"] > 0:
        restart_averages(folder / "case")
        notes.append(f"time averages start again at {new['average_from']:g} s")
    write_transient(folder / "case", study, plan)
    state = {"prepared": True, "segments": 0, "init_done": True, "started": True,
             "time_steps": int(old.get("time_steps", 0)), "init_iterations": old.get("init_iterations"),
             "continued_from": str(src), "continued_at": t}
    with open(folder / "prepared.pkl", "wb") as f:
        pickle.dump((mesh, patches, selection, plan, list(warns) + notes), f)
    (folder / "transient.json").write_text(json.dumps(state))
    return mesh, patches, selection, plan, list(warns) + notes, state


def _solve_transient(case: Path, study: dict, plan: dict, timeout: float, procs: int, folder: Path, state: dict,
                     t_job: float, warns: list) -> dict:
    """The steady start (once), then the transient run up to this job's time budget. When end_time is not reached
    and this is a SLURM job, the next job is submitted (after this one) and {'continued': True, ...} returned."""
    from . import foam
    from .transient import after_init, write_transient
    tr = plan["transient"]
    sf = folder / "transient.json"

    def save():
        sf.write_text(json.dumps(state))
    margin = min(900.0, max(60.0, 0.1 * timeout))
    if tr["init"] and not state.get("init_done"):
        info0 = foam.run(case, timeout, procs)
        state["init_iterations"] = info0["log"]["iterations"]
        after_init(case)
        write_transient(case, study, plan)
        state.update(init_done=True, started=True)
        save()
        resume = True
    else:
        resume = bool(state.get("started"))
        state["started"] = True
        save()
    left = timeout - (time.time() - t_job)
    info = foam.run(case, max(left, 120.0), procs, resume=resume, deadline=t_job + timeout - margin)
    state["time_steps"] = int(state.get("time_steps", 0)) + int(info["log"]["iterations"])
    state["time"] = info.get("time")
    if info.get("reached_end"):
        save()
        return info
    state["segments"] = int(state.get("segments", 0)) + 1
    save()
    job = os.environ.get("SLURM_JOB_ID")
    if job and state["segments"] < tr["max_jobs"]:
        from .executor import submit
        nxt = submit(folder.parent, folder.name, study, after=job, segment=state["segments"] + 1)
        if nxt.get("job_id"):
            return {"continued": True, "time": info.get("time"), "end_time": tr["end_time"], "next_job": nxt["job_id"],
                    "segment": state["segments"] + 1}
        warns.append(f"could not submit the next job ({nxt.get('why')}): results at t = {info.get('time')} s")
    else:
        warns.append(f"the run stopped at t = {info.get('time')} s of {tr['end_time']} s "
                     f"({'max_jobs reached' if job else 'one local run: raise run.timeout_s'}): the averages cover less "
                     "time than asked")
    if procs > 1:
        fm = foam.find_foam()
        foam._step(fm, "reconstructPar -latestTime", "log.reconstructPar", case, 3600)
    return info


def _save(out: dict, folder: Path, name: str) -> dict:
    """Atomically (temporary file, then rename): study_status never reads half a result."""
    from .executor import write_atomic
    write_atomic(folder / f"{name}.json", json.dumps(out, indent=1, default=float))
    return out
