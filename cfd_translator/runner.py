"""runner.py — one study end to end: mesh, patches, case, solve, results, checks.
Version: 2026.11.12.18

    run_study(name, study, folder, solver="foam" | "ref", timeout) -> result dict (also <folder>/<name>.json)
    prepare(study, folder) -> (mesh, patches, selection, plan, warnings)   (no solve)
    reprocess_study(folder, name) -> summary   results again from a finished case, without solving
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


# what decides the mesh and the patches: two studies that agree on these mesh the same way (an NPSH sweep's points)
MESH_KEYS = ("geometry", "domain", "mesh", "bcs", "rotating", "porous", "2d")
MESHED = "meshed.pkl"


def mesh_key(study: dict) -> str:
    import hashlib
    d = {k: study.get(k) for k in MESH_KEYS}
    d["version"] = __version__
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _request_studies(folder: Path) -> dict:
    try:
        return json.loads((Path(folder).parent / "request.json").read_text()).get("studies") or {}
    except (OSError, ValueError):
        return {}


def _mesh_dependents(study: dict, folder: Path) -> list[str]:
    """Studies of the same request that start from this one (init_from '<this study>') with the same mesh key: they
    take this study's mesh instead of meshing again."""
    from .transient import init_ref
    key, me = mesh_key(study), Path(folder).name
    return [n for n, st in _request_studies(folder).items()
            if n != me and isinstance(st, dict) and init_ref(st)[0] == me and mesh_key(st) == key]


def _reused_mesh(study: dict, folder: Path):
    """The mesh of the run this study starts from (init_from), when it was meshed from the same geometry, mesh
    settings and bcs (mesh_key) and saved for it (meshed.pkl): (mesh, patches, selection, warns, source) or None.
    A source of the same request still meshing is waited for (its SLURM job ran first; a local one runs alongside)."""
    import pickle

    from .transient import init_ref, init_source
    ref, _ = init_ref(study)
    if not ref:
        return None
    folder = Path(folder)
    src = init_source(ref, folder.parent).resolve()
    key = mesh_key(study)
    pf = src / MESHED
    if not pf.is_file() and src.parent == folder.resolve().parent:
        sst = _request_studies(folder).get(src.name)
        if not (isinstance(sst, dict) and mesh_key(sst) == key):
            return None
        t_end = time.time() + float(os.environ.get("CFDT_MESH_WAIT_S", "86400"))
        while not pf.is_file() and time.time() < t_end and not (src / f"{src.name}.json").is_file():
            if (folder / "cancel").exists():
                return None
            time.sleep(10.0)
    if not pf.is_file():
        return None
    try:
        with open(pf, "rb") as f:
            rec = pickle.load(f)
    except Exception:  # noqa: BLE001
        return None
    if rec.get("key") != key:
        return None
    others = [n for n, st in _request_studies(src).items()
              if n not in (src.name, folder.name) and isinstance(st, dict) and init_ref(st)[0] == src.name]
    if not others:                            # the next point saves its own: keep one copy at a time
        try:
            pf.unlink()
        except OSError:
            pass
    return rec["mesh"], rec["patches"], rec["selection"], list(rec["warns"]), src


def prepare(study: dict, folder: Path):
    from .case import write_case
    from .mesher import mesh_study
    from .select import assign_bcs, mesh_groups

    external = study.get("domain", {}).get("kind") == "external"
    reused = _reused_mesh(study, folder)
    if reused is not None:
        mesh, patches, selection, warns, src = reused
        warns = [w for w in warns if not w.startswith("mesh reused from")] + [
            f"mesh reused from {src} (same geometry, mesh settings and bcs): not meshed again"]
        mesh.info["reused_from"] = str(src)
    else:
        mesh = mesh_study(study, folder)
        if "assigned" in mesh.info:            # snappyHexMesh: the bcs were chosen on the part's surface before meshing
            patches, selection, warns = mesh.info.pop("assigned")
        else:
            groups = mesh_groups(mesh, external)
            patches, selection, warns = assign_bcs(groups, study, mesh.info.get("empty_tags", []))
    if _mesh_dependents(study, folder):
        # the next study of a chain (an NPSH sweep's next point) takes this mesh as it is before the case is written
        import pickle
        try:
            tmp = Path(folder) / (MESHED + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"key": mesh_key(study), "mesh": mesh, "patches": patches, "selection": selection,
                             "warns": list(warns)}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, Path(folder) / MESHED)
        except Exception as exc:  # noqa: BLE001
            warns = list(warns) + [f"could not save the mesh for the next study of the chain ({exc}): it meshes again"]
    try:
        plan = write_case(study, mesh, patches, Path(folder) / "case")
    except Exception as exc:
        # what meshing found (unmeshed CAD faces, retried surfaces, narrow gaps) belongs in the failed result too:
        # S50 (2026.11.01.5) stopped in the wall-leak check and its mesh notes were lost
        exc.notes = list(getattr(exc, "notes", [])) + [w for w in list(warns) + mesh.info.get("mesh_notes", [])
                                                       if w not in getattr(exc, "notes", [])]
        raise
    return mesh, patches, selection, plan, list(warns) + mesh.info.pop("mesh_notes", []) + plan.pop("notes", [])


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

    from . import foam as _foam
    _foam.PEAK_MB.clear()                     # this study's steps only (a worker runs several studies)

    def record(*a, **k):
        return {**execution_record(*a, **k), "limits": limits,
                **({"memory_peak_mb": dict(_foam.PEAK_MB)} if _foam.PEAK_MB else {})}
    out["execution"] = record(procs_req)
    if (folder / "cancel").exists():
        out.update(ok=False, state="cancelled", why="cancelled before it started")
        return _save(out, folder, name)
    transient = solver != "ref" and bool((study.get("solve") or {}).get("transient"))
    t_job = time.time()
    watcher = None
    if os.environ.get("SLURM_JOB_ID") and solver != "ref":       # anything else on this job's CPUs? (cpushare.py)
        from .cpushare import Watcher
        watcher = Watcher(folder, float(os.environ.get("CFDT_CPU_WATCH_S", "60"))).start()
    try:
        init_src = _init_source(study, folder)
        if init_src is not None:              # the case writer and the transient run see where the start comes from
            study = _with_init(study, str(init_src))
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
                                                       "boundary_layers_on", "cells_xyz", "snappy", "regions_removed", "reused_from") if k in mesh.info})
        try:                                   # for the progress of a running study (api._progress)
            from .executor import write_atomic
            write_atomic(folder / "mesh.json", json.dumps({**out["mesh"], "warnings": list(warns)}, default=float))
        except (OSError, TypeError, ValueError):
            pass
        q = plan["quality"]
        if q.get("max_non_orthogonality", 0) > 80 and q.get("worst_non_ortho"):
            gaps = ((mesh.info.get("snappy") or {}).get("narrow_gaps") or [])
            warns.append(f"{q['non_ortho_over_70']} faces are more than 70 degrees non-orthogonal (worst "
                         f"{q['max_non_orthogonality']:.1f}): " + "; ".join(f"{w['deg']} at {w['at']}" for w in q["worst_non_ortho"][:3])
                         + " mm. The pressure solver crawls on such cells (three correctors, lower relaxation)"
                         + (f"; the narrow gaps found are at {', '.join(str(g['at']) for g in gaps[:3])}" if gaps else "")
                         + ". Refine or widen the gap there, or leave layers out of it (mesh.layers.on)")
        if q.get("max_skewness", 0) > 4 and q.get("worst_skewness"):
            warns.append(f"{q['skewness_over_4']} faces have skewness above 4 (worst {q['max_skewness']:.3g}, as "
                         "checkMesh computes it): " + "; ".join(f"{w['skewness']} at {w['at']}" for w in q["worst_skewness"][:4])
                         + " mm. Usually a refinement-level jump or a snapped cell squeezed against a wall: move the "
                         "refinement region's edge away from there, or refine around it")
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
            if not os.environ.get("SLURM_JOB_ID"):          # a local run: this machine's cores (foam.local_procs)
                procs, cap_note = foam.local_procs(procs)
                if cap_note:
                    warns.append(cap_note)
            per = int(os.environ.get("CFDT_CELLS_PER_RANK", "5000"))
            use = max(1, min(procs, mesh.ncells // max(per, 1)))
            if use < procs:
                warns.append(f"ran on {use} MPI rank{'s' if use > 1 else ''} instead of {procs}: {mesh.ncells} cells "
                             f"(at least {per} per rank, CFDT_CELLS_PER_RANK)")
            if transient:
                info = _solve_transient(case, study, plan, timeout, use, folder, tstate, t_job, warns, mesh)
                if info.get("continued"):
                    if watcher is not None:
                        _cpu_note(watcher.stop(), info)
                    return {"name": name, "continued": True, **info}      # the next SLURM job carries on
            else:
                if init_src is not None:
                    out["init_from"] = _init_steady(case, study, plan, init_src, folder, t_job, timeout, warns)
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
                from .transient import courant_stats
                co = courant_stats(case / "log.foamRun")
                segs = tstate.get("courant") or []
                out["solve"]["time_step"] = {"fixed": bool(tr.get("fixed_step")), "dt": tr.get("dt"),
                                             **({"max_co": tr["max_co"]} if tr.get("max_co") else {}),
                                             **({"courant": co} if co else {}),
                                             **({"courant_by_segment": segs} if segs else {})}
                if tr.get("fixed_step") and co and co["max_median"] > 50:
                    warns.append(f"fixed time step: the largest Courant number is {co['max_median']:.3g} most steps "
                                 f"(mean {co['mean_last']:.3g}): watch the results for noise from those cells, or "
                                 "lower degrees_per_step / dt")
                if plan.get("sliding"):
                    sl = plan["sliding"]
                    out["solve"]["sliding"] = {
                        "zone": sl["zone"], "revolutions": tr.get("revolutions"),
                        "averaged_revolutions": tr.get("average_revolutions"), "revolution_s": tr.get("period"),
                        "degrees_per_step": tr.get("degrees_per_step"), "interface_faces": sl["interface_faces"],
                        "interface_lean_deg": sl["interface_lean_deg"], "rotor_walls": sl["moving_walls"],
                        "spinning_walls": sl.get("spinning_walls", []), "interface_cells": sl.get("interface_cells"),
                        "correct_mesh_phi": tr.get("correct_mesh_phi", True),
                        **({"ncc_owner": sl["ncc_owner"], "ncc_owner_by": sl.get("ncc_owner_by")}
                           if sl.get("ncc_owner") else {})}
                    from .transient import stitcher_stats
                    sst = stitcher_stats(case / "log.foamRun")
                    if sst:
                        out["solve"]["sliding"]["stitcher"] = sst
                        pv = (sl.get("interface_cells") or {}).get("projected_volume_fraction") or {}
                        side = "zone_owner" if sl.get("ncc_owner") == sl.get("rotor") else "other_owner"
                        est = (pv.get(side) or {}).get("max")
                        sst["estimated_before_run"] = est
                        if sst["projected_volume_fraction_max"] > 0.3:
                            warns.append(f"sliding interface: OpenFOAM's projected volume fraction reached "
                                         f"{sst['projected_volume_fraction_max']:g} ({sst['over_0.3']} of the last "
                                         f"{sst['seen']} steps above 0.3; estimated before the run: {est}): the "
                                         "stitched cells are distorted; results there may be noisy")
                try:
                    from .transient import step_profile
                    sp = step_profile((case / "log.foamRun").read_text(errors="replace")[-400000:])
                    if sp:
                        out["solve"]["step_profile"] = sp
                except OSError:
                    pass
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
        if transient:                             # p rms and the time below pressure thresholds (p_below)
            from .transient import pressure_statistics
            ps = pressure_statistics(study, mesh, plan, fields)
            if ps:
                res["pressure_statistics"] = ps
        if plan.get("vof"):                       # cavitation: vapour volume and where, NPSH at the inlet
            from .cavitation import results as _cav
            res["cavitation"], w = _cav(study, mesh, plan, fields, res)
            warns += w
        warns += _onset_into(study, mesh, plan, fields, res)
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
        if _foam.PEAK_MB and isinstance(out.get("execution"), dict):
            out["execution"]["memory_peak_mb"] = dict(_foam.PEAK_MB)     # where it stood when it stopped
        if isinstance(exc, Cancelled):
            out.update(ok=False, state="cancelled", why="cancelled while running", warnings=warns)
            return _save(out, folder, name)
        out.update(ok=False, why=type(exc).__name__, error=str(exc)[:3000],
                   warnings=warns + [n for n in getattr(exc, "notes", []) if n not in warns])
        from .foam import Killed
        if isinstance(exc, Killed):     # an OpenFOAM program killed for memory: said as such (it read 'failed')
            out["state"] = "out_of_memory"
        out["trace"] = traceback.format_exc()[-2000:]
    if watcher is not None:
        _cpu_note(watcher.stop(), out)
    return _save(out, folder, name)


def _init_source(study: dict, folder: Path) -> Path | None:
    """The study folder that solve.init_from / solve.transient.init_from names (same request: '<study>')."""
    from .transient import init_ref, init_source
    ref, _ = init_ref(study)
    return init_source(ref, Path(folder).parent).resolve() if ref else None


def _with_init(study: dict, src: str) -> dict:
    """The study with its init_from as an absolute folder (cavitation.write_vof reads the source's inlet pressure)."""
    s = dict(study.get("solve") or {})
    if (s.get("transient") or {}).get("init_from"):
        s["transient"] = dict(s["transient"], init_from=src)
    else:
        s["init_from"] = src
    return dict(study, solve=s)


def _source_ready(src: Path, folder: Path, t_job: float, timeout: float, warns: list) -> bool:
    """Wait for a source study of the same request (it runs before this one: a SLURM dependency, or this server's
    worker queue) and say whether it left fields to start from; a failed or missing one is warned about and the run
    starts from the uniform fields."""
    from .transient import source_times, wait_for
    src = Path(src)
    if not source_times(src):
        left = max(60.0, timeout - (time.time() - t_job) - 600)
        res = wait_for(src, left, Path(folder) / "cancel") if src.resolve().parent == Path(folder).resolve().parent else None
        if res is not None and not res.get("ok"):
            warns.append(f"init_from: {src.name} did not finish ({res.get('why') or res.get('state')}): this run starts "
                         "from the uniform fields")
            return False
    if not source_times(src):
        warns.append(f"init_from: {src} has no fields to start from: this run starts from the uniform fields")
        return False
    return True


def _init_steady(case: Path, study: dict, plan: dict, src: Path, folder: Path, t_job: float, timeout: float,
                 warns: list) -> dict | None:
    """solve.init_from on a steady run (MRF, cavitation): the source's fields in 0/ before the solve. Cavitation
    runs then start from the source's inlet pressure and at their full flow (cavitation.restart_tables)."""
    from . import foam
    from .transient import init_ref, map_from, map_timeout
    if not _source_ready(src, folder, t_job, timeout, warns):
        return {"from": str(src), "used": False}
    fm = foam.find_foam()
    try:
        notes, info = map_from(case, src, init_ref(study)[1],
                               lambda cmd, log: foam._step(fm, cmd, log, case, map_timeout()))
    except Exception as exc:  # noqa: BLE001
        warns.append(f"init_from: could not take the fields of {src} ({type(exc).__name__}: {exc}): this run starts "
                     "from the uniform fields")
        return {"from": str(src), "used": False}
    warns += notes
    if plan.get("vof"):
        from .cavitation import restart_tables
        warns += restart_tables(case, plan, src)
    return {"from": str(src), "used": True, **info}


def _cpu_note(summary: dict | None, out: dict) -> None:
    """Other processes on this job's CPUs: execution.cpu_sharing and a warning."""
    if not summary:
        return
    from .cpushare import warning
    out.setdefault("execution", {})["cpu_sharing"] = summary
    out.setdefault("warnings", []).append(warning(summary))


def reprocess_study(folder: Path, name: str) -> dict:
    """The results of a finished study again, from its case (the last time directory; transient: the mean fields)
    or its saved state, with the current post-processing and averaging, without solving. The old result is kept
    as <name>.json.<its version>.bak; the new one records reprocessed: {by, at, from, changes}."""
    from datetime import datetime

    from .post import check_expect, results
    from .render import load_state
    folder = Path(folder)
    f = folder / f"{name}.json"
    if not f.is_file():
        return {"name": name, "ok": False, "why": "no result yet (still running, or never started)"}
    old = json.loads(f.read_text())
    if not old.get("ok") or "results" not in old:
        return {"name": name, "ok": False, "why": "not a finished study with results"}
    if not (folder / "state.npz").is_file():
        return {"name": name, "ok": False, "why": "no saved mesh (state.npz)"}
    mesh, plan, fields, study = load_state(folder)
    if not study:
        return {"name": name, "ok": False, "why": "the saved state does not hold the study"}
    case = folder / "case"
    transient = bool((old.get("solve") or {}).get("transient"))
    source = "the saved state"
    from_case = False
    old_scalars = set((fields.get("scalars") or {}))
    if old.get("solver") != "ref" and case.is_dir():
        from .foam import read_fields
        try:
            fields = read_fields(case, mesh, plan, mean=transient)
            source = "the case" + (" (mean fields)" if fields.get("averaged") else "")
            from_case = True
        except Exception as exc:  # noqa: BLE001
            source = f"the saved state (the case could not be read: {exc})"
    upgraded = upgrade_plan(plan, study)
    res, w = results(study, mesh, plan, fields)
    if transient:
        from .transient import pressure_statistics
        ps = pressure_statistics(study, mesh, plan, fields)
        if ps:
            res["pressure_statistics"] = ps
    w = list(w) + _onset_into(study, mesh, plan, fields, res)
    if plan.get("vof"):
        if fields.get("alpha") is not None:
            from .cavitation import results as _cav
            res["cavitation"], w3 = _cav(study, mesh, plan, fields, res)
            w = list(w) + w3
        elif (old.get("results") or {}).get("cavitation"):
            res["cavitation"] = old["results"]["cavitation"]
    out = dict(old)
    solve = dict(old.get("solve") or {})
    if old.get("solver") != "ref" and plan.get("monitors") and case.is_dir():
        from .average import average_results
        try:
            avg, w2 = average_results(case, plan, res)
        except Exception as exc:  # noqa: BLE001
            avg, w2 = None, [f"averaging failed ({type(exc).__name__}: {exc}); final-iteration values kept"]
        w += w2
        if avg:
            res["averaged"] = avg
            solve["settled"] = bool(avg.get("settled"))
    if transient:
        for k in ("momentum_balance", "angular_momentum_balance"):
            if k in res.get("checks", {}):
                res["checks"][k]["note"] = ("from the time-averaged fields: the Reynolds stresses and the time "
                                            "derivative are not in it, so it is not judged")
    out["solve"] = solve
    out["results"] = res
    if study.get("expect"):
        out["expect"] = check_expect(res, study["expect"])
        out["passed"] = all(e["passed"] for e in out["expect"])
    kept = [x for x in old.get("warnings", []) if not x.startswith("averaging")]
    out["warnings"] = kept + [x for x in w if x not in kept]
    out.pop("unreliable_because", None)
    out["reliable"], why = _verdict(out, res)
    if why:
        out["unreliable_because"] = why
    oc, nc = old["results"].get("coefficients") or {}, res.get("coefficients") or {}
    changes = {k: [oc.get(k), nc.get(k)] for k in ("cd", "cl", "cm") if oc.get(k) != nc.get(k)}
    for k in ("part", "all_walls"):
        a_, b_ = (old["results"].get("forces") or {}).get(k), (res.get("forces") or {}).get(k)
        if a_ and b_ and any(a_.get(c) != b_.get(c) for c in ("fx", "fy", "fz")):
            changes[f"forces.{k}"] = [[a_.get(c) for c in ("fx", "fy", "fz")], [b_.get(c) for c in ("fx", "fy", "fz")]]
    for zn, z in (res.get("rotating") or {}).items():
        oz = (old["results"].get("rotating") or {}).get(zn) or {}
        for k in ("torque_Nm", "power_W"):
            if oz.get(k) != z.get(k):
                changes[f"rotating.{zn}.{k}"] = [oz.get(k), z.get(k)]
    for k in ("head_m", "power_W", "efficiency"):
        a_, b_ = (old["results"].get("pump") or {}).get(k), (res.get("pump") or {}).get(k)
        if a_ != b_:
            changes[f"pump.{k}"] = [a_, b_]
    if upgraded:
        changes["plan"] = upgraded
    out["reprocessed"] = {"by": __version__, "at": datetime.now().isoformat(timespec="seconds"),
                          "from": old.get("reprocessed", {}).get("by") or old.get("translator"), "fields": source,
                          "changes": changes}
    bak = folder / f"{name}.json.{old.get('translator', 'old')}.bak"
    if not bak.exists():
        bak.write_text(f.read_text())
    # the saved state from the fields just read, so render_images has what this version reads from a case (k, nut,
    # p_rms, p_below_* since 2026.11.12.4: run_20261108_s50_rc_open_s1's render_images field k gave KeyError 'k'
    # after reprocess, which read the case but kept the old state)
    added = []
    if from_case:
        from .render import save_state
        added = sorted(set(fields.get("scalars") or {}) - old_scalars)
        save_state(folder, mesh, plan, fields, study)
        out["reprocessed"]["state"] = {"saved": True, **({"fields_added": added} if added else {})}
    _save(out, folder, name)
    return {"name": name, "ok": True, "reliable": out["reliable"], "fields": source, "changes": changes,
            "backup": str(bak), **({"state_fields_added": added} if added else {})}


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


def _onset_into(study: dict, mesh, plan: dict, fields: dict, res: dict) -> list:
    """study.onset: results.onset and the p_eff field (images, render_images); its warnings."""
    if not study.get("onset"):
        return []
    from .onset import evaluate
    try:
        ob, p_eff, w = evaluate(study, mesh, plan, fields, res)
    except Exception as exc:  # noqa: BLE001 (a screen must never fail a finished run)
        return [f"onset: not evaluated ({type(exc).__name__}: {exc})"]
    if ob:
        res["onset"] = ob
        fields.setdefault("scalars", {})["p_eff"] = {"internal": p_eff, "patches": {}}
    return w


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
        if ch.get("wall_flux", {}).get("fraction_of_inflow", 0) > 0.02:
            why.append(f"{ch['wall_flux']['fraction_of_inflow']:.1%} of the inflow passes through walls (see warnings)")
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
    if ch.get("wall_flux", {}).get("fraction_of_inflow", 0) > 0.02:
        why.append(f"{ch['wall_flux']['fraction_of_inflow']:.1%} of the inflow passes through walls (see warnings)")
    if ch.get("gas_balance", {}).get("error", 0) > 0.05:
        gb = ch["gas_balance"]
        why.append(f"air not conserved: {gb['in_kg_s']:.3g} kg/s in, {gb['out_kg_s']:.3g} kg/s out "
                   f"({gb['error']:.0%})")
    if ch.get("alpha_bounds") and not ch["alpha_bounds"].get("ok", True):
        why.append(f"the liquid fraction left [0, 1] ({ch['alpha_bounds']['min']} .. {ch['alpha_bounds']['max']})")
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
             "continued_from": str(src), "continued_at": t, "sliding_done": bool(plan.get("sliding"))}
    with open(folder / "prepared.pkl", "wb") as f:
        pickle.dump((mesh, patches, selection, plan, list(warns) + notes), f)
    (folder / "transient.json").write_text(json.dumps(state))
    return mesh, patches, selection, plan, list(warns) + notes, state


def _solve_transient(case: Path, study: dict, plan: dict, timeout: float, procs: int, folder: Path, state: dict,
                     t_job: float, warns: list, mesh=None) -> dict:
    """The steady start (once), then the transient run up to this job's time budget. When end_time is not reached
    and this is a SLURM job, the next job is submitted (after this one) and {'continued': True, ...} returned."""
    from . import foam
    from .transient import after_init, write_transient
    tr = plan["transient"]
    sf = folder / "transient.json"

    def save():
        sf.write_text(json.dumps(state))
    margin = min(900.0, max(60.0, 0.1 * timeout))
    if state.get("correct_mesh_phi") is False:          # switched off by an earlier job of this run
        tr["correct_mesh_phi"] = False
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
    if tr.get("init_from") and not state.get("mapped") and not state.get("init_done"):
        # init_from: an earlier run's fields on this mesh instead of a uniform start (a variant that differs near one
        # wheel needs less start-up time)
        # once per run: transient.json's 'mapped' keeps later SLURM jobs from mapping again
        from .transient import map_from, map_timeout
        fm = foam.find_foam()
        notes = []
        if _source_ready(Path(tr["init_from"]), folder, t_job, timeout, notes):
            notes += map_from(case, tr["init_from"], tr.get("init_fields", "last"),
                              lambda cmd, log: foam._step(fm, cmd, log, case, map_timeout()))[0]
        state.update(mapped=True, init_notes=notes)
        save()
        warns += notes
    elif state.get("init_notes"):
        warns += [n for n in state["init_notes"] if n not in warns]
    if plan.get("sliding") and not state.get("sliding_done"):
        # the zone's cells start turning: the mesh is split at its cylinder and coupled non-conformally (sliding.py)
        from .sliding import to_sliding
        fm = foam.find_foam()
        if not (case / "log.checkMesh").exists():            # the conformal mesh, before it is split
            foam._step(fm, "checkMesh", "log.checkMesh", case, 900, raise_on_error=False)
        notes = to_sliding(case, plan, mesh, lambda cmd, log: foam._step(fm, cmd, log, case, 3600), procs,
                           foam.launch(procs) if procs > 1 else None)
        write_transient(case, study, plan)
        state.update(sliding_done=True, sliding_notes=notes)
        save()
        warns += notes
        resume = True
    elif state.get("sliding_notes"):
        warns += [n for n in state["sliding_notes"] if n not in warns]
    left = timeout - (time.time() - t_job)
    try:
        info = foam.run(case, max(left, 120.0), procs, resume=resume, deadline=t_job + timeout - margin)
    except foam.FoamError:
        # S50 on spark2 (2026.11.03): the first mesh motion crashed in OpenFOAM's mesh-flux correction at the
        # interface (fvMeshStitchers::moving::unconformInternalFaceCorrectMeshPhi, free(), exit 139). It is optional
        # (cfd.direct, 'Using NCC': conservation holds without it; it removes pressure noise): once more without it
        if not (plan.get("sliding") and tr.get("correct_mesh_phi", True) and mesh_phi_crash(case)):
            raise
        tr["correct_mesh_phi"] = False
        write_transient(case, study, plan)
        note = ("OpenFOAM crashed in the sliding interface's mesh-flux correction (unconformInternalFaceCorrectMeshPhi, "
                "see log.foamRun.part*): run again with PIMPLE correctMeshPhi off; expect some pressure noise at the "
                "interface. Poor cells there (sliding.interface_cells) are a likely cause: improve them and set "
                "solve.transient.correct_mesh_phi true again")
        warns.append(note)
        state["sliding_notes"] = list(state.get("sliding_notes", [])) + [note]
        state["correct_mesh_phi"] = False
        save()
        left = timeout - (time.time() - t_job)
        info = foam.run(case, max(left, 120.0), procs, resume=True, deadline=t_job + timeout - margin)
    state["time_steps"] = int(state.get("time_steps", 0)) + int(info["log"]["iterations"])
    state["time"] = info.get("time")
    from .transient import courant_stats
    co = courant_stats(case / "log.foamRun")       # this job's segment (the log is renamed when the next one starts)
    if co:
        state.setdefault("courant", []).append({"segment": int(state.get("segments", 0)) + 1, "time": info.get("time"),
                                                **co})
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
    if procs > 1 and not foam.decomposed(case):
        fm = foam.find_foam()
        foam._step(fm, "reconstructPar -latestTime", "log.reconstructPar", case, 3600)
    return info


def upgrade_plan(plan: dict, study: dict) -> list[str]:
    """A saved plan brought up to the current case writer's rules, for reprocess: walls with their own rotation at a
    zone's axis and speed count in the zone's torque (2026.11.03.5). The torque monitor of such an older run held
    the zone's named walls only (monitor_walls). Returns what changed."""
    from .case import same_speed_walls
    from .format import bc_names
    done = []
    bcs = [(n, b) for n, b in zip(bc_names(study), study.get("bcs") or []) if n in (plan.get("patches") or {})]
    radii = {z.get("name"): z.get("radius") for z in study.get("rotating") or [] if isinstance(z, dict)}
    for zn, zp in (plan.get("zones") or {}).items():
        if "radius_mm" not in zp and radii.get(zn) is not None:
            zp["radius_mm"] = float(radii[zn])
        if "rotating_walls" not in zp:
            continue
        zp.setdefault("monitor_walls", list(zp["rotating_walls"]))
        add = [n for n in same_speed_walls(bcs, zp) if n not in zp["rotating_walls"]]
        if add:
            zp["rotating_walls"] = list(zp["rotating_walls"]) + add
            done.append(f"zone {zn}: {', '.join(add)} (own rotation at the zone's speed) now count in its torque")
    return done


def mesh_phi_crash(case: Path) -> bool:
    """Did foamRun stop in the sliding interface's mesh-flux correction (a crash, not a fatal error)?"""
    try:
        t = (Path(case) / "log.foamRun").read_text(errors="replace")[-60000:]
    except OSError:
        return False
    return "CorrectMeshPhi" in t and any(k in t for k in ("Segmentation", "sigSegv", "free()", "signal 11", "Aborted",
                                                          "exited on signal", "SIGSEGV", "double free"))


def _save(out: dict, folder: Path, name: str) -> dict:
    """Atomically (temporary file, then rename): study_status never reads half a result."""
    from .executor import write_atomic
    write_atomic(folder / f"{name}.json", json.dumps(out, indent=1, default=float))
    return out
