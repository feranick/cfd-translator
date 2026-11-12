"""executor.py — where studies run: the local worker pool, or one SLURM job per study.
Version: 2026.11.12.5

From 2026.11.09.4 the SLURM layer is llmt-runtime's (llmt_runtime.slurm and llmt_runtime.jobs, configured by
runtime.RT): the settings, the machine (the cluster's nodes from `scontrol -o show node`, this node, pinning), the node's
memory, time limits (CFDT_SLURM_TIME), job.sh and submission (--dependency through after=), squeue/sacct and the states
of jobs (ended jobs squeue still lists, the lost-node diagnosis), the queue's CPUs, the cancel flags and scancel. The
names below stay as thin wrappers with cfdt's signatures (api, runner, bench, cavitation, transient, sliding, snappy and
the tests call them). What stays cfdt's: the sizing (cell_estimate, procs_for_job: ranks capped at cells /
CFDT_CELLS_PER_RANK and a node's cores, mem_plan / mem_auto: 2 GB + 4 GB per million cells, 8G when unknown, the time
estimate for validation), the solver's progress (api._progress), cpushare (leftover processes, core accounting:
cores_shared), segments of transient runs and chained studies (init_from), eta_start / running_etas (a queued job's start
from the running jobs' own ETAs, applied on top of the runtime's states), queue_summary's extras (cluster_queue,
starts_now, pending_for_node, machine.node), the queue position per node (pending_ahead), and writing a finished failure
as the study's result (slurm_states). They reach the runtime as hooks on runtime.RT (procs, mem_auto, job_extra:
cells_estimate, running_progress, oom_advice, timeout_hint). 2026.11.09.2 (phase 1): final_state and the runtime's
version.

Implements llm-translators' docs/EXECUTION.md. Settings (environment):
    CFDT_EXECUTOR          local | slurm (local)
    CFDT_SLURM_MIN_MINUTES studies estimated below this run locally even with slurm (0; no estimate: SLURM)
    CFDT_SLURM_ARGS        extra sbatch arguments, split like a shell does
    CFDT_SLURM_MEM         auto (from the cell estimate) | a size | none
    CFDT_SLURM_NODE        local | any | a node list
    CFDT_SLURM_TIME        timeout (default: ceil(timeout / 60) + 10 minutes) | max (the partition's limit) | estimate
                           (from the study's time estimate, within 'timeout') | minutes (llmt-runtime's meanings)
    CFDT_PROCS             MPI ranks per study (a study's solve.procs overrides it)
    CFDT_TIMEOUT           seconds per study
A study's "queue": true (always SLURM) or false (always local) overrides the minimum, not CFDT_EXECUTOR=local.

Every SLURM state is computed from the files and the queue (never from in-process handles), so any server
process, after restarts or from another chat, sees the same thing:
    done           <name>.json exists
    queued         squeue: PENDING (reason, expected_start, ahead)
    running        squeue: RUNNING / COMPLETING (the solver's progress, job_elapsed_min)
    timed_out, out_of_memory, cancelled, failed
                   gone from squeue without a result (or listed in a final state: MinJobAge): sacct, or job.log when
                   sacct knows nothing; these are written as the study's result ({"ok": false, "state", "why"}) so they
                   stay final
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from llmt_runtime import jobs as J
from llmt_runtime import slurm as S

from .runtime import RT

CLI = "cfdt"
FINAL = J.FINAL            # ("timed_out", "out_of_memory", "cancelled", "failed")


# ---- settings (llmt-runtime's) ----------------------------------------------------------------------
def kind() -> str:
    return S.kind(RT)


def min_minutes() -> float:
    return S.min_minutes(RT)


def timeout_s(study: dict | None = None) -> float:
    """The study's run.timeout_s, else CFDT_TIMEOUT (3600)."""
    return S.timeout_s(RT, study)


def time_limit_min(study: dict | None = None) -> int:
    """ceil(timeout / 60) + 10: the job's limit with CFDT_SLURM_TIME=timeout (the default; llmt-runtime's
    time_need_min, cfdt has no hook for it)."""
    return S.time_need_min(RT, study)


def procs_of(study: dict) -> int:
    """MPI ranks asked for: run.procs, then solve.procs, then CFDT_PROCS (before the size cap)."""
    return S.procs_of(RT, study)


def slurm_args() -> list[str]:
    """CFDT_SLURM_ARGS split as a shell does (several options in one value); unbalanced quotes: the plain words."""
    return S.slurm_args(RT)


def job_time(study: dict | None = None, estimate_min: float | None = None) -> tuple[str, dict]:
    """(the job's --time, details): CFDT_SLURM_TIME timeout (default) | max | estimate | minutes."""
    return S.job_time(RT, study, estimate_min)


# ---- per-study run settings (study.run) within the machine's caps ------------------------------------
RUN_KEYS = {"timeout_s", "mem", "procs", "queue"}
mem_mb = S.mem_mb          # '8G' -> 8192 (SLURM's units: K, M (default), G, T); None if malformed


class _Caches(dict):
    """cfdt's own caches of what SLURM said (core accounting). clear() also makes llmt-runtime forget the cluster's
    nodes and the partition's limit (the tests clear it between cases)."""

    def clear(self) -> None:
        super().clear()
        S.forget()


_NODES: dict = _Caches()


def _node_name(n: dict) -> str:
    return n.get("NodeName") or ""


def slurm_nodes(all_nodes: bool = False) -> list[dict]:
    """The SLURM nodes jobs may go to (all of the cluster's with all_nodes): [{name, host, cpus, threads, cores}]
    from llmt-runtime's `scontrol -o show node`; [] without SLURM. cores: physical cores (Sockets x CoresPerSocket,
    else CPUs / threads per core): with --ntasks-per-core=1 a job gets at most that many ranks on one node (mochi: 40
    CPUs, 2 threads per core: 20; a 36-rank request was 'Requested node configuration is not available')."""
    if kind() != "slurm":
        return []
    nodes = S.cluster_nodes() or []
    if not all_nodes:
        nodes = S.job_nodes(RT) if nodes else []
    out = []
    for n in nodes:
        try:
            thr = int(n.get("ThreadsPerCore") or 1)
        except ValueError:
            thr = 1
        out.append({"name": _node_name(n), "host": n.get("NodeHostName") or _node_name(n), "cpus": S.node_cpus(n),
                    "threads": thr, "cores": S.node_core_count(n)})
    return out


def _my_addresses() -> set:
    """This machine's addresses (llmt-runtime's: `hostname -I`, the host name's own)."""
    return S.addresses()


def local_node() -> str | None:
    """This machine's node in the SLURM cluster (llm-translators docs/EXECUTION.md, 'Clusters of several nodes'):
    llmt-runtime's this_node (NodeName or NodeHostName equal to the host name, else NodeAddr among this machine's
    addresses)."""
    me = S.this_node(S.cluster_nodes())
    return me.get("NodeName") if me else None


def core_accounting() -> dict | None:
    """Whether SLURM keeps a job's cores to itself: `scontrol show config` (SelectType, SelectTypeParameters)
    and the partitions' OverSubscribe. {shared: bool, why, select, params, oversubscribe}; None when SLURM does
    not say (asked again next time). Cores are shared when the select plugin is linear without CR_Core/CR_CPU
    counting, when the parameters count only memory (CR_Memory), or when a partition oversubscribes."""
    import re
    if "accounting" in _NODES:
        return _NODES["accounting"]
    rc, out, _ = _run(["scontrol", "show", "config"])
    if rc != 0 or "SelectType" not in out:
        return None
    kv = dict(re.findall(r"^(\w+)\s*=\s*(.*?)\s*$", out, re.M))
    sel, par = kv.get("SelectType", ""), kv.get("SelectTypeParameters", "")
    rc2, out2, _ = _run(["scontrol", "-o", "show", "partition"])
    over = sorted({m for m in re.findall(r"OverSubscribe=(\S+)", out2)}) if rc2 == 0 else []
    why = []
    counts = any(c in par.upper() for c in ("CR_CORE", "CR_CPU", "CR_SOCKET"))
    if "linear" in sel and not counts:
        why.append(f"SelectType {sel} gives whole nodes, but jobs here ask for cores")
    elif not counts:
        why.append(f"SelectTypeParameters {par or '(none)'} do not count cores or CPUs (CR_Core or CR_CPU)")
    for o in over:
        if o.upper().startswith(("YES", "FORCE")):
            why.append(f"a partition has OverSubscribe={o}")
    rec = {"shared": bool(why), "select": sel, "params": par, "oversubscribe": over, **({"why": why} if why else {})}
    _NODES["accounting"] = rec
    return rec


def cluster_nodes() -> int:
    """How many nodes `scontrol -o show node` lists (0 when SLURM does not answer)."""
    return len(S.cluster_nodes() or [])


def args_nodelist() -> str | None:
    """A --nodelist / -w in CFDT_SLURM_ARGS (split as a shell does), which wins over CFDT_SLURM_NODE."""
    return S.args_nodelist(RT)


def _args_nodelist() -> bool:
    return args_nodelist() is not None


def effective_nodelist() -> str | None:
    """The nodes a job may run on, as it asks: CFDT_SLURM_ARGS' --nodelist/-w, else node_list(); None: SLURM
    chooses. tool_status reports it (the router's sbatch --test-only must ask the same) and the rank caps use those
    nodes' cores."""
    return S.node_pin(RT)[1]


def node_setting() -> str:
    """CFDT_SLURM_NODE: local (default) | any | a node list."""
    return S.node_setting(RT)


def node_list() -> str | None:
    """The --nodelist cfdt adds (llm-translators docs/EXECUTION.md, 'Clusters of several nodes'): with
    CFDT_SLURM_NODE=local (the default) this machine's node when the cluster has several (the sparks: spark1 is
    spark-5da5, spark2 spark-0808, and their home folders are not shared, so a job sent to the other node vanished
    without a log); 'any': none (SLURM chooses; shared home folders); a node list: those. A --nodelist / -w in
    CFDT_SLURM_ARGS wins (then cfdt adds none). One-node clusters: none. (llmt-runtime's node_pin.)"""
    return S.node_pin(RT)[0]


def node_cores() -> int | None:
    """The most ranks one job can have: the largest node's cores (one rank per core) among the nodes its jobs may go
    to (the pinned ones, else the cluster's); None without SLURM."""
    if kind() != "slurm" or not S.cluster_nodes():
        return None
    ns = S.job_nodes(RT)
    return max(S.node_core_count(n) for n in ns) if ns else None


def node_mem_mb() -> float | None:
    """The most memory one job can ask for (MB): the largest RealMemory among the nodes jobs go to, or None when
    SLURM does not know it. mochi's jobs failed only at submission with 'Memory specification can not be satisfied'
    when run.mem asked for more (90G, 2026.11.07). llmt-runtime's (scontrol RealMemory)."""
    return S.node_mem_mb(RT)


NODES_LISTED = 8


def nodes_listing(nodes: list) -> dict:
    """status' nodes: every node on a small cluster; on a large one (Perlmutter: 5,288) the first NODES_LISTED and a
    summary (how many, each kind's CPUs and cores with its count, a few names)."""
    rows = [{k: n[k] for k in ("name", "cpus", "cores")} for n in nodes]
    if len(rows) <= NODES_LISTED:
        return {"nodes": rows}
    kinds = {}
    for r in rows:
        kinds[(r["cpus"], r["cores"])] = kinds.get((r["cpus"], r["cores"]), 0) + 1
    return {"nodes": rows[:NODES_LISTED],
            "nodes_summary": {"count": len(rows),
                              "kinds": [{"cpus": c, "cores": k, "count": n}
                                        for (c, k), n in sorted(kinds.items(), key=lambda kv: -kv[1])],
                              "names": [r["name"] for r in rows[:3]] + ["..."] + [rows[-1]["name"]],
                              "note": f"the first {NODES_LISTED} of {len(rows)} nodes are listed in 'nodes'"}}


def caps() -> dict:
    """The ceilings set in config/env (CFDT_MAX_TIMEOUT s, CFDT_MAX_MEM, CFDT_MAX_PROCS); unset: none. With
    SLURM, procs is also at most a node's cores, and mem at most a node's memory, exactly as SLURM gives it
    ('64000M'; it read '62G' before 2026.11.09.4), mem_mb in MB for the router."""
    out = {}
    for k, name in (("timeout_s", "MAX_TIMEOUT"), ("mem", "MAX_MEM"), ("procs", "MAX_PROCS")):
        v = RT.env(name, "").strip()
        if v:
            out[k] = v
    rc = S.caps(RT) if kind() == "slurm" else {}
    if rc.get("nodes"):
        # jobs may span nodes (CFDT_SLURM_NODES): llmt-runtime's cap, a node's cores x the nodes
        try:
            given = int(float(out["procs"])) if out.get("procs") else None
        except ValueError:
            given = None
        if given is None or given > int(rc["procs"]):
            out["procs"] = str(int(rc["procs"]))
            out["procs_from"] = rc.get("procs_from")
        out["nodes"] = rc["nodes"]
    else:
        nc = node_cores()
        if nc:
            try:
                given = int(float(out["procs"])) if out.get("procs") else None
            except ValueError:
                given = None
            if given is None or given > nc:
                out["procs"] = str(nc)
                out["procs_from"] = "a node's cores (SLURM, one rank per core)"
    nm = node_mem_mb()
    if nm and (not out.get("mem") or (mem_mb(out["mem"]) or 0) > nm):
        out["mem"] = S.mem_text(nm)
        out["mem_from"] = "a node's memory (SLURM RealMemory)"
    if out.get("mem") and mem_mb(out["mem"]):
        out["mem_mb"] = int(mem_mb(out["mem"]))
        if rc.get("mem_per"):
            out["mem_per"] = rc["mem_per"]         # SLURM's --mem is per node
    return out


def run_errors(st: dict) -> list[str]:
    """Validation of study.run: names, types, and the caps of this machine."""
    r = st.get("run")
    if r is None:
        return []
    if not isinstance(r, dict):
        return ["run: {timeout_s, mem, procs, queue}"]
    errs = []
    bad = set(r) - RUN_KEYS
    if bad:
        errs.append(f"run: unknown keys {sorted(bad)} (timeout_s, mem, procs, queue)")
    cap = caps()
    t = r.get("timeout_s")
    if t is not None:
        if not isinstance(t, (int, float)) or isinstance(t, bool) or t <= 0:
            errs.append("run.timeout_s: seconds (> 0)")
        elif "timeout_s" in cap and t > float(cap["timeout_s"]):
            errs.append(f"run.timeout_s: at most {float(cap['timeout_s']):g} s on this machine (CFDT_MAX_TIMEOUT)")
    m = r.get("mem")
    if m is not None:
        mb = mem_mb(m) if isinstance(m, str) and m.lower() != "auto" else (1 if m == "auto" else None)
        if not mb:
            errs.append("run.mem: SLURM memory per job, e.g. '8G' or '500M', or 'auto' (from the cell estimate)")
        elif "mem" in cap and mem_mb(cap["mem"]) and mb > mem_mb(cap["mem"]):
            errs.append(f"run.mem: at most {cap['mem']} on this machine ({cap.get('mem_from', 'CFDT_MAX_MEM')}); "
                        "choose a machine with more (machines(): executor.caps.mem) or ask for less")
    n = r.get("procs")
    if n is not None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            errs.append("run.procs: MPI ranks (integer >= 1)")
        elif "procs" in cap and n > int(cap["procs"]):
            errs.append(f"run.procs: at most {cap['procs']} on this machine "
                        f"({cap.get('procs_from', 'CFDT_MAX_PROCS')})")
        sp = (st.get("solve") or {}).get("procs")
        if sp is not None and sp != n:
            errs.append(f"run.procs ({n}) and solve.procs ({sp}) differ: give one")
    q = r.get("queue")
    if q is not None:
        if not isinstance(q, bool):
            errs.append("run.queue: true (always a SLURM job) or false (always local)")
        elif "queue" in st and st["queue"] != q:
            errs.append("run.queue and queue differ: give one")
    return errs


queue_of = S.queue_of      # run.queue, then the study's queue; None: decided by the time estimate


def mem_estimate_mb(cells: float | None) -> float | None:
    """Memory a study may need, from its cell count: 2 GB plus 4 GB per million cells, an upper bound that
    covers meshing (Gmsh, snappyHexMesh) and a kOmegaSST solve with MPI buffers."""
    return None if not cells else 2048.0 + 4096.0 * cells / 1e6


def mem_setting() -> str:
    """CFDT_SLURM_MEM (llm-translators docs/EXECUTION.md): auto (default: from the study's cell estimate) | a size
    (e.g. 16G) | none (SLURM's default)."""
    return RT.env("SLURM_MEM", "").strip() or "auto"


def _cap_mb() -> float | None:
    c = caps()
    return mem_mb(c["mem"]) if c.get("mem") else None


def mem_plan(study: dict, cells: float | None = None) -> tuple[float | None, str]:
    """(MB the job asks for, or None for SLURM's default; where it came from). In order: the study's run.mem (a size;
    'auto': the estimate); a --mem in CFDT_SLURM_ARGS (it wins over CFDT_SLURM_MEM); CFDT_SLURM_MEM: auto (2 GB +
    4 GB per million estimated cells; 8 GB when the cells cannot be told), a size, or none. Within CFDT_MAX_MEM and
    the node's memory. cells: the study's estimate (None: unknown). The job's own --mem is llmt-runtime's mem_plan with
    cfdt's mem_auto (job_mem); this one names the source and sizes the validation estimate."""
    cap = _cap_mb()

    def capped(mb):
        return float(min(mb, cap) if cap else mb)
    rm = (study.get("run") or {}).get("mem")
    if rm and str(rm).lower() != "auto":
        return float(mem_mb(rm) or 0), "study"
    a = S.opt_value(slurm_args(), ("--mem",))
    args_mem = mem_mb(a) if a else None
    if args_mem is not None and str(rm).lower() != "auto":
        return float(args_mem), "CFDT_SLURM_ARGS"
    setting = "auto" if str(rm).lower() == "auto" else mem_setting()
    if setting.lower() == "none":
        return None, "none"
    if setting.lower() == "auto":
        need = mem_estimate_mb(cells)
        return capped(need if need else 8192.0), "estimate" if need else "auto (no cell estimate: 8G)"
    mb = mem_mb(setting)
    return (capped(mb), "CFDT_SLURM_MEM") if mb else (None, "none")


def _mem_text(mb: float, cap: float | None = None) -> str:
    """MB as cfdt asks for memory: whole GB rounded up (MB below 1 GB), but never above the cap (then the cap exactly:
    a node's 64000M, not 63G = 64512 MB)."""
    t = f"{int(math.ceil(mb / 1024))}G" if mb >= 1024 else f"{int(math.ceil(mb))}M"
    if cap and (mem_mb(t) or 0) > cap:
        t = S.mem_text(cap)
    return t


def mem_auto(study: dict, cells: float | None = None) -> str:
    """CFDT_SLURM_MEM=auto (and run.mem 'auto'): the job's --mem from the cell estimate, 2 GB + 4 GB per million
    cells, 8G when the cells cannot be told; within CFDT_MAX_MEM and the node's memory. The runtime's hooks.mem_auto."""
    if cells is None:
        cells = _cells(study)
    cap = _cap_mb()
    need = mem_estimate_mb(cells) or 8192.0
    return _mem_text(min(need, cap) if cap else need, cap)


def _for_job(study: dict) -> dict:
    """The study as llmt-runtime sizes its job: run.mem 'auto' (cfdt's: the estimate, whatever CFDT_SLURM_MEM and a
    --mem in CFDT_SLURM_ARGS say) becomes that size."""
    r = S.run_of(study)
    if str(r.get("mem", "")).strip().lower() == "auto":
        return dict(study, run=dict(r, mem=mem_auto(study)))
    return study


def job_mem(study: dict) -> str | None:
    """The --mem in a study's job.sh (llmt-runtime's mem_plan: run.mem, else nothing when CFDT_SLURM_ARGS has a --mem,
    else CFDT_SLURM_MEM none | a size | auto: mem_auto); None: SLURM's default or CFDT_SLURM_ARGS's own."""
    return S.mem_for(RT, _for_job(study))


def sbatch_args(study: dict, cells: float | None = None) -> tuple[list[str], str | None]:
    """(sbatch's arguments from CFDT_SLURM_ARGS, where the job's memory comes from: mem_plan's source). With a
    study's run.mem, a --mem/--mem-per-cpu in CFDT_SLURM_ARGS is dropped. From 2026.11.09.4 the job's --mem is in
    job.sh (job_mem), not on sbatch's command line: `sbatch job.sh` by hand lost it."""
    return S.sbatch_args(RT, study), mem_plan(study, cells)[1]


def cell_estimate(study: dict) -> float | None:
    """An upper estimate of the mesh's cell count before meshing, or None when it cannot be told cheaply.
    Box cells are exact; Gmsh meshes: 2D/swept sections ~1.3 cells per size^2 with quads (2.5 with
    triangles), tetrahedra ~10 per size^3, plus every refine sphere at its size. A goal is built with
    build123d-translator for its volume (a second or so)."""
    import math
    g, m = study.get("geometry") or {}, study.get("mesh") or {}
    if m.get("cells"):
        return float(math.prod(int(c) for c in m["cells"]))
    h = m.get("size")
    if not h:
        return None
    h = float(h)
    if m.get("mesher") == "snappy":
        return snappy_estimate(study)
    dom = study.get("domain") or {}
    vol = thick = None
    if dom.get("kind") == "external" and dom.get("box"):
        b = [float(v) for v in dom["box"]]
        vol, thick = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]), b[5] - b[2]
    elif "box" in g:
        lx, ly, lz = (float(v) for v in g["box"])
        vol, thick = lx * ly * lz, lz
    elif "cylinder" in g:
        r, hh = (float(v) for v in g["cylinder"])
        vol, thick = math.pi * r * r * hh, hh
    elif "annulus" in g:
        ri, ro, hh = (float(v) for v in g["annulus"])
        vol, thick = math.pi * (ro * ro - ri * ri) * hh, hh
    elif "goal" in g:
        gs = goal_stats(g["goal"])
        if gs is None:
            return None
        vol, thick = gs[0], gs[2]
    if not vol or not thick or thick <= 0:
        return None
    layers = 1 if study.get("2d") else (int(m["extrude"]) if m.get("extrude") else None)
    refine = m.get("refine") or []
    ext = _refine_extents(study, refine, vol, thick)
    if layers:                                        # a 2D section, swept into layers
        per = 1.3 if m.get("quads", True) else 2.5
        n = vol / thick / h ** 2 * per + sum(a / float(r["size"]) ** 2 * per for r, (_, a) in zip(refine, ext))
        return n * layers
    return vol / h ** 3 * 10 + sum(v / float(r["size"]) ** 3 * 10 for r, (v, _) in zip(refine, ext))


def _refine_extents(study: dict, refine: list, vol: float, thick: float) -> list[tuple[float, float]]:
    """(volume mm3, area of its section across Z mm2) of each mesh.refine region, for the cell estimate: a
    sphere {at, r}, a box, a cylinder {center, axis, radius, span} or a band {distance} around the part. Boxes and
    cylinders are clipped to the domain (an external domain's box, else the fluid's own size)."""
    import math
    dom = study.get("domain") or {}
    db = [float(v) for v in dom["box"]] if dom.get("kind") == "external" and dom.get("box") else None
    L = max(db[3] - db[0], db[4] - db[1], db[5] - db[2]) if db else max(vol ** (1 / 3), thick)
    out = []
    for r in refine:
        if "box" in r:
            b = [float(v) for v in r["box"]]
            if db:
                b = [max(b[i], db[i]) for i in range(3)] + [min(b[i + 3], db[i + 3]) for i in range(3)]
            d = [max(b[i + 3] - b[i], 0.0) for i in range(3)]
            out.append((d[0] * d[1] * d[2], d[0] * d[1]))
        elif "cylinder" in r:
            c = r["cylinder"]
            R = float(c["radius"])
            s0, s1 = (float(v) for v in c.get("span", [-L, L]))
            ln = min(s1 - s0, 2 * L)
            a = [float(v) for v in c["axis"]]
            na = math.sqrt(sum(v * v for v in a)) or 1.0
            along_z = abs(a[2]) / na > 0.9
            Ri = float(c.get("inner_radius", 0) or 0)
            out.append((math.pi * (R * R - Ri * Ri) * ln, math.pi * (R * R - Ri * Ri) if along_z else 2 * (R - Ri) * ln))
        elif "distance" in r:
            ps = _part_size(study)
            d = float(r["distance"]) + float(r["size"])
            area = ps[1] if ps else 6 * vol ** (2 / 3)
            out.append((area * d, area / max(thick, 1e-9) * d))      # a shell; its section a band round the outline
        else:
            R = float(r["r"])
            out.append((4 / 3 * math.pi * R ** 3, math.pi * R * R))
    return out


_GOAL_STATS: dict = {}


def goal_stats(goal: dict) -> tuple[float, float, float] | None:
    """(volume mm3, area mm2, z extent mm) of a build123d-translator goal, built once per goal and process: the
    estimates ask for it several times per study (validation, use_slurm, submit), and a 13 kB car goal takes
    seconds to build on a busy machine (run_studies to mochi passed the router's minute, 2026.11.06)."""
    import hashlib
    key = hashlib.sha1(json.dumps(goal, sort_keys=True, default=str).encode()).hexdigest()
    if key not in _GOAL_STATS:
        try:
            from build123d_translator import translate as T
            solid, _ = T.build(goal)
            bb = solid.bounding_box()
            _GOAL_STATS[key] = (float(solid.volume), float(solid.area), float(bb.max.Z - bb.min.Z))
        except Exception:  # noqa: BLE001
            _GOAL_STATS[key] = None
        if len(_GOAL_STATS) > 64:
            _GOAL_STATS.pop(next(iter(_GOAL_STATS)))
    return _GOAL_STATS[key]


def _part_size(study: dict) -> tuple[float, float] | None:
    """(volume mm3, surface area mm2) of the part, or None."""
    import math
    g = study.get("geometry") or {}
    if "box" in g:
        a, b, c = (float(v) for v in g["box"])
        return a * b * c, 2 * (a * b + b * c + a * c)
    if "cylinder" in g:
        r, hh = (float(v) for v in g["cylinder"])
        return math.pi * r * r * hh, 2 * math.pi * r * (r + hh)
    if "annulus" in g:
        ri, ro, hh = (float(v) for v in g["annulus"])
        return math.pi * (ro * ro - ri * ri) * hh, 2 * math.pi * (ro + ri) * hh + 2 * math.pi * (ro * ro - ri * ri)
    if "goal" in g:
        gs = goal_stats(g["goal"])
        return (gs[0], gs[1]) if gs else None
    if "stl" in g:
        from .meshsurf import surface_stats
        ss = surface_stats(g["stl"], float(g.get("scale", 1.0) or 1.0))
        return (ss["volume_mm3"], ss["area_mm2"]) if ss else None
    return None


def snappy_estimate(study: dict) -> float | None:
    """Cells of a snappyHexMesh mesh: background hexahedra, a band of about four cells per level towards the
    part's surface (nCellsBetweenLevels 3), refinement regions, distance bands and layers; 20 % margin."""
    import math

    import numpy as np

    from .snappy import settings
    ps = _part_size(study)
    if ps is None:
        return None
    vol_p, area = ps
    ext = (study.get("domain") or {}).get("kind") == "external"
    if ext:
        b = [float(v) for v in study["domain"]["box"]]
    else:
        e = vol_p ** (1 / 3)
        b = [-e, -e, 0, e, e, 2 * e]
    try:
        st = settings(study, ext, b, None)
    except Exception:  # noqa: BLE001
        return None
    h0 = st["h0"]
    vol = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]) - (vol_p if ext else 0) if ext else vol_p
    n = vol / h0 ** 3
    Lmin, Lmax = st["surface"]
    for L in range(1, Lmax + 1):
        hl = h0 / 2 ** L
        n += area / hl ** 2 * 4 * (1 if L <= Lmin else 0.5)
    for e in st["regions"]:
        hl = e["cell_mm"]
        if e["shape"] == "box":
            bx = e["box"]
            v = (bx[3] - bx[0]) * (bx[4] - bx[1]) * (bx[5] - bx[2])
        elif e["shape"] in ("cylinder", "annulus"):
            v = math.pi * (e["radius"] ** 2 - e.get("inner_radius", 0.0) ** 2) * float(np.linalg.norm(np.subtract(e["p2"], e["p1"])))
        else:
            v = 4 / 3 * math.pi * e["radius"] ** 3
        n += min(v, vol) / hl ** 3
    for d, L in st["distance"]:
        n += area * d / (h0 / 2 ** L) ** 3
    if st["layers"]:
        n += area / (h0 / 2 ** Lmin) ** 2 * st["layers"]["n"]
    return 1.2 * n


def procs_for_job(study: dict, est: float | None = None) -> int:
    """The ranks a SLURM job asks for: the study's (or CFDT_PROCS), at most a node's cores times the nodes a job may
    use (CFDT_SLURM_NODES, default 1), and no more than the cell estimate allows
    at CFDT_CELLS_PER_RANK, so small studies do not hold cores they cannot use (the benchmarks on mochi held
    18 CPUs each for 1-3 ranks). The runtime's hooks.procs."""
    procs = procs_of(study)
    if kind() == "slurm" and S.cluster_nodes():
        # one rank per core, on one node or, with CFDT_SLURM_NODES, on several (llmt-runtime: a node's cores x nodes;
        # CFDT_PROCS may say more)
        procs = min(procs, S.max_job_procs(RT))
    est = cell_estimate(study) if est is None else est
    if est is None:
        return procs
    per = int(os.environ.get("CFDT_CELLS_PER_RANK", "5000") or 5000)
    return max(1, min(procs, int(math.ceil(est / max(per, 1)))))


# Time per cell and iteration on one rank (kOmegaSST, SIMPLE), and how ranks speed it up: mochi, 2026-10-19..22.
# The 900k-cell pump: 283 / 210 / 178 s for 200 iterations on 6 / 10 / 19 ranks (memory-bandwidth bound: little
# gain past 10). The Ahmed body's 600k-cell snappy mesh: 0.45 s per iteration (and per DES time step) on 19.
SEC_PER_CELL_ITER = 4.6e-6


def rank_speedup(p: int) -> float:
    p = max(1, int(p))
    return p ** 0.6 if p <= 10 else 10 ** 0.6 * (p / 10) ** 0.25


def _finest_mm(study: dict) -> float | None:
    """The smallest cell size the study asks for (mm): what limits a transient run's time step."""
    m = study.get("mesh") or {}
    sizes = []
    for k in ("size", "features"):
        if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool):
            sizes.append(float(m[k]))
    sf = m.get("surface")
    if sf is not None:
        sizes += [float(x) for x in (sf if isinstance(sf, list) else [sf])]
    sizes += [float(r["size"]) for r in (m.get("refine") or []) if "size" in r]
    if m.get("cells"):
        g = (study.get("geometry") or {}).get("box")
        if g:
            sizes += [float(L) / int(n) for L, n in zip(g, m["cells"]) if int(n) > 1]
    return min(sizes) if sizes else None


def transient_steps(study: dict) -> float | None:
    """Time steps of a transient study: the Courant limit on its finest cells at 1.5 x the reference speed
    (dt = 0.3 maxCo h / U: thin layer cells and fast spots make the steps about that small: the Ahmed body ran
    at 2.8e-5 s with 5 mm cells at 40 m/s, the 2D-2 cylinder at 2.6e-4 s with 2.5 mm at 1 m/s), and maxDeltaT."""
    tr = (study.get("solve") or {}).get("transient")
    if not tr:
        return None
    from .transient import _velocity
    T = float(tr["end_time"])
    T0 = 0.0
    if tr.get("continue"):                   # only the time still to run
        try:
            from .transient import source_folder
            T0 = float(json.loads((source_folder(tr["continue"]) / "transient.json").read_text()).get("time") or 0.0)
        except (OSError, ValueError):
            T0 = 0.0
    h = _finest_mm(study)
    U = _velocity(study)
    W = float(tr.get("write_interval", T / 10))
    dt = min(W / 10, T / 500)
    from .transient import sliding_zone
    sl = sliding_zone(study)
    if "max_co" in tr and tr["max_co"] is None:       # a fixed time step: dt, or degrees_per_step of a revolution
        fdt = (float(tr.get("degrees_per_step", 1.0)) / 360.0 * 60.0 / abs(float(sl["rpm"]))) if sl else \
            float(tr.get("dt") or dt)
        return max(T - T0, 0.0) / fdt * (1.6 if sl else 1.0)
    co = float(tr.get("max_co", 2.0 if sl else 1.0))
    if sl:
        # the rotor's tip speed sets the Courant number, and at most degrees_per_step per step
        U = max(U, abs(float(sl["rpm"])) * 2 * math.pi / 60 * float(sl["radius"]) * 1e-3)
        dt = min(dt, float(tr.get("degrees_per_step", 1.0)) / 360.0 * 60.0 / abs(float(sl["rpm"])))
    if h and U:
        dt = min(dt, 0.3 * co * h * 1e-3 / (1.5 * U))
    return max(T - T0, 0.0) / dt * (1.6 if sl else 1.0)      # sliding: two outer correctors, mesh motion


def estimate_minutes(study: dict, solver: str, cells: float | None = None, procs: int | None = None) -> float | None:
    """An upper bound of the run's wall time (minutes) on procs ranks: meshing, the solve (iterations, or the
    steady start plus time steps of a transient run, from the cell estimate and the per-rank rate above) and the
    results and pictures, plus 30 %. None when the cells cannot be estimated (a STEP file). The 2D reference
    solver takes seconds."""
    if solver == "ref":
        return 0.1
    n = cell_estimate(study) if cells is None else cells
    if not n:
        return None
    p = procs or procs_for_job(study, n)
    model = study.get("model", "laminar")
    per = SEC_PER_CELL_ITER * (0.7 if model == "laminar" else 1.0) * (1.6 if study.get("energy") else 1.0) * \
        (1.3 if (study.get("scalars") or study.get("age")) else 1.0)
    it = per * n / rank_speedup(p)
    s = study.get("solve") or {}
    tr = s.get("transient")
    if tr:
        from .transient import init_ref
        init = 0 if (tr.get("continue") or (init_ref(study)[0] and "init" not in tr)) else \
            int(tr.get("init", 0 if model == "laminar" else 500))
        solve = (init + (transient_steps(study) or 0)) * it
    elif study.get("cavitation"):
        # incompressibleVoF (cavitation.py): the vapour fraction (MULES, two correctors) and more pressure work
        cv = study["cavitation"] if isinstance(study["cavitation"], dict) else {}
        solve = int(cv.get("iterations", s.get("iterations", 4000))) * it * 1.8
    else:
        solve = int(s.get("iterations", 3000 if model != "laminar" else 2000)) * it
    snappy = (study.get("mesh") or {}).get("mesher") == "snappy"
    mesh = n * (1.2e-4 if snappy else 5e-5)
    post = n * (6e-5 + (4e-5 if study.get("images") else 0))
    return round(1.3 * (mesh + solve + post) / 60 + 0.5, 1)


def mem_needed_mb(study: dict, cells: float | None = None) -> float:
    """The memory the job asks for (mem_plan, as its --mem: a size as given, the estimate rounded up as mem_auto
    writes it); SLURM's default ('none'): the estimate's need, for the router."""
    mb, src = mem_plan(study, cells)
    if mb is None:
        return float(mem_estimate_mb(cells) or 8192.0)
    if src in ("study", "CFDT_SLURM_ARGS", "CFDT_SLURM_MEM"):
        return float(mb)
    return float(mem_mb(_mem_text(mb, _cap_mb())) or mb)


def estimate(study: dict, solver: str) -> dict | None:
    """llm-translators' contract (docs/EXECUTION.md, 'Estimates in validation'): {minutes, procs, mem_mb} from the
    same sizing as a SLURM job (ranks after the cell cap, memory as --mem, minutes an upper bound)."""
    ref_ok = bool(study.get("2d")) and "box" in (study.get("geometry") or {}) and \
        bool((study.get("mesh") or {}).get("cells")) and study.get("model", "laminar") == "laminar"
    if solver == "ref" and ref_ok:
        return {"minutes": 0.1, "procs": 1, "mem_mb": 1024}
    solver = "foam"                    # what the study needs (the reference solver runs 2D boxes only)
    n = cell_estimate(study)
    p = procs_for_job(study, n)
    minutes = estimate_minutes(study, solver, n, p)
    if minutes is None:
        return None
    out = {"minutes": minutes, "procs": int(p), "mem_mb": int(round(mem_needed_mb(study, n)))}
    if n:
        out["cells"] = int(n)
    return out


def use_slurm(study: dict, solver: str) -> bool:
    """Whether a study becomes a SLURM job: only with CFDT_EXECUTOR=slurm; then "queue" decides if given, else the
    time estimate (none: SLURM). llmt-runtime's use_slurm; the estimate is made only when it decides."""
    if kind() != "slurm":
        return False
    if queue_of(study) is not None:
        return S.use_slurm(RT, study, None)
    return S.use_slurm(RT, study, estimate_minutes(study, solver))


def status() -> dict:
    """The 'executor' key of tool_status (read by llmt status and the router): cfdt's keys (args, nodes, node,
    cores_shared; in local mode kind, workers, procs, timeout_s, caps, runtime), with llmt-runtime's time, slurm_time
    (CFDT_SLURM_TIME as read) and node_cores in SLURM mode."""
    if kind() == "slurm":
        _, nl, info = S.node_pin(RT)
        node = {"setting": node_setting(), "nodelist": nl, "cluster_nodes": cluster_nodes()}
        if info.get("warning"):
            node["warning"] = info["warning"]
        ts = S.time_setting(RT)
        ca = core_accounting()
        return {"kind": "slurm", "sbatch": shutil.which("sbatch"), "min_minutes": min_minutes(),
                "procs": S.default_procs(RT), "time_limit_min": time_limit_min(), "time": ts, "slurm_time": ts,
                "args": RT.env("SLURM_ARGS", ""), "timeout_s": timeout_s(), "caps": caps(),
                **nodes_listing(slurm_nodes()),
                "node": node, "node_cores": node_cores(),
                **({"cores_shared": ca["why"]} if ca and ca.get("shared") else {}),
                "mem": mem_setting(), "runtime": _runtime_version()}
    return {"kind": "local", "workers": int(os.environ.get("CFDT_WORKERS", "2")),
            "procs": S.default_procs(RT), "timeout_s": timeout_s(), "caps": caps(),
            "runtime": _runtime_version()}


def _runtime_version() -> str:
    """llmt-runtime's version (machines() and llmt status show it)."""
    return RT.runtime_version


# ---- submitting (llmt-runtime's) ---------------------------------------------------------------------
write_atomic = J.write_atomic      # a file written whole or not at all (a temporary file, then rename)

_CELLS: dict = {}
CELLS_TTL = 30.0


def _cells(study: dict) -> float | None:
    """cell_estimate, kept CELLS_TTL s per study: one submission asks for it in several hooks (ranks, memory,
    job.json)."""
    try:
        key = json.dumps(study, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return cell_estimate(study)
    hit = _CELLS.get(key)
    if hit is not None and time.time() - hit[0] < CELLS_TTL:
        return hit[1]
    v = cell_estimate(study)
    _CELLS[key] = (time.time(), v)
    while len(_CELLS) > 32:
        _CELLS.pop(next(iter(_CELLS)))
    return v


def job_extra(study: dict) -> dict:
    """cfdt's keys in job.json (the runtime's hooks.job_extra): cells_estimate (api._progress shows it next to the
    meshed count)."""
    est = _cells(study)
    return {"cells_estimate": int(est) if est else None}


def job_script(request: Path, name: str, procs: int, study: dict | None = None) -> str:
    """job.sh (llmt-runtime's job_script): the job's name cfdt:<request>:<name>, folder, log, --ntasks=N --nodes=1
    --ntasks-per-core=1, --time (CFDT_SLURM_TIME), --mem (job_mem; in job.sh from 2026.11.09.4), --nodelist
    (node_list), --export=ALL, then `python -m cfd_translator.cli run-study REQUEST NAME`."""
    st = study or {}
    return J.job_script(RT, Path(request), name, st, int(procs), job_time(st)[0], job_mem(st), node_list())


def _solver_of(request: Path) -> str:
    try:
        return json.loads((Path(request) / "request.json").read_text()).get("solver") or "foam"
    except (OSError, ValueError):
        return "foam"


def submit(request: Path, name: str, study: dict, after: str | None = None, segment: int | None = None) -> dict:
    """Writes <request>/<name>/job.sh, submits it, writes job.json (llmt-runtime's submit, with cfdt's hooks). On
    failure writes the study's result {"ok": false, "why": "could not submit: ..."} (never falls back to local).
    after: a job this one waits for (a transient study's next job, a chained study's source: --dependency=afterany),
    segment: its number (job.json segment, previous_jobs). job.json: the runtime's keys and cfdt's (cells_estimate,
    mem_from as cfdt names it, run as the study gave it, segment, previous_jobs)."""
    request = Path(request)
    d = request / name
    prev = J.read_job(d) if after and (d / "job.json").is_file() else None
    est_min = None
    if S.time_setting(RT) == "estimate" and not S.run_of(study).get("timeout_s"):
        try:                                   # CFDT_SLURM_TIME=estimate: the limit from the study's time estimate
            est_min = estimate_minutes(study, _solver_of(request))
        except Exception:  # noqa: BLE001 (no estimate: the timeouts)
            est_min = None
    extra = {"run": dict(S.run_of(study)), "mem_from": mem_plan(study, _cells(study))[1]}
    if segment:
        extra["segment"] = segment
        extra["previous_jobs"] = list((prev or {}).get("previous_jobs", [])) + \
            ([str(prev["job_id"])] if prev and prev.get("job_id") else [])
    info = J.submit(RT, request, name, _for_job(study), estimate_min=est_min,
                    after=[str(after)] if after else None, extra=extra)
    if info.get("job_id"):
        return info
    if not segment:                  # a continuation that fails leaves the running job to write its result
        return J.failed_submission(RT, request, name, info)
    return {"name": name, "ok": False, "state": "failed", "why": info.get("why", "could not submit"),
            "execution": {"executor": "slurm", "slurm_job": None}}


def run_study_job(request: str, name: str) -> int:
    """`cfdt run-study REQUEST NAME`: what a SLURM job (or a person) runs. Exit 0 when a result was written. The
    study's timeout is the one its job was sized with (job.json timeout_s: shorter with CFDT_SLURM_TIME=estimate)."""
    from .runner import run_study

    req = Path(request).expanduser().resolve()
    meta = json.loads((req / "request.json").read_text())
    study = dict(meta["studies"][name])
    n = os.environ.get("SLURM_NTASKS")
    procs = int(n) if n and n.isdigit() else procs_of(study)
    if procs != procs_of(study) or "procs" not in (study.get("solve") or {}):
        study["solve"] = dict(study.get("solve") or {}, procs=procs)
    if not os.environ.get("SLURM_JOB_ID"):
        os.environ["CFDT_EXECUTION"] = "by hand"            # the run record's executor
    reaped = []
    if os.environ.get("SLURM_JOB_ID"):
        from . import foam
        foam.install_signal_handlers()        # scancel takes this job's OpenFOAM processes down with it
        try:                                  # and leftovers of ended jobs go before this one starts
            from .cpushare import reap
            reaped = reap(job_active)
        except Exception:  # noqa: BLE001
            reaped = []
    t0 = time.perf_counter()
    r = run_study(name, study, req / name, meta.get("solver", "foam"), J.job_timeout_s(req / name, timeout_s(study)))
    if reaped:
        jobs_ = sorted({x["job"] for x in reaped})
        r.setdefault("warnings", []).append(
            f"stopped {len(reaped)} leftover OpenFOAM/MPI processes of ended SLURM job(s) {', '.join(jobs_)} on this "
            f"node before starting ({', '.join(sorted({x['cmd'] for x in reaped}))})")
    if r.get("continued"):              # a transient study goes on in the next job (job.json has its id)
        return 0
    r.setdefault("seconds", {})
    if isinstance(r["seconds"], dict):
        r["seconds"]["wall"] = round(time.perf_counter() - t0, 3)
    ex = r.get("execution") or {}
    # 'executor' is the record's former shape, kept for a while for readers of 2026.10.18.2-.5
    r["executor"] = {"kind": ex.get("executor", "slurm" if os.environ.get("SLURM_JOB_ID") else "by hand"),
                     "job_id": ex.get("slurm_job", os.environ.get("SLURM_JOB_ID")), "procs": procs}
    write_atomic(req / name / f"{name}.json", json.dumps(r, indent=1, default=float))
    return 0


def job_active(job_id: str) -> bool:
    """Is a SLURM job still pending or running (squeue)? Raises when squeue cannot tell."""
    rc, out, err = _run(["squeue", "-h", "-j", str(job_id), "-o", "%T"])
    if rc != 0:
        if "Invalid job id" in (err or ""):
            return False
        raise RuntimeError(err or f"squeue exit {rc}")
    return bool(out.strip())


# ---- the queue (llmt-runtime's) ----------------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 20):
    """(returncode, stdout, stderr) of a SLURM (or system) command, through llmt-runtime's slurm.run (within an MCP
    call's deadline when one is set); (1, '', why) when the command is missing or does not answer."""
    p = S.run(cmd, timeout)
    if p is None:
        return 1, "", f"{cmd[0]}: not found or no answer"
    return p.returncode, p.stdout or "", p.stderr or ""


_minutes = S.elapsed_minutes   # squeue's elapsed time ([D-]HH:MM:SS, MM:SS) in minutes


def squeue(ids: list[str]) -> dict | None:
    """One squeue call for these job ids: {id: {state, reason, start, elapsed_min, node}}; None when SLURM does not
    answer. Ended jobs squeue still lists (MinJobAge) are among them, in their final state."""
    return J.squeue([str(i) for i in ids])


# cfdt's own read of the whole queue (every user's jobs: id, name, state, CPUs, node, requested nodes, user, priority),
# kept a few seconds: pending_ahead, running_etas and queue_summary share it (they made three squeue calls of their
# own, next to the runtime's, in one study_status or jobs() answer)
QUEUE_FMT = "%i|%j|%T|%C|%N|%n|%u|%Q"
_QROWS: dict = {"at": 0.0, "v": None}


def queue_rows(max_age_s: float = 3.0) -> list[dict] | None:
    """[{id, name, state, cpus, node, req, user, prio}] for every job SLURM lists (one squeue call, kept max_age_s);
    None when SLURM does not answer."""
    if _QROWS["v"] is not None and time.time() - _QROWS["at"] < max_age_s:
        return _QROWS["v"]
    rc, out, _ = _run(["squeue", "-h", "-o", QUEUE_FMT])
    if rc != 0:
        return None
    rows = []
    for line in out.splitlines():
        f = (line.split("|") + [""] * 8)[:8]
        jid = f[0].strip().split("_")[0]
        if not jid.isdigit():
            continue
        try:
            prio = float(f[7]) if f[7].strip() else 0.0
        except ValueError:
            prio = 0.0
        rows.append({"id": jid, "name": f[1], "state": f[2].strip(), "cpus": int(f[3]) if f[3].strip().isdigit() else 0,
                     "node": f[4].strip(), "req": f[5].strip(), "user": f[6].strip(), "prio": prio})
    _QROWS.update(at=time.time(), v=rows)
    return rows


def pending_ahead(rows: list[dict] | None = None) -> dict:
    """Pending jobs ahead of each pending job: higher priority (or the same and an older id) and able to run on the
    same node: no requested nodes on either side, or requested node lists that share a node (%n). On the sparks
    (one controller, jobs pinned with --nodelist) spark1's job 506 counted spark2's 469-471 (2026.11.08). From
    queue_rows (no squeue call of its own); applied on top of llmt-runtime's `ahead` (by priority, every node's)."""
    rows = queue_rows() if rows is None else rows
    jobs = [(r["prio"], int(r["id"]), frozenset(_expand_nodes(r["req"])))
            for r in rows or [] if r["state"] == "PENDING"]

    def share(a, b):
        return not a or not b or bool(a & b)
    return {str(i): sum(1 for pp, ii, nn in jobs if (pp > p or (pp == p and ii < i)) and share(n, nn))
            for p, i, n in jobs}


_JOB_NODE: dict = {}


def sacct(ids: list[str]) -> dict | None:
    """{id: state} from accounting, or None when sacct fails or knows nothing (llmt-runtime's sacct_rows; the node
    each job ran on is kept in _JOB_NODE)."""
    if not ids:
        return {}
    rows = J.sacct_rows([str(i) for i in ids])
    for i, r in rows.items():
        if r.get("node"):
            _JOB_NODE[i] = r["node"]
    return {i: r["state"] for i, r in rows.items()} or None


def _tail(path: Path, n: int = 30) -> str:
    return J.tail(path, n) if Path(path).is_file() else "(no job.log)"


def final_state(acct: str | None, log: str, folder: Path | None = None) -> tuple[str, str]:
    """(state, why) of a job that left the queue without a result, from sacct's state and the end of job.log ('(no
    job.log)' when there is none): llmt-runtime's rules (jobs.classify_end), with cfdt's out-of-memory advice when the
    study folder is given. SLURM's out-of-memory wording 'Detected N oom_kill events' counts whatever sacct says
    (before 2026.11.09.2 only 'oom-kill', and only without accounting: such jobs came back as 'failed')."""
    has_log = log != "(no job.log)"
    job = J.read_job(Path(folder)) if folder is not None else {}
    return J.classify_end(RT, acct, log if has_log else "", has_log=has_log, folder=folder, job=job)


def slurm_states(request: Path, names: list[str], studies: dict | None = None, write: bool = True) -> dict:
    """{name: {"state", ...}} for the request's studies that have job.json and no result: llmt-runtime's job_states
    (one squeue call; sacct, else job.log, for jobs that left the queue or that squeue lists in a final state; a job
    lost on another node), with cfdt's additions: the queue position per node (pending_ahead), a sooner start from
    the running jobs' own ETAs (eta_start), the progress kept between a transient study's jobs, the solver's progress
    of running ones. Finished failures become the study's result file (when write)."""
    from .api import _progress

    request = Path(request)
    jobs = {}
    for n in names:
        if (request / n / "job.json").is_file() and not (request / n / f"{n}.json").is_file():
            j = J.read_job(request / n)
            if j.get("job_id"):
                jobs[n] = j
    if not jobs:
        return {}
    states = J.job_states(RT, request, list(jobs))
    ahead = None
    out = {}
    for n, j in jobs.items():
        s = states.get(n) or {}
        jid = str(j["job_id"])
        base = {"job_id": jid, "procs": j.get("procs")}
        state = s.get("state")
        if state == "done" or (request / n / f"{n}.json").is_file():       # finished between the calls
            continue
        if state == "unknown":
            out[n] = {"state": "unknown", **base, "why": s.get("why")}
            continue
        if state == "queued":
            if ahead is None:
                ahead = pending_ahead()
            st = {"state": "queued", **base, "reason": s.get("reason"), "ahead": ahead.get(jid, s.get("ahead"))}
            if s.get("expected_start"):
                st["expected_start"] = s["expected_start"]
            elif s.get("expected_start_note"):     # SLURM's one-year placeholder: no start time (llmt-runtime)
                st["expected_start_note"] = s["expected_start_note"]
            try:
                es = eta_start(j, s.get("reason") or "", st.get("expected_start"))
            except Exception:  # noqa: BLE001
                es = None
            if es:
                st.update({k: v for k, v in es.items() if v is not None})
            if j.get("segment") or (request / n / "mesh.json").is_file():
                # a later segment (or a job that meshed before): where the run got to, kept between jobs
                try:
                    prog = _progress(request / n, (studies or {}).get(n, {}))
                except Exception:  # noqa: BLE001
                    prog = {}
                keep = {k: prog[k] for k in ("time", "end_time", "eta_min_max", "mesh", "iteration",
                                             "max_iterations") if k in prog}
                if keep:
                    st.update(keep)
                    st["stage"] = (f"between jobs: segment {j.get('segment')} queued" if j.get("segment")
                                   else "queued again")
                    if "eta_min_max" in keep:
                        st["note"] = "eta_min_max: the simulated time left at the last job's pace, once running"
            out[n] = st
            continue
        if state == "running":
            try:
                prog = _progress(request / n, (studies or {}).get(n, {}))
            except Exception:  # noqa: BLE001
                prog = {}
            st = {**prog, "state": "running", **base, "job_elapsed_min": s.get("job_elapsed_min")}
            if s.get("node"):
                st["node"] = s["node"]
            out[n] = st
            continue
        why = s.get("why") or "the job left the queue without a result"
        if state not in FINAL:
            state = "failed"
        res = {"name": n, "ok": False, "state": state, "why": why, "job_id": jid,
               "execution": {"executor": "slurm", "slurm_job": jid}}
        if write:
            write_atomic(request / n / f"{n}.json", json.dumps(res, indent=1))
        out[n] = res
    return out


# ---- jobs and cancel ---------------------------------------------------------------------------
def _expand_nodes(spec: str | None) -> set[str]:
    """SLURM node names in a list such as 'spark-5da5,node[01-03]' (llmt-runtime's expand_nodelist)."""
    if not spec or spec in ("(null)", "None assigned", "N/A"):
        return set()
    return set(S.expand_nodelist(spec))


def _my_nodes() -> set[str]:
    """The nodes this machine's jobs go to (effective_nodelist), or empty: SLURM chooses among all."""
    return _expand_nodes(effective_nodelist())


def queue_summary() -> dict:
    """This translator's jobs in the queue and the machine's CPUs. 'machine' is llmt-runtime's (sinfo -N: each node
    once, where summing per-partition rows counted a node in two partitions twice), with 'node' when jobs are pinned.
    cfdt's on top: on a cluster where jobs are pinned to a node (CFDT_SLURM_NODE=local on the sparks), 'queue' counts
    this user's jobs of this machine's node only (running there, or pending with --nodelist naming it) and
    'cluster_queue' every node's; 'starts_now' says whether a job would start at once (all CPUs of the node idle and
    nothing pending for it, from any user), 'pending_for_node' how many are pending for it."""
    base = J.queue_summary(RT)
    mine = _my_nodes()
    rows = queue_rows()
    q = {"running": 0, "queued": 0, "cpus_in_use": 0}
    cq = {"running": 0, "queued": 0, "cpus_in_use": 0}
    pending_here_any = 0
    user = os.environ.get("USER", "")
    for r in rows or []:
        st = r["state"]
        nodes = _expand_nodes(r["node"]) or _expand_nodes(r["req"])
        here = not mine or bool(nodes & mine) or (st not in ("RUNNING", "COMPLETING") and not nodes)
        if st in J.QUEUED and here:
            pending_here_any += 1
        if not r["name"].startswith(f"{CLI}:") or (user and r["user"] and r["user"] != user):
            continue
        if st not in J.QUEUED and st not in J.ACTIVE:      # ended, still listed (MinJobAge): not in the queue
            continue
        for d, take in ((cq, True), (q, here)):
            if not take:
                continue
            if st in J.ACTIVE:
                d["running"] += 1
                d["cpus_in_use"] += r["cpus"]
            else:
                d["queued"] += 1
    machine = dict(base.get("machine") or {})
    res = {"queue": q, "machine": machine}
    if mine:
        machine["node"] = ",".join(sorted(mine))
        res["cluster_queue"] = dict(cq, note="this translator's jobs on every node of the SLURM cluster; 'queue' "
                                    "counts only those of this machine's node")
    if machine and rows is not None:
        free = machine.get("cpus_idle", 0) >= machine.get("cpus", 1) > 0
        res["starts_now"] = bool(free and pending_here_any == 0)
        res["pending_for_node"] = pending_here_any
    return res


# running jobs' own ETAs (from their logs), for the start of jobs queued behind them: SLURM counts a running job's
# whole time limit, which put later DES segments hours late (Rogue spat variants on spark1, 2026.11.06)
_ETAS: dict = {"at": 0.0, "v": {}}


def running_etas(max_age_s: float = 20.0) -> dict:
    """{job_id: {"node", "cpus", "eta_min", "name"}} for this translator's running jobs whose study reports an
    eta (its log); cached for max_age_s."""
    if time.time() - _ETAS["at"] < max_age_s:
        return _ETAS["v"]
    from .api import _progress_of
    user = os.environ.get("USER", "")
    v = {}
    for r in queue_rows() or []:
        if r["state"] != "RUNNING" or (user and r["user"] and r["user"] != user):
            continue
        parts = r["name"].split(":")
        if len(parts) != 3 or parts[0] != CLI:
            continue
        try:
            req = RT.out_root() / parts[1]
            meta = json.loads((req / "request.json").read_text())
            pr = _progress_of(req / parts[2], (meta.get("studies") or {}).get(parts[2], {}))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(pr, dict) and pr.get("eta_min_max") is not None:
            v[r["id"]] = {"node": r["node"], "cpus": r["cpus"] or None,
                          "eta_min": float(pr["eta_min_max"]), "name": f"{parts[1]}/{parts[2]}"}
    _ETAS.update(at=time.time(), v=v)
    return v


def eta_start(job: dict, reason: str, slurm_start: str | None) -> dict | None:
    """A sooner start than SLURM's for a queued job, from the ETAs of the running jobs it waits for: its previous
    segment (reason Dependency: the job.json's previous job) or, waiting for Resources/Priority on its node, the
    first of that node's running jobs of this translator to finish. None when SLURM's is as early."""
    etas = running_etas()
    if not etas:
        return None
    prev = (job.get("previous_jobs") or [])[-1:] if job.get("segment") else []
    cand = None
    if "Dependency" in (reason or "") and prev and str(prev[0]) in etas:
        cand = (etas[str(prev[0])]["eta_min"], f"its previous segment (job {prev[0]}) should end then")
    elif reason in ("Resources", "Priority"):
        mine = _my_nodes()
        here = [(e["eta_min"], j) for j, e in etas.items() if not mine or _expand_nodes(e["node"]) & mine]
        if here:
            m, j = min(here)
            cand = (m, f"the first running job on its node (job {j}, {etas[j]['name']}) should end then")
    if cand is None:
        return None
    t = time.time() + 60 * cand[0]
    when = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
    if slurm_start and slurm_start not in ("N/A", "Unknown") and slurm_start <= when:
        return None
    return {"expected_start": when, "expected_start_slurm": slurm_start or None,
            "expected_start_note": f"from the running jobs' own progress: {cand[1]}; SLURM counts their whole time "
                                   "limit (an upper bound)"}


def cancel_slurm(request: Path, names: list[str]) -> list[str]:
    """scancel for the SLURM jobs of these studies that have no result yet; returns their ids."""
    request = Path(request)
    ids = []
    for n in names:
        if (request / n / "job.json").is_file() and not (request / n / f"{n}.json").is_file():
            jid = J.read_job(request / n).get("job_id")
            if jid:
                ids.append(str(jid))
    if ids:
        J.scancel(ids)
    return ids


def queued_note(request: Path) -> str | None:
    """What to tell someone who interrupted a command that was waiting for a request's SLURM jobs (`cfdt bench`,
    `cfdt run`): the jobs still in the queue and how to stop them. They are not cancelled: they run without the
    command, and their results are collected later. None when no job of the request is left in the queue."""
    request = Path(request)
    try:
        names = json.loads((request / "request.json").read_text()).get("names", [])
    except (OSError, ValueError):
        names = [p.name for p in request.iterdir() if (p / "job.json").is_file()] if request.is_dir() else []
    ids = []
    for n in names:
        if not (request / n / f"{n}.json").is_file():
            jid = J.read_job(request / n).get("job_id")
            if jid:
                ids.append(str(jid))
    if not ids:
        return None
    q = J.squeue(ids)
    if q is not None:                       # SLURM did not answer: name them all
        ids = [i for i in ids if i in q and (q[i]["state"] in J.QUEUED or q[i]["state"] in J.ACTIVE)]
    if not ids:
        return None
    return (f"interrupted: SLURM job{'s' if len(ids) > 1 else ''} {', '.join(ids)} of {request} "
            f"{'are' if len(ids) > 1 else 'is'} still in the queue and will run without this command (nothing was "
            f"cancelled). Collect the results later with `{CLI} status {request}`; to stop "
            f"{'them' if len(ids) > 1 else 'it'}: `{CLI} cancel {request}` (cancel(request) from the MCP tools) or "
            f"`scancel {' '.join(ids)}`")


def print_queued_note(request: Path) -> None:
    """queued_note on stderr (stdout stays JSON); never raises."""
    try:
        note = queued_note(request)
    except Exception:  # noqa: BLE001
        note = None
    if note:
        print(f"[{CLI}] {note}", file=sys.stderr, flush=True)
