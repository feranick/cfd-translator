"""executor.py — where studies run: the local worker pool, or one SLURM job per study.
Version: 2026.10.31.1

Implements llm-translators' docs/EXECUTION.md. Settings (environment):
    CFDT_EXECUTOR          local | slurm (local)
    CFDT_SLURM_MIN_MINUTES studies estimated below this run locally even with slurm (0; no estimate: SLURM)
    CFDT_SLURM_ARGS        extra sbatch arguments, split like a shell does
    CFDT_PROCS             MPI ranks per study (a study's solve.procs overrides it)
    CFDT_TIMEOUT           seconds per study; the job's time limit is ceil(timeout / 60) + 10 minutes
A study's "queue": true (always SLURM) or false (always local) overrides the minimum, not CFDT_EXECUTOR=local.

Every SLURM state is computed from the files and the queue (never from in-process handles), so any server
process, after restarts or from another chat, sees the same thing:
    done           <name>.json exists
    queued         squeue: PENDING (reason, expected_start, ahead)
    running        squeue: RUNNING / COMPLETING (the solver's progress, job_elapsed_min)
    timed_out, out_of_memory, cancelled, failed
                   gone from squeue without a result: sacct, or job.log when sacct knows nothing; these
                   are written as the study's result ({"ok": false, "state", "why"}) so they stay final
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLI = "cfdt"
FINAL = ("timed_out", "out_of_memory", "cancelled", "failed")


# ---- settings ----------------------------------------------------------------------------------
def kind() -> str:
    k = os.environ.get("CFDT_EXECUTOR", "local").strip().lower() or "local"
    return k if k in ("local", "slurm") else "local"


def min_minutes() -> float:
    try:
        return float(os.environ.get("CFDT_SLURM_MIN_MINUTES", "0"))
    except ValueError:
        return 0.0


def timeout_s(study: dict | None = None) -> float:
    """The study's run.timeout_s, else CFDT_TIMEOUT (3600)."""
    t = ((study or {}).get("run") or {}).get("timeout_s")
    return float(t) if t else float(os.environ.get("CFDT_TIMEOUT", "3600"))


def time_limit_min(study: dict | None = None) -> int:
    return int(math.ceil(timeout_s(study) / 60.0)) + 10


def procs_of(study: dict) -> int:
    p = ((study.get("run") or {}).get("procs")) or (study.get("solve") or {}).get("procs")
    return int(p) if p else int(os.environ.get("CFDT_PROCS", "1") or 1)


# ---- per-study run settings (study.run) within the machine's caps ------------------------------------
RUN_KEYS = {"timeout_s", "mem", "procs", "queue"}
_MEM = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)B?$", re.I)


def mem_mb(text) -> float | None:
    """'8G' -> 8192 (SLURM's units: K, M (default), G, T)."""
    m = _MEM.match(str(text).strip())
    if not m:
        return None
    return float(m.group(1)) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2).upper()]


_NODES: dict = {}


def slurm_nodes(all_nodes: bool = False) -> list[dict]:
    """The SLURM nodes jobs may go to (all of the cluster's with all_nodes): [{name, host, cpus, threads, cores}]
    from sinfo; [] without SLURM. cores = CPUs / threads per core: with --ntasks-per-core=1 a job gets at most that
    many ranks on one node (mochi: 40 CPUs, 2 threads per core: 20; a 36-rank request was 'Requested node
    configuration is not available')."""
    if kind() != "slurm":
        return []
    if "v" in _NODES:
        known = _NODES["v"]
    else:
        rc, out, _ = _run(["sinfo", "-h", "-N", "-o", "%N %n %c %Z"])
        nodes = {}
        if rc == 0:
            for line in out.splitlines():
                f = line.split()
                if len(f) >= 4 and f[2].isdigit():
                    thr = int(f[3]) if f[3].isdigit() else 1
                    nodes[f[0]] = {"name": f[0], "host": f[1], "cpus": int(f[2]), "threads": thr,
                                   "cores": max(1, int(f[2]) // max(thr, 1))}
        known = list(nodes.values())
        if rc == 0 and known:                 # a failed sinfo is asked again next time, not remembered as empty
            _NODES["v"] = known
    want = None if all_nodes else effective_nodelist()
    return [n for n in known if not want or n["name"] in want.split(",")]


def _my_addresses() -> set:
    import socket
    out = set()
    try:
        for a in socket.getaddrinfo(socket.gethostname(), None):
            out.add(a[4][0])
    except OSError:
        pass
    rc, txt, _ = _run(["hostname", "-I"])
    if rc == 0:
        out.update(txt.split())
    return out


def local_node() -> str | None:
    """This machine's node in the SLURM cluster (llm-translators docs/EXECUTION.md, 'Clusters of several nodes'):
    `scontrol -o show node`, the host name matched to NodeName or NodeHostName, else one of this machine's
    addresses to NodeAddr."""
    if "local" in _NODES:
        return _NODES["local"]
    import socket
    host = socket.gethostname().split(".")[0]
    rc, out, _ = _run(["scontrol", "-o", "show", "node"])
    recs = []
    if rc == 0:
        for line in out.splitlines():
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            if kv.get("NodeName"):
                recs.append(kv)
    hit = next((r["NodeName"] for r in recs if host in (r["NodeName"], r.get("NodeHostName", "").split(".")[0])), None)
    if hit is None and recs:
        mine = _my_addresses()
        hit = next((r["NodeName"] for r in recs if r.get("NodeAddr") in mine), None)
    if rc == 0 and recs:                      # a failed scontrol is asked again next time
        _NODES["local"] = hit
        _NODES["scontrol_nodes"] = len(recs)
    return hit


def core_accounting() -> dict | None:
    """Whether SLURM keeps a job's cores to itself: `scontrol show config` (SelectType, SelectTypeParameters)
    and the partitions' OverSubscribe. {shared: bool, why, select, params, oversubscribe}; None when SLURM does
    not say (asked again next time). Cores are shared when the select plugin is linear without CR_Core/CR_CPU
    counting, when the parameters count only memory (CR_Memory), or when a partition oversubscribes."""
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
    """How many nodes `scontrol -o show node` lists (sinfo's count when scontrol is not available)."""
    if "scontrol_nodes" not in _NODES:
        local_node()
    return int(_NODES.get("scontrol_nodes") or len(slurm_nodes(all_nodes=True)))


def args_nodelist() -> str | None:
    """A --nodelist / -w in CFDT_SLURM_ARGS (split as a shell does), which wins over CFDT_SLURM_NODE."""
    a = shlex.split(os.environ.get("CFDT_SLURM_ARGS", ""))
    for i, x in enumerate(a):
        if x in ("-w", "--nodelist") and i + 1 < len(a):
            return a[i + 1]
        if x.startswith("--nodelist="):
            return x.split("=", 1)[1]
        if x.startswith("-w") and len(x) > 2:
            return x[2:]
    return None


def _args_nodelist() -> bool:
    return args_nodelist() is not None


def effective_nodelist() -> str | None:
    """The nodes a job may run on, as it asks: CFDT_SLURM_ARGS' --nodelist/-w, else node_list(); None: SLURM
    chooses. tool_status reports it (the router's sbatch --test-only must ask the same) and the rank caps use those
    nodes' cores."""
    return args_nodelist() or node_list()


def node_setting() -> str:
    """CFDT_SLURM_NODE: local (default) | any | a node list."""
    return os.environ.get("CFDT_SLURM_NODE", "").strip() or "local"


def node_list() -> str | None:
    """The --nodelist cfdt adds (llm-translators docs/EXECUTION.md, 'Clusters of several nodes'): with
    CFDT_SLURM_NODE=local (the default) this machine's node when the cluster has several (the sparks: spark1 is
    spark-5da5, spark2 spark-0808, and their home folders are not shared, so a job sent to the other node vanished
    without a log); 'any': none (SLURM chooses; shared home folders); a node list: those. A --nodelist / -w in
    CFDT_SLURM_ARGS wins (then cfdt adds none). One-node clusters: none."""
    if _args_nodelist():
        return None
    v = node_setting()
    if v.lower() == "any":
        return None
    if v.lower() != "local":
        return v
    if cluster_nodes() <= 1:
        return None
    return local_node()


def node_cores() -> int | None:
    """The most ranks one job can have: the largest node's cores (one rank per core)."""
    ns = slurm_nodes()
    return max(n["cores"] for n in ns) if ns else None


def caps() -> dict:
    """The ceilings set in config/env (CFDT_MAX_TIMEOUT s, CFDT_MAX_MEM, CFDT_MAX_PROCS); unset: none. With
    SLURM, procs is also at most a node's cores."""
    out = {}
    for k, env in (("timeout_s", "CFDT_MAX_TIMEOUT"), ("mem", "CFDT_MAX_MEM"), ("procs", "CFDT_MAX_PROCS")):
        v = os.environ.get(env, "").strip()
        if v:
            out[k] = v
    nc = node_cores()
    if nc and (not out.get("procs") or int(out["procs"]) > nc):
        out["procs"] = str(nc)
        out["procs_from"] = "a node's cores (SLURM, one rank per core)"
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
            errs.append(f"run.mem: at most {cap['mem']} on this machine (CFDT_MAX_MEM)")
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


def queue_of(study: dict):
    r = (study.get("run") or {}).get("queue")
    return r if r is not None else study.get("queue")


def mem_estimate_mb(cells: float | None) -> float | None:
    """Memory a study may need, from its cell count: 2 GB plus 4 GB per million cells, an upper bound that
    covers meshing (Gmsh, snappyHexMesh) and a kOmegaSST solve with MPI buffers."""
    return None if not cells else 2048.0 + 4096.0 * cells / 1e6


def _strip_mem(args: list[str]) -> tuple[list[str], float | None]:
    """args without memory requests, and the --mem they held (MB)."""
    out, mem, skip = [], None, None
    for a in args:
        if skip:
            if skip == "--mem":
                mem = mem_mb(a)
            skip = None
            continue
        if a in ("--mem", "--mem-per-cpu"):
            skip = a
            continue
        if a.startswith("--mem="):
            mem = mem_mb(a.split("=", 1)[1])
            continue
        if a.startswith("--mem-per-cpu="):
            continue
        out.append(a)
    return out, mem


def mem_setting() -> str:
    """CFDT_SLURM_MEM (llm-translators docs/EXECUTION.md): auto (default: from the study's cell estimate) | a size
    (e.g. 16G) | none (SLURM's default)."""
    return os.environ.get("CFDT_SLURM_MEM", "").strip() or "auto"


def mem_plan(study: dict, cells: float | None = None) -> tuple[float | None, str]:
    """(MB the job asks for, or None for SLURM's default; where it came from). In order: the study's run.mem (a size;
    'auto': the estimate); a --mem in CFDT_SLURM_ARGS (it wins over CFDT_SLURM_MEM); CFDT_SLURM_MEM: auto (2 GB +
    4 GB per million estimated cells; 8 GB when the cells cannot be told), a size, or none. Within CFDT_MAX_MEM."""
    cap = mem_mb(caps()["mem"]) if caps().get("mem") else None

    def capped(mb):
        return float(min(mb, cap) if cap else mb)
    rm = (study.get("run") or {}).get("mem")
    if rm and str(rm).lower() != "auto":
        return float(mem_mb(rm) or 0), "study"
    _, args_mem = _strip_mem(shlex.split(os.environ.get("CFDT_SLURM_ARGS", "")))
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


def sbatch_args(study: dict, cells: float | None = None) -> tuple[list[str], str | None]:
    """CFDT_SLURM_ARGS (split as a shell does: several options in one value) with the job's --mem from mem_plan.
    Returns (args, where the memory came from)."""
    args = shlex.split(os.environ.get("CFDT_SLURM_ARGS", ""))
    rest, _ = _strip_mem(args)
    mb, src = mem_plan(study, cells)
    if src == "CFDT_SLURM_ARGS":
        return args, src
    if mb is None:
        return rest, src
    return rest + [f"--mem={int(math.ceil(mb / 1024))}G" if mb >= 1024 else f"--mem={int(math.ceil(mb))}M"], src


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
        try:
            from build123d_translator import translate as T
            solid, _ = T.build(g["goal"])
            bb = solid.bounding_box()
            vol, thick = float(solid.volume), float(bb.max.Z - bb.min.Z)
        except Exception:  # noqa: BLE001
            return None
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
            out.append((math.pi * R * R * ln, math.pi * R * R if along_z else 2 * R * ln))
        elif "distance" in r:
            ps = _part_size(study)
            d = float(r["distance"]) + float(r["size"])
            area = ps[1] if ps else 6 * vol ** (2 / 3)
            out.append((area * d, area / max(thick, 1e-9) * d))      # a shell; its section a band round the outline
        else:
            R = float(r["r"])
            out.append((4 / 3 * math.pi * R ** 3, math.pi * R * R))
    return out


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
        try:
            from build123d_translator import translate as T
            solid, _ = T.build(g["goal"])
            return float(solid.volume), float(solid.area)
        except Exception:  # noqa: BLE001
            return None
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
        elif e["shape"] == "cylinder":
            v = math.pi * e["radius"] ** 2 * float(np.linalg.norm(np.subtract(e["p2"], e["p1"])))
        else:
            v = 4 / 3 * math.pi * e["radius"] ** 3
        n += min(v, vol) / hl ** 3
    for d, L in st["distance"]:
        n += area * d / (h0 / 2 ** L) ** 3
    if st["layers"]:
        n += area / (h0 / 2 ** Lmin) ** 2 * st["layers"]["n"]
    return 1.2 * n


def procs_for_job(study: dict, est: float | None = None) -> int:
    """The ranks a SLURM job asks for: the study's (or CFDT_PROCS), but no more than the cell estimate allows
    at CFDT_CELLS_PER_RANK, so small studies do not hold cores they cannot use (the benchmarks on mochi held
    18 CPUs each for 1-3 ranks)."""
    procs = procs_of(study)
    nc = node_cores()
    if nc:
        procs = min(procs, nc)                 # one node, one rank per core (CFDT_PROCS may say more)
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
    if h and U:
        dt = min(dt, 0.3 * float(tr.get("max_co", 1.0)) * h * 1e-3 / (1.5 * U))
    return max(T - T0, 0.0) / dt


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
        init = 0 if tr.get("continue") else int(tr.get("init", 0 if model == "laminar" else 500))
        solve = (init + (transient_steps(study) or 0)) * it
    else:
        solve = int(s.get("iterations", 3000 if model != "laminar" else 2000)) * it
    snappy = (study.get("mesh") or {}).get("mesher") == "snappy"
    mesh = n * (1.2e-4 if snappy else 5e-5)
    post = n * (6e-5 + (4e-5 if study.get("images") else 0))
    return round(1.3 * (mesh + solve + post) / 60 + 0.5, 1)


def mem_needed_mb(study: dict, cells: float | None = None) -> float:
    """The memory the job asks for (mem_plan, as its --mem; rounded up as sbatch_args writes it); SLURM's default
    ('none'): the estimate's need, for the router."""
    mb, _ = mem_plan(study, cells)
    if mb is None:
        return float(mem_estimate_mb(cells) or 8192.0)
    return float(math.ceil(mb / 1024) * 1024) if mb >= 1024 else float(math.ceil(mb))


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
    if kind() != "slurm":
        return False
    q = queue_of(study)
    if q is True:
        return True
    if q is False:
        return False
    est = estimate_minutes(study, solver)
    return est is None or est >= min_minutes()


def status() -> dict:
    """The 'executor' key of tool_status (read by llmt status)."""
    if kind() == "slurm":
        return {"kind": "slurm", "sbatch": shutil.which("sbatch"), "min_minutes": min_minutes(),
                "procs": int(os.environ.get("CFDT_PROCS", "1") or 1), "time_limit_min": time_limit_min(),
                "args": os.environ.get("CFDT_SLURM_ARGS", ""), "timeout_s": timeout_s(), "caps": caps(),
                "nodes": [{k: n[k] for k in ("name", "cpus", "cores")} for n in slurm_nodes()],
                "node": {"setting": node_setting(), "nodelist": effective_nodelist(),
                         "cluster_nodes": cluster_nodes()},
                **({"cores_shared": ca["why"]} if (ca := core_accounting()) and ca.get("shared") else {}),
                "mem": mem_setting()}
    return {"kind": "local", "workers": int(os.environ.get("CFDT_WORKERS", "2")),
            "procs": int(os.environ.get("CFDT_PROCS", "1") or 1), "timeout_s": timeout_s(), "caps": caps()}


# ---- submitting --------------------------------------------------------------------------------
def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def job_script(request: Path, name: str, procs: int, study: dict | None = None) -> str:
    d = request / name
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={CLI}:{request.name}:{name}",
        f"#SBATCH --chdir={d}",
        f"#SBATCH --output={d / 'job.log'}",
        f"#SBATCH --ntasks={procs}",
        "#SBATCH --nodes=1",                    # one node: the case is in one directory, on one machine
        "#SBATCH --ntasks-per-core=1",          # a whole core per rank (OpenMPI 5 maps ranks by core)
        *([f"#SBATCH --nodelist={node_list()}"] if node_list() else []),
        f"#SBATCH --time={time_limit_min(study)}",
        "#SBATCH --export=ALL",
        f"exec {shlex.quote(sys.executable)} -m cfd_translator.cli run-study {shlex.quote(str(request))} {shlex.quote(name)}",
        ""])


def submit(request: Path, name: str, study: dict, after: str | None = None, segment: int | None = None) -> dict:
    """Writes <request>/<name>/job.sh, submits it, writes job.json. On failure writes the study's result
    {"ok": false, "why": "could not submit: ..."} (never falls back to local). after: a job this one waits for
    (a transient study's next job: --dependency=afterany), segment: its number."""
    d = request / name
    prev = None
    if after and (d / "job.json").is_file():
        try:
            prev = json.loads((d / "job.json").read_text())
        except json.JSONDecodeError:
            prev = None
    d.mkdir(parents=True, exist_ok=True)
    est = cell_estimate(study)
    procs = procs_for_job(study, est)
    script = d / "job.sh"
    script.write_text(job_script(request, name, procs, study))
    script.chmod(0o755)
    sargs, mem_from = sbatch_args(study, est)
    cmd = ["sbatch", "--parsable"] + sargs + ([f"--dependency=afterany:{after}"] if after else []) + [str(script)]
    try:
        r = subprocess.run(cmd, cwd=str(d), capture_output=True, text=True, timeout=60)
        ok, msg = r.returncode == 0 and r.stdout.strip(), (r.stderr or r.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, msg = False, str(exc)
    if not ok:
        res = {"name": name, "ok": False, "state": "failed", "why": f"could not submit: {msg[-500:] or 'sbatch failed'}",
               "execution": {"executor": "slurm", "slurm_job": None}}
        if not after:                    # a continuation that fails leaves the running job to write its result
            write_atomic(d / f"{name}.json", json.dumps(res, indent=1))
        return res
    job_id = r.stdout.strip().splitlines()[-1].split(";")[0].strip()
    info = {"executor": "slurm", "job_id": job_id, "submitted": time.time(), "procs": procs,
            "time_min": time_limit_min(study), "command": cmd,
            "run": dict(study.get("run") or {}), "cells_estimate": int(est) if est else None, "mem_from": mem_from}
    if segment:
        info["segment"] = segment
        info["previous_jobs"] = list((prev or {}).get("previous_jobs", [])) + ([str(prev["job_id"])] if prev else [])
    write_atomic(d / "job.json", json.dumps(info, indent=1))
    return info


def run_study_job(request: str, name: str) -> int:
    """`cfdt run-study REQUEST NAME`: what a SLURM job (or a person) runs. Exit 0 when a result was written."""
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
    t0 = time.perf_counter()
    r = run_study(name, study, req / name, meta.get("solver", "foam"), timeout_s(study))
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


# ---- the queue ---------------------------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def _minutes(t: str) -> float | None:
    """SLURM elapsed time [d-]hh:mm:ss or mm:ss -> minutes."""
    t = t.strip()
    if not t or t in ("N/A", "INVALID"):
        return None
    days = 0
    if "-" in t:
        dd, t = t.split("-", 1)
        days = int(dd)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return round(days * 1440 + h * 60 + m + s / 60, 1)


def squeue(ids: list[str]) -> dict:
    """One squeue call for these job ids: {id: {state, reason, start, elapsed_min}}; None when squeue fails."""
    if not ids:
        return {}
    rc, out, err = _run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%r|%S|%M"])
    if rc != 0 and "Invalid job id" not in err:
        return None
    q = {}
    for line in out.splitlines():
        f = (line.split("|") + [""] * 5)[:5]
        q[f[0].strip()] = {"state": f[1].strip(), "reason": f[2].strip(), "start": f[3].strip(),
                           "elapsed_min": _minutes(f[4])}
    return q


def pending_ahead() -> dict:
    """All pending jobs on the machine as (priority, id), to count the jobs ahead of each."""
    rc, out, _ = _run(["squeue", "-h", "-t", "PD", "-o", "%i|%Q"])
    jobs = []
    for line in out.splitlines() if rc == 0 else []:
        i, _, p = line.partition("|")
        try:
            jobs.append((float(p or 0), int(i)))
        except ValueError:
            pass
    return {str(i): sum(1 for pp, ii in jobs if pp > p or (pp == p and ii < i)) for p, i in jobs}


def sacct(ids: list[str]) -> dict | None:
    """{id: state} from accounting, or None when sacct fails or knows nothing."""
    if not ids:
        return {}
    rc, out, _ = _run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "-o", "JobID,State,NodeList"])
    if rc != 0:
        return None
    st = {}
    for line in out.splitlines():
        f = line.split("|")
        i = f[0].strip()
        if i and "." not in i:
            st[i] = f[1].strip() if len(f) > 1 else ""
            if len(f) > 2 and f[2].strip() and f[2].strip() != "None assigned":
                _JOB_NODE[i] = f[2].strip()
    return st or None


_JOB_NODE: dict = {}


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no job.log)"


def final_state(acct: str | None, log: str) -> tuple[str, str]:
    """(state, why) of a job that left the queue without a result."""
    a = (acct or "").upper()
    if a.startswith("TIMEOUT") or (not acct and "DUE TO TIME LIMIT" in log):
        return "timed_out", ("the SLURM time limit ended the job: raise CFDT_TIMEOUT (the job asks for it plus 10 "
                             "minutes), or lower the study's solve.iterations or coarsen its mesh")
    if a.startswith("OUT_OF_MEMORY") or (not acct and ("oom-kill" in log.lower() or "out of memory" in log.lower())):
        return "out_of_memory", "out of memory: add --mem=... to CFDT_SLURM_ARGS, or coarsen the mesh"
    if a.startswith("CANCELLED") or (not acct and "CANCELLED" in log):
        who = a.split("BY", 1)[1].strip() if "BY" in a else None
        return "cancelled", "cancelled" + (f" by user {who}" if who else "")
    if not acct and log == "(no job.log)":     # never started: scancel while queued leaves nothing behind
        return "cancelled", ("left the queue without starting (no job.log): cancelled while queued (scancel), or "
                             "SLURM dropped it; submit it again if it was not cancelled on purpose")
    return "failed", f"the job ended without a result ({acct or 'no accounting'}); the end of job.log:\n{log}"


def slurm_states(request: Path, names: list[str], studies: dict | None = None, write: bool = True) -> dict:
    """{name: {"state", ...}} for the request's studies that have job.json and no result. Finished
    failures become the study's result file (when write)."""
    from .api import _progress

    jobs = {}
    for n in names:
        jf = request / n / "job.json"
        if jf.is_file() and not (request / n / f"{n}.json").is_file():
            try:
                jobs[n] = json.loads(jf.read_text())
            except json.JSONDecodeError:
                pass
    if not jobs:
        return {}
    ids = [str(j["job_id"]) for j in jobs.values()]
    q = squeue(ids)
    gone = [n for n, j in jobs.items() if q is not None and str(j["job_id"]) not in q]
    acct = sacct([str(jobs[n]["job_id"]) for n in gone]) if gone else {}
    ahead = pending_ahead() if q and any(v["state"] == "PENDING" for v in q.values()) else {}
    out = {}
    for n, j in jobs.items():
        jid = str(j["job_id"])
        base = {"job_id": jid, "procs": j.get("procs")}
        if (request / n / f"{n}.json").is_file():               # finished between the calls
            continue
        if q is None:
            out[n] = {"state": "unknown", **base, "why": "squeue failed (is slurmctld running?)"}
            continue
        if jid in q:
            s = q[jid]
            if s["state"] in ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"):
                st = {"state": "queued", **base, "reason": s["reason"], "ahead": ahead.get(jid)}
                if s["start"] and s["start"] not in ("N/A", "Unknown"):
                    st["expected_start"] = s["start"]
            else:
                prog = _progress(request / n, (studies or {}).get(n, {}))
                st = {**prog, "state": "running", **base, "job_elapsed_min": s["elapsed_min"]}
            out[n] = st
            continue
        state, why = final_state((acct or {}).get(jid), _tail(request / n / "job.log"))
        node, here = _JOB_NODE.get(jid), local_node()
        if (state in ("failed", "cancelled") and "by user" not in why and not (request / n / "job.log").exists()
                and node and here and node != here):
            state, why = "failed", (f"the job went to node {node} and left no job.log: that node cannot see {request} "
                                    "(its filesystem is not shared with this one). Keep jobs on this node "
                                    "(CFDT_SLURM_NODE=local, the default, in config/env) or share the home folders")
        if state == "failed" and (request / n / "cancel").exists():
            state, why = "cancelled", "cancelled with cancel(request) (it left the queue before running)"
        res = {"name": n, "ok": False, "state": state, "why": why, "job_id": jid,
               "execution": {"executor": "slurm", "slurm_job": jid}}
        if write:
            write_atomic(request / n / f"{n}.json", json.dumps(res, indent=1))
        out[n] = res
    return out


# ---- jobs and cancel ---------------------------------------------------------------------------
def queue_summary() -> dict:
    rc, out, _ = _run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T|%C"])
    run = qd = cpus = 0
    for line in out.splitlines() if rc == 0 else []:
        f = (line.split("|") + [""] * 4)[:4]
        if not f[1].startswith(f"{CLI}:"):
            continue
        if f[2] in ("RUNNING", "COMPLETING"):
            run += 1
            cpus += int(f[3]) if f[3].isdigit() else 0
        else:
            qd += 1
    machine = {}
    rc, out, _ = _run(["sinfo", "-h", "-o", "%C"])
    if rc == 0 and out.strip():
        a, i, o, t = (out.strip().splitlines()[0].split("/") + ["0"] * 4)[:4]
        try:
            machine = {"cpus": int(t), "cpus_allocated": int(a)}
        except ValueError:
            pass
    return {"queue": {"running": run, "queued": qd, "cpus_in_use": cpus}, "machine": machine}


def cancel_slurm(request: Path, names: list[str]) -> list[str]:
    ids = []
    for n in names:
        jf = request / n / "job.json"
        if jf.is_file() and not (request / n / f"{n}.json").is_file():
            ids.append(str(json.loads(jf.read_text())["job_id"]))
    if ids:
        _run(["scancel"] + ids)
    return ids
