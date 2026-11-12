"""cpushare.py — is anything else running on this job's CPUs? (Linux /proc; a SLURM job's view)
Version: 2026.11.12.10

On spark1 an 18-rank job ran at 36 s per iteration next to another run on a 20-core node (2026.10.30); the same
case took 0.5 s per step alone. SLURM should never give two jobs the same cores, but it does when its core
accounting is off (SelectTypeParameters without CR_Core/CR_CPU, OverSubscribe=YES/FORCE), and it cannot stop
programs started outside it (an MCP server running studies locally, a user's own runs, orphaned ranks).

    Watcher(folder, interval).start() / .stop() -> summary | None
        samples /proc every interval seconds while the solver runs: the CPU time that processes outside this
        job (not its descendants) spent on the job's CPUs; writes <folder>/cpu_share.json while it happens
    competitors(cpus, root, before, dt) -> (cores taken, [{pid, cmd, user, cores}])
    reap(job_active) -> [{pid, cmd, job}]   kill this user's OpenFOAM/MPI processes left by SLURM jobs that ended
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _stat(pid: str):
    """(ppid, last CPU, CPU seconds, command) of a process, or None."""
    try:
        t = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    r = t.rfind(")")
    f = t[r + 2:].split()
    try:
        return int(f[1]), int(f[36]), (int(f[11]) + int(f[12])) / HZ, t[t.find("(") + 1:r]
    except (IndexError, ValueError):
        return None


def snapshot() -> dict:
    out = {}
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            s = _stat(pid)
            if s:
                out[int(pid)] = s
    return out


def _descendants(snap: dict, root: int) -> set:
    kids: dict[int, list] = {}
    for pid, (ppid, *_r) in snap.items():
        kids.setdefault(ppid, []).append(pid)
    seen, todo = {root}, [root]
    while todo:
        for k in kids.get(todo.pop(), []):
            if k not in seen:
                seen.add(k)
                todo.append(k)
    return seen


def _user(pid: int) -> str | None:
    try:
        import pwd
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Uid:"):
                return pwd.getpwuid(int(line.split()[1])).pw_name
    except (OSError, KeyError, ImportError, ValueError):
        return None
    return None


def _in_job(pid: int, job: str | None, study: str | None) -> bool:
    """Whether a process belongs to this SLURM job (its cgroup or SLURM_JOB_ID) or to this study (CFDT_STUDY): srun
    starts a job's ranks under slurmstepd, outside the batch script's process tree (Perlmutter: the job's own 128
    foamRun ranks were reported as other processes on its CPUs)."""
    if job:
        try:
            cg = Path(f"/proc/{pid}/cgroup").read_text()
            if f"job_{job}/" in cg or cg.rstrip().endswith(f"job_{job}"):
                return True
        except OSError:
            pass
    env = _environ(pid)
    return bool((job and env.get("SLURM_JOB_ID") == job) or (study and env.get("CFDT_STUDY") == study))


def physical_cores(cpus: set) -> int:
    """The physical cores among logical CPUs (hardware threads of one core counted once; Linux sysfs)."""
    seen = set()
    for c in cpus:
        try:
            sib = Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text().strip()
        except OSError:
            sib = str(c)
        seen.add(sib)
    return len(seen) or len(cpus)


def competitors(cpus: set, root: int, before: dict, after: dict, dt: float, ours=None) -> tuple[float, list]:
    """Cores' worth of CPU time (over dt seconds) that processes outside root's tree spent on cpus. ours(pid) -> bool
    leaves out processes of this job started outside the tree (srun's ranks)."""
    mine = _descendants(after, root) | _descendants(before, root)
    taken, who = 0.0, []
    for pid, (ppid, cpu, sec, cmd) in after.items():
        if pid in mine or cpu not in cpus or pid not in before:
            continue
        if ours is not None and ours(pid):
            continue
        d = sec - before[pid][2]
        if d <= 0:
            continue
        taken += d / dt
        who.append({"pid": pid, "cmd": cmd, "cores": round(d / dt, 2)})
    who.sort(key=lambda w: -w["cores"])
    for w in who[:5]:
        w["user"] = _user(w["pid"])
    return taken, who[:5]


class Watcher:
    def __init__(self, folder: Path, interval: float = 60.0, root: int | None = None):
        self.folder, self.interval = Path(folder), float(interval)
        self.root = root or os.getpid()
        self.cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
        self.cores = physical_cores(self.cpus) if self.cpus else 0
        self.job = os.environ.get("SLURM_JOB_ID") or None
        self.study = str(Path(folder).resolve() / "case")
        self._ours: dict = {}
        self.max, self.samples, self.busy, self.top = 0.0, 0, 0, {}
        self._stop = threading.Event()
        self._t = None

    def start(self):
        if not self.cpus or not Path("/proc/self/stat").exists():
            return self
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        before, t0 = snapshot(), time.time()
        while not self._stop.wait(self.interval):
            after, t1 = snapshot(), time.time()
            self.sample(before, after, t1 - t0)
            before, t0 = after, t1

    def ours(self, pid: int) -> bool:
        if pid not in self._ours:
            self._ours[pid] = _in_job(pid, self.job, self.study)
        return self._ours[pid]

    def sample(self, before: dict, after: dict, dt: float) -> None:
        for gone in [p for p in self._ours if p not in after]:
            self._ours.pop(gone, None)
        taken, who = competitors(self.cpus, self.root, before, after, max(dt, 1e-3), self.ours)
        self.samples += 1
        if taken >= 0.5:
            self.busy += 1
            for w in who:
                k = (w["cmd"], w.get("user"))
                self.top[k] = max(self.top.get(k, 0.0), w["cores"])
        self.max = max(self.max, taken)
        if taken >= 0.5:
            try:
                (self.folder / "cpu_share.json").write_text(json.dumps(
                    {"cores_taken": round(taken, 2), "of": self.cores or len(self.cpus), "hw_threads": len(self.cpus),
                     "at": time.time(), "by": who}))
            except OSError:
                pass

    def stop(self) -> dict | None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=5)
        if self.busy == 0:
            return None
        top = sorted(self.top.items(), key=lambda kv: -kv[1])[:5]
        return {"max_cores_taken": round(self.max, 2), "cpus": self.cores or len(self.cpus), "hw_threads": len(self.cpus),
                "samples": self.samples,
                "samples_shared": self.busy, "by": [{"cmd": c, "user": u, "cores": round(v, 2)} for (c, u), v in top]}


def warning(summary: dict) -> str:
    who = ", ".join(b["cmd"] + (f" ({b['user']})" if b.get("user") else "") for b in summary["by"][:3])
    return (f"other processes ran on this job's CPUs: up to {summary['max_cores_taken']:g} of its {summary['cpus']} cores "
            f"({summary['samples_shared']} of {summary['samples']} one-minute samples; {who}), so it ran slower than it "
            "should. SLURM gives a job's cores to no one else only when it counts cores (SelectTypeParameters CR_Core "
            "or CR_CPU, OverSubscribe=NO): ask the admin, and check for runs started outside SLURM (MCP servers with "
            "a local executor, your own programs).")


_FOAM_CMDS = ("foamRun", "snappyHexMesh", "mpirun", "orted", "prted", "prterun", "decomposePar", "reconstructPar",
              "checkMesh", "surfaceFeatures", "blockMesh", "subsetMesh")


def _environ(pid: int) -> dict:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        k, _, v = item.partition(b"=")
        if k in (b"SLURM_JOB_ID", b"CFDT_STUDY"):
            out[k.decode()] = v.decode(errors="replace")
    return out


def reap(job_active, me: int | None = None) -> list[dict]:
    """This user's OpenFOAM and MPI processes (or any marked CFDT_STUDY) whose SLURM job has ended: leftovers of a
    cancelled job, which keep the node's cores busy (spark1, spark2: load 14-16 with no job, 2026.10.31). They are
    killed (their process group, TERM then KILL). job_active(job_id) -> bool (squeue); a job it cannot tell
    about counts as active, so nothing of a running job is touched. Returns what was killed."""
    import signal
    uid = os.getuid()
    me = me or os.getpid()
    mine = _descendants(snapshot(), me)
    cache: dict = {}
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in mine:
            continue
        pid = int(pid)
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
        except OSError:
            continue
        env = _environ(pid)
        job = env.get("SLURM_JOB_ID")
        if not job or job == os.environ.get("SLURM_JOB_ID"):
            continue
        st = _stat(str(pid))
        if st is None:
            continue
        cmd = st[3]
        if not (env.get("CFDT_STUDY") or any(c in cmd for c in _FOAM_CMDS)):
            continue
        if job not in cache:
            try:
                cache[job] = bool(job_active(job))
            except Exception:  # noqa: BLE001
                cache[job] = True
        if cache[job]:
            continue
        found.append({"pid": pid, "cmd": cmd, "job": job})
    for f in found:
        for sig in (signal.SIGTERM,):
            try:
                os.kill(f["pid"], sig)
            except OSError:
                pass
    if found:
        time.sleep(3)
        for f in found:
            try:
                os.kill(f["pid"], signal.SIGKILL)
            except OSError:
                pass
    return found
