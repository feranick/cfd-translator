"""cpushare.py — is anything else running on this job's CPUs? (Linux /proc; a SLURM job's view)
Version: 2026.10.31.1

On spark1 an 18-rank job ran at 36 s per iteration next to another run on a 20-core node (2026.10.30); the same
case took 0.5 s per step alone. SLURM should never give two jobs the same cores, but it does when its core
accounting is off (SelectTypeParameters without CR_Core/CR_CPU, OverSubscribe=YES/FORCE), and it cannot stop
programs started outside it (an MCP server running studies locally, a user's own runs, orphaned ranks).

    Watcher(folder, interval).start() / .stop() -> summary | None
        samples /proc every interval seconds while the solver runs: the CPU time that processes outside this
        job (not its descendants) spent on the job's CPUs; writes <folder>/cpu_share.json while it happens
    competitors(cpus, root, before, dt) -> (cores taken, [{pid, cmd, user, cores}])
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


def competitors(cpus: set, root: int, before: dict, after: dict, dt: float) -> tuple[float, list]:
    """Cores' worth of CPU time (over dt seconds) that processes outside root's tree spent on cpus."""
    mine = _descendants(after, root) | _descendants(before, root)
    taken, who = 0.0, []
    for pid, (ppid, cpu, sec, cmd) in after.items():
        if pid in mine or cpu not in cpus or pid not in before:
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

    def sample(self, before: dict, after: dict, dt: float) -> None:
        taken, who = competitors(self.cpus, self.root, before, after, max(dt, 1e-3))
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
                    {"cores_taken": round(taken, 2), "of": len(self.cpus), "at": time.time(), "by": who}))
            except OSError:
                pass

    def stop(self) -> dict | None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=5)
        if self.busy == 0:
            return None
        top = sorted(self.top.items(), key=lambda kv: -kv[1])[:5]
        return {"max_cores_taken": round(self.max, 2), "cpus": len(self.cpus), "samples": self.samples,
                "samples_shared": self.busy, "by": [{"cmd": c, "user": u, "cores": round(v, 2)} for (c, u), v in top]}


def warning(summary: dict) -> str:
    who = ", ".join(b["cmd"] + (f" ({b['user']})" if b.get("user") else "") for b in summary["by"][:3])
    return (f"other processes ran on this job's CPUs: up to {summary['max_cores_taken']:g} of its {summary['cpus']} cores "
            f"({summary['samples_shared']} of {summary['samples']} one-minute samples; {who}), so it ran slower than it "
            "should. SLURM gives a job's cores to no one else only when it counts cores (SelectTypeParameters CR_Core "
            "or CR_CPU, OverSubscribe=NO): ask the admin, and check for runs started outside SLURM (MCP servers with "
            "a local executor, your own programs).")
