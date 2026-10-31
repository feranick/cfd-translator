"""A fake SLURM for the tests: sbatch, squeue, scancel, sacct, sinfo (docs/EXECUTION.md, Tests).

Called as `python fake_slurm.py <command> [args]` by small wrappers on PATH. State lives in $FAKE_SLURM_DIR:
one <id>.json per job {name, script, output, pid, pending, final}. sbatch starts job.sh in the background
(its own session, like a real job it outlives the caller) unless FAKE_PENDING=1; squeue reports RUNNING
while the process lives; scancel kills its process group and writes SLURM's CANCELLED line to the log;
sacct reports `final` (tests set TIMEOUT, OUT_OF_MEMORY, ...), else COMPLETED; FAKE_SACCT_FAIL=1 makes
sacct fail as on installs without accounting; FAKE_SBATCH_FAIL=<msg> makes sbatch fail.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(os.environ["FAKE_SLURM_DIR"])


def rec(i):
    return json.loads((D / f"{i}.json").read_text())


def save(i, r):
    (D / f"{i}.json").write_text(json.dumps(r))


def alive(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return st != "Z"
    except (OSError, IndexError):
        return False


def state(r):
    if r.get("final"):
        return None
    if r.get("pending"):
        return "PENDING"
    return "RUNNING" if r.get("pid") and alive(r["pid"]) else None


def sbatch(args):
    if os.environ.get("FAKE_SBATCH_FAIL"):
        print(f"sbatch: error: {os.environ['FAKE_SBATCH_FAIL']}", file=sys.stderr)
        return 1
    script = Path(args[-1])
    opts = {}
    for line in script.read_text().splitlines():
        if line.startswith("#SBATCH --"):
            k, _, v = line[10:].partition("=")
            opts[k] = v
    n = D / "counter"
    i = int(n.read_text()) + 1 if n.exists() else 1001
    n.write_text(str(i))
    r = {"name": opts.get("job-name", ""), "script": str(script), "output": opts.get("output", "slurm.out"),
         "ntasks": int(opts.get("ntasks", "1")), "pid": None, "pending": bool(os.environ.get("FAKE_PENDING")),
         "final": None, "priority": 100}
    if not r["pending"]:
        r["pid"] = start(i, r, opts.get("chdir", "."))
    save(i, r)
    print(i)
    return 0


def start(i, r, cwd):
    env = dict(os.environ, SLURM_JOB_ID=str(i), SLURM_NTASKS=str(r["ntasks"]))
    delay = os.environ.get("FAKE_SBATCH_DELAY", "0")
    with open(r["output"], "a") as log:
        p = subprocess.Popen(["bash", "-c", f"sleep {delay}; exec bash {r['script']}"], cwd=cwd, env=env,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return p.pid


def squeue(args):
    ids, only_pd, fmt = None, False, "%i|%T"
    for k, v in zip(args, args[1:] + [""]):
        if k == "-j":
            ids = v.split(",")
        elif k == "-t":
            only_pd = v in ("PD", "PENDING")
        elif k == "-o":
            fmt = v
    jobs = sorted(int(p.stem) for p in D.glob("*.json"))
    for i in jobs:
        if ids is not None and str(i) not in ids:
            continue
        r = rec(i)
        st = state(r)
        if st is None or (only_pd and st != "PENDING"):
            continue
        vals = {"%i": str(i), "%T": st, "%r": "Resources" if st == "PENDING" else "None",
                "%S": "N/A", "%M": "0:07", "%Q": str(r.get("priority", 100)), "%j": r["name"],
                "%C": str(r["ntasks"])}
        print("|".join(vals.get(t, "") for t in fmt.split("|")))
    return 0


def scancel(args):
    for a in args:
        if not a.isdigit() or not (D / f"{a}.json").exists():
            continue
        r = rec(a)
        if r.get("pid") and alive(r["pid"]):
            try:
                os.killpg(r["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(50):
                if not alive(r["pid"]):
                    break
                time.sleep(0.1)
            with open(r["output"], "a") as log:
                log.write(f"slurmstepd: error: *** JOB {a} ON fake CANCELLED AT {time.strftime('%Y-%m-%dT%H:%M:%S')} ***\n")
        r["final"], r["pending"] = f"CANCELLED by {os.getuid()}", False
        save(a, r)
    return 0


def sacct(args):
    if os.environ.get("FAKE_SACCT_FAIL"):
        print("sacct: error: Slurm accounting storage is disabled", file=sys.stderr)
        return 1
    ids = args[args.index("-j") + 1].split(",")
    for i in ids:
        if (D / f"{i}.json").exists():
            r = rec(i)
            print(f"{i}|{r.get('final') or ('RUNNING' if state(r) else 'COMPLETED')}|{r.get('node', 'node1')}")
            print(f"{i}.batch|COMPLETED")
    return 0


def sinfo(args):
    if os.environ.get("FAKE_SINFO_FAIL"):
        print("sinfo: error: Unable to contact slurm controller", file=sys.stderr)
        return 1
    if "-N" in args:                     # node list: name, host, CPUs, threads per core (FAKE_NODES overrides)
        for line in os.environ.get("FAKE_NODES", "node1 localhost 128 1").split(";"):
            print(line)
        return 0
    print("2/6/0/8")
    return 0


def scontrol(args):
    """scontrol -o show node: one line per node of FAKE_NODES (name host cpus threads [addr])."""
    if os.environ.get("FAKE_SINFO_FAIL"):
        return 1
    if args[:3] == ["-o", "show", "node"]:
        for line in os.environ.get("FAKE_NODES", "node1 localhost 128 1").split(";"):
            f = line.split()
            addr = f[4] if len(f) > 4 else f[1]
            print(f"NodeName={f[0]} Arch=x86_64 CPUTot={f[2]} NodeAddr={addr} NodeHostName={f[1]} State=IDLE")
        return 0
    if args[:2] == ["show", "config"]:                # FAKE_SELECT: "select/cons_tres CR_Core_Memory" (default)
        sel, par = (os.environ.get("FAKE_SELECT") or "select/cons_tres CR_Core_Memory").split()
        print(f"Configuration data as of 2026-10-31T00:00:00\nSelectType              = {sel}\n"
              f"SelectTypeParameters    = {par}\nSlurmctldPort           = 6817")
        return 0
    if args[:3] == ["-o", "show", "partition"]:
        print(f"PartitionName=main Nodes=node1 OverSubscribe={os.environ.get('FAKE_OVERSUBSCRIBE', 'NO')} State=UP")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit({"sbatch": sbatch, "squeue": squeue, "scancel": scancel, "sacct": sacct, "sinfo": sinfo,
              "scontrol": scontrol}[sys.argv[1]](sys.argv[2:]))
