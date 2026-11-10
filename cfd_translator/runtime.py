"""runtime.py — cfdt's llmt-runtime: the executor code the translators share, with cfdt's settings and hooks.
Version: 2026.11.09.4

    RT                 llmt_runtime.Runtime for cfdt (CFDT_ settings, the `cfdt` CLI, MPI: --ntasks, one node)
    RUNTIME_VERSION    llmt-runtime's version (tool_status().executor.runtime)

Phase 1 (2026.11.09.2): workers, out of memory (the classification, the job's memory limit), how a job that left the
queue ended (final_state), version recording and ran_with, request_name come from llmt-runtime. Phase 2
(2026.11.09.4): the SLURM layer (settings, the machine, caps' node memory, CFDT_SLURM_TIME, job.sh and job.json,
submission with --dependency, states, the queue's CPUs, cancel's flags and scancel) is llmt-runtime's too, with cfdt's
sizing as hooks: the ranks (executor.procs_for_job: the cell cap), the memory (executor.mem_auto: 2 GB + 4 GB per
million cells), job.json's cells_estimate (executor.job_extra), the progress of running jobs (api.job_progress), the
out-of-memory advice and the timeout hint. The job's time limit with CFDT_SLURM_TIME=timeout is the runtime's default
(ceil(timeout / 60) + 10 minutes, as cfdt had it), and its resource lines are the runtime's MPI lines with one_node
(--ntasks=N, --nodes=1, --ntasks-per-core=1, as cfdt wrote them). The MCP server's stale handling stays in
mcp_server.py until phase 3.

The hooks are small wrappers that import executor and api when called: executor imports RT from here.
"""
from __future__ import annotations

import json
from pathlib import Path

from llmt_runtime import Hooks, Runtime
from llmt_runtime import __version__ as RUNTIME_VERSION  # noqa: F401

from . import __version__


def oom_advice(folder: Path) -> str:
    """Why and what to ask for when SLURM killed a study's job for memory: the job's --mem (job.json 'mem'; before
    2026.11.09.4 only on sbatch's command line) and ranks."""
    try:
        job = json.loads((Path(folder) / "job.json").read_text())
    except (OSError, ValueError):
        job = {}
    mem = job.get("mem") or next((a.split("=", 1)[1] for a in job.get("command", []) if str(a).startswith("--mem=")),
                                 None)
    n = int(job.get("procs") or 1)
    had = f" ({mem} with {n} rank{'s' if n != 1 else ''})" if mem else ""
    cells = f"; about {int(job['cells_estimate']):,} cells were estimated" if job.get("cells_estimate") else ""
    return (f"SLURM killed the job for memory{had}{cells}: give the study more, \"run\": {{\"mem\": \"<more>\"}} "
            "(within study_format's status.executor.caps; it replaces a --mem in CFDT_SLURM_ARGS), or run it on a "
            "machine with more memory (run.host), or make fewer cells")


def _study_procs(study: dict):
    return (study.get("solve") or {}).get("procs")


def _procs(study: dict) -> int:
    from .executor import _cells, procs_for_job
    return procs_for_job(study, _cells(study))


def _mem_auto(study: dict) -> str:
    from .executor import mem_auto
    return mem_auto(study)


def _job_extra(study: dict) -> dict:
    from .executor import job_extra
    return job_extra(study)


def _running_progress(folder: Path, study: dict) -> dict:
    from .api import job_progress
    return job_progress(folder, study)


RT = Runtime(prefix="CFDT", cli="cfdt", package="cfd_translator", version=__version__, kind="mpi",
             out_default="~/cfdt_runs", min_minutes=0.0, timeout_default=3600.0, slurm_time_default="timeout",
             one_node=True, hooks=Hooks(study_procs=_study_procs, procs=_procs, mem_auto=_mem_auto,
                                        job_extra=_job_extra, oom_advice=oom_advice,
                                        timeout_hint="or lower the study's solve.iterations or coarsen its mesh",
                                        running_progress=_running_progress))
