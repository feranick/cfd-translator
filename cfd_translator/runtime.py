"""runtime.py — cfdt's llmt-runtime: the executor code the translators share, with cfdt's settings and hooks.
Version: 2026.11.09.2

    RT                 llmt_runtime.Runtime for cfdt (CFDT_ settings, the `cfdt` CLI, MPI: --ntasks)
    RUNTIME_VERSION    llmt-runtime's version (tool_status().executor.runtime)

Phase 1 (2026.11.09.2): workers, out of memory (the classification, the job's memory limit), how a job that left the
queue ended (final_state), version recording and ran_with, request_name come from llmt-runtime. The SLURM layer
(settings, caps, job.sh, states, jobs, cancel) and the MCP server's stale handling stay in executor.py and
mcp_server.py until phases 2 and 3.
"""
from __future__ import annotations

import json
from pathlib import Path

from llmt_runtime import Hooks, Runtime
from llmt_runtime import __version__ as RUNTIME_VERSION  # noqa: F401

from . import __version__


def oom_advice(folder: Path) -> str:
    """Why and what to ask for when SLURM killed a study's job for memory: the job's --mem and ranks (job.json)."""
    try:
        job = json.loads((Path(folder) / "job.json").read_text())
    except (OSError, ValueError):
        job = {}
    mem = next((a.split("=", 1)[1] for a in job.get("command", []) if str(a).startswith("--mem=")), None)
    n = int(job.get("procs") or 1)
    had = f" ({mem} with {n} rank{'s' if n != 1 else ''})" if mem else ""
    cells = f"; about {int(job['cells_estimate']):,} cells were estimated" if job.get("cells_estimate") else ""
    return (f"SLURM killed the job for memory{had}{cells}: give the study more, \"run\": {{\"mem\": \"<more>\"}} "
            "(within study_format's status.executor.caps; it replaces a --mem in CFDT_SLURM_ARGS), or run it on a "
            "machine with more memory (run.host), or make fewer cells")


RT = Runtime(prefix="CFDT", cli="cfdt", package="cfd_translator", version=__version__, kind="mpi",
             out_default="~/cfdt_runs", min_minutes=0.0, timeout_default=3600.0, slurm_time_default="timeout",
             one_node=True, hooks=Hooks(oom_advice=oom_advice,
                                        timeout_hint="or lower the study's solve.iterations or coarsen its mesh",
                                        study_procs=lambda st: (st.get("solve") or {}).get("procs")))
