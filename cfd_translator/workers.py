"""workers.py — cfdt's worker processes end with the server that started them (from llmt-runtime).
Version: 2026.11.09.2

    watch_parent(ppid)   initializer of the study pool and the aux pool (api._Pool, api._aux_pool): a daemon thread
                         ends the worker, and the OpenFOAM process groups it started (foam.kill_all), when its parent
                         (or the server ppid) is gone; SIGTERM (Pool.terminate) does the same

The code is llmt-runtime's workers module (llmt_runtime.workers): polling os.getppid(), not PR_SET_PDEATHSIG (it fires
when the *thread* that created the child exits, and the pools are created from the MCP server's call threads). cfdt's
own list of OpenFOAM runs (foam.kill_all) is the runtime's on_orphan cleanup. Before 2026.11.09.2 a worker given SIGTERM
died at once and its mpirun and ranks, in their own process groups, ran on. SLURM jobs do not depend on the server.
"""
from __future__ import annotations

from llmt_runtime import workers as _W

PARENT_POLL_S = _W.PARENT_POLL_S
_STARTED = _W._STARTED


def _kill_foam() -> None:
    try:
        from .foam import kill_all
        kill_all()
    except Exception:  # noqa: BLE001 (the worker ends anyway)
        pass


def watch_parent(ppid: int | None = None) -> None:
    """Starts (once per process) the watcher that ends this worker, and the OpenFOAM runs it started, when its parent
    `ppid` (default: the parent now) is gone (llmt_runtime.workers.watch_parent)."""
    _W.PARENT_POLL_S = PARENT_POLL_S       # (a test may set a shorter poll here)
    _W.on_orphan(_kill_foam)
    _W.watch_parent(ppid, name="cfdt")
