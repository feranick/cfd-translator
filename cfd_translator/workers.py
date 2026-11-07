"""workers.py — cfdt's worker processes end with the server that started them.
Version: 2026.11.07.2

    watch_parent(ppid)   initializer of the study pool and the aux pool (api._Pool, api._aux_pool): a daemon thread
                         ends the worker, and the OpenFOAM process groups it started (foam.kill_all), when its parent
                         is gone

A spawn worker holds both ends of its task queue, so it never sees EOF when the MCP server dies (crash, kill -9, an
app restart) and would wait forever; one in the middle of a local solve would also leave mpirun and its ranks
running. feat found 7 orphaned workers from three releases on carbonio (2026.11.02) and fixed it the same way.

Not PR_SET_PDEATHSIG: on Linux it fires when the *thread* that created the child exits, and the pools are created
from the MCP server's call threads, which come and go. The worker polls os.getppid() instead (PARENT_POLL_S): when
the parent dies the worker is re-parented and its parent pid changes. SLURM jobs do not depend on the server and
are not affected.
"""
from __future__ import annotations

import os
import threading
import time

PARENT_POLL_S = 1.0
_STARTED = {"pid": None}        # the process the watcher runs in (a fork would inherit the flag)


def _orphaned() -> None:
    try:
        from .foam import kill_all
        kill_all()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


def watch_parent(ppid: int | None = None) -> None:
    """Starts (once per process) a daemon thread that ends this process, and the OpenFOAM runs it started, when its
    parent `ppid` (default: the parent now) is gone."""
    if _STARTED["pid"] == os.getpid():
        return
    _STARTED["pid"] = os.getpid()
    ppid = int(ppid or os.getppid())
    if os.getppid() != ppid:      # the parent died before the worker started
        _orphaned()

    def loop():
        while True:
            time.sleep(PARENT_POLL_S)
            if os.getppid() != ppid:
                _orphaned()
    threading.Thread(target=loop, daemon=True, name="cfdt-parent-watch").start()
