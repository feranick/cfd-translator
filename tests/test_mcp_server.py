"""The MCP server's contract with llm-translators (docs/SHARED.md): stale servers after a deploy, errors as
ToolError messages, and that tools called in-process (tests, scripts) never refuse or end the process."""
from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest

from cfd_translator import mcp_server as M


def _release(tmp_path: Path, name: str) -> Path:
    pkg = tmp_path / "releases" / name / "cfd-translator" / "cfd_translator"
    pkg.mkdir(parents=True)
    (tmp_path / "releases" / name / "RELEASE.json").write_text("{}")
    (pkg / "__init__.py").write_text('__version__ = "1"\n')
    return pkg


def test_stale_reason_shared_and_checkout(tmp_path):
    a = _release(tmp_path, "2026.10.25.1")
    _release(tmp_path, "2026.10.25.2")
    cur = tmp_path / "current"
    cur.symlink_to(tmp_path / "releases" / "2026.10.25.1")
    assert M.stale_reason(a, "1") == (None, True)
    cur.unlink()
    cur.symlink_to(tmp_path / "releases" / "2026.10.25.2")
    why, shared = M.stale_reason(a, "1")
    assert shared and "2026.10.25.2 is installed" in why
    co = tmp_path / "checkout" / "cfd_translator"
    co.mkdir(parents=True)
    (co / "__init__.py").write_text('__version__ = "2026.10.25.9"\n')
    assert M.stale_reason(co, "2026.10.25.9") == (None, False)
    why, shared = M.stale_reason(co, "2026.10.25.4")
    assert not shared and "2026.10.25.9 is installed" in why


def _call(name, args):
    async def go():
        return await M.server.call_tool(name, args)
    return anyio.run(go)


def test_in_process_tools_never_refuse_or_exit(monkeypatch):
    """llmt deploy --test runs the new release's tests before switching current: they see a 'stale' install."""
    monkeypatch.setitem(M._STATE, "serving", False)
    monkeypatch.setitem(M._STATE, "checked", 1e18)            # keep the 'why' below
    monkeypatch.setitem(M._STATE, "why", "release X is installed; this server runs Y")
    monkeypatch.setitem(M._STATE, "shared", True)
    monkeypatch.setenv("CFDT_EXIT_WHEN_STALE", "1")
    exits = []
    monkeypatch.setattr(M, "_exit_soon", lambda why, delay: exits.append(why))
    assert M._enter_call() is None
    M._leave_call()
    r = _call("guide", {"topic": ""})
    data = json.loads(r.content[0].text)
    assert "topics" in data and "restart the MCP server" in data["server_stale"] and not exits


def test_a_serving_stale_server_refuses_when_idle(monkeypatch):
    monkeypatch.setitem(M._STATE, "serving", True)
    monkeypatch.setitem(M._STATE, "checked", 1e18)
    monkeypatch.setitem(M._STATE, "why", "release X is installed; this server runs Y")
    monkeypatch.setitem(M._STATE, "shared", True)
    monkeypatch.setitem(M._STATE, "exiting", False)
    monkeypatch.setitem(M._STATE, "inflight", 0)
    exits = []
    monkeypatch.setattr(M, "_exit_soon", lambda why, delay: exits.append((why, delay)))
    msg = M._enter_call()
    assert msg and msg.startswith(M.STALE_MARK) and exits == [("release X is installed; this server runs Y", 1.0)]
    with pytest.raises(Exception) as e:
        _call("guide", {"topic": ""})
    assert M.STALE_MARK in str(e.value)
    # busy (a call in flight): answered with the note, no exit
    exits.clear()
    monkeypatch.setitem(M._STATE, "inflight", 1)
    assert M._enter_call() is None and not exits
    M._leave_call()
    # CFDT_EXIT_WHEN_STALE=0: never
    monkeypatch.setitem(M._STATE, "inflight", 0)
    monkeypatch.setenv("CFDT_EXIT_WHEN_STALE", "0")
    assert M._enter_call() is None and not exits
    M._leave_call()


def test_errors_reach_the_caller_as_tool_errors(monkeypatch):
    monkeypatch.setitem(M._STATE, "serving", False)
    with pytest.raises(Exception) as e:
        _call("get_image", {"path": "/etc/passwd"})
    assert type(e.value).__name__ == "ToolError" and "ValueError: no image /etc/passwd" in str(e.value)


def test_no_test_ends_the_process():
    """llmt test counts a pytest run without its JUnit report as failed: nothing here may os._exit or sys.exit."""
    import re
    here = Path(__file__).parent
    for f in here.glob("test_*.py"):
        txt = f.read_text()
        code = "\n".join(l for l in txt.splitlines() if not l.strip().startswith(("#", '"', "assert")))
        assert not re.search(r"\bos\._exit\(|\bsys\.exit\(", code), f.name


CH = {"geometry": {"box": [1000, 100, 10]}, "2d": True, "fluid": {"rho": 1.0, "mu": 1e-3},
      "bcs": [{"kind": "inlet", "faces": {"nx": -1}, "velocity": 0.01}, {"kind": "outlet", "faces": {"nx": 1}}],
      "mesh": {"cells": [40, 16, 1]}}


def test_slow_tool_work_runs_in_workers_and_returns_in_time():
    """Every tool returns within its wait_s: meshing (get_case) and rendering run in worker processes; a call that
    finds them unfinished says 'running' and the same call again collects the result."""
    import time
    from cfd_translator import api
    t0 = time.time()
    r = api.in_worker("meshing", "case", {"study": CH, "name": "w"}, 0)
    assert time.time() - t0 < 5 and r["state"] == "running" and "get_case again" in r["next"]
    r = api.in_worker("meshing", "case", {"study": CH, "name": "w"}, 50)
    assert "case" in r and r["mesh"]["cells"] == 640 and not api.aux_busy()
    assert "error" in api.in_worker("rendering", "render_images", {"study_folder": "/nope", "images": True}, 50)


def test_benchmarks_from_the_tool_are_submitted_not_run_in_the_call(tmp_path):
    import time
    from cfd_translator import api
    t0 = time.time()
    r = api.bench(["couette_2d"], "ref", wait_s=0)
    assert time.time() - t0 < 20 and r.get("bench")
    for _ in range(60):
        r = api.bench_status(r["bench"], 2)
        if r["done"]:
            break
    assert r["passed"] == ["couette_2d"]
