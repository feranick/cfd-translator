"""cli.py — the `cfdt` command line. Every command prints JSON.
Version: 2026.11.13.3

    cfdt status                          versions; OpenFOAM and Gmsh found or not
    cfdt format > study_format.json      the study format
    cfdt validate studies.json           static checks (exit 1 when a study is invalid)
    cfdt run studies.json [--solver S] [--wait SEC]
                                         mesh, solve, check (S: auto | foam | ref); waits for all, or SEC seconds
                                         (--wait 0 on a cluster's login node: submit and return the request)
    cfdt status REQUEST                  results of a request folder (finished and pending studies)
    cfdt case studies.json NAME          write the OpenFOAM case of one study (no solve)
    cfdt bench [NAMES...] [--solver S]   the benchmark studies with known answers
    cfdt guide [TOPIC]                   how-to guides (vehicle_aero)
    cfdt images FOLDER [IMAGES.json]     pictures of a finished study (default views, or the given list)
    cfdt jobs [--limit N] [--active]     recent requests with SLURM jobs, their states; the queue
    cfdt cancel REQUEST [NAMES...]       stop a request's studies (scancel, or the local cancel flag)
    cfdt reprocess REQUEST [NAMES...] [--onset JSON|FILE] [--average JSON]   results again from finished cases
                                         (current post-processing, no solve); --onset: a new onset block for them;
                                         --average: other averaging settings (steady), e.g. '{"window": 0.5}'
    cfdt run-study REQUEST NAME          run one study of a request (what a SLURM job runs; exit 0 = result written)
    cfdt pack REQUEST [--state] [--out F] on a cluster: the request's results as a .tgz to bring back (other/supercomputers/clusters.md)
    cfdt import PATH [--name N]          a packed request (.tgz or folder) as a request of this machine

'-' reads the studies from stdin. Interrupted (Ctrl-C) while `run` or `bench` waits for SLURM jobs, the command says
which jobs are left in the queue and how to cancel them (it cancels nothing), then exits with 130.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import api


def _load(path: str) -> dict:
    return json.load(sys.stdin) if path == "-" else json.loads(open(path).read())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cfdt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status")
    p.add_argument("request", nargs="?")
    sub.add_parser("format")
    p = sub.add_parser("guide")
    p.add_argument("topic", nargs="?")
    p = sub.add_parser("validate")
    p.add_argument("studies")
    p = sub.add_parser("run")
    p.add_argument("studies")
    p.add_argument("--solver", default=None)
    p.add_argument("--wait", type=float, default=None, help="seconds to wait (default: until all finish)")
    p = sub.add_parser("case")
    p.add_argument("studies")
    p.add_argument("name")
    p = sub.add_parser("images")
    p.add_argument("folder")
    p.add_argument("images", nargs="?")
    p = sub.add_parser("jobs")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--active", action="store_true")
    p = sub.add_parser("cancel")
    p.add_argument("request")
    p.add_argument("names", nargs="*")
    p = sub.add_parser("reprocess", help="results again from finished cases, without solving")
    p.add_argument("request")
    p.add_argument("names", nargs="*")
    p.add_argument("--onset", default=None, help="a new onset block (JSON, or a file holding it) for these studies")
    p.add_argument("--average", default=None, help="other averaging settings for steady runs (JSON), e.g. '{\"window\": 0.5}'")
    p = sub.add_parser("run-study")
    p.add_argument("request")
    p.add_argument("name")
    p = sub.add_parser("pack", help="a request's results as a .tgz (on a cluster), to import elsewhere")
    p.add_argument("request")
    p.add_argument("--state", action="store_true", help="add the mesh and fields (state.npz) for render_images")
    p.add_argument("--out", default=None)
    p = sub.add_parser("import", help="a packed request (.tgz or folder) as a request of this machine")
    p.add_argument("path")
    p.add_argument("--name", default=None)
    p = sub.add_parser("bench")
    p.add_argument("names", nargs="*")
    p.add_argument("--solver", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run-study":
        sys.exit(api.run_study_job(a.request, a.name))
    try:
        out = _command(a)
    except KeyboardInterrupt:     # run / bench printed the SLURM jobs left in the queue (api.run, bench.run)
        sys.exit(130)
    print(json.dumps(out, indent=1, default=float))
    code = exit_code(a.cmd, out)
    if code:
        sys.exit(code)


def exit_code(cmd: str, out) -> int:
    """1 when the command did not do what it was asked, so `cfdt validate s.json && cfdt run s.json` stops at an
    invalid study: validate with ok false, or any answer with a top-level 'error'. Results (a study that failed, a
    benchmark that did not pass) are data in the JSON and keep 0."""
    if not isinstance(out, dict):
        return 0
    if cmd == "validate" and out.get("ok") is False:
        return 1
    return 1 if out.get("error") else 0


def _command(a) -> dict:
    if a.cmd == "status":
        out = api.status(a.request) if a.request else api.tool_status()
    elif a.cmd == "format":
        out = api.study_format()
    elif a.cmd == "guide":
        out = api.guide(a.topic)
    elif a.cmd == "validate":
        out = api.validate(_load(a.studies))
    elif a.cmd == "run":
        out = api.run(_load(a.studies), a.solver, wait=a.wait)
    elif a.cmd == "case":
        st = _load(a.studies)
        out = api.case(st[a.name], a.name)
    elif a.cmd == "jobs":
        out = api.jobs(a.limit, a.active)
    elif a.cmd == "cancel":
        out = api.cancel(a.request, a.names or None)
    elif a.cmd == "reprocess":
        on = None
        if getattr(a, "onset", None):
            txt = Path(a.onset).read_text() if Path(a.onset).is_file() else a.onset
            on = json.loads(txt)
        av = None
        if getattr(a, "average", None):
            av = json.loads(Path(a.average).read_text() if Path(a.average).is_file() else a.average)
        out = api.reprocess(a.request, a.names or None, on, av)
    elif a.cmd == "pack":
        out = api.pack(a.request, a.state, a.out)
    elif a.cmd == "import":
        out = api.import_results(a.path, a.name)
    elif a.cmd == "images":
        out = api.render_images(a.folder, _load(a.images) if a.images else True)
    else:
        out = api.bench(a.names or None, a.solver)
    return out


if __name__ == "__main__":
    main()
