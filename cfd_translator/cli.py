"""cli.py — the `cfdt` command line. Every command prints JSON.
Version: 2026.10.17.1

    cfdt status                          versions; OpenFOAM and Gmsh found or not
    cfdt format > study_format.json      the study format
    cfdt validate studies.json           static checks
    cfdt run studies.json [--solver S]   mesh, solve, check (S: auto | foam | ref); waits for all
    cfdt status REQUEST                  results of a request folder (finished and pending studies)
    cfdt case studies.json NAME          write the OpenFOAM case of one study (no solve)
    cfdt bench [NAMES...] [--solver S]   the benchmark studies with known answers
    cfdt images FOLDER [IMAGES.json]     pictures of a finished study (default views, or the given list)

'-' reads the studies from stdin.
"""
from __future__ import annotations

import argparse
import json
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
    p = sub.add_parser("validate")
    p.add_argument("studies")
    p = sub.add_parser("run")
    p.add_argument("studies")
    p.add_argument("--solver", default=None)
    p = sub.add_parser("case")
    p.add_argument("studies")
    p.add_argument("name")
    p = sub.add_parser("images")
    p.add_argument("folder")
    p.add_argument("images", nargs="?")
    p = sub.add_parser("bench")
    p.add_argument("names", nargs="*")
    p.add_argument("--solver", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        out = api.status(a.request) if a.request else api.tool_status()
    elif a.cmd == "format":
        out = api.study_format()
    elif a.cmd == "validate":
        out = api.validate(_load(a.studies))
    elif a.cmd == "run":
        out = api.run(_load(a.studies), a.solver)
    elif a.cmd == "case":
        st = _load(a.studies)
        out = api.case(st[a.name], a.name)
    elif a.cmd == "images":
        out = api.render_images(a.folder, _load(a.images) if a.images else True)
    else:
        out = api.bench(a.names or None, a.solver)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
