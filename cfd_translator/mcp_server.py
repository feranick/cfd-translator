"""mcp_server.py — MCP server: lets LLM-based tools run CFD studies with Gmsh and OpenFOAM.
Version: 2026.10.16.1

Tools (all JSON):
  study_format      the study format: geometry, domain, fluids, models, boundary conditions, selectors. Read it first.
  validate_studies  static checks of {name: study} (fast; fix errors before running)
  run_studies       meshes, solves and checks {name: study} (returns within ~45 s; long studies stay pending)
  study_status      collects the pending studies of a run_studies request
  get_case          the OpenFOAM case of one study (dictionaries as text), without solving
  run_benchmarks    the benchmark studies with known answers (is the toolchain right?)
  get_file          a written file as base64 (for callers on another machine)

Run:  cfdt-mcp                       (stdio)
      cfdt-mcp --http 8768 [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse

from . import api

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
    _V2 = True
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    _V2 = False

INSTRUCTIONS = """cfd-translator runs steady incompressible CFD (OpenFOAM, laminar or k-omega SST) on one
geometry: internal flow (the geometry is the fluid: pipes, channels, manifolds) or external flow (a
domain box minus the part: drag, lift, wall pressures). A study names the geometry (a
build123d-translator goal, a STEP file, a box or a cylinder), the fluid, the model, and boundary
conditions (inlet, outlet, opening, wall, slip, symmetry) on faces chosen by normal, point or cylinder
radius as in build123d-translator and fea-translator, plus domain-box sides for external flow; faces no
bc takes are no-slip walls. Results: flow rates, pressures, pressure drop, wall forces and
coefficients, probes, and loads for fea-translator (external flow).
Workflow: call study_format once and follow its conventions; validate_studies and fix every error;
run_studies (long studies come back as 'pending': poll study_status with the request path). Check
'selection' (what each bc took), warnings, solve.converged and checks (mass and momentum balance)
before trusting numbers; rerun with a finer mesh to check convergence.
Units: geometry mm; velocity m/s; pressure Pa; flow m3/s; forces N."""

server = _Server("cfd-translator", instructions=INSTRUCTIONS)


@server.tool()
def study_format() -> dict:
    """The study format: geometry, domain, fluids, models, boundary-condition kinds with their
    parameters, face selectors, conventions, an example; and whether OpenFOAM and Gmsh are installed."""
    return api.study_format()


@server.tool()
def validate_studies(studies: dict) -> dict:
    """Static checks of studies given as {name: study}. Returns per study ok, errors, warnings."""
    return api.validate(studies)


@server.tool()
def run_studies(studies: dict, solver: str = "auto", wait_s: float = 45) -> dict:
    """Meshes, solves (OpenFOAM, or the 2D reference solver) and checks each study. Per study: results
    (flow rates, pressures, pressure drop, wall forces, coefficients, probes, wall shear, y+, loads for
    fea-translator), what each bc's selector took, solver convergence, mass and momentum balance,
    'expect' checks, warnings, and files (the OpenFOAM case and a .tgz of it). Returns after at most
    wait_s seconds: studies still running are listed in 'pending'; collect them with study_status."""
    return api.run(studies, solver, wait=min(max(float(wait_s), 0.0), 50.0))


@server.tool()
def study_status(request: str, wait_s: float = 45) -> dict:
    """The studies of a run_studies request (its 'request' path): finished ones with their results,
    the others in 'pending'. Waits up to wait_s seconds (at most 50) for them to finish."""
    return api.status(request, min(max(float(wait_s), 0.0), 50.0))


@server.tool()
def get_case(study: dict, name: str = "study") -> dict:
    """Meshes one study and writes its OpenFOAM case without solving: mesh quality, what each bc took,
    inlet values, and the case dictionaries as text."""
    return api.case(study, name)


@server.tool()
def run_benchmarks(names: list[str] | None = None, solver: str = "auto") -> dict:
    """Runs the benchmark studies with known answers (Couette, plane Poiseuille, lid-driven cavity vs
    Ghia, Hagen-Poiseuille pipe, Schaefer-Turek cylinder; slow ones by name: cavity_re1000,
    pipe_turbulent; chain_plate with build123d-translator and fea-translator) and reports pass or fail."""
    return api.bench(names, solver)


@server.tool()
def get_file(path: str, max_mb: float = 20) -> dict:
    """A file written by run_studies (a path from its result) as base64."""
    return api.read_file(path, max_mb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, help="serve MCP over streamable HTTP on this port (default: stdio)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.http is None:
        server.run()
    elif _V2:
        server.run(transport="streamable-http", host=a.host, port=a.http)
    else:
        server.settings.host, server.settings.port = a.host, a.http
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
