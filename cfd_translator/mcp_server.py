"""mcp_server.py — MCP server: lets LLM-based tools run CFD studies with Gmsh and OpenFOAM.
Version: 2026.10.20.1

Tools (all JSON):
  study_format      the study format: geometry, domain, fluids, models, boundary conditions, selectors. Read it first.
  validate_studies  static checks of {name: study} (fast; fix errors before running)
  run_studies       meshes, solves and checks {name: study} (returns within ~45 s; long studies stay pending)
  study_status      collects the pending studies of a run_studies request (states: queued, running, ...)
  jobs              recent and active SLURM runs on this machine (from any conversation), the queue
  cancel            stops a request's studies
  get_case          the OpenFOAM case of one study (dictionaries as text), without solving
  run_benchmarks    the benchmark studies with known answers (is the toolchain right?)
  get_image         a PNG (flow slices, streamlines, wall pressure) as an image
  render_images     more pictures of a finished study, without solving again
  get_file          a written file as base64 (for callers on another machine)

Run:  cfdt-mcp                       (stdio)
      cfdt-mcp --http 8768 [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse

from . import api

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver import MCPServer as _Server
    _V2 = True
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp import Image
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
before trusting numbers; if 'reliable' is false, do not report the numbers as results (rerun as
'unreliable_because' says). Rerun with a finer mesh to check mesh convergence. Use head_loss_m (total
pressure) for losses. Pumps and fans: a 'rotating' zone (MRF) around the impeller, its faces a wall
with rotation: the zone's name; results give rotating.<zone>.torque_Nm, power_W and pump.head_m,
efficiency. Long runs: set solve.procs (MPI ranks). Pending studies report 'progress' with eta_min_max:
polling costs tokens, so when a run needs more than about 5 minutes, tell the user when it should
finish and stop; collect it later with study_status.
Heat and compressible gas: "energy": true (OpenFOAM fluid; air becomes a perfect gas up to transonic
speeds, water keeps constant density; temperatures in degrees C; walls with T, q, power or h + T_inf;
"gravity" for natural convection); results add heat.<wall bc>, flow.<bc>.mdot/T_mean_C/mach, max_mach,
checks.energy_balance and feat_thermal (bcs for fea-translator's thermal analysis, external flow).
Long studies run as SLURM jobs on the machine (when CFDT_EXECUTOR=slurm): run_studies returns them
pending with job ids. They do not depend on this conversation. Report the expected finish time and
stop; do not poll. Later, study_status(request) collects them; jobs() lists recent and active runs
(also those started in other conversations); cancel(request) stops them. A study's "run": {timeout_s, mem,
procs, queue} overrides the machine's defaults for that study, within the caps in study_format's status.
Pictures: add "images": true (or slices and surfaces) to a study; results.images.<name>.path then
get_image shows each; render_images makes other views of a finished study without solving again.
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
def run_benchmarks(names: list[str] | None = None, solver: str = "auto", wait_s: float = 30) -> dict:
    """Runs the benchmark studies with known answers (Couette, plane Poiseuille, lid-driven cavity vs
    Ghia, Hagen-Poiseuille pipe, Schaefer-Turek cylinder; slow ones by name: cavity_re1000,
    pipe_turbulent; chain_plate with build123d-translator and fea-translator) and reports pass or fail.
    With SLURM the cases are submitted as jobs and this returns after at most wait_s seconds: collect the
    rest with bench_status(bench)."""
    return api.bench(names, solver, wait_s=float(wait_s))


@server.tool()
def bench_status(bench: str, wait_s: float = 45) -> dict:
    """Pass/fail of benchmarks submitted by run_benchmarks (its 'bench' folder): finished cases with their
    checks, the others pending (SLURM jobs that keep running without this conversation)."""
    return api.bench_status(bench, min(float(wait_s), 50.0))


@server.tool()
def jobs(limit: int = 20, active: bool = False) -> dict:
    """Recent requests with SLURM jobs on this machine (from any conversation): each study's state
    (queued, running, done, timed_out, out_of_memory, cancelled, failed), job id and ETA; the queue of
    this translator's jobs and the machine's CPUs. active=True: only requests still queued or running."""
    return api.jobs(int(limit), bool(active))


@server.tool()
def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops a request's studies (or the named ones): scancel for SLURM jobs, a cancel flag for local
    ones. Returns their states."""
    return api.cancel(request, names)


@server.tool()
def get_image(path: str) -> Image:
    """A PNG written by run_studies (results' images.<name>.path) or render_images, as an image you can
    see and show: slices of velocity or pressure (with streamlines) and wall surfaces."""
    return Image(path=str(api.image_path(path)))


@server.tool()
def render_images(study_folder: str, images: list | bool = True) -> dict:
    """New pictures of a finished study without solving again: study_folder is <request>/<study name>
    (the folder holding <name>.json). images: true (defaults) or [{name, slice: {axis, at}, field,
    streamlines, window} | {name, surface, field, view}] as in study_format. Then get_image each path."""
    return api.render_images(study_folder, images)


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
