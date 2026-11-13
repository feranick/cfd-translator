"""mcp_server.py — MCP server: lets LLM-based tools run CFD studies with Gmsh and OpenFOAM.
Version: 2026.11.13.4

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

Stale: a server keeps the code it started with. When a newer release is installed (shared install: <prefix>/current
      points elsewhere; checkout: cfdt's or llmt-runtime's __init__.py has another version), a call that finds the
      server idle is refused with STALE_SERVER (the router repeats it on a new server) and the server exits; an idle
      server exits within about 2 x CFDT_STALE_CHECK_S (10 s) by itself; a busy one answers (with server_stale) and
      exits once idle. CFDT_EXIT_WHEN_STALE=1 always, 0 never; default: shared installs only. The exit stops the
      worker pools for at most 10 s, then exits whatever happens. Errors reach the caller as ToolError messages
      (MCP SDK 1.x and 2.x). Every tool runs in a worker thread under a 48 s deadline (SLURM commands and waits keep
      to it). Since 2026.11.09.6 the stale handling, the tool wrapper, the server's log ($CFDT_OUT/.server) and the
      signal handlers (SIGTERM, SIGHUP, SIGINT stop the worker pools first) are llmt-runtime's (llmt_runtime.server).
Run:  cfdt-mcp                       (stdio)
      cfdt-mcp --http 8768 [--host 0.0.0.0]
"""
# (no `from __future__ import annotations`: the tool wrapper is llmt-runtime's, and MCP SDK 1.x evaluates string
# annotations such as `-> Image` in the wrapper's module, where they are not defined; Python >= 3.10 takes them as is)
import argparse

from llmt_runtime import workers as _W
from llmt_runtime.server import Server
from llmt_runtime.server import STALE_MARK, stale_reason  # noqa: F401 (cfdt's names: tests and other callers)

from . import api, workers
from .runtime import RT

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
    _V2 = True
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp import Image
    from mcp.server.fastmcp.exceptions import ToolError
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
efficiency. The zone is frozen (MRF) unless motion is 'sliding' (the mesh turns: transient, revolution
averages, no frozen-rotor dependence); cavitation {npsh: [m, ...]} gives the NPSH curve and NPSH3 (liquid +
vapour); read guide('pumps') first. Long runs: set solve.procs (MPI ranks). Pending studies report 'progress' with eta_min_max:
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
Vehicles and 3D external aero: read guide('vehicle_aero') first; mesh.mesher "snappy" (snappyHexMesh: surface and edge refinement, refine boxes and
distance bands, prism layers from mesh.layers.y_plus with a coverage report in mesh.layers, half models on a
symmetry side); check mesh.layers.coverage_pct, mesh.snappy.attempts and warnings before trusting forces.
Unsteady flow (vortex shedding, separated wakes): solve.transient {end_time, average_from} with model laminar,
kOmegaSST (URANS) or kOmegaSSTDES (DES); results are time averages (coefficients.cd mean, cd_min/max,
frequency_Hz, strouhal); a run longer than one SLURM job continues in the next by itself.
Pictures: add "images": true (or slices and surfaces) to a study; results.images.<name>.path then
get_image shows each; render_images makes other views of a finished study without solving again.
Units: geometry mm; velocity m/s; pressure Pa; flow m3/s; forces N."""

server = _Server("cfd-translator", instructions=INSTRUCTIONS)


# ---- a stale server, the tool wrapper, the server's log: llmt-runtime's (llmt_runtime.server) since 2026.11.09.6 ----
# The router keeps its servers for hours and each keeps the release it started from, so after `llmt deploy`
# requests ran old code. A stale server with nothing running refuses the next call with STALE_SERVER (the router
# drops it and repeats the call on a new server) and exits; an idle one exits by itself (a look every
# CFDT_STALE_CHECK_S, 10 s; cfdt's own looked every 20 s); a busy one answers with a server_stale note and exits once
# idle. Only a server started by main() (GUARD.serve(): 'serving') may refuse or exit: llmt deploy --test runs the new
# release's tests before switching current, so in-process callers (tests, scripts) see a 'stale' install and must be
# served (omct: an os._exit(0) ended pytest with exit code 0). The names below are thin wrappers over GUARD, kept for
# the tests and for anyone who used them.

def _pool_busy() -> bool:
    """Studies still running in this process's worker pool (they would die with it), or work in the aux pool (meshing
    for get_case, rendering, reprocessing). SLURM jobs do not count."""
    return any(not r.ready() for jobs in api._POOL.jobs.values() for r in jobs.values()) or api.aux_busy()


def _shutdown_workers() -> None:
    """Everything this server started that would outlive it: the study pool, the aux pool (each SIGTERM, then SIGKILL
    after 3 s: llmt_runtime.workers.stop_pool; a worker given SIGTERM ends its OpenFOAM runs), and OpenFOAM runs this
    process started itself (foam.kill_all; none in a server normally). Each runs even when another fails."""
    _W.shutdown(lambda: _W.stop_pool(api._POOL.pool), lambda: _W.stop_pool(api._AUX["pool"]), workers._kill_foam)


# busy and shutdown look the module's names up when called (a test may replace them)
GUARD = Server(RT, api.__file__, busy=lambda: _pool_busy(), shutdown=lambda: _shutdown_workers())
_STATE = GUARD.state
_LOCK = GUARD.lock


def _stale() -> str | None:
    """stale_reason for this process, checked at most every 2 s (GUARD.stale)."""
    return GUARD.stale()


def _may_exit() -> bool:
    """Only a server started by main(). Then CFDT_EXIT_WHEN_STALE: 1 always, 0 never; default: in a shared install
    (GUARD.may_exit)."""
    return GUARD.may_exit()


def _exit_soon(why: str, delay: float) -> None:
    """Exits in `delay` s after stopping the workers (at most 10 s). Caller holds _LOCK (GUARD.exit_soon)."""
    GUARD.exit_soon(why, delay)


def _enter_call() -> str | None:
    """Counts a call in; a STALE_MARK message when the call is refused (GUARD.enter_call)."""
    return GUARD.enter_call()


def _leave_call() -> None:
    GUARD.leave_call()


def _watch_install(period: float | None = None) -> None:
    """Exits this server when it is stale, may exit and is idle (GUARD.watch; period: GUARD.period, 10 s or
    CFDT_STALE_CHECK_S)."""
    if period is not None:
        GUARD.period = float(period)
    GUARD.watch()


# every tool: @GUARD.tool(ToolError): refused when stale and idle; run in a worker thread (run_studies and study_status
# wait up to 50 s; on the event loop they would block other requests and pings) under a 48 s deadline (SLURM commands,
# submissions and waits keep to it); any exception re-raised as a ToolError with its text (MCP SDK 2.x passes on only a
# ToolError's text); the server_stale note on a stale server's answers; calls in the server's log.


@server.tool()
@GUARD.tool(ToolError)
def study_format() -> dict:
    """The study format: geometry, domain, fluids, models, boundary-condition kinds with their
    parameters, face selectors, conventions, an example; and whether OpenFOAM and Gmsh are installed."""
    return api.study_format()


@server.tool()
@GUARD.tool(ToolError)
def validate_studies(studies: dict) -> dict:
    """Static checks of studies given as {name: study}. Returns per study ok, errors, warnings."""
    return api.validate(studies)


@server.tool()
@GUARD.tool(ToolError)
def run_studies(studies: dict, solver: str = "auto", wait_s: float = 45, detail: str = "summary",
                request_name: str | None = None) -> dict:
    """Meshes, solves (OpenFOAM, or the 2D reference solver) and checks each study. Per study: results
    (flow rates, pressures, pressure drop, wall forces, coefficients, probes, wall shear, y+, loads for
    fea-translator), what each bc's selector took, solver convergence, mass and momentum balance,
    'expect' checks, warnings, and files (the OpenFOAM case and a .tgz of it). Returns after at most
    wait_s seconds: studies still running are listed in 'pending'; collect them with study_status.
    A study with "mesh_scales": [1, 0.707, 0.5] or "variants": {suffix: {changes}} runs as several
    (<name>_L1.., <name>_<suffix>); 'variants' in the result compares them (mesh convergence). geometry
    {"ref": name} uses a geometry registered with add_geometry.
    detail: "summary" (default: bc selections as counts, no per-face loads), "brief" (verdict, main
    numbers, mesh, timing) or "full". request_name ('run_<id>', optional): the request folder's name, so a caller
    knows the request path before the answer; repeating it returns that request's status (no second submission)."""
    return api.run(studies, solver, wait=min(max(float(wait_s), 0.0), 50.0), detail=detail, request_name=request_name)


@server.tool()
@GUARD.tool(ToolError)
def study_status(request: str, wait_s: float = 45, detail: str = "summary", window: list | None = None,
                 reference: str | None = None) -> dict:
    """The studies of a run_studies request (its 'request' path): finished ones with their results,
    the others in 'pending'. Waits up to wait_s seconds (at most 50) for them to finish. detail: "summary"
    (default: coefficients, forces, checks, image paths; bc selections as counts, no per-face loads),
    "brief" (verdict, warnings, main numbers, mesh and layers, timing: for many studies) or "full"
    (everything, as in <request>/<name>/<name>.json). Variants and mesh_scales are compared in 'variants'.
    window [t0, t1] (s; t1 null = to the end): instead, for finished transient studies, the mean, std, standard error
    and drift of cd, cl, the part's force and each wall bc's force over that window, from the force histories (no
    download of the full result). reference: a study of the request; 'comparison' then sets every other study
    against it (head and power differences with their standard errors, significant beyond 2 sigma), as variants do."""
    return api.status(request, min(max(float(wait_s), 0.0), 50.0), detail, window, reference)


@server.tool()
@GUARD.tool(ToolError)
def guide(topic: str = "") -> dict:
    """How to set up a kind of study, from the runs that validated it (longer than study_format holds):
    'vehicle_aero' (cars, bluff bodies: domain, snappy sizes, layers, steady RANS vs DES, averaging windows,
    reporting, which Ahmed benchmark to run); 'pumps' (MRF or sliding mesh, the rotating zone, time steps and
    revolution averages, cavitation and NPSH sweeps). Without a topic: the list."""
    return api.guide(topic or None)


@server.tool()
@GUARD.tool(ToolError)
def add_geometry(name: str, goal: dict | str | None = None, step: str | None = None, step_base64: str | None = None,
                 stl: str | None = None, stl_base64: str | None = None, scale: float = 1.0,
                 groups: dict | list | str | None = None) -> dict:
    """Registers a geometry on this machine once, so studies name it instead of carrying it: a
    build123d-translator goal (a dict, or the path of a goal JSON file; checked by building it: volume, area,
    bounding box), a STEP file (step: path, or step_base64), or a closed triangle mesh (stl: path of an STL, OBJ or
    PLY file, or stl_base64 for an STL; checked watertight; named solids / OBJ groups become face groups that bcs
    select with {"group": name}; scale: factor to mm, 1000 for metres; groups: {name: [triangle indices]}, one name
    per triangle, or a JSON file of either). Studies then use "geometry": {"ref": name}."""
    return api.add_geometry(name, goal, step, step_base64, stl, stl_base64, scale, groups)


@server.tool()
@GUARD.tool(ToolError)
def list_geometry() -> dict:
    """The registered geometries (name, kind, size, bounding box, when added)."""
    return api.list_geometry()


@server.tool()
@GUARD.tool(ToolError)
def get_case(study: dict, name: str = "study", wait_s: float = 45) -> dict:
    """Meshes one study and writes its OpenFOAM case without solving: mesh quality, what each bc took,
    inlet values, and the case dictionaries as text. Meshing runs in a worker: when it takes longer than wait_s
    (at most 50) the answer is state 'running'; call again with the same arguments to collect it."""
    return api.in_worker("meshing", "case", {"study": study, "name": name}, wait_s)


@server.tool()
@GUARD.tool(ToolError)
def run_benchmarks(names: list[str] | None = None, solver: str = "auto", wait_s: float = 30) -> dict:
    """Runs the benchmark studies with known answers (Couette, plane Poiseuille, lid-driven cavity vs
    Ghia, Hagen-Poiseuille pipe, Schaefer-Turek cylinder; slow ones by name: cavity_re1000,
    pipe_turbulent; chain_plate with build123d-translator and fea-translator) and reports pass or fail.
    With SLURM the cases are submitted as jobs and this returns after at most wait_s seconds: collect the
    rest with bench_status(bench)."""
    return api.bench(names, solver, wait_s=float(wait_s))


@server.tool()
@GUARD.tool(ToolError)
def bench_status(bench: str, wait_s: float = 45) -> dict:
    """Pass/fail of benchmarks submitted by run_benchmarks (its 'bench' folder): finished cases with their
    checks, the others pending (SLURM jobs that keep running without this conversation)."""
    return api.bench_status(bench, min(float(wait_s), 50.0))


@server.tool()
@GUARD.tool(ToolError)
def jobs(limit: int = 20, active: bool = False) -> dict:
    """Recent requests with SLURM jobs on this machine (from any conversation): each study's state
    (queued, running, done, timed_out, out_of_memory, cancelled, failed), job id and ETA; the queue of
    this translator's jobs and the machine's CPUs. active=True: only requests still queued or running."""
    return api.jobs(int(limit), bool(active))


@server.tool()
@GUARD.tool(ToolError)
def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops a request's studies (or the named ones): scancel for SLURM jobs, a cancel flag for local
    ones. Returns their states."""
    return api.cancel(request, names)


@server.tool()
@GUARD.tool(ToolError)
def get_image(path: str) -> Image:
    """A PNG written by run_studies (results' images.<name>.path) or render_images, as an image you can
    see and show: slices of velocity or pressure (with streamlines) and wall surfaces."""
    return Image(path=str(api.image_path(path)))


@server.tool()
@GUARD.tool(ToolError)
def render_images(study_folder: str, images: list | bool = True, wait_s: float = 45) -> dict:
    """New pictures of a finished study without solving again: study_folder is <request>/<study name>
    (the folder holding <name>.json). images: true (defaults) or [{name, slice: {axis, at}, field,
    streamlines, window, clim} | {name, surface, field, view, clim, quality: high} | {name, iso: {field: q | u |
    cp_total, value | auto}, color, window, view}] as in study_format (field cp: pressure coefficient from
    reference.velocity). quality high and iso draw with VTK (lit 3D, half models mirrored); Q isosurfaces show
    vortices (DES wakes). Then get_image each path."""
    return api.in_worker("rendering", "render_images", {"study_folder": study_folder, "images": images}, wait_s)


@server.tool()
@GUARD.tool(ToolError)
def reprocess(request: str, studies: list[str] | None = None, onset: dict | bool | None = None,
              average: dict | bool | None = None, wait_s: float = 45) -> dict:
    """Results again for a request's finished studies (all, or those named) from their cases, with this version's
    post-processing, without solving: e.g. porous-zone reactions in the part's forces (2026.10.28.2) for runs
    made before. onset: a new onset block for those studies (corrected regions, thresholds, c; false removes it):
    the cavitation-onset screen again without solving. average: other averaging settings for steady runs
    (solve.average, e.g. {"window": 0.5}): window means, standard errors and the variants' differences again from the
    saved monitor histories. Keeps the old result as a .bak and reports what changed; then study_status(request)."""
    return api.in_worker("reprocessing", "reprocess", {"request": request, "studies": studies, "onset": onset,
                                                       "average": average}, wait_s)


@server.tool()
@GUARD.tool(ToolError)
def import_results(path: str, name: str | None = None) -> dict:
    """A request run on a cluster this server cannot reach (NERSC Perlmutter, MIT Engaging: cfdt runs there from its
    command line, other/supercomputers/clusters.md), brought back: path is the .tgz made there with `cfdt pack REQUEST` (a local
    file: the router copies it to the machine), or the request folder copied back. It becomes a request of this
    machine: study_status (results, reliable, averages, revolution averages, drift), window averages, variants and
    render_images (when packed with --state) work on it as on any other."""
    return api.import_results(path, name)


@GUARD.tool(ToolError)
def get_file(path: str, max_mb: float = 20) -> dict:
    """A file written by run_studies (a path from its result) as base64."""
    return api.read_file(path, max_mb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, help="serve MCP over streamable HTTP on this port (default: stdio)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    # only now may a stale server refuse calls and exit; the install watch, the server's log ($CFDT_OUT/.server) and
    # the signal handlers (SIGTERM, SIGHUP, SIGINT: the worker pools stop first) start here
    GUARD.serve()
    if a.http is None:
        server.run()
    elif _V2:
        server.run(transport="streamable-http", host=a.host, port=a.http)
    else:
        server.settings.host, server.settings.port = a.host, a.http
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
