# cfd-translator: context for a new session

Copy to `~/Desktop/cfd-translator/CLAUDE.md`, connect the folder, and start the chat with: "Read CLAUDE.md and start."

## What this is

A deterministic **translator + checker** for LLMs: the LLM writes a structured spec (the flexible part), the translator computes everything else (no model, no guessing), an independent checker confirms the result, and errors go back to the LLM as fixable messages. Same design as:

- **build123d-translator** (`b3dt`; `~/Desktop/build123d-translator`; github.com/feranick/build123d-translator): goal (ordered CAD features) → build123d solid, STEP, standalone script. Checker: IoU vs FreeCAD (6,761/6,762 DeepCAD goals), exact volumes.
- **fea-translator** (`feat`; `~/Desktop/fea-translator`; github.com/feranick/fea-translator): study (part, material, analysis, BCs on faces) → Gmsh mesh, CalculiX deck, results. Checker: 11 closed-form benchmarks, equilibrium/heat balance every run, `expect`, convergence; reference solver checks decks without CalculiX.

**This tool:**

- Name / CLI / MCP: `cfd-translator` / `cfdt` / `cfdt-mcp`; repo github.com/feranick/cfd-translator
- Domain: steady incompressible CFD; target: **OpenFOAM .org** (current apt release; check whether `foamRun -solver incompressibleFluid` replaces `simpleFoam`), meshing with Gmsh (`gmshToFoam`) and `blockMesh`; runs on carbonio.
- Typical requests:
  - "Pressure drop through this manifold (b3dt goal) at 2 L/min water."
  - "Drag and lift on this bracket in 10 m/s air."
  - "Velocity profile and flow split between the two outlets of this tee."
  - "Wall pressures on this plate in 30 m/s air, then stresses with feat."
- Spec contents (first idea; challenge it):
  - `geometry`: `{"goal": <b3dt goal>}` or `{"step": path}`. Set `domain: internal` (the part is the fluid) or `external` (a box around the part, with sizes in body lengths, minus the part).
  - `fluid`: preset (`air`, `water`) or `rho` [kg/m3], `mu` [Pa·s].
  - `analysis`: `steady`; `model: laminar | kOmegaSST`; iterations; residual target.
  - `bcs`: `inlet` (`velocity` [m/s] or `flow_rate` [m3/s or L/min], turbulence intensity), `outlet` (`pressure` [Pa]), `wall` (no-slip), `slip`, `symmetry`. Faces use feat selectors.
  - `mesh`: `size`, `boundary_layers` (n, first height, growth), `refine` near selected faces; `blockMesh` for box/channel cases.
  - `outputs`: Δp between patches, flow per patch, forces/moments and Cd/Cl (reference area/velocity), max velocity, probes (points, lines), wall shear, y⁺.
  - `expect`: as in feat (dotted result paths, value, tolerance).
  - Units: mm for geometry (converted to m for OpenFOAM), m/s, Pa, N, kg/m3.
- Independent checks:
  - Every run: mass balance (Σ patch flux ≈ 0), control-volume momentum balance (wall forces vs. inlet/outlet momentum flux + pressure), residual drop, `checkMesh` quality, y⁺ and convergence warnings.
  - Closed-form benchmarks: Hagen–Poiseuille pipe (Δp, centreline velocity), plane Poiseuille, Couette, developing-flow entrance length (loose).
  - Reference data: lid-driven cavity vs. Ghia et al. 1982 (Re 100/400/1000), Schäfer–Turek cylinder 2D-1 (Cd ≈ 5.58, Cl ≈ 0.0106, Δp ≈ 0.1175), creeping-flow sphere (Stokes drag, with wall-effect correction), turbulent pipe friction factor vs. Blasius/Colebrook (loose, kOmegaSST).
  - Reference implementation: a small numpy finite-volume solver (2D channel/cavity, laminar) to check case translation and results without OpenFOAM.
- First version scope: steady incompressible, laminar + kOmegaSST, internal and external flow, Gmsh + blockMesh meshes, Δp/flow/force/probe outputs, the benchmarks above, b3dt goals inline, wall pressures exported to feat.

## Interconnection

Works **together** with b3dt and feat:

- Accept b3dt specs inline, as feat accepts `{"part": {"goal": <b3dt goal>}}` or `{"step": path}`. Import b3dt optionally (warning in validation, tests skipped when missing).
- **CFD → feat:** export wall pressures (and optionally shear) on the part's faces as feat surface loads (`pressure` per face set, or mapped nodal loads). Same geometry, same selectors. Chain benchmark: b3dt plate goal → cfdt flat plate in uniform flow → feat static; check that feat's reaction equals cfdt's force.
- Same conventions: mm, N, MPa, °C; Z up, base on z = 0 centred in X/Y; face selectors `nx/ny/nz` (+ `at`), `pts`, `cyl_r`, `all`, lists = union; JSON in/out; dotted result paths.
- Result fields another tool reads are an interface: keep them stable and list them in the README.
- One environment on carbonio: shared venv `~/Software/build123d-translator/.venv` (alias `cad`), `pip install -e ~/Software/cfd-translator[mcp,test]`.
- At least one chain benchmark (b3dt goal → cfdt → feat), skipped when a piece is missing.
- MCP servers may sit on different machines (b3dt on the Mac, feat and cfdt on carbonio): pass specs between tools, not local paths; `get_file` returns files as base64.

## Conventions

- Versions `YYYY.MM.DD.x` in every module docstring, `__init__.py`, `pyproject.toml`, and the README's version table; update the README with every change.
- Author: Nicola Ferralis <ferralis@generalflash.com>. I commit and push from the Mac; carbonio clones in `~/Software/`.
- Keep answers concise; give commands ready to paste.
- Python ≥ 3.10; `pyproject.toml` extras `[mcp, test]` (+ `gmsh`); entry points `cfdt`, `cfdt-mcp`.

## Architecture

- `format.py`: spec format (kinds, parameters, units, conventions, example) as JSON; static `validate()` with precise messages.
- Core translation (`case.py`): spec → OpenFOAM case dictionaries (`controlDict`, `fvSchemes`, `fvSolution`, `0/U`, `0/p`, `0/k`, `0/omega`, `0/nut`, `momentumTransport`, `physicalProperties`, function objects). Pure Python, testable without OpenFOAM; patch names come from our own face selection, never from OpenFOAM numbering.
- `mesh.py`: Gmsh volume mesh with physical groups per selected face set (internal/external domain, boundary layers) and `blockMesh` for box cases.
- Backend (`foam.py`): the only module calling OpenFOAM (`gmshToFoam`, `blockMesh`, `checkMesh`, solver, `postProcess`), as subprocesses with timeouts and logs.
- `results.py`: tolerant readers of `postProcessing/` function-object output (forces, surfaceFieldValue, probes, residuals) and logs, with test fixtures.
- `runner.py`: one study end to end, with checks (`expect`, mass/momentum balance, residuals, warnings).
- `bench.py`: cases with known answers; `cfdt bench`.
- `refsolver.py`: small independent laminar FV solver (2D).
- `api.py` (JSON, spawned workers with time limits, `CFDT_OUT` folder, long runs as start + status), `cli.py` (`format|validate|run|bench|status`), `mcp_server.py` (`study_format`, `validate_studies`, `run_studies`, `start_run`, `run_status`, `run_benchmarks`, `get_file`; MCP SDK 1.x/2.x; stdio and `--http`).
- `tests/` (skip what is not installed), `examples/`, `README.md` (install, use, format, checker results, versions).

## Lessons from b3dt and feat

1. Research the target program first (manual, output formats), then the spec, then code. List unverified assumptions in the README, each with a benchmark that would catch it (feat: CalculiX's RF includes applied loads; caught by `gravity_block`). Likely CFD examples: kinematic vs. static pressure (OpenFOAM's incompressible `p` is p/ρ), the sign of the `forces` output, units after `gmshToFoam` (mm → m scaling), and flux sign conventions.
2. Never rely on the program's internal numbering or ordering; recompute from geometry or data.
3. Exact checks where possible (1e-6), loose only where physics says so, with the reason. For CFD: laminar closed-form ≤ 1–2 % on converged meshes; turbulent correlations ±10 %.
4. Every run checks itself (conservation, validity), not only the benchmarks.
5. Validate before running; an LLM must fix every error from its message alone.
6. Compute derived quantities the way the target does (feat's flat-triangle loads were 0.1 % off on curved faces until isoparametric). For CFD: integrate forces/fluxes on the same faces and the same way as OpenFOAM's function objects.
7. Keep `study_format` compact (local LLMs need ≥ 16k context for it).
8. If the sandbox cannot run the target (ARM Linux, no apt/conda/Homebrew), say so up front, test the rest, give exact carbonio commands, iterate on my output.

## Environment

- Mac (Apple Silicon): development, Claude desktop, LM Studio. No Homebrew; MacPorts gfortran fails for Fortran builds.
- carbonio (Ubuntu, Python 3.14): heavy programs (CalculiX, Gmsh, OpenFOAM .org via apt), shared venv `cad`. OpenFOAM needs its `etc/bashrc` sourced; the backend must do this in each subprocess.
- MCP servers on carbonio, in `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
"cfdt": {"command": "/usr/bin/ssh",
         "args": ["-T", "-o", "BatchMode=yes", "nicola@carbonio",
                  "/home/nicola/Software/build123d-translator/.venv/bin/cfdt-mcp"]}
```

## How to start

1. Design questions are already answered: OpenFOAM .org; steady incompressible (laminar + kOmegaSST); Gmsh + blockMesh; b3dt goals inline and CFD → feat loads. Ask only what is still open.
2. Research OpenFOAM .org (current release: solver names, dictionary keywords, function objects, `gmshToFoam` patch handling). Summarise the proposed spec format and benchmark list before coding.
3. Build, test what can run here, write the README; give carbonio commands (`sudo apt install openfoamNN`, `git pull`, `pip install -e`, `pytest -q`, `cfdt bench`).
4. Finish with an end-to-end test prompt using cfdt with b3dt and feat.

## State

`<filled in by Claude at the end of each session: version, what passes, open ideas>`
