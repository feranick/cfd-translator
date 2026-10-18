# cfd-translator: context for a new session

Session context for Claude (the project documentation is README.md). Start a chat with: "Read planning.md and start."

## What this is

A deterministic **translator + checker** for LLMs: the LLM writes a structured spec (the flexible part), the translator computes everything else (no model, no guessing), an independent checker confirms the result, and errors go back to the LLM as fixable messages. Same design as:

- **build123d-translator** (`b3dt`; `~/Desktop/build123d-translator`; github.com/feranick/build123d-translator): goal (ordered CAD features) → build123d solid, STEP, standalone script. Checker: IoU vs FreeCAD (6,761/6,762 DeepCAD goals), exact volumes.
- **fea-translator** (`feat`; `~/Desktop/fea-translator`; github.com/feranick/fea-translator): study (part, material, analysis, BCs on faces) → Gmsh mesh, CalculiX deck, results. Checker: 11 closed-form benchmarks, equilibrium/heat balance every run, `expect`, convergence; reference solver checks decks without CalculiX.

**This tool:**

- Name / CLI / MCP: `cfd-translator` / `cfdt` / `cfdt-mcp`; repo github.com/feranick/cfd-translator
- Domain: steady incompressible CFD; target: **OpenFOAM .org 14** (`foamRun`, `solver incompressibleFluid`; works from 11), meshing with Gmsh or our own structured boxes; the polyMesh is written by our code (no `gmshToFoam`, no `blockMesh`); runs on carbonio.
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

- `format.py`: study format (`study_format`) and static `validate()` with precise messages.
- `polymesh.py`: cells -> OpenFOAM polyMesh (faces, owner, neighbour, patches; OpenFOAM's geometry formulas); structured box; extrusion of 2D meshes.
- `mesher.py`: Gmsh (OCC geometry, external cut, tets or swept 2D mesh with boundary layers); `select.py`: face groups, selectors, bc -> patch assignment.
- `case.py`: the OpenFOAM case (all values computed here); `foam.py`: the only module running OpenFOAM; `foamio.py`: tolerant readers (fields, logs, checkMesh, polyMesh).
- `post.py`: results and invariants from the final fields; `refsolver.py`: 2D MAC solver that reads the written case.
- `runner.py`, `bench.py`, `api.py`, `cli.py` (`status|format|validate|run|case|bench`), `mcp_server.py` (`study_format`, `validate_studies`, `run_studies`, `study_status`, `get_case`, `run_benchmarks`, `get_file`).

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

1. Read README.md (format, checker, unverified assumptions) and the State below.
2. Ask me for the carbonio output of `pytest -q` and `cfdt bench` (and the slow benchmarks) if I have not pasted it; fix from it, one assumption at a time.
3. Keep README (versions table, checker results) and State up to date; give carbonio commands ready to paste.
4. End with an end-to-end test prompt using cfdt with b3dt and feat.

## State

**2026.10.18.7/.8**: fixes from the mochi energy rerun (.8: GAMG cap, isothermal h, per-axis angular balance, benchmark fixes; .7 was pinned without them) (request run_20261018_182109_c20fd9, 2026.10.18.5).
- conduction_slab_2d: p_rgh GAMG diverged at iteration 3 (final residual 6.5e43), then T inversion failed. Closed + no gravity now uses rhoConst; the same slab with water (rhoConst, 18.5 code, run locally on mochi) gave Q = 0.04794 W (k dT/H: 0.0479) and mid T 39.987 C in 6 s.
- venturi_subsonic: outline not closed (b3dt BuildError); cfdt validation now runs b3dt's validate_goal.
- heated_channel_2d: Nu 8.36 (ref 8.235), outlet T 23.826 (ref 23.833), but not converged (Uy) and a false inlet-flow warning (mass vs volume): both fixed.
- natural_cavity_ra1e4: Nu 2.253 (ref 2.243), p_rgh 1.18e-6 vs 1e-6 at 6000 iterations: target now 1e-5.
- Rotation benchmarks: Taylor-Couette torque 6.74e-7 (fixed wall) and 6.75e-7 N m (MRF) against 6.70e-7 analytic (mu 0.01, omega 1): +0.6 %; swirl_pipe reliable.
- pipe_air_low_mach: dp over 0.1 m 0.2897 Pa against 0.2920 (-0.8 %), but 78 min: h stalled (isothermal) and GAMG ran 1000 sweeps per iteration on absolute p. Fixed by energy.isothermal (h out of residualControl) and maxIter 50.
- taylor_couette_2d passes all checks (u_t at 30 mm -0.2 %). taylor_couette_mrf: torques +0.7/+1.0 %, power ok, but u_t at the probe +7.8 %: the probe sat on the zone's jagged 30 mm edge; zone moved to 25 mm.
- swirl_pipe: angular momentum in -0.65 %; its flow check compared with pi R^2 (polygon area is 0.16 % less): now the meshed area. Its angular balance (0.22) and pipe_air's (1.09) were the scale: now per axis against gross face moments.

**2026.10.18.6**: the run record (`execution` in every result), as added to EXECUTION.md together with lmpt's `slurm_job`/`mpi` fields.
- The former `executor` block of run-study results is kept as an alias; drop it once llmt and lmpt read `execution`.
- 47 passed in the sandbox.

**2026.10.18.5**: on mochi, conduction_slab_2d diverged at iteration 4 (p GAMG 4.9e65, then SIGFPE in MachNo): closed energy domains now use the p_rgh/heRhoThermo closed-cavity formulation with g = 0. Queueing confirmed on mochi: 15-rank jobs take 30 CPUs (whole cores) and the rest wait (Resources, then Priority, with ahead). Also: Couette's 'not converged' came from p (zero everywhere, residual 0.05), not only Uy. `zero_fields` now covers U components, p/p_rgh and h.

**2026.10.18.4**: mochi with SLURM works.
- `llmt test cfdt`: 47 passed.
- couette_2d and pipe_poiseuille_mpi pass. pipe_poiseuille_mpi ran on 4 ranks in a job: 17 s solve, against 41 s serial on carbonio.
- The zero-component convergence rule now uses 1e-6.
- Still to test on mochi:
  - jobs queueing when they ask for more cores than the machine has;
  - disconnecting Claude mid-run and collecting the result;
  - the energy and rotation benchmarks.

**2026.10.18.3** (mochi, first SLURM run): the jobs were submitted and collected, but OpenFOAM's `mpirun -np 9` failed with OpenMPI 5 'allocation-overload'.
- Cause: with hyperthreads, `--ntasks=9` gets fewer physical cores than ranks.
- Fix: `--ntasks-per-core=1` in job.sh, the `--map-by hwthread` fallback, and ranks capped by cells (`CFDT_CELLS_PER_RANK` 5000).
- A test assumed CFDT_PROCS=1; the fixture now clears the settings.

**2026.10.18.2**: SLURM execution per llm-translators `docs/EXECUTION.md`.
- `executor.py`: submit, `run-study`, states from squeue (one call per request), sacct or job.log; `jobs`, `cancel`.
- `api._collect` judges SLURM studies only from files and the queue, so any process sees them. Local pool studies are unchanged.
- OpenFOAM now runs in its own process group (killed whole on timeout or cancel); the cancel flag works in local mode.
- Sandbox: 42 passed, including 16 against `tests/fake_slurm.py`. A result was collected by a new process; queued (reason, ahead), running, cancel, every finished state from sacct and from job.log, sbatch failure, routing, run-study by hand, process-group kill, bench through SLURM.
- Next on mochi with `CFDT_EXECUTOR=slurm`:
  - couette_2d;
  - two servers at once (they should queue);
  - disconnecting mid-run and collecting the result.
- lmpt and feat follow the same contract.

**2026.10.18.1**: energy and compressible flow.
- `energy` (OpenFOAM `fluid`): air as a perfect gas (hePsiThermo, or heRhoThermo with gravity), water as rhoConst.
- `gravity` gives buoyancy through `p_rgh` + `g` + `pRef`.
- Walls: `T`/`q`/`power`/`h` (externalTemperature). Inlets: temperature, mass_flow, total_pressure/total_temperature.
- Results: heat, mdot, Mach number, energy balance, `feat_thermal`; images of T, Mach number and q.
- The OpenFOAM 14 keywords were researched in the sources by an agent: externalTemperature, prghPressure, totalPressure/totalTemperature, wallHeatFlux sign, phi mass flux, buoyancy switched on by `0/p_rgh`, default Prt 1 (we write 0.85).
- Sandbox: 26 passed. The energy post-processing was verified on the exact conduction solution written in OpenFOAM's format.
- Next on carbonio: `cfdt bench conduction_slab_2d heated_channel_2d natural_cavity_ra1e4 venturi_subsonic pipe_air_low_mach chain_thermal`.
- Open: supersonic (shockFluid), conjugate heat transfer (solid regions), radiation, energy with MRF, transient.

**2026.10.17.1**: images.
- `render.py` (matplotlib/Agg): slices with streamlines and masked solids; wall surfaces.
- `state.npz` + `state.json` per study for re-rendering.
- MCP `get_image` (an MCP Image) and `render_images`; `cfdt images`.
- Checked here: cavity (vortex at Ghia's position), channel, potential flow around a cylinder (hole masked, 2U at the sides), 3D box surfaces. 22 tests pass here.
- Not yet seen: images of real OpenFOAM 3D tetrahedral results (slab selection at about 0.6 cell; check the look on carbonio).
- ParaView (pvbatch) rendering is an open idea for nicer 3D (streamtubes, iso-surfaces).

**2026.10.16.6** (sandbox: 20 passed, 3 skipped; new OpenFOAM benchmarks not run yet). On OpenFOAM 14 (carbonio), every benchmark up to .4 passes:
- couette, channel, cavity_re100/1000, pipe_poiseuille, cylinder_2d1 (cD 5.581, Δp 0.11733; cL 0.0078 is the weakest), pipe_turbulent (3.1 %, but its residuals stall at 2e-4), chain_plate (with feat >= .6).
- Invariants: mass <= 4e-10, momentum <= 4e-4.

.6 came from a pump study in another chat:
- It reported unconverged numbers (momentum off 5-9 %) as ±15 %.
- It fed a swirl-free uniform inlet at the impeller exit.
- It ran serially for 90 min and polled every 50 s.

So .6 adds: the `reliable` verdict, `head_loss_m`, `swirl` inlets, rotating walls, rotating zones (MRF: cellZones + MRFProperties, torque/power/pump head and efficiency, frame source in the balances), the annulus primitive, `solve.procs`, progress/ETA for pending studies, and MCP instructions to stop polling long runs.

Next: on carbonio run the new benchmarks (`cfdt bench taylor_couette_2d taylor_couette_mrf swirl_pipe pipe_poiseuille_mpi`). Unverified for .6 (OpenFOAM 14):
- MRF (checked against the v14 MRFZone.C: no `nonRotatingPatches`; rotating walls need `MRFnoSlip`, written with its value): `omega` in rad/s as a plain number; whether MRFnoSlip writes a value (the reader falls back to ω × r);
- the cellZones file format;
- decomposeParDict (`decomposer scotch`, from the v14 motorBike tutorial's form);
- that phi stays absolute with MRF (the balances assume it).

Open ideas:
- pipe_turbulent residual stall; cylinder cL (refine);
- `converge: true`; 3D prism layers; cyclic patches;
- heat transfer/CHT with feat; transient (PIMPLE, sliding mesh for impellers); snappyHexMesh; porous zones;
- a pump example (volute + impeller goal) once MRF is confirmed.
