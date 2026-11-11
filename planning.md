# cfd-translator: context for a new session

Session context for Claude (the project documentation is README.md: format, results, checks, benchmarks and a
version table with the reason for every change). Start a chat with: "Read planning.md and start."

## What this is

A deterministic **translator + checker** for LLMs: the LLM writes a structured study (the flexible part), the
translator computes everything else (no model, no guessing), checks confirm the result, and errors go back to the
LLM as fixable messages. Same design as the other translators bundled by **llm-translators** (llmt):
build123d-translator (b3dt, CAD), fea-translator (feat, CalculiX FEA), lammps-translator (lmpt),
openmodelica-translator (omct), print3d-translator (p3dt).

**This tool** (repo github.com/feranick/cfd-translator; CLI `cfdt`, MCP `cfdt-mcp`; local checkout
`~/Software/cfd-translator`):

- Target: **OpenFOAM .org 14** (`foamRun`; modules `incompressibleFluid`, `fluid`, `incompressibleVoF`).
- Meshes: our own polyMesh writer (structured boxes, Gmsh tets and swept 2D meshes with layers) or
  **snappyHexMesh** (`mesh.mesher: "snappy"`: Gmsh triangulates the STEP/goal surface, bcs chosen on it).
- Physics now: steady incompressible (laminar, kOmegaSST), **transient** (URANS, kOmegaSSTDES, laminar; averages,
  continues across SLURM jobs), **energy** (perfect-gas air, water; buoyancy), **tracers and mean age**,
  **porous zones**, **rotating zones** (MRF frozen rotor; **sliding mesh** with OpenFOAM's non-conformal coupling),
  **cavitation** (water + vapour VoF, MRF, pseudo-steady, NPSH sweeps with NPSH3).
- Results: flows, pressures, head loss, forces/coefficients, torque/power/pump head and efficiency, probes,
  y+, images (Matplotlib, optional VTK), `feat_loads` / `feat_thermal` for fea-translator; every run judged
  (`reliable` + `unreliable_because`).
- Pump work in other chats drives most recent changes: **S180** (large, 1450 rpm, snappy 6-13 M cells) and **S50**
  (self-priming, 2900 rpm, STEP `fluid_s50_df.step`); also vehicle aero (Ahmed body, Nissan Rogue DES on mochi).

## Interconnection

- Geometry from b3dt goals inline or STEP files; `add_geometry` / `geometry.ref` register one once per machine.
- CFD → feat: wall pressures per CAD face as feat loads (`feat_loads`), wall heat as feat thermal bcs.
- Same conventions as feat/b3dt: mm, N, Pa (results), m/s, m3/s; Z up; face selectors `nx/ny/nz` (+ `at`), `pts`,
  `cyl_r`, `all`; JSON in/out; dotted result paths. Result fields other tools read are an interface (README).
- Execution follows llm-translators' docs/EXECUTION.md and SHARED.md (SLURM jobs, run record `execution`,
  estimates in validation, stale servers after `llmt deploy`, `run: {timeout_s, mem, procs, queue, host}`).
- MCP: Claude reaches cfdt through the **llmt router**, which places work on carbonio, mochi, spark1 or spark2
  (`machines()`, `run.host`) and stages local files (`staged`, `~/.llmt/uploads/<sha>/`).
- **Do not edit, commit or push llm-translators** (or its submodules): the user syncs all tools and updates llmt
  at the end. Changes for other tools go to the user as prompts or doc lines.

## llmt-runtime (since 2026.11.09; the user's TRANSLATOR_CHATS.md)

The code the translators used to copy is **llmt-runtime** (`~/Software/llmt-runtime`, package `llmt_runtime`, a
submodule of llm-translators; read its README.md and PLANNING.md): workers (parent watch, solver runs, SIGTERM,
stopping pools), memory and out of memory (cgroups, job vs node, wording), the SLURM layer (settings `CFDT_EXECUTOR`,
`_SLURM_ARGS`, `_SLURM_NODE`, `_SLURM_MEM`, `_SLURM_TIME`, caps; nodes, pinning, cores, node memory; job.sh / job.json,
submit with dependencies and late-sbatch adoption, job states, `queue_summary`, `list_jobs`, cancel, extend), requests
(folders, `request_name`, `request_meta`, `ran_with`, `read_file`), the MCP server (`server.Server`: stale installs,
the tool wrapper with the 48 s deadline, server log in `$CFDT_OUT/.server`) and the fake SLURM for tests
(`llmt_runtime.testing.fake_slurm`, `FAKE_*`).

- cfdt's side: `runtime.py` makes `RT = Runtime(prefix="CFDT", ...)` with cfdt's sizing as hooks (`study_procs` =
  solve.procs, `procs` = procs_for_job with the cells cap, `mem_auto` 2 GB + 4 GB per million cells, `job_extra`
  cells_estimate, `running_progress`, `oom_advice`, `timeout_hint`; `one_node`, `min_minutes=0`,
  `slurm_time_default="timeout"`). `executor.py` keeps its public names as thin wrappers over `RT.slurm.*` /
  `RT.jobs.*` (keep them: callers and tests use them). `mcp_server.py`: `GUARD = Server(RT, ...)`, every tool
  `@GUARD.tool(ToolError)`, `main()` -> `GUARD.serve()`. `workers.py` wraps the runtime's with `foam.kill_all` as the
  orphan cleanup.
- Stays cfdt's: `eta_start` / `running_etas`, `pending_ahead` (per node), transient segments (`RT.jobs.submit(...,
  after=[ids])`), `queue_summary`'s extras (`cluster_queue`, `starts_now`, `pending_for_node`), `cores_shared`
  (cpushare), failures written as the study's result; and all the CFD (sizing, meshing, solver, progress, results,
  rendering, benchmarks, advice texts).
- **Shared behaviour is fixed once, in llmt-runtime** (SLURM, job states, memory/OOM, workers, requests, the MCP
  server's common code), never by patching cfdt's wrappers. This chat does not edit llmt-runtime: such fixes go to the
  user as a prompt for the runtime's chat (additions only, backward compatible; the runtime bumps its version, adds a
  test and tags the release).
- `pyproject.toml` declares a floor only, `llmt-runtime>=2026.11.11.2` (also requirements-standalone.txt's tag and
  the README install line); raise it only when cfdt needs something newer. Order: runtime pushed, then cfdt, then the
  user pins (`./llmt pin runtime cfdt`).
- Tests use the runtime's fake SLURM; patch the runtime's names where its code runs (`llmt_runtime.slurm.*`), not
  cfdt's wrappers; a helper taking a folder plus `**kwargs` must not name the folder `state`; MCP answer tests pin
  `GUARD.stale` to None.
- Open (cfdt): two tests fail only on macOS (tool errors: SDK version; the Ahmed snappy benchmark); the bench test
  depends on timing; `extend` (runtime PLANNING.md, section 2). 2026.11.09.7: cfdt's extras (pending_ahead,
  running_etas, queue_summary) share one queue read (`queue_rows`); the runtime's `pending_jobs` call for its `ahead`
  remains: one call less only with a runtime option (`ahead_of` with a 'can share a node' test, also for lmpt/omct). An explicit
  `CFDT_SLURM_MEM` size is no longer cut to `CFDT_MAX_MEM`.

## Conventions

- Versions `YYYY.MM.DD.x` (x increments within a day) in every changed module's docstring, `__init__.py`,
  `pyproject.toml`, and a README "Versions" row saying what happened and why (the run that showed it).
- Author: Nicola Ferralis <ferralis@generalflash.com>. The user commits from the Mac; give only the cfdt command:
  `cd ~/Software/cfd-translator && rm -f .git/index.lock && git add -A && git commit -m "..." && git push`
- Tests in the sandbox: copy llm-translators/llmt-runtime to /tmp/rt (the sandbox disk is too full for pip), then
  `PYTHONPATH=/tmp/rt CFDT_OUT=/tmp/h/out TMPDIR=/tmp/h python3 -m pytest -q -p no:cacheprovider` (no OpenFOAM or
  Gmsh here: stand-ins and synthetic fields; say what is unverified on OpenFOAM). 247 passed, 5 skipped (2026.11.09.6).
  On servers the user runs `$R/llmt test cfdt` (or `runtime` too when the runtime changed) in tmux.
- `study_format` without its machine-dependent `status` must stay under 12,300 characters (test; local models): it
  is at 12,054 (2026.11.09.1). New keywords get one short line there; the how-to goes into `guides.py`
  (`guide('vehicle_aero')`, `guide('pumps')`).
- Before writing any OpenFOAM keyword, read it in the v14 source or tutorials (github.com/OpenFOAM/OpenFOAM-14,
  cpp.openfoam.org/v14): bad keys are fatal at startup.

## Architecture

```
format.py     study format, validation           case.py       the OpenFOAM case (wall_velocity, write_mrf)
polymesh.py   polyMesh writer, box, extrusion     foam.py       runs OpenFOAM, reads fields (also decomposed cases)
mesher.py     Gmsh (generate(): surface retries)  foamio.py     tolerant readers (fields, logs, checkMesh, polyMesh)
snappy.py     snappyHexMesh (surface, layers, retries, pockets, revolution tags)
meshsurf.py   STL/OBJ/PLY geometry: read, watertight check, orient, faces by feature angle, groups, inside test
sliding.py    sliding mesh (checks, faceZone, createBaffles/splitBaffles/createNonConformalCouples, parallel)
cavitation.py incompressibleVoF + VoFCavitation case, results, NPSH curve
select.py     selectors, bc -> patches            post.py       results and invariants (balances, wall_flux)
average.py    monitors, window/time averages, window_average   transient.py  URANS/DES, start, continue, init_from
scalars.py    tracers, mean age                   porous.py     porous zones, reactions on the part
variants.py   geometry refs, variants, mesh_scales, NPSH sweeps (compare)
executor.py   SLURM: submit, run-study, states, estimates, caps   cpushare.py  other processes on our CPUs, reap
render.py / render_vtk.py   images                refsolver.py  2D MAC reference solver
runner.py     one study end to end, verdict       bench.py      benchmarks
api.py, cli.py, mcp_server.py, guides.py, workers.py (pool workers end with the server)
```

MCP tools: study_format, validate_studies, run_studies, study_status, get_case, guide, add_geometry, list_geometry,
run_benchmarks, bench_status, jobs, cancel, get_image, render_images, reprocess, get_file.

## Lessons (b3dt, feat, and this tool)

1. Research the target program first (source, tutorials), then the spec, then code; list what is unverified.
2. Never rely on the program's internal numbering; recompute from geometry (CAD-face tags survive meshing).
3. Exact checks where possible, loose only where physics says so, with the reason.
4. Every run checks itself (mass, momentum, angular momentum, energy, scalars, wall flux, alpha bounds), and the
   verdict says why a result must not be reported.
5. Validate before running; an LLM must fix every error from its message alone (where, what, which key).
6. Compute derived quantities the way the target does (forces, fluxes on the same faces).
7. Keep `study_format` compact; long guidance in guides.
8. OpenFOAM traps: MRF subtracts Ω×r on every face of a zone cell, walls included (stationary walls inside a zone
   leak); snapped faces of a surface of revolution are jagged (judge by CAD face, not by facet); aarch64 cannot
   trap FPEs (NaN runs crawl: watch the log); decomposePar loops over every face pair of a non-conformal couple
   (decompose first); `createBaffles fields true` needs a bc for every field in 0/; `flowRateOutletVelocity`
   divides by the set flow (never 0); Gmsh can leave a surface empty without raising (check, retry).
9. Failures must say what happened: killed (memory) vs mesh error, cancelled, diverged, timed out.

## Environment

- Mac (Apple Silicon): development, Claude desktop. The sandbox mounts the repo; no OpenFOAM/Gmsh there.
- Machines (OpenFOAM 14, SLURM, cfdt from llmt's shared install), as of 2026.11.03:
  - **carbonio**: 12 CPUs / 6 cores, default 2 ranks (cap 6), `--mem=8G` default (snappy layer addition was
    killed there for memory: use run.mem), 190 min time limit.
  - **mochi**: 40 CPUs / 20 cores, default 9 ranks (cap 20), 510 min limit; busy with Rogue DES runs.
  - **spark1 / spark2** (aarch64, 20 cores each, own nodes spark-5da5 / spark-0808): default 10 ranks (cap 18),
    mem cap 96G, 190 min; `CFDT_SLURM_NODE=local`. No FPE trapping on aarch64.
- Transient and long runs continue in dependent SLURM jobs (max_jobs); results are collected with study_status.

## How to start

1. Read README.md (format, checks, versions) and "Open work" below.
2. Ask the user for the latest run results (request paths, machine) if not given; `jobs()`, `study_status` and
   `get_file` reach run folders on any machine through the router.
3. Fix from real output, one cause at a time; README row + versions + the commit command each time.

## Open work (2026.11.06, in order)

1. **Cavitation, first run** (2026.11.03.4): S50 at 20 m3/h, NPSH 15 m on carbonio stopped at once on SIGFPE
   (outlet flow ramp from 0; fixed in .4). Rerun that point: expect almost no vapour and the MRF head within a few %.
   Watch: VoF writes `p` (else p_rgh is read), `uniformTotalPressure` on p_rgh with the mixture's rho, pressure and
   forces function objects with `rho rho;` and p in Pa (absolute, pRef = target total pressure), LTS stability
   (maxCo 5, maxAlphaCo 2), the ramp's start level (p0 + rho max(U_tip, 3 U_ref)^2). Then the sweep
   (`cavitation.npsh: [...]`) and `variants.<name>.npsh_curve` / `npsh3_m`. No wall shear or y+ in VoF runs yet.
   Cavitation NPSH 15 m completed on .03.4 (6b0313, spark2): no vapour to speak of, but head/torque never settled
   (torque -47..+2 N m): LTS let the pseudo-time step reach 1 s in slow parts of the MRF zone (Omega dt ~ 300).
   11.04.4: maxDeltaT = 0.2/Omega, maxCo 2 (cavitation.max_co). Rerun the 15 m point; the 3 m point (on the old
   settings) will likely oscillate the same way.
   S50 sliding 6th try (a30e06, .04.4): NaN at 1.44e-4 s; projected volume fraction tracks the rotation up to one
   cell then ~1; k negative from step 7, pcorr NaN at 28; coupled faces up to 117 deg apart. Suspect: the snapped
   interface is not round at the zone's circular edges / next to walls. 11.04.5 reports roundness and lean places.
   Possible next fix: give snappy the zone cylinder's circular edges as features (an .eMesh), since only the part's
   edges are feature-snapped now (implicitFeatureSnap false).
   get_case on .04.5: the zone boundary is stepped next to the vane tips (r 59.66, 87 deg; tips 59.45, bore 60.0,
   gap 0.55 mm = 2.2 cells of 0.25 mm): snappy cannot snap the faceZone there. 11.04.6 measures distance/angle to
   the ideal cylinder and stops before the solve. Options: refine the gap to ~0.1 mm (hollow cylinder r 59.3-60.1;
   costly: one cell of motion per ~5e-6 s at 18 m/s tip speed), or frozen rotor (MRF) at several rotor angles.
   Cavitation NPSH 15/3 m (d01011, .04.7): outlet held a VOLUME flow, the inlet's liquid flow fell 1.6/12 % as
   vapour formed; head rose at 3 m (lower flow). 11.05.1: mass-flow outlet, cavitation.flows (mass, volume source,
   net evaporation), mass-based balance, NPSH3 leaves out points off the flow by > 1 %. Rerun the sweep (20, 10,
   6, 4, 3.5, 3, 2.5, 2 m) on the fine-front-gap mesh.
2. **Sliding mesh, first full run** (2026.11.03.2): S180 (13.3 M cells, 18 ranks, spark2) built the couple in
   serial, then decomposePar never finished (O(N^2) over the couple's faces). Now: decompose the conformal case,
   build the couple with `-parallel`, keep the case decomposed, read results from processor* (cfdt_addressing);
   no reconstructPar. Rerun as a new request. Watch: parallel createBaffles with faceZone faces on processor
   boundaries, the MRF steady start then `movingWallVelocity`, foamRun with the mover and NCC, time per
   revolution (1 deg steps, maxCo 2), results read from processors, averages over whole revolutions. The case
   archive (.tgz) has no results for decomposed cases.
   S50 sliding (spark2, 2026.11.03, 7.25 M cells, checkMesh failing): steady start and parallel couple fine, then a
   crash in unconformInternalFaceCorrectMeshPhi (free(), exit 139) at the first motion. 11.03.5 retries with
   correctMeshPhi off and reports the interface cells' quality; round rotor walls outside the zone now spin
   (rotatingWallVelocity); own-rotation walls count in the zone's torque. Open: whether the crash is the mesh
   (poor cells at the interface) or OpenFOAM's stitcher; a small sliding benchmark would tell.
   S50 sliding rerun (spark1, 2026.11.04, no layers, skewness 8.12): NaN in pcorr at the 8th step (t 1.5e-5 s);
   OpenFOAM's projected volume fraction grew 0.31 -> 1.06 in 0.26 deg: a size mismatch across the interface (the
   0.2 mm band straddled the zone's end but was only 1.1 mm thick; elsewhere on the rim/back end not checked). 11.04.1: smaller-cell side as NCC neighbour,
   interface size/skewness report, skewness places. Rerun ec3392 (spark1, .04.1): one 0.25 mm band z -47.5..-45.0 matching the tongue boxes and back-gap cylinder;
   the user sends interface_cells / worst_skewness if it still flags. Another suspect (pump chat): 18 boxes of 0.25 mm
   at the vane tips at fixed angles; their cells pass 1.25 mm stator cells as the rotor turns. 11.04.2 checks
   ring-wise size ratios and warns on non-round refine regions at the interface; if ec3392 diverges the same
   way, the user replaces the tip boxes with refinement all the way round the rim: 11.04.3 adds hollow refine
   cylinders (inner_radius; a triangulated annulus for snappy) for a 0.25 mm band r 58.5-61, z -77.1..-45.0.
3. **S50 meshing**: layered meshes failing checkMesh now rebuilt without layers (.11.03.3; restored if that fails
   too); Gmsh surfaces left empty retried with other algorithms (.11.02.2; face 121); a 2.8e-7 mm gap the user
   closes in CAD. The S50 MRF zone had wall leaks (impeller faces outside the zone, casing faces inside, 22-44 %
   of the inflow at 11.01.5): the zone or the geometry must change, for MRF and sliding alike (`guide('pumps')`).
3b. **Wall-flux warning after reprocess** (2026.11.03): 1.3 % of the inflow through walls on the S50 15 % and 35 %
   runs; every S50 run (2 Nov baseline too) has 0.3-0.9 % more outflow than inflow, so it was always there.
   11.03.7 adds `checks.wall_flux.where` (inside/outside the zone, CAD faces, places) to find it. Found (pump chat):
   the wear plate (z -46.0, zone end at -46.25, 0.5 mm cells: its cells rotate) 0.70 %, hub face just outside the
   zone 0.23 %, tongue nothing; same in every volute variant (comparisons unbiased); absolute head/torque ~ +1 %.
   Fix: 0.2 mm front-gap refinement (+4 M cells); the user may run one 40 m3/h baseline with it on spark2.
   11.03.8: revolution faces count by their net per CAD face in the judged wall flux (they were left out). Earlier note:
   the reprocessed S50 15 % run (11.03.6) warns that 1.3 % of the flow passes through walls. The user compares
   it with the baseline rerun and looks at where it happens (`checks.wall_flux.by_wall`) before relying on it.
   Suspect to rule out first: the saved state (render.save_state) keeps only scalar/list `mesh.info`, so
   `revolution_tags` (a dict) is lost and a reprocess judges wall flux with the older per-CAD-face net rule,
   not the run's own (face by face, surfaces of revolution left out). If so: keep revolution_tags in state.json,
   and say in the warning which rule was used.
4. **S180 MRF**: wall-leak and own-rotation fixes through 11.01.6; s180_base_Q250_v6 timed out on spark2 (job 322):
   check its state with the user.
5. **#25 transient / #26 Ahmed and Rogue DES**: cylinder benchmarks pass (10.22.5); Rogue spatial DES runs
   (spatA/C/D/E) were running/queued on mochi on 2026.11.02.
   2026.11.06.1 (cfdt_dev_prompt_router_geometry.md, all 10 items done on the cfdt side; untested on OpenFOAM):
   stl geometry (meshsurf.py) and group/solid/in_box/in_cyl selectors, several solids for external flow
   (forces_by_solid), get_case ref, goal files, request_name, eta-based expected_start, progress between segments,
   study_status(window), init_from (mapFields with an empty patchMap: internal fields only), per-node queue.
   Verify on a machine: (a) the Rogue as stl with wheel groups through snappy (cut face on ymin dropped, groups in
   the final patches, forces_by_solid); (b) init_from on a spat variant: mapFields' log, the first steps' Courant
   number, whether nut needs mapping or recomputes; (c) per-bc force histories (cfdtFp_*) cost nothing visible;
   (d) Gmsh path with several STEP solids (occ.getDistance exists in Gmsh >= 4.11; else bounding boxes).
   2026.11.06.2: steady init_from and chained NPSH sweeps; verify on S50: mapFields on 7 M cells (time, log),
   the chained sweep's head at the first iterations, whether the pressure ramp from the previous point's p0 is
   smooth, and whether fewer iterations per point (e.g. 3000) still settle.
   Router side (llm-translators): done by the user 2026.11.06 (sections 0-4); llmt needs a deploy for pending jobs'
   requested node. Was: node-filtered queue and sbatch --test-only, starts_now,
   cluster_queue; pass request_name and return the request path on a time-out; stage local paths for
   add_geometry(goal=<file>), add_geometry(stl=...) and geometry.stl / geometry.groups.
6. Benchmarks still missing: a sliding-mesh one (e.g. Taylor-Couette with a sliding inner zone vs the analytic
   torque, then a rotor-stator case), cavitation ones (cavitating orifice/venturi vs Nurick's discharge
   coefficient; a non-cavitating VoF point equal to the single-phase result).
7. Other tools (other chats): feat S50 inspection-cover joint (contact chatter in the preload step: prompt sent
   2026.11.02).

Future (agreed, not started):
- **Sliding-mesh cavitation**: the cavitation case on the sliding zone (incompressibleVoF supports the NCC mover,
  as its propeller tutorial: correctPhi, correctMeshPhi, movingWallVelocity / movingWallSlipVelocity, alpha
  zeroGradient on the couple's patches); averages over whole revolutions; cavity shedding and pressure
  pulsations. Needs to_sliding to set alpha.water / p_rgh patchFields and a transient VoF controlDict.
- One mesh shared by all points of a sweep: done for chained studies (2026.11.06.3, cavitation.chain or init_from with
  the same mesh key); unchained parallel sweeps (flow points) still mesh each point.
- Free-surface VoF (tanks, priming, sloshing, air entrainment); multi-region CHT (dual-fluid heat exchangers);
  Euler-Euler and particles (larger efforts, see the 2026.11.03 discussion).
- Energy with MRF; supersonic; radiation.

## Reference: cavitation (incompressibleVoF + VoFCavitation), implemented 2026.11.03.1

User's choices (2026.11.03): MRF pseudo-steady first; a built-in NPSH sweep; sliding-mesh cavitation later.

OpenFOAM 14 facts (tutorials incompressibleVoF/propeller and DTCHull, sources read on GitHub):
- solver incompressibleVoF; constant/phaseProperties `phases (water vapour); pSat; sigma;`;
  physicalProperties.water / .vapour `viscosityModel constant; nu; rho;`; constant/g (0 0 0) for a pump.
- constant/fvModels `VoFCavitation { type VoFCavitation; model SchnerrSauer; SchnerrSauer { pSat; liquid water;
  n 1.6e13; dNuc 2e-6; Cc 1; Cv 1; } }` (Kunz, Merkle: UInf, tInf, Cc, Cv).
- Fields alpha.water, U, p_rgh (Pa, absolute; = p with g = 0), k/omega/nut. BCs: walls fixedFluxPressure (p_rgh),
  zeroGradient (alpha); inlet `uniformTotalPressure` (p0 a Function1 of time: the ramp) with
  pressureInletOutletVelocity; outlet `flowRateOutletVelocity` (volumetricFlowRate Function1; it divides by the
  set flow: the ramp starts at 5 %).
- VoFSolver has MRFZones and LTS (ddt localEuler; PIMPLE maxCo, maxAlphaCo, rDeltaTSmoothingCoeff,
  rDeltaTDampingCoeff, maxDeltaT). Schemes: div(phi,alpha) Gauss noInterfaceCompression vanLeer, div(rhoPhi,U),
  div(((rho*nuEff)*dev2(T(grad(U))))). Solvers "alpha.water.*" (MULES, MULESCorr), p_rgh, p_rghFinal, "pcorr.*".
- Function objects with p in Pa: `pressure` needs `rho rho;` (it looks rho up as a field); `forces` with
  `rho rho;` and `pRef` (the target total pressure).

Design: `cavitation: {npsh: m | [m, ...], temperature | p_vapour, vapour {rho, mu}, model, iterations (4000),
ramp (0.3), head_drop (0.03), sigma}`; one inlet (its flow rate moves to the outlet) and one outlet; inlet total
pressure p_v + rho g NPSH, ramped down from a vapour-free level; the last quarter averaged; an npsh list expands
into `<name>_npsh<value>` studies and compare() gives the NPSH curve and NPSH3. Reader: p (Pa, absolute) ->
kinematic gauge (p - p_ref) / rho_l; alpha.water -> `vapour` scalar for images. Not with transient, sliding,
energy, tracers, porous zones.

## Reference: sliding mesh (OpenFOAM 14 NCC), implemented 2026.11.02.1-11.03.2

- propeller tutorial: createBaffles (faceZone -> two patches, `fields true` with patchFields), splitBaffles,
  createNonConformalCouples `-fields` p1 p2 (overwrite by default in v14; `-noOverwrite`); dynamicMeshDict
  `mover { type solidBody; libs ("libpointMeshMovers.so"); cellZone; solidBodyMotionFunction { type
  rotatingMotion; origin; axis; omega N [rpm]; } }`; U `movingWallVelocity` on rotor walls,
  `movingWallSlipVelocity` on the couple's patches; `#includeEtc "caseDicts/setConstraintTypes"`; PIMPLE
  correctPhi, correctMeshPhi, MeshPhi and "pcorr.*" solvers.
- Zone files: classes `cellZoneList` / `faceZoneList`.
- decomposePar on a coupled mesh loops over every pair of the couple's faces (domainDecompositionDecompose.C,
  processInterCyclics): decompose the conformal case first, build the couple in parallel, never reconstruct.

## Reference: tracers and porous zones (done 2026.10.19, benchmarks pass on mochi)

For gas and liquid reactors: what mixes, how long the fluid stays, and how packed beds, catalyst monoliths and filters
resist and distribute the flow. Steady, single-phase, no reactions. Tracers first (low risk), then porous zones.
Before writing case files, check every OpenFOAM 14 keyword against its source (as for the .16 monitors): a bad key
in a function object or fvModel is fatal at startup.

### 1. Tracers (passive scalars) and mean age — 2026.10.x.a

- **Format.** `"scalars": [{"name": "tracer", "Sc": 1.0 | "diffusivity": m2/s, "inlets": {"<inlet bc>": value, ...}}]`.
  Inlets not listed take 0; walls zero flux; outlets/openings inletOutlet (0 on backflow).
  `"age": true` adds the mean age of fluid (time since entering): source 1 per unit volume (rho for gas), 0 at
  every inlet. Turbulent runs add nut/Sct (Sct 0.7, settable) to the diffusivity.
- **Case.** OpenFOAM's `scalarTransport` function object (libsolverFunctionObjects), one per scalar, solved every
  iteration with the flow; `0/<name>` with the bcs above; `div(phi,<name>)` bounded limitedLinear; a solver entry
  in fvSolution. Age: the same object with a unit source (fvModel `semiImplicitSource` on the field, or the
  object's own source if v14 has one). To verify in the v14 source: keywords (`field`, `diffusivity`, `D`,
  `alphal`/`alphat`, `schemesField`, `nCorr`), how the source is attached, and that it runs in steady
  (SIMPLE) mode. Energy runs: phi is a mass flux; check how scalarTransport takes rho.
- **Results.** Per outlet: flow-weighted mean, min, max and the mixing index (1 − CoV of the scalar over the
  outlet, flow-weighted); volume mean; at probes. Age: per outlet flow-weighted mean age (the mean residence
  time), volume-mean age, max age and where (dead zones), the fraction of volume older than 2 V/Q.
  Images: `field: "<scalar name>"` and `"age"` in slices and surfaces.
- **Checks on every run.**
  - Scalar conservation: sum over open boundaries of phi*s = 0 (no source), so the outlet mean is
    sum(Q_i s_i)/Q exactly; error reported and judged like the mass balance.
  - Age: with one inlet and no diffusion through inlets/outlets, the outlet flow-weighted mean age is V/Q exactly
    (any geometry, any flow): judged within 1 %.
  - Bounds: 0 <= s <= max inlet value (no overshoot), reported.
- **Benchmarks.**
  - `mixing_layer_2d`: uniform flow (slip walls), tracer 1 on half the inlet, 0 on the other half:
    c = ½ erfc(y / (2 sqrt(D x / U))) at two stations (exact for constant D and U).
  - `tee_mixing`: two inlets 1 and 0 with flows Q1, Q2: outlet mean = Q1/(Q1+Q2) (exact, conservation).
  - `age_pipe`: laminar pipe, outlet mean age = V/Q (exact); the volume-mean age differs (laminar profile), so
    both numbers are checked.
  - `age_cavity_tank`: a tank with an inlet and outlet: mean age = V/Q exactly, and the dead zone shows.
- **Limits stated in study_format.** Mean age gives the mean residence time and where fluid stagnates, not the
  full residence-time distribution (that needs a transient tracer pulse: later, with transient).

### 2. Porous zones (packed beds, monoliths, filters, screens) — 2026.10.x.b

- **Format.** `"porous": [{"name": "bed", "region": {"box": [x0,y0,z0,x1,y1,z1]} | {"cylinder": {...}} |
  {"sphere": {...}}, "darcy": [dx, dy, dz] (1/m2), "forchheimer": [fx, fy, fz] (1/m), "axes": [[e1], [e2]]}]`, or
  `"ergun": {"particle_diameter": mm, "porosity": eps}` (d = 150 (1-eps)^2 / (eps^3 dp^2), f = 3.5 (1-eps) /
  (eps^3 dp), OpenFOAM's convention: dp/L = mu d U + ½ rho |U| f U), or `"monolith": {"channel": mm,
  "open_fraction": eps, "shape": "round" | "square"}` (laminar channels along e1: d1 = 32 / (eps dh^2) round, 28.4 / (eps dh^2) square; d2 = d3 = 1000 d1).
  Velocities are superficial (OpenFOAM's convention), said in study_format.
- **Case.** A cellZone from the region (cell centres, as for rotating zones; zones may not overlap).
  `constant/fvModels`: `explicitPorositySource` with `DarcyForchheimer` (d, f, coordinateSystem from `axes`).
  To verify in the v14 source: the fvModel keywords (cellZone selection key, `d`, `f`, the coordinate system
  layout). Works with energy runs for the flow; the solid heat capacity/conduction is not modelled (said so).
- **Results.** Per zone: superficial velocity mean and range, pressure drop across the zone (flow-weighted
  means on the zone's inflow and outflow faces), the uniformity index of the flow leaving the zone
  (1 − ½ mean|u − ū|/ū, the catalyst/filter maldistribution number), and the zone's force on the fluid.
- **Checks on every run.**
  - Momentum balance with the porous force: sum over zone cells of −(mu d + ½ rho |U| f)·U V added to the
    balance (as the MRF Coriolis term is now); without it the balance would flag every porous run.
  - The zone's force equals the pressure drop × area for a 1D zone (reported).
- **Benchmarks.**
  - `porous_channel_1d`: slip walls, uniform flow through a block: dp = L (mu d U + ½ rho f U²) exact.
  - `ergun_bed`: the same with `ergun` input against the Ergun equation (exact by construction, checks the
    conversion and units).
  - `porous_bypass`: a porous plug in half of a channel (slip walls, inviscid split): the flow divides so the
    pressure drops match (loose, 5 %).
  - `monolith_straightening`: angled inflow into an anisotropic block: the outflow leaves along e1 (angle < 1°).
- **Limits.** No solid-phase energy, no reactions on the surface, no particle-resolved beds.

## Reference: snappyHexMesh in OpenFOAM 14 (implemented 2026.10.21, snappy.py)

OpenFOAM-14 source facts (checked on GitHub, 2026-10-21):
- snappyHexMesh overwrites constant/polyMesh by default (-noOverwrite to keep times). No fvSchemes/fvSolution needed;
  controlDict: writeFormat ascii needs mergeTolerance >= 10^-writePrecision (1e-6 with 6 digits is fine).
- Geometry: `type triSurface; file "part.stl";` from constant/geometry; `type box; min; max;`; `type cylinder; point1;
  point2; radius;`; `type sphere; centre; radius;`. Multi-solid STL surface `part` gives patches `part_<solid>`
  (single solid: `part`); `regions { <solid> { name <patch>; } }` renames.
- refinementSurfaces: `part { level (a b); regions { <solid> { level (a b); patchInfo { type wall; } } } }` (level
  mandatory per region; no patchInfo -> wall). refinementRegions: `mode inside|outside; level N;` or
  `mode distance; levels ((d1 l1) (d2 l2));` (increasing d, non-increasing level), allowed on the part surface too.
- castellatedMeshControls mandatory: maxGlobalCells, maxLocalCells, minRefinementCells, resolveFeatureAngle,
  nCellsBetweenLevels, allowFreeStandingZoneFaces, insidePoint. features ( { file "part.eMesh"; level N; } ) from
  `surfaceFeatures` (system/surfaceFeaturesDict: surfaces ("part.stl"); includedAngle 150;).
- snapControls mandatory: nSmoothPatch, tolerance, nSolveIter, nRelaxIter; optional nFeatureSnapIter,
  explicitFeatureSnap, implicitFeatureSnap.
- addLayersControls mandatory: relativeSizes, minThickness, featureAngle, nGrow, maxFaceThicknessRatio,
  nBufferCellsNoExtrude, nLayerIter, layers {"<patch regex>" {nSurfaceLayers N;}}, nSmoothSurfaceNormals,
  minMedialAxisAngle, nSmoothNormals, maxThicknessToMedialRatio, nSmoothThickness, nRelaxIter; exactly two of
  firstLayerThickness / finalLayerThickness / thickness / expansionRatio. nRelaxedIter enables meshQualityControls.relaxed.
- meshQualityControls mandatory: maxNonOrtho, minVol, minTetQuality, maxConcave, maxInternalSkewness,
  maxBoundarySkewness, minFaceWeight, minVolRatio, minTwist, minDeterminant, errorReduction, nSmoothScale
  (minFaceFlatness optional; tutorial's minFlatness is ignored). `#includeEtc "caseDicts/mesh/generation/meshQualityDict"`.
- Log: layer table after "patch faces layers overall thickness" / "[m] [%]" / "----- -----" header: rows
  `patch faces avgLayers thickness_m percent`; "Layer mesh : cells:N faces:.. points:.."; "Finished meshing ...".
- Parallel: decomposePar (-force), mpirun snappyHexMesh -parallel, then `reconstructPar -constant` (no times otherwise).
- blockMesh: every block face in a patch (unlisted -> empty -> snappy fails "not fully 3D"), or defaultPatch {type patch;}.

## State (history, newest first)

**2026.11.11.3**: llmt-runtime>=2026.11.11.2 (`--nodes=1` for every one-node MPI job); job.sh test updated.

**2026.11.11.2**: multi-node jobs (not one_node; CFDT_SLURM_NODES; ranks capped at S.max_job_procs; caps from the
runtime when jobs span nodes); clusters.md: Perlmutter 4 x 128 with CFDT_JOB_SETUP; Engaging one node.

**2026.11.11.1**: clusters (Perlmutter, Engaging): cfdt installed there and run from its CLI (other/supercomputers/clusters.md),
CFDT_LAUNCH, `cfdt run --wait 0`, `cfdt pack` / `import_results`. Multi-node: 2026.11.11.2. Verify the OpenFOAM 14 builds and other/supercomputers/clusters.md section 5 on both clusters.

**2026.11.08.2**: queued jobs' `ahead` counts only pending jobs that can run on the same node.

**2026.11.08.1**: init_from transient runs no longer reported as steady iterations (they were running); mapping
stage in progress, CFDT_MAP_TIMEOUT. mapFields 911k -> 865k cells: < 1 min on mochi.

**2026.11.07.2**: worker pools end with the server (workers.watch_parent, as feat).

**2026.11.07.1**: node memory (SLURM RealMemory) as caps.mem / caps.mem_mb; run.mem above it fails validation.
Router (user's): skip machines whose executor.caps.mem_mb < run.mem before staging.

**2026.11.06.4**: window averages built by the run's own averaging code (porous part was not scaled by rho: v3g
cd 0.4029 vs 0.4078); no window cut on an in-window frequency; regression test window = run's averages.

**2026.11.06.3**: a chained study with the same mesh key takes its source's mesh (meshed.pkl), fields copied.

**2026.11.06.2**: solve.init_from for steady runs (MRF, cavitation), same-request sources (SLURM afterany),
cavitation.chain for NPSH sweeps, internal fields copied on identical meshes. See README row.

**2026.11.06.1**: mesh geometry (STL/OBJ/PLY, groups), region selectors, several solids, window averages,
init_from, per-node queue, request_name, eta-based starts, NaN-safe images. See README row.

**2026.11.03.1-.4**: cavitation (incompressibleVoF, MRF, LTS, NPSH sweep, NPSH3); sliding mesh in parallel
(decompose first, read processor*); layered snappy meshes failing checkMesh rebuilt without layers; killed
programs reported as `Killed` with a run.mem hint; outlet flow ramp from 5 % (SIGFPE). See README rows.

**2026.11.02.1-.3**: sliding mesh (rotating[].motion "sliding"; checks, NCC steps, revolution averages,
guide('pumps')); Gmsh surfaces left empty are retried and mesh notes kept in failed results; createBaffles fields
(only solved fields in 0/), zone class names.

**2026.11.01.1-.6**: MRF wall leaks (pre-run check per wall against its cell's frame, post `checks.wall_flux`
with verdict), Gmsh surface retries (MeshAdapt, Delaunay, Frontal-Delaunay), own-rotation walls (full Ω×r in a
same-speed zone, tangential on CAD surfaces of revolution found by snappy), layer folds remade without layers,
every bad region reported, `mesh.layers.at_openings`.

**2026.10.31.1-.5**: refinement shapes in the estimate, no outsidePoint (region zoning), rotating walls at the
zone's edge accepted, snappy flatness and mergeFaces retries, pockets cut off from the flow removed (subsetMesh),
NaN watch on running logs (aarch64), CPU-share watcher, leftover OpenFOAM processes reaped after scancel.

**2026.10.25.1-10.28.4**: SLURM on mochi and the sparks (node pinning, rank caps, memory auto/size), llmt rules
(stale servers, ToolError, tools return within wait_s), Ahmed benchmarks, VTK renderer, porous reactions on the
part (and their checks), study_format trimmed under 12k.

**2026.10.22.5-.10**: transient benchmarks pass on mochi (cylinder 2D-2 and its restart), test isolation from
the machine's config, estimates in validation.

**2026.10.22.4**: #25 transient (transient.py): URANS / kOmegaSSTDES / laminar, steady start, fieldAverage, time averages with frequency/Strouhal, stop (stopAt writeNow) and continue in a dependent SLURM job. Unverified on OpenFOAM: run cylinder_2d2 (quick), cylinder_2d2_restart (job chaining), then ahmed25_des / ahmed35_des. Watch: fieldAverage startTime key (timeControl), stopAt writeNow picked up in parallel (timeStamp check), reconstructPar after the last segment, Courant-limited dt with thin layer cells.

**Ahmed results (2026.10.22, steady kOmegaSST, wall functions, y+ ~60, half model, moving ground)**: Cd by mesh L1/L2/L3/L4 (95k/240k/600k/1.64M cells): 25 deg 0.322/0.293/0.266/0.253 (extrapolated ~0.24, tunnel 0.285: -15 %); 35 deg 0.310/0.297/0.273/0.265 (extrapolated ~0.261, tunnel 0.26: +0.5 %, fully steady). 25 deg L3 in a larger domain (blockage 1 % instead of 2 %): Cd 0.276, Cl 0.283 and not settled (Cl scatter 2.8 %) against 0.266 / ~0.17 settled: the 25 deg slant flow changes state with small changes (partial separation, bistable), so steady RANS is not a reliable model there. 35 deg passes; 25 deg goes to #25 (DDES). Layers 98-99 %, balances ~1e-5, L4 ~1.6 h on 19 ranks.

**2026.10.22.3**: #24 geometry by reference (add_geometry / list_geometry, geometry.ref) and variants (mesh_scales, variants) with a mesh-convergence comparison; detail 'brief'. Ahmed (.2): L1-L3 done (L3 within 10 %, settled, layers ok, but Cd not mesh-converged and 25/35 order reversed); L4 x2 and L3 in a larger domain submitted as studies (run_20261022_121720_044316). Next: #25 transient URANS/DDES with restarts.

**2026.10.21.2**: snappyHexMesh path (#1 layers + coverage + y+ plan, #5 refinement shapes, #10 quality: checkMesh report, stricter retry, Netgen for Gmsh tets, non-orthogonal schemes). User's list from the vehicle chat also covered: per-CAD-face tags survive meshing, ground layers optional (layers.on), feature-edge snapping, half-car cut at a domain side, radiator/MRF as snapped cellZones, narrow gaps refined (gapLevelIncrement) and flagged, quality retry. Unverified on OpenFOAM (no OpenFOAM/Gmsh in the sandbox): run sphere_snappy_re100, sphere_half_snappy, pipe_snappy on mochi first; then the 2D layer regression, Ahmed 25/35 at three levels, Rogue v3 (underbody, radiator) as the stress test. Watch: region renaming (patch names = bc names, else part_r<k> aliases are accepted), parallel snappy finding constant/extendedFeatureEdgeMesh (copied into processor dirs), reconstructPar -constant, binary label size (no 'arch' in v14).

**2026.10.21.1**: vehicle-aero quick fixes (#4 y+ from shear, part-only warning; #3 Cd/Cl averaged from a forces monitor on the part walls; #6 --mem from the cell estimate; #8 summary detail; #9 cp and clim). Next, in order: snappyHexMesh path (#1 layers with collapse + y+ report, #5 box/cylinder/distance refinement, #10 quality controls), geometry by reference and mesh variants (#7), transient URANS/DDES with averaging and restarts across SLURM jobs (#2); acceptance: Ahmed body 25 and 35 degrees, three meshes, layers, y+ 30-300, Cd within ~10 % (0.285 / 0.26).

**2026.10.20.1**: per-study `run` {timeout_s, mem, procs, queue} over config/env, with CFDT_MAX_* caps (the user cannot ask Claude to edit config/env on mochi); execution.limits in results; EXECUTION.md section 'Per-study run settings' for lmpt and feat; translators.json lists CFDT_MAX_*.

**2026.10.19.3/.4**: run_benchmarks submits and returns, bench_status collects (all six new benchmarks pass on mochi with .3). Jobs ask for ranks from an upper cell estimate before meshing (executor.cell_estimate / procs_for_job).

**2026.10.19.2**: the six tracer/age/porous benchmarks pass on mochi; OpenFOAM 14 took every new keyword (scalarTransport, semiImplicitSource cellZone all, porosityForce fixedCoeff + axesRotation). Fixed: age_cavity selectors; porous Δp −2.6 % from the collocated-grid velocity dip at zone faces (20 cells) → 100 cells: −0.4 %, plus face_cell_fraction warning.

**2026.10.19.1**: tracers, mean age (scalars.py) and porous zones (porous.py), per the plan above. OpenFOAM 14 sources read: scalarTransport (keys field, diffusivity none|constant|viscosity, D, alphal, alphat; sources via fvModels; it writes every iteration unless writeControl writeTime); semiImplicitSource (volumeMode, sources {f {explicit; implicit}}, cellZone); porosityForce (tutorial fluid/angledDuct: type porosityForce; porosityForce {cellZone; type DarcyForchheimer|fixedCoeff; ...; coordinateSystem}). fixedCoeff chosen: DarcyForchheimer looks up nu/mu fields, which incompressibleFluid may not register; fixedCoeff's rhoRef must be 1 for the kinematic equation. Unverified on OpenFOAM: `cellZone all` for the age source, scalarTransport inside the fluid solver (rho lookup), fixedCoeff's rotation (e1/e3). Benchmarks to run on mochi (all small, < 1 min): mixing_layer_2d, tracer_two_streams, age_cavity_2d, porous_channel_1d, ergun_bed, porous_anisotropic.
- Scaling test on mochi (pump_v5 Q115, 899k tets, kOmegaSST + MRF, 200 iterations, idle machine): solve 178 s on 19 ranks, 210 s on 10, 283 on 6 (19 ranks only 1.18x faster than 10): memory-bandwidth bound. Two 10-rank jobs at once: 316 s each, so a pair takes 316 s of solve against 2 x 178 = 356 s one after the other on 19 (11 % more throughput; 24 % on wall time, 413 s vs 540 s, because meshing (40 s) and results (52 s) are serial and overlap). For flow-point sweeps: pairs on 10 ranks; single studies: 19.

**2026.10.18.15/.16**: iteration averaging (average.py, monitors in controlDict). On mochi with .15: the torque monitor works (Taylor-Couette MRF, local run: torque -6.706e-7 vs 6.702e-7 analytic, settled, drift 0.14 %, scatter 0.10 %); the inlet/outlet surfaceFieldValue stopped OpenFOAM 14 at startup (weightedAverage not an operation). .16: operation average + weightField phi, patch only, no mode in the pressure FO. To check on mochi: swirl_pipe with solve.average true (cfdtP_inlet/outlet written, no dropped series, total_pressure_drop = its _last). Caveat: phi-weighted average with backflow faces on an outlet; the 5 % agreement check drops it if it goes wrong.

**2026.10.18.13**: venturi variants on mochi (.12, local runs): rest / estimate / SIMPLE / relax 0.5,0.2 / ramp without transonic: all diverge (negative T or GAMG). Ramp 400 + transonic: stable, 4000 iterations, residuals ~2e-5 (h 2.4e-5). Now the default for total-pressure gas inlets. Results then failed on KeyError 'Q' (total-pressure inlet in inlet_flow_error): fixed. On mochi with .13: venturi_subsonic PASS, converged in 1889 iterations (32 s): mass flow -0.12 %, throat pressure -13795 Pa vs -13970 isentropic (1.25 %), energy balance 0.7 %. All energy benchmarks now pass on mochi (conduction_slab, heated_channel, natural_cavity, venturi, pipe_air_low_mach), as do the rotation ones.
- Disconnect test (EXECUTION.md) PASSED on mochi (2026.10.18.13): job 2534 ran on while Claude was closed; a new server found it with jobs() and collected the full result (execution: slurm, 6 of 6 ranks, mpirun -np 6, 6 cores / 12 hwthreads). lmpt already writes the same `execution` record (lmpt 2026.10.18.5).

**2026.10.18.12**: .11 on mochi: heated_channel_2d PASS and converged. Venturi variants (local runs): default, consistent false, relax 0.5/0.1, total pressure 100 and 1000 Pa: all SIGFPE in MachNo after iteration 1 (limitPressure min -1.3e6 at 10 kPa); 60 m/s velocity inlet runs fine (throat 86.9 m/s, M 0.25). MachNo dropped; solve.start and solve.ramp added to try next.

**2026.10.18.11**: .10 on mochi: heated_channel_2d and pipe_air_low_mach PASS (87 s, 28 s). venturi_subsonic still diverged at iteration 2 (p GAMG to 1e6, limitPressure saw +-1e25). Now the flow starts at half the isentropic inlet speed; if it still fails, try solve.consistent false and solve.relax.p on mochi with queue false.

**2026.10.18.10**: .9's pMinFactor/pMaxFactor in PIMPLE is a fatal error in OpenFOAM 14 ("Pressure limits should now be specified in fvConstraints"): now system/fvConstraints limitPressure.

**2026.10.18.9**: mochi run of .8 (request run_20261018_203019_cbfc51, --mem=16G now in CFDT_SLURM_ARGS).
- pass: taylor_couette_mrf, swirl_pipe, natural_cavity_ra1e4 (reliable), conduction_slab_2d (in the .7 run). pipe_air_low_mach: converged in 177 iterations, 20 s, dp -0.8 %, but 'unreliable' from the energy balance of an isothermal run: no longer judged.
- heated_channel_2d: Nu +1.7 %, outlet T ok, energy 0.2 %, but Uy (|Uy| <= 1e-6 m/s, 1.3e-3 of the speed) stalled at 7e-3: residuals now scaled by component size.
- venturi_subsonic: builds now; diverged at iteration 2 (continuity errors 1e4, then MachNo sqrt of negative T), also with relax 0.3. Now: transonic off below Mach 0.7, p relaxed 0.3, pMin/MaxFactor. Unverified: if it still fails, try solve.transonic and solve.relax.p on mochi with queue: false.

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
- heat transfer/CHT with feat; transient (PIMPLE, sliding mesh for impellers); snappyHexMesh; porous zones and tracers (plan above);
- a pump example (volute + impeller goal) once MRF is confirmed.

