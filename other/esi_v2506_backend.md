# ESI OpenFOAM v2506 backend for sliding meshes: specification (draft, 2026-11-17)

For the cfd-translator developers. It turns priority 2 of `planning.md` ("limited ESI v2506 support for specific
cases") into what the backend must write, run, parse and check. Everything here comes from cases that ran: the
hand-built S50 v2506 case on Perlmutter (as built, 5 design variants, DES, movies; 10-17 Nov 2026), the frozen-rotor
variant on MIT ORCD, and the coarse carbonio case. Companion to `other/sliding_meshes.md` (which compares the two
couplings); this file is about the code.

## 1. Scope and selection

- **In scope:** incompressible, isothermal, single-phase flow with a sliding (turning) cell zone coupled through a
  cyclicAMI pair; laminar or k-omega SST (RAS), optionally kOmegaSSTDDES; steady MRF start or impulsive start;
  fixed time step in degrees; the same results as the v14 path (pump head, power, efficiency, flows, probes,
  revolution averages, pulsation statistics, images).
- **Out of scope (refuse with a clear message, suggest the v14 path):** energy, compressible gas, cavitation and
  VoF, gas release, porous zones, scalar transport, fvModels sources, external aero, steady-only studies.
- **Selection:** explicit only. Study key `solve.backend: "esi"` (default `"openfoam.org"`). It sits at the
  `solve` level because the steady MRF start runs on the backend too. Validation allows it only with
  `rotating[].motion: "sliding"` and refuses everything out of scope. cfdt may *suggest* it in a warning when its
  v14 cost estimate for a sliding run exceeds a threshold (section 7); it never switches silently.
- **Mesh once.** The ESI backend takes cfdt's v14 mesh and its meshing checks unchanged, and only writes and runs
  the ESI case. To test first:
  - whether v2506 reads OpenFOAM 14's zone files (`cellZoneList`/`faceZoneList` header classes) and v14 binary
    meshes: a v2506 `checkMesh` on a cfdt-written mesh settles it, and if not the headers are converted when the
    ESI case is written;
  - v2506's directional refinement in snappyHexMesh stays a later option.

## 2. Status of the decision gate (17 Nov)

- v2506 cyclicAMI runs the S50 at a fixed 1 degree per step on 13.7 M cells: 5.1 s per step on 512 Perlmutter
  ranks, about 2 node-hours (260 core-hours) per revolution.
- v14 NCC is stable only up to about 2 interface cells passed per step. The long-cell O-grid (`OGRID_ANISO=1`)
  was meant to allow 1 degree per step:
  - run 629: the interface had been meshed at level 4 (4.3 cells per degree) by a port bug. It diverged from step 2.
  - run 634: the corrected mesh (2.13 cells per degree max, 8.65 M cells). The gate at rest passed, it was stable
    for 6 steps (Courant max 43-70), then ran away from step 7 at the top rim corner under the wear plate
    (θ ≈ 94°). Stopped at step 10 by the new CO_STOP watch. 47 s per step on 18 ranks.
- **What 634 showed (dev's analysis).** The interface coupling stayed healthy throughout: coverage min 0.92,
  projected volume fraction under 0.15, 2.13 cells per degree. The run failed on the Courant number of the
  impulsive start-up jet, not on the interface rate:
  - steps 1-6: the vanes push fluid over their tips at 37-55 m/s relative to the mesh, across 0.125 mm cells
    under the top face (θ 93-101°, moving with a vane), giving Courant 43-70 at 1°;
  - from step 7: the runaway moves to the stator-side cells beside it (θ ≈ 94°, fixed in space), first through a
    coupled face.

  The v2506 impulsive start (61358002) showed the same jet is a start-up transient. In developed flow, about
  +10 m/s beside the tips, those cells would be near Courant 13 at 1°.
- **Next v14 test:** spin up over 90° at 1°, then 60° at full speed (`RAMP_STEPS=90, RUN_DEG=150`). If it still
  runs away at θ ≈ 94°, try 30° at 0.25° (`FINE_STEPS=120 FINE_DEG=0.25`) before the 1° steps.
- **Cost if v14 runs at 1°:** 47 s per step on 18 cores and 8.65 M cells is about 98 core-seconds per million
  cells per step, against 190 for v2506 on Perlmutter. The hardware differs, so this is not a conclusion, but it is
  nowhere near 25 times; that factor belongs to the 0.17° rate. Each step takes about 130 `MeshPhi` iterations
  (plan item 3).
- **Gate stays open** until the ramp test and the like-for-like carbonio comparison (section 7). For now the ESI
  backend is the route for gap-dominated pumps until v14 runs at 1°; the two tests could reverse that within days.

## 3. Installation and environment

- **x86_64 (mochi, carbonio):** the openfoam.com apt package (`openfoam2506-default`), environment
  `/usr/lib/openfoam/openfoam2506/etc/bashrc`.
- **Perlmutter:** a user build (`~/software/openfoam/OpenFOAM-v2506`, GCC + cray-mpich, `srun --cpu-bind=cores`).
- **MIT ORCD (Engaging):** a user build (gcc 12.2.0, openmpi 5.0.8, `mpirun`).
- **arm64 (sparks):** the openfoam.com repository's `resolute` (Ubuntu 26.04) branch carries arm64 builds of
  openfoam2506 too; being installed on the sparks (17 Nov), same path as on x86_64. The machines run Ubuntu resolute with v14 from
  `http://dl.openfoam.org/ubuntu` (package `openfoam14`, `/opt/openfoam14`).
- **One OpenFOAM per shell** (the dev's point). Proposed `CFDT_FOAM_ESI` alongside `CFDT_FOAM`, each command
  sourcing only the backend's bashrc, as `foam.py` does today. `find_esi()` mirrors `find_foam()`: env var, else
  `/usr/lib/openfoam/openfoam2*/etc/bashrc`, else `~/OpenFOAM/OpenFOAM-v2*/etc/bashrc`. It checks
  `WM_PROJECT_VERSION` starts with `v2` and that `pimpleFoam` exists; it refuses an openfoam.org environment and
  vice versa. `machines()` reports both backends per machine.

## 4. Dialect mapping (what the case writer emits)

| Item | v14 path (today) | ESI v2506 (working reference) |
|---|---|---|
| Solver | `controlDict: solver incompressibleFluid;`, run `foamRun` | `application pimpleFoam;` (transient), `simpleFoam` (steady start); run `pimpleFoam -parallel` |
| Viscosity | `constant/physicalProperties` (`viscosityModel constant; nu`) | `constant/transportProperties`: `transportModel Newtonian; nu 1.003e-06;` |
| Turbulence | `constant/momentumTransport` | `constant/turbulenceProperties`: `simulationType RAS; RAS { RASModel kOmegaSST; turbulence on; }` (DES: `simulationType LES; LES { model kOmegaSSTDDES; delta cubeRootVol; cubeRootVolCoeffs { deltaCoeff 1; } }`) |
| Mesh motion | `mover { type solidBody; libs ("libpointMeshMovers.so"); cellZone; solidBodyMotionFunction { type rotatingMotion; origin; axis; omega N [rpm]; } }` | `dynamicFvMesh dynamicMotionSolverFvMesh; motionSolverLibs (fvMotionSolvers); motionSolver solidBody; cellZone rotor; solidBodyMotionFunction rotatingMotion; origin; axis; omega <rad/s>;` |
| MRF (steady start) | MRF as in cfdt today | `constant/MRFProperties`: `MRF1 { cellZone rotor; active yes; nonRotatingPatches (); origin; axis; omega <rad/s>; }`; delete it before the sliding run |
| Interface | `createBaffles` (plain patches) → `splitBaffles` → `createNonConformalCouples` on the decomposed case | `createBaffles -overwrite` with a cyclicAMI pair from the faceZone (`matchTolerance 1e-4; lowWeightCorrection 0.2; transform noOrdering;` each side naming the other), then `mergeOrSplitBaffles -split -overwrite`; decompose afterwards (no v14-style problem decomposing a coupled mesh) |
| Rotor walls | `movingWallVelocity` | `movingWallVelocity` (as v14); a wall of revolution outside the zone may use `rotatingWallVelocity` |
| Patch fields on the interface | from createBaffles with patch fields | the `cyclicAMI` types in `0/` come from createBaffles with patch fields, as in v14 |
| `fvSolution` solvers | cfdt's set | ESI form for `pcorr` (and `pcorrFinal`) and the `…Final` entries of every solved field |
| PIMPLE | cfdt's settings | `nOuterCorrectors 3; nCorrectors 2; nNonOrthogonalCorrectors 1; correctPhi yes; checkMeshCourantNo yes;` + `outerCorrectorResidualControl { p 1e-4; U 1e-5 }` |
| Time step | `cells_per_step` cap | fixed 1 degree (`deltaT = 1/(6·rpm)` s, `adjustTimeStep no`); `maxCo` informational only |
| Function objects | cfdt's set (incl. the `cfdtWhere` coded locator) | inside `controlDict` `functions {}`: `forces` (`patches (impeller impeller_back); rho rhoInf; rhoInf 998.2; CofR`), `pressure` (`mode total; result pTot`), `surfaceFieldValue` (ESI: `operation weightedAverage; weightField phi;` against v14's `operation average;` + `weightField`, the 2026.10.18.16 lesson; `sum` of phi for flows), `probes`, `fieldAverage`; `#include "controlDict.extra"` for per-stage additions. The coded locator is left out at first: v2506's coded function object API differs |
| Outputs | `postProcessing/<fo>/<t>/…` | the same layout; the moment is in `torque/<t>/moment.dat` (`total_z` column, or `pressure_z + viscous_z`), probes in `probes/<t>/p` and `U` |
| Sampling | `foamPostProcess` | `postProcess -func … -fields '(U)'`: without `-fields` v2506 samples nothing (Perlmutter, 12 Nov) |

## 5. Running, logs and checks

- **Commands:** blockMesh → surfaceFeatureExtract → decomposePar → `snappyHexMesh -parallel -overwrite` →
  reconstructParMesh → topoSet/subsetMesh (drop sealed pockets) → createBaffles → mergeOrSplitBaffles →
  checkMesh → decomposePar → [simpleFoam with MRF] → pimpleFoam. The reference is `scripts/01_mesh.sbatch` and
  `03_transient.sbatch` in the S50 case.
- **`foamDictionary -set`** rewrites the file with `#include`s expanded unless `-disableFunctionEntries` is given.
  That silently lost the movie's added function objects once (04_movie v1). Always pass it.
- **Log parsing differences:**
  - time lines read `Time = 0.1652866` (v14 prints a trailing `s`);
  - the outer loop prints `PIMPLE: iteration N` (v14: `PIMPLE: Iteration N`);
  - `Courant Number mean: … max: …`, `ExecutionTime = … s  ClockTime = … s` and the residual lines read the
    same in both;
  - AMI health lines replace the stitcher's: `AMI: Patch source sum(weights) min/max/average`. Check the
    average is about 1, the max well below 2 and the min not stuck at 0 except at snapped corners;
  - there is no `projected volume fraction` or `coverage`.
- **NaN:** v2506 traps floating-point exceptions on x86_64 (`FOAM_SIGFPE`), so a NaN aborts. On arm64 it would
  not, so keep cfdt's log watch (NaN, `CO_STOP`) for both backends.
- **Mass balance:** report inlet and outlet flow from `surfaceFieldValue` sums, as on v14.
- **Steady start before the split, as on v14.** cfdt runs the steady MRF start on the conformal mesh (faceZone not
  yet split), then builds the coupling. The ESI writer does the same: simpleFoam with MRF, then createBaffles to
  the cyclicAMI pair. That way the frozen-rotor flux loss across AMI never arises. It does arise when MRF runs on an
  already-split cyclicAMI mesh: the first Perlmutter attempt lost 11.4 of 20 m³/h. A frozen-rotor study on an
  already-split mesh therefore needs the interface joined first (`stitchMesh -perfect AMI1 AMI2`, then drop the
  empty patches; reference `engaging/06a_conformal_prep.sbatch`; after joining Qout/Qin was 1.000).

## 6. Physics and post-processing lessons (the backend must handle or warn)

Lessons 3 and 4 apply to both backends' results, not only ESI's. The chamber filter (4) is a reporting option that
says so in the results.

1. **Development time.** 8 revolutions at fixed flow before switching the inlet to a fixed total pressure. With 4,
   the flow drifted to 18.4-19.4 m³/h after the switch. Averages over the last 2 revolutions of each stage.
2. **The inlet condition sets the pulse amplitude.** With a fixed flow the eye-pressure swing is about 2.3 times
   larger than with a fixed inlet total pressure (19.7 against 8.7 kPa rms); head and power agree within 1 %.
   Pulsation results should state the inlet condition, and a pressure inlet is the default for pulsation.
3. **`fieldAverage` in the rotor is per cell**, so in the rotating frame: `pMean` there is not the mean at a fixed
   point. Mean and fluctuation at fixed points must come from samples (or probes) at fixed positions.
4. **Two-step pressure oscillation** in nearly closed chambers (the seal chamber behind the impeller): up to
   ±25 kPa alternating step to step. It is numerical; filter over 3 steps before reporting chamber pulsation.
   The vane-passing component (12.6 kPa as built) survives the filter.
5. **Turbulence model.** DES (kOmegaSSTDDES, backward/LUST) gave the same eye pulse as k-omega SST (8.6 against
   8.7 kPa rms) and the same head, 6 % less power. SST is the default; DES is an option for checking.
6. **Movies and maps.** `surfaces` with `cuttingPlane`, `surfaceFormat raw`, at every step for one vane passage
   (120 steps at 1 degree for 3 vanes). Then raster to a grid (nearest sample, gaps filled by the local sample
   spacing). Reference: `movie2/12_movie.sbatch`, `movie_raster.py`, `render_movies.py`.

## 7. Cost model (for the suggestion in section 1)

- **Like for like:** the carbonio coarse case on both backends (same mesh, same ranks, 1° per step) gives the
  ratio for the suggestion rule. v2506 is installed on carbonio and mochi (17 Nov); the v2506 run is the first half.
- **Stored per backend and machine:** measured core-seconds per million cells per step, as a bench entry, not one
  constant. Figures so far: v2506 about 190 (Perlmutter, 13.7 M cells, 512 ranks); v14 about 98 (spark2, 8.65 M
  cells, 18 ranks, 634's steps 3-10). Different hardware, so not yet comparable.
- **Suggestion rule:** suggest the ESI backend when the v14 estimate (steps from `cells_per_step`, cells, cost
  per cell) exceeds about 4 times the ESI estimate for the same study.

## 8. Tests

- **Golden files:** the writer's dictionaries for a rotor-in-cylinder case, compared with hand-checked files
  (`dynamicMeshDict`, `createBafflesDict`, `turbulenceProperties`, `transportProperties`, `controlDict`
  functions).
- **Mesh compatibility:** a v2506 `checkMesh` on a cfdt-written v14 mesh (section 1).
- **Parser:** both forms of the time line and of `PIMPLE: iteration N`, and the AMI weight lines, from captured
  v2506 logs (Perlmutter: `results/s50_sliding_v2506/log.pimpleFoam.61124833.head/.tail`)
  give time, Courant, ExecutionTime, AMI weights and residuals.
- **Rung 2 (g = 5) on both backends:** head and torque within 2 %, at the same probes.
- **S50 coarse on carbonio:** runs 2 revolutions; the cost line is recorded; same results format as v14.
- **Refusals:** each out-of-scope feature in section 1 is refused with a message naming the v14 alternative.

## 9. Reference files (all in `~/b3d_parts/s50/nersc/`)

| File | What |
|---|---|
| `s50_sliding_v2506.tar.gz` | as-built case: mesh, steady MRF start, sliding run, post-processing |
| `v2506_design.tar.gz` | design variants (`make_variant.sh`), `09_variant` (fixed flow then pressure inlet), `11_extend`, DES (`10a`, `10b`), `eye_stats.py` |
| `movie2.tar.gz`, `movie2_render_movies.py` | one-vane-passage movies |
| `orcd_mrf_conf.tar.gz` | frozen rotor on the sliding mesh: interface joined (`06a`), run with self-resubmission (`06b`) |
| `carbonio_v2506_coarse.tar.gz` | coarse case for local cost on the apt install |
