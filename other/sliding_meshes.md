# Rotating machines with a sliding mesh: ESI cyclicAMI against OpenFOAM 14 NCC

For people choosing how to run a pump, fan or mixer with a turning mesh, and for an AI that has to set one up: what
each method does, what it costs, the meshing rules, what happens at the start, how to validate, the pitfalls we hit,
and a decision guide. Written 15 Nov 2026 from the S50 investigation, 12-15 Nov 2026. The evidence base is the NCC
ladder record in `b3d_parts/s50/nersc/ncc_ladder/README.md`, with the scripts beside it: the v14 port
`s50_sliding_v14/` and the v2506 case `s50_sliding_v2506/`.

In cfd-translator, a sliding zone is `rotating[].motion: "sliding"`. It uses OpenFOAM 14 (openfoam.org) and its
non-conformal coupling (NCC). cyclicAMI is ESI's (openfoam.com), used here by hand for comparison and for production
runs on Perlmutter.

## 1. How the two couple the meshes

The rotor's cells turn as a solid body. At the zone's boundary they slide along the stationary cells, and the two
meshes must exchange mass and momentum across faces that no longer match.

| | ESI cyclicAMI (v2506) | OpenFOAM 14 NCC (`nonConformalCyclic`) |
|---|---|---|
| **How** | Interpolation weights: each face's value is a weighted sum over the faces it overlaps; the weights come from the overlap areas, recomputed as the rotor turns | Stitching: every step the overlapping parts of the two sides' faces are cut into new, exact coupled faces, and the mesh is rebuilt around them (fvMeshStitcher) |
| **Conservation** | Approximate: the weights of a face sum to about 1. Where they fall below `lowWeightCorrection` (0.2 here) the flux is corrected, but some is lost | Exact by construction. Volume the coupling cannot place goes to error patches; mesh fluxes are corrected to the swept volumes (`correctPhi`, `correctMeshPhi`, the `MeshPhi` solve) |
| **Sensitivity** | Tolerant of large slides per step: the S50 ran at 1 degree a step, 8.5 interface cells, with Courant numbers of about 25 in a few cells | Sensitive to the sliding rate: the rotor must pass no more than about 2 interface cells per step (section 2) |
| **What it reports** | `AMI: Patch ... sum(weights)` (the average should be about 1, the maximum well below 2) | coverage (min/average/max per face), the projected volume fraction (fvMeshStitcher warns above 0.3), mesh-flux errors |
| **Setup** | `createBaffles` with a cyclicAMI pair, `mergeOrSplitBaffles -split` | `createBaffles` (plain patches) -> `splitBaffles` -> `createNonConformalCouples -fields A B`, on the decomposed case. The second patch named is the neighbour; the volume error goes to the first (the owner), so the side with smaller or thinner cells should be the neighbour |

Both reproduce the same flow. At the same probe in the S50, from the same impulsive start, the vane-tip start-up jet
was -59.33 m/s on NCC against -55.10 on cyclicAMI. The stator cell next to it was -2.06 against -1.99 (Perlmutter
jobs 61305325 and 61358002).

## 2. Step limits and costs (measured)

**Cells per step:** the number of interface cells the rotor passes in one time step, along the rotation. It is the
arc at the interface's radius over the cells' size along it. On an O-grid it does not depend on the radius: 96 x 2^L
/ 360 per degree for 96 cells round at refinement level L (2.1 at level 3, 4.3 at 4, 8.5 at 5).

| case | cells per step | Courant max | result |
|---|---|---|---|
| propeller tutorial (v14, rung 1) shipped | adjustable, Co 2 | 2 | runs |
| propeller, fixed 1 degree | 0.8 | 7.6 | runs (also with the S50's PIMPLE settings) |
| propeller, maxCo 20 | 2.3 (2.8 degrees) | 20 | runs, 141 steps |
| propeller, S50's rate | 8.5 (10.3 degrees) | 190 | diverged at step 3 |
| S50 v14, fixed 1 degree | 8.5 | 22 plateau, then runaway | diverged within a few steps of full speed (every mesh and ramp) |
| S50 v14, maxCo 5 | about 1.5 (0.17 degrees) | 5 | runs, 149 steps from full speed |
| S50 v2506 cyclicAMI, fixed 1 degree | 8.5 | about 25 in a few cells | runs, 8 revolutions |
| S50 v14, long cells (`OGRID_ANISO`, run 634), fixed 1 degree | 2.13 | 43-70 | diverged from step 7 (start-up jet over the vane tops) |
| S50 v14, long cells, 90-degree ramp then full speed (run 636) | 2.13 | 22 plateau in the ramp, 24 -> 53 at full speed | diverged about 10 steps after full speed |
| rung 2 g = 5 v14 (job 639), fixed 1 degree | 1.74 | rising | diverged at step 28 |
| rung 2 g = 5 ESI cyclicAMI (cfd-translator `solve.backend: "esi"`, job 638), same mesh, fixed 1 degree | 1.74 | 6 typical, 11 max | runs, 3 revolutions; AMI weights 0.93-1.41, flux balance 1.4e-6 |

- **The limit (17 Nov):** the cells-per-step cap alone is not OpenFOAM 14's limit. On rung 2 at 1.74 cells per step
  and on the S50's long-cell mesh at 2.13, v14 diverged at a fixed 1 degree, while ESI's cyclicAMI ran rung 2's
  identical mesh at 1 degree with a Courant number of 6-11. v14 NCC needs a Courant-limited step (cfd-translator's
  default: Courant 2, at most `degrees_per_step`) on these meshes. The S50's flow Courant number at 1 degree comes from
  the leakage over the vane tops, which scales with the rotor's speed: a ramp delays the runaway (636), it does not
  cure it.
- **cfd-translator:** `solve.transient.cells_per_step` (default 2; null for no cap) caps the step once the mesh is
  known. The run-time estimate plans with the capped step.

**Cost per revolution on the S50 (Perlmutter, 4 nodes):** v14 NCC about 55-65 node-hours, v2506 cyclicAMI about 2.
The factor of about 25 is made of:
- **steps, about 6 x:** 0.17 degrees per step against 1;
- **cells, about 1.5 x:** 20.6 M on the O-grid mesh against 13.7 M;
- **cost per cell per step, about 2.7 x:** re-stitching every step, the `MeshPhi` solve, the locator.

Levers, largest first:
- **Interface cells long in the direction of rotation:** about 0.5 mm round, still 0.125 mm across the gap. This
  gives about 2 cells per degree, so about 1 degree a step at the cap, and about 16 x fewer interface faces.
  snappyHexMesh refines isotropically (openfoam.org has no directional refinement; ESI's `levelIncrement` does it),
  so the anisotropy has to come from the background mesh (`s50_sliding_v14/tools/ogrid.py aniso`, `OGRID_ANISO=1`).
- **The real step limit:** measure it rather than keep the safe 2.
- **Per-step overhead:** locator off; measure the stitching and `MeshPhi` share; try a looser `MeshPhi` tolerance.
- **Fewer cells away from the gap:** `COARSE=1`.
- **Shorter runs:** start variants from a developed flow (`init_from`, `mapFields`) rather than from rest.

## 3. Meshing rules, and the checks that enforce them

These are the rules the S50 taught. None of these defects caused its blow-up, which was the sliding rate, but each
was real and each has a check.

1. **The interface on grid rings and planes.** Snapping Cartesian cells onto a cylinder leaves slivers and a
   faceted interface whose two sides stop matching as the rotor turns.
   - Use a polar (O-grid) background whose cell faces lie on the interface, the rim and the bore. Make the ring radii
     slightly secant, because refinement puts new points on chords, not arcs.
   - End faces (caps and steps) go on z planes of the refined levels.
   - **Checks:** `interface_cells.off_surface` (faces off the cylinder, in cells and degrees);
     `interface_cells.end_faces` (the share of each end's faces on its plane, and the plane's offset; under 90 % is
     noted); the port's `precheck.json` `caps_planes`.
   - On a Cartesian background cfd-translator puts the end faces on grid planes; the radius needs the O-grid.
2. **No slivers within 2 cells of the interface.** Small wall features next to the interface are snapped onto, not
   meshed by whole cells: a bore-entry taper, a sloped floor at a bore's exit, rounded vane tips, a tongue. They
   left cells of 0.017-0.02 mm against 0.12 mm.
   - **Check:** `interface_cells.thin_wall_cells`. Cells at walls within 2 cells of the interface, thinner than 0.35
     of the interface's cells, are noted; below 0.2 the run is unreliable.
   - **Fixes:** move the interface or the zone's end clear of the feature, or defeature it for the sliding run (and
     check the effect with MRF).
3. **The zone's ends clear of the rotor's corners and of the casing.** The S50's original zone floor lay inside the
   bore's entry taper, 0.06 mm from the casing.
   - Raised to just under the back disc, with a step of smaller radius keeping the hub inside the zone, the stator
     side got 0.25 mm all round. In the generic rotor, the zone's floor sits 3 gap cells under the disc.
   - A step's own faces need the grid too (the O-grid's rings from r 24 for a step at r 25).
   - **The end faces' edges where they meet the cylinder next to a wall: fine cells along the axis.** On the S50's
     long-cell mesh (0.25 mm along the axis at the ends) snapping dragged the zone's top rim corner off its end face
     toward the wear plate (coupling coverage 0 to 1.95 at rest, diverged within 17 steps), although the flat part of
     the end face had 2 whole cells to the plate. The level-5 mesh (0.125 mm there) coupled cleanly. Finer
     background planes along the axis at the ends give that without raising the level.
     **Checks:** `interface_cells.stray_end_faces` (interface faces facing along the axis that lie on no end plane:
     the dragged edge; mesh 621 had 15 'planes' of a few faces between z -46.11 and -46.27, where most of its poor
     faces were) and `interface_cells.end_gap_cells` (whole cells from each end face to the wall along the axis, 2
     or more wanted).
4. **Whole cells across narrow gaps:** about 2 or more each side of the interface.
   - Prism layers do not fit: snappy thinned them to 6 micrometres in a 0.25 mm half-gap, because of its
     thickness-to-medial-axis rule.
   - **Checks:** `interface_cells.projected_volume_fraction` (an estimate of the stitcher's measure for each owner
     choice; OpenFOAM warns above 0.3, cfd-translator stops above 1); the size ratio across the interface.
5. **No refinement above the interface's level where it touches the interface.**
   - On an O-grid the cells passed per degree depend only on the refinement level: 96 x 2^L / 360 for 96 cells
     round, so 2.1 at level 3, 4.3 at level 4 and 8.5 at level 5, at any radius.
   - A refinement region (a gap band, a vane band, a sphere at a tongue) that brings any interface face to a higher
     level quietly halves the step once per extra level, because the cap follows the finest faces.
   - Keep every region that meets the interface at the interface's level, and the interface surface's own level
     too (snappyHexMesh's `refinementSurfaces` entry for the faceZone): the S50's long-cell meshes 621 and 624 left
     it at (4 4) and had every interface face at level 4 (4.3 cells per degree, 1.28 M faces) instead of 3. Get fine planes along the axis from the
     background (finer z layers at the zone's ends), not from an extra level.
   - **Check:** `interface_cells.cells_per_step`, and on the port `ogrid.py check`.
6. **One fine band over the whole interface, side and ends.** A study that takes an MRF study's refinements gets
   bands made for the frozen rotor's flow, not for the interface. S50 noboth with ESI (spark2, job 643): rc_noboth's
   bands left the side from z -82.25 to -77.1 and the floor in 1.25 mm cells while the rest of the side was 0.25
   mm; the interface had unmatched faces, a stray end face 90 degrees off and 141 one-cell floor faces, and the run
   diverged at its second sliding step (Courant 5.5e7). The hand-built sliding7 set (a 0.125 mm ring r 59.2-60.2
   over the whole span, a 0.25 mm band r 58.5-61, 0.25 mm cylinders over both ends) ran ESI at 1 degree on all
   five designs and DES.
   - **Check:** validation warns when mesh.refine leaves part of the zone's cylinder (side or ends, 2 of a region's
     cells either side) in cells more than 2x its finest band's (`sliding.refine_coverage`), for MRF zones too since
     2026.11.18.9.
   - **The same holds for the frozen rotor.** S50 noboth, MRF: `noboth_s1` (rc_noboth's MRF refinements, 8.26 M
     cells) gave 16.44 m and 2026 W; job 648 (sliding7's mesh block, 13.03 M cells) 15.69 m and 1671 +- 7 W, 51.1 %,
     torque -5.27 pressure / -0.25 viscous N m. The 355 W were all pressure torque, and faces 32 and 39 no longer
     stood out. The "power gap" against ORCD's frozen rotor (1794 W) was cfdt's old mesh, not the method; cfdt now
     reads 7 % below ORCD in power and 0.7 m lower in head.
   - **`rotating[].interface_refine: h`** (2026.11.18.9) adds sliding7's set scaled to h: a ring of h/2 cells at the
     radius +- 2 h, a band of h cells at the radius +- 5 h, both over the span and 4 h beyond it, and a cylinder of h
     cells over each end (4 h either side, out to the radius + 5 h). h = 0.25 on the S50 gives sliding7's set except
     its geometry-specific cylinder at the ledge (z -77.1 to -70.9). It is not the default: it costs cells (the S50
     went from 5.97 M to 13 M), and the right h depends on the pump.
7. **Surfaces of revolution inside or across the zone:** hub, shaft and back shroud may cross the zone's boundary
   as `rotatingWallVelocity` walls. Blades must lie inside it (`movingWallVelocity`). A hexagonal socket is not a
   surface of revolution.

## 4. Start-up behaviour

A sliding run usually starts from rest at full speed. In the first steps the vanes displace the fluid at once.
- **Beside the tips:** the fluid flows back over the tips at two to three times the tip speed: -55 to -59 m/s
  against +18 on the S50, in both NCC and cyclicAMI.
- **Narrow gaps:** an inflow switched on at once gives tens of m/s too (-31 m/s through the S50's 0.45 mm tongue
  gap, on a still mesh).
- **Decay:** it decays over about the first revolution. cyclicAMI's developed flow beside the tips is about +10 m/s.

What follows:
- **Averages:** only after the start-up revolutions. cfd-translator notes the start on every sliding run, and warns
  when `average_from` lies within the first revolution.
- **Not a coupling error:** this flow vanishes when the vanes do not displace fluid (a still mesh with spinning
  walls), and flips with the rotation.
- **Ramps:** a ramp in the rotation softens it but does not remove it.

## 5. Validation

1. **Sliding against MRF on the same mesh:** a near-axisymmetric case (a radial outlet) should give head and torque
   within a few % of the frozen rotor, both converged. A volute or tangential outlet adds rotor-stator interaction
   that MRF cannot capture; compare with cyclicAMI instead.
2. **v14 NCC against v2506 cyclicAMI at the same probes:** the same start (from rest, the same step rule), the same
   points. Compare the developed flow over whole revolutions.
3. **Flux balance through the coupling:** `solve.sliding.interface_balance`, the rotor and stator sides' sums and
   the error patches against the through-flow. Above 0.1 % it is noted; above 1 % the run is unreliable.
4. **When it fails, use the locator:** `solve.transient.locator: true`. A coded function object writes, for the
   first steps, the cells with the largest projected volume fraction and Courant number, each with:
   - its velocity and the mesh's;
   - every face's flux and the value on the face's other side (a wall's boundary value, the coupling's).

   The divergence message adds where the last steps' worst cells were.
5. **Rule out the coupling step by step:**
   - the wall's value: `U_other` on the rotor's wall faces should be omega r;
   - `correctMeshPhi no`;
   - a still mesh with spinning walls;
   - the rotation reversed;
   - the same start on cyclicAMI.

## 6. Pitfalls we hit

- **Snapped STL coordinates are single precision:** -0.079 m is -0.0790000036, so a face snapped onto -79 mm lies
  3.6 micrometres off it. Count faces on a plane within 1 micrometre (or by plane, rounded), not 1e-6 mm.
- **Refinement puts points on chords:** hexRef8 splits edges at their midpoints and adds face and cell centroids, so
  refined points of a polar background lie inside its arcs (0.03 mm at 96 cells round, r 60). Scale the ring radii
  so the refined lines straddle the cylinders (`ogrid.py check` simulates this at the refined level).
- **`foamDictionary` expands `#include`:** editing a dictionary with `-set` writes it back with its includes
  expanded. On v2506 `foamDictionary -set` on `controlDict` replaced `#include "controlDict.extra"` with its
  contents as read then, so the function objects added to `controlDict.extra` later (the movie's) silently never
  loaded. Use `foamDictionary -disableFunctionEntries`, or keep pristine `.orig` copies and copy them in at every
  start.
- **Per-run edits must be undone at every start:** a test option that changes `processor*/0/U` (the STATIC run's
  `rotatingWallVelocity` impeller) persisted into the next two runs and made them useless. Reset every edited entry
  (and check it) at the start of every run.
- **`set -o pipefail` with `grep ... | head`:** head closes the pipe, grep fails, and a finished job is marked
  FAILED. Add `|| true`.
- **A stale marker skips the coupling:** a `.ncc_done` kept after remeshing made the solve run the new mesh without
  the coupling. Delete it when remeshing, and check for the coupling's patches. Refuse a different owner order on an
  existing couple.
- **v2506 `postProcess` needs `-fields`:** a `sets` function object run by `postProcess` wrote nothing ("Cannot
  find registered field matching 1(U)") until the fields were named: `postProcess -fields '(U)' -time ...`. On a
  decomposed case reconstruct those fields first (`reconstructPar -fields '(U)' -time ...`).
- **OpenFOAM 14 API (coded function objects):**
  - `DimensionedField::field()` is private: use `static_cast<const scalarField&>(mesh.V())`;
  - `fvMesh` has no `findCell`: use the nearest cell centre, and keep only the rank with the nearest of all;
  - `fvMesh::phi()` aborts before the mesh has moved (at the t = 0 call): guard with `mesh.moving()`;
  - write a `std::string` with `.c_str()` (an OpenFOAM `string` is written quoted).
- **Perlmutter's system `python3` is 3.6:** `module load python`, or avoid `capture_output`/`text`.
- **Open MPI 5 on hyperthreaded nodes** (carbonio, mochi) refuses `mpirun -np N` in `sbatch -n N`: use
  `sbatch -n N -c 2` and `mpirun --bind-to none`. The sparks are not hyperthreaded: `-n N` alone.
- **Coverage below 1 once the rotor turns** (0.92-0.96) is the worst single face, not the share of the interface
  uncovered. The uncovered parts get the original patch's condition (`movingWallSlipVelocity`), which cannot drag
  the fluid.

## 7. Decision guide

- **First numbers, design trends, many variants:** MRF (frozen rotor). Check two or three rotor angles.
- **Performance, pulsations, rotor-stator interaction, production:**
  - **cyclicAMI (ESI v2506) at 1 degree a step: the production route for sliding studies** (cfd-translator
    `solve.backend: "esi"`, 2026.11.17; the same mesh, checks and steady start as OpenFOAM 14). It holds 1 degree a
    step where v14 does not (rung 2 job 638 against 639, the S50).
  - **OpenFOAM 14 NCC (cfd-translator's default sliding zone)** with its Courant-2 step: exact conservation, every
    cfd-translator machine, and validation. Keep it to 2 interface cells per step and mesh the gap by the rules
    above. Its cost against cyclicAMI on the same mesh and machine: rung 2 job 642 (spark1) measures it.
- **What to run first, on new hardware, a new OpenFOAM build or a new kind of machine:**
  1. **The propeller tutorial** (`ncc_ladder/rung1_propeller`): shipped, fixed 1 degree, the target's cells per
     step, and maxCo 20. It tells environment problems from coupling limits in an hour.
  2. **The generic rotor** (`ncc_ladder/rung2_rotor`): a rim in a close bore at the target's gap, with the target's
     features one at a time, sliding against MRF.
  3. **The production geometry:** meshed by the rules above, checked (`precheck.json` or cfd-translator's
     `interface_cells`) before a long run. Run with the locator for the first steps, then without.
- **Before trusting a sliding result:**
  - the averages start after the start-up revolutions and have settled;
  - mass and the coupling's flux balance;
  - no thin cells at walls next to the interface, and the end faces on their planes;
  - head and torque agree with MRF (or cyclicAMI) where they should.
