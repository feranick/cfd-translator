# sliding7 on Perlmutter, 2026-11-12: what the logs say

OpenFOAM 14, 13.1 M cells, 512 ranks (4 x 128), 1 degree a step (fixed step, 5.75e-5 s), kOmegaSST, cfdt 2026.11.12.4.
43 steps before it was cancelled. `log.foamRun.head` and `log.foamRun.tail` are the same file (the log is 165 KB).
Numbers below come from `cfd_translator.transient.step_profile` / `stitcher_stats` (2026.11.12.12) on that log.

## Where a step's time goes

| | first 4 steps | last 8 steps (diverged) |
|---|---|---|
| wall s per step (median) | 44 | 64 |
| U sweeps per step (Ux+Uy+Uz) | ~440 | ~3,000 (UFinal hits maxIter 1000) |
| p GAMG iterations per step | 46 | 48 |
| k + omega sweeps | 48 | 63 |
| couplings / time | 13.1 M / 0.81 s | 13.7 M / 0.77 s |

* About 2,550 more U sweeps cost about 20 s more: ~8 ms per component sweep on 26k cells a rank. The first steps'
  ~440 sweeps are then ~3.5 s, k/omega ~0.4 s, and the ~52 GAMG cycles (pcorr + 4 p solves) a few seconds:
  **the linear solves are ~6-7 s of the 44 s.** Loosening the final U/k/omega solve (2026.11.12.11) saves ~3 s.
* Wall distance: `Selecting patchDistMethod meshWave` is printed every step (OpenFOAM.org rebuilds wallDist when the
  mesh changes; no updateInterval there). In the frozen-rotor start the whole first iteration, meshWave included,
  took 1.4 s: **not the main cost.**
* The rest, **~35-38 s a step, is the mesh change**: the rotor moves, the couple is re-intersected (0.8 s for 13 M
  couplings), the stitched mesh gets ~13 M non-conformal faces (3.0 M at rest: 2 per interface face; 9 per face once
  turned) and everything that depends on the mesh is rebuilt (geometry, addressing, GAMG agglomeration, wall
  distance, old-time volumes, pcorr).
* The interface has 1,467,202 faces: the 0.125 mm refinement band (r 59.2-60.2 mm over 38.5 mm) alone is ~0.9 M.
  The rotor moves 1.04 mm (8 such faces) a step at r 59.7 mm. Fewer interface faces (0.25 mm: ~4x fewer) should cut
  the per-step mesh work roughly in proportion. To measure the split: 3 steps of the same mesh as a transient MRF
  (no motion) against 3 sliding steps.

## Why it diverged

* At rest the couple is clean: coverage 1/1/1, projected volume fraction (pvf) max 0.076.
* From the first step on and at every step: **pvf max 0.54-0.95** (warned above 0.3), **coverage min 0.51-0.78**,
  **coupled faces up to 42-60 degrees apart**. These stay the same after the flow has blown up: they are the mesh's,
  not the flow's.
* cfdt's own note on that mesh: the interface "strays from its cylinder: up to 1.13 of a cell and 64.7 degrees, e.g.
  at [9.9, 9.9, -82.1] mm (end at s0); [-7.1, 12.1, -82.1] mm (end at s0)": the zone's lower end (z -82.25) where it
  meets the hub/shaft at r ~14 mm. Faces there are not on a surface of revolution, so once turned they do not match:
  low coverage, large angles, volume error on thin cells.
* Courant max 62 at t = 0 (before any motion: tiny cells with the fixed 1-degree step), 21, 95, 2,656, 1.3e5 ...;
  Ux initial residual 0.17 at step 3. Which cells blew up first is not in the logs (no fields were written).
* 2026.11.12.11 estimates pvf per side before the run and stops above 1.0; 2026.11.12.10's divergence watch would have
  stopped this run at step 4 (Courant > 1e4).

## What to change in the study

1. The zone's caps on background-grid planes, as the hand-built ESI case had them (z -82.25 and -46.25 mm on 0.25 mm
   cells; AMI weights 0.98-1.16 with the hub crossing the lower cap): cfdt 2026.11.12.13 does this by itself
   (`mesh.snappy.caps_on_grid`). Check `interface_cells.off_surface` (max_deg should be well below 30) and
   `projected_volume_fraction` before solving (`get_case`).
2. The interface band at 0.25 mm instead of 0.125 mm (both sides alike, round about the axis).
3. The time step: Courant 62 at rest in a few cells; the hand-built case ran at ~25.
