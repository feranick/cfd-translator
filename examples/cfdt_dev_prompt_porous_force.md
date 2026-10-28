# cfdt: porous-zone forces are missing from the part's drag

## Problem

In external flow, `results.forces.part` and `coefficients` (cd, cl, cm) contain only the forces on the part's wall patches. A porous zone, such as a car radiator, removes momentum from the air. Its reaction acts on whatever holds the porous medium, which in practice is the part. cfdt reports this force only as `results.porous.<zone>.force_on_fluid_N` and leaves it out of the part's forces, coefficients and averages. Any model with a porous zone therefore under-reports drag.

## Evidence (cfdt 2026.10.25.6, kOmegaSSTDES, half car, 30 m/s, q = 541.8 Pa, A = 1.31934 m²)

| Run | forces.part fx (N) | porous force_on_fluid x (N) | cd reported | cd with porous reaction |
|---|---|---|---|---|
| rogue_v3c_des6_L1, spark1 `/home/feranick/cfdt_runs/run_20261025_181435_6bfd94` | 269.19 | −26.91 | 0.377 | 0.414 |
| rogue_v3c_des6_L2, spark1 `/home/feranick/cfdt_runs/run_20261026_110547_58387b` | 265.33 | −24.15 | 0.372 | 0.405 |
| rogue_v3e_des6_L1 (grille sealed, no porous zone), spark1 `/home/feranick/cfdt_runs/run_20261028_012112_2a8564` | 280.42 | none | 0.393 | 0.393 |

Two independent checks support the corrected values:
- **Pressure drop across the core:** the radiator's `pressure_drop_Pa` (about 150 Pa) times the core's flow area (`flow_in_m3s` / superficial velocity ≈ 0.756 / 3.5 ≈ 0.22 m²) gives about 32 N, consistent with the 27 N above.
- **Cooling drag:** with the reaction included, the open grille costs ΔCd ≈ 0.021 over the sealed one. That is in the usual range for road cars (0.01–0.03). Without it, the open grille appears to *lower* drag by 0.016, which is not physical.

## Requested changes

1. **Report the reaction:** add `results.forces.porous.<zone>` with fx, fy, fz, mx, my, mz about `reference.cofr` (the reaction on the part, = −force_on_fluid, with moments).
2. **Include it by default:** for external flow, include porous-zone reactions in `forces.part`, `coefficients` and `feat_loads`. Add a per-zone switch, `porous[].on_part`, default true; false is for a separate screen or a fan that isn't attached to the part.
3. **Keep the old number visible:** also report `coefficients.cd_walls_only` / `cl_walls_only`, so earlier runs can still be compared.
4. **Average it:** record the porous force in the per-step force history, so transient means, min/max, drift, standard error and frequency include it (`averaged.quantities.cd`).
5. **Check it:** add a momentum check: the inlet-minus-outlet momentum flux should equal the wall forces plus the porous forces. Flag the run when porous forces are present but missing from the part.
6. **Re-process finished runs:** add a way to recompute results from a finished case without solving again, for example `study_status(..., reprocess: true)` or a `postprocess(request, study)` tool. The three runs above, and any other run with a `porous` block, could then be corrected from their saved time-averaged fields (UMean in the porous cells).
7. **Document it:** in `study_format` conventions, add one sentence saying porous-zone forces count as forces on the part, with the switch above.
8. **Test it:** a porous slab in uniform flow, alone in the domain as a "part" with no walls. Its reported drag should equal Δp × A within 2%.

Context: I found this in the Rogue study while comparing an open-grille run (radiator as Darcy–Forchheimer zone, d = 1e7 m⁻², f = 300 m⁻¹ along x) with a sealed one. The Ahmed-body benchmarks have no porous zone and are not affected.
