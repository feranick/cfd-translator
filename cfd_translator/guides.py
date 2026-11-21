"""guides.py — how-to guides for kinds of study, longer than study_format can hold (it stays under 12k characters
for local models). study_format's conventions point to them; the MCP tool `guide(topic)` returns one.
Version: 2026.11.21.1

The vehicle-aero guide condenses the Ahmed-body and Nissan Rogue studies on mochi (cfdt 2026.10.22-25,
OpenFOAM 14; examples/cfdt_dev_prompt_vehicle_aero.md and README's versions table).
"""
from __future__ import annotations

GUIDES = {
    "pumps": {
        "title": "Pumps, fans and mixers: MRF (frozen rotor) or a sliding mesh; cavitation and NPSH curves; air with slip",
        "choose": [
            "MRF (rotating[].motion 'mrf', the default): steady, fast. The rotor is frozen where the geometry puts it, "
            "so head and torque depend on where the blades stand against the cutwater or the stator vanes; the "
            "rotor-stator interaction is lost. Good for trends and first designs; check two or three rotor angles.",
            "Sliding mesh (motion 'sliding'): the zone's cells turn with the rotor (OpenFOAM's non-conformal coupling "
            "on the zone's cylinder); transient. Head, torque, power and efficiency are averages over whole "
            "revolutions, with their scatter. For performance numbers, pressure pulsations, and when MRF results "
            "change with the rotor's angle.",
            "Before a sliding study, read other/sliding_meshes.md in the cfd-translator repository: OpenFOAM 14's "
            "coupling against ESI cyclicAMI, measured step limits and costs, the meshing rules and their checks, the "
            "start-up jet, validation and pitfalls. In short: at most 2 interface cells passed per step "
            "(solve.transient.cells_per_step caps it; the S50 diverged at 8.5), no thin cells at walls within 2 cells "
            "of the interface (interface_cells.thin_wall_cells), end faces on grid planes (interface_cells.end_faces), "
            "averages after the first revolution (the start from rest drives tens of m/s beside blade tips and in "
            "narrow gaps), and validate against MRF on the same mesh. OpenFOAM 14 also needs its Courant-limited step "
            "(the default, Courant 2): at a fixed 1 degree it diverged on rung 2 (1.74 cells per step) and on the S50 "
            "where ESI cyclicAMI ran the same mesh stably; so it costs many more steps per revolution.",
            "solve.backend 'esi' (explicit, sliding only): the same mesh, checks and steady MRF start, then the sliding "
            "run in ESI OpenFOAM (v2506 pimpleFoam, a cyclicAMI pair on the zone's cylinder) at a fixed "
            "degrees_per_step (1 by default, no cells_per_step cap): the production route for sliding studies at 1 "
            "degree a step (rung 2: stable at Courant 6-11 where OpenFOAM 14 diverged). Results as on OpenFOAM 14, plus solve.sliding.ami (the AMI's weights). "
            "Cost about 150 core-seconds per million cells per step: the S50 on a 13.2 M-cell mesh takes 110 s a step on "
            "18 ranks, so 8 revolutions (2880 steps) take about 4 days on one spark or mochi (the run continues across "
            "jobs by itself); run half a revolution first to check the coupling. Cover the whole interface with one fine "
            "band (validation warns otherwise). Needs "
            "ESI OpenFOAM on the machine (machines(): status openfoam_esi). Not with energy, cavitation, porous zones, "
            "scalars, external flow, p_below or the locator (other/esi_v2506_backend.md).",
        ],
        "zone": [
            "A cylinder (center, axis, radius, span) around the rotor in open fluid: between the impeller's outer "
            "diameter and the cutwater or diffuser vanes (about mid-gap), its ends mid-way in the side gaps.",
            "Inside it only the rotor's walls (rotation: the zone's name); stationary walls only if they are "
            "surfaces of revolution about the axis (casing side discs, a stationary shaft sleeve). A blade tip "
            "touching cells outside the zone, or a casing wall touching cells inside it, stops the run with where "
            "it is: move the zone's ends into the gap and refine the gap (mesh.refine box) to 3 cells or more.",
            "The same zone works for MRF and for sliding: the sliding run's steady start (solve.transient.init) uses "
            "MRF.",
            "Mesh the zone's surface finely all along it, for MRF as well as sliding: interface_refine: <band size mm> on "
            "the zone adds a ring of half that size at its radius and a band over its side and both ends (the hand-built "
            "S50 sliding7 set: 0.125 mm ring, 0.25 mm band). Validation warns when mesh.refine leaves part of the zone's "
            "cylinder in coarser cells. S50 noboth, frozen rotor: 2026 W on an 8.26 M-cell mesh whose refinements left "
            "the zone's floor and lower side at 1.25 mm, 1671 +- 7 W on 13.03 M cells with sliding7's set (head 16.44 "
            "and 15.69 m): the power was 21 % high because of the mesh at the zone's surface, all of it pressure torque, "
            "not because of the frozen-rotor method (ORCD's frozen rotor: 1794 W, 16.37 m). Check power for mesh "
            "dependence before comparing designs.",
            "Round rotor walls (hub, shaft, back shroud) may reach outside the sliding zone if their bc holds nothing "
            "else: such a wall gets the rotating-wall velocity everywhere and its friction counts in the torque. "
            "Give them a wall bc of their own (rotation: the zone's name, or {axis, rpm} at the zone's speed); a bc "
            "that also holds blades must lie inside the zone.",
            "Sliding: the cells on the two sides of the zone's boundary must match in size, all the way round: "
            "refinement there must be round about the axis (cylinders on the axis; a band: inner_radius), since the rotor's cells pass "
            "every stator cell of their ring as it turns (boxes at the vane tips do not work). S50 blew up within a "
            "degree of rotation. solve.sliding.interface_cells reports the size ratio facing and around each ring, "
            "skewness and non-orthogonality; the warnings say it before the run.",
            "The cells along the zone's boundary matter: the interface's mesh-flux correction works on them "
            "(solve.sliding.interface_cells). If OpenFOAM crashes there, cfdt runs again with "
            "solve.transient.correct_mesh_phi false (some pressure noise at the interface).",
            "The zone's caps (the ends of its span) lie on background-grid planes by themselves (internal flow, axis "
            "along x, y or z: mesh.snappy.caps_on_grid), so they need no snapping; give the zone's band the cell size of "
            "the cells it meets (0.25 mm on S50, not 0.125: every step re-intersects the interface's faces). Before a "
            "long run, check interface_cells.projected_volume_fraction (OpenFOAM warns above 0.3; cfdt stops above 1) "
            "and off_surface.max_deg with get_case.",
        ],
        "time": [
            "solve.transient.end_time in seconds: revolutions x 60 / rpm; about 10 revolutions (5 to settle after "
            "the steady start, 5 averaged). average_from about 4-5 revolutions in: the window is cut to whole "
            "revolutions ending at end_time (solve.sliding.averaged_revolutions).",
            "degrees_per_step (default 1) caps the rotation per time step; max_co (default 2) the Courant number; "
            "cells_per_step (default 2) caps the step so that the rotor passes at most that many interface cells, "
            "once the mesh is known (OpenFOAM 14's coupling ran the S50 at 1.5 cells a step, 0.17 deg, and diverged "
            "at 8.5, 1 deg: NCC ladder 2026.11.14; solve.sliding.interface_cells.cells_per_step says what it chose). "
            "Cost: 360 steps per revolution or more (the S50 at 0.17 deg: about 2,100), with two outer correctors "
            "each; long runs continue over SLURM jobs by themselves (max_jobs), or later with "
            "solve.transient.continue.",
            "Screening impeller variants before two-phase sweeps: a steady MRF run with onset {c: 0.3, below: "
            "['vapour', {air_release: <saturation Pa>}], regions: ['eye', 'vane_leading_edges', 'tip_clearance']} "
            "gives per region the volume with p_eff = p_abs - c rho k below each threshold, the deepest point and the "
            "margin (results.onset; image field p_eff). conditions.p sets the absolute level. The regions come from "
            "the rotating zone and its walls (no boxes per variant). Calibrate c against a transient run's p_below "
            "at the same thresholds. Judge variants on each region's robust values (0.1 %/1 % volume quantiles, "
            "minimum and volumes without cells of skewness > 4 or within 2 cells of the zone's boundary); air "
            "release relative to the inlet's own saturation: {air_release: 'inlet', ratios: [0.5]}. reprocess(request, "
            "onset={...}) screens finished runs again with new regions or thresholds, without solving. Frozen-rotor "
            "heads oscillate: compare variants by pump.head_m with head_m_stderr (variants.differences), and give "
            "them solve.average {window: 0.5} and enough iterations for ten oscillations in the window.",
            "Pressure pulsations of a sliding run: solve.transient.free_flow true. The flow is held while the run "
            "settles (after {revolutions: 4}), then the inlet holds the mean total pressure of the last {revolutions: "
            "2} and the flow follows it (from a finished transient run it continues or init_froms: that run's averaged "
            "inlet total pressure, from the start). With the flow held the whole pulsation lands on the suction side "
            "(S50: head 11-21 m and eye pressure +-48 kPa held, +-19 kPa free, means unchanged). average_from at least 4 "
            "revolutions after the switch (the flow settles); results.free_flow gives the flow, the inlet and outlet "
            "pressures and the probes per revolution (probes also min_f / max_f: the extremes after [1 2 1]/4 over time "
            "steps, without the step-to-step jitter, S50 +-3 kPa; reprocess adds them to older runs).",
            "Pressure statistics of a sliding (or any transient) run: solve.transient.p_below [Pa absolute, e.g. "
            "2339, 20000, 50000: vapour pressure and air-release levels] writes, per cell, the fraction of the "
            "averaging window spent below each (image field p_below_<Pa>), summarised for the domain and each "
            "p_below_regions box or cylinder (results.pressure_statistics: volume-time below, mean and largest "
            "fraction, volume below for over 10 % of the time); absolute = conditions.p + the gauge pressure, so set "
            "conditions.p (or the outlet's pressure) to the pump's real level. p rms over the window is the image "
            "field p_rms. Steady and cavitation runs: image fields k and nut (the turbulent pressure fluctuation is "
            "about 0.2-0.4 rho k in the eye and the clearances). kOmegaSSTDES works with a sliding zone.",
            "Narrow gaps (an impeller's 0.5 mm clearances): snappyHexMesh leaves a few micrometre cells there, and a "
            "Courant limit then shrinks the step to a sliver (S50 at max_co 2: 0.02 deg a step, ~150,000 steps for 8 "
            "revolutions). Use max_co null: a fixed step of exactly degrees_per_step (1 deg ran stable on the S50 with "
            "Courant ~25 in a handful of cells, mean 0.12, ESI v2506's cyclicAMI on Perlmutter; OpenFOAM 14's "
            "coupling needs the cells_per_step cap). solve.time_step.courant (and "
            "courant_by_segment) report the maximum, its median over the steps and the mean.",
        ],
        "cavitation": [
            "cavitation: {npsh: [12, 8, 6, 5, 4, 3.5, 3, 2.5]} turns the pump study into an NPSH sweep: one study per "
            "value (<name>_npsh<value>), liquid and vapour (OpenFOAM incompressibleVoF with Schnerr-Sauer mass "
            "transfer), MRF frozen rotor, pseudo-steady (local time step), iterations 4000 by default.",
            "The study keeps its inlet flow_rate and outlet: the inlet then holds the absolute total pressure p_vapour "
            "+ rho g NPSH, the outlet draws the flow rate as a mass flow, so the liquid flow stays the study's while "
            "vapour forms (the outlet's pressure is not used). Points whose inlet liquid flow is more than 1 % off "
            "are left out of NPSH3 (cavitation.flows.liquid_flow_error). temperature (20 C) sets "
            "water's vapour pressure; another liquid needs p_vapour and vapour {rho, mu}.",
            "The highest NPSH must be free of vapour (its head is the reference); space the points closer where the "
            "head starts to fall. 'variants' in study_status gives npsh_curve: head and vapour volume per point and "
            "npsh3_m (3 % head drop, interpolated; head_drop changes the 3 %).",
            "Each point: results.cavitation (vapour_volume_mm3, vapour_region, walls with vapour, min_pressure_Pa, "
            "npsh_inlet_m from the flow-weighted inlet total pressure, sigma_tip) and pump.head_m, power_W averaged "
            "over the last quarter. images: a 'vapour' slice (field vapour) shows where it forms.",
            "Air coming out of solution (a self-priming pump whose head falls gradually from NPSHa ~7 m): "
            "cavitation.inlet_gas_fraction runs liquid + air entering with it: a fraction (0-0.2), one per NPSH "
            "point (a list as long as npsh), or {saturated_at: 101325} (the air a liquid saturated at that pressure "
            "cannot hold at each point's inlet pressure, Henry's law: about 0.02 (p_sat / p_inlet - 1) at 20 C, so "
            "~1 % at NPSH 7 m and ~3 % at 4 m; release < 1 for slower outgassing) (OpenFOAM "
            "compressibleVoF: water at constant density, air a perfect gas that expands where the pressure falls, "
            "isothermal), with the same NPSH sweep, mass-flow outlet and results. No vapour in the same run: the "
            "pressure is held at the vapour pressure, and cells reaching it are reported (cavitation.at_vapour_"
            "pressure). Compare its NPSH curve with the vapour-only one; results.cavitation gives gas_volume_mm3, "
            "gas_holdup (the air held in the pump against the inlet's fraction), gas_region, walls, and the air's "
            "mass in and out (flows.gas_in_kg_s, gas_out_kg_s).",
            "Mesh: the blades' leading edges and suction sides need fine cells (refine there, 0.2-0.5 mm on a "
            "small pump); vapour sheets thinner than a few cells are not resolved. Each NPSH point meshes again, "
            "except in a chain (below), where each point takes the previous point's mesh.",
            "Starts from earlier fields (solve.init_from: '<request>/<study>' of a finished run, or '<study>' of the same "
            "request, which then runs first): a clearance or rotor-angle variant from the baseline MRF run (U, p, k, "
            "omega, nut mapped with mapFields), the first NPSH point from the single-phase run (pressure not carried "
            "over), and with cavitation.chain true each NPSH point from the previous one (U, p_rgh, alpha, k, omega, "
            "nut; copied when the mesh is the same; the inlet pressure ramps down from the previous point's, the flow "
            "is held from the start). A chained point with the same geometry, mesh settings and bcs as the point it "
            "starts from takes that point's mesh (not meshed again) and its fields are copied, not mapped. Chained "
            "points run one after the other, not side by side: fewer iterations each (cavitation.iterations) is what "
            "pays for it.",
        ],
        "air_with_slip": [
            "two_phase {...} (OpenFOAM 14 multiphaseEuler, two-fluid): air and water each with their own velocity, so "
            "air slips through the water under gravity and, in an MRF zone, under the rotating frame's pressure "
            "gradient (what drives air to an impeller's eye and lets it separate in a discharge chamber). cavitation's "
            "inlet_gas_fraction (VoF, one velocity) cannot show either. MRF only (no sliding zone, no ESI); fluid water; "
            "laminar or kOmegaSST (water kOmegaSSTSato, air kOmegaSST); air a perfect gas.",
            "bubble_size {fixed: mm}: the answers depend on it; run every case at several sizes (0.3, 1, 3 mm) and say "
            "when a verdict changes with it ({iate} is reserved for a later version). gravity is required with a free "
            "surface or an opening (default [0, 0, -9.81] otherwise) and must match the pump's orientation: the S50 "
            "and S45+50 have a horizontal shaft (axis z, y up): [0, -9.81, 0].",
            "mode 'steady' (local time step, iterations) for frozen-rotor points at a fixed inlet gas fraction; "
            "'transient' (end_time, max_co) for air moving in time. inlet {gas_fraction | air_flow_lpm}; initial "
            "{gas_fraction, fill_level_mm (air above), velocity 'rotation'}; regions [{name, cylinder | box}] (or the study's) report "
            "their gas fraction and the phases' velocities (an eye region for gas locking).",
            "Results two_phase: air_mass_balance (judged: within 1 %), flows per open patch (water m3/s, air kg/s), "
            "regions, pump {head_m, torque_Nm, power_W}; with reference '<request>/<study>' (a single-phase run on "
            "the same mesh) the head and torque ratios, which at a gas fraction of 0 must be within 2 %. Compare "
            "two-phase heads as ratios to that single-phase head, not as absolute heads.",
            "Swirl gate: OpenFOAM 14's multiphaseEuler adds angular momentum to a rotating flow (a closed cylinder in "
            "solid-body rotation gains about 5 % in 1 s at 10 rad/s, with or without MRF, water alone too; single "
            "phase holds it; ESI's reacting Euler solvers lose 6-8 %; mrf_hold, 20 Nov 2026). Before a gas-fraction "
            "sweep give both runs (single phase and two-fluid at zero air, same mesh) the same top-level regions "
            "(eye, chamber pockets, reservoir): results.swirl / two_phase.swirl give each region's mean tangential "
            "velocity and angular momentum about the zone's axis, and two_phase.reference adds swirl_gate per region: "
            "gate 'relative' (u_theta within 2 %) or 'absolute' (swirl ratios u_theta / omega r within 0.03: "
            "stationary pockets and reservoirs with little swirl), or {relative|absolute: tol}; unset, absolute where "
            "the single-phase swirl ratio is below 0.25 (head and torque 2 %); or, for any of them, within twice the two runs' combined standard error (steady runs with regions write U snapshots over their averaging window for it: give both the same window length). Boxes as [x0..z1] or {min, max}. Run the gate at the sweep's flow and at a low flow (near "
            "shut-off the water recirculates as in priming); if it fails there, the priming question cannot use it.",
            "Validate first: run_benchmarks bubble_column_euler and rotating_cylinder_euler. Cost: a two-fluid step is "
            "several times a single-phase one, and the time step follows the fastest flow through the smallest cells "
            "(clearance jets): estimate before a pump run.",
        ],
        "reporting": [
            "Give pump.head_m, power_W and efficiency as revolution averages with results.averaged (std, drift, "
            "standard error); 'settled' needs drift < 2 % and standard error < 1 %.",
            "Compare with the MRF result at the same flow to show how much the frozen rotor was off.",
        ],
    },
    "vehicle_aero": {
        "title": "Vehicles and bluff-body aerodynamics (drag, lift, wakes)",
        "domain": [
            "Half model with symmetry on ymin (the part cut at y = 0: its cut face is left out automatically).",
            "About 3 L upstream, 5-6 L downstream, about 1.3 L to the side and above (L: body length); blockage "
            "2-3 % (Ahmed: a 2 % domain and a 1 % one differed by a few % in Cd). The Rogue used "
            "[-14000, 0, 0, 25000, 6000, 8000] mm for L = 4630 mm.",
            "Road vehicles: moving ground (wall velocity = the free stream) and rotating wheels (wall rotation). "
            "Wind-tunnel validation bodies: the floor as in the experiment (Ahmed: stationary).",
        ],
        "mesh": [
            "mesher snappy. Background cell (size) about L/10; surface [~L/400, ~L/200]; features true; gap 1.",
            "Refinement boxes around the body and the underbody, a wake box 1-1.5 L downstream, and a distance band "
            "of ~L/50 at the surface size.",
            "Check mesh convergence with mesh_scales [1, 0.707, 0.5] (and 0.354): 'variants' gives the change "
            "between the finest levels and the observed order. The Ahmed body (steady, moving ground) still changed "
            "~10 % per level until 1.6M cells.",
        ],
        "layers": [
            "y_plus 60-100 (wall functions); growth 1.2-1.3. Leave n out: cfdt then fits the stack into one of the "
            "finest surface cells (thicker stacks collapse: 13 layers on 10-20 mm cells kept 1.5).",
            "Accept forces only when mesh.layers.coverage_pct is above ~60 %; aim for > 90 % (Ahmed 96-99 %, the "
            "Rogue 62 %).",
        ],
        "geometry_pitfalls": [
            "Faces that touch tangentially make near-zero gaps that collapse the layers (a mirror stalk resting on a "
            "flat housing took the Rogue's coverage from 62 % to 13 %): sink joined parts into each other or meet at "
            "an angle; also rounded edges meeting flat faces.",
            "Avoid sliver faces from ruled lofts. A part touching a domain side only at an edge or point is warned "
            "about: let it cross the side by a millimetre or end on it with a flat face (tyre contact patches).",
            "Scanned or downloaded bodies: give the closed triangle mesh itself, geometry {stl: path (STL/OBJ/PLY), "
            "scale} or add_geometry(name, stl=path), not a faceted STEP (the Rogue's 67k triangles were a 170 MB "
            "STEP). Wheels may be separate closed bodies (external flow): name them as STL solids and select them "
            "with {group: 'wheel_front'}, or take any faces with {in_cyl: {axis [0, 1, 0], center, r, y: [y0, y1]}} "
            "or {in_box: [...]}; results add forces_by_solid.",
        ],
        "steady_rans": [
            "kOmegaSST; relax U 0.4, p 0.2, other 0.4; consistent false; start estimate; 3000-4000 iterations; "
            "average on (cd and cl means over the last 25 %).",
            "Ahmed with layers: 1-11 % over-prediction of drag on the L1-L3 meshes (stationary floor, 60 m/s); "
            "without layers on tetrahedra 11-30 %. With a moving ground at 40 m/s the series converged to 0.24 at "
            "25 deg (tunnel 0.285) and 0.261 at 35 deg (0.26): the 25 deg slant's partial separation is a known "
            "weakness of steady SST.",
            "Fully separated SUV or fastback wakes often do not settle (+-7 % on the Rogue). The 35 deg slant is "
            "bistable: the finest stationary-floor mesh flipped to attached flow (Cd 0.404).",
        ],
        "des": [
            "model kOmegaSSTDES with solve.transient {end_time, average_from, max_co 1, init 500}.",
            "Discard the first 6-10 body lengths of flow (L / U); then average over at least 15-20 body lengths and at "
            "least 3 periods of the slowest significant oscillation (results.averaged.frequency). The Rogue's lift "
            "had a 0.5 Hz oscillation (St ~ 0.05 on wheelbase): a 2 s window held one cycle, so lift did not settle; "
            "drag did. 0.3 s windows on the Ahmed body (11 body lengths) still drifted 5-8 %.",
            "DES puts the 25 and 35 deg slants in the tunnel's order where steady SST did not; Ahmed 35 deg L2: -8 % "
            "against the tunnel.",
            "Cost: 726k cells, 66,584 time steps, ~22 h on 9 procs over 3 SLURM jobs (queue included). To extend a "
            "finished run use solve.transient.continue: '<request>/<study>' with a later end_time (and a later "
            "average_from to start the averages again) instead of starting over.",
            "Variants that differ near one feature (a wheel spat): start them from the baseline's flow with "
            "solve.transient.init_from '<request>/<study>' (U, p, k, omega mapped with mapFields; no steady init) "
            "and a shorter run, e.g. end_time 2.5 with average_from 0.5 instead of 4 and 1.",
            "Compare runs over the same window: study_status(request, window=[t0, t1]) gives cd, cl and each wall "
            "bc's force (mean, std, standard error, drift) without downloading the result.",
        ],
        "reporting": [
            "Give the mean Cd with its min-max and drift, and say whether lift is settled (it needs longer windows "
            "than drag).",
            "Run the matching Ahmed benchmark before quoting absolute numbers: ahmed25_snappy / ahmed35_snappy "
            "(steady) or ahmed35_des (DES), run_benchmarks by name.",
            "Pictures: {surface: part, field: cp, quality: high} (lit, mirrored to the full body) and, for DES or "
            "separated wakes, {iso: {field: q, value: auto}, color: u} (vortices) or {iso: {field: cp_total, "
            "value: 0}} (the wake's losses); render_images makes them after the run.",
        ],
    },
}


def guide(topic: str | None = None) -> dict:
    if not topic or topic not in GUIDES:
        return {"topics": {k: v["title"] for k, v in GUIDES.items()},
                **({"error": f"no guide {topic!r}"} if topic else {})}
    return {"topic": topic, **GUIDES[topic]}
