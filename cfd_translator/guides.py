"""guides.py — how-to guides for kinds of study, longer than study_format can hold (it stays under 12k characters
for local models). study_format's conventions point to them; the MCP tool `guide(topic)` returns one.
Version: 2026.11.12.3

The vehicle-aero guide condenses the Ahmed-body and Nissan Rogue studies on mochi (cfdt 2026.10.22-25,
OpenFOAM 14; examples/cfdt_dev_prompt_vehicle_aero.md and README's versions table).
"""
from __future__ import annotations

GUIDES = {
    "pumps": {
        "title": "Pumps, fans and mixers: MRF (frozen rotor) or a sliding mesh; cavitation and NPSH curves",
        "choose": [
            "MRF (rotating[].motion 'mrf', the default): steady, fast. The rotor is frozen where the geometry puts it, "
            "so head and torque depend on where the blades stand against the cutwater or the stator vanes; the "
            "rotor-stator interaction is lost. Good for trends and first designs; check two or three rotor angles.",
            "Sliding mesh (motion 'sliding'): the zone's cells turn with the rotor (OpenFOAM's non-conformal coupling "
            "on the zone's cylinder); transient. Head, torque, power and efficiency are averages over whole "
            "revolutions, with their scatter. For performance numbers, pressure pulsations, and when MRF results "
            "change with the rotor's angle.",
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
        ],
        "time": [
            "solve.transient.end_time in seconds: revolutions x 60 / rpm; about 10 revolutions (5 to settle after "
            "the steady start, 5 averaged). average_from about 4-5 revolutions in: the window is cut to whole "
            "revolutions ending at end_time (solve.sliding.averaged_revolutions).",
            "degrees_per_step (default 1) caps the rotation per time step; max_co (default 2) the Courant number. "
            "Cost: 360 steps per revolution or more, with two outer correctors each; long runs continue over SLURM "
            "jobs by themselves (max_jobs), or later with solve.transient.continue.",
            "Narrow gaps (an impeller's 0.5 mm clearances): snappyHexMesh leaves a few micrometre cells there, and a "
            "Courant limit then shrinks the step to a sliver (S50 at max_co 2: 0.02 deg a step, ~150,000 steps for 8 "
            "revolutions). Use max_co null: a fixed step of exactly degrees_per_step (1 deg ran stable on the S50 with "
            "Courant ~25 in a handful of cells, mean 0.12, ESI v2506 on Perlmutter). solve.time_step.courant (and "
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
