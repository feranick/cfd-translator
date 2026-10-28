"""guides.py — how-to guides for kinds of study, longer than study_format can hold (it stays under 12k characters
for local models). study_format's conventions point to them; the MCP tool `guide(topic)` returns one.
Version: 2026.10.28.1

The vehicle-aero guide condenses the Ahmed-body and Nissan Rogue studies on mochi (cfdt 2026.10.22-25,
OpenFOAM 14; examples/cfdt_dev_prompt_vehicle_aero.md and README's versions table).
"""
from __future__ import annotations

GUIDES = {
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
