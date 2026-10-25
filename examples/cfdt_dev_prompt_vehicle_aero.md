# cfdt: Ahmed-body benchmarks, a vehicle-aero guide, a README note, and four router/executor fixes

Context: a full vehicle study (2015 Nissan Rogue, half car, snappyHexMesh + prism layers, steady k-omega SST and
kOmegaSSTDES, cfdt 2026.10.22.10, OpenFOAM 14 on mochi) validated the method on the Ahmed body first. These
results should stay available to future chats, not just this one. Please do four things.

---

## 1. Add Ahmed-body benchmarks to `run_benchmarks`

Add three named, opt-in cases (not in the default fast set). Each case gets two checks:
- **physics**: compared with the wind tunnel, at a tolerance that reflects the method's known accuracy.
- **regression**: compared with the value this version produced, at a tight tolerance. This catches a change in the tool's own behaviour.

| Name | Model / mesh | Wind tunnel Cd | Physics tol | Value produced (cfdt 2026.10.22.10) | Regression tol | Cost (mochi, 9 procs) |
|---|---|---|---|---|---|---|
| `ahmed25_snappy` | kOmegaSST, steady, snappy L1 (448k cells) | 0.2875 | ±10% | Cd 0.289, Cl 0.353 | ±3% | ~50 min |
| `ahmed35_snappy` | kOmegaSST, steady, snappy L1 (450k cells) | 0.26 | ±15% | Cd 0.285, Cl 0.026 (settled) | ±3% | ~50 min |
| `ahmed35_des` | kOmegaSSTDES, snappy L2 (990k cells), 0.4 s, average from 0.15 s | 0.26 | ±12% | Cd 0.2387 (min 0.230, max 0.247), Cl −0.078, 22.7 Hz | ±4% | 3 SLURM jobs, ~1 day incl. queue |

Reference values:
- 25° slant: Cd 0.2875 from the [SimScale Ahmed validation](https://www.simscale.com/docs/validation-cases/aerodynamics-flow-around-the-ahmed-body/), quoting Ahmed et al. (1984).
- 35° slant: ~0.26, the figure commonly cited from Ahmed et al. (1984). It was not checked against the original paper. Please verify it if you can.
- Test case description: [ERCOFTAC AC1-05](https://www.kbwiki.ercoftac.org/w/index.php/Description_AC1-05).

Geometry: 1044 × 389 × 288 mm, R100 nose, 222 mm slant, 50 mm ground clearance. Half model split at y = 0, no stilts, stationary floor (as in the experiment), 60 m/s.

```json
{"ahmed25": {"features": [{"kind": "combine", "params": {"op": "add", "goal": {"features": [
   {"kind": "base_box", "params": {"w": 1044, "d": 389, "h": 288}},
   {"kind": "fillet_edges", "params": {"r": 100, "nx": -1, "ny": 0, "nz": 0}},
   {"kind": "profile_pocket", "params": {"nx": 0, "ny": 1, "nz": 0, "off": 0, "x": 0, "y": 0, "z": 0,
     "outline": {"start": [320.8, 288], "segs": [["L", 560, 176.46], ["L", 560, 400], ["L", 320.8, 400], ["L", 320.8, 288]]}}}]},
   "move": [0, 0, 50]}}, {"kind": "split", "params": {"nx": 0, "ny": 1, "nz": 0, "off": 0}}]},
 "ahmed35": "same, with the pocket outline start [340.15, 288] and first segment to [560, 134.06]"}
```

Steady study (both slants; L1 = these sizes):

```json
{"geometry": {"ref": "ahmed35"}, "domain": {"kind": "external", "box": [-3650, 0, 0, 6800, 1500, 1500]},
 "fluid": "air", "model": "kOmegaSST",
 "bcs": [{"kind": "inlet", "name": "inlet", "faces": {"box": "xmin"}, "velocity": 60, "intensity": 0.01, "length_scale": 45},
         {"kind": "outlet", "name": "outlet", "faces": {"box": "xmax"}, "pressure": 0},
         {"kind": "symmetry", "name": "sym", "faces": {"box": "ymin"}},
         {"kind": "slip", "name": "farfield", "faces": {"box": ["ymax", "zmax"]}},
         {"kind": "wall", "name": "ground", "faces": {"box": "zmin"}}],
 "mesh": {"mesher": "snappy", "size": 100, "surface": [3, 6], "features": true, "gap": 1,
          "refine": [{"box": [-700, 0, 0, 800, 350, 500], "size": 12}, {"box": [522, 0, 0, 1800, 350, 500], "size": 16},
                     {"distance": 20, "size": 6}],
          "layers": {"y_plus": 60, "n": 5, "growth": 1.2}},
 "solve": {"iterations": 4000, "procs": 9, "relax": {"U": 0.4, "p": 0.2, "other": 0.4}, "consistent": false,
           "start": "estimate", "average": true},
 "reference": {"velocity": 60, "area": 56016, "length": 1044, "cofr": [0, 0, 0], "drag_dir": [1, 0, 0], "lift_dir": [0, 0, 1]}}
```

DES study: same, except `"model": "kOmegaSSTDES"`, mesh sizes × 0.707 (size 70.7, surface [2.12, 4.24], features 2.12, refine 8.5 / 11.3 / 4.24), `"layers": {"y_plus": 60, "n": 3, "growth": 1.2}`, and
`"solve": {"procs": 9, "transient": {"end_time": 0.4, "average_from": 0.15, "max_co": 1, "init": 500, "max_jobs": 10}}`.

Also record these results from the same runs (useful for the guide, not as benchmarks):

| Case | L1 (scale 1) | L2 (0.707) | L3 (0.5) |
|---|---|---|---|
| 25°, steady | 0.289 (Cl scatter 5.7%, not settled) | 0.311 | 0.295 |
| 35°, steady | 0.285 (settled) | 0.288 (drift 3.4%, not settled) | **0.404**: flow reattached on the slant, the wrong state for 35° |

Nice to have: an `ahmed25_des` case, once someone has run it.

---

## 2. Add a "Vehicles and bluff-body aero" section to `study_format` conventions

Every chat reads `study_format` first, so these lessons belong there. Suggested content, condensed from the runs above and the Rogue study:

- **Domain:**
  - Half model with symmetry on ymin.
  - About 3 L upstream, 5–6 L downstream, about 1.3 L to the side and above.
  - Blockage 2–3%. The Rogue used [-14000, 0, 0, 25000, 6000, 8000] for L = 4630 mm.
  - Road vehicles: moving ground and rotating wheels (wall `rotation`). Wind-tunnel validation bodies: stationary floor, as in the experiment.
- **Mesh (snappy):**
  - Background cell about L/10.
  - Surface cells [~L/400, ~L/200].
  - `features` on, `gap` 1.
  - Refinement boxes around the body, the underbody, and a wake box reaching 1–1.5 L downstream.
  - A distance band of ~L/50 at the surface size.
- **Prism layers:**
  - `y_plus` 60–100, `n` 5, `growth` 1.2–1.3.
  - Keep the stack within about 1–2 surface cells.
  - Accept forces only when `coverage_pct` is above ~60%; aim for >90%. The Ahmed body reached 96–99%, the Rogue 62%.
- **Geometry pitfalls:**
  - Faces that touch tangentially create near-zero gaps. Examples: a cylinder stalk resting on a flat housing, rounded edges meeting flat faces.
  - These gaps collapse the layers. On the Rogue, a mirror stalk took coverage from 62% down to 13%. Sink the joined parts into each other, or make them meet at an angle.
  - Avoid sliver faces from ruled lofts.
- **Steady RANS:**
  - Settings: relax U 0.4, p 0.2, other 0.4; consistent false; start estimate; 3000–4000 iterations; average on.
  - Expect 1–11% over-prediction of drag on the Ahmed body with layers. Without layers on tetrahedral meshes it was 11–30%.
  - Steady results on fully separated SUV or fastback wakes often do not settle: ±7% swings on the Rogue.
  - At 35° the solver is bistable: the finest Ahmed mesh flipped to attached flow (Cd 0.404).
- **DES (kOmegaSSTDES):**
  - Settings: `max_co` 1, `init` 500.
  - Discard the first 6–10 body lengths of flow (L / U), then average over at least 15–20 body lengths, and over at least 3 periods of the slowest significant oscillation. The Rogue's lift had a 0.5 Hz oscillation (St ≈ 0.05 on wheelbase), so a 2 s window covered only one cycle and lift did not settle; drag did.
  - On Ahmed 35° (L2), DES gave −8% versus the tunnel.
  - Cost: 726k cells and 66,584 time steps took about 22 h wall on 9 procs across 3 SLURM jobs, queueing included.
- **Reporting:**
  - Give the mean Cd with its min–max and drift, and say whether lift is settled. Lift needs longer windows than drag.
  - Run the Ahmed benchmark that matches the case (steady or DES) before quoting absolute numbers.

---

## 3. README note: optional personal skill

Add a short README paragraph:

> **Reusing a workflow across chats.** The benchmarks (`run_benchmarks`) and the conventions in `study_format` are the shared record of what works. If you repeat a multi-tool workflow often (for example photos → build123d-translator CAD → cfdt mesh check → Ahmed benchmark → DES), you can also save it as a personal Claude skill (e.g. "cfdt vehicle aero"). The skill holds the steps and your own defaults, and works in any of your chats. Keep the physics and accuracy numbers in the repo, so there is one source of truth, and have the skill point to them.

---

## 4. Router / executor issues found in the same session

1. **spark1 drops every SLURM job.** Jobs 33 (18 procs, `mem` 16G) and 35 (10 procs, default mem) left the queue within seconds, with no job.log. The spark cluster has two nodes (spark-0808, spark-5da5) and `shared_home: false`, so a job scheduled on the other node cannot find its run folder. spark2 was unreachable at the time. Suggested fix: pin jobs to the machine's own node (`-w <node>` or a per-machine partition). Also report "dropped before start" with the SLURM reason (`sacct`).
2. **mochi cannot take more than ~20 procs.** 36 procs was refused: "Requested node configuration is not available". The executor seems to give each rank 2 hardware threads (a 9-proc run showed cores 9 / hwthreads 18). Please put the real maximum in `executor.caps.procs`, and have validation reject larger requests.
3. **Mixed `run.host` in one request.** A request with studies for mochi and spark1 was sent whole to mochi; the spark1 study queued there. The router should either split a request by `run.host` or refuse a mixed one.
4. **Feature: continue a finished transient run.** Add `solve.transient.continue: "<request>/<study>"`: restart from the latest time, extend `end_time`, and either keep the existing averages or restart them from a given time. Today, extending the Rogue DES from 3 s to 6 s means a fresh run of about 30 h instead of about 15 h of extra time.
