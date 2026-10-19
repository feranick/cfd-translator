# cfd-translator

Turns a **CFD study** (a geometry, a fluid, a turbulence model, boundary conditions on chosen faces) into an [OpenFOAM](https://openfoam.org) case and checked results, deterministically. It includes a command line (`cfdt`), a Python API and an MCP server (`cfdt-mcp`), so LLM-based tools can set up flows as studies and get the numbers back with their checks.

It is the CFD counterpart of [build123d-translator](https://github.com/feranick/build123d-translator) and [fea-translator](https://github.com/feranick/fea-translator). An LLM writes the study (the flexible part). The translator computes everything else: the mesh and its polyMesh files, which faces form which patch, inlet profiles face by face, turbulence inlet values, the case dictionaries. A checker then confirms the result with closed-form and published benchmarks, mass and momentum balance on every run, solver convergence, and expected values. Geometries are build123d-translator goals or STEP files, so the same face selectors name CAD features, CFD boundary conditions and FEA loads. External-flow results include the wall forces as fea-translator loads.

Scope:
- **Steady flow**, laminar or k-ω SST (wall functions).
- Internal flow (the geometry is the fluid) or external flow (a domain box minus the part), 3D or 2D.
- Incompressible by default (OpenFOAM `incompressibleFluid`).
- With `"energy": true`, temperature is solved too (OpenFOAM `fluid`):
  - air is a compressible perfect gas, from low speed to transonic;
  - water keeps a constant density;
  - walls can be heated or cooled;
  - `gravity` adds buoyancy (natural convection).

## Install

On the machine that runs OpenFOAM (carbonio, Ubuntu):

```bash
# OpenFOAM 14 (openfoam.org), once
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update && sudo apt -y install openfoam14

# the package, in the shared venv (alias cad)
cd ~/Software && git clone https://github.com/feranick/cfd-translator.git    # or: cd cfd-translator && git pull
source ~/Software/build123d-translator/.venv/bin/activate
pip install -e "$HOME/Software/cfd-translator[mesh,mcp,test]"
```

On **Linux ARM (aarch64**, e.g. DGX Spark) PyPI has no Gmsh wheels, so the `mesh` extra skips it; use Ubuntu's (Gmsh 4.12 on 24.04) and link its Python module into the venv (the venv must use the system's Python version, 3.12 on 24.04):

```bash
sudo apt install -y python3-gmsh
ln -sf /usr/lib/python3/dist-packages/gmsh.py "$VIRTUAL_ENV/lib/python3.12/site-packages/gmsh.py"
python -c "import gmsh; print(gmsh.__version__)"
```

On ARM, check that openfoam.org has an arm64 package before installing (`apt-cache policy openfoam14` after adding the repository). Without OpenFOAM, the package still meshes, writes cases (`cfdt case`) and runs the 2D reference solver.

The translator finds OpenFOAM through `CFDT_FOAM=/opt/openfoam14/etc/bashrc` (or `PATH` when the shell is already sourced). Without that variable, it uses `foamRun` on the PATH, or else the newest `/opt/openfoam*/etc/bashrc`. OpenFOAM .org 11 or later is needed (`foamRun`, `solver incompressibleFluid`), and 14 is the target. Without OpenFOAM, the package's reference solver runs the 2D box studies (`--solver ref`).

```bash
cfdt status                 # versions; OpenFOAM, Gmsh, b3dt, feat found?
pytest -q                   # 50 tests: 47 run anywhere (19 with a fake SLURM), 3 need Gmsh or OpenFOAM
cfdt bench                  # benchmarks, pass/fail against known answers
cfdt bench cavity_re1000 pipe_turbulent chain_plate      # the slow and chained ones, by name
```

## Use

```bash
cfdt format > study_format.json          # the study format (what an LLM reads first)
cfdt validate examples/studies.json      # static checks with fixable messages
cfdt run examples/studies.json           # mesh, solve, check; results in ~/cfdt_runs/run_*/
cfdt case examples/studies.json pipe_laminar   # write the OpenFOAM case only (inspect, run by hand)
cfdt images ~/cfdt_runs/run_.../<study> [views.json]   # more pictures of a finished study, without solving again
```

Every run writes `<request>/<study>/case` (an ordinary OpenFOAM case: open `case.foam` in ParaView), `<study>_case.tgz` and `<study>.json`. Environment: `CFDT_OUT` (~/cfdt_runs), `CFDT_WORKERS` (2), `CFDT_TIMEOUT` (3600 s per study), `CFDT_SOLVER` (auto | foam | ref), `CFDT_PROCS` (MPI ranks; 1).

## Running on SLURM

With `CFDT_EXECUTOR=slurm`, each study becomes a SLURM job instead of running in the MCP server's own workers. This follows llm-translators' [docs/EXECUTION.md](https://github.com/feranick/llm-translators/blob/main/docs/EXECUTION.md). Studies from every chat then share the machine through one queue. A job keeps running when the conversation, the server or its ssh connection ends, and any later server process collects it.

**Settings:**

| variable | default | meaning |
|---|---|---|
| `CFDT_EXECUTOR` | `local` | `slurm`: studies become SLURM jobs |
| `CFDT_SLURM_MIN_MINUTES` | 0 | studies estimated below this run locally; cfdt estimates only reference-solver studies (0.1 min), so OpenFOAM studies always go to SLURM |
| `CFDT_SLURM_ARGS` | | extra `sbatch` arguments, e.g. `--mem=64G` |
| `CFDT_PROCS` | 1 | MPI ranks per study (`--ntasks`); a study's `solve.procs` overrides it |
| `CFDT_TIMEOUT` | 3600 | seconds per study; the job asks for ceil(timeout/60) + 10 minutes, so the translator stops itself first |
| `CFDT_MPIRUN` | `mpirun` | the MPI launcher; inside a job, OpenMPI's mpirun uses the allocation |
| `CFDT_CELLS_PER_RANK` | 5000 | a study runs on at most cells/5000 ranks, so small cases run serially |

A study's `"queue": true` or `false` forces SLURM or local; it never overrides `CFDT_EXECUTOR=local`.

**What a submission writes**, in `<request>/<name>/`:
- `job.sh`: the job name `cfdt:<request>:<name>`, `--chdir`, `--output=job.log`, `--ntasks`, `--ntasks-per-core=1` (whole cores: OpenMPI 5 refuses more ranks than cores, 'allocation-overload'), `--time`, `--export=ALL`, then `<python> -m cfd_translator.cli run-study <request> <name>`. `sbatch job.sh` reruns it by hand.
- `job.json`: the job id, submission time, ranks and time limit.

If `sbatch` fails, the study's result is `{"ok": false, "why": "could not submit: ..."}`. There is no silent local fallback.

**States** (`study_status` → `progress.<name>.state`, and `jobs()`):

| state | how it is known | what it reports |
|---|---|---|
| `queued` | `squeue` says PENDING | `reason`, `expected_start`, and `ahead` (pending jobs before it) |
| `running` | `squeue` says RUNNING | iteration, `eta_min_max`, `job_elapsed_min` |
| done | `<name>.json` exists | the result |
| `timed_out`, `out_of_memory`, `cancelled`, `failed` | the job left the queue without a result: `sacct`, or `job.log` when there is no accounting | `ok: false`, `why`, and what to change |

The four finished-failure states are written as the study's result, so they stay final. `study_status` makes one `squeue` call per request. It returns SLURM studies still pending after at most 15 s, since they don't need the server.

**Tools:**
- `jobs(limit, active)`: the newest requests with jobs, from any conversation, with the queue of cfdt jobs and the machine's CPUs.
- `cancel(request, names)`: `scancel`, plus a `cancel` flag file. In local mode the flag stops the OpenFOAM run, in whatever process runs it, within a second. The solver's whole process group is killed, mpirun and every rank, on cancel and on timeout.

**The run record.** Every study result, local or SLURM, has the `execution` block of EXECUTION.md: `{executor: slurm | local | by hand, slurm_job, procs_requested, procs_used, mpi: {launch, binding, cores, hwthreads} | null}`.
- `procs_used` can be lower than `procs_requested` (the cell cap), and `launch` is the command as run.
- The former `executor: {kind, job_id, procs}` block of `run-study` results stays for now.

`tool_status()` / `cfdt status` reports `executor: {kind, sbatch, min_minutes, procs, time_limit_min}`, which `llmt status` reads. The CLI has `cfdt jobs [--active]`, `cfdt cancel REQUEST [NAMES...]` and `cfdt run-study REQUEST NAME`.

## A study

```json
{"pipe_laminar": {
  "geometry": {"cylinder": [5, 200]},
  "fluid": "water",
  "model": "laminar",
  "bcs": [{"kind": "inlet", "faces": {"nz": -1}, "flow_rate_lpm": 0.5, "profile": "parabolic"},
          {"kind": "outlet", "faces": {"nz": 1}, "pressure": 0}],
  "mesh": {"size": 0.8, "extrude": 80},
  "expect": [{"quantity": "pressure_drop", "value": 6.80, "tol": 0.03}]}}
```

- **geometry**: `{"goal": <b3dt goal>}`, `{"step": path}`, `{"box": [lx, ly, lz]}`, `{"cylinder": [r, h]}` or `{"annulus": [r_in, r_out, h]}` (axis Z). Lengths are in mm, with Z up, the base on z = 0, and the part centred in X/Y (as in b3dt).
- **domain**: `internal` (default) means the geometry is the fluid. `{"kind": "external", "box": [x0, y0, z0, x1, y1, z1]}` means the fluid is that box minus the geometry.
- **2d**: `true` gives a one-cell-thick mesh along Z with `empty` front and back. The domain must be a prism along Z.
- **fluid**: `air`, `water`, `{"rho", "mu"}`, or a name with overrides. **model**: `laminar` or `kOmegaSST`.
- **energy** (heat transfer and compressible gas): `"energy": true` switches to OpenFOAM's `fluid` solver.
  - **Fluids:** `air` is a perfect gas (M 28.96, cp 1005, μ 1.825e-5, Pr 0.71), with density from p and T. `water` has constant density (cp 4182, Pr 7). Custom fluids are `{mu, cp, pr or k}` plus `molar_mass` for a gas or `rho` for a liquid.
  - **conditions:** `{p, T}` sets the operating pressure (Pa absolute, default 101325) and the initial or ambient temperature (°C, default 20). All study pressures stay gauge, relative to `conditions.p`; temperatures are in °C.
  - **gravity:** `[gx, gy, gz]` (m/s²) adds buoyancy. OpenFOAM then solves `p_rgh`, with `constant/g` and `pRef` written for it.
  - **Inlets:** a `temperature`, plus velocity or flow rate as before, or `mass_flow` (kg/s). Gases can instead use `total_pressure` (Pa gauge) with `total_temperature` (isentropic `totalPressure`/`totalTemperature`).
  - **Outlets and openings:** a backflow `temperature`.
  - **Walls:** `T` (°C), `q` (W/m² into the fluid), `power` (W), or `h` with `T_inf`, the same names as fea-translator's thermal bcs. Otherwise they're adiabatic.
  - **Transonic formulation:** switched on when the inlet's estimated Mach number exceeds 0.3.
  - **Not yet combined with energy:** rotating zones and the reference solver.
- **bcs**: the kinds are listed below. Faces use the selectors of b3dt and feat: `nx/ny/nz` (+ `at`), `pts`, `cyl_r` (+ `side`), `all`, lists for a union. External studies add `{"box": "xmin" … "zmax"}` for the domain sides. For external flow the normals are the solid part's, so one selector names the same face in b3dt, feat and here. A face is taken by at most one bc (`all` yields to the others). Faces no bc takes become the no-slip patch `walls`.

| kind | parameters | OpenFOAM (U / p) |
|---|---|---|
| inlet | `velocity` (number = normal in, or `[vx, vy, vz]`), or `flow_rate` (m3/s), or `flow_rate_lpm`; `profile` uniform or parabolic; `swirl {axis, center, rpm or tangential}`; `intensity`, `length_scale` | fixedValue (our values, face by face) / zeroGradient |
| outlet | `pressure` (Pa, 0) | inletOutlet / fixedValue p/ρ |
| opening | `pressure` (Pa, 0) | pressureInletOutletVelocity / fixedValue p/ρ |
| wall | `velocity` (moving wall; tangential part), or `rotation` (a rotating zone's name, or `{axis, center, rpm}`) | noSlip, fixedValue (sliding: tangential part; rotating: rigid ω × r), or MRFnoSlip (rotating with a zone) / zeroGradient |
| slip | | slip / zeroGradient |
| symmetry | | symmetry |

The parabolic profile is the fully developed laminar one for circular, rectangular (series solution) and 2D inlets. It is scaled so the flow rate is exact. `swirl` adds a tangential velocity face by face (solid body with `rpm`, uniform speed with `tangential` in m/s) without changing the flow rate. Use it for flow leaving an impeller that isn't modelled, or a swirler.

- **rotating** (pumps, fans, mixers): `[{name, axis, center, rpm, radius, span}]` defines rotating zones (MRF, frozen rotor). Each is a cylinder about `axis` through `center` (mm), with `span` [s0, s1] along the axis (default: all). The cells whose centres lie inside form the zone, as `topoSet cylinderToCell` would.
  - The impeller's faces are a wall with `"rotation": "<zone name>"`. They must lie inside the zone.
  - Those walls get OpenFOAM 14's `MRFnoSlip`, with the rigid velocity ω × r. Other walls stay at rest.
  - The zone's cylinder should contain only fluid and the rotating walls.

- **scalars** (mixing): `[{name, inlets: {<inlet bc>: value}, Sc (1) | diffusivity (m²/s), Sct (0.7)}]`. Passive tracers carried by the converged flow; inlets not listed get 0, walls have zero flux. Laminar runs with a `diffusivity` use it as given; otherwise D = ν/Sc (+ ν_t/Sct for kOmegaSST).
- **age** (residence time): `true` or `{Sc, Sct}`. The mean age of fluid a (s): div(φ a) − div(D ∇a) = 1, a = 0 at every inlet. Its flow-weighted outlet mean is the mean residence time; it equals V/Q when nothing diffuses back through the inlets, whatever the flow. Old fluid marks dead zones. (The full residence-time distribution needs a transient tracer pulse: not available.)
- **porous** (packed beds, catalyst monoliths, filters): `[{name, region, darcy, forchheimer, axes} | ergun | monolith]`.
  - `region`: `{box: [x0, y0, z0, x1, y1, z1]}`, `{cylinder: {center, axis, radius, span}}` or `{sphere: {center, radius}}` (mm). The cells whose centres lie inside form the zone (zones may not overlap).
  - Resistance per unit volume −(μ d + ½ ρ |U| f) U, with U the superficial velocity. `darcy` d (1/m²) and `forchheimer` f (1/m) are numbers or `[along e1, e2, e3]` with `axes: [[e1], [e2]]`.
  - `ergun: {particle_diameter (mm), porosity}`: d = 150 (1−ε)²/(ε³ d_p²), f = 3.5 (1−ε)/(ε³ d_p).
  - `monolith: {channel (mm), open_fraction, shape: round | square}` with `axes` (e1 = channel direction): laminar channels, d₁ = 32 (round) or 28.4 (square) /(ε d_h²); across, 1000 d₁.
  - Give the zone enough cells along the flow: OpenFOAM's collocated grid makes the velocity dip in the zone's first and last cells, so Δp is low by roughly a quarter of the fraction of zone cells on its faces (−2.6 % with 20 cells along a block, −0.4 % with 100); `face_cell_fraction` is reported, with a warning above 5 %.
  - Not modelled: heat in the solid, reactions on its surface.

- **mesh**:
  - A box geometry with `cells [nx, ny, nz]` (+ `grading`) gets a structured hex mesh, as blockMesh would make.
  - Otherwise Gmsh meshes the domain. `size` is in mm. `refine` takes `[{at, r, size}]`.
  - For prisms along Z, `extrude: n` sweeps a 2D mesh of the bottom face (quads by default) into n layers.
  - `layers: {n, first, growth}` adds boundary layers on the walls, in 2D or swept meshes only.
  - Plain 3D meshes are tetrahedra.
- **solve**: `iterations` (2000; 3000 for kOmegaSST), `residual` (1e-5), `relax {U, other, p}`, `procs` (MPI ranks; default `CFDT_PROCS` or 1), `average`; energy runs also `transonic`, `consistent`, `start`, `ramp`.
- **reference**: `velocity`, `area` (mm2), `length`, `cofr`, `drag_dir`, `lift_dir`, `p` (gauge zero).
- **probes**: `{name, at}` or `{name, line: {from, to, n}}`.
- **expect**: `[{quantity, value, tol | abs}]`, where `quantity` is a dotted path into the results.
- **images**: `true` for the default views, or a list of views. The PNGs are written after the solve.
  - Defaults: 2D studies get |U| with streamlines and the pressure on the plane. External 3D studies get top and side slices through the part with streamlines, plus the pressure on the part. Internal 3D studies get a mid slice and the wall pressure.
  - Slice: `{name, slice: {axis: x | y | z, at: mm}, field: u | ux | uy | uz | p (| T | mach | a tracer | age), streamlines, window: [a0, a1, b0, b1]}`. Tracers and the age are added to the default views.
  - Surface: `{name, surface: part | walls | <wall bc>, field: p | wall_shear, view: [elev, azim]}`.
  - Every run also saves its mesh and final fields (`state.npz`), so `render_images` / `cfdt images` can make other views later without solving again.

### Results

| key | what |
|---|---|
| `flow.<bc>.Q`, `Q_lpm`, `mdot`, `U_mean` | flow through inlets, outlets and openings (positive = into the domain) |
| `pressure.<bc>.p_mean`, `p_total` | area-mean static pressure; flow-weighted total pressure (Pa) |
| `pressure_drop`, `total_pressure_drop`, `head_loss_m` | mean inlet minus mean outlet (Pa); total-pressure loss / (ρ g) in m (use it for losses: static drops include velocity changes) |
| `flow.<bc>.angular_momentum_in` | moment-of-momentum flux into the domain about `reference.cofr` (N m) |
| `rotating.<zone>.torque_Nm`, `power_W` | torque by the fluid on the zone's rotating walls (about its axis) and power to the fluid |
| `pump.head_m`, `flow_m3h`, `power_W`, `efficiency` | with a rotating zone, an inlet and an outlet: total-head rise and hydraulic efficiency ρ g Q H / P |
| `forces.<wall bc>.fx/fy/fz`, `mx/my/mz`, `pressure`, `viscous` | force by the fluid on each wall patch (N), moments about `reference.cofr` (N m) |
| `forces.part` (external) / `forces.all_walls` (internal) | total over the part's faces / all walls |
| `coefficients.cd`, `cl`, `cm` | with `reference.velocity` and `area` (drag along the inlet velocity unless `drag_dir` is given) |
| `probes.<name>.ux/uy/uz/u/p` | values at points; lists for lines |
| `wall_shear.<bc>.mean/max`, `y_plus.<bc>` | Pa; y⁺ for kOmegaSST |
| `max_velocity` | largest cell velocity and where it is |
| `feat_loads` | external flow: one fea-translator `force` bc per part face (`pts` selector, N), ready to paste into a feat study of the same part |
| `images.<name>.path`, `what`, `min`, `max` | the PNGs (show them with the MCP tool `get_image`) |
| `heat.<wall bc>.Q_W`, `q_mean`, `T_mean_C` (+ `h`, `Nu` with `reference.T` and `length`) | energy: heat into the fluid through each wall (OpenFOAM's `wallHeatFlux`), wall temperatures |
| `flow.<bc>.mdot`, `T_mean_C`, `mach`; `max_mach`, `T_min_C`, `T_max_C`; `probes.<name>.T`, `mach` | energy: mass flow, flow-weighted temperature, Mach number |
| `feat_thermal` | energy, external flow: fea-translator thermal bcs per part face. `convection` gives h from the CFD heat and wall temperature, with `T_inf`; `heat_flux` is the heat into the part; plus the total |
| `scalars.<name>.outlets.<bc>.mean`, `min`, `max`, `mixing_index`; `volume_mean`; `probes.<name>.<scalar>` | tracers: flow-weighted outlet values; mixing index = 1 − CoV over the outlet |
| `age.mean_residence_time_s`, `V_over_Q_s`, `volume_mean`, `max_s`, `max_at`, `volume_fraction_older_than_2tau` | mean age: outlet mean (residence time), V/Q, dead zones |
| `porous.<zone>.pressure_drop_Pa`, `superficial_velocity`, `uniformity_index_out`, `force_on_fluid_N`, `darcy_1_m2`, `forchheimer_1_m` | porous zones: flow-weighted pressure upstream minus downstream of the zone, flow uniformity leaving it (1 − ½ mean\|u − ū\|/ū) |
| `checks` | `mass_balance`, `momentum_balance` (with MRF and porous terms), `angular_momentum_balance`, `energy_balance` (energy), `scalar_balance.<name>`, `age_balance`, `inlet_flow_error`, `wall_shear_vs_ours` |

Each study's result has **`reliable`**. It is false, with `unreliable_because`, when the solver did not converge, or the mass balance is off by more than 0.1%, the momentum balance by more than 3%, or a tracer's or the age's balance by more than 1%. Those numbers must not be reported as results.

It also has `selection` (what every bc took: face kind, area, normal, radius, side), `mesh` (cells, volume, non-orthogonality, closedness), `solve` (converged, iterations, last residuals, checkMesh), `warnings` and `files`.

**Interface** (read by other tools; kept stable): `forces.part.*`, `feat_loads.bcs`, `feat_loads.total`, `flow.<bc>.Q`, `pressure_drop`, `coefficients.*`.

## The checker

**Every run checks itself:**

- **Mass:** the boundary fluxes (`phi`) sum to zero.
- **Momentum over the whole boundary:**

  ```
  Σ_faces ( ρ U_f φ_f + (p_f − p_ref) S_f + F_viscous,f ) = 0
  ```

  - Wall terms use OpenFOAM's `wallShearStress`, open boundaries ν_eff·snGrad(U).
  - With OpenFOAM's conservative face fluxes this holds to convergence. Above 3% the run warns.
- **Angular momentum** (about `reference.cofr`): the same sum of moments. Finite volumes don't conserve it exactly, so it's reported, and the swirl and Taylor–Couette benchmarks bound it at 3%.
- **Rotating zones:** both balances include the frame's source, ρ Ω × U over the zone's cells.
- **Inlets:** the measured inlet flow must equal the prescribed one (this checks the case writing and the reading).
- **Wall shear:** our own wall shear (ν_eff (U_f − U_P)·Δ, tangential) is compared with OpenFOAM's.
- **Solver and mesh:** solver convergence, the residuals, and checkMesh, plus our own mesh checks: closed cells, positive volumes, outward boundary normals, upper-triangular face order, non-orthogonality. The schemes and non-orthogonal correctors follow the non-orthogonality.

**Benchmarks** (`cfdt bench`):

| case | reference | tolerance | reference solver | OpenFOAM |
|---|---|---|---|---|
| couette_2d | Q = UH/2, τ = μU/H | 0.5 % | pass (exact, 1e-16) | pass (exact: 0) |
| channel_2d | dp = 12 μ L U / H², wall force | 1 % | pass (0.13 %) | pass (0.13 %, 0.17 %) |
| cavity_re100 | Ghia, Ghia & Shin 1982, 30 centreline points | 0.02 U | pass (max 0.0086) | pass (max 0.0087) |
| pipe_poiseuille | dp = 8 μ L Q / (π R⁴), wall force (swept disk, Gmsh) | 2 % | — | pass (0.80 %, 0.88 %) |
| cylinder_2d1 | Schäfer & Turek 1996, Re 20: cD 5.58, cL 0.0106, Δp 0.1174 Pa | 2 % / 0.005 / 5 % | — | pass (cD 5.581: 0.03 %; Δp 0.11733: 0.06 %; cL 0.0078: low, inside 0.005) |
| cavity_re1000 (slow) | Ghia 1982 | 0.03 U | — | pass (max 0.0101; 16k cells, 28 s) |
| pipe_turbulent (slow) | Colebrook, Re 5×10⁴, kOmegaSST | 10 % | — | pass (3.1 %: dp 1591 vs 1642 Pa; residuals stall at 2e-4, 'not converged' after 3000 iterations, 184 s) |
| taylor_couette_2d | rotating inner cylinder: torque 4πμωR₁²R₂²/(R₂²−R₁²), u_t = Ar + B/r, angular momentum balance | 2 % / 2 % / 3 % | — | to run |
| taylor_couette_mrf | the same with a rotating zone (MRF) around the inner cylinder; power = Tω | 2 % | — | to run |
| swirl_pipe | solid-body swirl inlet: angular momentum in = ρUωπR⁴/2, flow unchanged, balance | 2 % / 1e-6 / 3 % | — | to run |
| pipe_poiseuille_mpi | pipe_poiseuille on 4 MPI ranks (`solve.procs`) | 2 % | — | to run |
| conduction_slab_2d | still air between plates at 60/20 °C: q = kΔT/H, linear T | 1 % / 0.2 K | — | to run |
| heated_channel_2d | water, both walls 100 W/m², fully developed: Nu = 8.235; outlet bulk T from the energy balance | 3 % / 0.05 K | — | to run |
| natural_cavity_ra1e4 | de Vahl Davis (1983) square cavity, Ra 10⁴, air: mean hot-wall Nu 2.243 | 2 % | — | to run |
| venturi_subsonic | 2D venturi (b3dt spline outline), slip walls: quasi-1D isentropic mass flow and throat pressure (M_t ≈ 0.60) | 2 % / 3 % | — | to run |
| pipe_air_low_mach | air with energy at Mach 0.0015: Hagen–Poiseuille | 2 % | — | to run |
| chain_thermal | b3dt block at 60 °C in 20 °C air → `feat_thermal` convection + heater with the CFD heat → copper block at 60 °C | 1 K | — | to run |
| chain_plate | b3dt plate → cfdt → feat: support reaction = −CFD force | 1 % | — | pass (reaction = −CFD force to 2e-5; feat 2026.10.16.8; 58 s) |
| mixing_layer_2d | two streams after a splitter plate, slip walls: c = ½ erfc(−y / (2√(Dx/U))); outlet mean ½ | 0.02 / 1e-3 | — | pass (5 probes within 0.002; outlet mean 0.50004; balance 9e-5; 18k cells, 16 s on 3 ranks) |
| tracer_two_streams | unequal streams: outlet mean = Q₁/(Q₁+Q₂) (conservation) | 1e-3 | — | pass (0.66667; balance 5e-6) |
| age_cavity_2d | channel with a side cavity, laminar: outlet mean age = V/Q (exact for any flow), age balance, dead zone | 1 % / 1e-3 | — | pass (13.987 s vs 14.0: −0.09 %; balance 1.5e-4; 29 % of the volume older than 2V/Q, max 115 s in the cavity corner) |
| porous_channel_1d | Darcy–Forchheimer block in a slip channel: Δp = L(μ d U + ½ ρ f U²), force, momentum balance | 1 % | — | pass (52.31 vs 52.5 Pa: −0.4 % with 100 cells along the block; −2.6 % with 20: the cell velocity dips at the block's faces on a collocated grid; momentum balance 3e-11) |
| ergun_bed | the same from particle size and porosity: Ergun's equation | 1 % | — | pass (501.1 vs 503.9 Pa: −0.55 %) |
| porous_anisotropic | block with axes at 45°: the momentum balance closes only if OpenFOAM's rotated resistance equals ours; transverse/axial force vs the 1D 0.82 | 2e-3 / 25 % | — | pass (balance 5.7e-4; ratio 0.71: the flow turns near the block's ends) |

The **reference solver** (`refsolver.py`) is a 2D steady Navier–Stokes solver: MAC grid, central differences, Picard iteration with a sparse direct solve. It reads the written case (polyMesh, `0/U`, `0/p`, `physicalProperties`), not the study, so a mistake in the translation shows up there too. A test changes ν in the case file and checks that the answer follows.

**Assumptions about OpenFOAM and Gmsh** (each with what would catch it; all confirmed by the OpenFOAM 14 runs on carbonio, 2026-10-16, including the Ghia Re 1000 values typed from the paper):

1. **Sign of `wallShearStress`.** It is R·n with n into the domain and R = −viscous stress (from the v14 source), so the viscous force on a wall is −ρ τ A. → channel_2d and pipe_poiseuille wall forces, and the momentum balance on every run.
2. **Missing boundary values.** Patches written without `value` (zeroGradient, noSlip, slip, symmetry) are evaluated as OpenFOAM does: cell value, zero, or the tangential part. → couette_2d and the momentum balance; the reader is tested on files in OpenFOAM's output style.
3. **What `phi` contains.** Its boundary values are outward volumetric fluxes (m³/s) at the last time. → `inlet_flow_error` on every run.
4. **Function-object fields.** `wallShearStress` and `yPlus` (in controlDict, `writeControl writeTime`) write their fields at the converged time, which `writeAndEnd` writes (from the v14 source). If they are missing, our snGrad estimate is used and `wall_shear_vs_ours` is absent. → channel_2d.
5. **Accepted input syntax.** Numeric dimension sets, `nu` without dimensions, and `simulationType laminar;` alone (Stokes, from the v14 source) are accepted. → any run.
7. **Energy (2026.10.18.1, checked against the OpenFOAM 14 sources, not yet run):**
   - `externalTemperature` takes `q`/`Q` as Function1s (`constant x`) and `h`/`Ta` for convection.
   - `prghPressure` takes `p` (absolute).
   - `totalPressure` takes `p0`, `psi`, `gamma`; `totalTemperature` takes `T0`, `gamma`.
   - `wallHeatFlux` is positive from the wall into the fluid.
   - `phi` is a mass flux.
   - Buoyancy is switched on by `0/p_rgh`.
   - Inferred, not seen in a tutorial: the `constant/pRef` file format, and `pressureInletOutletVelocity` paired with `prghPressure`. → conduction_slab_2d, natural_cavity_ra1e4, venturi_subsonic and the energy balance on every run.
6. **Gmsh details.** Boundary layers in 2D come from the `BoundaryLayer` field. After the side faces are removed, the bottom face's curves are matched by bounding box. → the Gmsh tests and cylinder_2d1.

**Limits:**

- Plain 3D meshes are tetrahedra without boundary layers, so turbulent external flow needs fine meshes. y⁺ is reported, with a warning outside 30–300.
- Steady flow only. No supersonic shocks (no density-based solver), no conjugate heat transfer inside solids (one-way to feat via `feat_thermal`), and no radiation.
- Rotating zones are frozen rotor: the impeller's position relative to the volute is fixed, and blade-passing effects are absent.

## How it works

1. `format.py` validates the study. Every error names the key and the fix.
2. `mesher.py` builds the fluid domain, which is one of:
   - a structured box;
   - Gmsh (OpenCASCADE): the goal/STEP/primitive, cut out of the domain box for external flow, meshed as tetrahedra or as a 2D mesh of the bottom face (quads, boundary layers) swept along Z.

   Boundary faces keep the tag of the CAD face they lie on.
3. `polymesh.py` turns cells into OpenFOAM's polyMesh. Faces are oriented outward from their cells (checked against centroids, so mesher node order never matters). Internal faces are matched and sorted by owner and neighbour. Boundary faces are grouped by patch. Geometry is computed with OpenFOAM's formulas.
4. `select.py` evaluates the selectors and assigns faces to patches. The `selection` report says what each bc took.
5. `case.py` writes the case:
   - inlet values face by face, scaled to the exact flow rate;
   - k = 1.5 (I U)² and ω = √k / (C_μ^¼ L);
   - kinematic pressures;
   - schemes chosen from the mesh's non-orthogonality.
6. `foam.py` runs `checkMesh` and `foamRun` (or MPI). `foamio.py` reads the logs and the last time's fields. `refsolver.py` is the alternative solver.
7. `post.py` computes results and invariants from the fields; `runner.py` ties one study together. `api.py`, `cli.py` and `mcp_server.py` are the interfaces.

## For LLM tools: the MCP server

Tools: `study_format`, `validate_studies`, `run_studies` (returns within ~45 s; long studies come back `pending` with `progress`: state, iteration, s/iteration, `eta_min_max`, job id), `study_status`, `jobs` (SLURM runs from any conversation), `cancel`, `get_case`, `get_image` (a PNG as an image Claude can see and show), `render_images` (other views of a finished study), `run_benchmarks`, `get_file` (base64, for callers on another machine). In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
"cfdt": {"command": "/usr/bin/ssh",
         "args": ["-T", "-o", "BatchMode=yes", "nicola@carbonio",
                  "/home/nicola/Software/build123d-translator/.venv/bin/cfdt-mcp"]}
```

Polling costs tokens: every `study_status` call is a model turn. The server's instructions therefore tell the model to report the expected finish time and stop when a run needs more than about 5 minutes, and to collect the result later.

`cfdt-mcp --http 8768 --host 0.0.0.0` serves streamable HTTP instead of stdio.

## Files

```
cfd_translator/
  format.py     study format, validation          case.py       the OpenFOAM case
  polymesh.py   cells -> polyMesh, box, extrusion  foam.py       runs OpenFOAM, reads fields
  mesher.py     Gmsh meshing                       foamio.py     readers (fields, logs, checkMesh)
  select.py     selectors, bc -> patches           post.py       results and invariants
  refsolver.py  2D MAC reference solver            runner.py     one study end to end
  render.py     images (matplotlib), saved state
  bench.py      benchmarks                         api.py, cli.py, mcp_server.py
  executor.py   SLURM jobs: submit, run-study, states, jobs, cancel
  average.py    per-iteration monitors, window averages, 'settled'
  scalars.py    tracers and the mean age            porous.py     porous zones
tests/test_cfd.py, test_slurm.py, test_average.py, test_scalars.py, test_porous.py
examples/studies.json      planning.md (session context)
```

## Versions

| version | changes |
|---|---|
| 2026.10.19.2 | The six new benchmarks on mochi (OpenFOAM 14 accepted every new keyword: scalarTransport, semiImplicitSource with `cellZone all`, porosityForce/fixedCoeff with axesRotation): all pass. Fixes from the first run: age_cavity_2d's inlet and outlet now chosen with `at` (`nx: -1` alone took the cavity's side wall); porous blocks need cells across them: the cell velocity dips in the zone's first and last cells (collocated grid), so Δp came out −2.6 % with 20 cells along the block and −0.4 % with 100. `porous.<zone>.face_cell_fraction` and a warning above 5 %. porous_anisotropic checks the momentum balance (which tests the rotation) instead of a 1D formula the flow does not follow. |
| 2026.10.19.1 | Tracers, mean age and porous zones (checked against the OpenFOAM 14 source, not yet run). `scalars` (passive tracers) and `age` (mean age of fluid, residence time): OpenFOAM's `scalarTransport` function object per field, solved every iteration without relaxation; the age's unit source via `semiImplicitSource` in `constant/fvModels` (`cellZone all`, `volumeMode specific`). Results `scalars.<name>` (outlet mean, mixing index), `age` (mean residence time, V/Q, dead zones), `checks.scalar_balance`, `checks.age_balance` (exact conservation, in the verdict), images of the fields. `porous` zones (box, cylinder, sphere; Darcy–Forchheimer, Ergun, monolith; anisotropic with axes): `porosityForce` with `fixedCoeff` (needs no viscosity field), the porous force in the momentum and angular momentum balances, results `porous.<zone>` (Δp, superficial velocity, uniformity index, force). Benchmarks mixing_layer_2d, tracer_two_streams, age_cavity_2d, porous_channel_1d, ergun_bed, porous_anisotropic. New modules scalars.py, porous.py. |
| 2026.10.18.16 | Monitors for OpenFOAM 14 (2026.10.18.15 stopped every study with an inlet and an outlet at startup: "weightedAverage is not in enumeration"). `cfdtP_<patch>`: `operation average` with `weightField phi` (the flow-weighted mean), `patch <name>` only (no select/regionType/name). `cfdtTotalP`: no `mode` (calcTotal, rho rhoInf, result total(p); in Pa). Checked on mochi with .15: the torque monitor works (Taylor-Couette MRF: averaged torque +0.06 % from analytic, settled). |
| 2026.10.18.15 | Iteration averaging for runs whose residuals stall (MRF frozen rotor). With a rotating zone (or `solve.average`), controlDict gets function objects that write every iteration: `forces` on each zone's rotating walls (torque about the axis) and flow-weighted `p` and `total(p)` (`pressure` function object) on every inlet and outlet. `average.py` reads postProcessing, checks each series against the final fields (last value within 5 %, units factor 1 or rho; otherwise the series is dropped with a warning; without total(p) it falls back to static p plus the final dynamic-pressure difference), and computes over the window (last 25 %, at least 500 iterations) mean, std, drift (second half vs first half) and scatter. `results.averaged` holds them; the means replace torque_Nm, power_W, pump.head_m/power_W/efficiency and total_pressure_drop (final iteration as *_last). `solve.settled`: head and power (or pressure drop and torque) drift < 1 % and scatter < 2 %; `reliable` now accepts converged or settled. `solve.average`: true/false/{window, min_iterations, drift, scatter}. foam.run clears postProcessing. Tests: tests/test_average.py (synthetic monitor files). Not yet with energy runs. The function-object syntax is untested on OpenFOAM 14: check postProcessing/cfdt* after the first run. |
| 2026.10.18.14 | The disconnect test of EXECUTION.md passed on mochi: SLURM job 2534 kept running while Claude was closed, and a new server found it with jobs() and collected it. Progress at the last iteration says 'finishing (fields, results, pictures)' and `next` gives an estimate (it said 'no estimate yet' when the ETA was 0). |
| 2026.10.18.13 | Variants of the venturi run on mochi with .12: from rest, from the estimate, with SIMPLE, with strong relaxation, with a ramp but without transonic: all diverged. A total-pressure ramp (400 iterations) with the transonic formulation ran 4000 iterations to residuals of 2e-5. Gas studies with a total-pressure inlet now default to that: `solve.ramp` 400 (at most half the iterations), `transonic` on, `solve.start` 'rest' ('estimate' is opt-in). Fixed: results stopped with KeyError 'Q' on a total-pressure inlet (inlet_flow_error). venturi_subsonic: 6000 iterations, residual 3e-5. |
| 2026.10.18.12 | From mochi with .11: heated_channel_2d converges now (PASS, 72 s). The venturi still stops at iteration 1: in OpenFOAM's MachNo function object (SIGFPE in sqrt), also at 100 and 1000 Pa total pressure and with SIMPLE; the same venturi with a 60 m/s velocity inlet runs. MachNo is no longer written (our results compute Mach from U and T). New `solve.start` ('estimate' | 'rest') and `solve.ramp` (a total-pressure inlet rises from the outlet pressure over that many iterations: uniformTotalPressure). |
| 2026.10.18.11 | From mochi with .10 (heated_channel and pipe_air_low_mach pass). An internal flow driven by one total-pressure inlet starts at half the isentropic inlet speed along the inlet's inward normal (`case.initial_U`): from rest the venturi diverged at iteration 2 even with transonic off, p relaxed and p limited. Small velocity components are scaled by mean |U_i| / mean |U| (was the max ratio, which left heated_channel's Uy just above the target). New `solve.consistent` (false: SIMPLE). |
| 2026.10.18.10 | OpenFOAM 14 refuses `pMinFactor`/`pMaxFactor` in PIMPLE (every open energy run failed on mochi with .9): the pressure limits are now `system/fvConstraints` (`limitPressure`, minFactor 0.5, maxFactor 2). |
| 2026.10.18.9 | From the mochi run of .8 (pipe_air_low_mach now 20 s instead of 78 min, heated_channel 85 s instead of 17 min). 'converged' judges each velocity component's residual relative to the peak speed (OpenFOAM normalises it by the component itself: heated channel's Uy, at most 1.3e-3 of the speed, stalled at 7e-3); `solve.residuals_scaled` shows the values used. With no heat put in, the energy balance is reported but not judged (note). Energy runs: `transonic` only above an estimated Mach 0.7 (the subsonic venturi diverged in 2 iterations with it) or when `solve.transonic` asks; gas above Mach 0.1 relaxes p by 0.3 (`solve.relax.p`); `pMinFactor 0.5`/`pMaxFactor 2`; maxIter 100 on U/h/k/omega. A SLURM job that left the queue without a job.log is `cancelled` (scancel while queued), not `failed`. |
| 2026.10.18.8 | From the mochi rerun's timings and checks (2026.10.18.7 was pinned before these). Energy runs cap the pressure solver at 50 GAMG sweeps: on absolute pressure it stalled at a round-off floor and took 1000 sweeps per iteration (heated channel: 0.26 s per iteration for 4000 cells). With no heat put in and Mach < 0.1 (`energy.isothermal`), h leaves OpenFOAM's residualControl and 'converged' ignores it while T varies less than u^2/cp (pipe_air_low_mach: h stalled at 1.5e-4 and ran all 3000 iterations, 78 min); the energy balance is scaled by the kinetic energy flux there. `angular_momentum_balance` judges each axis against the gross face moments about it (it used the largest net moment, ~0 in non-swirling flows: swirl_pipe 0.22, pipe_air 1.09 from noise). Benchmarks: taylor_couette_mrf's zone edge moved to 25 mm, off the probe at 30 mm (jagged edge: u_t 7.8 % high there); swirl_pipe's flow check uses the meshed inlet area. Residual targets: heated_channel 1e-5, natural_cavity 1e-5, pipe_air_low_mach 1e-5, conduction_slab 1e-7. |
| 2026.10.18.7 | From the energy benchmarks on mochi: a closed domain without gravity takes the fluid's density as constant (`rhoConst`; perfectGas diverged in p_rgh in 3 iterations, then enthalpy -> T failed), with relaxation 0.9 on h. With energy, `inlet_flow_error` compares volumetric flow (phi is a mass flux there; heated_channel showed a factor of rho), and `mass_balance` says its units are kg/s. A velocity component below 1e-4 of the peak speed counts as zero for 'converged' (heated channel: Uy's residual stalled at 7e-3). A `goal` geometry is checked by build123d-translator's validator when installed (venturi_subsonic's outline was not closed: fixed). Benchmark residual targets: heated_channel 1e-6, natural_cavity 1e-5, conduction_slab 1e-7. |
| 2026.10.18.6 | The run record of EXECUTION.md in every study result: `execution: {executor, slurm_job, procs_requested, procs_used, mpi: {launch, binding, cores, hwthreads}}`, also on failures (`could not submit`, timed_out ...); `run-study`'s former `executor` block kept as an alias. |
| 2026.10.18.5 | Closed energy domains (no inlet, outlet or opening) use the p_rgh formulation of OpenFOAM's closed-cavity tutorial (heRhoThermo, reference cell, relaxation 0.3), with zero gravity when none is given: on mochi the conduction slab diverged at iteration 4 with the open-flow settings (psiThermo, consistent). Also: 'converged' ignores a pressure that is uniform everywhere (spread below 1e-6 of ½ρu²) and, with energy, an enthalpy of a uniform temperature: their normalised residuals are round-off. Seen on mochi: Couette's p residual 0.05 with p ≈ 1e-19 Pa; the 18.4 change alone (Uy) could not fix it. |
| 2026.10.18.4 | A velocity component counts as zero everywhere (and its residual is ignored for 'converged') below 1e-6 of the largest velocity, not 1e-9: OpenFOAM's round-off in Couette's Uy is above 1e-9. On mochi (SLURM, OpenMPI 5): 47 tests pass; couette_2d (1 rank) and pipe_poiseuille_mpi (4 ranks in a job, 0.8 %) pass. |
| 2026.10.18.3 | From mochi (OpenMPI 5, 20 cores × 2 threads, `CFDT_PROCS=9`), where `mpirun -np 9` failed with 'allocation-overload':
- Jobs ask `--ntasks-per-core=1`, and `mpirun --map-by hwthread` is used when a job's cpuset has fewer cores than ranks.
- Ranks are capped at one per 5000 cells (`CFDT_CELLS_PER_RANK`), with a warning.
- The SLURM tests clear CFDT_PROCS and the other settings from the environment. |
| 2026.10.18.2 | SLURM execution (llm-translators docs/EXECUTION.md):
- `CFDT_EXECUTOR=slurm` submits each study as a job (`job.sh`, `job.json`), run by `cfdt run-study`.
- States queued / running / timed_out / out_of_memory / cancelled / failed, from squeue, sacct or job.log; finished failures are kept as results.
- `jobs` and `cancel` tools (and CLI); the `executor` key in `tool_status`; the study key `queue`; `CFDT_SLURM_MIN_MINUTES`, `CFDT_SLURM_ARGS`, `CFDT_MPIRUN`.
- Results are written atomically.
- OpenFOAM runs in its own process group, killed whole on timeout or cancel; local mode can now cancel too (a flag file).
- `cfdt bench` and `run_benchmarks` submit their cases as SLURM jobs too.
- 16 tests against a fake SLURM. |
| 2026.10.18.1 | Energy:
- `energy` (OpenFOAM `fluid`): compressible perfect-gas air up to transonic, constant-density water.
- `conditions`, and `gravity` (buoyancy, `p_rgh`).
- Thermal walls (`T`, `q`, `power`, `h` + `T_inf`).
- Inlets by temperature, mass flow, or total pressure/temperature.
- Results: heat, mass flow, Mach number, `T` probes, energy balance, `feat_thermal`.
- Images of `T`, Mach number, and wall heat flux `q`.
- Benchmarks: conduction_slab_2d, heated_channel_2d, natural_cavity_ra1e4, venturi_subsonic, pipe_air_low_mach, chain_thermal. |
| 2026.10.17.1 | Images: `images` in a study (slices of velocity or pressure with streamlines, wall-surface pressure or shear; solid parts masked); each run saves its mesh and fields (`state.npz`) for `render_images` / `cfdt images` without solving again; MCP `get_image` returns PNGs as images. matplotlib becomes a dependency. |
| 2026.10.16.6 | Rotating zones (MRF): `rotating`, walls with `rotation` (zone name or axis/rpm), torque, power, pump head and efficiency; `swirl` at inlets; `annulus` geometry; `solve.procs` (MPI); `reliable` verdict; `head_loss_m`; angular momentum flux and balance; pending studies report progress and ETA, and the MCP instructions stop long polling; benchmarks taylor_couette_2d, taylor_couette_mrf, swirl_pipe, pipe_poiseuille_mpi. |
| 2026.10.16.5 | Linux ARM (aarch64): the `mesh` extra skips Gmsh (no PyPI wheels); README: Ubuntu's python3-gmsh linked into the venv, as in fea-translator. |
| 2026.10.16.4 | First OpenFOAM 14 benchmark results (table above): all default cases pass; chain_plate needs fea-translator >= 2026.10.16.6 (checked) and uses a lighter mesh (it took 15 min on 230k cells); the wall-shear comparison warns only above 50 % (our estimate omits the dev2 term, ~20 % on tetrahedra; the momentum balance uses OpenFOAM's values). |
| 2026.10.16.3 | A velocity component that is zero everywhere (exact 2D solutions) no longer blocks "converged" (its normalised residual is round-off); `cfdt bench` prints one progress line per case on stderr. First OpenFOAM 14 runs on carbonio: mass balance ≤ 4e-10, momentum balance ≤ 4e-4, OpenFOAM vs our wall shear ≤ 1e-4 (cavity 5 %, lid corners). |
| 2026.10.16.2 | Boundary layers: the walls are found on the swept mesh itself (no separate Gmsh surface mesh); cylinder fit accepts coarse, short triangulations (facet normals tilted ~20°). |
| 2026.10.16.1 | First version: steady incompressible (laminar, kOmegaSST), internal/external, 2D/3D; own polyMesh writer; Gmsh tets and swept meshes with boundary layers; reference 2D solver; 8 benchmarks; feat loads; MCP server. Tested without OpenFOAM and Gmsh (15 tests; reference benchmarks pass). |

Author: Nicola Ferralis <ferralis@generalflash.com>
