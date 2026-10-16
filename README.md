# cfd-translator

Turns a **CFD study** (a geometry, a fluid, a turbulence model, boundary conditions on chosen faces) into an [OpenFOAM](https://openfoam.org) case and checked results, deterministically. It includes a command line (`cfdt`), a Python API and an MCP server (`cfdt-mcp`), so LLM-based tools can set up flows as studies and get the numbers back with their checks.

It is the CFD counterpart of [build123d-translator](https://github.com/feranick/build123d-translator) and [fea-translator](https://github.com/feranick/fea-translator). An LLM writes the study (the flexible part). The translator computes everything else: the mesh and its polyMesh files, which faces form which patch, inlet profiles face by face, turbulence inlet values, the case dictionaries. A checker then confirms the result with closed-form and published benchmarks, mass and momentum balance on every run, solver convergence, and expected values. Geometries are build123d-translator goals or STEP files, so the same face selectors name CAD features, CFD boundary conditions and FEA loads. External-flow results include the wall forces as fea-translator loads.

Scope (first version): **steady incompressible flow**, laminar or k-ω SST (wall functions), internal flow (the geometry is the fluid) or external flow (a domain box minus the part), 3D or 2D.

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
pytest -q                   # 19 tests: 16 run anywhere, 3 need Gmsh or OpenFOAM
cfdt bench                  # benchmarks, pass/fail against known answers
cfdt bench cavity_re1000 pipe_turbulent chain_plate      # the slow and chained ones, by name
```

## Use

```bash
cfdt format > study_format.json          # the study format (what an LLM reads first)
cfdt validate examples/studies.json      # static checks with fixable messages
cfdt run examples/studies.json           # mesh, solve, check; results in ~/cfdt_runs/run_*/
cfdt case examples/studies.json pipe_laminar   # write the OpenFOAM case only (inspect, run by hand)
```

Every run writes `<request>/<study>/case` (an ordinary OpenFOAM case: open `case.foam` in ParaView), `<study>_case.tgz` and `<study>.json`. Environment: `CFDT_OUT` (~/cfdt_runs), `CFDT_WORKERS` (2), `CFDT_TIMEOUT` (3600 s per study), `CFDT_SOLVER` (auto | foam | ref), `CFDT_PROCS` (MPI ranks; 1).

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

- **geometry**: `{"goal": <b3dt goal>}`, `{"step": path}`, `{"box": [lx, ly, lz]}` or `{"cylinder": [r, h]}` (axis Z). Lengths are in mm, with Z up, the base on z = 0, and the part centred in X/Y (as in b3dt).
- **domain**: `internal` (default) means the geometry is the fluid. `{"kind": "external", "box": [x0, y0, z0, x1, y1, z1]}` means the fluid is that box minus the geometry.
- **2d**: `true` gives a one-cell-thick mesh along Z with `empty` front and back. The domain must be a prism along Z.
- **fluid**: `air`, `water`, `{"rho", "mu"}`, or a name with overrides. **model**: `laminar` or `kOmegaSST`.
- **bcs**: the kinds are listed below. Faces use the selectors of b3dt and feat: `nx/ny/nz` (+ `at`), `pts`, `cyl_r` (+ `side`), `all`, lists for a union. External studies add `{"box": "xmin" … "zmax"}` for the domain sides. For external flow the normals are the solid part's, so one selector names the same face in b3dt, feat and here. A face is taken by at most one bc (`all` yields to the others). Faces no bc takes become the no-slip patch `walls`.

| kind | parameters | OpenFOAM (U / p) |
|---|---|---|
| inlet | `velocity` (number = normal in, or `[vx, vy, vz]`), or `flow_rate` (m3/s), or `flow_rate_lpm`; `profile` uniform or parabolic; `intensity`, `length_scale` | fixedValue (our values, face by face) / zeroGradient |
| outlet | `pressure` (Pa, 0) | inletOutlet / fixedValue p/ρ |
| opening | `pressure` (Pa, 0) | pressureInletOutletVelocity / fixedValue p/ρ |
| wall | `velocity` (moving wall; tangential part) | noSlip or fixedValue / zeroGradient |
| slip | | slip / zeroGradient |
| symmetry | | symmetry |

The parabolic profile is the fully developed laminar one for circular, rectangular (series solution) and 2D inlets. It is scaled so the flow rate is exact.

- **mesh**:
  - A box geometry with `cells [nx, ny, nz]` (+ `grading`) gets a structured hex mesh, as blockMesh would make.
  - Otherwise Gmsh meshes the domain. `size` is in mm. `refine` takes `[{at, r, size}]`.
  - For prisms along Z, `extrude: n` sweeps a 2D mesh of the bottom face (quads by default) into n layers.
  - `layers: {n, first, growth}` adds boundary layers on the walls, in 2D or swept meshes only.
  - Plain 3D meshes are tetrahedra.
- **solve**: `iterations` (2000; 3000 for kOmegaSST), `residual` (1e-5), `relax {U, other}`.
- **reference**: `velocity`, `area` (mm2), `length`, `cofr`, `drag_dir`, `lift_dir`, `p` (gauge zero).
- **probes**: `{name, at}` or `{name, line: {from, to, n}}`.
- **expect**: `[{quantity, value, tol | abs}]`, where `quantity` is a dotted path into the results.

### Results

| key | what |
|---|---|
| `flow.<bc>.Q`, `Q_lpm`, `mdot`, `U_mean` | flow through inlets, outlets and openings (positive = into the domain) |
| `pressure.<bc>.p_mean`, `p_total` | area-mean static pressure; flow-weighted total pressure (Pa) |
| `pressure_drop`, `total_pressure_drop` | mean inlet minus mean outlet (Pa) |
| `forces.<wall bc>.fx/fy/fz`, `mx/my/mz`, `pressure`, `viscous` | force by the fluid on each wall patch (N), moments about `reference.cofr` (N m) |
| `forces.part` (external) / `forces.all_walls` (internal) | total over the part's faces / all walls |
| `coefficients.cd`, `cl`, `cm` | with `reference.velocity` and `area` (drag along the inlet velocity unless `drag_dir` is given) |
| `probes.<name>.ux/uy/uz/u/p` | values at points; lists for lines |
| `wall_shear.<bc>.mean/max`, `y_plus.<bc>` | Pa; y⁺ for kOmegaSST |
| `max_velocity` | largest cell velocity and where it is |
| `feat_loads` | external flow: one fea-translator `force` bc per part face (`pts` selector, N), ready to paste into a feat study of the same part |
| `checks` | `mass_balance`, `momentum_balance`, `inlet_flow_error`, `wall_shear_vs_ours` |

Each study's result also has `selection` (what every bc took: face kind, area, normal, radius, side), `mesh` (cells, volume, non-orthogonality, closedness), `solve` (converged, iterations, last residuals, checkMesh), `warnings` and `files`.

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
| chain_plate | b3dt plate → cfdt → feat: support reaction = −CFD force | 1 % | — | pass (reaction = −CFD force to 2e-5; feat 2026.10.16.8; 58 s) |

The **reference solver** (`refsolver.py`) is a 2D steady Navier–Stokes solver: MAC grid, central differences, Picard iteration with a sparse direct solve. It reads the written case (polyMesh, `0/U`, `0/p`, `physicalProperties`), not the study, so a mistake in the translation shows up there too. A test changes ν in the case file and checks that the answer follows.

**Assumptions about OpenFOAM and Gmsh** (each with what would catch it; all confirmed by the OpenFOAM 14 runs on carbonio, 2026-10-16, including the Ghia Re 1000 values typed from the paper):

1. **Sign of `wallShearStress`.** It is R·n with n into the domain and R = −viscous stress (from the v14 source), so the viscous force on a wall is −ρ τ A. → channel_2d and pipe_poiseuille wall forces, and the momentum balance on every run.
2. **Missing boundary values.** Patches written without `value` (zeroGradient, noSlip, slip, symmetry) are evaluated as OpenFOAM does: cell value, zero, or the tangential part. → couette_2d and the momentum balance; the reader is tested on files in OpenFOAM's output style.
3. **What `phi` contains.** Its boundary values are outward volumetric fluxes (m³/s) at the last time. → `inlet_flow_error` on every run.
4. **Function-object fields.** `wallShearStress` and `yPlus` (in controlDict, `writeControl writeTime`) write their fields at the converged time, which `writeAndEnd` writes (from the v14 source). If they are missing, our snGrad estimate is used and `wall_shear_vs_ours` is absent. → channel_2d.
5. **Accepted input syntax.** Numeric dimension sets, `nu` without dimensions, and `simulationType laminar;` alone (Stokes, from the v14 source) are accepted. → any run.
6. **Gmsh details.** Boundary layers in 2D come from the `BoundaryLayer` field. After the side faces are removed, the bottom face's curves are matched by bounding box. → the Gmsh tests and cylinder_2d1.

**Limits:**

- Plain 3D meshes are tetrahedra without boundary layers, so turbulent external flow needs fine meshes. y⁺ is reported, with a warning outside 30–300.
- Steady flow only; no heat transfer; no rotating frames.
- Parallel runs (`CFDT_PROCS`) are untested.

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

Tools: `study_format`, `validate_studies`, `run_studies` (returns within ~45 s; long studies come back `pending`), `study_status`, `get_case`, `run_benchmarks`, `get_file` (base64, for callers on another machine). In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
"cfdt": {"command": "/usr/bin/ssh",
         "args": ["-T", "-o", "BatchMode=yes", "nicola@carbonio",
                  "/home/nicola/Software/build123d-translator/.venv/bin/cfdt-mcp"]}
```

`cfdt-mcp --http 8768 --host 0.0.0.0` serves streamable HTTP instead of stdio.

## Files

```
cfd_translator/
  format.py     study format, validation          case.py       the OpenFOAM case
  polymesh.py   cells -> polyMesh, box, extrusion  foam.py       runs OpenFOAM, reads fields
  mesher.py     Gmsh meshing                       foamio.py     readers (fields, logs, checkMesh)
  select.py     selectors, bc -> patches           post.py       results and invariants
  refsolver.py  2D MAC reference solver            runner.py     one study end to end
  bench.py      benchmarks                         api.py, cli.py, mcp_server.py
tests/test_cfd.py      examples/studies.json      planning.md (session context)
```

## Versions

| version | changes |
|---|---|
| 2026.10.16.5 | Linux ARM (aarch64): the `mesh` extra skips Gmsh (no PyPI wheels); README: Ubuntu's python3-gmsh linked into the venv, as in fea-translator. |
| 2026.10.16.4 | First OpenFOAM 14 benchmark results (table above): all default cases pass; chain_plate needs fea-translator >= 2026.10.16.6 (checked) and uses a lighter mesh (it took 15 min on 230k cells); the wall-shear comparison warns only above 50 % (our estimate omits the dev2 term, ~20 % on tetrahedra; the momentum balance uses OpenFOAM's values). |
| 2026.10.16.3 | A velocity component that is zero everywhere (exact 2D solutions) no longer blocks "converged" (its normalised residual is round-off); `cfdt bench` prints one progress line per case on stderr. First OpenFOAM 14 runs on carbonio: mass balance ≤ 4e-10, momentum balance ≤ 4e-4, OpenFOAM vs our wall shear ≤ 1e-4 (cavity 5 %, lid corners). |
| 2026.10.16.2 | Boundary layers: the walls are found on the swept mesh itself (no separate Gmsh surface mesh); cylinder fit accepts coarse, short triangulations (facet normals tilted ~20°). |
| 2026.10.16.1 | First version: steady incompressible (laminar, kOmegaSST), internal/external, 2D/3D; own polyMesh writer; Gmsh tets and swept meshes with boundary layers; reference 2D solver; 8 benchmarks; feat loads; MCP server. Tested without OpenFOAM and Gmsh (15 tests; reference benchmarks pass). |

Author: Nicola Ferralis <ferralis@generalflash.com>
