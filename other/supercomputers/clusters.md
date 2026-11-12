# Running cfdt on clusters its servers cannot reach (NERSC Perlmutter, MIT Engaging)

cfdt runs on the cluster itself, from its command line. It meshes, writes the case, runs the steady start, the
sliding conversion and the transient segments, and judges the results, in SLURM jobs, exactly as it does on carbonio,
mochi and the sparks. Only the router and the MCP server stay at home. When the runs are done, `cfdt pack` makes a
small archive of the results; `import_results` (or `cfdt import`) turns it into a request on a cfdt machine, where
`study_status`, `window`, `variants` and `render_images` work as usual.

Jobs may span several nodes (llmt-runtime 2026.11.11.1, cfdt 2026.11.11.2): with `CFDT_SLURM_NODES` a study whose
ranks exceed a node's cores runs on `--nodes=K` with the ranks balanced (`--ntasks-per-node`). Memory is always per
node (SLURM's `--mem`): `run.mem` is what each node gets, and the estimate gives each node what one node would
(2 GB + its share of 4 GB per million cells; 8G a node when the size is unknown). On Perlmutter the S50 sliding case runs on 4 nodes x 128 ranks (about 27k cells a rank), as the
hand-built run did. Engaging's per-user cap (96 CPUs) keeps jobs on one node there.

OpenFOAM 14 is built on both clusters (build 14-a4867fc914c1; `env-of14-perlmutter.sh`, `env-of14-engaging.sh` here,
installed as `~/software/openfoam/env-of14.sh`). Unverified (2026.11.11): Gmsh's wheel on the clusters, `sbatch` from
inside a job (the transient segments resubmit themselves) on Engaging, and multi-node jobs on Perlmutter. Check them
with the steps at the end.

---

## 1. OpenFOAM 14 (openfoam.org)

The verified build recipes, the env scripts and the test runs (Perlmutter: test job 61153641, 2 nodes x 8 ranks; NCC
mixerVessel2D on 16 ranks) are in `openfoam-v2506-hpc-recipes.md` (sections "OpenFOAM-14"), next to this file. Both
install OpenFOAM-14 and ThirdParty-14 in `~/software/openfoam`, next to v2506, with `env-of14.sh` there. The outline:

The Foundation's release, the one cfdt writes cases for (`foamRun` with modules). ESI's v2506 (openfoam-v2506-hpc-recipes.md, next to
this file) runs other solvers and does not read cfdt's cases.

### MIT Engaging (ORCD)

```bash
salloc -p mit_normal -N 1 -c 32 --mem=64G -t 03:00:00
exec bash -l
module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8
mkdir -p ~/software/openfoam && cd ~/software/openfoam        # next to OpenFOAM-v2506
git clone https://github.com/OpenFOAM/OpenFOAM-14.git
git clone https://github.com/OpenFOAM/ThirdParty-14.git
source OpenFOAM-14/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMOPENMPI
foamSystemCheck
(cd $WM_THIRD_PARTY_DIR && ./Allwmake -j $SLURM_CPUS_PER_TASK)      # Scotch (decomposition, snappyHexMesh)
cd $WM_PROJECT_DIR && ./Allwmake -j $SLURM_CPUS_PER_TASK -s -l
foamRun -help | head -3; snappyHexMesh -help | head -3
```

### NERSC Perlmutter

Build in `$PSCRATCH` (home quota 40 GB), then copy the slim install home.

```bash
salloc -N 1 -C cpu -q interactive -t 02:00:00
module load PrgEnv-gnu cray-mpich cmake
mkdir -p $PSCRATCH/openfoam14 && cd $PSCRATCH/openfoam14
git clone https://github.com/OpenFOAM/OpenFOAM-14.git
git clone https://github.com/OpenFOAM/ThirdParty-14.git
# cray-mpich through OpenFOAM's SYSTEMMPI (it reads these four variables)
export MPI_ROOT=$MPICH_DIR
export MPI_ARCH_FLAGS="-DMPICH_SKIP_MPICXX"
export MPI_ARCH_INC="-I$MPI_ROOT/include"
export MPI_ARCH_LIBS="-L$MPI_ROOT/lib -lmpich"
source OpenFOAM-14/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMMPI
foamSystemCheck
(cd $WM_THIRD_PARTY_DIR && ./Allwmake -j 128)
cd $WM_PROJECT_DIR && ./Allwmake -j 128 -s -l
```

Slim copy home, from a login node (as for v2506: binaries and libraries, no sources, tutorials or build trees):

```bash
cd $PSCRATCH/openfoam14
rsync -a --exclude 'OpenFOAM-14/applications/' --exclude 'OpenFOAM-14/src/' --exclude 'OpenFOAM-14/tutorials/' \
  --exclude 'OpenFOAM-14/doc/' --exclude 'OpenFOAM-14/build/' --exclude 'OpenFOAM-14/platforms/*/applications/' \
  --exclude 'OpenFOAM-14/platforms/*/src/' --exclude '.git*' \
  --include 'ThirdParty-14/' --include 'ThirdParty-14/platforms/***' --include 'ThirdParty-14/etc/***' \
  --exclude 'ThirdParty-14/*' ./ $HOME/software/openfoam/         # next to OpenFOAM-v2506
```

The four `MPI_*` variables must be set again before sourcing the bashrc in every shell: `env-of14-perlmutter.sh`
(here; installed as `~/software/openfoam/env-of14.sh`) does it, with the values the build used. If `foamSystemCheck` or the Pstream build complains about MPI, check `$MPICH_DIR` with `module show cray-mpich`.

## 2. Python and cfdt

```bash
# Perlmutter: module load python      Engaging: module load miniforge
python3 -m venv ~/venvs/cfdt && source ~/venvs/cfdt/bin/activate
cd ~/software && git clone https://github.com/feranick/llmt-runtime.git && git clone https://github.com/feranick/cfd-translator.git
pip install -e llmt-runtime && pip install -e cfd-translator
pip install gmsh            # only for STEP or goal geometry; STL geometry needs no Gmsh
```

## 3. Settings

Put them in `~/cfdt.env` and source it before any `cfdt` command. `CFDT_JOB_SETUP` writes the module loads, the
OpenFOAM environment and the venv into every job script, so the jobs (and the transient segments they submit) do not
depend on the shell that submitted them.

**Perlmutter**

```bash
module load python
source ~/software/openfoam/env-of14.sh        # PrgEnv-gnu, cray-mpich, SYSTEMMPI, the OpenFOAM 14 bashrc
source ~/venvs/cfdt/bin/activate
export CFDT_FOAM=$HOME/software/openfoam/env-of14.sh   # sourced before each OpenFOAM command
export CFDT_JOB_SETUP="module load python\nsource ~/software/openfoam/env-of14.sh\nsource ~/venvs/cfdt/bin/activate"
export CFDT_EXECUTOR=slurm
export CFDT_OUT=$PSCRATCH/cfdt_runs           # scratch is purged on a schedule: pack results soon
export CFDT_SLURM_ARGS="-A m526 -C cpu -q regular"   # -C cpu: the 128-core CPU nodes size the jobs
export CFDT_SLURM_NODE=any                    # SLURM chooses the nodes (the login node is not one)
export CFDT_SLURM_NODES=4                     # a job may span up to 4 nodes
export CFDT_LAUNCH="srun -n {n} --cpu-bind=cores"
export CFDT_PROCS=512                         # 4 x 128; studies with fewer cells get fewer (5000 cells a rank)
export CFDT_TIMEOUT=42600                     # s per job: with cfdt's 10 min, 12 h
```

A study's `run.procs` (or `solve.procs`) up to 512 is accepted (`caps.procs`); 512 ranks run as 4 nodes x 128, 300 as
3 x 100, up to 128 on one node. Memory is per node, `run.mem` and the estimate alike: `run.mem: "120G"` on a 4-node job
asks `--mem=120G` on each node (480G in all); the estimate gives each node 2 GB + its share of 4 GB per million cells
(job.json: `mem` per node, `mem_total`, `mem_per: node`). Before 2026.11.12.8 an unknown size on 4 nodes asked 2G a node.

A node's memory cap (`caps.mem`, per node) is what a job can get there (llmt-runtime 2026.11.12.1, cfdt 2026.11.12.9):
RealMemory less MemSpecLimit, within the partition's MaxMemPerNode or MaxMemPerCPU x the node's CPUs; `mem_from` in
`cfdt status` says which limit applies. Perlmutter refused `--mem=480G` with RealMemory 515100M before this. If
`cfdt status` shows a cap SLURM still refuses, set `CFDT_MAX_MEM` (per node) below it and check with `sbatch --test-only`
(no job is queued):

```bash
cfdt status | python3 -c "import json,sys; c=json.load(sys.stdin)['executor']['caps']; print(c.get('mem'), '|', c.get('mem_from'))"
for m in 440G 460G 480G; do echo -n "$m: "; sbatch --test-only -A m526 -C cpu -q regular -N 1 --mem=$m --wrap true 2>&1 | tail -1; done
```

**Engaging**

```bash
source ~/software/openfoam/env-of14.sh        # module purge; gcc 12.2.0, openmpi 5.0.8; the OpenFOAM 14 bashrc
module load miniforge
source ~/venvs/cfdt/bin/activate
export CFDT_FOAM=$HOME/software/openfoam/env-of14.sh
export CFDT_JOB_SETUP="source ~/software/openfoam/env-of14.sh\nmodule load miniforge\nsource ~/venvs/cfdt/bin/activate"
export CFDT_EXECUTOR=slurm
export CFDT_OUT=$HOME/orcd/scratch/cfdt_runs
export CFDT_SLURM_ARGS="-p mit_normal"
export CFDT_SLURM_NODE=any
# no CFDT_SLURM_NODES: the per-user cap on mit_normal (cpu=96) is one 96-core node
export CFDT_LAUNCH="mpirun -n {n}"            # if mpirun refuses (slots, PMIx): "srun --mpi=pmix -n {n}"
export CFDT_PROCS=96                          # the per-user cap: cpu=96
export CFDT_MAX_MEM=360G                      # the per-user cap is 386G; never --mem=0 (1.5 TB nodes)
export CFDT_TIMEOUT=42600                     # mit_normal: 12 h per job
```

`job.sh` runs without strict mode (no `set -e`), so a `CFDT_JOB_SETUP` line that returns non-zero (`module load` and
the OpenFOAM bashrc often do) does not end the job; cfdt checks for OpenFOAM itself when it runs. The first hand-built
Perlmutter job died in 7 s on exactly that under `set -euo pipefail`.

`CFDT_FOAM` names the file each OpenFOAM command sources first (cfdt runs every command in its own `bash -c`). The env
script is the robust choice: `CFDT_FOAM=PATH` relies on the calling shell's environment, and on Perlmutter an
interactive login shell with the nersc-python conda environment active did not find OpenFOAM that way, though a fresh
`bash -lc` did. The env script sets the modules, the MPI variables and the OpenFOAM bashrc itself.

`CFDT_LAUNCH` is the MPI launcher cfdt uses for snappyHexMesh, foamRun and the sliding mesh's parallel utilities
(`{n}`: the ranks). Without it cfdt uses `mpirun -np N`, as on its own machines.

## 4. Running

```bash
source ~/cfdt.env
cfdt validate studies.json                    # as validate_studies
cfdt run studies.json --wait 0                # submit and return: the request folder and the job ids
cfdt jobs --active                            # what is queued or running
cfdt status $CFDT_OUT/run_...                 # as study_status (pending ones with progress and ETA)
cfdt pack $CFDT_OUT/run_...                   # when done: run_....cfdt.tgz next to the request
cfdt pack $CFDT_OUT/run_... --state           # with the mesh and fields, for render_images at home (large)
```

Queueing on Engaging: a full 96-core node on `mit_normal` can wait hours (the first S50 job pended 8 h and was then
cancelled with no recorded reason). `squeue --start -j <id>` gives SLURM's estimate; a job cancelled without a reason
(`sacct -j <id> -o JobID,State,Reason,Start,End`) is one to ask ORCD about.

Long transient runs (DES, sliding mesh) write and stop before each job's limit and continue in the next job by
themselves (`solve.transient.max_jobs`), as on mochi. `cfdt cancel REQUEST` stops them.

Copy the archive home (`scp perlmutter:$PSCRATCH/cfdt_runs/run_....cfdt.tgz .`) and ask Claude to
`import_results` it (a local path: the router copies it to a cfdt machine), or run `cfdt import run_....cfdt.tgz`
on a cfdt machine. The archive holds each study's result, images, logs (long ones cut to their last 2 MB), the
monitors' `postProcessing/` (window averages) and the plan; not the time directories, `processor*` or the mesh.

Geometry: STL needs nothing on the cluster; a STEP file is copied with the studies (a path on the cluster); a
build123d goal needs build123d-translator there, or export it as STL first.

## 5. Checks before a big run

1. `cfdt status` shows OpenFOAM 14 and (for STEP) Gmsh found; `cfdt format | head` works.
2. `cfdt bench channel_2d sphere_snappy_stl` from the login node: SLURM jobs that pass in minutes and need only
   OpenFOAM (the sphere is an STL cfdt writes itself: snappyHexMesh in parallel with layers, the launcher with several
   ranks, the environment in jobs, scratch). With Gmsh, also `pipe_poiseuille_mpi` and `pipe_snappy`; with
   build123d-translator, `sphere_snappy_re100`. Bench jobs ask for 30 minutes each (run.timeout_s 20 min;
   `CFDT_BENCH_TIMEOUT` sets another), so a short QOS takes them:
   `CFDT_SLURM_ARGS="-A m526 -C cpu -q debug" cfdt bench channel_2d sphere_snappy_stl` on Perlmutter.
3. `cfdt bench cylinder_2d2_restart`: a transient run in 2-minute jobs that must stop, resubmit itself (sbatch from
   inside a job) and give the same answers. Needs Gmsh.
4. Perlmutter: a short multi-node job before the real one, e.g. `pipe_poiseuille_mpi` with `"run": {"procs": 256}`
   and a mesh fine enough for 256 ranks: job.sh has `--nodes=2 --ntasks-per-node=128`, and `log.foamRun` shows
   `nProcs : 256` on two hosts.
5. The first real case: the S50 `sliding7` (the fine ring at the interface) as built, at 20 m3/h, with
   `solve.transient` `max_co: null` and `degrees_per_step: 1`, on Perlmutter at 4 x 128 ranks. It is the check of
   cfdt's sliding mesh (OpenFOAM 14's non-conformal coupling, NCC) on the S50: cfdt's six earlier sliding attempts
   there failed in the first steps, and `sliding7` has never run, while ESI's cyclicAMI case ran with interface
   weights averaging 1.0008. Compare `pump.head_m`, `power_W` and `efficiency` (revolution averages) with the
   hand-built v2506 run on Perlmutter, and look at `solve.time_step.courant_by_segment` and the sliding block
   (`solve.sliding`: interface cells, correct_mesh_phi).
6. Then the other cases: on Perlmutter up to 4 x 128 ranks; on Engaging one node (96).

## 6. What the hand-built S50 run on Perlmutter needed, and where cfdt does it

| needed | in cfdt |
|---|---|
| pockets cut off from the flow removed | snappy: `cut_off_regions` + `subsetMesh` (mesh.regions_removed) |
| interface faces with no overlap, a stepped interface | sliding: `check` stops before the solve; NCC (`createNonConformalCouples`) instead of AMI |
| sliver cells limiting the time step | `solve.transient.max_co: null`: a fixed step of `degrees_per_step` (or `dt`), no Courant limit; the Courant numbers reported per job segment (`solve.time_step`). A report of the cells alone would not have saved the run |
| pcorr tolerance | the transient fvSolution's pcorr/MeshPhi settings (correctPhi), the retry without correctMeshPhi |
| write and stop before the time limit, continue, cap the jobs | transient segments (deadline writeNow, latestTime, max_jobs) |
