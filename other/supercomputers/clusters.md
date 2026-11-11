# Running cfdt on clusters its servers cannot reach (NERSC Perlmutter, MIT Engaging)

cfdt runs on the cluster itself, from its command line. It meshes, writes the case, runs the steady start, the
sliding conversion and the transient segments, and judges the results, in SLURM jobs, exactly as it does on carbonio,
mochi and the sparks. Only the router and the MCP server stay at home. When the runs are done, `cfdt pack` makes a
small archive of the results; `import_results` (or `cfdt import`) turns it into a request on a cfdt machine, where
`study_status`, `window`, `variants` and `render_images` work as usual.

What is not there yet: **jobs over several nodes**. A job is one node today (Perlmutter: up to 128 ranks; Engaging:
96, the per-user cap anyway). Multi-node jobs (`--nodes`, ranks per node) are llmt-runtime's to add; until then the
S50 sliding case runs on 128 ranks on Perlmutter (about 107k cells a rank), not 512.

Unverified (2026.11.11): the OpenFOAM 14 builds below, Gmsh's wheel on the clusters, and `sbatch` from inside a job
(the transient segments resubmit themselves) on Engaging. Check them in that order with the steps at the end.

---

## 1. OpenFOAM 14 (openfoam.org)

The Foundation's release, the one cfdt writes cases for (`foamRun` with modules). ESI's v2506 (docs/openfoam-v2506-
hpc-recipes.md) runs other solvers and does not read cfdt's cases.

### MIT Engaging (ORCD)

```bash
salloc -p mit_normal -N 1 -c 32 --mem=64G -t 03:00:00
exec bash -l
module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8
mkdir -p ~/software/openfoam14 && cd ~/software/openfoam14
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
  --exclude 'ThirdParty-14/*' ./ $HOME/software/openfoam14/
```

The four `MPI_*` variables must be set again before sourcing the bashrc in every shell (put them in the env file
below). If `foamSystemCheck` or the Pstream build complains about MPI, check `$MPICH_DIR` with `module show cray-mpich`.

## 2. Python and cfdt

```bash
# Perlmutter: module load python      Engaging: module load miniforge
python3 -m venv ~/venvs/cfdt && source ~/venvs/cfdt/bin/activate
cd ~/software && git clone https://github.com/feranick/llmt-runtime.git && git clone https://github.com/feranick/cfd-translator.git
pip install -e llmt-runtime && pip install -e cfd-translator
pip install gmsh            # only for STEP or goal geometry; STL geometry needs no Gmsh
```

## 3. Settings

Put them in `~/cfdt.env` and source it, with the modules, the OpenFOAM bashrc and the venv, before any `cfdt`
command. Jobs inherit that environment (`--export=ALL`), and so do the later transient segments they submit.

**Perlmutter**

```bash
module load PrgEnv-gnu cray-mpich python
export MPI_ROOT=$MPICH_DIR MPI_ARCH_FLAGS="-DMPICH_SKIP_MPICXX" MPI_ARCH_INC="-I$MPICH_DIR/include" \
       MPI_ARCH_LIBS="-L$MPICH_DIR/lib -lmpich"
source $HOME/software/openfoam14/OpenFOAM-14/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMMPI
source ~/venvs/cfdt/bin/activate
export CFDT_FOAM=PATH                         # OpenFOAM is already in the environment
export CFDT_EXECUTOR=slurm
export CFDT_OUT=$PSCRATCH/cfdt_runs           # scratch is purged on a schedule: pack results soon
export CFDT_SLURM_ARGS="-A m526 -C cpu -q regular"
export CFDT_SLURM_NODE=any                    # SLURM chooses the node (the login node is not one)
export CFDT_LAUNCH="srun -n {n} --cpu-bind=cores"
export CFDT_PROCS=128                         # one node; a study's solve.procs may ask for fewer
export CFDT_TIMEOUT=42600                     # s per job: with cfdt's 10 min, the 12 h of 'regular'
```

**Engaging**

```bash
module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8 miniforge
source $HOME/software/openfoam14/OpenFOAM-14/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMOPENMPI
source ~/venvs/cfdt/bin/activate
export CFDT_FOAM=PATH
export CFDT_EXECUTOR=slurm
export CFDT_OUT=$HOME/orcd/scratch/cfdt_runs
export CFDT_SLURM_ARGS="-p mit_normal"
export CFDT_SLURM_NODE=any
export CFDT_LAUNCH="mpirun -n {n}"            # if mpirun refuses (slots, PMIx): "srun --mpi=pmix -n {n}"
export CFDT_PROCS=96                          # the per-user cap: cpu=96
export CFDT_MAX_MEM=360G                      # the per-user cap is 386G; never --mem=0 (1.5 TB nodes)
export CFDT_TIMEOUT=42600                     # mit_normal: 12 h per job
```

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
2. `cfdt bench pipe_poiseuille_mpi sphere_snappy_re100` from the login node: SLURM jobs that pass in minutes (the
   launcher with several ranks, snappyHexMesh in parallel, the environment in jobs, scratch).
3. `cfdt bench cylinder_2d2_restart`: a transient run in 2-minute jobs that must stop, resubmit itself (sbatch from
   inside a job) and give the same answers. Needs Gmsh.
4. Then the real case, at most one node's ranks.

## 6. What the hand-built S50 run on Perlmutter needed, and where cfdt does it

| needed | in cfdt |
|---|---|
| pockets cut off from the flow removed | snappy: `cut_off_regions` + `subsetMesh` (mesh.regions_removed) |
| interface faces with no overlap, a stepped interface | sliding: `check` stops before the solve; NCC (`createNonConformalCouples`) instead of AMI |
| sliver cells limiting the time step | mesh quality and skewness reports with places; layers dropped when checkMesh fails |
| pcorr tolerance | the transient fvSolution's pcorr/MeshPhi settings (correctPhi), the retry without correctMeshPhi |
| write and stop before the time limit, continue, cap the jobs | transient segments (deadline writeNow, latestTime, max_jobs) |
