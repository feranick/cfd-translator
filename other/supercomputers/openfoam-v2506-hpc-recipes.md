# OpenFOAM v2506 (ESI) and OpenFOAM-14 (Foundation) — HPC build recipes

- Part 1: OpenFOAM v2506 (ESI) on Engaging and Perlmutter (below).
- Part 2: OpenFOAM-14 (Foundation), build `14-a4867fc914c1`, matching the cfdt servers (at the end).
  Installed side by side with v2506; each version is loaded only by its own env script / bashrc.

## MIT Engaging (ORCD)

Install location: `~/software/openfoam` (home = 200 GB flash, backed up)
Runs: `~/orcd/scratch/runs`

### Build
```bash
salloc -p mit_normal -N 1 -c 32 --mem=64G -t 03:00:00
exec bash -l
module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8
cd ~/software/openfoam
source OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMOPENMPI
echo $WM_MPLIB $FOAM_MPI $MPI_ARCH_PATH   # SYSTEMOPENMPI sys-openmpi /orcd/software/core/001/spack/pkg/openmpi/5.0.8/...
foamSystemCheck
cd $WM_PROJECT_DIR && ./Allwmake -j $SLURM_CPUS_PER_TASK -s -l
```

### Limits (mit_normal)
- MaxTime 12:00:00 per job; 62 nodes (mostly 96-core, some 192/64)
- QoS mit_normal per user: **cpu=96, mem=386G** running at once; MaxSubmitPU=448
- Use 1 node × 96 ranks; never `--mem=0` (1.5 TB nodes exceed the 386G cap)

### Production job script (1 full node, 96 ranks, 12 h)
Save as `run_openfoam.sh` inside the case directory and submit with `sbatch run_openfoam.sh`.
```bash
#!/bin/bash
#SBATCH -J of-mycase
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=96          # = per-user cap (cpu=96); one full 96-core node
#SBATCH --mem=360G                    # under the 386G per-user cap; never --mem=0
#SBATCH -t 12:00:00                   # partition max
#SBATCH -o %x-%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<kerb>@mit.edu

SOLVER=simpleFoam                     # change to your solver
CASE=$SLURM_SUBMIT_DIR                # run from the case dir in ~/orcd/scratch/runs/...

module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8
source ~/software/openfoam/OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMOPENMPI

cd $CASE
foamDictionary -entry numberOfSubdomains -set $SLURM_NTASKS system/decomposeParDict

# First job: decompose. Restart jobs (processor dirs already there): continue from latest time.
if [ ! -d processor0 ]; then
    decomposePar > log.decomposePar 2>&1
else
    foamDictionary -entry startFrom -set latestTime system/controlDict
fi

mpirun -n $SLURM_NTASKS $SOLVER -parallel > log.$SOLVER.$SLURM_JOB_ID 2>&1

reconstructPar -latestTime > log.reconstructPar 2>&1
```
Notes:
- 96 ranks → good for meshes of ~1–10 M cells (aim ≥10k cells/rank).
- Runs >12 h: set `writeInterval` so results are saved regularly, then just `sbatch` the same script again — it restarts from the latest time. Chain jobs with `sbatch --dependency=afterany:<jobid> run_openfoam.sh`.
- Changing rank count mid-run requires `reconstructPar` + `decomposePar -force` first.

### SLURM test run (2 nodes × 8 ranks, pitzDaily)
```bash
mkdir -p ~/orcd/scratch/runs && cd ~/orcd/scratch/runs
cat > test_pitzDaily.sh << 'EOF'
#!/bin/bash
#SBATCH -J of-test
#SBATCH -p mit_normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o %x-%j.out

module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8
source ~/software/openfoam/OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMOPENMPI

srun hostname | sort | uniq -c

CASE=~/orcd/scratch/runs/pitzDaily_$SLURM_JOB_ID
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily $CASE
cd $CASE
cp $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDailyExptInlet/system/decomposeParDict system/
foamDictionary -entry numberOfSubdomains -set $SLURM_NTASKS system/decomposeParDict
foamDictionary -entry method -set scotch system/decomposeParDict

blockMesh > log.blockMesh 2>&1
decomposePar > log.decomposePar 2>&1
mpirun -n $SLURM_NTASKS simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime > log.reconstructPar 2>&1

grep -E "nProcs|Hosts" -A3 log.simpleFoam | head
tail -n 5 log.simpleFoam
EOF

sbatch test_pitzDaily.sh
squeue -u $USER
cat of-test-<jobid>.out      # when finished
```
Success = 2 nodes × 8 tasks in hostname count, `nProcs : 16`, both hosts listed, log ends with `End`.
If mpirun complains about slots/PMIx: use `srun --mpi=pmix simpleFoam -parallel`.

---

## NERSC Perlmutter

Install location: `/global/homes/f/feranick/software/openfoam` (slim copy, ~835 MB; built in `$PSCRATCH`)
Runs: `$PSCRATCH/runs`

Notes:
- `/global/common/software` is read-only on compute nodes; m526 project quota there was full.
- Home quota is 40 GB (was ~36 GB used) — build in `$PSCRATCH`, then copy the slim install.

### Build
```bash
salloc -N 1 -C cpu -q interactive -t 02:00:00
module load PrgEnv-gnu cray-mpich cmake
mkdir -p $PSCRATCH/openfoam && cd $PSCRATCH/openfoam
wget https://dl.openfoam.com/source/v2506/OpenFOAM-v2506.tgz
wget https://dl.openfoam.com/source/v2506/ThirdParty-v2506.tgz
tar xzf OpenFOAM-v2506.tgz && tar xzf ThirdParty-v2506.tgz
source OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=CRAY-MPICH
cd OpenFOAM-v2506 && ./Allwmake -j 128 -s -l
```

### Slim copy to home (from a login node)
```bash
cd $PSCRATCH/openfoam
rsync -a \
  --exclude 'OpenFOAM-v2506/applications/' \
  --exclude 'OpenFOAM-v2506/tutorials/' \
  --exclude 'OpenFOAM-v2506/doc/' \
  --exclude 'OpenFOAM-v2506/build/' \
  --exclude '.git*' \
  --include 'ThirdParty-v2506/' \
  --include 'ThirdParty-v2506/platforms/***' \
  --include 'ThirdParty-v2506/etc/***' \
  --exclude 'ThirdParty-v2506/*' \
  ./ /global/homes/f/feranick/software/openfoam/
```

### Production job script (2 full CPU nodes, 256 ranks, 12 h)
Save as `run_openfoam.sh` inside the case directory and submit with `sbatch run_openfoam.sh`.
```bash
#!/bin/bash
#SBATCH -J of-mycase
#SBATCH -A m526
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2                          # scale nodes with mesh size (~10k+ cells/rank)
#SBATCH --ntasks-per-node=128         # 128 physical cores per node (2x AMD EPYC 7763)
#SBATCH -c 2                          # 2 hyperthreads per rank -> 1 rank per physical core
#SBATCH -t 12:00:00                   # regular QoS allows longer (check NERSC queue policy)
#SBATCH -o %x-%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<email>
# Memory: nodes are allocated exclusively (512 GB each) - no --mem needed.

SOLVER=simpleFoam                     # change to your solver
CASE=$SLURM_SUBMIT_DIR                # run from the case dir in $PSCRATCH/runs/...

module load PrgEnv-gnu cray-mpich
source /global/homes/f/feranick/software/openfoam/OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=CRAY-MPICH

cd $CASE
foamDictionary -entry numberOfSubdomains -set $SLURM_NTASKS system/decomposeParDict

# First job: decompose. Restart jobs (processor dirs already there): continue from latest time.
if [ ! -d processor0 ]; then
    decomposePar > log.decomposePar 2>&1
else
    foamDictionary -entry startFrom -set latestTime system/controlDict
fi

srun -n $SLURM_NTASKS --cpu-bind=cores $SOLVER -parallel > log.$SOLVER.$SLURM_JOB_ID 2>&1

reconstructPar -latestTime > log.reconstructPar 2>&1
```
Notes:
- 256 ranks → meshes of ~2.5–25 M cells. Use `-N 1` for smaller cases; add nodes for bigger ones.
- For ≥ a few hundred ranks, add `-fileHandler collated` to decomposePar/solver/reconstructPar to avoid thousands of processor dirs.
- Restart: just `sbatch` the same script again (or chain with `--dependency=afterany:<jobid>`).
- Use `-q debug -t 00:30:00` for quick tests.

### SLURM test run (2 nodes × 8 ranks, pitzDaily)
```bash
mkdir -p $PSCRATCH/runs && cd $PSCRATCH/runs
cat > test_pitzDaily.sh << 'EOF'
#!/bin/bash
#SBATCH -J of-test
#SBATCH -A m526
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:15:00
#SBATCH -o %x-%j.out

module load PrgEnv-gnu cray-mpich
source /global/homes/f/feranick/software/openfoam/OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=CRAY-MPICH

srun hostname | sort | uniq -c

CASE=$PSCRATCH/runs/pitzDaily_$SLURM_JOB_ID
mkdir -p $CASE && tar xzf ~/software/openfoam-tutorials.tgz -C $CASE --strip-components=4 \
    tutorials/incompressible/simpleFoam/pitzDaily
cd $CASE
cat > system/decomposeParDict << 'DICT'
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains 16;
method scotch;
DICT
foamDictionary -entry numberOfSubdomains -set $SLURM_NTASKS system/decomposeParDict

blockMesh > log.blockMesh 2>&1
decomposePar > log.decomposePar 2>&1
srun -n $SLURM_NTASKS --cpu-bind=cores simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime > log.reconstructPar 2>&1

grep -E "nProcs|Hosts" -A3 log.simpleFoam | head
tail -n 5 log.simpleFoam
EOF

sbatch test_pitzDaily.sh
```
Note: the Perlmutter home install has no `tutorials/` — this uses `~/software/openfoam-tutorials.tgz`
(created with `tar czf ~/software/openfoam-tutorials.tgz -C $PSCRATCH/openfoam/OpenFOAM-v2506 tutorials`
before scratch was cleaned). If that archive doesn't exist, copy pitzDaily from Engaging or re-download the tarball.

---

## Quick parallel test (both systems)
```bash
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily . && cd pitzDaily
blockMesh
cp $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDailyExptInlet/system/decomposeParDict system/
foamDictionary -entry numberOfSubdomains -set 8 system/decomposeParDict
foamDictionary -entry method -set scotch system/decomposeParDict
decomposePar
mpirun -n 8 simpleFoam -parallel > log.simpleFoam 2>&1   # Perlmutter: srun -n 8
grep nProcs log.simpleFoam && tail log.simpleFoam
```

---
---

# Part 2 — OpenFOAM-14 (Foundation), matching the cfdt servers

## Target
| Item | Value |
|---|---|
| Distribution | OpenFOAM Foundation (openfoam.org), version 14 |
| OpenFOAM-14 commit | `a4867fc914c10ee10c2f6f83c294c3d3160dd573` (2026-10-08, "Function1s: Fixes to writing in user-time units") → build string **`14-a4867fc914c1`** (verified to exist on GitHub) |
| ThirdParty-14 commit | `6f75cb6aa1a3d8f1ba768840f9ee535244b4c70a` (2026-07-10, "Updated for OpenFOAM-14"; the current master and the latest commit before the OpenFOAM commit) → Scotch/PT-Scotch 7.0.12, Zoltan 3.90 |
| METIS | 5.1.0, not bundled in ThirdParty-14: downloaded into `ThirdParty-14/metis-5.1.0` and built with `METIS_TYPE=ThirdParty` (Foundation default is `METIS_TYPE=none`) |
| Options | `linux64GccDPInt32Opt`: Gcc, DP, 32-bit labels, Opt (Foundation defaults, set explicitly in the env scripts) |
| Must run | `foamRun` with `incompressibleFluid` (steady, MRF, NCC sliding mesh), `incompressibleVoF` (Schnerr-Sauer cavitation), `snappyHexMesh`; scotch/ptscotch/metis decomposition |

Notes that apply to both clusters:
- **Use `git clone` + `checkout`, not a GitHub zip.** The build string comes from `wmakePrintBuild` (`14-` + the first 12 characters of the git HEAD). Without `.git` it would print only `14`.
- **Directory names must be `OpenFOAM-14` and `ThirdParty-14`**, side by side. The Foundation bashrc finds ThirdParty that way. They sit next to `OpenFOAM-v2506` and `ThirdParty-v2506` without conflict.
- **No `foamInstallationTest` or `foamSystemCheck` in OpenFOAM-14.** Verification is by the solver header, `ldd` and the test job below.
- **Objects:** Foundation puts object files in `OpenFOAM-14/platforms/linux64GccDPInt32Opt/{src,applications}/` (ESI used `build/`). Binaries and libraries are in `platforms/linux64GccDPInt32Opt/{bin,lib}`. The slim copy excludes the object trees. This was confirmed on the Perlmutter build: the first dry run without those excludes came to 2.08 GB.
- **Build command:** the top-level `Allwmake` also builds ThirdParty (Scotch, PT-Scotch, Zoltan, METIS). Scotch needs `flex` and `bison`, and METIS needs `cmake`.

## Shared verification tests (`of14-tests.sh`, same file on both clusters)
Save as `~/software/openfoam/of14-tests.sh`. It is sourced by the per-cluster test jobs below.
```bash
cat > ~/software/openfoam/of14-tests.sh << 'EOF_TESTS'
# of14-tests.sh - OpenFOAM-14 verification tests (same file on Perlmutter and Engaging)
# Source from a Slurm job (2 nodes x 8 ranks) AFTER env-of14.sh. Needs WORK, SLURM_NTASKS,
# OF14_LAUNCH (set by env-of14.sh) and FOAM_TUTORIALS.
#  1. version / build string + installed decomposition libraries
#  2. serial   incompressibleFluid/pitzDailySteady
#  3. parallel incompressibleFluid/pitzDailySteady (scotch, all ranks, 2 nodes) + metis decomposition
#  4. parallel incompressibleFluid/mixerVessel2D: sliding mesh with non-conformal coupling (NCC)

RUN="$OF14_LAUNCH -n $SLURM_NTASKS"

dpd() {   # dpd <nSubdomains> <method>: write system/decomposeParDict
cat > system/decomposeParDict << EOF
FoamFile { format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains $1;
method $2;
EOF
}

check() { # check <log> <label>
    if grep -q "^End" "$1" && ! grep -q "FOAM FATAL" "$1"; then
        echo "PASS  $2"
    else
        echo "FAIL  $2  (see $PWD/$1)"
    fi
}

mkdir -p "$WORK" && cd "$WORK" || exit 1

echo "== nodes / ranks"
srun hostname | sort | uniq -c
echo "== environment"
echo "$WM_PROJECT_DIR  $WM_OPTIONS  $WM_MPLIB  $FOAM_MPI  $MPI_ARCH_PATH"
echo "== decomposition libraries"
ls "$FOAM_LIBBIN" "$FOAM_LIBBIN/$FOAM_MPI" 2> /dev/null | grep -i decomp | sort -u

echo "== serial pitzDailySteady"
cp -r "$FOAM_TUTORIALS/incompressibleFluid/pitzDailySteady" serial && cd serial
blockMesh > log.blockMesh 2>&1
grep -E "^(Version|Build|Arch)" log.blockMesh      # expect Build : 14-a4867fc914c1
foamRun > log.foamRun 2>&1
check log.foamRun "serial pitzDailySteady"
grep -i "converged" log.foamRun | tail -1
cd ..

echo "== parallel pitzDailySteady"
cp -r "$FOAM_TUTORIALS/incompressibleFluid/pitzDailySteady" parallel && cd parallel
blockMesh > log.blockMesh 2>&1
dpd "$SLURM_NTASKS" scotch
decomposePar > log.decomposePar 2>&1
$RUN foamRun -parallel > log.foamRun 2>&1
check log.foamRun "parallel pitzDailySteady ($SLURM_NTASKS ranks, scotch)"
grep -E "^nProcs" log.foamRun
grep -i "converged" log.foamRun | tail -1
reconstructPar -latestTime > log.reconstructPar 2>&1
check log.reconstructPar "reconstructPar"
dpd "$SLURM_NTASKS" metis
decomposePar -force > log.decomposePar.metis 2>&1
check log.decomposePar.metis "decomposePar with metis"
cd ..

echo "== parallel NCC mixerVessel2D"
cp -r "$FOAM_TUTORIALS/incompressibleFluid/mixerVessel2D" ncc && cd ncc
blockMesh -dict "$FOAM_TUTORIALS/resources/blockMesh/mixerVessel2D" > log.blockMesh 2>&1
createBaffles > log.createBaffles 2>&1
splitBaffles > log.splitBaffles 2>&1
createNonConformalCouples nonCouple1 nonCouple2 > log.createNonConformalCouples 2>&1
check log.createNonConformalCouples "createNonConformalCouples"
foamDictionary -entry endTime -set 0.25 system/controlDict > /dev/null
dpd "$SLURM_NTASKS" scotch
decomposePar > log.decomposePar 2>&1
$RUN foamRun -parallel > log.foamRun 2>&1
check log.foamRun "parallel NCC mixerVessel2D ($SLURM_NTASKS ranks)"
grep -E "^Time = " log.foamRun | tail -1
cd ..

echo "== done: results in $WORK"
EOF_TESTS
```
- **Tutorial choice:** in OpenFOAM-14, `mixerVessel2D` is the NCC sliding-mesh case. Its `Allrun` runs `createNonConformalCouples nonCouple1 nonCouple2`, and `dynamicMeshDict` sets solidBody rotation of cellZone `rotor` at 60 rpm. There is no `mixerVessel2DNCC` any more. `mixerVessel2DMRF` is the MRF variant. `pitzDailySteady` is the steady case; `pitzDaily` is transient.
- **Expected output:** `Build : 14-a4867fc914c1`, every line `PASS`, `nProcs : 16`, and 8 ranks on each of 2 nodes.

---

## MIT Engaging (ORCD) — OpenFOAM-14

Install location: `~/software/openfoam/OpenFOAM-14` + `~/software/openfoam/ThirdParty-14` (next to v2506; full tree, no slimming needed)
Env script: `~/software/openfoam/env-of14.sh`
Runs: `~/orcd/scratch/runs`

### Environment snippet (`env-of14.sh`)
```bash
cat > ~/software/openfoam/env-of14.sh << 'EOF'
# env-of14.sh - OpenFOAM-14 (Foundation, build 14-a4867fc914c1) on MIT Engaging (ORCD)
# Use: source ~/software/openfoam/env-of14.sh   (runs `module purge`, like the v2506 job scripts)
# Does not touch OpenFOAM-v2506 or ~/.bashrc. Use one OpenFOAM version per shell / job.

OF14_INST=${OF14_INST:-$HOME/software/openfoam}

module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8

if ! command -v mpicc > /dev/null 2>&1; then
    echo "env-of14: mpicc not found - openmpi/5.0.8 module did not load" >&2
    return 1 2> /dev/null || exit 1
fi

. "$OF14_INST/OpenFOAM-14/etc/bashrc" \
    WM_COMPILER=Gcc WM_PRECISION_OPTION=DP WM_LABEL_SIZE=32 WM_COMPILE_OPTION=Opt \
    WM_MPLIB=SYSTEMOPENMPI \
    SCOTCH_TYPE=ThirdParty METIS_TYPE=ThirdParty ZOLTAN_TYPE=ThirdParty \
    ParaView_TYPE=none

# Launcher for job scripts (fallback if mpirun misbehaves: "srun --mpi=pmix")
export OF14_LAUNCH="mpirun"
EOF
```

### Build
```bash
# --- login node: sources at the pinned commits ---
cd ~/software/openfoam
git clone https://github.com/OpenFOAM/OpenFOAM-14.git
git clone https://github.com/OpenFOAM/ThirdParty-14.git
git -C OpenFOAM-14  checkout a4867fc914c10ee10c2f6f83c294c3d3160dd573
git -C ThirdParty-14 checkout 6f75cb6aa1a3d8f1ba768840f9ee535244b4c70a
# METIS 5.1.0: the original glaros.dtc.umn.edu link is dead (404); use the Spack source mirror (keyed by sha256)
wget -O ThirdParty-14/metis-5.1.0.tar.gz \
  https://mirror.spack.io/_source-cache/archive/76/76faebe03f6c963127dbb73c13eab58c9a3faeae48779f049066a21c087c5db2.tar.gz
echo "76faebe03f6c963127dbb73c13eab58c9a3faeae48779f049066a21c087c5db2  ThirdParty-14/metis-5.1.0.tar.gz" | sha256sum -c
tar xzf ThirdParty-14/metis-5.1.0.tar.gz -C ThirdParty-14 && rm ThirdParty-14/metis-5.1.0.tar.gz
ls -d ThirdParty-14/{scotch_7.0.12,Zoltan-3.90,metis-5.1.0}     # all three must exist

# --- build node ---
salloc -p mit_normal -N 1 -c 64 --mem=128G -t 04:00:00
exec bash -l
source ~/software/openfoam/env-of14.sh
module load cmake 2> /dev/null; command -v cmake flex bison      # METIS needs cmake; Scotch needs flex + bison
echo $WM_OPTIONS $WM_MPLIB $FOAM_MPI $MPI_ARCH_PATH
#   -> linux64GccDPInt32Opt SYSTEMOPENMPI openmpi-system /orcd/software/core/001/spack/pkg/openmpi/5.0.8/...
wmakePrintBuild                                                   # -> 14-a4867fc914c1
cd $WM_PROJECT_DIR && ./Allwmake -j $SLURM_CPUS_PER_TASK > log.Allwmake 2>&1
```
- **Time:** expect roughly 1–2 h on 64 cores. Foundation builds more modules than ESI.
- **If the allocation runs out:** source `env-of14.sh` again and rerun the same `Allwmake`. It continues where it stopped.
- **Core count:** 64 cores leaves room under the 96-core per-user cap. `-c 32` (as for v2506) also works, just slower.

### Post-build checks
```bash
tail -n 15 log.Allwmake
grep -n "error:\|\*\*\* .*Error" log.Allwmake | head             # should print nothing
ls $FOAM_APPBIN | wc -l
ls $FOAM_LIBBIN $FOAM_LIBBIN/$FOAM_MPI | grep -i decomp           # scotchDecomp, ptscotchDecomp, metisDecomp, zoltan...
ls $FOAM_LIBBIN | grep -E "incompressibleFluid|incompressibleVoF" # solver modules for foamRun
ldd $FOAM_APPBIN/foamRun | grep "not found"                      # should print nothing
ldd $FOAM_LIBBIN/$FOAM_MPI/libPstream.so | grep libmpi           # -> openmpi 5.0.8 module path
```
Optional, after the tests pass: delete the objects (several GB; a later rebuild then starts from scratch):
`rm -rf ~/software/openfoam/OpenFOAM-14/platforms/linux64GccDPInt32Opt/{src,applications}`

### Limits
Same as v2506 on `mit_normal`: 96 cores and 386G per user, 12 h per job. Production uses 1 node × 96 ranks with `--mem=360G`. Never use `--mem=0`.

### SLURM test run (2 nodes × 8 ranks)
```bash
mkdir -p ~/orcd/scratch/runs && cd ~/orcd/scratch/runs
cat > test_of14.sh << 'EOF'
#!/bin/bash
#SBATCH -J of14-test
#SBATCH -p mit_normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o %x-%j.out

source ~/software/openfoam/env-of14.sh
WORK=~/orcd/scratch/runs/of14-test-$SLURM_JOB_ID
source ~/software/openfoam/of14-tests.sh
EOF

sbatch test_of14.sh
cat of14-test-<jobid>.out      # when finished: all PASS, Build : 14-a4867fc914c1
```
If `mpirun` complains about slots/PMIx: `export OF14_LAUNCH="srun --mpi=pmix"` after sourcing `env-of14.sh`.

### Production job script (1 full node, 96 ranks, 12 h)
Save as `run_of14.sh` in the cfdt case directory and submit with `sbatch run_of14.sh`.
```bash
#!/bin/bash
#SBATCH -J of14-mycase
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=96          # = per-user cap (cpu=96)
#SBATCH --mem=360G                    # under the 386G per-user cap; never --mem=0
#SBATCH -t 12:00:00                   # partition max
#SBATCH -o %x-%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<kerb>@mit.edu

source ~/software/openfoam/env-of14.sh
cd $SLURM_SUBMIT_DIR                  # the cfdt case, copied to ~/orcd/scratch/runs/...

# decomposeParDict sized to the job (created if the case has none)
[ -f system/decomposeParDict ] || printf 'FoamFile { format ascii; class dictionary; object decomposeParDict; }\nmethod scotch;\n' > system/decomposeParDict
foamDictionary -entry numberOfSubdomains -set $SLURM_NTASKS system/decomposeParDict > /dev/null

if [ ! -d processor0 ]; then
    # First job. Serial mesh-preparation steps of the case (e.g. createNonConformalCouples for
    # sliding mesh) must be done before this point. Optional parallel meshing:
    #   blockMesh; decomposePar; $OF14_LAUNCH -n $SLURM_NTASKS snappyHexMesh -parallel -overwrite
    decomposePar > log.decomposePar 2>&1
else
    # Restart: continue from the latest written time
    foamDictionary -entry startFrom -set latestTime system/controlDict > /dev/null
fi

$OF14_LAUNCH -n $SLURM_NTASKS foamRun -parallel > log.foamRun.$SLURM_JOB_ID 2>&1

reconstructPar -latestTime > log.reconstructPar 2>&1
```

---

## NERSC Perlmutter — OpenFOAM-14

> **Verified on Perlmutter (Nov 2026)** with test job 61153641 on 2 nodes × 8 ranks:
> - Build : 14-a4867fc914c1; libPstream linked to Cray MPICH `libmpi_gnu.so.12`.
> - Serial pitzDailySteady: converged in 285 iterations.
> - Parallel pitzDailySteady (16 ranks, scotch): converged in 298 iterations.
> - reconstructPar and METIS decomposition: PASS.
> - NCC mixerVessel2D on 16 ranks: ran to t = 0.25 s, PASS.
> - Slim home copy: 530 MB plus a 32 MB tutorials archive.
>
> The 285 vs 298 iteration counts show the normal effect of decomposition (see the last section).

Build location: `$PSCRATCH/openfoam14/{OpenFOAM-14,ThirdParty-14}`
Install location: slim copy in `~/software/openfoam/{OpenFOAM-14,ThirdParty-14}` (next to v2506) + tutorials archive `~/software/openfoam/OpenFOAM-14-tutorials.tgz`
Env script: `~/software/openfoam/env-of14.sh`
Runs: `$PSCRATCH/runs`

### Environment snippet (`env-of14.sh`)
cray-mpich is used through Foundation's `WM_MPLIB=SYSTEMMPI`, which reads `MPI_ROOT`, `MPI_ARCH_FLAGS`, `MPI_ARCH_INC` and `MPI_ARCH_LIBS`. SYSTEMMPI does not add the MPI library directory to `LD_LIBRARY_PATH`, so the script adds it, plus an rpath.
```bash
mkdir -p ~/software/openfoam
cat > ~/software/openfoam/env-of14.sh << 'EOF'
# env-of14.sh - OpenFOAM-14 (Foundation, build 14-a4867fc914c1) on NERSC Perlmutter
# Use:            source ~/software/openfoam/env-of14.sh
# During build:   OF14_INST=$PSCRATCH/openfoam14 source ~/software/openfoam/env-of14.sh
# Does not touch OpenFOAM-v2506 or ~/.bashrc. Use one OpenFOAM version per shell / job.

OF14_INST=${OF14_INST:-$HOME/software/openfoam}

module load PrgEnv-gnu cray-mpich > /dev/null 2>&1

if [ -z "$MPICH_DIR" ] || [ ! -d "$MPICH_DIR" ]; then
    echo "env-of14: cray-mpich is not loaded (MPICH_DIR unset)" >&2
    return 1 2> /dev/null || exit 1
fi

# cray-mpich through Foundation's SYSTEMMPI (read by etc/config.sh/mpi and wmake)
export MPI_ROOT=$MPICH_DIR
export MPI_ARCH_FLAGS="-DMPICH_SKIP_MPICXX"
export MPI_ARCH_INC="-isystem $MPICH_DIR/include"
export MPI_ARCH_LIBS="-L$MPICH_DIR/lib -lmpi -Wl,-rpath,$MPICH_DIR/lib"
export LD_LIBRARY_PATH=$MPICH_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

. "$OF14_INST/OpenFOAM-14/etc/bashrc" \
    WM_COMPILER=Gcc WM_PRECISION_OPTION=DP WM_LABEL_SIZE=32 WM_COMPILE_OPTION=Opt \
    WM_MPLIB=SYSTEMMPI \
    SCOTCH_TYPE=ThirdParty METIS_TYPE=ThirdParty ZOLTAN_TYPE=ThirdParty \
    ParaView_TYPE=none

# Launcher for job scripts
export OF14_LAUNCH="srun --cpu-bind=cores"
EOF
```

### Build
```bash
# --- login node: sources at the pinned commits ---
mkdir -p $PSCRATCH/openfoam14 && cd $PSCRATCH/openfoam14
git clone https://github.com/OpenFOAM/OpenFOAM-14.git
git clone https://github.com/OpenFOAM/ThirdParty-14.git
git -C OpenFOAM-14  checkout a4867fc914c10ee10c2f6f83c294c3d3160dd573
git -C ThirdParty-14 checkout 6f75cb6aa1a3d8f1ba768840f9ee535244b4c70a
# METIS 5.1.0: the original glaros.dtc.umn.edu link is dead (404); use the Spack source mirror (keyed by sha256)
wget -O ThirdParty-14/metis-5.1.0.tar.gz \
  https://mirror.spack.io/_source-cache/archive/76/76faebe03f6c963127dbb73c13eab58c9a3faeae48779f049066a21c087c5db2.tar.gz
echo "76faebe03f6c963127dbb73c13eab58c9a3faeae48779f049066a21c087c5db2  ThirdParty-14/metis-5.1.0.tar.gz" | sha256sum -c
tar xzf ThirdParty-14/metis-5.1.0.tar.gz -C ThirdParty-14 && rm ThirdParty-14/metis-5.1.0.tar.gz
ls -d ThirdParty-14/{scotch_7.0.12,Zoltan-3.90,metis-5.1.0}     # all three must exist

# --- build node (interactive QoS allows up to 4 h) ---
salloc -N 1 -C cpu -q interactive -t 03:00:00
OF14_INST=$PSCRATCH/openfoam14 source ~/software/openfoam/env-of14.sh
module load cmake; command -v cmake flex bison                   # METIS needs cmake; Scotch needs flex + bison
echo $WM_OPTIONS $WM_MPLIB $FOAM_MPI $MPI_ARCH_PATH
#   -> linux64GccDPInt32Opt SYSTEMMPI mpi-system /opt/cray/pe/mpich/9.1.0/ofi/gnu/12.3
wmakePrintBuild                                                   # -> 14-a4867fc914c1
cd $WM_PROJECT_DIR && ./Allwmake -j 128 > log.Allwmake 2>&1
```
Then run the same post-build checks as on Engaging. `libPstream.so` should link `/opt/cray/pe/mpich/.../libmpi.so`. If `sourcing` prints SYSTEMMPI warnings about `MPI_ROOT` or `MPI_ARCH_*`, `cray-mpich` was not loaded.

### Slim copy to home (from a login node) + tutorials archive
Home has about 3 GB free, so check the size first.
```bash
cd $PSCRATCH/openfoam14
tar czf ~/software/openfoam/OpenFOAM-14-tutorials.tgz -C OpenFOAM-14 tutorials

OPTS=(
  --exclude '.git*'
  --exclude 'log.Allwmake'
  --exclude 'Make/linux64GccDPInt32Opt/'
  --exclude 'OpenFOAM-14/platforms/*/src/'
  --exclude 'OpenFOAM-14/platforms/*/applications/'
  --exclude 'OpenFOAM-14/applications/'
  --exclude 'OpenFOAM-14/tutorials/'
  --exclude 'OpenFOAM-14/doc/'
  --exclude 'OpenFOAM-14/test/'
  --include 'ThirdParty-14/'
  --include 'ThirdParty-14/platforms/***'
  --include 'ThirdParty-14/etc/***'
  --exclude 'ThirdParty-14/*'
)
rsync -a --dry-run --stats "${OPTS[@]}" ./ ~/software/openfoam/ | grep -E "Number of files|Total file size"
# if it fits:
rsync -a --info=progress2 "${OPTS[@]}" ./ ~/software/openfoam/

source ~/software/openfoam/env-of14.sh
echo $WM_PROJECT_DIR                                  # -> /global/homes/f/feranick/software/openfoam/OpenFOAM-14
ldd $FOAM_APPBIN/foamRun | grep "not found"           # should print nothing
```
- **What the slim copy keeps:** binaries, libraries, `etc`, `bin`, `wmake` and the `src` headers (needed for coded/`#codeStream` entries), plus ThirdParty `platforms` and `etc`.
- **What it drops:** object files, `applications/` sources, tutorials, docs and `.git`. The build string is compiled into the libraries, so it survives.
- **Keep `$PSCRATCH/openfoam14`** until the test job below passes from the home copy.

### Limits
- **`interactive`:** up to 4 h. Use it for building.
- **`debug`:** short tests, 30 min here.
- **`regular`:** production. Nodes are exclusive (128 cores and 512 GB each), so no `--mem` is needed.

### SLURM test run (2 nodes × 8 ranks)
The slim home install has no tutorials, so the job unpacks the archive into scratch once.
```bash
mkdir -p $PSCRATCH/runs && cd $PSCRATCH/runs
cat > test_of14.sh << 'EOF'
#!/bin/bash
#SBATCH -J of14-test
#SBATCH -A m526
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:30:00
#SBATCH -o %x-%j.out

source ~/software/openfoam/env-of14.sh

TUT=$PSCRATCH/of14-tutorials
if [ ! -d $TUT/incompressibleFluid ]; then
    mkdir -p $TUT && tar xzf ~/software/openfoam/OpenFOAM-14-tutorials.tgz -C $TUT --strip-components=1
fi
export FOAM_TUTORIALS=$TUT

WORK=$PSCRATCH/runs/of14-test-$SLURM_JOB_ID
source ~/software/openfoam/of14-tests.sh
EOF

sbatch test_of14.sh
cat of14-test-<jobid>.out      # when finished: all PASS, Build : 14-a4867fc914c1
```

### Production job script (2 full CPU nodes, 256 ranks, 12 h)
Save as `run_of14.sh` in the cfdt case directory and submit with `sbatch run_of14.sh`.
```bash
#!/bin/bash
#SBATCH -J of14-mycase
#SBATCH -A m526
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2                          # scale with mesh size (aim >= 10k cells/rank)
#SBATCH --ntasks-per-node=128         # 128 physical cores per node (2x AMD EPYC 7763)
#SBATCH -c 2                          # 2 hyperthreads per rank -> 1 rank per physical core
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=<email>
# Memory: nodes are exclusive (512 GB each) - no --mem needed.

source ~/software/openfoam/env-of14.sh
cd $SLURM_SUBMIT_DIR                  # the cfdt case, copied to $PSCRATCH/runs/...

[ -f system/decomposeParDict ] || printf 'FoamFile { format ascii; class dictionary; object decomposeParDict; }\nmethod scotch;\n' > system/decomposeParDict
foamDictionary -entry numberOfSubdomains -set $SLURM_NTASKS system/decomposeParDict > /dev/null

if [ ! -d processor0 ]; then
    # First job. Serial mesh-preparation steps of the case (e.g. createNonConformalCouples for
    # sliding mesh) must be done before this point. Optional parallel meshing:
    #   blockMesh; decomposePar; $OF14_LAUNCH -n $SLURM_NTASKS snappyHexMesh -parallel -overwrite
    decomposePar > log.decomposePar 2>&1
else
    foamDictionary -entry startFrom -set latestTime system/controlDict > /dev/null
fi

$OF14_LAUNCH -n $SLURM_NTASKS foamRun -parallel > log.foamRun.$SLURM_JOB_ID 2>&1

reconstructPar -latestTime > log.reconstructPar 2>&1
```
For ≥ a few hundred ranks (e.g. the 13.7 M-cell pump on 4+ nodes), add `-fileHandler collated` to decomposePar, foamRun and reconstructPar.

---

## Differences from the cfdt servers' build, and whether they matter

| | cfdt servers (`/opt/openfoam14`) | Perlmutter | Engaging |
|---|---|---|---|
| OpenFOAM source | 14-a4867fc914c1 | same commit | same commit |
| Options | linux64GccDPInt32Opt | same | same |
| Compiler flags | Foundation wmake defaults | same (`wmake/rules/linux64Gcc`) | same |
| Compiler | to check (likely the Ubuntu system GCC) | GCC 14.3 (`gcc-native/14`, PrgEnv-gnu) | GCC 12.2 (Spack) |
| MPI | to check (likely the Ubuntu system OpenMPI) | Cray MPICH 9.1.0 (SYSTEMMPI) | OpenMPI 5.0.8 (SYSTEMOPENMPI) |
| Scotch / Zoltan | to check | ThirdParty-14: Scotch 7.0.12, Zoltan 3.90 | same |
| METIS | to check (Foundation default is none) | 5.1.0 (ThirdParty) | 5.1.0 (ThirdParty) |
| OS / glibc | to check (Ubuntu) | SLES 15 (Cray OS) | Rocky 8.10 |

`/opt/openfoam14` is where the Foundation's Ubuntu packages install, so the servers probably run the packaged build. To fill in the "to check" column, run this on a server:
```bash
. /opt/openfoam14/etc/bashrc
head -2 /etc/os-release; dpkg -l | grep -i openfoam
gcc --version | head -1; mpirun --version | head -1; echo $WM_MPLIB $FOAM_MPI
ls $FOAM_LIBBIN $FOAM_LIBBIN/$FOAM_MPI | grep -i decomp
ldd $FOAM_LIBBIN/libscotchDecomp.so | grep -i scotch       # ThirdParty or system Scotch, and its version
```

**Could these change results?**
- **Same physics.** The source commit is identical, so models, discretisation, solver modules and defaults are the same. The case files from cfdt run unchanged.
- **Compiler, glibc, CPU:** these cause round-off-level differences only. The flags are the same, with no `-march`/`-ffast-math`.
- **MPI:** the order of global sums differs between implementations and rank counts. This gives last-digit differences per iteration, the same effect as changing the number of cores on the servers.
- **Decomposition:** a different rank count, or Scotch/METIS version, gives a different partition. Linear-solver iterations then converge to the same tolerance along a slightly different path.
- **Steady runs (MRF, steady incompressibleFluid):** converged results agree within solver/residual tolerance, far below engineering significance.
- **Transient sliding-mesh and cavitating runs:** instantaneous histories drift apart over time, since the flow is chaotic. Revolution-averaged head, torque, efficiency and NPSH3 should agree within their statistical scatter. Compare averages, not snapshots.
- **Suggested check:** run one small cfdt case on a server and on each cluster with the same rank count, and compare its reported quantities.

Sources: [OpenFOAM-14 commit a4867fc](https://github.com/OpenFOAM/OpenFOAM-14/commit/a4867fc914c10ee10c2f6f83c294c3d3160dd573) · [ThirdParty-14](https://github.com/OpenFOAM/ThirdParty-14) · OpenFOAM-14 `etc/bashrc`, `etc/config.sh/mpi`, `etc/config.sh/metis`, `wmake/wmakePrintBuild`, `Allwmake`, `tutorials/incompressibleFluid/mixerVessel2D` at that commit.
