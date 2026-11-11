<!-- Reference for building and running OpenFOAM on HPC clusters (MIT Engaging, NERSC Perlmutter). Note: these
     recipes are for ESI's OpenFOAM v2506 (openfoam.com: simpleFoam, ...). cfd-translator targets OpenFOAM 14 from
     openfoam.org (foamRun with modules): cases it writes do not run on v2506 as they are. -->

# OpenFOAM v2506 (ESI) — HPC build recipes

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

### Job script
```bash
#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=96
#SBATCH --mem=0
#SBATCH -t 01:00:00

module purge
module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0 openmpi/5.0.8
source ~/software/openfoam/OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=SYSTEMOPENMPI

cd ~/orcd/scratch/runs/mycase
decomposePar -force
mpirun -n $SLURM_NTASKS simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime
```

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

### Job script
```bash
#!/bin/bash
#SBATCH -A m526
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 01:00:00

module load PrgEnv-gnu cray-mpich
source /global/homes/f/feranick/software/openfoam/OpenFOAM-v2506/etc/bashrc WM_COMPILER=Gcc WM_MPLIB=CRAY-MPICH

cd $PSCRATCH/runs/mycase
decomposePar -force
srun -n $SLURM_NTASKS --cpu-bind=cores simpleFoam -parallel > log.simpleFoam 2>&1
reconstructPar -latestTime
```

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
