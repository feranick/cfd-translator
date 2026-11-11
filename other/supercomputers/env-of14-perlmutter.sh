# env-of14.sh - OpenFOAM-14 (Foundation, build 14-a4867fc914c1) on NERSC Perlmutter
#
# Install as:  ~/software/openfoam/env-of14.sh
# Use:         source ~/software/openfoam/env-of14.sh
# During the build (tree in scratch):
#              OF14_INST=$PSCRATCH/openfoam14 source ~/software/openfoam/env-of14.sh
#
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
