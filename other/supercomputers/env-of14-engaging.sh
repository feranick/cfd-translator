# env-of14.sh - OpenFOAM-14 (Foundation, build 14-a4867fc914c1) on MIT Engaging (ORCD)
#
# Install as:  ~/software/openfoam/env-of14.sh
# Use:         source ~/software/openfoam/env-of14.sh
#
# Note: runs `module purge` (same as the v2506 job scripts).
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
