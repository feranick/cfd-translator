# of14-tests.sh - OpenFOAM-14 verification tests (same file on Perlmutter and Engaging)
#
# Install as ~/software/openfoam/of14-tests.sh and source it from a Slurm job
# (2 nodes x 8 ranks) AFTER sourcing env-of14.sh. Needs: WORK, SLURM_NTASKS,
# OF14_LAUNCH (set by env-of14.sh) and FOAM_TUTORIALS.
#
# 1. version / build string + installed decomposition libraries
# 2. serial   incompressibleFluid/pitzDailySteady
# 3. parallel incompressibleFluid/pitzDailySteady (scotch, all ranks, 2 nodes) + metis decomposition
# 4. parallel incompressibleFluid/mixerVessel2D: sliding mesh with non-conformal coupling (NCC)

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

# --- 1+2. serial pitzDailySteady, build string --------------------------------
echo "== serial pitzDailySteady"
cp -r "$FOAM_TUTORIALS/incompressibleFluid/pitzDailySteady" serial && cd serial
blockMesh > log.blockMesh 2>&1
grep -E "^(Version|Build|Arch)" log.blockMesh      # expect Build : 14-a4867fc914c1
foamRun > log.foamRun 2>&1
check log.foamRun "serial pitzDailySteady"
grep -i "converged" log.foamRun | tail -1
cd ..

# --- 3. parallel pitzDailySteady (scotch) + metis decomposition ---------------
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

# --- 4. NCC sliding mesh in parallel: mixerVessel2D, 0.25 s = 1/4 revolution --
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
