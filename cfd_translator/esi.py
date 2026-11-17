"""esi.py — the ESI OpenFOAM (openfoam.com v2506) backend for sliding meshes: cyclicAMI instead of OpenFOAM 14's
non-conformal coupling, for gap-dominated pumps where NCC needs far smaller steps (other/esi_v2506_backend.md).
Version: 2026.11.17.1

Selected explicitly with solve.backend 'esi' (never silently), only for a transient study with a sliding zone. The
mesh, its checks and the steady MRF start are cfdt's OpenFOAM 14 path, unchanged (mesh once; the start runs on the
conformal mesh, so the frozen-rotor flux loss across an AMI never arises). Then, in the ESI environment only:

    to_sliding(case, plan, mesh, run_step, procs, launch)
        solved fields only, rotor walls movingWallVelocity (rotatingWallVelocity for surfaces of revolution), MRF
        off; zone files' header class for ESI; decomposePar of the conformal case (its addressing kept, as on v14:
        foam.read_fields reads the processor directories); createBaffles -overwrite (a cyclicAMI pair from the
        faceZone, the zone's side first) and mergeOrSplitBaffles -split -overwrite, in parallel; cyclicAMI entries
        added to the fields
    write_transient(case, study, plan)
        constant/{dynamicMeshDict, transportProperties, turbulenceProperties}, system/{controlDict (pimpleFoam,
        the function objects in ESI form), fvSchemes, fvSolution} as in the hand-built S50 v2506 case
    foam.run(..., backend='esi')
        pimpleFoam; its log is log.foamRun, so cfdt's progress, divergence watch and results read it as before

Environment: CFDT_FOAM_ESI (an ESI etc/bashrc), else the newest /usr/lib/openfoam/openfoam2*/etc/bashrc (the
openfoam.com packages), else ~/OpenFOAM/OpenFOAM-v2*/etc/bashrc. One OpenFOAM per shell: each command sources only
its backend's bashrc. Not in scope (refused): energy, cavitation, gas, porous zones, scalars, external flow, steady
studies, p_below, the locator (its coded function object is OpenFOAM 14's).
"""
from __future__ import annotations

import glob
import os
import re
import shutil
from pathlib import Path

import numpy as np

BACKENDS = ("openfoam.org", "esi")
SCALE = 1e-3
LOW_WEIGHT = 0.2           # cyclicAMI lowWeightCorrection (the S50 v2506 case: no 0/0 on snapped corners)
MATCH_TOL = 1e-4
AMI_WARN = 1.5             # sum(weights) max above this, or average off 1 by more than AMI_AVG_WARN: noted
AMI_AVG_WARN = 0.01


def backend(study: dict) -> str:
    return str((study.get("solve") or {}).get("backend") or "openfoam.org")


def is_esi(plan_or_study: dict) -> bool:
    if "backend" in plan_or_study and isinstance(plan_or_study.get("backend"), str):
        return plan_or_study["backend"] == "esi"
    return backend(plan_or_study) == "esi"


V14 = "leave solve.backend out (OpenFOAM 14, the default)"


def errors(st: dict) -> list[str]:
    """Validation of solve.backend (format.validate_study)."""
    s = st.get("solve") or {}
    if "backend" not in s:
        return []
    b = s["backend"]
    if b not in BACKENDS:
        return [f"solve.backend: one of {list(BACKENDS)} (default 'openfoam.org': OpenFOAM 14)"]
    if b != "esi":
        return []
    out = []
    tr = s.get("transient")
    if not isinstance(tr, dict):
        out.append("solve.backend 'esi': only for transient runs (solve.transient) with a sliding zone; a steady study "
                   f"runs on OpenFOAM 14: {V14}")
    if not any(isinstance(z, dict) and z.get("motion") == "sliding" for z in st.get("rotating") or []):
        out.append("solve.backend 'esi': only with a sliding zone (rotating[].motion 'sliding'): cyclicAMI is what it is "
                   f"for; {V14}")
    for key, what in (("energy", "energy"), ("cavitation", "cavitation"), ("porous", "porous zones"),
                      ("scalars", "scalars and mean age")):
        if st.get(key):
            out.append(f"solve.backend 'esi': {what} ({key}) is not in the ESI backend; {V14}")
    if (st.get("domain") or {}).get("kind") == "external":
        out.append(f"solve.backend 'esi': external flow is not in the ESI backend; {V14}")
    if st.get("gravity"):
        out.append(f"solve.backend 'esi': gravity is not in the ESI backend; {V14}")
    if isinstance(tr, dict):
        if tr.get("p_below"):
            out.append("solve.backend 'esi': solve.transient.p_below uses an OpenFOAM 14 coded function object; leave it "
                       f"out, or {V14}")
        if tr.get("locator"):
            out.append("solve.backend 'esi': solve.transient.locator is OpenFOAM 14's (NCC); leave it out, or " + V14)
        if "cells_per_step" in tr:
            out.append("solve.backend 'esi': solve.transient.cells_per_step is OpenFOAM 14's NCC limit; cyclicAMI runs "
                       "the asked degrees_per_step (leave cells_per_step out)")
        if "correct_mesh_phi" in tr:
            out.append("solve.backend 'esi': solve.transient.correct_mesh_phi is OpenFOAM 14's (NCC); leave it out")
    return out


# ---- environment -------------------------------------------------------------------------------------------------
_FOUND: dict = {}


def _ver(p: str) -> int:
    m = re.search(r"(?:openfoam|OpenFOAM-v)(\d{4})", p)
    return int(m.group(1)) if m else -1


def candidates() -> list[str]:
    env = os.environ.get("CFDT_FOAM_ESI")
    if env:
        return [env]
    c = glob.glob("/usr/lib/openfoam/openfoam2*/etc/bashrc") + \
        glob.glob(os.path.expanduser("~/OpenFOAM/OpenFOAM-v2*/etc/bashrc")) + glob.glob("/opt/OpenFOAM-v2*/etc/bashrc")
    c = [x for x in c if _ver(x) >= 0]
    return sorted(c, key=_ver, reverse=True)


def find_esi(refresh: bool = False) -> dict | None:
    """{'bashrc', 'version', 'pimpleFoam'} of an ESI OpenFOAM, {'error'} when the one found is not usable, None when
    there is none. An openfoam.org environment (foamRun, no pimpleFoam) is refused."""
    if "v" in _FOUND and not refresh:
        return _FOUND["v"]
    from .foam import _bash
    res = None
    for c in candidates():
        foam = {"bashrc": c}
        try:
            r = _bash(foam, 'echo "v=$WM_PROJECT_VERSION"; echo "p=$(command -v pimpleFoam)"; '
                            'echo "f=$(command -v foamRun)"', Path.cwd(), timeout=60, capture=True)
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"cannot source {c}: {exc}"}
            continue
        kv = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if ln[:2] in ("v=", "p=", "f="))
        v, p, f = kv.get("v") or "", kv.get("p") or "", kv.get("f") or ""
        if not v.startswith("v") or not p:
            res = {"error": f"{c} is not an ESI OpenFOAM (version {v or '?'}, pimpleFoam {'found' if p else 'missing'}"
                            f"{', foamRun found: openfoam.org' if f else ''})"}
            continue
        res = {"bashrc": c, "version": v, "pimpleFoam": p}
        break
    _FOUND["v"] = res
    return res


def require() -> dict:
    from .foam import FoamError
    e = find_esi()
    if not e or "error" in e:
        raise FoamError("solve.backend 'esi': " + ((e or {}).get("error") or
                        "no ESI OpenFOAM found (install openfoam2506-default from openfoam.com, or set CFDT_FOAM_ESI "
                        "to its etc/bashrc)"))
    return e


# ---- dictionaries ------------------------------------------------------------------------------------------------
def _dict(obj: str, location: str, body: str) -> str:
    from .case import _dict as d
    return d(obj, location, body)


def dynamic_mesh_dict(sl: dict) -> str:
    o = np.asarray(sl["center_mm"], float) * SCALE
    ax = sl["axis"]
    w = float(sl["omega_rad_s"])
    body = ("dynamicFvMesh       dynamicMotionSolverFvMesh;\n\nmotionSolverLibs    (fvMotionSolvers);\n\n"
            f"motionSolver        solidBody;\n\ncellZone            {sl['zone']};\n\n"
            "solidBodyMotionFunction rotatingMotion;\n\n"
            f"origin              ({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});\n"
            f"axis                ({ax[0]:.12g} {ax[1]:.12g} {ax[2]:.12g});\n"
            f"omega               {w:.12g};   // rad/s ({w * 60 / (2 * np.pi):.6g} rpm)")
    return _dict("dynamicMeshDict", "constant", body)


def create_baffles_dict(sl: dict) -> str:
    """A cyclicAMI pair from the faceZone (ESI: patches master/slave; the master takes the faces on the zone's side,
    which write_face_zone's flipMap makes the owner side)."""
    def side(key, name, other):
        return (f"            {key}\n            {{\n                name            {name};\n"
                f"                type            cyclicAMI;\n                matchTolerance  {MATCH_TOL:g};\n"
                f"                lowWeightCorrection {LOW_WEIGHT:g};\n                neighbourPatch  {other};\n"
                "                transform       noOrdering;\n            }\n")
    body = (f"internalFacesOnly true;\n\nbaffles\n{{\n    {sl['group']}\n    {{\n        type            faceZone;\n"
            f"        zoneName        {sl['face_zone']};\n\n        patches\n        {{\n"
            + side("master", sl["rotor"], sl["stator"]) + side("slave", sl["stator"], sl["rotor"]) + "        }\n    }\n}")
    return _dict("createBafflesDict", "system", body)


def transport_properties(nu: float) -> str:
    return _dict("transportProperties", "constant", f"transportModel  Newtonian;\n\nnu              {nu:.8g};")


def turbulence_properties(model: str) -> str:
    if model == "laminar":
        body = "simulationType  laminar;"
    elif model == "kOmegaSSTDES":
        body = ("simulationType  LES;\n\nLES\n{\n    model           kOmegaSSTDES;\n    turbulence      on;\n"
                "    printCoeffs     on;\n    delta           cubeRootVol;\n    cubeRootVolCoeffs\n    {\n"
                "        deltaCoeff      1;\n    }\n}")
    else:
        body = "simulationType  RAS;\n\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n    printCoeffs     on;\n}"
    return _dict("turbulenceProperties", "constant", body)


_LIBS = {"libforces.so": "forces", "libfieldFunctionObjects.so": "fieldFunctionObjects",
         "libsampling.so": "sampling", "libutilityFunctionObjects.so": "utilityFunctionObjects"}


def function(block: str) -> tuple[str | None, str | None]:
    """One of cfdt's controlDict function objects (OpenFOAM 14) in ESI form, or (None, why) when it has none.
    surfaceFieldValue: 'patch X' -> 'regionType patch; name X', 'average' with weightField -> 'weightedAverage';
    pressure: calcTotal/calcCoeff -> mode total|static; fieldAverage: startTime -> timeStart; libs by short name."""
    m = re.search(r"\btype\s+(\w+);", block)
    typ = m.group(1) if m else ""
    name = block.strip().split(None, 1)[0] if block.strip() else "?"
    if typ == "coded":
        return None, (f"function object {name}: OpenFOAM 14 coded code, left out on the ESI backend")
    out = block
    for long, short in _LIBS.items():
        out = out.replace(f'("{long}")', f"({short})")
    if typ == "surfaceFieldValue":
        out = re.sub(r"(?m)^(\s*)patch(\s+)(\w+);", r"\1regionType      patch;\n\1name\2\3;", out)
        if re.search(r"\bweightField\b", out):
            out = re.sub(r"\boperation(\s+)average;", r"operation\1weightedAverage;", out)
    elif typ == "pressure":
        total = bool(re.search(r"calcTotal\s+yes;", out))
        out = re.sub(r"(?m)^\s*calcTotal\s+\w+;\n?", "", out)
        out = re.sub(r"(?m)^(\s*)calcCoeff\s+\w+;", r"\1mode            " + ("total" if total else "static") + ";", out)
        if "mode " not in out:
            out = re.sub(r"(\btype\s+pressure;)", r"\1\n        mode            " + ("total" if total else "static") + ";",
                         out, count=1)
    elif typ == "fieldAverage":
        out = re.sub(r"\bstartTime(\s+)", r"timeStart\1", out)
    elif typ not in ("forces", "probes", "wallShearStress", "yPlus", "fieldAverage"):
        return out, f"function object {name} ({typ}): passed to ESI unchanged, not checked against v2506"
    return out, None


def functions(blocks: list[str]) -> tuple[list[str], list[str]]:
    out, notes = [], []
    for b in blocks:
        f, why = function(b)
        if f is not None:
            out.append(f)
        if why:
            notes.append(why)
    return out, notes


def control_dict(tr: dict, funcs: list[str]) -> str:
    fixed = bool(tr.get("fixed_step")) or tr.get("max_co") is None
    cd = (f"application     pimpleFoam;\n\nstartFrom       latestTime;\n\nstartTime       0;\n\nstopAt          endTime;\n\n"
          f"endTime         {tr['end_time']:.8g};\n\ndeltaT          {tr['dt']:.6g};\n\n"
          + ("adjustTimeStep  no;\n\nwriteControl    runTime;\n\n" if fixed else
             f"adjustTimeStep  yes;\n\nmaxCo           {tr['max_co']:g};\n\nmaxDeltaT       {tr['max_dt']:.6g};\n\n"
             "writeControl    adjustable;\n\n") +
          f"writeInterval   {tr['write_interval']:.8g};\n\npurgeWrite      2;\n\nwriteFormat     ascii;\n\n"
          "writePrecision  10;\n\nwriteCompression off;\n\ntimeFormat      general;\n\ntimePrecision   8;\n\n"
          "runTimeModifiable true;\n\nfunctions\n{\n" + "\n".join(funcs) + "\n}")
    return _dict("controlDict", "system", cd)


def fv_schemes(model: str) -> str:
    """The hand-built S50 v2506 case's schemes (Euler; DES: backward + LUST as its 10b run); wallDist to the wall
    faces' centres (correctWalls false: no zero distance in skewed gap cells, an FPE in SST's F2 otherwise)."""
    des = model == "kOmegaSSTDES"
    ddt = "backward" if des else "Euler"
    divU = "Gauss LUST grad(U)" if des else "Gauss linearUpwind grad(U)"
    fs = (f"ddtSchemes\n{{\n    default         {ddt};\n}}\n\n"
          "gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n"
          "    grad(k)         cellLimited Gauss linear 1;\n    grad(omega)     cellLimited Gauss linear 1;\n}\n\n"
          "divSchemes\n{\n    default         none;\n"
          f"    div(phi,U)      {divU};\n"
          "    div(phi,k)      Gauss limitedLinear 1;\n    div(phi,omega)  Gauss limitedLinear 1;\n"
          "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
          "laplacianSchemes\n{\n    default         Gauss linear limited corrected 0.33;\n}\n\n"
          "interpolationSchemes\n{\n    default         linear;\n}\n\n"
          "snGradSchemes\n{\n    default         limited corrected 0.33;\n}\n\n"
          "wallDist\n{\n    method          meshWave;\n    correctWalls    false;\n}")
    return _dict("fvSchemes", "system", fs)


def fv_solution(model: str, nonorth: int = 1, fixed_pressure: bool = True) -> str:
    """The hand-built S50 v2506 case's solution settings: 3 outer correctors with residual control, 2 pressure
    correctors, correctPhi (pcorr after the mesh moves), checkMeshCourantNo."""
    fields = "U" if model == "laminar" else "U|k|omega"
    ref = "" if fixed_pressure else "    pRefCell            0;\n    pRefValue           0;\n"
    fv = ("solvers\n{\n"
          "    p\n    {\n        solver          GAMG;\n        smoother        DICGaussSeidel;\n"
          "        nCellsInCoarsestLevel 50;\n        tolerance       1e-06;\n        relTol          0.05;\n"
          "        maxIter         200;\n    }\n\n"
          "    pFinal\n    {\n        $p;\n        relTol          0.01;\n    }\n\n"
          "    \"pcorr.*\"\n    {\n        solver          GAMG;\n        smoother        DICGaussSeidel;\n"
          "        tolerance       1e-02;\n        relTol          0;\n        maxIter         50;\n    }\n\n"
          f"    \"({fields})\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          "        tolerance       1e-07;\n        relTol          0.1;\n    }\n\n"
          f"    \"({fields})Final\"\n    {{\n        $U;\n        relTol          0;\n    }}\n}}\n\n"
          "PIMPLE\n{\n    momentumPredictor   yes;\n    nOuterCorrectors    3;\n    nCorrectors         2;\n"
          f"    nNonOrthogonalCorrectors {max(1, int(nonorth))};\n    correctPhi          yes;\n"
          "    checkMeshCourantNo  yes;\n" + ref +
          "    outerCorrectorResidualControl\n    {\n        p\n        {\n            tolerance   1e-4;\n"
          "            relTol      0;\n        }\n        U\n        {\n            tolerance   1e-5;\n"
          "            relTol      0;\n        }\n    }\n}\n\n"
          "relaxationFactors\n{\n    fields\n    {\n        p               0.7;\n        pFinal          1;\n    }\n"
          f"    equations\n    {{\n        \"({fields})\"     0.9;\n        \"({fields})Final\" 1;\n    }}\n}}")
    return _dict("fvSolution", "system", fv)


def write_transient(case: Path, study: dict, plan: dict) -> list[str]:
    """The ESI dialect's constant/ and system/ files of the sliding run (the 0/ fields stay). Returns notes."""
    case = Path(case)
    tr = plan["transient"]
    model = tr["model"]
    (case / "constant" / "transportProperties").write_text(transport_properties(float(plan["nu"])))
    (case / "constant" / "turbulenceProperties").write_text(turbulence_properties(model))
    if plan.get("sliding"):
        (case / "constant" / "dynamicMeshDict").write_text(dynamic_mesh_dict(plan["sliding"]))
    blocks = []
    for f in plan.get("functions", []):
        if re.match(r"\s*(wallShearStress|yPlus)\b", f):         # fields: only when written
            f = f.replace("writeControl    writeTime;", "executeControl  writeTime;\n        writeControl    writeTime;")
        blocks.append(f)
    blocks.append("    cfdtAverage\n    {\n        type            fieldAverage;\n"
                  "        libs            (\"libfieldFunctionObjects.so\");\n        writeControl    writeTime;\n"
                  f"        startTime       {tr['average_from']:.8g};\n        restartOnRestart false;\n"
                  "        fields\n        (\n            U { mean on; prime2Mean on; base time; }\n"
                  "            p { mean on; prime2Mean on; base time; }\n        );\n    }")
    funcs, notes = functions(blocks)
    (case / "system" / "controlDict").write_text(control_dict(tr, funcs))
    (case / "system" / "fvSchemes").write_text(fv_schemes(model))
    (case / "system" / "fvSolution").write_text(fv_solution(model, int(plan.get("nonorth", 1)),
                                                            bool(plan.get("fixed_pressure", True))))
    return notes


# ---- the mesh: zone files, the cyclicAMI pair ---------------------------------------------------------------------
def zone_headers(poly: Path) -> list[str]:
    """OpenFOAM 14 writes the zone files with class cellZoneList / faceZoneList / pointZoneList; ESI writes (and
    reads) regIOobject. Only the header is changed (binary files too). Returns the files changed."""
    done = []
    for nm in ("cellZones", "faceZones", "pointZones"):
        p = Path(poly) / nm
        if not p.is_file():
            continue
        raw = p.read_bytes()
        m = re.search(rb"FoamFile\s*\{(.*?)\}", raw[:4096], re.S)
        if not m:
            continue
        hdr = re.sub(rb"(class\s+)\w+(ZoneList)(\s*;)", rb"\1regIOobject\3", m.group(1))
        if hdr != m.group(1):
            p.write_bytes(raw[:m.start(1)] + hdr + raw[m.end(1):])
            done.append(nm)
    return done


_ENTRY = "    {name}\n    {{\n        type            cyclicAMI;\n    }}\n"


def add_ami_fields(field_dir: Path, patches: list[str]) -> int:
    """cyclicAMI entries for the pair in every field file of field_dir that has none (createBaffles leaves the
    fields as they are). Returns the files changed."""
    n = 0
    for f in sorted(Path(field_dir).iterdir()) if Path(field_dir).is_dir() else []:
        if not f.is_file():
            continue
        txt = f.read_text(errors="replace")
        m = re.search(r"(?m)^boundaryField\s*\n?\s*\{", txt)
        if not m:
            continue
        add = "".join(_ENTRY.format(name=p) for p in patches
                      if not re.search(r"(?m)^\s*" + re.escape(p) + r"\s*(\{|\n\s*\{)", txt[m.end():]))
        if add:
            f.write_text(txt[:m.end()] + "\n" + add + txt[m.end():])
            n += 1
    return n


def to_sliding(case: Path, plan: dict, mesh, run_step, procs: int = 1, launch: str | None = None) -> list[str]:
    """The steady start's case, made the ESI sliding case (see the module's notes). run_step(command, log) runs an
    ESI utility in the case (raising on failure). Returns notes."""
    import json

    from .sliding import (DECOMPOSED, SOLVED, SOLVED_TURB, SlidingError, _boundary, _turbulent, interface,
                          set_patch_type, write_face_zone)
    case = Path(case)
    sl = plan["sliding"]
    zero = case / "0"
    for d in sorted(case.glob("processor*")):
        shutil.rmtree(d, ignore_errors=True)
    (case / DECOMPOSED).unlink(missing_ok=True)
    keep = set(SOLVED_TURB if _turbulent(plan) else SOLVED)
    aside = case / "0.steady"
    for f in sorted(zero.iterdir()) if zero.is_dir() else []:
        if f.is_file() and f.name not in keep:
            aside.mkdir(exist_ok=True)
            shutil.move(str(f), str(aside / f.name))
    uf = zero / "U"
    txt = uf.read_text()
    for w in sl["moving_walls"]:
        txt, ok = set_patch_type(txt, w, "movingWallVelocity")
        if not ok:
            raise SlidingError(f"0/U has no entry for wall {w!r}")
    for w in sl.get("spinning_walls", []):
        o = np.asarray(sl["center_mm"], float) * SCALE
        ax = sl["axis"]
        txt, ok = set_patch_type(txt, w, "rotatingWallVelocity;\n        origin          "
                                 f"({o[0]:.12g} {o[1]:.12g} {o[2]:.12g});\n        axis            "
                                 f"({ax[0]:.12g} {ax[1]:.12g} {ax[2]:.12g});\n        omega           "
                                 f"{float(sl['omega_rad_s']):.12g}")
        if not ok:
            raise SlidingError(f"0/U has no entry for wall {w!r}")
    if re.search(r"type\s+MRFnoSlip;", txt):
        raise SlidingError("0/U still has an MRFnoSlip wall (OpenFOAM 14 only) after the rotor's walls were set: "
                           "every wall turning with the zone must be one of its moving or spinning walls")
    uf.write_text(txt)
    (case / "constant" / "MRFProperties").unlink(missing_ok=True)
    (case / "constant" / "dynamicMeshDict").write_text(dynamic_mesh_dict(sl))
    poly = case / "constant" / "polyMesh"
    f, flip = interface(mesh, mesh.zones[sl["zone"]])
    write_face_zone(poly, sl["face_zone"], f, flip)
    changed = zone_headers(poly)
    (case / "system" / "createBafflesDict").write_text(create_baffles_dict(sl))
    before = {p["name"]: p for p in mesh.patches}
    par = ""
    if procs > 1:
        (case / "system" / "decomposeParDict").write_text(
            "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      decomposeParDict;\n}\n\n"
            f"numberOfSubdomains {procs};\n\nmethod          scotch;\n")
        run_step("decomposePar -force", "log.decomposePar")
        for k in range(procs):
            d = case / f"processor{k}"
            src = d / "constant" / "polyMesh"
            dst = d / "cfdt_addressing"
            dst.mkdir(exist_ok=True)
            for nm in ("cellProcAddressing", "faceProcAddressing", "boundary"):
                if not (src / nm).is_file():
                    raise SlidingError(f"decomposePar (ESI) wrote no {src / nm}")
                shutil.copyfile(src / nm, dst / nm)
        lp = (launch or "mpirun").strip()
        par = (lp if re.search(r"(^|\s)(-np|-n)\s+\d+", lp) else f"{lp} -np {procs}") + " "
    pf = " -parallel" if par else ""
    run_step(f"{par}createBaffles -overwrite{pf}", "log.createBaffles")
    run_step(f"{par}mergeOrSplitBaffles -split -overwrite{pf}", "log.mergeOrSplitBaffles")
    pair = [sl["rotor"], sl["stator"]]
    nf = 0
    for d in ([case / f"processor{k}" for k in range(procs)] if par else [case]):
        nf += add_ami_fields(d / "0", pair)
    if par:
        after = {}
        for k in range(procs):
            for p in _boundary(case / f"processor{k}" / "constant" / "polyMesh"):
                after.setdefault(p["name"], {"name": p["name"], "n": 0, "type": p.get("type")})["n"] += p["n"]
    else:
        after = {p["name"]: p for p in _boundary(poly)}
    for nm, p in before.items():
        q = after.get(nm)
        if q is None or q["n"] != p["n"]:
            raise SlidingError(f"after createBaffles (ESI) patch {nm!r} has {q['n'] if q else 'no'} faces instead of {p['n']}")
    for nm in pair:
        q = after.get(nm, {})
        if q.get("n", 0) != len(f) or q.get("type") != "cyclicAMI":
            raise SlidingError(f"createBaffles (ESI) made {q.get('n', 0)} faces of type {q.get('type')} on {nm!r}, "
                               f"expected {len(f)} cyclicAMI")
    sl["patches_added"] = pair
    sl["ami_patches"] = {nm: "cyclicAMI" for nm in pair}
    sl["backend"] = "esi"
    if par:
        (case / DECOMPOSED).write_text(json.dumps({"procs": procs, "why": "sliding mesh (ESI cyclicAMI): built on the "
                                                  "decomposed case; results are read from processor*"}))
        sl["decomposed"] = procs
    return [f"sliding mesh (ESI {find_esi() and find_esi().get('version')} cyclicAMI): zone {sl['zone']!r} turns at "
            f"{sl['omega_rad_s'] * 60 / (2 * np.pi):g} rpm; {len(f)} faces on its cylinder coupled by an AMI pair "
            f"({sl['rotor']} | {sl['stator']}, lowWeightCorrection {LOW_WEIGHT:g})" +
            (f", built on the case decomposed for {procs} ranks (it stays decomposed)" if par else "") +
            (f"; zone files' header class set for ESI ({', '.join(changed)})" if changed else "") +
            (f"; cyclicAMI entries added to {nf} field files" if nf else "")]


# ---- the log and the fields after the run ------------------------------------------------------------------------
_AMI = re.compile(r"AMI: Patch (source|target) sum\(weights\) min:([-+0-9.eE]+|nan) max:([-+0-9.eE]+|nan) "
                  r"average:([-+0-9.eE]+|nan)")
_LOW = re.compile(r"AMI: Patch (source|target) identified (\d+) faces with weights less than ([-+0-9.eE]+)")


def ami_stats(text: str) -> dict | None:
    """The AMI's weights over the run (log.foamRun): each side's sum(weights) min, max and average, worst over the
    steps (the first, at rest, separately), and the faces with low weights. Sound: average about 1, max well below 2,
    min not stuck at 0 except at snapped corners."""
    rows = _AMI.findall(text)
    if not rows:
        return None
    vals = [(s, float(a), float(b), float(c)) for s, a, b, c in rows]
    at_rest = {s: {"min": a, "max": b, "average": c} for s, a, b, c in vals[:2]}
    moving = vals[2:] or vals
    out = {"at_rest": at_rest, "updates": len(vals) // 2,
           "min": round(min(v[1] for v in moving), 4), "max": round(max(v[2] for v in moving), 4),
           "average_min": round(min(v[3] for v in moving), 6), "average_max": round(max(v[3] for v in moving), 6)}
    low = _LOW.findall(text)
    if low:
        out["low_weight_faces_max"] = max(int(n) for _, n, _ in low)
        out["low_weight_threshold"] = float(low[-1][2])
    return out


def ami_stats_file(log: Path, head: int = 400_000, tail: int = 20_000_000) -> dict | None:
    """ami_stats of a log file, reading its start (the AMI at rest) and its last 20 MB."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(0)
            a = f.read(min(n, head))
            if n > head + tail:
                f.seek(n - tail)
                b = f.read()
            else:
                b = f.read()
    except OSError:
        return None
    return ami_stats((a + b).decode(errors="replace"))


def ami_notes(st: dict | None) -> list[str]:
    if not st:
        return []
    out = []
    if st["max"] > AMI_WARN or abs(st["average_max"] - 1) > AMI_AVG_WARN or abs(st["average_min"] - 1) > AMI_AVG_WARN:
        out.append(f"sliding interface (cyclicAMI): sum of weights from {st['min']} to {st['max']} per face, averages "
                   f"{st['average_min']}-{st['average_max']} (sound: about 1 on average, well below 2 at most): the two "
                   "sides' surfaces do not match well; check the zone's cylinder and caps against the background grid")
    return out


def ami_balance(case: Path, sl: dict, q_through: float | None) -> dict | None:
    """What crosses the AMI pair in the last written fields (phi on the two patches, every processor): the rotor
    side's and the stator side's sums (equal and opposite when the flux is conserved), as sliding.interface_balance."""
    from .foamio import read_foam
    case = Path(case)
    dirs = sorted(case.glob("processor*")) or [case]

    def times(d):
        out = []
        for x in d.iterdir() if d.is_dir() else []:
            try:
                t = float(x.name)
            except ValueError:
                continue
            if t > 0 and (x / "phi").is_file():
                out.append((t, x.name))
        return sorted(out)
    tl = times(dirs[0])
    if not tl:
        return None
    t, tname = tl[-1]
    rot = sto = cross = 0.0
    found = 0
    for d in dirs:
        f = d / tname / "phi"
        if not f.is_file():
            continue
        bf = read_foam(f.read_text(errors="replace")).get("boundaryField") or {}
        for name, side in ((sl["rotor"], "r"), (sl["stator"], "s")):
            e = bf.get(name) or {}
            v = e.get("value")
            if v is None or isinstance(v, str):
                continue
            a = np.atleast_1d(np.asarray(v, float))
            found += a.size
            if side == "r":
                rot += float(a.sum())
                cross += float(a[a > 0].sum())
            else:
                sto += float(a.sum())
    if not found:
        return None
    q = abs(q_through) if q_through else None
    imb = rot + sto
    out = {"time": t, "rotor_side_m3s": float(f"{rot:.6g}"), "stator_side_m3s": float(f"{sto:.6g}"),
           "imbalance_m3s": float(f"{imb:.4g}"), "error_patches_m3s": 0.0,
           "crossing_one_way_m3s": float(f"{cross:.6g}"), "faces": found, "coupling": "cyclicAMI"}
    if q:
        out["through_flow_m3s"] = float(f"{q:.6g}")
        out["error"] = float(f"{abs(imb) / q:.4g}")
    return out
