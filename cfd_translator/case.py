"""case.py — an OpenFOAM case (incompressibleFluid, steady SIMPLE) from a study and its mesh.
Version: 2026.10.16.1

    write_case(study, mesh, patches, case_dir) -> plan

Everything is computed here, not left to OpenFOAM: inlet velocities face by face (uniform or the fully
developed laminar profile, scaled so the flow rate is exact), moving-wall velocities (tangential part),
turbulence inlet values (k = 1.5 (I U)^2, omega = sqrt(k) / (Cmu^0.25 L)), kinematic pressures (p / rho).
Files: system/{controlDict, fvSchemes, fvSolution}, constant/{physicalProperties, momentumTransport,
polyMesh/*}, 0/{U, p[, k, omega, nut]}, case.foam. Target: OpenFOAM (openfoam.org) 11 or later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .format import fluid_props
from .polymesh import SCALE, MeshError, header

CMU = 0.09


def _uniform(v) -> str:
    if np.ndim(v) == 0:
        return f"uniform {float(v):.12g}"
    return "uniform (" + " ".join(f"{float(x):.12g}" for x in v) + ")"


def _nonuniform(a: np.ndarray) -> str:
    a = np.asarray(a, float)
    if a.ndim == 1:
        body = "\n".join(f"{x:.12g}" for x in a)
        return f"nonuniform List<scalar>\n{len(a)}\n(\n{body}\n)"
    body = "\n".join(f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in a)
    return f"nonuniform List<vector>\n{len(a)}\n(\n{body}\n)"


def _field(cls: str, obj: str, dims: str, internal: str, patches: dict) -> str:
    lines = [header(cls, obj, "0"), f"dimensions      {dims};", "", f"internalField   {internal};", "",
             "boundaryField", "{"]
    for name, entries in patches.items():
        lines += [f"    {name}", "    {"]
        for k, v in entries.items():
            lines.append(f"        {k:<15} {v};")
        lines.append("    }")
    lines += ["}", "", "// " + "*" * 73 + " //", ""]
    return "\n".join(lines)


def _dict(obj: str, location: str, body: str) -> str:
    return header("dictionary", obj, location) + "\n" + body.strip() + "\n\n// " + "*" * 73 + " //\n"


# ---- inlet profiles ----------------------------------------------------------------------------
def _plane_frame(n: np.ndarray):
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _boundary_edges(rings: np.ndarray) -> np.ndarray:
    """Edges (pairs of node indices) used by exactly one face of the set."""
    e = []
    for r in rings:
        k = int((r >= 0).sum())
        for i in range(k):
            a, b = int(r[i]), int(r[(i + 1) % k])
            e.append((min(a, b), max(a, b)))
    e = np.array(e, np.int64)
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def inlet_shape(mesh, faces: np.ndarray, two_d: bool, profile: str):
    """Relative velocity (mean 1 over the area) of each inlet face, the inlet's hydraulic diameter (m),
    and a description."""
    g = mesh.geometry()
    Sf, Cf = g["Sf"][faces], g["Cf"][faces]
    A = np.linalg.norm(Sf, axis=1)
    n = Sf / A[:, None]
    rings = mesh.faces[faces]
    nodes = np.unique(rings[rings >= 0])
    P = mesh.points[nodes]
    edges = _boundary_edges(rings)
    perim = float(np.linalg.norm(mesh.points[edges[:, 0]] - mesh.points[edges[:, 1]], axis=1).sum())
    Atot = float(A.sum())
    planar = bool(np.all(n @ n[0] > 1 - 1e-6))
    if two_d:
        t = np.cross([0, 0, 1.0], n[0])
        if np.linalg.norm(t) < 1e-9:
            raise MeshError("2d inlet faces must be edge-on (normal in the XY plane)")
        t /= np.linalg.norm(t)
        s = P @ t
        W = float(s.max() - s.min())
        dh = 2 * W * SCALE
        desc = {"shape": "2D channel", "width_mm": W}
        if profile == "parabolic":
            if not planar:
                raise MeshError("a parabolic 2D inlet must be straight")
            sc = Cf @ t
            mid = 0.5 * (s.max() + s.min())
            rel = np.clip(1 - (2 * (sc - mid) / W) ** 2, 0, None)
        else:
            rel = np.ones(len(faces))
    else:
        dh = 4 * Atot / max(perim, 1e-300) * SCALE
        desc = {"shape": "inlet", "area_mm2": Atot, "perimeter_mm": perim}
        rel = np.ones(len(faces))
        if profile == "parabolic":
            if not planar:
                raise MeshError("a parabolic inlet must be planar")
            e1, e2 = _plane_frame(n[0])
            c0 = (Cf * A[:, None]).sum(axis=0) / Atot
            u, v = (P - c0) @ e1, (P - c0) @ e2
            R = float(np.sqrt(u * u + v * v).max())
            uc, vc = (Cf - c0) @ e1, (Cf - c0) @ e2
            if abs(Atot / (np.pi * R * R) - 1) < 0.03:
                rel = np.clip(1 - (uc * uc + vc * vc) / R ** 2, 0, None)
                desc = {"shape": "circle", "radius_mm": R}
            else:
                # rectangle: principal axes of the boundary points
                w, V = np.linalg.eigh(np.cov(np.column_stack([u, v]).T))
                a1 = V[:, 1]
                pu, pv = np.column_stack([u, v]) @ a1, np.column_stack([u, v]) @ np.array([-a1[1], a1[0]])
                la, lb = float(pu.max() - pu.min()), float(pv.max() - pv.min())
                if abs(Atot / (la * lb) - 1) > 0.03:
                    raise MeshError("a parabolic inlet must be a circle, a rectangle, or 2D (this one is neither)")
                q = np.column_stack([uc, vc])
                y = q @ a1 - 0.5 * (pu.max() + pu.min())
                z = q @ np.array([-a1[1], a1[0]]) - 0.5 * (pv.max() + pv.min())
                a, b = la / 2, lb / 2
                acc = np.zeros(len(faces))
                for i in range(1, 80, 2):
                    acc += (-1) ** ((i - 1) // 2) * (1 - np.cosh(i * np.pi * z / (2 * a)) / np.cosh(i * np.pi * b / (2 * a))) \
                        * np.cos(i * np.pi * y / (2 * a)) / i ** 3
                rel = np.clip(acc, 0, None)
                desc = {"shape": "rectangle", "sides_mm": [la, lb]}
    mean = float((rel * A).sum() / Atot)
    if mean <= 0:
        raise MeshError("inlet profile has zero mean (inlet too coarse: refine it)")
    return rel / mean, dh, desc


# ---- the case ----------------------------------------------------------------------------------
def write_case(study: dict, mesh, patches: list, case: Path) -> dict:
    """Writes the case; returns the plan (inlet values, patches, schemes chosen) for the results."""
    case = Path(case)
    for d in ("system", "constant", "0"):
        (case / d).mkdir(parents=True, exist_ok=True)
    fl = fluid_props(study["fluid"])
    rho, mu = float(fl["rho"]), float(fl["mu"])
    nu = mu / rho
    model = study.get("model", "laminar")
    turb = model != "laminar"
    two_d = bool(study.get("2d"))
    external = study.get("domain", {}).get("kind") == "external"
    mesh.set_patches([{"name": p["name"], "type": p["type"], "tags": p["tags"]} for p in patches])
    g = mesh.geometry()
    q = mesh.quality()
    if not q["ok"]:
        raise MeshError(f"mesh check failed: {q}")
    plan = {"rho": rho, "nu": nu, "mu": mu, "model": model, "patches": {}, "inlets": {}, "quality": q}
    # inlets first (they set the turbulence reference)
    U = {}
    kin = {}
    for p in patches:
        f = mesh.patch_faces(p["name"])
        b = p["bc"]
        plan["patches"][p["name"]] = {"kind": p["kind"], "type": p["type"], "faces": int(len(f)),
                                      "tags": [int(t) for t in p["tags"]]}
        if p["kind"] != "inlet":
            continue
        Sf = g["Sf"][f]
        A = np.linalg.norm(Sf, axis=1)
        n = Sf / A[:, None]
        Am2 = float(A.sum()) * SCALE ** 2
        rel, dh, desc = inlet_shape(mesh, f, two_d, b.get("profile", "uniform"))
        v = b.get("velocity")
        if isinstance(v, (list, tuple)):
            Uf = np.tile(np.asarray(v, float), (len(f), 1))
            vmean = float(np.linalg.norm(v))
        else:
            if v is None:
                Q = float(b["flow_rate"]) if "flow_rate" in b else float(b["flow_rate_lpm"]) / 60000.0
                v = Q / Am2
            vmean = float(v)
            Uf = -vmean * rel[:, None] * n
        Qin = float(-(np.einsum("ij,ij->i", Uf, Sf)).sum() * SCALE ** 2)
        if Qin <= 0:
            raise MeshError(f"inlet {p['name']!r}: the velocity points out of the domain")
        I = float(b.get("intensity", 0.05))
        if "length_scale" in b:
            L = float(b["length_scale"]) * SCALE
        elif external:
            ptags = [t for t, i in mesh.tag_info.items() if i.get("on") == "part"]
            pf = mesh.faces[mesh.ninternal:][np.isin(mesh.btag, ptags)]
            pp = mesh.points[np.unique(pf[pf >= 0])]
            L = 0.07 * float((pp.max(axis=0) - pp.min(axis=0)).max()) * SCALE if len(pp) else 0.07 * dh
        else:
            L = 0.07 * dh
        k = float(1.5 * (vmean * I) ** 2)
        om = float(np.sqrt(k) / (CMU ** 0.25 * L))
        U[p["name"]] = Uf
        kin[p["name"]] = (k, om)
        plan["inlets"][p["name"]] = {"Q": Qin, "U_mean": Qin / Am2, "area_m2": Am2, "hydraulic_diameter_m": dh,
                                     "profile": b.get("profile", "uniform"), **desc,
                                     **({"k": k, "omega": om, "length_scale_m": L, "intensity": I} if turb else {})}
    if kin:
        k0 = max(v[0] for v in kin.values())
        om0 = max(v[1] for v in kin.values())
        uref = max(pl["U_mean"] for pl in plan["inlets"].values())
    else:
        uref = float(study.get("reference", {}).get("velocity", 1.0))
        k0 = 1.5 * (uref * 0.05) ** 2
        Lr = float(study.get("reference", {}).get("length", 0)) * SCALE or mesh.info.get("diag", 1.0) * SCALE * 0.07
        om0 = float(np.sqrt(k0) / (CMU ** 0.25 * Lr))
    plan["U_ref"] = uref
    # fields
    Ub, pb, kb, wb, nb = {}, {}, {}, {}, {}
    has_fixed_p = False
    for p in patches:
        nm, kind, b = p["name"], p["kind"], p["bc"]
        if kind == "empty":
            for d in (Ub, pb, kb, wb, nb):
                d[nm] = {"type": "empty"}
            continue
        if kind == "symmetry":
            for d in (Ub, pb, kb, wb, nb):
                d[nm] = {"type": "symmetry"}
            continue
        if kind == "inlet":
            Uf = U[nm]
            same = np.allclose(Uf, Uf[0], rtol=0, atol=1e-14 * max(1.0, float(np.abs(Uf).max())))
            Ub[nm] = {"type": "fixedValue", "value": _uniform(Uf[0]) if same else _nonuniform(Uf)}
            pb[nm] = {"type": "zeroGradient"}
            k, om = kin[nm]
            kb[nm] = {"type": "fixedValue", "value": _uniform(k)}
            wb[nm] = {"type": "fixedValue", "value": _uniform(om)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind in ("outlet", "opening"):
            has_fixed_p = True
            pk = float(b.get("pressure", 0.0)) / rho
            pb[nm] = {"type": "fixedValue", "value": _uniform(pk)}
            if kind == "outlet":
                Ub[nm] = {"type": "inletOutlet", "inletValue": _uniform([0, 0, 0]), "value": _uniform([0, 0, 0])}
            else:
                Ub[nm] = {"type": "pressureInletOutletVelocity", "value": _uniform([0, 0, 0])}
            kb[nm] = {"type": "inletOutlet", "inletValue": _uniform(k0), "value": _uniform(k0)}
            wb[nm] = {"type": "inletOutlet", "inletValue": _uniform(om0), "value": _uniform(om0)}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        elif kind == "wall":
            vw = b.get("velocity")
            if vw and np.linalg.norm(vw) > 0:
                f = mesh.patch_faces(nm)
                Sf = g["Sf"][f]
                n = Sf / np.linalg.norm(Sf, axis=1)[:, None]
                vw = np.asarray(vw, float)
                Ut = vw[None, :] - np.einsum("ij,j->i", n, vw)[:, None] * n
                same = np.allclose(Ut, Ut[0], rtol=0, atol=1e-12 * float(np.linalg.norm(vw)))
                Ub[nm] = {"type": "fixedValue", "value": _uniform(Ut[0]) if same else _nonuniform(Ut)}
                plan["patches"][nm]["moving"] = True
            else:
                Ub[nm] = {"type": "noSlip"}
            pb[nm] = {"type": "zeroGradient"}
            kb[nm] = {"type": "kqRWallFunction", "value": _uniform(k0)}
            wb[nm] = {"type": "omegaWallFunction", "value": _uniform(om0)}
            nb[nm] = {"type": "nutkWallFunction", "value": _uniform(0)}
        elif kind == "slip":
            Ub[nm] = {"type": "slip"}
            pb[nm] = {"type": "zeroGradient"}
            kb[nm] = {"type": "zeroGradient"}
            wb[nm] = {"type": "zeroGradient"}
            nb[nm] = {"type": "calculated", "value": _uniform(0)}
        else:
            raise MeshError(f"unknown bc kind {kind}")
    U0 = [0.0, 0.0, 0.0]
    if external and len(U) == 1:
        Uf = next(iter(U.values()))
        U0 = Uf.mean(axis=0).tolist()
    (case / "0" / "U").write_text(_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", _uniform(U0), Ub))
    (case / "0" / "p").write_text(_field("volScalarField", "p", "[0 2 -2 0 0 0 0]", _uniform(0), pb))
    if turb:
        (case / "0" / "k").write_text(_field("volScalarField", "k", "[0 2 -2 0 0 0 0]", _uniform(k0), kb))
        (case / "0" / "omega").write_text(_field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", _uniform(om0), wb))
        (case / "0" / "nut").write_text(_field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", _uniform(0), nb))
    # constant
    (case / "constant" / "physicalProperties").write_text(_dict("physicalProperties", "constant",
        f"viscosityModel  constant;\n\nnu              {nu:.12g};\n\n// rho = {rho:g} kg/m3 (results only; the solver is kinematic)"))
    if turb:
        mt = "simulationType  RAS;\n\nRAS\n{\n    model           kOmegaSST;\n\n    turbulence      on;\n\n    printCoeffs     on;\n}"
    else:
        mt = "simulationType  laminar;"
    (case / "constant" / "momentumTransport").write_text(_dict("momentumTransport", "constant", mt))
    mesh.write(case)
    # system
    s = study.get("solve", {})
    iters = int(s.get("iterations", 3000 if turb else 2000))
    res = float(s.get("residual", 1e-5))
    unstructured = q["max_non_orthogonality"] > 5.0
    plan["schemes"] = "unstructured" if unstructured else "orthogonal"
    relax = s.get("relax", {})
    rU = float(relax.get("U", 0.7 if unstructured else 0.9))
    rO = float(relax.get("other", 0.7 if unstructured else 0.9))
    plan.update({"iterations": iters, "residual": res, "relax": {"U": rU, "other": rO}, "fixed_pressure": has_fixed_p})
    walls = [p["name"] for p in patches if p["type"] == "wall"]
    funcs = []
    if walls:
        funcs.append("    wallShearStress\n    {\n        type            wallShearStress;\n"
                     "        libs            (\"libfieldFunctionObjects.so\");\n"
                     f"        patches         ({' '.join(walls)});\n        writeControl    writeTime;\n    }}")
        if turb:
            funcs.append("    yPlus\n    {\n        type            yPlus;\n"
                         "        libs            (\"libfieldFunctionObjects.so\");\n        writeControl    writeTime;\n    }")
    cd = (f"solver          incompressibleFluid;\n\nstartFrom       startTime;\n\nstartTime       0;\n\n"
          f"stopAt          endTime;\n\nendTime         {iters};\n\ndeltaT          1;\n\nwriteControl    timeStep;\n\n"
          f"writeInterval   {iters};\n\npurgeWrite      0;\n\nwriteFormat     ascii;\n\nwritePrecision  12;\n\n"
          "writeCompression off;\n\ntimeFormat      general;\n\ntimePrecision   6;\n\nrunTimeModifiable false;\n\n"
          "functions\n{\n" + "\n".join(funcs) + "\n}")
    (case / "system" / "controlDict").write_text(_dict("controlDict", "system", cd))
    if unstructured:
        grad = "default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n    grad(k)         cellLimited Gauss linear 1;\n    grad(omega)     cellLimited Gauss linear 1;"
        lap, sn, nonorth = "Gauss linear limited corrected 0.5", "limited corrected 0.5", 1
    else:
        grad, lap, sn, nonorth = "default         Gauss linear;", "Gauss linear corrected", "corrected", 0
    fs = (f"ddtSchemes\n{{\n    default         steadyState;\n}}\n\ngradSchemes\n{{\n    {grad}\n}}\n\n"
          "divSchemes\n{\n    default         Gauss linear;\n    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
          "    div(phi,k)      bounded Gauss limitedLinear 1;\n    div(phi,omega)  bounded Gauss limitedLinear 1;\n"
          "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n\n"
          f"laplacianSchemes\n{{\n    default         {lap};\n}}\n\ninterpolationSchemes\n{{\n    default         linear;\n}}\n\n"
          f"snGradSchemes\n{{\n    default         {sn};\n}}\n\nwallDist\n{{\n    method          meshWave;\n}}")
    (case / "system" / "fvSchemes").write_text(_dict("fvSchemes", "system", fs))
    tol_p, tol_u = res * 1e-2, res * 1e-2
    fv = ("solvers\n{\n    p\n    {\n        solver          GAMG;\n"
          f"        tolerance       {tol_p:.3g};\n        relTol          0.05;\n        smoother        GaussSeidel;\n    }}\n\n"
          "    pcorr\n    {\n        solver          GAMG;\n"
          f"        tolerance       {tol_p:.3g};\n        relTol          0;\n        smoother        GaussSeidel;\n    }}\n\n"
          "    \"(U|k|omega)\"\n    {\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
          f"        tolerance       {tol_u:.3g};\n        relTol          0.1;\n    }}\n}}\n\n"
          f"SIMPLE\n{{\n    nNonOrthogonalCorrectors {nonorth};\n    consistent      yes;\n"
          "    pRefCell        0;\n    pRefValue       0;\n\n    residualControl\n    {\n"
          f"        p               {res:.3g};\n        U               {res:.3g};\n        \"(k|omega)\"       {res:.3g};\n    }}\n}}\n\n"
          f"relaxationFactors\n{{\n    equations\n    {{\n        U               {rU:g};\n        \".*\"            {rO:g};\n    }}\n}}")
    (case / "system" / "fvSolution").write_text(_dict("fvSolution", "system", fv))
    (case / "case.foam").write_text("")
    return plan
