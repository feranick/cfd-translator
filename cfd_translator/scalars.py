"""scalars.py — passive tracers (mixing) and the mean age of fluid (residence time).
Version: 2026.10.19.1

A tracer is a passive scalar carried by the converged flow: c = 1 in one stream and 0 in another shows how
they mix. The mean age a is the mean time the fluid at a point has spent in the domain: div(phi a) -
div(D grad a) = 1 per unit volume (rho for gas), a = 0 at every inlet. Its flow-weighted mean over the outlets
is the mean residence time, V/Q exactly when nothing diffuses back through the inlets, whatever the flow; old
fluid marks dead zones.

OpenFOAM 14 (checked against the source): the scalarTransport function object (libsolverFunctionObjects)
solves each field every iteration with the current phi (volumetric or mass flux); keys field, phi, rho,
diffusivity none|constant|viscosity, D, alphal, alphat (D = alphal nu + alphat nut); it reads 0/<field>, the
scheme div(phi,<field>) and the solver entry <field>; sources come from constant/fvModels
(semiImplicitSource: volumeMode, sources {<field> {explicit; implicit}}, cellZone all); writeControl
writeTime, or it writes the field every iteration.

    validate(study, names, errs)      the study's scalars/age (format.py)
    specs(study)                      [{name, kind: tracer|age, Sc|D, Sct, inlets}]
    case_parts(study, plan, patches, turb, nu, rho_src) -> dict   0/ files, function objects, schemes, solvers
    results(study, mesh, plan, fields, out, warns)                 results.scalars / results.age, checks
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SCALE = 1e-3
RESERVED = {"U", "p", "p_rgh", "T", "h", "e", "k", "omega", "nut", "alphat", "phi", "rho", "age", "s"}
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,30}$")
SCT = 0.7


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(st: dict, bc_kinds: dict, errs: list) -> None:
    """bc_kinds: {bc name: kind}."""
    inlets = {n for n, k in bc_kinds.items() if k == "inlet"}
    sc = st.get("scalars")
    if sc is not None:
        if not isinstance(sc, list) or not sc:
            errs.append("scalars: a list of {name, inlets: {<inlet bc>: value}, Sc | diffusivity, Sct}")
            sc = []
        seen = set()
        for i, s in enumerate(sc):
            w = f"scalars[{i}]"
            if not isinstance(s, dict):
                errs.append(f"{w}: {{name, inlets, Sc | diffusivity}}")
                continue
            bad = set(s) - {"name", "inlets", "Sc", "diffusivity", "Sct"}
            if bad:
                errs.append(f"{w}: unknown keys {sorted(bad)} (name, inlets, Sc, diffusivity, Sct)")
            n = s.get("name")
            if not isinstance(n, str) or not NAME.match(n) or n in RESERVED:
                errs.append(f"{w}.name: letters, digits, _ (starting with a letter), not one of {sorted(RESERVED)}")
            elif n in seen:
                errs.append(f"{w}.name: {n!r} given twice")
            else:
                seen.add(n)
            inl = s.get("inlets")
            if not isinstance(inl, dict) or not inl:
                errs.append(f"{w}.inlets: {{<inlet bc name>: value}} (inlets not listed get 0)")
            else:
                for k, v in inl.items():
                    if k not in inlets:
                        errs.append(f"{w}.inlets: {k!r} is not an inlet bc (inlets: {sorted(inlets)})")
                    if not _num(v):
                        errs.append(f"{w}.inlets.{k}: a number")
            _diff(s, w, errs)
    age = st.get("age")
    if age is not None and age is not False:
        if age is not True and not isinstance(age, dict):
            errs.append("age: true or {Sc, Sct}")
        elif isinstance(age, dict):
            if set(age) - {"Sc", "Sct"}:
                errs.append("age: keys Sc, Sct")
            _diff(age, "age", errs)
        if not inlets:
            errs.append("age: needs an inlet (the age is 0 where fluid enters)")


def _diff(s: dict, w: str, errs: list) -> None:
    if "Sc" in s and "diffusivity" in s:
        errs.append(f"{w}: give Sc or diffusivity, not both")
    for k in ("Sc", "diffusivity", "Sct"):
        if k in s and (not _num(s[k]) or s[k] <= 0):
            errs.append(f"{w}.{k}: a positive number" + (" (m2/s)" if k == "diffusivity" else ""))


def specs(study: dict) -> list[dict]:
    out = []
    for s in study.get("scalars") or []:
        out.append({"name": s["name"], "kind": "tracer", "Sc": s.get("Sc"), "D": s.get("diffusivity"),
                    "Sct": float(s.get("Sct", SCT)), "inlets": {k: float(v) for k, v in s["inlets"].items()}})
    age = study.get("age")
    if age:
        a = age if isinstance(age, dict) else {}
        out.append({"name": "age", "kind": "age", "Sc": float(a.get("Sc", 1.0)), "D": None,
                    "Sct": float(a.get("Sct", SCT)), "inlets": {}})
    return out


def case_parts(study: dict, plan: dict, patches: list, case: Path, turb: bool, nu: float, rho_src: float | None) -> dict:
    """Writes 0/<name> and returns the controlDict function objects, fvSchemes and fvSolution lines and the
    constant/fvModels entries. rho_src: the age source density for mass-flux (energy) runs, else None."""
    sp = specs(study)
    parts = {"funcs": [], "div": "", "solvers": "", "relax": "", "fvModels": [], "names": [s["name"] for s in sp]}
    if not sp:
        return parts
    plan["scalars"] = {}
    for s in sp:
        nm = s["name"]
        bnd = {}
        for p in patches:
            k = p["kind"]
            if k in ("empty", "symmetry"):
                bnd[p["name"]] = {"type": k}
            elif k == "inlet":
                bnd[p["name"]] = {"type": "fixedValue", "value": f"uniform {s['inlets'].get(p['name'], 0.0):.12g}"}
            elif k in ("outlet", "opening"):
                bnd[p["name"]] = {"type": "inletOutlet", "inletValue": "uniform 0", "value": "uniform 0"}
            else:
                bnd[p["name"]] = {"type": "zeroGradient"}
        dims = "[0 0 1 0 0 0 0]" if s["kind"] == "age" else "[0 0 0 0 0 0 0]"
        from .case import _field
        (case / "0" / nm).write_text(_field("volScalarField", nm, dims, "uniform 0", bnd))
        # diffusivity: constant D for laminar runs given one; otherwise D = nu / Sc (+ nut / Sct when turbulent)
        if s["D"] is not None and not turb:
            dline = f"diffusivity     constant;\n        D               {s['D']:.12g};"
            alphal, alphat = s["D"] / nu, 0.0
        else:
            alphal = (s["D"] / nu) if s["D"] is not None else 1.0 / float(s["Sc"] or 1.0)
            alphat = 1.0 / s["Sct"] if turb else 0.0
            dline = f"diffusivity     viscosity;\n        alphal          {alphal:.12g};\n        alphat          {alphat:.12g};"
        parts["funcs"].append(
            f"    cfdtS_{nm}\n    {{\n        type            scalarTransport;\n"
            "        libs            (\"libsolverFunctionObjects.so\");\n"
            f"        field           {nm};\n        {dline}\n        writeControl    writeTime;\n    }}")
        lim = "limitedLinear01 1" if s["kind"] == "tracer" else "limitedLinear 1"
        parts["div"] += f"    div(phi,{nm})  bounded Gauss {lim};\n"
        plan["scalars"][nm] = {"kind": s["kind"], "D": float(alphal * nu), "alphal": alphal, "alphat": alphat,
                               "inlets": s["inlets"]}
        if s["kind"] == "age":
            src = 1.0 if rho_src is None else float(rho_src)
            parts["fvModels"].append(
                f"cfdtAgeSource\n{{\n    type            semiImplicitSource;\n    cellZone        all;\n"
                f"    volumeMode      specific;\n    sources\n    {{\n        {nm}\n        {{\n"
                f"            explicit    {src:.12g};\n            implicit    0;\n        }}\n    }}\n}}")
            plan["scalars"][nm]["source_per_m3"] = src
    names = "|".join(parts["names"])
    tol = float(plan.get("residual", 1e-5)) * 1e-2
    # solved to tolerance every iteration with the current flux (no equation relaxation): when the flow has
    # converged, so has every scalar
    parts["solvers"] = (f"    \"({names})\"\n    {{\n        solver          smoothSolver;\n        smoother        symGaussSeidel;\n"
                        f"        tolerance       {tol:.3g};\n        relTol          0;\n        maxIter         200;\n    }}\n")
    parts["relax"] = "".join(f"        {n:<15} 1;\n" for n in parts["names"])
    return parts


# ---- results -------------------------------------------------------------------------------------------
def results(study: dict, mesh, plan: dict, fields: dict, out: dict, warns: list) -> None:
    sc = plan.get("scalars") or {}
    if not sc or not fields.get("scalars"):
        return
    from .post import _Probe, _r
    g = mesh.geometry()
    ni = mesh.ninternal
    dc = mesh.delta_coeffs_boundary() / SCALE
    en = plan.get("energy")
    V = g["V"] * SCALE ** 3
    vol = float(V.sum())
    two_d = bool(study.get("2d"))
    checks = out.setdefault("checks", {})
    for nm, sp in sc.items():
        cfield = fields["scalars"].get(nm)
        if cfield is None:
            warns.append(f"{nm}: OpenFOAM wrote no field (scalarTransport not run?)")
            continue
        c = np.asarray(cfield["internal"], float)
        conv_in = conv_out = diff_in = 0.0
        per = {}
        Qin = 0.0
        for P in mesh.patches:
            pn = P["name"]
            kind = plan["patches"][pn]["kind"]
            if kind not in ("inlet", "outlet", "opening"):
                continue
            f = np.arange(P["start"], P["start"] + P["n"])
            own = mesh.owner[f]
            pf = fields["patches"][pn]
            phi = np.asarray(pf["phi"], float)
            cb = np.asarray(cfield["patches"].get(pn, c[own]), float)
            A = np.linalg.norm(g["Sf"][f], axis=1) * SCALE ** 2
            nut = pf.get("nut")
            nut = np.asarray(nut, float) if nut is not None else 0.0
            if en:
                rho_f = np.asarray(pf.get("rho") if pf.get("rho") is not None else np.full(len(f), en["rho"]), float)
                nu_f = en["mu"] / rho_f
                Qp = phi / rho_f
            else:
                rho_f, nu_f, Qp = 1.0, plan["nu"], phi
            D = sp["alphal"] * nu_f + sp["alphat"] * nut
            # diffusive flux out of the domain through fixed-value faces (inlets); zero where zeroGradient
            dflux = -(rho_f * D * (cb - c[own]) * dc[f - ni] * A) if kind == "inlet" else np.zeros(len(f))
            cflux = phi * cb
            conv_out += float(cflux[phi > 0].sum())
            conv_in += float(-cflux[phi < 0].sum())
            diff_in += float(-dflux.sum())
            w = np.abs(Qp)
            if w.sum() > 0:
                m = float((w * cb).sum() / w.sum())
                rec = {"mean": _r(m), "min": _r(cb.min()), "max": _r(cb.max())}
                if kind != "inlet" and sp["kind"] == "tracer" and abs(m) > 1e-300:
                    cov = float(np.sqrt((w * (cb - m) ** 2).sum() / w.sum())) / abs(m)
                    rec["mixing_index"] = _r(max(0.0, 1.0 - cov))
                per[pn] = rec
            if kind == "inlet":
                Qin += float(-Qp.sum())
        rec = {"outlets": {k: v for k, v in per.items() if plan["patches"][k]["kind"] != "inlet"},
               "volume_mean": _r(float((c * V).sum() / vol)), "min": _r(c.min()), "max": _r(c.max())}
        if sp["kind"] == "tracer":
            # conservation: what enters (convected and diffused through the inlets) leaves through the outlets
            src_in = conv_in + diff_in
            scale = max(abs(conv_in), abs(conv_out), 1e-300)
            err = abs(conv_out - src_in) / scale
            checks.setdefault("scalar_balance", {})[nm] = {"in": _r(src_in), "out": _r(conv_out), "error": _r(err)}
            if err > 1e-3:
                warns.append(f"tracer {nm}: balance off by {err:.2%} (not converged?)")
            hi = max(list(sp["inlets"].values()) + [0.0])
            lo = min(list(sp["inlets"].values()) + [0.0])
            over = max(float(c.max()) - hi, lo - float(c.min()), 0.0)
            if over > 1e-3 * max(hi - lo, 1e-300):
                warns.append(f"tracer {nm}: values outside the inlet range by {over:.3g} (numerical over/undershoot)")
            out.setdefault("scalars", {})[nm] = rec
        else:
            # mean age: sum(out phi a) + diffusion back through the inlets = source * V (exact); and the outlet
            # mean is V/Q (the mean residence time) when that back-diffusion is small
            src = sp.get("source_per_m3", 1.0) * vol
            total = conv_out + (-diff_in)      # diff_in < 0: age diffuses out through the inlets
            err = abs(total - src) / src
            outs = rec["outlets"]
            # flow-weighted outlet mean over all outlets
            num = den = 0.0
            for pn in outs:
                P = next(P for P in mesh.patches if P["name"] == pn)
                f = np.arange(P["start"], P["start"] + P["n"])
                phi = np.asarray(fields["patches"][pn]["phi"], float)      # flow- (mass-) weighted
                cb = np.asarray(cfield["patches"].get(pn, c[mesh.owner[f]]), float)
                num += float((np.clip(phi, 0, None) * cb).sum())
                den += float(np.clip(phi, 0, None).sum())
            mean_age = num / den if den > 0 else float("nan")
            tau = vol / Qin if Qin > 0 else float("nan")
            k = int(np.argmax(c))
            rec.update({"mean_residence_time_s": _r(mean_age), "V_over_Q_s": _r(tau),
                        "max_s": _r(c[k]), "max_at": _r(g["C"][k], 6),
                        "volume_fraction_older_than_2tau": _r(float(V[c > 2 * tau].sum()) / vol) if np.isfinite(tau) else None})
            del rec["min"]
            checks["age_balance"] = {"source": _r(src), "out": _r(total), "error": _r(err),
                                     "outlet_mean_vs_V_over_Q": _r(mean_age / tau - 1) if np.isfinite(tau) else None}
            if err > 1e-3:
                warns.append(f"age: balance off by {err:.2%} (not converged?)")
            out["age"] = rec
    # probes
    if study.get("probes") and out.get("probes") is not None:
        names = [n for n in sc if n in fields["scalars"]]
        if names:
            Xs = [g["C"]]
            Vs = [np.column_stack([np.asarray(fields["scalars"][n]["internal"], float) for n in names])]
            for P in mesh.patches:
                if P["type"] == "empty":
                    continue
                f = np.arange(P["start"], P["start"] + P["n"])
                own = mesh.owner[f]
                Xs.append(g["Cf"][f])
                Vs.append(np.column_stack([np.asarray(fields["scalars"][n]["patches"].get(P["name"],
                           np.asarray(fields["scalars"][n]["internal"], float)[own]), float) for n in names]))
            pr = _Probe(np.concatenate(Xs), np.concatenate(Vs), two_d)
            for p in study["probes"]:
                if p["name"] not in out["probes"]:
                    continue
                pts = [p["at"]] if "at" in p else np.linspace(p["line"]["from"], p["line"]["to"], int(p["line"].get("n", 11)))
                vals = np.array([pr(q)[0] for q in pts])
                for j, n in enumerate(names):
                    out["probes"][p["name"]][n] = _r(vals[0, j]) if "at" in p else _r(vals[:, j])
