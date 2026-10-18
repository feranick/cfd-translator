"""post.py — results and invariant checks from the final fields (OpenFOAM's or the reference solver's).
Version: 2026.10.18.1

    results(study, mesh, plan, fields) -> dict

fields: {"U": (cells, 3) m/s, "p": (cells,) kinematic (p / rho, m2/s2),
         "patches": {name: {"U": (n, 3), "p": (n,), "phi": (n,) m3/s outward,
                            "wss": (n, 3) | None (OpenFOAM wallShearStress, kinematic), "nut": (n,) | None,
                            "yPlus": (n,) | None}}}
On a boundary face (outward unit normal n, area A):
  pressure force on the boundary    (p_f - p_ref) n A
  viscous force on a wall           -rho wallShearStress A   (wallShearStress = R . n_in, R = -viscous stress)
  viscous force elsewhere (ours)    -rho nuEff (U_f - U_P) deltaCoeff A   (nuEff = nu + nut_f; what wallShearStress
                                    reduces to without the small dev2(T(grad U)) term)
  momentum outflow                  rho U_f phi_f
Steady momentum over the whole boundary: sum(outflow + forces) = 0 (momentum_balance); mass: sum(phi) = 0.
"""
from __future__ import annotations

import numpy as np

from .polymesh import SCALE


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _r(x, n=6):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, n) for v in x]
    x = float(x)
    return float(f"{x:.{n}g}") if np.isfinite(x) else x


def _vec3(F, prefix="f"):
    return {f"{prefix}x": _r(F[0]), f"{prefix}y": _r(F[1]), f"{prefix}z": _r(F[2])}


class _Probe:
    """Values at points: weighted least-squares linear fit over the nearest cell centres and boundary
    face centres (2D: in the XY plane)."""

    def __init__(self, X, vals, two_d):
        from scipy.spatial import cKDTree
        self.two_d = two_d
        self.X = X[:, :2] if two_d else X
        self.vals = vals
        self.tree = cKDTree(self.X)
        self.k = min(len(self.X), 10 if two_d else 16)

    def __call__(self, q):
        q = np.asarray(q, float)[:2] if self.two_d else np.asarray(q, float)
        d, i = self.tree.query(q, k=self.k)
        d, i = np.atleast_1d(d), np.atleast_1d(i)
        h = max(float(d.max()), 1e-12)
        w = 1.0 / (d / h + 1e-3) ** 2
        A = np.column_stack([np.ones(len(i)), self.X[i] - q]) * np.sqrt(w)[:, None]
        B = self.vals[i] * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A, B, rcond=None)
        return coef[0], float(d.min())


def results(study: dict, mesh, plan: dict, fields: dict) -> tuple[dict, list]:
    rho, nu = plan["rho"], plan["nu"]
    ref = study.get("reference", {})
    p_ref = float(ref.get("p", 0.0))
    cofr = np.asarray(ref.get("cofr", [0, 0, 0]), float) * SCALE
    two_d = bool(study.get("2d"))
    external = study.get("domain", {}).get("kind") == "external"
    g = mesh.geometry()
    ni = mesh.ninternal
    dc = mesh.delta_coeffs_boundary() / SCALE                     # 1/m
    Uc, pc = np.asarray(fields["U"], float), np.asarray(fields["p"], float)
    en = plan.get("energy")                     # energy: p absolute (Pa), phi mass flux (kg/s), wss dynamic (Pa)
    if en:
        p_op, cp, T_op = en["p_op"], en["cp"], en["T_op"]
        gas = en["kind"] == "gas"
        rho_c = np.asarray(fields.get("rho", np.full(len(pc), rho)), float)
        Tc = np.asarray(fields.get("T", np.full(len(pc), T_op)), float)
        pc_pa = pc - p_op
    else:
        pc_pa = pc * rho
    heat, heat_faces = {}, []
    e_out, q_walls, e_scale = 0.0, 0.0, 0.0
    warns = []
    out = {"flow": {}, "pressure": {}, "forces": {}, "wall_shear": {}}
    totals = np.zeros(3)
    scale_m = 0.0
    ang, scale_a = np.zeros(3), 0.0
    face_forces = {}
    net_phi, inflow = 0.0, 0.0
    wall_faces = []          # (face indices, F per face) of every wall patch
    visc_cmp = [0.0, 0.0]
    yplus = {}
    probes_X, probes_V = [], []
    for P in mesh.patches:
        nm = P["name"]
        kind = plan["patches"][nm]["kind"]
        if kind == "empty":
            continue
        f = np.arange(P["start"], P["start"] + P["n"])
        own = mesh.owner[f]
        pf = fields["patches"][nm]
        Sf = g["Sf"][f] * SCALE ** 2
        A = np.linalg.norm(Sf, axis=1)
        Ub = np.asarray(pf["U"], float).reshape(-1, 3)
        phi = np.asarray(pf["phi"], float)
        nut = pf.get("nut")
        nut = np.asarray(nut, float) if nut is not None else 0.0
        if en:
            pb = np.asarray(pf["p"], float) - p_op
            rho_f = np.asarray(pf.get("rho") if pf.get("rho") is not None else rho_c[own], float)
            mueff = en["mu"] + rho_f * nut
            Tb = np.asarray(pf.get("T") if pf.get("T") is not None else Tc[own], float)
        else:
            pb = np.asarray(pf["p"], float) * rho
            rho_f = rho
            mueff = rho * (nu + nut)
        Fp = (pb - p_ref)[:, None] * Sf
        dU = Ub - Uc[own]
        if kind == "wall":       # no-slip: the normal velocity's normal derivative vanishes; keep the shear
            nh = Sf / A[:, None]
            dU = dU - np.einsum("ij,ij->i", dU, nh)[:, None] * nh
        Fv_ours = -np.asarray(mueff)[..., None] * dU * dc[f - ni][:, None] * A[:, None]
        wss = pf.get("wss")
        if kind == "wall" and wss is not None:
            Fv = -(1.0 if en else rho) * np.asarray(wss, float) * A[:, None]
            visc_cmp[0] += float(np.linalg.norm(Fv.sum(axis=0)))
            visc_cmp[1] += float(np.linalg.norm(Fv.sum(axis=0) - Fv_ours.sum(axis=0)))
        else:
            Fv = Fv_ours
        Mo = (1.0 if en else rho) * Ub * phi[:, None]
        F = Fp + Fv
        totals += (Mo + F).sum(axis=0)
        scale_m = max(scale_m, float(np.linalg.norm(F.sum(axis=0))), float(np.linalg.norm(Mo.sum(axis=0))),
                      float(np.linalg.norm(Fp.sum(axis=0))))
        face_forces[nm] = (f, F)
        rf = g["Cf"][f] * SCALE - cofr
        Ma, Fa = np.cross(rf, Mo).sum(axis=0), np.cross(rf, F).sum(axis=0)
        ang += Ma + Fa
        scale_a = max(scale_a, float(np.linalg.norm(Ma)), float(np.linalg.norm(Fa)))
        net_phi += float(phi.sum())
        inflow += float(np.clip(-phi, 0, None).sum())
        At = float(A.sum())
        if en:
            mdot = -float(phi.sum())
            Q = -float((phi / rho_f).sum())
        else:
            Q = -float(phi.sum())
            mdot = rho * Q
        flow = {"Q": _r(Q), "Q_lpm": _r(Q * 60000.0), "mdot": _r(mdot), "U_mean": _r(Q / At), "area_mm2": _r(At / SCALE ** 2),
                "angular_momentum_in": _r(-Ma)}
        if en and gas:
            mach_f = np.linalg.norm(Ub, axis=1) / np.sqrt(en["gamma"] * en["R"] * Tb)
            pt = (pb + p_op) * (1 + 0.5 * (en["gamma"] - 1) * mach_f ** 2) ** (en["gamma"] / (en["gamma"] - 1)) - p_op
        else:
            pt = pb + 0.5 * rho_f * np.einsum("ij,ij->i", Ub, Ub)
        if en:
            aw = np.abs(phi)
            if aw.sum() > 0:
                flow["T_mean_C"] = _r((Tb * aw).sum() / aw.sum() - 273.15)
            if gas:
                flow["mach"] = _r((mach_f * A).sum() / At)
            if kind in ("inlet", "outlet", "opening", "slip"):
                e_flux = phi * (cp * (Tb - T_op) + 0.5 * np.einsum("ij,ij->i", Ub, Ub))
                e_out += float(e_flux.sum())
                e_scale = max(e_scale, float(np.abs(phi * cp * (Tb - T_op)).sum()))
            if kind == "wall":
                whf = pf.get("whf")
                if whf is not None:
                    qf = np.asarray(whf, float)
                else:          # ours: k_eff (T_wall - T_cell) deltaCoeff, positive into the fluid
                    at = pf.get("alphat")
                    keff = en["k"] + (cp * np.asarray(at, float) if at is not None else 0.0)
                    qf = keff * (Tb - Tc[own]) * dc[f - ni]
                Qw = float((qf * A).sum())
                q_walls += Qw
                e_scale = max(e_scale, abs(Qw))
                heat[nm] = {"Q_W": _r(Qw), "q_mean": _r(Qw / At), "T_mean_C": _r((Tb * A).sum() / At - 273.15),
                            "T_max_C": _r(Tb.max() - 273.15), "T_min_C": _r(Tb.min() - 273.15)}
                heat_faces.append((f, qf * A, Tb, A))
        aphi = np.abs(phi)
        pres = {"p_mean": _r((pb * A).sum() / At)}
        if aphi.sum() > 0:
            pres["p_total"] = _r((pt * aphi).sum() / aphi.sum())
        if kind in ("inlet", "outlet", "opening"):
            out["flow"][nm] = flow
        out["pressure"][nm] = pres
        if kind == "wall":
            r = (g["Cf"][f] * SCALE) - cofr
            Mm = np.cross(r, F).sum(axis=0)
            Ft, Fpt, Fvt = F.sum(axis=0), Fp.sum(axis=0), Fv.sum(axis=0)
            out["forces"][nm] = {**_vec3(Ft), **_vec3(Mm, "m"), "pressure": _r(Fpt), "viscous": _r(Fvt)}
            tau = np.linalg.norm(Fv - np.einsum("ij,ij->i", Fv, Sf / A[:, None])[:, None] * (Sf / A[:, None]), axis=1) / A
            out["wall_shear"][nm] = {"mean": _r((tau * A).sum() / At), "max": _r(tau.max())}
            wall_faces.append((f, F, r))
            yp = pf.get("yPlus")
            if yp is not None and len(yp):
                yp = np.asarray(yp, float)
                yplus[nm] = {"min": _r(yp.min()), "mean": _r((yp * A).sum() / At), "max": _r(yp.max())}
        probes_X.append(g["Cf"][f])
        probes_V.append(np.column_stack([Ub, pb] + ([Tb - 273.15] if en else [])))
    # totals over walls
    if wall_faces:
        tags = mesh.btag
        allF = np.concatenate([F for _, F, _ in wall_faces])
        allf = np.concatenate([f for f, _, _ in wall_faces])
        allr = np.concatenate([r for _, _, r in wall_faces])
        if external:
            ptags = {t for t, i in mesh.tag_info.items() if i.get("on") == "part"}
            sel = np.isin(tags[allf - ni], list(ptags))
            key = "part"
        else:
            sel = np.ones(len(allf), bool)
            key = "all_walls"
        Fsum = allF[sel].sum(axis=0)
        out["forces"][key] = {**_vec3(Fsum), **_vec3(np.cross(allr[sel], allF[sel]).sum(axis=0), "m")}
        # coefficients
        if ref.get("velocity") and ref.get("area"):
            q = 0.5 * rho * float(ref["velocity"]) ** 2
            Aref = float(ref["area"]) * SCALE ** 2
            d = _unit(ref["drag_dir"]) if "drag_dir" in ref else None
            if d is None:
                inl = [n for n, v in plan["patches"].items() if v["kind"] == "inlet"]
                if inl:
                    d =_unit(np.asarray(fields["patches"][inl[0]]["U"]).reshape(-1, 3).mean(axis=0))
                else:
                    d = np.array([1.0, 0, 0])
            if "lift_dir" in ref:
                l = _unit(ref["lift_dir"])
            elif two_d:
                l = _unit([-d[1], d[0], 0.0])
            else:
                z = np.array([0, 0, 1.0])
                l = z - (z @ d) * d
                l = _unit(l) if np.linalg.norm(l) > 1e-6 else np.array([0, 1.0, 0])
            out["coefficients"] = {"cd": _r(Fsum @ d / (q * Aref)), "cl": _r(Fsum @ l / (q * Aref)),
                                   "drag_dir": _r(d), "lift_dir": _r(l), "q_Pa": _r(q), "area_m2": _r(Aref)}
            if ref.get("length"):
                Lr = float(ref["length"]) * SCALE
                Mt = np.cross(allr[sel], allF[sel]).sum(axis=0)
                out["coefficients"]["cm"] = _r(Mt @ np.cross(d, l) / (q * Aref * Lr))
        # loads for fea-translator (external flow: one force per part face)
        if external:
            from .select import mesh_groups
            groups = mesh_groups(mesh, True)
            loads, total = [], np.zeros(3)
            for t in sorted(set(int(x) for x in tags[allf[sel] - ni])):
                m = tags[allf - ni] == t
                Ft = allF[m].sum(axis=0)
                total += Ft
                loads.append({"kind": "force", "name": f"cfd_{t}", "faces": {"pts": [groups.groups[t].point_on()]},
                              **_vec3(Ft)})
            out["feat_loads"] = {"bcs": loads, "total": _r(total),
                                 "note": "fea-translator force bcs (N) on the same part: add supports; "
                                         "each face's total fluid force (pressure + shear), spread uniformly"}
    # pressure drop
    kinds = {n: v["kind"] for n, v in plan["patches"].items()}
    ins = [n for n, k in kinds.items() if k == "inlet"]
    outs = [n for n, k in kinds.items() if k in ("outlet", "opening")]

    def amean(names, key):
        num = den = 0.0
        for n in names:
            P = mesh.patch(n)
            f = np.arange(P["start"], P["start"] + P["n"])
            A = np.linalg.norm(g["Sf"][f], axis=1) * SCALE ** 2
            Ub = np.asarray(fields["patches"][n]["U"]).reshape(-1, 3)
            pb = np.asarray(fields["patches"][n]["p"], float) * rho
            if key == "p":
                num += float((pb * A).sum())
                den += float(A.sum())
            else:
                w = np.abs(np.asarray(fields["patches"][n]["phi"], float))
                num += float(((pb + 0.5 * rho * np.einsum("ij,ij->i", Ub, Ub)) * w).sum())
                den += float(w.sum())
        return num / den if den > 0 else None
    if ins and outs:
        out["pressure_drop"] = _r(amean(ins, "p") - amean(outs, "p"))
        a, b = amean(ins, "pt"), amean(outs, "pt")
        if a is not None and b is not None:
            out["total_pressure_drop"] = _r(a - b)
            out["head_loss_m"] = _r((a - b) / (rho * 9.80665))
    # field extremes and probes
    um = np.linalg.norm(Uc, axis=1)
    k = int(np.argmax(um))
    out["max_velocity"] = {"u": _r(um[k]), "at": _r(g["C"][k], 6)}
    if study.get("probes"):
        X = np.concatenate([g["C"]] + probes_X)
        V = np.concatenate([np.column_stack([Uc, pc_pa] + ([Tc - 273.15] if en else []))] + probes_V)
        pr = _Probe(X, V, two_d)
        if two_d:
            h = float(np.sqrt(g["V"].mean() / max(float(np.ptp(mesh.points[:, 2])), 1e-300)))
        else:
            h = float(g["V"].mean() ** (1 / 3))
        out["probes"] = {}
        for p in study["probes"]:
            pts = [p["at"]] if "at" in p else np.linspace(p["line"]["from"], p["line"]["to"], int(p["line"].get("n", 11)))
            vals, far = [], 0.0
            for q in pts:
                v, dmin = pr(q)
                vals.append(v)
                far = max(far, dmin)
            vals = np.array(vals)
            res = {"ux": vals[:, 0], "uy": vals[:, 1], "uz": vals[:, 2], "u": np.linalg.norm(vals[:, :3], axis=1),
                   "p": vals[:, 3]}
            if en:
                res["T"] = vals[:, 4]
                if gas:
                    res["mach"] = res["u"] / np.sqrt(en["gamma"] * en["R"] * (vals[:, 4] + 273.15))
            if "at" in p:
                res = {k2: _r(v[0]) for k2, v in res.items()}
            else:
                res = {k2: _r(v) for k2, v in res.items()}
                res["points"] = _r(np.asarray(pts), 6)
            out["probes"][p["name"]] = res
            if far > 5 * h:
                warns.append(f"probe {p['name']!r}: {far:.3g} mm from the nearest cell centre (outside the fluid?)")
    if yplus:
        out["y_plus"] = yplus
        if plan["model"] != "laminar":
            mean = np.mean([v["mean"] for v in yplus.values()])
            if mean < 25 or mean > 400:
                warns.append(f"mean y+ on the walls is {mean:.3g}: wall functions want 30-300 at the first cell "
                             "(change mesh.size or mesh.layers.first)")
    # energy: gravity in the momentum balances, heat results, energy balance, loads for feat's thermal analysis
    if en:
        V = g["V"] * SCALE ** 3
        if en.get("buoyancy"):
            gv = np.asarray(en["gravity"], float)
            body = rho_c[:, None] * V[:, None] * gv[None, :]
            totals -= body.sum(axis=0)
            ang -= np.cross(g["C"] * SCALE - cofr, body).sum(axis=0)
            scale_m = max(scale_m, float(np.linalg.norm(body.sum(axis=0))))
        out["heat"] = heat
        out["T_min_C"], out["T_max_C"] = _r(Tc.min() - 273.15), _r(Tc.max() - 273.15)
        if gas:
            mc = np.linalg.norm(Uc, axis=1) / np.sqrt(en["gamma"] * en["R"] * Tc)
            k = int(np.argmax(mc))
            out["max_mach"] = {"mach": _r(mc[k]), "at": _r(g["C"][k], 6)}
        Tref = ref.get("T")
        if Tref is not None:
            for nm, h in heat.items():
                dT = h["T_mean_C"] - float(Tref)
                if abs(dT) > 1e-9:
                    h["h"] = _r(h["q_mean"] / dT)
                    if ref.get("length"):
                        h["Nu"] = _r(h["h"] * float(ref["length"]) * SCALE / en["k"])
        e_err = abs(e_out - q_walls) / e_scale if e_scale > 0 else 0.0
        energy_check = {"out_W": _r(e_out), "walls_in_W": _r(q_walls), "error": _r(e_err)}
        if e_err > 0.02:
            warns.append(f"energy balance off by {e_err:.2%} (not converged?)")
        if external and heat_faces:
            ptags = {t for t, i in mesh.tag_info.items() if i.get("on") == "part"}
            from .select import mesh_groups
            groups = mesh_groups(mesh, True)
            T_inf = float(Tref) if Tref is not None else (
                np.mean([v["T_mean_C"] for n, v in out["flow"].items()
                         if plan["patches"][n]["kind"] == "inlet" and "T_mean_C" in v]) if any(
                    plan["patches"][n]["kind"] == "inlet" for n in out["flow"]) else T_op - 273.15)
            allf = np.concatenate([f for f, _, _, _ in heat_faces])
            allQ = np.concatenate([q for _, q, _, _ in heat_faces])
            allT = np.concatenate([t for _, _, t, _ in heat_faces])
            allA = np.concatenate([a for _, _, _, a in heat_faces])
            conv, flux, tot = [], [], 0.0
            for t in sorted(set(int(x) for x in mesh.btag[allf - ni]) & ptags):
                m = mesh.btag[allf - ni] == t
                Qt, At_ = float(allQ[m].sum()), float(allA[m].sum())
                Tw = float((allT[m] * allA[m]).sum() / At_) - 273.15
                tot += Qt
                pt_ = groups.groups[t].point_on()
                flux.append({"kind": "heat_flux", "name": f"cfd_{t}", "faces": {"pts": [pt_]}, "q": _r(-Qt / At_)})
                if abs(Tw - T_inf) > 1e-6:
                    conv.append({"kind": "convection", "name": f"cfd_{t}", "faces": {"pts": [pt_]},
                                 "h": _r(Qt / At_ / (Tw - T_inf)), "T_inf": _r(T_inf)})
            out["feat_thermal"] = {"convection": conv, "heat_flux": flux, "total_W": _r(tot), "T_inf": _r(T_inf),
                                   "note": "fea-translator thermal bcs on the same part. convection: h per face from "
                                           "the CFD heat and wall temperature, to T_inf (best when the solid's "
                                           "temperature is unknown); heat_flux: the CFD heat leaving the fluid into "
                                           "the part (W/m2, add a temperature somewhere). Q_W > 0: heat into the fluid."}
    # rotating zones (MRF): the frame's source rho Omega x U enters the momentum balances; torque, power, pump
    for zn, zp in plan.get("zones", {}).items():
        cells = mesh.zones[zn]
        a = np.asarray(zp["axis"], float)
        Om = zp["omega_rad_s"] * a
        V = g["V"][cells] * SCALE ** 3
        src = rho * np.cross(Om, Uc[cells]) * V[:, None]
        totals += src.sum(axis=0)
        scale_m = max(scale_m, float(np.linalg.norm(src.sum(axis=0))))
        ang += np.cross(g["C"][cells] * SCALE - cofr, src).sum(axis=0)
        c0 = np.asarray(zp["center_mm"], float) * SCALE
        torque = 0.0
        for wn in zp["rotating_walls"]:
            f, F = face_forces[wn]
            torque += float(np.cross(g["Cf"][f] * SCALE - c0, F).sum(axis=0) @ a)
        power = -torque * zp["omega_rad_s"]
        out.setdefault("rotating", {})[zn] = {"torque_Nm": _r(torque), "power_W": _r(power),
                                              "rpm": _r(zp["omega_rad_s"] * 60 / (2 * np.pi)), "cells": zp["cells"],
                                              "rotating_walls": zp["rotating_walls"]}
        if not zp["rotating_walls"]:
            warns.append(f"rotating zone {zn!r} has no rotating walls (give the impeller faces rotation: {zn!r})")
    if out.get("rotating") and "total_pressure_drop" in out:
        P = sum(z["power_W"] for z in out["rotating"].values())
        H = -out["total_pressure_drop"] / (rho * 9.80665)
        Qin = sum(v["Q"] for n, v in out["flow"].items() if plan["patches"][n]["kind"] == "inlet")
        out["pump"] = {"head_m": _r(H), "flow_m3h": _r(Qin * 3600), "power_W": _r(P),
                       "efficiency": _r(rho * 9.80665 * Qin * H / P) if P > 0 else None,
                       "note": "hydraulic efficiency of the modelled passages (frozen rotor, steady)"}
    # checks
    vol = float(g["V"].sum()) * SCALE ** 3
    qscale = max(inflow, plan.get("U_ref", 1.0) * vol ** (2 / 3) * 1e-3 * (rho if en else 1.0))
    mass_err = abs(net_phi) / qscale if qscale > 0 else 0.0
    mom_err = float(np.linalg.norm(totals)) / scale_m if scale_m > 0 else 0.0
    checks = {"mass_balance": {"inflow_m3s": _r(inflow), "net_m3s": _r(net_phi), "error": _r(mass_err)},
              "momentum_balance": {"residual_N": _r(totals), "scale_N": _r(scale_m), "error": _r(mom_err)},
              **({"energy_balance": energy_check} if en else {}),
              "angular_momentum_balance": {"residual_Nm": _r(ang), "scale_Nm": _r(scale_a),
                                           "error": _r(float(np.linalg.norm(ang)) / scale_a if scale_a > 0 else 0.0),
                                           "about": _r(cofr / SCALE)}}
    if visc_cmp[0] > 0:
        checks["wall_shear_vs_ours"] = _r(visc_cmp[1] / visc_cmp[0])
        # our estimate leaves out OpenFOAM's dev2(T(grad U)) term, which is not small on tetrahedra (~20 %
        # there); the momentum balance already uses OpenFOAM's values, so only a gross mismatch is a warning
        if visc_cmp[1] / visc_cmp[0] > 0.5:
            warns.append(f"OpenFOAM's wall shear and ours (snGrad) differ by {visc_cmp[1] / visc_cmp[0]:.1%}: "
                         "check the wall cells (very coarse or skewed?)")
    for n, pl in plan.get("inlets", {}).items():
        got = -float(np.asarray(fields["patches"][n]["phi"], float).sum())
        if abs(got - pl["Q"]) > 1e-3 * abs(pl["Q"]):
            warns.append(f"inlet {n}: flow {got:.6g} m3/s, but the case prescribes {pl['Q']:.6g}")
        checks.setdefault("inlet_flow_error", {})[n] = _r((got - pl["Q"]) / pl["Q"])
    if mass_err > 1e-3:
        warns.append(f"mass balance off by {mass_err:.2%} of the inflow (not converged?)")
    if fields.get("source") == "reference solver":
        checks["momentum_balance"]["note"] = ("indicative only: the MAC solver's boundary values (wall pressure = "
                                              "cell value) do not telescope exactly like OpenFOAM's face fluxes")
    elif mom_err > 0.03:
        warns.append(f"momentum balance off by {mom_err:.2%} (not converged, or coarse mesh at open boundaries)")
    out["checks"] = checks
    return out, warns


def get_path(d, path: str):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict):
            if k not in cur:
                raise KeyError(path)
            cur = cur[k]
        elif isinstance(cur, list):
            cur = cur[int(k)]
        else:
            raise KeyError(path)
    return cur


def check_expect(res: dict, expect: list) -> list:
    out = []
    for e in expect or []:
        q = e["quantity"]
        try:
            got = float(get_path(res, q))
        except (KeyError, ValueError, IndexError, TypeError):
            out.append({"quantity": q, "expected": e["value"], "got": None, "passed": False,
                        "why": "no such result (see the result keys)"})
            continue
        v = float(e["value"])
        if "abs" in e:
            ok, err = abs(got - v) <= float(e["abs"]), abs(got - v)
        else:
            tol = float(e.get("tol", 0.02))
            err = abs(got - v) / abs(v) if v != 0 else abs(got)
            ok = err <= tol
        out.append({"quantity": q, "expected": v, "got": _r(got), "error": _r(err), "passed": bool(ok)})
    return out
