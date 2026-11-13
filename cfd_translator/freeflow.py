"""freeflow.py — a transient pump run whose inlet flow is free (solve.transient.free_flow): the flow is held while
the run settles, then the inlet holds the total pressure that flow needed and the flow follows the pressure.
Version: 2026.11.13.5

Why (S50, the hand-built v2506 sliding run continued 5 revolutions, 2026.11.13): with the flow held, the head swung
11-21 m and the eye pressure +-48 kPa (rms 19.7 kPa); with the inlet's total pressure held instead, the head was steady,
the flow pulsed +-2 % and the eye pressure +-19 kPa (rms 8.7 kPa), the means unchanged (15.34 / 15.37 m, 1660 /
1648 W). A held flow puts the whole pulsation on the suction side.

    settings(study, plan, folder) -> cfg | None
    errors(study) -> [str]
    p0_from_monitor(case, inlet, t0, t1, rho) -> (Pa, info)      the held phase's mean inlet total pressure
    switch_inlet(case, inlet, p0_kin) -> [files]                  flow-held inlet -> total pressure (latest times)
    set_end_time(case, t)
    report(case, plan, cfg, state, study, inlet) -> (block, warnings)

p0 comes, in order, from free_flow.p0 (Pa gauge); from the run a study starts from (init_from or continue of a
finished transient: its averaged inlet total pressure, results.pressure.<inlet>.p_total); else from this run's own
held phase: after {revolutions: 4} (or {time: s}), its last window {revolutions: 2} (or {time: s}). Not from a
steady (frozen-rotor) start: its head differs from the sliding run's (S50: 16.2 against 15.4 m) and would shift the
flow. After the switch the flow settles over a few revolutions (S50: 19.62, 19.87, 19.99, 20.20 m3/h for 20.00):
the report gives the flow per revolution and its error; a run should have at least MIN_FREE revolutions free.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

MIN_FREE = 4                     # revolutions after the switch before the averages are judged
FLOW_TOL = 0.01                  # the last free revolution's mean flow within 1 % of the target


def _ff(study: dict):
    return ((study.get("solve") or {}).get("transient") or {}).get("free_flow")


def errors(study: dict) -> list[str]:
    ff = _ff(study)
    if ff is None or ff is False:
        return []
    if ff is not True and not isinstance(ff, dict):
        return ["solve.transient.free_flow: true or {after: {revolutions | time}, window: {revolutions | time}, p0 (Pa)}"]
    errs = []
    d = {} if ff is True else ff
    bad = set(d) - {"after", "window", "p0"}
    if bad:
        errs.append(f"solve.transient.free_flow: unknown keys {sorted(bad)} (after, window, p0)")
    for k in ("after", "window"):
        v = d.get(k)
        if v is None:
            continue
        if not (isinstance(v, dict) and len(v) == 1 and set(v) <= {"revolutions", "time"} and
                isinstance(next(iter(v.values())), (int, float)) and next(iter(v.values())) > 0):
            errs.append(f"solve.transient.free_flow.{k}: {{revolutions: n}} or {{time: s}}")
    if "p0" in d and (not isinstance(d["p0"], (int, float)) or isinstance(d["p0"], bool)):
        errs.append("solve.transient.free_flow.p0: the inlet's total pressure, Pa gauge")
    from .transient import sliding_zone
    sl = sliding_zone(study)
    for k in ("after", "window"):
        if isinstance(d.get(k), dict) and "revolutions" in d[k] and not (sl or study.get("rotating")):
            errs.append(f"solve.transient.free_flow.{k}.revolutions: needs a rotating zone; give time (s)")
    if not sl and not study.get("rotating") and "p0" not in d and not (isinstance(d.get("after"), dict)
                                                                       and "time" in d["after"]):
        errs.append("solve.transient.free_flow: without a rotating zone give after: {time: s} (or p0)")
    inl = [b for b in study.get("bcs") or [] if b.get("kind") == "inlet"]
    if len(inl) != 1:
        errs.append("solve.transient.free_flow: one inlet")
    elif not ({"flow_rate", "velocity", "mass_flow"} & set(inl[0])):
        errs.append("solve.transient.free_flow: the inlet must hold a flow (flow_rate, velocity or mass_flow) to free")
    if study.get("energy") or study.get("cavitation"):
        errs.append("solve.transient.free_flow: incompressible single-phase runs only")
    a = d.get("after") or {"revolutions": 4}
    w = d.get("window") or {"revolutions": 2}
    if not (isinstance(a, dict) and isinstance(w, dict) and set(a) <= {"revolutions", "time"} and
            set(w) <= {"revolutions", "time"} and a and w):
        return errs
    if set(a) == set(w) and next(iter(w.values())) > next(iter(a.values())):
        errs.append("solve.transient.free_flow: the window lies within 'after' (window <= after)")
    # the averages belong to the free phase: average_from at or after the switch (a run that holds its own flow)
    tr = (study.get("solve") or {}).get("transient") or {}
    z = sl or next((z for z in study.get("rotating") or [] if isinstance(z, dict) and z.get("rpm")), None)
    own = "p0" not in d and not tr.get("continue") and not tr.get("init_from") and \
        not (study.get("solve") or {}).get("init_from")
    if own and isinstance(tr.get("end_time"), (int, float)):
        T = float(tr["end_time"])
        hold = float(a["time"]) if "time" in a else (float(a["revolutions"]) * 60.0 / abs(float(z["rpm"])) if z else None)
        if hold is not None:
            avf = float(tr.get("average_from", T / 3))
            if avf < hold - 1e-9:
                per = 60.0 / abs(float(z["rpm"])) if z else None
                errs.append(f"solve.transient.average_from ({avf:g} s) lies before the flow is freed ({hold:g} s): the "
                            "averages must belong to the free phase; set average_from to "
                            f"{hold + (MIN_FREE * per if per else 0):.6g} s or later (the flow settles over about "
                            f"{MIN_FREE} revolutions after the switch)" + (f", and end_time a few revolutions after "
                                                                           "that" if per else ""))
    return errs


def _source_p0(study: dict, folder: Path, inlet: str) -> tuple[float | None, str | None]:
    """The averaged inlet total pressure (Pa gauge) of the finished transient run this one continues or starts from."""
    from .transient import init_ref, init_source, source_folder
    tr = (study.get("solve") or {}).get("transient") or {}
    ref = tr.get("continue")
    src = source_folder(ref) if ref else None
    if src is None:
        r, _ = init_ref(study)
        src = init_source(r, Path(folder).parent) if r else None
    if src is None:
        return None, None
    f = Path(src) / f"{Path(src).name}.json"
    try:
        res = json.loads(f.read_text())
    except (OSError, ValueError):
        return None, None
    if not (res.get("solve") or {}).get("transient"):
        return None, None                     # a steady (frozen-rotor) start: its head is not the sliding run's
    pt = (((res.get("results") or {}).get("pressure") or {}).get(inlet) or {}).get("p_total")
    if pt is None:
        return None, None
    return float(pt), f"the averaged inlet total pressure of {src.name} (its time averages)"


def settings(study: dict, plan: dict, folder: Path) -> dict | None:
    ff = _ff(study)
    if not ff:
        return None
    d = {} if ff is True else dict(ff)
    tr = plan.get("transient") or {}
    period = tr.get("period")
    if period is None:
        z = next((z for z in study.get("rotating") or [] if isinstance(z, dict) and z.get("rpm")), None)
        period = 60.0 / abs(float(z["rpm"])) if z else None

    def span(v, default_rev):
        v = v or {"revolutions": default_rev}
        if "time" in v:
            return float(v["time"]), f"{float(v['time']):g} s"
        return float(v["revolutions"]) * float(period), f"{float(v['revolutions']):g} revolutions"
    inl = next(b for b in study["bcs"] if b.get("kind") == "inlet")
    from .format import bc_names
    name = next(n for n, b in zip(bc_names(study), study["bcs"]) if b is inl)
    cfg = {"inlet": name, "period": period, "rho": float(plan.get("rho") or 998.2),
           "Q_target": float(((plan.get("inlets") or {}).get(name) or {}).get("Q") or 0.0) or None}
    if "p0" in d:
        cfg.update(p0_Pa=float(d["p0"]), p0_from="free_flow.p0 (given)", hold_s=0.0)
        return cfg
    if "after" not in d:
        p0, why = _source_p0(study, folder, name)
        if p0 is not None:
            cfg.update(p0_Pa=p0, p0_from=why, hold_s=0.0)
            return cfg
    cfg["hold_s"], cfg["hold"] = span(d.get("after"), 4)
    cfg["window_s"], cfg["window"] = span(d.get("window"), 2)
    cfg["p0_Pa"], cfg["p0_from"] = None, f"this run's held phase: the mean over its last {cfg['window']}"
    return cfg


def _weighted_mean(t: np.ndarray, v: np.ndarray) -> float:
    w = np.gradient(t) if len(t) > 1 else np.ones(1)
    w = np.maximum(w, 0)
    return float((w * v).sum() / max(w.sum(), 1e-300))


def p0_from_monitor(case: Path, inlet: str, t0: float, t1: float, rho: float) -> tuple[float | None, dict]:
    """The time-weighted mean of the inlet's flow-weighted total pressure (monitor cfdtP_<inlet>) over [t0, t1], Pa.
    OpenFOAM's pressure function object gives total(p) in Pa with rhoInf (p kinematic): told from the static column."""
    from .average import _pressure_columns, read_monitor
    got = read_monitor(case, f"cfdtP_{inlet}")
    if got is None:
        return None, {"why": f"no postProcessing/cfdtP_{inlet}"}
    t, header, vals = got
    st, tt = _pressure_columns(header, vals.shape[1])
    if tt is None:
        return None, {"why": "no total(p) column"}
    m = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)
    if m.sum() < 5:
        return None, {"why": f"{int(m.sum())} samples in [{t0:g}, {t1:g}] s"}
    tot = vals[m, tt]
    stat = vals[m, st] if st is not None else None
    pa = True
    if stat is not None and np.median(np.abs(stat)) > 0:
        pa = np.median(np.abs(tot)) > np.sqrt(rho) * np.median(np.abs(stat))
    mean = _weighted_mean(t[m], tot)
    p0 = mean if pa else mean * rho
    return float(p0), {"samples": int(m.sum()), "window_s": [float(t[m][0]), float(t[m][-1])],
                       "std_Pa": float(np.std(tot) * (1 if pa else rho)), "units": "Pa" if pa else "kinematic x rho"}


def set_end_time(case: Path, t: float) -> None:
    f = Path(case) / "system" / "controlDict"
    s = f.read_text()
    f.write_text(re.sub(r"(?m)^(\s*endTime\s+)[^;]+;", lambda m: f"{m.group(1)}{t:.10g};", s, count=1))


def _latest_dirs(case: Path) -> list[Path]:
    out = []
    for d in [Path(case)] + sorted(Path(case).glob("processor*")):
        best, bt = None, -1.0
        for p in d.iterdir() if d.is_dir() else []:
            if p.is_dir():
                try:
                    t = float(p.name)
                except ValueError:
                    continue
                if t > bt:
                    best, bt = p, t
        if best is not None and (best / "U").is_file():
            out.append(best)
    return out


def _block(text: str, patch: str) -> tuple[int, int] | None:
    """The [start, end) of a patch's entry in a field file's boundaryField (its name to its closing brace)."""
    k = text.find("boundaryField")
    if k < 0:
        return None
    m = re.compile(r"(?m)^[ \t]*" + re.escape(patch) + r"\s*\{").search(text, k)
    if not m:
        return None
    i, depth = text.index("{", m.start()), 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return m.start(), j + 1
    return None


def _value_entry(block: str) -> str | None:
    """'value ...;' of a patch entry (a nonuniform list's parentheses included)."""
    m = re.search(r"\bvalue\b", block)
    if not m:
        return None
    depth = 0
    for j in range(m.end(), len(block)):
        c = block[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ";" and depth == 0:
            return block[m.start():j + 1]
    return None


def switch_inlet(case: Path, inlet: str, p0_kin: float) -> list[str]:
    """The inlet's U and p entries in the latest time of the case and of every processor directory: the velocity
    from the pressure (pressureInletOutletVelocity), p held at the total pressure p0 (totalPressure, kinematic)."""
    done = []
    for d in _latest_dirs(case):
        for fld, new in (("U", lambda v: "type            pressureInletOutletVelocity;\n        " + (v or
                                                                                                  "value           uniform (0 0 0);")),
                         ("p", lambda v: f"type            totalPressure;\n        p0              uniform {p0_kin:.10g};\n        "
                          + (v or f"value           uniform {p0_kin:.10g};"))):
            f = d / fld
            if not f.is_file():
                continue
            text = f.read_text(errors="surrogateescape")
            b = _block(text, inlet)
            if b is None:
                continue
            old = text[b[0]:b[1]]
            body = "    " + inlet + "\n    {\n        " + new(_value_entry(old)) + "\n    }"
            f.write_text(text[:b[0]] + body + text[b[1]:], errors="surrogateescape")
            done.append(str(f.relative_to(case)))
    return done


def _per_period(t: np.ndarray, v: np.ndarray, t0: float, period: float) -> list[dict]:
    out = []
    k = 0
    while t0 + (k + 1) * period <= t[-1] + 1e-9:
        m = (t >= t0 + k * period - 1e-12) & (t < t0 + (k + 1) * period - 1e-12)
        if m.sum() >= 3:
            x = v[m]
            mean = _weighted_mean(t[m], x)
            out.append({"rev": k + 1, "mean": float(f"{mean:.6g}"), "min": float(f"{x.min():.6g}"),
                        "max": float(f"{x.max():.6g}"), "rms": float(f"{np.sqrt(np.mean((x - mean) ** 2)):.4g}")})
        k += 1
    return out


def report(case: Path, plan: dict, cfg: dict, state: dict, study: dict) -> tuple[dict, list]:
    """results.free_flow: p0 and where from, when the flow was freed, the flow per revolution (mean, min, max, rms,
    error against the held flow), the inlet's and outlet's pressures and the probes' per revolution."""
    from .average import _pressure_columns, read_monitor
    warns = []
    ff = state.get("free_flow") or {}
    inlet, rho, per = cfg["inlet"], cfg["rho"], cfg.get("period")
    t_sw = ff.get("switched_at")
    out = {"p0_Pa": ff.get("p0_Pa"), "p0_from": ff.get("p0_from"), "switched_at_s": t_sw,
           **({"p0_window": ff["p0_window"]} if ff.get("p0_window") else {}),
           "held_flow_m3h": float(f"{cfg['Q_target'] * 3600:.5g}") if cfg.get("Q_target") else None}
    if t_sw is None:
        out["note"] = "the flow was not freed yet (the run ended within its held phase)"
        return out, warns
    span = per or max(cfg.get("window_s") or 0.0, 1e-9)
    q = read_monitor(case, f"cfdtQ_{inlet}")
    if q is not None:
        t, _h, v = q
        sel = t >= t_sw - 1e-12
        if sel.sum() >= 3:
            Q = -v[sel, 0] * 3600.0                     # phi out of the domain is positive: the inlet's is negative
            revs = _per_period(t[sel], Q, t_sw, span)
            for r in revs:
                if cfg.get("Q_target"):
                    r["error_pct"] = float(f"{100 * (r['mean'] - cfg['Q_target'] * 3600) / (cfg['Q_target'] * 3600):.3g}")
                r["pulsation_pct"] = float(f"{100 * (r['max'] - r['min']) / 2 / max(abs(r['mean']), 1e-300):.3g}")
            out["flow_m3h_by_revolution"] = revs
            if len(revs) < MIN_FREE:
                warns.append(f"free_flow: {len(revs)} revolution(s) after the flow was freed, fewer than {MIN_FREE}: "
                             "the flow is still settling (S50: 19.62, 19.87, 19.99, 20.20 m3/h for 20.00); run longer")
            if revs and "error_pct" in revs[-1] and abs(revs[-1]["error_pct"]) > 100 * FLOW_TOL:
                warns.append(f"free_flow: the last revolution's mean flow is {revs[-1]['error_pct']:+.2g} % off the "
                             f"held {out['held_flow_m3h']} m3/h: the point sits at another flow (head and power belong "
                             "to it); p0 (free_flow.p0) or more revolutions")
    ends = [n for n, p in (plan.get("patches") or {}).items() if p.get("kind") in ("inlet", "outlet")]
    pr = {}
    for n in ends:
        got = read_monitor(case, f"cfdtP_{n}")
        if got is None:
            continue
        t, header, v = got
        st, tt = _pressure_columns(header, v.shape[1])
        sel = t >= t_sw - 1e-12
        if sel.sum() < 3 or st is None:
            continue
        rec = {"static_Pa": _per_period(t[sel], v[sel, st] * rho, t_sw, span)}
        if tt is not None:
            tot = v[sel, tt]
            pa = np.median(np.abs(tot)) > np.sqrt(rho) * max(np.median(np.abs(v[sel, st])), 1e-300)
            rec["total_Pa"] = _per_period(t[sel], tot if pa else tot * rho, t_sw, span)
        pr[n] = rec
    if pr:
        out["pressure_by_revolution"] = pr
    probes = _probe_series(case, (plan.get("monitors") or {}).get("probes"))
    if probes:
        pb = {}
        for nm, (t, p) in probes.items():
            sel = t >= t_sw - 1e-12
            if sel.sum() >= 3:
                pb[nm] = _per_period(t[sel], p[sel] * rho, t_sw, span)
        if pb:
            out["probes_p_Pa_by_revolution"] = pb
    out["note"] = ("per revolution after the switch (whole revolutions; mean, min, max, rms about the mean): the flow "
                   "pulsates with the inlet's total pressure held; pressures in Pa gauge")
    return out, warns


def _probe_series(case: Path, names: list | None = None) -> dict:
    """{probe name: (time, p kinematic)} from postProcessing/cfdtProbes/<t>/p (the study's probes, every step)."""
    root = Path(case) / "postProcessing" / "cfdtProbes"
    if not root.is_dir():
        return {}
    rows: dict[float, list] = {}
    n = None
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: float(p.name) if _isnum(p.name) else 0):
        f = d / "p"
        if not f.is_file():
            continue
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            x = line.split()
            try:
                vals = [float(v) for v in x]
            except ValueError:
                continue
            if n is None:
                n = len(vals) - 1
            if len(vals) == n + 1:
                rows[vals[0]] = vals[1:]
    if not rows:
        return {}
    t = np.array(sorted(rows))
    a = np.array([rows[k] for k in t])
    if not names or len(names) != a.shape[1]:
        names = [f"probe{i}" for i in range(a.shape[1])]
    return {names[i]: (t, a[:, i]) for i in range(a.shape[1])}


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
