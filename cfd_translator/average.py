"""average.py — per-iteration monitors, window averages and the 'settled' criterion.
Version: 2026.10.22.4

Steady RANS of a machine with a rotating zone (MRF, frozen rotor) rarely reaches the residual target: the
blade-volute interaction keeps the residuals at 1e-3..1e-2 while the integral quantities only wobble around a
mean. The final iteration is then one snapshot of that wobble. Here OpenFOAM function objects record, every
iteration, the torque on the rotating walls and the flow-weighted (total) pressure on every inlet and outlet;
the last part of the run (the window) gives means, scatter and drift, and a study whose monitored quantities
neither drift nor scatter is 'settled' (an alternative to residual convergence in the reliability verdict).

    monitor_functions(study, plan, patches) -> (controlDict entries, monitors)   (case.py; plan["monitors"])
    read_monitor(case, name) -> (iterations, header, rows)                        (postProcessing/<name>/*/*.dat)
    window_stats(values, cfg) -> {mean, std, min, max, drift, scatter, n, ...}
    average_results(case, plan, res, study) -> (averaged block, warnings)         (updates res in place)

Settings (study.solve.average): true / false / {window: 0.25 (last fraction of the iterations), min_iterations:
500 (at least this many in the window), drift: 0.01, scatter: 0.02}. Default: on with a rotating zone, off
otherwise. drift = |mean of the window's second half - mean of its first half| / |mean|; scatter = std / |mean|.
Every monitored quantity must meet both for 'settled'.

Each series is checked against the final-iteration value computed from the fields (post.results): the monitor's
last value must agree within 5 % (after a units factor of 1 or rho, since function objects may or may not scale
kinematic pressure). A series that does not agree is dropped with a warning, so a monitor that OpenFOAM wrote
differently from what is assumed here cannot change the results silently.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

G0 = 9.80665
DEFAULTS = {"window": 0.25, "min_iterations": 500, "drift": 0.01, "scatter": 0.02}
AGREE = 0.05


def _r(x, n=6):
    x = float(x)
    return float(f"{x:.{n}g}") if np.isfinite(x) else x


def settings(study: dict, plan: dict | None = None) -> dict | None:
    """The averaging settings, or None when off. Transient runs (solve.transient) always average, in time, from
    average_from on."""
    tr = (study.get("solve") or {}).get("transient")
    if tr:
        T = float(tr["end_time"])
        return {"transient": True, "from": float(tr.get("average_from", T / 3)), "drift": 0.02, "stderr": 0.01,
                "min_samples": 20}
    a = (study.get("solve") or {}).get("average")
    if a is False:
        return None
    if a is None:
        has_zones = bool((plan or {}).get("zones")) if plan is not None else bool(study.get("rotating"))
        # turbulent external flow with a reference (drag and lift coefficients): the forces wobble the same way
        ext = (study.get("domain") or {}).get("kind") == "external" and study.get("model", "laminar") != "laminar" \
            and bool((study.get("reference") or {}).get("velocity")) and bool((study.get("reference") or {}).get("area"))
        # snappyHexMesh meshes: residuals often stop at a floor while nothing changes (pipe_snappy: p 4.5e-5 at
        # iteration 2000 and 4000 alike); 'settled' tells a converged run from one still moving
        snappy = (study.get("mesh") or {}).get("mesher") == "snappy"
        if not (has_zones or ext or snappy):
            return None
        a = True
    cfg = dict(DEFAULTS)
    if isinstance(a, dict):
        cfg.update({k: a[k] for k in DEFAULTS if k in a})
    return cfg


# ---- controlDict entries ----------------------------------------------------------------------------
def _fo(name: str, body: str) -> str:
    lines = "\n".join("        " + l for l in body.strip().splitlines())
    return f"    {name}\n    {{\n{lines}\n    }}"


def monitor_functions(study: dict, plan: dict, rho: float) -> tuple[list, dict]:
    """controlDict function objects that write every iteration: total pressure on every inlet and outlet
    (flow-weighted, with the static pressure beside it) and the moment on each rotating zone's walls."""
    cfg = settings(study, plan)
    if cfg is None:
        return [], {}
    kinds = {n: v["kind"] for n, v in plan["patches"].items()}
    ends = [n for n, k in kinds.items() if k in ("inlet", "outlet", "opening")]
    funcs, mon = [], {"pressure": {}, "torque": {}, "settings": cfg}
    if any(k == "inlet" for k in kinds.values()) and any(k in ("outlet", "opening") for k in kinds.values()):
        # total pressure as a field (OpenFOAM's pressure function object), evaluated every iteration. OpenFOAM 14
        # keys (pressure.H): calcTotal/calcCoeff, rho rhoInf (no rho field when incompressible), result; the
        # result is in Pa, p rhoInf + 0.5 rhoInf |U|^2 (the calibration below allows a factor rho)
        funcs.append(_fo("cfdtTotalP", f"""
type            pressure;
libs            ("libfieldFunctionObjects.so");
calcTotal       yes;
calcCoeff       no;
rho             rhoInf;
rhoInf          {rho:.12g};
pRef            0;
result          total(p);
executeControl  timeStep;
executeInterval 1;
writeControl    writeTime;"""))
        for n in ends:
            fo = f"cfdtP_{n}"
            # OpenFOAM 14 (surfaceFieldValue.C): the selection follows from the key given (patch); there is no
            # weightedAverage: 'average' with weightField phi is sum(phi v) / sum(phi), the flow-weighted mean
            # (phi's sign cancels on one inlet or outlet). Any bad key here is fatal at startup (2026.10.18.15).
            funcs.append(_fo(fo, f"""
type            surfaceFieldValue;
libs            ("libfieldFunctionObjects.so");
writeControl    timeStep;
writeInterval   1;
log             false;
writeFields     false;
surfaceFormat   none;
patch           {n};
operation       average;
weightField     phi;
fields          (p total(p));"""))
            mon["pressure"][n] = fo
    body = plan.get("body_walls") or []
    ref = study.get("reference") or {}
    if body and ref.get("velocity") and ref.get("area"):
        c = np.asarray(ref.get("cofr", [0, 0, 0]), float) * 1e-3
        funcs.append(_fo("cfdtF_body", f"""
type            forces;
libs            ("libforces.so");
writeControl    timeStep;
writeInterval   1;
log             false;
patches         ({' '.join(body)});
rho             rhoInf;
rhoInf          {rho:.12g};
pRef            0;
CofR            ({c[0]:.12g} {c[1]:.12g} {c[2]:.12g});"""))
        mon["forces"] = {"fo": "cfdtF_body", "patches": list(body)}
    for zn, zp in (plan.get("zones") or {}).items():
        if not zp.get("rotating_walls"):
            continue
        c = np.asarray(zp["center_mm"], float) * 1e-3
        fo = f"cfdtM_{zn}"
        funcs.append(_fo(fo, f"""
type            forces;
libs            ("libforces.so");
writeControl    timeStep;
writeInterval   1;
log             false;
patches         ({' '.join(zp['rotating_walls'])});
rho             rhoInf;
rhoInf          {rho:.12g};
pRef            0;
CofR            ({c[0]:.12g} {c[1]:.12g} {c[2]:.12g});"""))
        mon["torque"][zn] = fo
    if not mon["pressure"] and not mon["torque"] and not mon.get("forces"):
        return [], {}
    return funcs, mon


# ---- reading postProcessing ---------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)", re.I)


def _read_dat(path: Path) -> tuple[list, list]:
    """(header names, rows of floats) of an OpenFOAM .dat file; vectors '(x y z)' become three numbers."""
    header, rows = [], []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            h = [t for t in re.split(r"\s+", s.lstrip("#").strip()) if t]
            if h and h[0].lower() == "time":
                header = h
            continue
        vals = [float(v) for v in _NUM.findall(s.replace("(", " ").replace(")", " "))]
        if vals:
            rows.append(vals)
    return header, rows


def read_monitor(case: Path, name: str, prefer: str | None = None) -> tuple[np.ndarray, list, np.ndarray] | None:
    """Iterations, header and values of postProcessing/<name>/<start time>/<file>.dat (all start times, later
    ones replacing earlier iterations: a restarted run). prefer: a file name prefix to pick (e.g. 'moment')."""
    root = Path(case) / "postProcessing" / name
    if not root.is_dir():
        return None
    data: dict[float, list] = {}
    header: list = []
    width = None
    starts = []
    for d in root.iterdir():
        try:
            starts.append((float(d.name), d))
        except ValueError:
            continue
    for _, d in sorted(starts):
        files = sorted(d.glob("*.dat"))
        if prefer:
            pf = [f for f in files if f.name.lower().startswith(prefer)]
            files = pf or files
        if not files:
            continue
        h, rows = _read_dat(files[0])
        header = h or header
        for r in rows:
            if width is None:
                width = len(r)
            if len(r) == width:
                data[r[0]] = r[1:]
    if not data:
        return None
    its = np.array(sorted(data))
    return its, header, np.array([data[t] for t in its], float)


def _moment_series(case: Path, fo: str):
    """Total moment (x, y, z) per iteration from a forces function object, whichever layout it wrote:
    moments.dat with total first (OpenFOAM.org 10+), moment.dat with pressure/viscous(/porous) parts, or one
    forces.dat with forces then moments (older)."""
    got = read_monitor(case, fo, "moment")
    if got is None:
        return None
    its, header, vals = got
    root = Path(case) / "postProcessing" / fo
    has_moment_file = any(f.name.lower().startswith("moment") for f in root.glob("*/*.dat"))
    if has_moment_file:
        if any("total" in h.lower() for h in header) or vals.shape[1] == 3:
            return its, vals[:, :3]
        nv = vals.shape[1] // 3
        return its, sum(vals[:, 3 * k:3 * k + 3] for k in range(nv))
    if vals.shape[1] >= 18:                       # forces(pressure viscous porous) moment(pressure viscous porous)
        return its, vals[:, 9:12] + vals[:, 12:15] + vals[:, 15:18]
    if vals.shape[1] >= 12:                       # forces(pressure viscous) moment(pressure viscous)
        return its, vals[:, 6:9] + vals[:, 9:12]
    return None


def _force_series(case: Path, fo: str):
    """Total force (x, y, z) per iteration from a forces function object (force.dat/forces.dat: total first,
    or pressure + viscous (+ porous) parts)."""
    got = read_monitor(case, fo, "force")
    if got is None:
        return None
    its, header, vals = got
    if any("total" in h.lower() for h in header) or vals.shape[1] == 3:
        return its, vals[:, :3]
    if vals.shape[1] in (6, 9):                    # pressure, viscous (, porous)
        return its, sum(vals[:, 3 * k:3 * k + 3] for k in range(vals.shape[1] // 3))
    if vals.shape[1] >= 18:                        # forces(p v porous) moments(p v porous) in one file
        return its, vals[:, 0:3] + vals[:, 3:6] + vals[:, 6:9]
    if vals.shape[1] >= 12:
        return its, vals[:, 0:3] + vals[:, 3:6]
    return None


def _pressure_columns(header: list, ncol: int) -> tuple[int | None, int | None]:
    """(static, total) column indices (after the time column) of a surfaceFieldValue file."""
    names = header[1:] if header and header[0].lower() == "time" else []
    st = tt = None
    for i, h in enumerate(names):
        if "total(p)" in h:
            tt = i
        elif re.search(r"\(p\)$", h):
            st = i
    if not names:                                  # no header: the order the fields were asked for
        st, tt = (0, 1) if ncol >= 2 else (0, None)
    return st, tt


# ---- statistics ---------------------------------------------------------------------------------------
def window_stats(its: np.ndarray, v: np.ndarray, cfg: dict, scale: float | None = None) -> dict | None:
    """Mean, scatter and drift of the last part of a series; None when too short. scale: what drift and
    scatter are relative to (default |mean|; a lift coefficient near zero is judged against the drag's)."""
    n = len(v)
    m = int(cfg.get("min_iterations", DEFAULTS["min_iterations"]))
    w = max(int(round(float(cfg.get("window", DEFAULTS["window"])) * n)), m)
    if n < m or w < 4:
        return None
    w = min(w, n)
    x = np.asarray(v[-w:], float)
    mean = float(x.mean())
    std = float(x.std())
    h = w // 2
    d = float(x[h:].mean() - x[:h].mean())
    scale = abs(mean) if scale is None else max(abs(mean), float(scale))
    return {"mean": _r(mean), "std": _r(std), "min": _r(x.min()), "max": _r(x.max()), "last": _r(x[-1]),
            "drift": _r(abs(d) / scale) if scale > 0 else float("inf"),
            "scatter": _r(std / scale) if scale > 0 else float("inf"),
            "window": [int(its[-w]), int(its[-1])], "n": int(w)}


def _calibrate(last: float, ref: float, rho: float, tol: float = AGREE) -> float | None:
    """The units factor (1, rho or 1/rho) that makes the monitor's last value match the field value, or None.
    Transient runs compare the window's mean with the mean fields, more loosely (tol 0.3: the factors differ by
    rho, a thousand for water)."""
    if ref is None or not np.isfinite(last):
        return None
    if abs(ref) < 1e-12:
        return 1.0 if abs(last) < 1e-9 else None
    best = min((1.0, rho, 1.0 / rho), key=lambda s: abs(s * last - ref))
    return best if abs(best * last - ref) <= tol * abs(ref) else None


def time_window_stats(t: np.ndarray, v: np.ndarray, cfg: dict, scale: float | None = None,
                      period: float | None = None) -> dict | None:
    """Transient: time-weighted mean, std, min, max, drift (second half of the window against the first) and the
    standard error of the mean from batch means (ten, or whole numbers of periods when the signal has a period,
    so a few vortex-shedding cycles do not look like drift), over t >= cfg['from']; None with too few samples."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    m = t >= cfg["from"]
    if m.sum() < int(cfg.get("min_samples", 20)):
        return None
    tt, x = t[m], v[m]
    w = np.gradient(tt) if len(tt) > 1 else np.ones(1)
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        return None

    def wmean(sel):
        return float((w[sel] * x[sel]).sum() / max(w[sel].sum(), 1e-300))
    span = tt[-1] - tt[0]
    if period and span >= 4 * period:
        # whole periods only: the mean over the last k periods, drift between the two halves of them, and
        # batches of whole periods
        k = int(span // period)
        t0 = tt[-1] - k * period
        sel = tt >= t0 - 1e-12
        tt, x, w = tt[sel], x[sel], np.maximum(np.gradient(tt[sel]), 0)
        h = k // 2
        first = (tt < t0 + h * period)
        second = (tt >= tt[-1] - h * period)
        nb = min(10, k)
        per = k // nb
        edges = [t0 + i * per * period for i in range(nb + 1)]
    else:
        mid = 0.5 * (tt[0] + tt[-1])
        first, second = tt < mid, tt >= mid
        edges = list(np.linspace(tt[0], tt[-1], 11))
    mean = wmean(np.ones(len(x), bool))
    std = float(np.sqrt((w * (x - mean) ** 2).sum() / w.sum()))
    d = wmean(second) - wmean(first) if first.any() and second.any() else 0.0
    bm = [wmean((tt >= a) & (tt < b)) for a, b in zip(edges[:-1], edges[1:]) if ((tt >= a) & (tt < b)).any()]
    se = float(np.std(bm, ddof=1) / np.sqrt(len(bm))) if len(bm) > 2 else float("inf")
    sc = abs(mean) if scale is None else max(abs(mean), float(scale))
    return {"mean": _r(mean), "std": _r(std), "min": _r(x.min()), "max": _r(x.max()), "last": _r(x[-1]),
            "drift": _r(abs(d) / sc) if sc > 0 else float("inf"),
            "scatter": _r(std / sc) if sc > 0 else float("inf"),
            "stderr": _r(se / sc) if sc > 0 else float("inf"),
            "window": [_r(tt[0]), _r(tt[-1])], "n": int(len(tt))}


def dominant_frequency(t: np.ndarray, v: np.ndarray, t0: float) -> dict | None:
    """The strongest frequency (Hz) of v(t) for t >= t0 (resampled uniformly, Hann window, parabolic peak),
    and how many periods the window holds."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    m = t >= t0
    if m.sum() < 32:
        return None
    tt, x = t[m], v[m]
    n = int(min(8192, max(256, 2 * len(tt))))
    tu = np.linspace(tt[0], tt[-1], n)
    xu = np.interp(tu, tt, x)
    xu = (xu - xu.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(xu))
    f = np.fft.rfftfreq(n, d=(tu[-1] - tu[0]) / (n - 1))
    if len(spec) < 4 or spec[1:].max() <= 0:
        return None
    k = int(np.argmax(spec[1:])) + 1
    if 1 <= k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        den = a - 2 * b + c
        dk = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        dk = 0.0
    fr = float((k + dk) * (f[1] - f[0]))
    return {"frequency_Hz": _r(fr), "periods": _r(fr * (tt[-1] - tt[0]))}


# ---- the averaged results -----------------------------------------------------------------------------
def average_results(case: Path, plan: dict, res: dict) -> tuple[dict | None, list]:
    """Window statistics of the monitored quantities; when they agree with the final fields, the means
    replace the final-iteration values in res (rotating.<zone>.torque_Nm, power_W; pump.head_m, power_W,
    efficiency; total_pressure_drop, head_loss_m), which move to *_last."""
    mon = plan.get("monitors") or {}
    if not mon:
        return None, []
    cfg = mon.get("settings") or dict(DEFAULTS)
    rho = float(plan["rho"])
    warns, series = [], {}
    tr = bool(cfg.get("transient"))

    def cal(its, raw, ref):
        """Units factor from the last value (steady) or the window's mean against the mean fields (transient)."""
        if tr:
            mm = np.asarray(its) >= cfg["from"]
            x = float(np.asarray(raw)[mm].mean()) if mm.any() else float(raw[-1])
            return _calibrate(x, ref, rho, 0.3)
        return _calibrate(float(raw[-1]), ref, rho)
    # torque per rotating zone
    for zn, fo in mon.get("torque", {}).items():
        got = _moment_series(case, fo)
        if got is None:
            warns.append(f"averaging: no moment history from {fo} (postProcessing/{fo}); final-iteration values kept")
            continue
        its, M = got
        a = np.asarray(plan["zones"][zn]["axis"], float)
        T = M @ a
        ref = (res.get("rotating") or {}).get(zn, {}).get("torque_Nm")
        s = cal(its, T, ref)
        if s is None:
            warns.append(f"averaging: zone {zn}: the monitored torque at the last iteration ({T[-1]:.6g}) does not "
                         f"match the fields' ({ref}); its history is not used")
            continue
        series[f"torque_Nm.{zn}"] = (its, s * T)
    # total pressure drop, inlets minus outlets, each side weighted by its flow
    ins = [n for n in mon.get("pressure", {}) if plan["patches"][n]["kind"] == "inlet"]
    outs = [n for n in mon.get("pressure", {}) if plan["patches"][n]["kind"] in ("outlet", "opening")]
    if ins and outs:
        cols, ok = {}, True
        for n in ins + outs:
            got = read_monitor(case, mon["pressure"][n])
            if got is None:
                warns.append(f"averaging: no pressure history for {n} (postProcessing/{mon['pressure'][n]})")
                ok = False
                break
            its, header, vals = got
            st, tt = _pressure_columns(header, vals.shape[1])
            cols[n] = (its, vals[:, st] if st is not None and st < vals.shape[1] else None,
                       vals[:, tt] if tt is not None and tt < vals.shape[1] else None)
        if ok:
            common = cols[ins[0]][0]
            for n in cols:
                common = np.intersect1d(common, cols[n][0])
            if len(common):
                wq = {n: abs(float(res.get("flow", {}).get(n, {}).get("Q", 1.0))) or 1.0 for n in cols}

                def side(names, j):
                    num = 0.0
                    for n in names:
                        its, *c = cols[n]
                        idx = np.searchsorted(its, common)
                        num = num + wq[n] * c[j][idx]
                    return num / sum(wq[n] for n in names)
                ref = res.get("total_pressure_drop")
                tp = None
                if all(cols[n][2] is not None for n in cols):
                    d = side(ins, 1) - side(outs, 1)
                    s = cal(common, d, ref)
                    if s is not None:
                        tp, method = s * d, "flow-weighted total pressure"
                if tp is None and all(cols[n][1] is not None for n in cols) and ref is not None:
                    # static pressure history plus the final iteration's dynamic-pressure difference
                    d = side(ins, 0) - side(outs, 0)
                    sp = res.get("pressure_drop")
                    s = cal(common, d, sp) if sp is not None else None
                    if s is not None:
                        tp, method = s * d + (ref - sp), "static pressure + final dynamic-pressure difference"
                if tp is None:
                    warns.append("averaging: the monitored inlet/outlet pressures do not match the fields at the last "
                                 "iteration; their history is not used")
                else:
                    series["total_pressure_drop"] = (common, tp)
    # drag and lift coefficients from the forces on the body
    fm = mon.get("forces")
    co = res.get("coefficients") or {}
    if fm and co.get("q_Pa") and co.get("area_m2"):
        got = _force_series(case, fm["fo"])
        if got is None:
            warns.append(f"averaging: no force history (postProcessing/{fm['fo']}); final-iteration coefficients kept")
        else:
            its, Fh = got
            d, l = np.asarray(co["drag_dir"], float), np.asarray(co["lift_dir"], float)
            qa = float(co["q_Pa"]) * float(co["area_m2"])
            cd_ref = co.get("cd")
            s = cal(its, (Fh @ d) / qa, cd_ref)
            if s is None:
                warns.append(f"averaging: the monitored drag at the last iteration ({float(Fh[-1] @ d) / qa:.6g}) does not "
                             f"match the fields' cd ({cd_ref}); its history is not used")
            else:
                series["cd"] = (its, s * (Fh @ d) / qa)
                series["cl"] = (its, s * (Fh @ l) / qa)
    if not series:
        return None, warns
    # derived: power, head, efficiency
    zones = [k.split(".", 1)[1] for k in series if k.startswith("torque_Nm.")]
    if zones:
        its = series[f"torque_Nm.{zones[0]}"][0]
        for z in zones[1:]:
            its = np.intersect1d(its, series[f"torque_Nm.{z}"][0])
        P = sum(-series[f"torque_Nm.{z}"][1][np.searchsorted(series[f"torque_Nm.{z}"][0], its)]
                * plan["zones"][z]["omega_rad_s"] for z in zones)
        series["power_W"] = (its, P)
    if "total_pressure_drop" in series:
        its, tp = series["total_pressure_drop"]
        series["head_m"] = (its, -tp / (rho * G0))
        if "power_W" in series and res.get("pump"):
            Q = float(res["pump"]["flow_m3h"]) / 3600.0
            com = np.intersect1d(its, series["power_W"][0])
            H = series["head_m"][1][np.searchsorted(its, com)]
            P = series["power_W"][1][np.searchsorted(series["power_W"][0], com)]
            with np.errstate(divide="ignore", invalid="ignore"):
                series["efficiency"] = (com, np.where(P > 0, rho * G0 * Q * H / P, np.nan))
    stats = {}
    fq, period = None, None
    if tr:
        sig = series.get("cl") or series.get("cd")
        fq = dominant_frequency(sig[0], sig[1], cfg["from"]) if sig is not None else None
        if fq and fq["periods"] >= 4:
            period = 1.0 / fq["frequency_Hz"]
    for k, (its, v) in series.items():
        # cl is judged against the drag's size (its own mean can be near zero)
        sc = float(np.abs(series["cd"][1][-len(v):]).mean()) if k == "cl" and "cd" in series else None
        st = time_window_stats(its, v, cfg, sc, period) if tr else window_stats(its, v, cfg, sc)
        if st is not None:
            stats[k] = st
    total = max(len(v) for _, v in series.values())
    if not stats and tr:
        warns.append(f"averaging: fewer than {cfg['min_samples']} time steps after average_from ({cfg['from']:g} s); "
                     "last-time values kept")
        return {"settings": cfg, "time_steps_recorded": total, "settled": False,
                "why": ["too few time steps to average"]}, warns
    if not stats:
        warns.append(f"averaging: {total} iterations recorded, fewer than min_iterations ({cfg['min_iterations']}); "
                     "final-iteration values kept")
        return {"settings": cfg, "iterations_recorded": total, "settled": False,
                "why": "too few iterations to average"}, warns
    # the quantities judged: pump head and power, else the pressure drop and the torques
    judged = [k for k in ("head_m", "power_W") if k in stats] or \
             [k for k in stats if k == "total_pressure_drop" or k.startswith("torque_Nm.")]
    judged += [k for k in ("cd", "cl") if k in stats]
    bad = []
    for k in judged:
        st = stats[k]
        if st["drift"] > cfg["drift"]:
            bad.append(f"{k} drifts {st['drift']:.2%} (> {cfg['drift']:.0%})")
        if tr:
            if st["stderr"] > cfg["stderr"]:
                bad.append(f"{k}: the mean's standard error is {st['stderr']:.2%} (> {cfg['stderr']:.0%}): run longer")
        elif st["scatter"] > cfg["scatter"]:
            bad.append(f"{k} scatters {st['scatter']:.2%} (> {cfg['scatter']:.0%})")
    if tr:
        block = {"basis": f"time averages from {cfg['from']:g} s to the end (time-weighted); *_last: the last time step",
                 "settings": cfg, "judged": judged, "settled": not bad, **({"why": bad} if bad else {}),
                 "quantities": stats, "time_steps_recorded": total}
        if fq:
            L, U = (plan.get("transient") or {}).get("length_m"), (plan.get("transient") or {}).get("velocity")
            if L and U:
                fq["strouhal"] = _r(fq["frequency_Hz"] * L / U)
            fq["of"] = "cl" if "cl" in series else "cd"
            block["frequency"] = fq
    else:
        block = {"basis": "mean over the window (the last iterations); *_last in the results are the final iteration",
                 "settings": cfg, "judged": judged, "settled": not bad, **({"why": bad} if bad else {}),
                 "quantities": stats, "iterations_recorded": total}
    if "cd" in series:                               # coefficient history, at most 200 points, for plots
        its, cd = series["cd"]
        k = max(1, len(its) // 200)
        xk = "time" if tr else "iteration"
        block["history"] = {xk: [(_r(i) if tr else int(i)) for i in its[::k]], "cd": [_r(x) for x in cd[::k]],
                            "cl": [_r(x) for x in series["cl"][1][::k]]}
    _apply_means(res, stats, plan)
    co = res.get("coefficients")
    if tr and co is not None:
        for k in ("cd", "cl"):
            if k in stats:
                co[f"{k}_min"], co[f"{k}_max"] = stats[k]["min"], stats[k]["max"]
        if block.get("frequency"):
            co["frequency_Hz"] = block["frequency"]["frequency_Hz"]
            if "strouhal" in block["frequency"]:
                co["strouhal"] = block["frequency"]["strouhal"]
    return block, warns


def _apply_means(res: dict, stats: dict, plan: dict) -> None:
    rho = float(plan["rho"])
    co = res.get("coefficients")
    if co is not None:
        for k in ("cd", "cl"):
            if k in stats:
                co[f"{k}_last"] = co.get(k)
                co[k] = stats[k]["mean"]
                co[f"{k}_std"] = stats[k]["std"]
        if "cd" in stats:
            co["averaged_over"] = stats["cd"]["window"]
    for k, st in stats.items():
        if k.startswith("torque_Nm."):
            zn = k.split(".", 1)[1]
            z = res.get("rotating", {}).get(zn)
            if z is not None:
                z["torque_Nm_last"], z["power_W_last"] = z["torque_Nm"], z["power_W"]
                z["torque_Nm"] = st["mean"]
                z["power_W"] = _r(-st["mean"] * plan["zones"][zn]["omega_rad_s"])
                z["averaged_over"] = st["window"]
    if "total_pressure_drop" in stats and "total_pressure_drop" in res:
        res["total_pressure_drop_last"] = res["total_pressure_drop"]
        res["total_pressure_drop"] = stats["total_pressure_drop"]["mean"]
        if "head_loss_m" in res:
            res["head_loss_m_last"] = res["head_loss_m"]
            res["head_loss_m"] = _r(stats["total_pressure_drop"]["mean"] / (rho * G0))
    pump = res.get("pump")
    if pump:
        for k in ("head_m", "power_W", "efficiency"):
            if k in stats and np.isfinite(stats[k]["mean"]):
                pump[f"{k}_last"] = pump.get(k)
                pump[k] = stats[k]["mean"]
                pump[f"{k}_std"] = stats[k]["std"]
        if "head_m" in stats:
            pump["averaged_over"] = stats["head_m"]["window"]
            # efficiency of the means (rho g Q mean(H) / mean(P)) is steadier than the mean of the ratio
            if "power_W" in stats and stats["power_W"]["mean"] > 0:
                Q = float(pump["flow_m3h"]) / 3600.0
                pump["efficiency"] = _r(rho * G0 * Q * stats["head_m"]["mean"] / stats["power_W"]["mean"])
        pump["note"] = (pump.get("note", "") + "; window means over iterations "
                        f"{pump.get('averaged_over', '')} (see averaged)").lstrip("; ")
