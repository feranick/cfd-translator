"""bubbles.py — one-way bubble tracking through a finished single-phase run's fields (frozen rotor).
Version: 2026.11.26.6

A study {"bubbles": {...}} with no geometry: bubbles.source names a finished single-phase run ('<request>/<study>' or
an absolute folder) whose latest U and p (kinematic) are read with its mesh's cell centres and volumes. Each bubble
(mass neglected) obeys, per unit liquid mass of its volume,

    0 = (a_f - g) + C_A (a_f - A_b) + D,      a_f = -grad p  (the liquid's absolute acceleration, viscous stress
                                                               neglected; p has no hydrostatic part),
    D = 18 nu / d^2 (1 + 0.15 Re^0.687) (u - v)   (Schiller-Naumann),  Re = |u - v| d / nu,

so A_b = ((1 + C_A) a_f - g + D) / C_A: the pressure-gradient force (fluid acceleration and buoyancy) and added mass.
Inside an MRF zone the run is steady in the rotating frame: there the bubble moves with its relative velocity w
(v = w + Omega x r) and dw/dt = A_b - 2 Omega x w - Omega x (Omega x r), against the liquid's relative velocity
W = U - Omega x r; outside, in the absolute frame. Crossing the zone's edge converts the velocity. Drag is
integrated implicitly (a 0.1 mm bubble responds in 0.3 ms), the rest explicitly, each bubble with its own step
(a quarter of the local cell per step, Omega dt <= 0.05 inside the zone).

Fields at a point: a weighted linear least-squares fit over the 12 nearest cell centres (scipy cKDTree) gives U, p
and grad p. A point farther from its nearest centre than that cell's size is outside the fluid: the step is undone and
the bubble takes the liquid's velocity there (it slides along walls); next to an inlet or outlet patch it leaves.

Results per release and diameter: the share that reaches the target region (bubbles.target, default the 'eye'
region), the share leaving the rotating zone through its outer radius (impeller exit), the share in the target at the
end, through the outlet or inlet, stuck at walls; residence in the target (s and revolutions) and the time to reach
it; images of trajectories. One-way: the bubbles do not change the flow; no coalescence or break-up.

With k and omega in the source (k-omega SST): hinze_d_max_mm, Hinze's largest stable bubble 0.725 (sigma/rho)^0.6
epsilon^-0.4 (epsilon = 0.09 k omega) where the bubbles are held or free at the end, and per stayer; probes give it
(and U, p, k, omega) at named points. dispersion: true adds turbulent dispersion by a discrete random walk
(Gosman-Ioannides: u' normal with variance 2k/3 per component, kept for the eddy lifetime 2 C_L k/epsilon or the
shorter time to cross the eddy). Escape: outlet_time_s / _rev and escape (share out per revolution, by revolution).
Stayers report their crossings of a rotating zone's face over the last half (a point that toggles across the face is
an interface effect of the frozen rotor).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

KEYS = {"source", "releases", "diameters_mm", "n", "t_max_s", "revolutions", "target", "targets", "gravity",
        "added_mass", "dt_max_s", "seed", "keep", "max_steps", "dry_run", "tracer", "bins", "slices", "path_speeds",
        "dispersion", "probes", "surface_tension"}
G_DEFAULT = (0.0, 0.0, -9.80665)
SCALE = 1e-3                                  # mm -> m
MAX_STEPS = 8_000_000                         # loop passes (each bubble its own step: the fastest region sets them)
HELD_S = 0.1                                  # within a cell of one place this long (s), having met a wall: held there


# ---- validation ------------------------------------------------------------------------------------------------
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _region_errors(r, where: str) -> list[str]:
    from .twophase import region_errors
    if not isinstance(r, dict):
        return [f"{where}: a region {{name, cylinder | box}}"]
    e = region_errors([r], where)
    ang = (r.get("cylinder") or {}).get("angles") if isinstance(r.get("cylinder"), dict) else None
    if ang is not None and not (isinstance(ang, list) and len(ang) == 2 and all(_num(v) for v in ang)):
        e.append(f"{where}.cylinder.angles: [a0, a1] degrees about the axis (a sector, counter-clockwise from a0; "
                 "0 along x for an axis along z)")
    return e


def errors(st: dict) -> list[str]:
    b = st.get("bubbles")
    if not isinstance(b, dict):
        return ["bubbles: {source, releases, diameters_mm, ...}"]
    e = [f"bubbles: unknown key {k!r} (keys: {sorted(KEYS)})" for k in b if k not in KEYS]
    if not isinstance(b.get("source"), str) or not b["source"].strip():
        e.append("bubbles.source: '<request>/<study>' of a finished single-phase run on this machine (its latest U "
                 "and p are the flow the bubbles move in)")
    rel = b.get("releases")
    analysis_only = bool(b.get("slices") or b.get("path_speeds") or b.get("probes")) and not rel
    pr = b.get("probes")
    if pr is not None and not (isinstance(pr, list) and pr and all(
            isinstance(q, dict) and isinstance(q.get("name"), str) and isinstance(q.get("at"), (list, tuple))
            and len(q["at"]) == 3 and all(_num(v) for v in q["at"]) for q in pr)):
        e.append("bubbles.probes: [{name, at: [x, y, z] mm}, ...]: U, p, k, omega, epsilon and Hinze's largest "
                 "stable bubble there")
    if "dispersion" in b and not isinstance(b["dispersion"], bool):
        e.append("bubbles.dispersion: true adds turbulent dispersion (a discrete random walk from the source's k and "
                 "omega)")
    if "surface_tension" in b and not (_num(b["surface_tension"]) and 0 < b["surface_tension"] < 1):
        e.append("bubbles.surface_tension: N/m (default 0.072, water) for Hinze's largest stable bubble")
    for i, sl in enumerate(b.get("slices") or []):
        if not (isinstance(sl, dict) and isinstance(sl.get("name"), str) and
                (_num(sl.get("z")) or _num(sl.get("meridional_deg")))):
            e.append(f"bubbles.slices[{i}]: {{name, z: mm (a plane across the axis) | meridional_deg: angle (a half "
                     "plane through the axis), r_max, z_range, mark: [[x, y, z], ...]}: axial velocity and p there")
    ps = b.get("path_speeds")
    if ps is not None and not (isinstance(ps, dict) and isinstance(ps.get("paths"), str) and ps.get("region")):
        e.append("bubbles.path_speeds: {paths: '<request>/<study>' (a tracer or bubble run's trajectories), region: "
                 "a region or its name, direction: [dx, dy, dz] (default against gravity: down)}: the liquid's speed "
                 "along those paths there, against each diameter's rise velocity")
    if analysis_only:
        pass
    elif not isinstance(rel, list) or not rel:
        e.append("bubbles.releases: [{name, box: {min, max} | cylinder: {...} | points: [[x, y, z], ...], n}] (mm)")
    else:
        for i, r in enumerate(rel):
            w = f"bubbles.releases[{i}]"
            if not isinstance(r, dict) or not isinstance(r.get("name"), str):
                e.append(f"{w}: {{name, box | cylinder | points, n}}")
                continue
            kinds = [k for k in ("box", "cylinder", "points") if k in r]
            if len(kinds) != 1:
                e.append(f"{w}: one of box, cylinder, points")
            elif kinds[0] == "points":
                if not (isinstance(r["points"], list) and r["points"] and
                        all(isinstance(p, (list, tuple)) and len(p) == 3 and all(_num(v) for v in p)
                            for p in r["points"])):
                    e.append(f"{w}.points: [[x, y, z], ...] (mm)")
            else:
                e += _region_errors({"name": r["name"], kinds[0]: r[kinds[0]]}, w)
            if "n" in r and not (isinstance(r["n"], int) and 1 <= r["n"] <= 100000):
                e.append(f"{w}.n: bubbles from this release (1-100000)")
    d = b.get("diameters_mm", [1.0])
    if not (isinstance(d, list) and d and all(_num(x) and 0.001 <= x <= 20 for x in d)):
        e.append("bubbles.diameters_mm: [d, ...] between 0.001 and 20 mm")
    for k, lo, hi in (("n", 1, 100000), ("max_steps", 100, 10**7)):
        if k in b and not (isinstance(b[k], int) and lo <= b[k] <= hi):
            e.append(f"bubbles.{k}: an integer {lo}-{hi}")
    for k in ("t_max_s", "revolutions", "dt_max_s", "added_mass"):
        if k in b and not (_num(b[k]) and b[k] > 0):
            e.append(f"bubbles.{k}: a positive number")
    if "gravity" in b and not (isinstance(b["gravity"], list) and len(b["gravity"]) == 3 and
                               all(_num(v) for v in b["gravity"])):
        e.append("bubbles.gravity: [gx, gy, gz] (m/s2)")
    for i, t in enumerate(b.get("targets") or []):
        if isinstance(t, dict):
            e += _region_errors(t, f"bubbles.targets[{i}]")
        elif not isinstance(t, str):
            e.append(f"bubbles.targets[{i}]: a region {{name, cylinder | box}} or the name of one in regions")
    if "targets" in b and not isinstance(b["targets"], list):
        e.append("bubbles.targets: a list of regions or region names (several targets reported side by side)")
    if "tracer" in b and b["tracer"] not in ("forward", "backward"):
        e.append("bubbles.tracer: 'backward' (massless liquid tracers moving against the liquid: where the liquid "
                 "reaching the releases comes from) or 'forward' (where it goes); diameters are then ignored")
    for i, r_ in enumerate(b.get("bins") or []):
        if isinstance(r_, dict):
            e += _region_errors(r_, f"bubbles.bins[{i}]")
        elif not isinstance(r_, str):
            e.append(f"bubbles.bins[{i}]: a region or the name of one in regions (where the paths end, counted)")
    if "dry_run" in b and not isinstance(b["dry_run"], bool):
        e.append("bubbles.dry_run: true reads the source and counts the fluid cells in every release and target, "
                 "without tracking")
    if "target" in b:
        t = b["target"]
        if not (isinstance(t, str) or isinstance(t, dict)):
            e.append("bubbles.target: a region {name, cylinder | box} or the name of one in the study's regions")
        elif isinstance(t, dict):
            e += _region_errors(t, "bubbles.target")
    if "keep" in b and not (isinstance(b["keep"], int) and 0 <= b["keep"] <= 2000):
        e.append("bubbles.keep: trajectories kept per release and diameter for the pictures (0-2000)")
    if "regions" in st:
        from .twophase import region_errors
        e += region_errors(st["regions"], "regions")
    # solve: the SLURM job adds solve.procs to every study (executor.run_study_job); nothing else of it is read
    other = sorted(k for k in st if k not in ("bubbles", "regions", "run", "images", "name", "solve"))
    if other:
        e.append(f"a bubbles study takes only bubbles, regions, run and images (not {other}): the flow is the source's")
    return e


# ---- reading the source ----------------------------------------------------------------------------------------
def _times(case: Path) -> list:
    out = []
    for d in Path(case).iterdir():
        try:
            t = float(d.name)
        except ValueError:
            continue
        if d.is_dir() and t > 0:
            out.append((t, d))
    return sorted(out)


def _boundary_vectors(path: Path) -> dict:
    """The boundaryField of an ascii volVectorField (C: face centres) as {patch: (n, 3) array}; uniform or empty
    patches are left out."""
    txt = path.read_bytes()
    i = txt.find(b"boundaryField")
    out = {}
    if i < 0:
        return out
    for m in re.finditer(rb"\n    (\w[\w.:-]*)\s*\n    \{", txt[i:]):
        name = m.group(1).decode()
        at = i + m.end()
        end = txt.find(b"\n    }", at)
        blk = txt[at:end]
        mm = re.search(rb"nonuniform\s+List<vector>\s*(\d+)\s*\(", blk)
        if not mm:
            continue
        n = int(mm.group(1))
        e2 = blk.rfind(b")")
        a = np.fromstring(blk[mm.end():e2].replace(b"(", b" ").replace(b")", b" ").decode(), sep=" ")
        if n and a.size >= 3 * n:
            out[name] = a[:3 * n].reshape(-1, 3)
    return out


def source_case(src: str) -> Path:
    from .transient import source_folder
    p = source_folder(src)
    return p / "case" if (p / "case").is_dir() else p


def _mrf(case: Path) -> list:
    from .foamio import read_foam
    p = Path(case) / "constant" / "MRFProperties"
    if not p.is_file():
        return []
    d = read_foam(p.read_text(errors="replace"))
    out = []
    for name, z in d.items():
        if not isinstance(z, dict) or "omega" not in z:
            continue
        if str(z.get("active", "yes")).lower() in ("no", "false", "off"):
            continue
        om = float(str(z["omega"]).split()[0])
        num = lambda v: np.asarray([float(x) for x in re.findall(r"[-+0-9.eE]+", str(v))] if isinstance(v, str)
                                   else v, float)
        ax = num(z["axis"])
        out.append({"name": name, "cellZone": str(z.get("cellZone", name)), "origin": num(z["origin"]),
                    "axis": ax / np.linalg.norm(ax), "omega": om})
    return out


def _source_study(src_folder: Path) -> dict:
    try:
        meta = json.loads((src_folder.parent / "request.json").read_text())
        return dict(meta["studies"][src_folder.name])
    except (OSError, ValueError, KeyError):
        return {}


def cell_data(case: Path, run_step=None) -> dict:
    """C (m), V (m3), U (m/s), p (m2/s2) of the latest time, the MRF zones with their cells, the open patches' face
    centres. C and V come from writeCellCentres / writeCellVolumes, run in a shadow case beside the tracking study
    (the source is not touched) when the latest time does not already hold them."""
    from .snappy import _read_cell_zones
    from .swirl import _scalars_fast, _vectors_fast
    case = Path(case)
    ts = [t for t in _times(case) if (t[1] / "U").is_file() and (t[1] / "p").is_file()]
    if not ts:
        raise RuntimeError(f"bubbles.source: no written time with U and p in {case} (a finished, reconstructed run)")
    t, tdir = ts[-1]
    cdir = tdir
    if not ((tdir / "C").is_file() and ((tdir / "V").is_file() or (tdir / "Vc").is_file())):
        if run_step is None:
            raise RuntimeError(f"no cell centres in {tdir}")
        cdir = run_step(case, tdir)
    n = None
    V = None
    for nm in ("V", "Vc"):
        if (cdir / nm).is_file():
            txt = (cdir / nm).read_bytes()
            m = re.search(rb"internalField\s+nonuniform\s+List<scalar>\s*(\d+)", txt)
            n = int(m.group(1)) if m else None
            V = _scalars_fast(cdir / nm, n) if n else None
            break
    if V is None:
        raise RuntimeError(f"could not read the cell volumes in {cdir}")
    C = _vectors_fast(cdir / "C", n)
    U = _vectors_fast(tdir / "U", n)
    p = _scalars_fast(tdir / "p", n)
    if C is None or U is None or p is None:
        raise RuntimeError(f"could not read C, U or p ({n} cells) at time {t:g} of {case}")
    turb = {}
    for nm in ("k", "omega"):                     # turbulence (k-omega SST runs): dispersion and break-up size
        if (tdir / nm).is_file():
            a = _scalars_fast(tdir / nm, n)
            if a is not None:
                turb[nm] = a
    zones = _read_cell_zones(case / "constant" / "polyMesh")
    mrf = _mrf(case)
    zone_of = np.full(n, -1, np.int32)
    for i, z in enumerate(mrf):
        cells = zones.get(z["cellZone"])
        if cells is None:
            raise RuntimeError(f"MRF zone {z['name']}: no cellZone {z['cellZone']!r} in the mesh")
        zone_of[cells] = i
    return {"time": t, "C": C, "V": V, "U": U, "p": p, "mrf": mrf, "zone_of": zone_of,
            "faces": _boundary_vectors(cdir / "C"), "k": turb.get("k"), "omega": turb.get("omega")}


def shadow_cell_data(case: Path, tdir: Path, work: Path, run_foam) -> Path:
    """writeCellCentres and writeCellVolumes in a shadow of the source (links to its constant/ and the latest time's
    U, p; a copy of system/): returns the shadow time directory holding C and V."""
    import shutil
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    (work / tdir.name).mkdir(parents=True)
    (work / "constant").symlink_to(Path(case).resolve() / "constant")
    shutil.copytree(Path(case) / "system", work / "system")
    (work / "system" / "controlDict").write_text(        # no function objects of the source (its monitors)
        "FoamFile\n{\n    format      ascii;\n    class       dictionary;\n    object      controlDict;\n}\n\n"
        "application     foamRun;\nstartFrom       latestTime;\nstartTime       0;\nstopAt          endTime;\n"
        f"endTime         {tdir.name};\ndeltaT          1;\nwriteControl    timeStep;\nwriteInterval   1;\n"
        "writeFormat     ascii;\nwritePrecision  10;\nwriteCompression off;\ntimeFormat      general;\n"
        "timePrecision   10;\nrunTimeModifiable false;\n")
    for f in ("U", "p"):
        (work / tdir.name / f).symlink_to((tdir / f).resolve())
    run_foam(work, f"foamPostProcess -func writeCellCentres -time {tdir.name}", "log.writeCellCentres")
    run_foam(work, f"foamPostProcess -func writeCellVolumes -time {tdir.name}", "log.writeCellVolumes")
    return work / tdir.name


# ---- the field at points ---------------------------------------------------------------------------------------
class Field:
    """U, p and grad p at points: weighted linear least squares over the k nearest cell centres."""

    def __init__(self, C, V, U, p, zone_of=None, mrf=(), faces=None, open_patches=(), k: int = 12,
                 tke=None, omega=None):
        from scipy.spatial import cKDTree
        self.tke = None if tke is None else np.maximum(np.asarray(tke, float), 0.0)
        self.om_t = None if omega is None else np.maximum(np.asarray(omega, float), 1e-12)
        self.C = np.asarray(C, float)
        self.h = np.cbrt(np.asarray(V, float))
        self.U = np.asarray(U, float)
        self.p = np.asarray(p, float)
        self.zone_of = np.full(len(self.C), -1, np.int32) if zone_of is None else np.asarray(zone_of, np.int32)
        self.mrf = list(mrf)
        self.k = int(min(k, len(self.C)))
        self.tree = cKDTree(self.C)
        self.F = np.column_stack([self.p, self.U])                      # fitted together: (n, 4)
        fc, fk = [], []
        for name, a in (faces or {}).items():
            if name in open_patches:
                fc.append(a)
                fk += [name] * len(a)
        self.open_tree = cKDTree(np.vstack(fc)) if fc else None
        self.open_names = np.asarray(fk)
        wc, wk = [], []                               # every other patch's face centres: which wall a point is at
        for name, a in (faces or {}).items():
            if name not in open_patches:
                wc.append(a)
                wk += [name] * len(a)
        self.wall_tree = cKDTree(np.vstack(wc)) if wc else None
        self.wall_names = np.asarray(wk)

    def at(self, X):
        """(U (n,3), p (n,), grad p (n,3), nearest cell (n,), outside (n,) bool, h (n,))."""
        X = np.atleast_2d(np.asarray(X, float))
        d, idx = self.tree.query(X, k=self.k)
        if self.k == 1:
            d, idx = d[:, None], idx[:, None]
        near = idx[:, 0]
        h = self.h[near]
        dx = self.C[idx] - X[:, None, :]                                   # (n, k, 3)
        w = 1.0 / (d * d + (0.25 * h[:, None]) ** 2)                       # (n, k)
        A = np.concatenate([np.ones(dx.shape[:2] + (1,)), dx], axis=2)    # (n, k, 4)
        Aw = A * w[:, :, None]
        M = np.einsum("nki,nkj->nij", Aw, A)
        reg = np.zeros((4, 4))
        reg[1:, 1:] = np.eye(3)
        M = M + 1e-8 * (h * h)[:, None, None] * w.sum(axis=1)[:, None, None] * reg     # flat stencils (2D cases)
        rhs = np.einsum("nki,nkf->nif", Aw, self.F[idx])                 # (n, 4, 4 fields)
        coef = np.linalg.solve(M, rhs)                                     # (n, 4 coefs, 4 fields)
        p = coef[:, 0, 0]
        U = coef[:, 0, 1:4]
        gp = coef[:, 1:4, 0]
        # in the fluid when within some nearby cell's reach (0.87 of its size: a cube's half diagonal); judging by the
        # nearest centre alone puts a point in a large cell next to small ones outside
        outside = ~np.any(d <= 0.87 * self.h[idx], axis=1)
        return U, p, gp, near, outside, h

    @property
    def turbulent(self) -> bool:
        return self.tke is not None and self.om_t is not None

    def turb(self, near):
        """(k m2/s2, omega 1/s) of the nearest cells."""
        return self.tke[near], self.om_t[near]

    def wall_normal(self, X):
        """An outward wall normal estimate at points next to a wall: away from the centroid of the nearest centres
        (a wall's cells all lie on one side)."""
        X = np.atleast_2d(np.asarray(X, float))
        k = int(min(40, len(self.C)))                 # wide enough to see a sloping wall through stair-stepped cells
        _, idx = self.tree.query(X, k=k)
        if k == 1:
            idx = idx[:, None]
        n = X - self.C[idx].mean(axis=1)
        return n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]

    def nearest_wall(self, X):
        """(the nearest wall patch's name, its distance in m) at points."""
        if self.wall_tree is None:
            return np.full(len(X), "", dtype=object), np.full(len(X), np.inf)
        d, i = self.wall_tree.query(np.atleast_2d(X))
        return self.wall_names[i].astype(object), d

    def exit_patch(self, X, h):
        """The open patch a point outside the fluid is next to (within a cell), else ''."""
        if self.open_tree is None:
            return np.full(len(X), "", dtype=object)
        d, i = self.open_tree.query(np.atleast_2d(X))
        out = np.where(d <= 1.5 * h, self.open_names[i], "")
        return out.astype(object)


# ---- turbulence: break-up size and dispersion -------------------------------------------------------------------
C_MU = 0.09
C_L = 0.15                                    # Lagrangian integral time T_L = C_L k / epsilon (Gosman-Ioannides)
SIGMA_WATER = 0.072                           # N/m


def epsilon(k, om):
    """Dissipation rate (m2/s3) from k-omega: epsilon = C_mu k omega."""
    return C_MU * np.asarray(k, float) * np.asarray(om, float)


def hinze_dmax(k, om, rho: float, sigma: float = SIGMA_WATER):
    """Hinze's largest stable bubble (m): 0.725 (sigma/rho)^0.6 epsilon^-0.4."""
    eps = np.maximum(epsilon(k, om), 1e-30)
    return 0.725 * (sigma / rho) ** 0.6 * eps ** -0.4


def _stats_mm(a_m) -> dict | None:
    a = np.asarray(a_m, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    q = np.percentile(a, [10, 50, 90]) / SCALE
    return {"n": int(len(a)), "p10": float(f"{q[0]:.3g}"), "median": float(f"{q[1]:.3g}"), "p90": float(f"{q[2]:.3g}")}


def eddy(k, om, slip, k_drag, rng):
    """A new eddy for each bubble (discrete random walk): the fluctuation u' (n, 3; each component normal with variance
    2k/3) and its interaction time: the eddy lifetime 2 T_L (T_L = C_L k/epsilon = C_L/(C_mu omega)) or, when shorter,
    the time the bubble takes to cross the eddy (size L_e = C_mu^0.75 k^1.5/epsilon) at its slip speed."""
    n = len(k)
    up = rng.standard_normal((n, 3)) * np.sqrt(2.0 * k / 3.0)[:, None]
    eps = np.maximum(epsilon(k, om), 1e-30)
    life = 2.0 * C_L * k / eps
    le = C_MU ** 0.75 * k ** 1.5 / eps
    tau = 1.0 / np.maximum(k_drag, 1e-30)
    arg = le / np.maximum(tau * slip, 1e-30)
    cross = np.where(arg < 1.0, -tau * np.log(np.maximum(1.0 - arg, 1e-30)), np.inf)
    return up, np.minimum(life, cross)


# ---- regions ---------------------------------------------------------------------------------------------------
def region_mask(r: dict, X_m: np.ndarray) -> np.ndarray:
    """Points (m) inside a region given in mm ({cylinder: {center, axis, radius, span}} or {box})."""
    from .twophase import box6
    X = np.asarray(X_m) / SCALE
    if "cylinder" in r:
        c = r["cylinder"]
        a = np.asarray(c.get("axis", [0, 0, 1]), float)
        a = a / np.linalg.norm(a)
        rel = X - np.asarray(c["center"], float)
        s = rel @ a
        rad = np.linalg.norm(rel - s[:, None] * a, axis=1)
        lo, hi = c.get("span", [-1e30, 1e30])
        m = (rad <= float(c["radius"])) & (rad >= float(c.get("inner_radius", 0.0))) & (s >= lo) & (s <= hi)
        if c.get("angles") is not None:              # a sector: angles [a0, a1] deg about the axis, counter-clockwise
            e1 = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
            e1 = e1 - (e1 @ a) * a
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(a, e1)
            radv = rel - s[:, None] * a
            th = np.degrees(np.arctan2(radv @ e2, radv @ e1))
            a0, a1 = (float(v) for v in c["angles"])
            m &= ((th - a0) % 360.0) <= ((a1 - a0) % 360.0)
        return m
    b = box6(r["box"])
    return np.all((X >= np.asarray(b[:3])) & (X <= np.asarray(b[3:])), axis=1)


def release_points(field: Field, rel: dict, n: int, rng) -> np.ndarray:
    """n start points (m): random fluid cell centres inside the release region (points: as given, repeated)."""
    if "points" in rel:
        P = np.asarray(rel["points"], float) * SCALE
        return P[np.arange(n) % len(P)]
    m = region_mask({k: rel[k] for k in ("box", "cylinder") if k in rel}, field.C)
    cells = np.flatnonzero(m)
    if not len(cells):
        raise RuntimeError(f"release {rel['name']!r}: no cell centre inside its region")
    pick = rng.choice(cells, size=n, replace=len(cells) < n)
    return field.C[pick].copy()


# ---- tracking --------------------------------------------------------------------------------------------------
def track(field: Field, X0, d_m, nu: float, t_max: float, g=G_DEFAULT, c_a: float = 0.5,
          dt_max: float = 1e-3, target=None, keep=0, max_steps: int = MAX_STEPS, zone_r=None,
          tracer: int = 0, dispersion: bool = False, rng=None) -> dict:
    """Moves bubbles of diameter d_m (m; one value or one per bubble) from X0 (m) for t_max seconds, all in one loop
    (each with its own step). keep: how many (the first) or which bubbles' paths to record. target: a region dict
    (mm) or None. zone_r: per
    MRF zone, (outer radius m, span lo m, span hi m) for the impeller-exit count, or None. Returns per bubble: fate
    ('flow' | 'outlet' | 'inlet' | 'wall'), reached (target), t_first, t_in (seconds in the target), in_at_end,
    exited (left a zone through its outer radius), and kept trajectories. tracer +1 / -1: massless liquid tracers
    moving with (+1) or against (-1, backwards in time) the liquid's velocity in their frame; no forces.
    dispersion: turbulent dispersion by a discrete random walk (the liquid velocity a bubble sees is U + u', u' from k
    and omega, renewed after each eddy interaction time); then a bubble held at a wall is not stopped (an eddy can take
    it off again): it counts as held ('wall') only if it is held at the end. Also returned: t_end (when it left
    through a patch), crossings and crossings_late (frame changes at a zone's edge, all and over the last half)."""
    X = np.array(X0, float)
    nb = len(X)
    disp = bool(dispersion) and not tracer and field.turbulent
    if disp and rng is None:
        rng = np.random.default_rng(1)
    up = np.zeros((nb, 3))
    t_eddy = np.zeros(nb)
    n_cross = np.zeros(nb, np.int32)
    n_cross_late = np.zeros(nb, np.int32)
    t_end = np.full(nb, np.nan)
    dall = np.broadcast_to(np.asarray(d_m, float), (nb,)).copy()
    g = np.asarray(g, float)
    U, p, gp, near, out0, h = field.at(X)
    zid = field.zone_of[near]
    om = np.zeros((nb, 3))
    org = np.zeros((nb, 3))
    for i, z in enumerate(field.mrf):
        m = zid == i
        om[m] = z["omega"] * z["axis"]
        org[m] = z["origin"]
    v = U - np.cross(om, X - org)                  # start with the liquid (relative velocity inside a zone)
    t = np.zeros(nb)
    alive = np.ones(nb, bool)
    fate = np.array(["flow"] * nb, dtype=object)
    reached = np.zeros(nb, bool)
    t_first = np.full(nb, np.nan)
    t_in = np.zeros(nb)
    exited = np.zeros(nb, bool)
    rejects = np.zeros(nb, np.int32)
    touched = np.zeros(nb, bool)
    anchor, anchor_h, anchor_t = X.copy(), h.copy(), np.zeros(nb)
    wall_seen = np.zeros(nb, bool)
    tr_keep = list(range(min(keep, nb))) if isinstance(keep, int) else [int(i) for i in keep]
    paths = {i: [X[i].copy()] for i in tr_keep}
    tg = [] if target is None else (list(target) if isinstance(target, (list, tuple)) else [target])
    nt = len(tg)
    reached_k = np.zeros((nt, nb), bool)
    t_first_k = np.full((nt, nb), np.nan)
    t_in_k = np.zeros((nt, nb))
    in_k = np.array([region_mask(r_, X) for r_ in tg]) if nt else np.zeros((0, nb), bool)
    reached_k |= in_k
    t_first_k[in_k] = 0.0
    steps = 0
    rec_every = min(dt_max, t_max / 2000.0)        # path points (a rotating-frame path turns fast)
    last_rec = np.zeros(nb)
    while alive.any() and steps < max_steps:
        steps += 1
        a = np.flatnonzero(alive)
        Xa, va = X[a], v[a]
        U, p, gp, near, outside, h = field.at(Xa)
        zid_new = field.zone_of[near]
        # frame changes at a zone's edge: v absolute = w + omega x r
        om_old, org_old = om[a], org[a]
        om_new = np.zeros((len(a), 3))
        org_new = np.zeros((len(a), 3))
        for i, z in enumerate(field.mrf):
            m = zid_new == i
            om_new[m] = z["omega"] * z["axis"]
            org_new[m] = z["origin"]
        changed = np.any(om_new != om_old, axis=1)
        if changed.any():
            n_cross[a[changed]] += 1
            n_cross_late[a[changed & (t[a] >= 0.5 * t_max)]] += 1
            vabs = va + np.cross(om_old, Xa - org_old)
            va = np.where(changed[:, None], vabs - np.cross(om_new, Xa - org_new), va)
            if zone_r is not None:                  # left a zone through its outer radius: the impeller exit
                for i, z in enumerate(field.mrf):
                    leave = changed & (zid[a] == i) & (zid_new != i)
                    if leave.any():
                        rel = Xa[leave] - z["origin"]
                        s = rel @ z["axis"]
                        rad = np.linalg.norm(rel - s[:, None] * z["axis"], axis=1)
                        R, lo, hi = zone_r[i]
                        hit = (rad >= 0.9 * R) & (s >= lo) & (s <= hi)
                        exited[a[np.flatnonzero(leave)[hit]]] = True
        om[a], org[a], zid[a] = om_new, org_new, zid_new
        da = dall[a]

        def forcing(Xp, U_, gp_, vb):
            """The bubble's explicit acceleration F, its drag rate Kc and the liquid's velocity, in its frame."""
            r_rel = Xp - org_new
            u_f_ = U_ - np.cross(om_new, r_rel)                          # the liquid in the bubble's frame
            re_ = np.linalg.norm(u_f_ - vb, axis=1) * da / nu
            Kc_ = 18.0 * nu / (da * da) * (1.0 + 0.15 * re_ ** 0.687) / c_a      # 1/s
            F_ = ((1.0 + c_a) * (-gp_) - g) / c_a - 2.0 * np.cross(om_new, vb) - \
                np.cross(om_new, np.cross(om_new, r_rel))
            return F_, Kc_, u_f_
        if disp:                                   # a new eddy where the last one's interaction time has run out
            new = t_eddy[a] <= 0.0
            if new.any():
                u0 = U[new] - np.cross(om_new[new], Xa[new] - org_new[new])     # the mean liquid in its frame
                slip = np.linalg.norm(u0 - va[new], axis=1)
                K0 = 18.0 * nu / (da[new] ** 2) * (1.0 + 0.15 * (slip * da[new] / nu) ** 0.687) / c_a
                kk, oo = field.turb(near[new])
                up[a[new]], t_eddy[a[new]] = eddy(kk, oo, slip, K0, rng)
            U = U + up[a]
        F1, K1, u_f = forcing(Xa, U, gp, va)
        if tracer:
            va = tracer * u_f
        speed = np.maximum(np.linalg.norm(va, axis=1), np.linalg.norm(u_f, axis=1)) + 1e-9
        dt = np.minimum.reduce([0.4 * h / speed, np.full(len(a), dt_max), t_max - t[a] + 1e-12,
                                np.where(np.linalg.norm(om_new, axis=1) > 0,
                                         0.05 / np.maximum(np.linalg.norm(om_new, axis=1), 1e-30), np.inf)])
        if disp:                                   # not past the eddy's end, but no less than a fifth of the step
            dt = np.minimum(dt, np.maximum(t_eddy[a], 0.2 * dt))
        # Heun (second order; explicit Euler spirals outward on curved paths), drag implicit in both stages
        v1 = (va + dt[:, None] * (F1 + K1[:, None] * u_f)) / (1.0 + dt * K1)[:, None] if not tracer else va
        X1 = Xa + dt[:, None] * v1
        U2, _, gp2, _, _, _ = field.at(X1)
        if disp:
            U2 = U2 + up[a]
        F2, K2, u_f2 = forcing(X1, U2, gp2, v1)
        # corrector: v' = -K v + G(t), G linear over the step (exact exponential integration: stable for the stiff
        # drag of small bubbles, second order in the forcing)
        Km = 0.5 * (K1 + K2)
        x = Km * dt
        ex = np.exp(-x)
        phi1 = -np.expm1(-x) / Km
        phi2 = np.where(x > 1e-6, (1.0 - (-np.expm1(-x)) / np.maximum(x, 1e-300)) / Km, 0.5 * dt)
        G0 = F1 + Km[:, None] * u_f
        G1 = F2 + Km[:, None] * u_f2
        vn = ex[:, None] * va + phi1[:, None] * G0 + phi2[:, None] * (G1 - G0) if not tracer else tracer * u_f2
        Xn = Xa + 0.5 * dt[:, None] * (va + vn)
        _, _, _, near_n, out_n, h_n = field.at(Xn)
        ok = ~out_n
        # outside: an open patch nearby takes the bubble out; elsewhere the step is undone and the bubble rides with
        # the liquid (it slides along the wall)
        if (~ok).any():
            bad = np.flatnonzero(~ok)
            ex = field.exit_patch(Xn[bad], h_n[bad])
            for j, nm in zip(bad, ex):
                if nm:
                    fate[a[j]] = "outlet" if "out" in nm else ("inlet" if "in" in nm else str(nm))
                    alive[a[j]] = False
                    t_end[a[j]] = t[a[j]] + dt[j]
            stay = bad[ex == ""]
            if len(stay):
                # at a wall: slide (the move less its part into the wall), else ride with the liquid, else wait
                nrm = field.wall_normal(Xa[stay])
                vt = vn[stay] - np.maximum((vn[stay] * nrm).sum(axis=1), 0.0)[:, None] * nrm
                Xt = Xa[stay] + dt[stay, None] * vt - 0.1 * h[stay, None] * nrm   # a little off the wall
                o1 = field.at(Xt)[4]
                Xl = Xa[stay] + dt[stay, None] * u_f[stay]
                o2 = field.at(Xl)[4]
                Xn[stay] = np.where(~o1[:, None], Xt, np.where(~o2[:, None], Xl, Xa[stay]))
                vn[stay] = np.where(~o1[:, None], vt, u_f[stay])
                touched[a[stay]] = True
                wall_seen[a[stay]] = True
        # held: within a cell of where it was HELD_S ago, having met a wall meanwhile (a ceiling in slow liquid)
        far = np.linalg.norm(Xn - anchor[a], axis=1) > anchor_h[a]
        ra = a[far]
        anchor[ra], anchor_h[ra], anchor_t[ra], wall_seen[ra] = Xn[far], h[far], t[ra] + dt[far], False
        X[a], v[a] = Xn, vn
        t_old = t[a].copy()
        t[a] = t[a] + dt
        if disp:
            t_eddy[a] -= dt
        for j, r_ in enumerate(tg):
            inside = region_mask(r_, Xn)
            was = in_k[j, a]
            t_in_k[j, a] += np.where(inside & was, dt, np.where(inside | was, 0.5 * dt, 0.0))
            new = inside & ~reached_k[j, a]
            t_first_k[j, a[new]] = t_old[new] + dt[new]
            reached_k[j, a] |= inside
            in_k[j, a] = inside
        stuck = wall_seen[a] & (t[a] - anchor_t[a] > HELD_S)
        if not disp:                               # held: stopped there (with dispersion an eddy may free it)
            fate[a[stuck]] = "wall"
            alive[a[stuck]] = False
        alive[a[t[a] >= t_max - 1e-12]] = False
        aset = set(a.tolist()) if tr_keep else ()
        for i in tr_keep:
            if i in aset and (t[i] - last_rec[i] >= rec_every or not alive[i]):
                paths[i].append(X[i].copy())
                last_rec[i] = t[i]
    if disp:                                       # held at the end
        fate[(fate == "flow") & wall_seen & (t - anchor_t > HELD_S)] = "wall"
    # a bubble held at a wall inside a target stays there: it counts as in the target at the end, its residence
    # running to t_max (it was not moved after being held)
    held = fate == "wall"
    stay = (fate == "flow") | held
    t_in_k = t_in_k + np.where(in_k & held[None, :], np.maximum(t_max - t, 0.0)[None, :], 0.0)
    if nt:
        reached, t_first, t_in = reached_k[0], t_first_k[0], t_in_k[0]
    in_t = in_k[0] if nt else np.zeros(nb, bool)
    per = {(r_.get("name") or f"target{j}"): {"reached": reached_k[j], "t_first": t_first_k[j], "t_in": t_in_k[j],
                                              "in_at_end": in_k[j] & stay} for j, r_ in enumerate(tg)}
    return {"targets": per, "fate": fate, "reached": reached, "t_first": t_first, "t_in": t_in,
            "in_at_end": in_t & stay,
            "exited": exited, "touched_wall": touched, "t": t, "t_max": t_max, "steps": steps, "X": X,
            "t_end": t_end, "crossings": n_cross, "crossings_late": n_cross_late, "dispersion": disp,
            "paths": {i: np.asarray(p) for i, p in paths.items()}}


def summarize(r: dict, omega: float | None) -> dict:
    n = len(r["fate"])
    rev = (lambda s: s * omega / (2 * math.pi)) if omega else (lambda s: None)
    tf = r["t_first"][np.isfinite(r["t_first"])]
    tin = r["t_in"][r["reached"]]

    def fr(m):
        return round(float(np.count_nonzero(m)) / n, 4) if n else None
    out = {"n": n, "reached_target": fr(r["reached"]), "in_target_at_end": fr(r["in_at_end"]),
           "impeller_exit": fr(r["exited"]), "outlet": fr(r["fate"] == "outlet"), "inlet": fr(r["fate"] == "inlet"),
           "stuck_at_wall": fr(r["fate"] == "wall"), "still_in_flow": fr(r["fate"] == "flow"),
           **({"touched_wall": fr(r["touched_wall"])} if "touched_wall" in r else {})}
    if "t_max" in r:
        done = (r["t"] >= r["t_max"] - 1e-9) | (r["fate"] != "flow")
        out["finished"] = fr(done)
        if (~done).any():
            out["unfinished_reached_s"] = {"min": float(f"{r['t'][~done].min():.4g}"),
                                           "median": float(f"{np.median(r['t'][~done]):.4g}")}
    if "X" in r and (r["fate"] == "wall").any():
        P = np.asarray(r["X"])[r["fate"] == "wall"] / SCALE
        out["held_at_mm"] = {"centroid": [round(float(v), 1) for v in P.mean(axis=0)],
                             "points": [[round(float(v), 1) for v in q] for q in P[:12]]}
    te = np.asarray(r["t_end"])[r["fate"] == "outlet"] if "t_end" in r else np.zeros(0)
    te = te[np.isfinite(te)]
    if len(te):                                  # escape through the outlet: when, and the rate per revolution
        q = np.percentile(te, [10, 50, 90])
        out["outlet_time_s"] = {"p10": float(f"{q[0]:.4g}"), "median": float(f"{q[1]:.4g}"),
                                "p90": float(f"{q[2]:.4g}")}
        if omega:
            out["outlet_time_rev"] = {k: float(f"{rev(v):.4g}") for k, v in out["outlet_time_s"].items()}
    if omega and "t_max" in r and n:
        nrev = rev(r["t_max"])
        f_out = float(np.count_nonzero(r["fate"] == "outlet")) / n
        te_rev = np.sort(te * omega / (2 * math.pi))
        marks = [m for m in (1, 2, 5, 10, 20, 50, 100, 200, 500) if m <= nrev + 1e-9]
        out["escape"] = {"per_rev_mean": float(f"{f_out / nrev:.4g}"),
                         "per_rev_hazard": float(f"{-math.log(max(1.0 - f_out, 1e-12)) / nrev:.4g}")
                         if f_out < 1 else None,
                         "out_by_rev": {str(m): round(float(np.searchsorted(te_rev, m, side='right')) / n, 4)
                                        for m in marks}}
    if "dmax" in r:                              # Hinze's largest stable size where they are at the end
        dm = np.asarray(r["dmax"])
        dd = {"held": _stats_mm(dm[r["fate"] == "wall"]), "free_at_end": _stats_mm(dm[r["fate"] == "flow"])}
        d_b = r.get("d")
        if d_b is not None:
            for k_, m_ in (("held", r["fate"] == "wall"), ("free_at_end", r["fate"] == "flow")):
                if dd[k_]:
                    dd[k_]["share_below_d"] = round(float(np.count_nonzero(dm[m_] < d_b)) / max(int(m_.sum()), 1), 4)
        out["hinze_d_max_mm"] = {k: v for k, v in dd.items() if v}
    if "crossings_late" in r:
        cl = np.asarray(r["crossings_late"])
        stay_ = (r["fate"] == "flow")
        if stay_.any():
            out["zone_crossings_late_free"] = {"median": float(np.median(cl[stay_])), "max": int(cl[stay_].max()),
                                               "share_ge_10": round(float(np.count_nonzero(cl[stay_] >= 10))
                                                                    / int(stay_.sum()), 4)}
    if len(tf):
        out["time_to_target_s"] = {"median": float(f"{np.median(tf):.4g}"), "min": float(f"{tf.min():.4g}")}
    if len(tin):
        out["residence_in_target_s"] = {"mean": float(f"{tin.mean():.4g}"), "median": float(f"{np.median(tin):.4g}"),
                                        "max": float(f"{tin.max():.4g}")}
        if omega:
            out["residence_in_target_rev"] = {k: float(f"{rev(v):.4g}") for k, v in out["residence_in_target_s"].items()}
    if len(r.get("targets") or {}) > 1:          # every target side by side
        out["targets"] = {}
        for nm, q in r["targets"].items():
            d = {"reached": fr(q["reached"]), "in_at_end": fr(q["in_at_end"])}
            tf_, tin_ = q["t_first"][np.isfinite(q["t_first"])], q["t_in"][q["reached"]]
            if len(tf_):
                d["time_to_s_median"] = float(f"{np.median(tf_):.4g}")
            if len(tin_):
                d["residence_s"] = {"median": float(f"{np.median(tin_):.4g}"), "max": float(f"{tin_.max():.4g}")}
                if omega:
                    d["residence_rev"] = {k: float(f"{rev(v):.4g}") for k, v in d["residence_s"].items()}
            out["targets"][nm] = d
    return out


# ---- the study -------------------------------------------------------------------------------------------------
def _target(st: dict, b: dict) -> dict | None:
    t = b.get("target", "eye")
    if isinstance(t, dict):
        return t
    for r in (st.get("regions") or []):
        if isinstance(r, dict) and r.get("name") == t:
            return r
    return None


def run(name: str, study: dict, folder: Path, run_foam=None) -> dict:
    """The tracking study: reads the source, releases and moves the bubbles, writes trajectories and pictures."""
    import time
    t0 = time.perf_counter()
    folder = Path(folder)
    b = study["bubbles"]
    src = Path(b["source"])
    case = source_case(b["source"])
    src_study = _source_study(case.parent)
    warns, notes = [], []
    data = cell_data(case, (lambda c, tdir: shadow_cell_data(c, tdir, folder / "shadow", run_foam)) if run_foam else None)
    from .format import FLUIDS
    fl = src_study.get("fluid", "water")
    props = FLUIDS.get(fl) if isinstance(fl, str) else fl
    rho, mu = float(props["rho"]), float(props["mu"])
    nu = mu / rho
    opens = [bc.get("name") for bc in src_study.get("bcs") or []
             if isinstance(bc, dict) and bc.get("kind") in ("inlet", "outlet", "opening") and bc.get("name")]
    if not opens:
        opens = [k for k in data["faces"] if "inlet" in k or "outlet" in k]
    field = Field(data["C"], data["V"], data["U"], data["p"], data["zone_of"], data["mrf"], data["faces"], opens,
                  tke=data.get("k"), omega=data.get("omega"))
    sigma = float(b.get("surface_tension", SIGMA_WATER))
    if b.get("dispersion") and not field.turbulent:
        warns.append("bubbles.dispersion: the source has no k and omega (not a k-omega run): no dispersion")
    zone_r = []
    rot = {z.get("name"): z for z in src_study.get("rotating") or [] if isinstance(z, dict)}
    for z in data["mrf"]:
        s = rot.get(z["name"]) or {}
        cells = field.C[data["zone_of"] == data["mrf"].index(z)]
        rel = cells - z["origin"]
        sax = rel @ z["axis"]
        rad = np.linalg.norm(rel - sax[:, None] * z["axis"], axis=1)
        R = float(s["radius"]) * SCALE if s.get("radius") else float(rad.max()) if len(rad) else 0.0
        zone_r.append((R, float(sax.min()) if len(sax) else 0.0, float(sax.max()) if len(sax) else 0.0))
    omega = abs(data["mrf"][0]["omega"]) if data["mrf"] else None
    T = float(b["t_max_s"]) if "t_max_s" in b else (float(b.get("revolutions", 20)) * 2 * math.pi / omega
                                                    if omega else 1.0)
    g = b.get("gravity") or src_study.get("gravity") or list(G_DEFAULT)
    targets = []
    for t_ in (b.get("targets") or [b.get("target", "eye")]):
        r_ = _target(study, {"target": t_})
        if r_ is None:
            warns.append(f"bubbles target {t_!r}: no such region: not counted")
        else:
            targets.append(r_)
    target = targets[0] if targets else None
    if b.get("slices") or b.get("path_speeds") or b.get("probes"):  # field pictures, path statistics, point values
        extra, rho_ = {}, rho
        if b.get("slices"):
            extra["slices"] = slices(field, data, b["slices"], rho_, folder)
        if b.get("path_speeds"):
            extra["path_speeds"] = path_speeds(field, b["path_speeds"], study, b, nu, warns)
        if b.get("probes"):
            extra["probes"] = probes(field, data, b["probes"], rho, sigma)
        if not b.get("releases"):
            return {"name": name, "ok": True, "results": {"bubbles": {"source": str(src), "source_time": data["time"],
                    "cells": int(len(field.C)), **extra}}, "warnings": warns, "notes": notes, "reliable": True,
                    "seconds": {"read": round(time.perf_counter() - t0, 3)}}
    else:
        extra = {}
    if b.get("dry_run"):                         # the cells each release and target holds, nothing tracked
        cnt = {}
        for rel in b["releases"]:
            if "points" in rel:
                P = np.asarray(rel["points"], float) * SCALE
                cnt[rel["name"]] = {"points": len(P), "in_fluid": int(np.count_nonzero(~field.at(P)[4]))}
            else:
                m = region_mask({k: rel[k] for k in ("box", "cylinder") if k in rel}, field.C)
                cnt[rel["name"]] = {"cells": int(np.count_nonzero(m))}
                if m.any():
                    c = field.C[m] / SCALE
                    cnt[rel["name"]].update(centroid_mm=[round(float(v), 2) for v in c.mean(axis=0)],
                                            cell_size_mm=round(float(np.median(field.h[m])) / SCALE, 3))
        for t_ in targets:
            m = region_mask(t_, field.C)
            cnt[f"target:{t_.get('name')}"] = {"cells": int(np.count_nonzero(m))}
        bad = [k for k, v in cnt.items() if not v.get("cells", v.get("in_fluid", 0))]
        if bad:
            warns.append(f"bubbles: no fluid cells in {bad}: move those regions")
        return {"name": name, "ok": True, "results": {"bubbles": {"source": str(src), "source_time": data["time"],
                "cells": int(len(field.C)), "dry_run": cnt}}, "warnings": warns, "notes": notes, "reliable": True,
                "seconds": {"read": round(time.perf_counter() - t0, 3)}}
    rng = np.random.default_rng(int(b.get("seed", 1)))
    keep = int(b.get("keep", 30))
    res = {"source": str(src), "source_time": data["time"], "cells": int(len(field.C)), "t_max_s": T,
           "revolutions": round(T * omega / (2 * math.pi), 3) if omega else None, "gravity": g,
           "added_mass": float(b.get("added_mass", 0.5)), "target": target,
           **({"targets": [t_.get("name") for t_ in targets]} if len(targets) > 1 else {}), **extra,
           "releases": {}}
    tracer = {"backward": -1, "forward": 1}.get(b.get("tracer"), 0)
    if tracer:
        res["tracer"] = b["tracer"]
        notes.append(f"massless liquid tracers moving {'against' if tracer < 0 else 'with'} the liquid's velocity "
                     f"({'where the liquid reaching the releases comes from' if tracer < 0 else 'where it goes'}); "
                     "the diameter is not used")
    bins = []
    for b_ in b.get("bins") or []:
        r_ = _target(study, {"target": b_})
        if r_ is None:
            warns.append(f"bubbles.bins {b_!r}: no such region")
        else:
            bins.append(r_)
    X0, dd, groups = [], [], []
    for rel in b["releases"]:
        n = int(rel.get("n", b.get("n", 100)))
        P = release_points(field, rel, n, rng)
        for dmm in ([0.1] if tracer else b.get("diameters_mm", [1.0])):
            i0 = sum(len(x) for x in X0)
            X0.append(P)
            dd.append(np.full(n, float(dmm) * SCALE))
            groups.append((rel["name"], float(dmm), np.arange(i0, i0 + n)))
    X0, dd = np.vstack(X0), np.concatenate(dd)
    kidx = [int(i) for _, _, ix in groups for i in ix[:keep]]
    r = track(field, X0, dd, nu, T, g, float(b.get("added_mass", 0.5)), float(b.get("dt_max_s", 1e-3)), targets,
              kidx, int(b.get("max_steps", MAX_STEPS)), zone_r, tracer, bool(b.get("dispersion")), rng)
    res["steps"] = int(r["steps"])
    if r["dispersion"]:
        res["dispersion"] = {"model": "discrete random walk", "C_L": C_L, "C_mu": C_MU}
    if field.turbulent and not tracer:            # Hinze's largest stable size where each bubble is at the end
        near_end = field.tree.query(r["X"])[1]
        r["dmax"] = hinze_dmax(*field.turb(near_end), rho, sigma)
        r["dmax"][np.isin(r["fate"], ["outlet", "inlet"])] = np.nan
        res["surface_tension_N_m"] = sigma
    if r["steps"] >= int(b.get("max_steps", MAX_STEPS)):
        warns.append("bubbles: stopped at max_steps before every bubble reached t_max (raise bubbles.max_steps)")
    paths_all, stay_all = {}, {}
    for rn, dmm, ix in groups:
        sub = {k: (r[k][ix] if isinstance(r[k], np.ndarray) and r[k].shape[:1] == (len(dd),) else r[k])
               for k in ("fate", "reached", "t_first", "t_in", "in_at_end", "exited", "touched_wall", "t", "X",
                         "t_end", "crossings", "crossings_late", "dmax") if k in r}
        sub["targets"] = {nm: {k: v[ix] for k, v in q.items()} for nm, q in r["targets"].items()}
        sub["t_max"] = T
        sub["d"] = dmm * SCALE
        res["releases"].setdefault(rn, {})[f"{dmm:g}"] = summarize(sub, omega)
        if bins or tracer:
            res["releases"][rn][f"{dmm:g}"]["ends"] = ends(sub, bins, field, data)
        if targets and not tracer:
            st_ = stayers(field, data, sub, targets)
            if st_:
                res["releases"][rn][f"{dmm:g}"]["stayers"] = st_
                stay_all.setdefault(rn, {})[dmm] = st_
        loc = {int(j): r["paths"][int(i)] for j, i in enumerate(ix[:keep]) if int(i) in r["paths"]}
        paths_all[(rn, dmm)] = (loc, sub["fate"], sub["reached"])
    traj = folder / "trajectories.npz"
    np.savez_compressed(traj, **{f"{k[0]}__{k[1]:g}__{i}": p for k, (ps, _, _) in paths_all.items()
                                 for i, p in ps.items()})
    res["trajectories"] = str(traj)
    try:
        res["images"] = draw(paths_all, field, data, target, folder)
        if stay_all:
            res["images"].update(draw_stayers(stay_all, field, data, targets, folder))
    except Exception as exc:  # noqa: BLE001 (pictures never fail a run)
        warns.append(f"bubbles: no pictures ({type(exc).__name__}: {exc})")
    notes.append("one-way: the bubbles do not change the flow; liquid acceleration from -grad p (viscous stress "
                 "neglected); " + ("turbulent dispersion by a discrete random walk; " if r["dispersion"] else
                                   "no turbulent dispersion; ") +
                 "no break-up (hinze_d_max_mm: the largest stable size where they end) or coalescence; a frozen-rotor "
                 "field: the bubbles see the impeller at one position")
    return {"name": name, "ok": True, "results": {"bubbles": res}, "warnings": warns, "notes": notes,
            "reliable": True, "seconds": {"track": round(time.perf_counter() - t0, 3)}}


def _cyl(data: dict, X):
    """(r mm, axial mm from the first zone's origin, angle deg) about the first MRF zone's axis (else z)."""
    z = data["mrf"][0] if data["mrf"] else {"origin": np.zeros(3), "axis": np.array([0, 0, 1.0])}
    ax_ = z["axis"]
    e1 = np.array([1.0, 0, 0]) if abs(ax_[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = e1 - (e1 @ ax_) * ax_
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax_, e1)
    rel = (np.atleast_2d(X) - z["origin"]) / SCALE
    s = rel @ ax_
    rr = rel - s[:, None] * ax_
    return np.linalg.norm(rr, axis=1), s, np.degrees(np.arctan2(rr @ e2, rr @ e1))


def stayers(field: Field, data: dict, sub: dict, targets: list) -> dict:
    """Per target, the bubbles in it at the end: where (x, y, z mm; r, axial from the zone's origin, angle about its
    axis in the frozen impeller's frame), whether held at a wall (and which, how far) or free in the liquid."""
    out = {}
    for t_ in targets:
        nm = t_.get("name")
        q = (sub.get("targets") or {}).get(nm)
        if q is None:
            continue
        idx = np.flatnonzero(q["in_at_end"])
        if not len(idx):
            continue
        X = np.asarray(sub["X"])[idx]
        r, a, th = _cyl(data, X)
        wn, wd = field.nearest_wall(X)
        held = np.asarray(sub["fate"])[idx] == "wall"
        cl = np.asarray(sub["crossings_late"])[idx] if "crossings_late" in sub else None
        dm = np.asarray(sub["dmax"])[idx] if "dmax" in sub else None
        pts = [{"xyz_mm": [round(float(v) / SCALE, 1) for v in X[i]], "r_mm": round(float(r[i]), 1),
                "axial_mm": round(float(a[i]), 1), "angle_deg": round(float(th[i]), 1),
                "held": bool(held[i]), "nearest_wall": str(wn[i]), "wall_mm": round(float(wd[i]) / SCALE, 2),
                **({"zone_crossings_late": int(cl[i])} if cl is not None else {}),
                **({"d_max_mm": float(f"{dm[i] / SCALE:.3g}")} if dm is not None and np.isfinite(dm[i]) else {})}
               for i in range(len(idx))]
        walls = {}
        for w_, h_ in zip(wn, held):
            if h_:
                walls[str(w_)] = walls.get(str(w_), 0) + 1
        out[nm] = {"n": int(len(idx)), "held": int(held.sum()), "free": int((~held).sum()),
                   "held_by_wall": walls, "r_mm": {"min": round(float(r.min()), 1), "max": round(float(r.max()), 1)},
                   "axial_mm": {"min": round(float(a.min()), 1), "max": round(float(a.max()), 1)},
                   "points": pts[:40]}
        if cl is not None:                        # back and forth across a zone's face over the last half
            out[nm]["zone_crossings_late"] = {"median": float(np.median(cl)), "max": int(cl.max()),
                                              "share_ge_10": round(float(np.count_nonzero(cl >= 10)) / len(cl), 4)}
        if dm is not None and _stats_mm(dm):
            out[nm]["hinze_d_max_mm"] = _stats_mm(dm)
    return out


def ends(sub: dict, bins: list, field: Field, data: dict) -> dict:
    """Where the paths ended (tracers: where the liquid came from or goes): the share in each bin region, by fate,
    and a sample of end points."""
    X = np.asarray(sub["X"])
    n = len(X)
    out = {"by_fate": {f: round(float(np.count_nonzero(np.asarray(sub["fate"]) == f)) / n, 4)
                       for f in sorted(set(np.asarray(sub["fate"]).tolist()))}}
    if bins:
        out["in_bins"] = {b_.get("name"): round(float(np.count_nonzero(region_mask(b_, X))) / n, 4) for b_ in bins}
        anyb = np.zeros(n, bool)
        for b_ in bins:
            anyb |= region_mask(b_, X)
        out["in_bins"]["elsewhere"] = round(float(np.count_nonzero(~anyb)) / n, 4)
    r, a, th = _cyl(data, X)
    out["centroid_mm"] = [round(float(v), 1) for v in (X.mean(axis=0) / SCALE)]
    out["r_mm_quartiles"] = [round(float(v), 1) for v in np.percentile(r, [25, 50, 75])]
    out["axial_mm_quartiles"] = [round(float(v), 1) for v in np.percentile(a, [25, 50, 75])]
    out["points_mm"] = [[round(float(v) / SCALE, 1) for v in q] for q in X[:30]]
    return out


def probes(field: Field, data: dict, spec: list, rho: float, sigma: float = SIGMA_WATER) -> dict:
    """The flow at named points (mm): U (m/s), p (Pa), k, omega, epsilon (m2/s3) and Hinze's largest stable bubble
    (mm), whether in the fluid and in a rotating zone; with r, axial and angle about the first zone's axis."""
    out = {}
    if not spec:
        return out
    X = np.asarray([q["at"] for q in spec], float) * SCALE
    U, p, _, near, outside, h = field.at(X)
    r, a, th = _cyl(data, X)
    for i, q in enumerate(spec):
        d = {"at_mm": [float(v) for v in q["at"]], "r_mm": round(float(r[i]), 2), "axial_mm": round(float(a[i]), 2),
             "angle_deg": round(float(th[i]), 1), "in_fluid": bool(not outside[i]),
             "zone": (field.mrf[field.zone_of[near[i]]]["name"] if field.zone_of[near[i]] >= 0 else None),
             "U_m_s": [round(float(v), 3) for v in U[i]], "U_mag_m_s": round(float(np.linalg.norm(U[i])), 3),
             "p_Pa": round(float(p[i]) * rho, 1), "cell_mm": round(float(h[i]) / SCALE, 3)}
        if field.turbulent:
            k_, o_ = field.turb(near[i:i + 1])
            d.update(k_m2_s2=float(f"{k_[0]:.4g}"), omega_1_s=float(f"{o_[0]:.4g}"),
                     epsilon_m2_s3=float(f"{epsilon(k_, o_)[0]:.4g}"),
                     hinze_d_max_mm=float(f"{hinze_dmax(k_, o_, rho, sigma)[0] / SCALE:.3g}"))
        out[q["name"]] = d
    return out


def slices(field: Field, data: dict, spec: list, rho: float, folder: Path) -> dict:
    """Axial velocity (along the first zone's axis, + in the axis direction) and p (Pa) on a plane across the axis
    (z, mm) or a half plane through it (meridional_deg), from the cells within about half a cell of it: a picture each,
    with u_axial = 0 drawn, marked points and the pressure minimum; backflow_share = the share of those cells with
    u_axial > 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    z0 = data["mrf"][0] if data["mrf"] else {"origin": np.zeros(3), "axis": np.array([0, 0, 1.0])}
    ax_ = z0["axis"]
    r, a, th = _cyl(data, field.C)
    hmm = field.h / SCALE
    uax = field.U @ ax_
    out = {}
    for sl in spec:
        r_max = float(sl.get("r_max", 30.0))
        if _num(sl.get("z")):
            a0 = float(sl["z"]) - float(z0["origin"] @ ax_) / SCALE
            m = (np.abs(a - a0) <= 0.6 * hmm) & (r <= r_max)
            X2 = np.column_stack([field.C[m] @ np.array([1.0, 0, 0]), field.C[m] @ np.array([0, 1.0, 0])]) / SCALE
            labels = ("x (mm)", "y (mm)")
        else:
            th0 = float(sl["meridional_deg"])
            dth = np.radians((th - th0 + 180.0) % 360.0 - 180.0)
            zr = sl.get("z_range")
            m = (np.abs(r * np.sin(dth)) <= 0.6 * hmm) & (np.cos(dth) > 0) & (r <= r_max)
            if zr:
                zz = field.C @ ax_ / SCALE
                m &= (zz >= float(zr[0])) & (zz <= float(zr[1]))
            X2 = np.column_stack([r[m], field.C[m] @ ax_ / SCALE])
            labels = ("r (mm)", "z (mm)")
        if m.sum() < 10:
            out[sl["name"]] = {"cells": int(m.sum()), "note": "too few cells on this plane"}
            continue
        ua, pp = uax[m], field.p[m] * rho
        i_min = int(np.argmin(pp))
        res_ = {"cells": int(m.sum()), "u_axial_m_s": {"min": round(float(ua.min()), 3), "max": round(float(ua.max()), 3)},
                "backflow_share": round(float(np.mean(ua > 0)), 3),
                "p_min_Pa": round(float(pp[i_min]), 1), "p_min_at_mm": [round(float(v), 1) for v in field.C[m][i_min] / SCALE],
                "p_min_r_mm": round(float(r[m][i_min]), 1), "p_min_angle_deg": round(float(th[m][i_min]), 1)}
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        try:
            tri = mtri.Triangulation(X2[:, 0], X2[:, 1])
            ln = np.hypot(*(X2[tri.triangles[:, [1, 2, 0]]] - X2[tri.triangles]).transpose(2, 0, 1)).max(axis=1)
            hm = hmm[m][tri.triangles].max(axis=1)        # against the local cell size: coarse cells are not gaps
            tri.set_mask(ln > 2.5 * hm)
        except Exception:  # noqa: BLE001
            tri = None
        for k_, (val, ttl, cmap) in enumerate(((ua, "axial velocity (m/s; + along the axis)", "RdBu_r"),
                                               (pp, "p (Pa, gauge)", "viridis"))):
            axk = axs[k_]
            lim = np.percentile(np.abs(val), 98) if k_ == 0 else None
            if tri is not None:
                cf = axk.tripcolor(tri, val, shading="gouraud", cmap=cmap,
                                   **({"vmin": -lim, "vmax": lim} if lim else {}))
                if k_ == 0:
                    try:
                        axk.tricontour(tri, val, levels=[0.0], colors="k", linewidths=0.8)
                    except Exception:  # noqa: BLE001
                        pass
            else:
                cf = axk.scatter(X2[:, 0], X2[:, 1], c=val, s=4, cmap=cmap)
            fig.colorbar(cf, ax=axk, shrink=0.8)
            for q in sl.get("mark") or []:
                q = np.asarray(q, float)
                if _num(sl.get("z")):
                    axk.plot(q[0], q[1], "kx", ms=8)
                else:
                    rq, _, _ = _cyl(data, q * SCALE)
                    axk.plot(rq[0], q[2], "kx", ms=8)
            if k_ == 1:
                axk.plot(X2[i_min, 0], X2[i_min, 1], "w*", ms=10, mec="k")
            axk.set_aspect("equal")
            axk.set_xlabel(labels[0])
            axk.set_ylabel(labels[1])
            axk.set_title(f"{sl['name']}: {ttl}", fontsize=9)
        fig.tight_layout()
        p_ = folder / f"slice_{sl['name']}.png"
        fig.savefig(p_, dpi=120)
        plt.close(fig)
        res_["image"] = str(p_)
        out[sl["name"]] = res_
    return out


def path_speeds(field: Field, spec: dict, study: dict, b: dict, nu: float, warns: list) -> dict:
    """The liquid's speed along another run's kept paths inside a region, in a direction (default: down, against
    gravity), against each diameter's rise velocity (Schiller-Naumann, still liquid)."""
    from .transient import source_folder
    from .twophase import slip_velocity
    src = source_folder(spec["paths"])
    f = src / "trajectories.npz"
    if not f.is_file():
        raise RuntimeError(f"bubbles.path_speeds: no trajectories.npz in {src}")
    reg = spec["region"] if isinstance(spec["region"], dict) else _target(study, {"target": spec["region"]})
    if reg is None:
        raise RuntimeError(f"bubbles.path_speeds: no region {spec['region']!r}")
    g = np.asarray(b.get("gravity") or G_DEFAULT, float)
    dirn = np.asarray(spec.get("direction") or (g / np.linalg.norm(g)), float)
    dirn = dirn / np.linalg.norm(dirn)
    P = np.vstack([v for v in np.load(f).values() if len(v)])
    P = P[region_mask(reg, P)]
    if not len(P):
        return {"points": 0, "note": "no path points inside the region"}
    U = field.at(P)[0]
    w = U @ dirn
    q = np.percentile(w, [10, 50, 90])
    rho_l = 998.2
    gmag = float(np.linalg.norm(g))
    rise = {f"{float(d):g}": round(slip_velocity(gmag, float(d) * SCALE, rho_l, 0.0, nu * rho_l)[0], 4)
            for d in b.get("diameters_mm", [0.1, 0.3, 1, 2])}
    share = {k: round(float(np.mean(w > v)), 3) for k, v in rise.items()}
    return {"paths": str(src), "region": reg.get("name"), "points": int(len(P)),
            "direction": [round(float(v), 3) for v in dirn],
            "liquid_speed_m_s": {"p10": round(float(q[0]), 4), "median": round(float(q[1]), 4),
                                 "p90": round(float(q[2]), 4)},
            "rise_velocity_m_s": rise, "share_of_path_points_where_liquid_beats_rise": share}


def draw_stayers(stay_all: dict, field: Field, data: dict, targets: list, folder: Path) -> dict:
    """Per release, where the bubbles still in a target at the end sit: r against axial (meridional) and the angle
    about the axis against r, every size, held (filled) or free (open)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    cols = {0.1: "tab:blue", 0.3: "tab:green", 1.0: "tab:orange", 2.0: "tab:red"}
    for rn, by_d in stay_all.items():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        for dmm, st_ in by_d.items():
            for tn, q in st_.items():
                P = q["points"]
                mk = "o" if tn == (targets[0].get("name") if targets else "") else "s"
                for held_ in (True, False):
                    pp = [p_ for p_ in P if p_["held"] == held_]
                    if not pp:
                        continue
                    kw = dict(c=cols.get(dmm, "k")) if held_ else dict(facecolors="none", edgecolors=cols.get(dmm, "k"))
                    axs[0].scatter([p_["r_mm"] for p_ in pp], [p_["axial_mm"] for p_ in pp], marker=mk, s=18, **kw,
                                   label=f"{dmm:g} mm, {tn}, {'held' if held_ else 'free'}")
                    axs[1].scatter([p_["angle_deg"] for p_ in pp], [p_["r_mm"] for p_ in pp], marker=mk, s=18, **kw)
        axs[0].set_xlabel("r (mm)")
        axs[0].set_ylabel("axial from the zone origin (mm)")
        axs[1].set_xlabel("angle about the axis (deg, frozen impeller frame)")
        axs[1].set_ylabel("r (mm)")
        axs[0].legend(fontsize=6)
        axs[0].set_title(f"{rn}: bubbles in a target at the end")
        fig.tight_layout()
        p_ = folder / f"stayers_{rn}.png"
        fig.savefig(p_, dpi=110)
        plt.close(fig)
        out[f"stayers_{rn}"] = {"path": str(p_)}
    return out


def draw(paths_all: dict, field: Field, data: dict, target, folder: Path) -> dict:
    """Per release and diameter, two views of the kept trajectories (plan view along the first zone's axis and the
    meridional r-z view), coloured by fate: reached the target (red), left through the outlet (blue), others grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = data["mrf"][0] if data["mrf"] else {"origin": np.zeros(3), "axis": np.array([0, 0, 1.0])}
    ax_ = z["axis"]
    # plan view in the frame's own x and y when the axis is z (the S50: y up), else any two perpendiculars
    e1 = np.array([1.0, 0, 0]) if abs(ax_[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = e1 - (e1 @ ax_) * ax_
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax_, e1)
    out = {}
    sub = field.C[:: max(1, len(field.C) // 40000)]

    def proj(P):
        rel = (P - z["origin"]) / SCALE
        s = rel @ ax_
        return rel @ e1, rel @ e2, np.linalg.norm(rel - s[:, None] * ax_, axis=1), s
    bx, by, br, bs = proj(sub)
    for (rn, d), (ps, fate, reached) in paths_all.items():
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        axs[0].scatter(bx, by, s=0.2, c="0.85")
        axs[1].scatter(br, bs, s=0.2, c="0.85")
        for i, P in ps.items():
            if len(P) < 2:
                continue
            c = "tab:red" if reached[i] else ("tab:blue" if fate[i] == "outlet" else "0.4")
            x, y, r, s = proj(P)
            axs[0].plot(x, y, lw=0.6, c=c)
            axs[1].plot(r, s, lw=0.6, c=c)
            axs[0].plot(x[:1], y[:1], "k.", ms=2)
        axs[0].set_aspect("equal")
        axs[0].set_title(f"{rn}, {d:g} mm: view along the axis (mm)")
        axs[0].set_xlabel("x" if abs(e1[0]) > 0.9 else "e1")
        axs[0].set_ylabel("y" if abs(e2[1]) > 0.9 else "e2")
        axs[1].set_title("meridional view: r against axial (mm)")
        axs[1].set_aspect("equal")
        p = folder / f"bubbles_{rn}_{d:g}mm.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out[f"{rn}_{d:g}mm"] = {"path": str(p)}
    return out
