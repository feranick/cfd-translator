"""bubbles.py — one-way bubble tracking through a finished single-phase run's fields (frozen rotor).
Version: 2026.11.25.4

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
it; images of trajectories. One-way: the bubbles do not change the flow; no turbulent dispersion, no coalescence.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

KEYS = {"source", "releases", "diameters_mm", "n", "t_max_s", "revolutions", "target", "gravity", "added_mass",
        "dt_max_s", "seed", "keep", "max_steps"}
G_DEFAULT = (0.0, 0.0, -9.80665)
SCALE = 1e-3                                  # mm -> m


# ---- validation ------------------------------------------------------------------------------------------------
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _region_errors(r, where: str) -> list[str]:
    from .twophase import region_errors
    return region_errors([r], where) if isinstance(r, dict) else [f"{where}: a region {{name, cylinder | box}}"]


def errors(st: dict) -> list[str]:
    b = st.get("bubbles")
    if not isinstance(b, dict):
        return ["bubbles: {source, releases, diameters_mm, ...}"]
    e = [f"bubbles: unknown key {k!r} (keys: {sorted(KEYS)})" for k in b if k not in KEYS]
    if not isinstance(b.get("source"), str) or not b["source"].strip():
        e.append("bubbles.source: '<request>/<study>' of a finished single-phase run on this machine (its latest U "
                 "and p are the flow the bubbles move in)")
    rel = b.get("releases")
    if not isinstance(rel, list) or not rel:
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
    zones = _read_cell_zones(case / "constant" / "polyMesh")
    mrf = _mrf(case)
    zone_of = np.full(n, -1, np.int32)
    for i, z in enumerate(mrf):
        cells = zones.get(z["cellZone"])
        if cells is None:
            raise RuntimeError(f"MRF zone {z['name']}: no cellZone {z['cellZone']!r} in the mesh")
        zone_of[cells] = i
    return {"time": t, "C": C, "V": V, "U": U, "p": p, "mrf": mrf, "zone_of": zone_of,
            "faces": _boundary_vectors(cdir / "C")}


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

    def __init__(self, C, V, U, p, zone_of=None, mrf=(), faces=None, open_patches=(), k: int = 12):
        from scipy.spatial import cKDTree
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

    def exit_patch(self, X, h):
        """The open patch a point outside the fluid is next to (within a cell), else ''."""
        if self.open_tree is None:
            return np.full(len(X), "", dtype=object)
        d, i = self.open_tree.query(np.atleast_2d(X))
        out = np.where(d <= 1.5 * h, self.open_names[i], "")
        return out.astype(object)


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
        return (rad <= float(c["radius"])) & (s >= lo) & (s <= hi)
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
          dt_max: float = 1e-3, target=None, keep=0, max_steps: int = 2_000_000, zone_r=None) -> dict:
    """Moves bubbles of diameter d_m (m; one value or one per bubble) from X0 (m) for t_max seconds, all in one loop
    (each with its own step). keep: how many (the first) or which bubbles' paths to record. target: a region dict
    (mm) or None. zone_r: per
    MRF zone, (outer radius m, span lo m, span hi m) for the impeller-exit count, or None. Returns per bubble: fate
    ('flow' | 'outlet' | 'inlet' | 'wall'), reached (target), t_first, t_in (seconds in the target), in_at_end,
    exited (left a zone through its outer radius), and kept trajectories."""
    X = np.array(X0, float)
    nb = len(X)
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
    tr_keep = list(range(min(keep, nb))) if isinstance(keep, int) else [int(i) for i in keep]
    paths = {i: [X[i].copy()] for i in tr_keep}
    in_t = region_mask(target, X) if target else np.zeros(nb, bool)
    reached |= in_t
    t_first[in_t] = 0.0
    steps = 0
    rec_every = max(dt_max, t_max / 400.0)
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
        F1, K1, u_f = forcing(Xa, U, gp, va)
        speed = np.maximum(np.linalg.norm(va, axis=1), np.linalg.norm(u_f, axis=1)) + 1e-9
        dt = np.minimum.reduce([0.25 * h / speed, np.full(len(a), dt_max), t_max - t[a] + 1e-12,
                                np.where(np.linalg.norm(om_new, axis=1) > 0,
                                         0.05 / np.maximum(np.linalg.norm(om_new, axis=1), 1e-30), np.inf)])
        # Heun (second order; explicit Euler spirals outward on curved paths), drag implicit in both stages
        v1 = (va + dt[:, None] * (F1 + K1[:, None] * u_f)) / (1.0 + dt * K1)[:, None]
        X1 = Xa + dt[:, None] * v1
        U2, _, gp2, _, _, _ = field.at(X1)
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
        vn = ex[:, None] * va + phi1[:, None] * G0 + phi2[:, None] * (G1 - G0)
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
                blocked = stay[o1 & o2]
                rejects[a[blocked]] += 1
                rejects[a[stay[~(o1 & o2)]]] = 0
        rejects[a[ok]] = 0
        X[a], v[a] = Xn, vn
        t_old = t[a].copy()
        t[a] = t[a] + dt
        if target is not None:
            inside = region_mask(target, Xn)
            t_in[a] += np.where(inside & in_t[a], dt, np.where(inside | in_t[a], 0.5 * dt, 0.0))
            new = inside & ~reached[a]
            t_first[a[new]] = t_old[new] + dt[new]
            reached[a] |= inside
            in_t[a] = inside
        stuck = rejects[a] > 2000
        fate[a[stuck]] = "wall"
        alive[a[stuck]] = False
        alive[a[t[a] >= t_max - 1e-12]] = False
        aset = set(a.tolist()) if tr_keep else ()
        for i in tr_keep:
            if i in aset and (t[i] - last_rec[i] >= rec_every or not alive[i]):
                paths[i].append(X[i].copy())
                last_rec[i] = t[i]
    return {"fate": fate, "reached": reached, "t_first": t_first, "t_in": t_in, "in_at_end": in_t & (fate == "flow"),
            "exited": exited, "touched_wall": touched, "t": t, "steps": steps, "X": X, "paths": {i: np.asarray(p) for i, p in paths.items()}}


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
    if len(tf):
        out["time_to_target_s"] = {"median": float(f"{np.median(tf):.4g}"), "min": float(f"{tf.min():.4g}")}
    if len(tin):
        out["residence_in_target_s"] = {"mean": float(f"{tin.mean():.4g}"), "median": float(f"{np.median(tin):.4g}"),
                                        "max": float(f"{tin.max():.4g}")}
        if omega:
            out["residence_in_target_rev"] = {k: float(f"{rev(v):.4g}") for k, v in out["residence_in_target_s"].items()}
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
    field = Field(data["C"], data["V"], data["U"], data["p"], data["zone_of"], data["mrf"], data["faces"], opens)
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
    target = _target(study, b)
    if target is None:
        warns.append(f"bubbles.target {b.get('target', 'eye')!r}: no such region: residence not counted")
    rng = np.random.default_rng(int(b.get("seed", 1)))
    keep = int(b.get("keep", 30))
    res = {"source": str(src), "source_time": data["time"], "cells": int(len(field.C)), "t_max_s": T,
           "revolutions": round(T * omega / (2 * math.pi), 3) if omega else None, "gravity": g,
           "added_mass": float(b.get("added_mass", 0.5)), "target": target, "releases": {}}
    X0, dd, groups = [], [], []
    for rel in b["releases"]:
        n = int(rel.get("n", b.get("n", 100)))
        P = release_points(field, rel, n, rng)
        for dmm in b.get("diameters_mm", [1.0]):
            i0 = sum(len(x) for x in X0)
            X0.append(P)
            dd.append(np.full(n, float(dmm) * SCALE))
            groups.append((rel["name"], float(dmm), np.arange(i0, i0 + n)))
    X0, dd = np.vstack(X0), np.concatenate(dd)
    kidx = [int(i) for _, _, ix in groups for i in ix[:keep]]
    r = track(field, X0, dd, nu, T, g, float(b.get("added_mass", 0.5)), float(b.get("dt_max_s", 1e-3)), target,
              kidx, int(b.get("max_steps", 2_000_000)), zone_r)
    res["steps"] = int(r["steps"])
    if r["steps"] >= int(b.get("max_steps", 2_000_000)):
        warns.append("bubbles: stopped at max_steps before every bubble reached t_max (raise bubbles.max_steps)")
    paths_all = {}
    for rn, dmm, ix in groups:
        sub = {k: (r[k][ix] if isinstance(r[k], np.ndarray) and r[k].shape[:1] == (len(dd),) else r[k])
               for k in ("fate", "reached", "t_first", "t_in", "in_at_end", "exited", "touched_wall", "t")}
        res["releases"].setdefault(rn, {})[f"{dmm:g}"] = summarize(sub, omega)
        loc = {int(j): r["paths"][int(i)] for j, i in enumerate(ix[:keep]) if int(i) in r["paths"]}
        paths_all[(rn, dmm)] = (loc, sub["fate"], sub["reached"])
    traj = folder / "trajectories.npz"
    np.savez_compressed(traj, **{f"{k[0]}__{k[1]:g}__{i}": p for k, (ps, _, _) in paths_all.items()
                                 for i, p in ps.items()})
    res["trajectories"] = str(traj)
    try:
        res["images"] = draw(paths_all, field, data, target, folder)
    except Exception as exc:  # noqa: BLE001 (pictures never fail a run)
        warns.append(f"bubbles: no pictures ({type(exc).__name__}: {exc})")
    notes.append("one-way: the bubbles do not change the flow; liquid acceleration from -grad p (viscous stress "
                 "neglected); no turbulent dispersion or coalescence; a frozen-rotor field: the bubbles see the impeller "
                 "at one position")
    return {"name": name, "ok": True, "results": {"bubbles": res}, "warnings": warns, "notes": notes,
            "reliable": True, "seconds": {"track": round(time.perf_counter() - t0, 3)}}


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
