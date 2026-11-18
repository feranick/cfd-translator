"""locator.py — where a sliding (NCC) run goes wrong: a coded function object that writes, for each of the first
steps, each rank's cells with the largest projected volume fraction (fvMeshStitcher's measure) and Courant number,
with the velocity, the mesh velocity and every face's flux of the worst (and of watched cells), to
processor*/cfdtWhere.dat; and the reader that turns those files into results.solve.sliding.locator.
Version: 2026.11.17.9

From the S50 v14 port (nersc/s50_sliding_v14, jobs 61249082-61290075), where it found the cells the divergence
started in. Opt-in (solve.transient.locator): the function object is compiled when foamRun starts (OpenFOAM's
coded function objects need its compiler; about a minute).
"""
from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = {"steps", "top", "detail", "watch"}

TEMPLATE = """    cfdtWhere
    {{
        type            coded;
        libs            ("libutilityFunctionObjects.so");
        steps           {steps};
        top             {top};
        detail          {detail};
        watch           {watch};

        codeInclude
        #{{
            #include "volFields.H"
            #include "surfaceFields.H"
            #include "fvcSurfaceIntegrate.H"
            #include "fvcReconstruct.H"
            #include "OFstream.H"
        #}};

        codeData
        #{{
            label steps_ = 30;
            label top_ = 8;
            label done_ = 0;
            label detail_ = 3;
            List<vector> watch_;
            labelList watchCells_;
            autoPtr<OFstream> os_;
        #}};

        codeRead
        #{{
            steps_ = dict.lookupOrDefault<label>("steps", 30);
            top_ = dict.lookupOrDefault<label>("top", 8);
            detail_ = dict.lookupOrDefault<label>("detail", 3);
            watch_ = dict.lookupOrDefault<List<vector>>("watch", List<vector>());
        #}};

        codeExecute
        #{{
            if (done_ < steps_)
            {{
                done_++;
                const fvMesh& m = mesh();
                if (!os_.valid())
                {{
                    os_.reset(new OFstream(m.time().path()/"cfdtWhere.dat"));
                }}
                const auto tsi = fvc::surfaceIntegrate(m.Sf() & m.Cf());
                const scalarField si(static_cast<const scalarField&>(tsi()));
                const scalarField pvf(mag(si/scalar(m.nSolutionD()) - 1.0));
                const surfaceScalarField& phi = m.lookupObject<surfaceScalarField>("phi");
                const auto tsum = fvc::surfaceSum(mag(phi));
                const scalarField sumPhi(static_cast<const scalarField&>(tsum()));
                const scalarField& V = static_cast<const scalarField&>(m.V());   // v14: DimensionedField::field() is private (OldTimeField)
                const scalarField Co(0.5*sumPhi/V*m.time().deltaTValue());
                const volVectorField& C = m.C();
                const scalar t = m.time().value();
                labelList order;
                sortedOrder(pvf, order);
                for (label k = 0; k < min(top_, order.size()); k++)
                {{
                    const label c = order[order.size() - 1 - k];
                    os_() << t << " pvf " << pvf[c] << " " << C[c].x() << " " << C[c].y() << " " << C[c].z()
                        << " " << V[c] << nl;
                }}
                sortedOrder(Co, order);
                for (label k = 0; k < min(top_, order.size()); k++)
                {{
                    const label c = order[order.size() - 1 - k];
                    os_() << t << " Co " << Co[c] << " " << C[c].x() << " " << C[c].y() << " " << C[c].z()
                        << " " << V[c] << nl;
                }}
                // details (tools/where.py 'Co_detail', 'watch'): the cell's velocity, the mesh's (reconstructed from
                // the mesh fluxes), and each face's flux as the solver keeps it (phi: relative to the mesh motion in
                // incompressibleFluid) with the mesh flux, the area, the centre and the unit normal (out of the
                // cell): internal faces and every fv patch face of the cell, the non-conformal ones included
                if (done_ == 1)
                {{
                    watchCells_.setSize(watch_.size(), -1);
                    // v14: fvMesh/polyMesh have no findCell; nearest cell centre (watch points sit well inside cells)
                    const vectorField& Cc = m.C().primitiveField();
                    forAll(watch_, i)
                    {{
                        label best = -1; scalar dmin = 1e30;
                        forAll(Cc, c) {{ const scalar d = magSqr(Cc[c] - watch_[i]); if (d < dmin) {{ dmin = d; best = c; }} }}
                        // every rank has a nearest cell: only the rank with the nearest of all keeps it (the lowest
                        // rank on a tie), so each point is followed once
                        const scalar gmin = returnReduce(dmin, minOp<scalar>());
                        const label owner = returnReduce(dmin == gmin ? Pstream::myProcNo() : Pstream::nProcs(),
                                                         minOp<label>());
                        watchCells_[i] = (Pstream::myProcNo() == owner) ? best : -1;
                    }}
                }}
                const volVectorField& U = m.lookupObject<volVectorField>("U");
                // the mesh flux exists only once the mesh has moved: the call at start (time 0, before the first
                // step) would abort in fvMesh::phi(), so use zero mesh fluxes until then
                const tmp<surfaceScalarField> tmphi
                (
                    m.moving()
                  ? tmp<surfaceScalarField>(m.phi())
                  : surfaceScalarField::New("cfdtMeshPhi0", m, dimensionedScalar(dimVolume/dimTime, 0))
                );
                const surfaceScalarField& mphi = tmphi();
                const volVectorField Um(fvc::reconstruct(mphi));
                DynamicList<label> cells;
                for (label k = 0; k < min(detail_, order.size()); k++) {{ cells.append(order[order.size() - 1 - k]); }}
                forAll(watchCells_, i) {{ if (watchCells_[i] >= 0) {{ cells.append(watchCells_[i]); }} }}
                const std::string pres(" p" + Foam::name(Pstream::myProcNo()) + "c");   // written raw: a word may not hold
                const char* pre = pres.c_str();                                       // the space, a string is quoted
                forAll(cells, j)
                {{
                    const label c = cells[j];
                    const word tag(j < min(detail_, order.size()) ? "cell" : "watch");
                    os_() << t << " " << tag << pre << c << " " << Co[c] << " " << C[c].x() << " " << C[c].y()
                        << " " << C[c].z() << " " << V[c] << " " << U[c].x() << " " << U[c].y() << " " << U[c].z()
                        << " " << Um[c].x() << " " << Um[c].y() << " " << Um[c].z() << " " << pvf[c] << nl;
                    const cell& cf = m.cells()[c];
                    forAll(cf, i)
                    {{
                        const label f = cf[i];
                        if (f >= m.nInternalFaces()) {{ continue; }}
                        const scalar sg = (m.owner()[f] == c) ? 1.0 : -1.0;
                        const vector n(sg*m.Sf()[f]/mag(m.Sf()[f]));
                        // the velocity on the other side: the neighbouring cell's
                        const vector Uo(U[m.owner()[f] == c ? m.neighbour()[f] : m.owner()[f]]);
                        os_() << t << " face" << pre << c << " internal " << sg*phi[f] << " " << sg*mphi[f] << " "
                            << mag(m.Sf()[f]) << " " << m.Cf()[f].x() << " " << m.Cf()[f].y() << " " << m.Cf()[f].z()
                            << " " << n.x() << " " << n.y() << " " << n.z()
                            << " " << Uo.x() << " " << Uo.y() << " " << Uo.z() << nl;
                    }}
                    forAll(m.boundary(), pi)
                    {{
                        const fvPatch& pp = m.boundary()[pi];
                        const labelUList& fc = pp.faceCells();
                        forAll(fc, i)
                        {{
                            if (fc[i] != c) {{ continue; }}
                            const scalar a = mag(pp.Sf()[i]);
                            const vector n(a > vSmall ? pp.Sf()[i]/a : vector::zero);
                            // the velocity the boundary condition holds there: a wall's (movingWallVelocity: the
                            // face's own motion), the coupling's (the other side's, interpolated)
                            const vector Ub(U.boundaryField()[pi][i]);
                            os_() << t << " face" << pre << c << " " << pp.name() << " " << phi.boundaryField()[pi][i]
                                << " " << mphi.boundaryField()[pi][i] << " " << a << " " << pp.Cf()[i].x() << " "
                                << pp.Cf()[i].y() << " " << pp.Cf()[i].z() << " " << n.x() << " " << n.y() << " "
                                << n.z() << " " << Ub.x() << " " << Ub.y() << " " << Ub.z() << nl;
                        }}
                    }}
                }}
                os_().flush();
            }}
        #}};
    }}
"""


def settings(tr: dict) -> dict | None:
    """solve.transient.locator: true | {steps (30), top (8), detail (3), watch: [[x, y, z] mm, ...]}."""
    lo = tr.get("locator")
    if not lo:
        return None
    lo = {} if lo is True else dict(lo)
    return {"steps": int(lo.get("steps", 30)), "top": int(lo.get("top", 8)), "detail": int(lo.get("detail", 3)),
            "watch": [[float(v) for v in p] for p in lo.get("watch", [])]}


def errors(tr: dict) -> list[str]:
    lo = tr.get("locator")
    if lo is None or lo is True or lo is False:
        return []
    if not isinstance(lo, dict):
        return ["solve.transient.locator: true or {steps, top, detail, watch}"]
    e = [f"solve.transient.locator: unknown key {k!r} (keys: {sorted(KEYS)})" for k in lo if k not in KEYS]
    for k, lim in (("steps", 1000), ("top", 100), ("detail", 20)):
        v = lo.get(k)
        if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= lim):
            e.append(f"solve.transient.locator.{k}: an integer 0-{lim}")
    w = lo.get("watch", [])
    if not (isinstance(w, list) and all(isinstance(p, (list, tuple)) and len(p) == 3 and
                                        all(isinstance(x, (int, float)) for x in p) for p in w)):
        e.append("solve.transient.locator.watch: a list of points [x, y, z] (mm)")
    return e


def function(cfg: dict) -> str:
    """The function object's controlDict entry (watch points in metres)."""
    w = "(" + " ".join(f"({x * 1e-3:.9g} {y * 1e-3:.9g} {z * 1e-3:.9g})" for x, y, z in cfg["watch"]) + ")"
    return TEMPLATE.format(steps=cfg["steps"], top=cfg["top"], detail=cfg["detail"], watch=w)


def read(case: Path, axis, center, top: int = 3, steps: int | None = None) -> dict | None:
    """The locator's files merged: per step the largest projected volume fractions and Courant numbers (value, centre
    mm, r and s along the axis, theta, volume mm3) and the worst cell's detail (velocity, mesh velocity, the largest
    face fluxes as normal velocities and their patches)."""
    files = glob.glob(str(Path(case) / "processor*" / "cfdtWhere.dat")) or glob.glob(str(Path(case) / "cfdtWhere.dat"))
    if not files:
        return None
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    o = np.asarray(center, float)
    e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)

    def place(x, y, z):
        d = np.array([x, y, z]) * 1e3 - o
        s_ = float(d @ a)
        rad = d - s_ * a
        return {"at_mm": [round(x * 1e3, 2), round(y * 1e3, 2), round(z * 1e3, 2)],
                "r_mm": round(float(np.linalg.norm(rad)), 3), "s_mm": round(s_, 3),
                "theta_deg": round(float(np.degrees(np.arctan2(rad @ e2, rad @ e1))) % 360, 1)}
    rows = defaultdict(lambda: {"pvf": [], "Co": []})
    cells = defaultdict(dict)
    for f in files:
        for line in open(f, errors="replace"):
            p = line.split()
            try:
                if len(p) == 7 and p[1] in ("pvf", "Co"):
                    rows[float(p[0])][p[1]].append(tuple(map(float, p[2:])))
                elif len(p) in (14, 15) and p[1] in ("cell", "watch"):
                    cells[float(p[0])].setdefault(p[2], {"faces": []}).update(kind=p[1], v=list(map(float, p[3:])))
                elif len(p) in (13, 16) and p[1] == "face":
                    cells[float(p[0])].setdefault(p[2], {"faces": []})["faces"].append([p[3]] + list(map(float, p[4:])))
            except ValueError:
                continue
    def cyl(v, x, y, z):
        """(radial, along the rotation (+axis, right-handed), axial) components of v at (x, y, z) metres."""
        d = np.array([x, y, z]) * 1e3 - o
        rad = d - (d @ a) * a
        r = np.linalg.norm(rad)
        if r <= 0:
            return [0.0, 0.0, round(float(v @ a), 3)]
        er = rad / r
        et = np.cross(a, er)
        return [round(float(v @ er), 3), round(float(v @ et), 3), round(float(v @ a), 3)]
    out = []
    for i, t in enumerate(sorted(rows)[:steps]):
        rec = {"step": i + 1, "time": t}
        for kind in ("pvf", "Co"):
            rec[kind] = [{"value": float(f"{v:.4g}"), **place(x, y, z), "volume_mm3": float(f"{vol * 1e9:.3g}")}
                         for v, x, y, z, vol in sorted(rows[t][kind], key=lambda r_: -r_[0])[:top]]
        for kind, key in (("cell", "Co_detail"), ("watch", "watch")):
            got = [(k_, d) for k_, d in cells.get(t, {}).items() if d.get("kind") == kind and "v" in d]
            if kind == "cell":
                got = sorted(got, key=lambda g_: -g_[1]["v"][0])[:1]
            if got:
                rec[key] = [_detail(k_, d, place, cyl) for k_, d in got]
        out.append(rec)
    return {"steps": out, "files": len(files)}


def _detail(cid, d, place, cyl) -> dict:
    v = d["v"]
    co, x, y, z, vol = v[:5]
    U, Um = np.array(v[5:8]), np.array(v[8:11])
    faces = []
    for fr in d["faces"]:
        patch, ph, mph, area = fr[:4]
        if area > 0:
            rec = {"patch": patch, "v_n_m_s": float(f"{ph / area:.4g}"),
                   "mesh_v_n_m_s": float(f"{mph / area:.4g}"), "area_mm2": float(f"{area * 1e6:.3g}")}
            if len(fr) >= 13:            # the other side: the neighbouring cell, or the boundary condition's value
                rec["U_other_r_theta_s"] = cyl(np.array(fr[10:13]), *fr[4:7])
            faces.append(rec)
    faces.sort(key=lambda f_: -abs(f_["v_n_m_s"]))
    return {"id": cid, "Co": float(f"{co:.4g}"), **place(x, y, z), "volume_mm3": float(f"{vol * 1e9:.3g}"),
            **({"pvf": float(f"{v[11]:.4g}")} if len(v) > 11 else {}),
            "speed_m_s": round(float(np.linalg.norm(U)), 3), "relative_speed_m_s": round(float(np.linalg.norm(U - Um)), 3),
            "U_r_theta_s": cyl(U, x, y, z), "Umesh_r_theta_s": cyl(Um, x, y, z),
            "net_flux_m3s": float(f"{sum(f_[1] for f_ in d['faces']):.3g}"), "faces": faces[:4]}


def summary(loc: dict, n: int = 3) -> str:
    """One line per step of the last n steps: where the largest Courant number and projected volume fraction were."""
    parts = []
    for s in (loc or {}).get("steps", [])[-n:]:
        c = (s.get("Co") or [{}])[0]
        p = (s.get("pvf") or [{}])[0]
        parts.append(f"step {s['step']}: Co {c.get('value')} at r {c.get('r_mm')} s {c.get('s_mm')} theta "
                     f"{c.get('theta_deg')}, pvf {p.get('value')} at r {p.get('r_mm')} s {p.get('s_mm')}")
    return "; ".join(parts)


FRAME = "cfdt_locator.json"


def write_frame(case: Path, axis, center, radius=None, span=None, omega=None) -> None:
    """The rotating zone's axis and centre (mm) beside the case, for diverged_note and diverge.report (which have no
    plan); with its radius, span (mm along the axis from the centre) and speed (rad/s) when given."""
    import json
    fr = {"axis": list(map(float, axis)), "center_mm": list(map(float, center))}
    if radius is not None:
        fr["radius_mm"] = float(radius)
    if span is not None:
        fr["span_mm"] = [float(x) for x in span]
    if omega is not None:
        fr["omega_rad_s"] = float(omega)
    (Path(case) / FRAME).write_text(json.dumps(fr))


def diverged_note(case: Path) -> str:
    """For a divergence message: where the locator saw the largest Courant numbers in the last steps ('' without
    it)."""
    import json
    f = Path(case) / FRAME
    if not f.is_file():
        return ""
    try:
        fr = json.loads(f.read_text())
        loc = read(case, fr["axis"], fr["center_mm"])
    except Exception:  # noqa: BLE001
        return ""
    return f" Locator (solve.transient.locator; r, s along the axis and theta in mm and degrees): {summary(loc)}" if loc else ""

