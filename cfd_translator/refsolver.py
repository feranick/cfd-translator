"""refsolver.py — an independent 2D steady Navier-Stokes solver that runs the written OpenFOAM case.
Version: 2026.10.16.6

    solve_case(case_dir) -> fields (same layout as the OpenFOAM reader's)

It reads the case files, not the study: constant/polyMesh (points, faces, owner, boundary),
0/U, 0/p and constant/physicalProperties, so a wrong translation (a viscosity, a velocity, a pressure,
a patch in the wrong place) gives a wrong answer here too. Scope: one-cell-thick (2D) uniform
structured boxes, one patch per side; U fixedValue / noSlip / slip / symmetry / zero-gradient types
with p fixedValue or zeroGradient (what cfd-translator writes for inlet, outlet, opening, wall, slip,
symmetry); laminar.

Method: staggered (MAC) grid, central differences, Picard iteration on the coupled velocity-pressure
system with a sparse direct solver. Boundary faces of fixed-pressure sides carry their own momentum
equation with a ghost pressure (p at the face = the fixed value), as OpenFOAM's fixedValue p does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import foamio


class RefUnsupported(Exception):
    """The case is outside the reference solver's scope."""


def _read(case: Path):
    pm = case / "constant" / "polyMesh"
    pts = foamio.read_list((pm / "points").read_text())
    faces = foamio.read_faces((pm / "faces").read_text())
    owner = foamio.read_list((pm / "owner").read_text()).astype(np.int64)
    nei = foamio.read_list((pm / "neighbour").read_text()).astype(np.int64)
    bnd = foamio.read_boundary((pm / "boundary").read_text())
    U = foamio.read_foam((case / "0" / "U").read_text())
    p = foamio.read_foam((case / "0" / "p").read_text())
    pp = foamio.read_foam((case / "constant" / "physicalProperties").read_text())
    if (case / "constant" / "MRFProperties").exists():
        raise RefUnsupported("the reference solver has no rotating zones (MRF)")
    mt = case / "constant" / "momentumTransport"
    if mt.is_file() and foamio.read_foam(mt.read_text()).get("simulationType", "laminar") != "laminar":
        raise RefUnsupported("the reference solver is laminar only")
    return pts, faces, owner, nei, bnd, U, p, float(pp["nu"])


def _grid(v):
    u = np.unique(np.round(v, 12))
    d = np.diff(u)
    if len(d) and np.ptp(d) > 1e-6 * d.mean():
        raise RefUnsupported("the reference solver needs uniform spacing")
    return u


def solve_case(case, max_iter: int = 300, tol: float = 1e-10) -> dict:
    case = Path(case)
    pts, faces, owner, nei, bnd, Uf, pf, nu = _read(case)
    xs, ys, zs = _grid(pts[:, 0]), _grid(pts[:, 1]), _grid(pts[:, 2])
    if len(zs) != 2:
        raise RefUnsupported("the reference solver is 2D (one cell along Z)")
    nx, ny = len(xs) - 1, len(ys) - 1
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    nc = int(max(owner.max(), nei.max() if len(nei) else 0)) + 1
    fc = np.array([pts[f].mean(axis=0) for f in faces])
    cc = np.zeros((nc, 3))
    cnt = np.zeros(nc)
    np.add.at(cc, owner, fc)
    np.add.at(cnt, owner, 1)
    np.add.at(cc, nei, fc[:len(nei)])
    np.add.at(cnt, nei, 1)
    cc /= cnt[:, None]
    ci = np.rint((cc[:, 0] - xs[0]) / dx - 0.5).astype(int)
    cj = np.rint((cc[:, 1] - ys[0]) / dy - 0.5).astype(int)
    cell = -np.ones((nx, ny), int)
    cell[ci, cj] = np.arange(nc)
    if np.any(cell < 0):
        raise RefUnsupported("not a structured grid")
    tol_x = 1e-9 * max(xs[-1] - xs[0], ys[-1] - ys[0])
    # sides: W, E, S, N -> patch, and the face index along the side
    sides = {}
    pinfo = {}
    for b in bnd:
        if b["type"] == "empty" or b["n"] == 0:
            continue
        f = np.arange(b["start"], b["start"] + b["n"])
        c = fc[f]
        ub, pb = Uf["boundaryField"][b["name"]], pf["boundaryField"][b["name"]]
        ut, pt = ub.get("type"), pb.get("type")
        uval = ub.get("value")
        if ut == "fixedValue":
            a = np.asarray(uval, float)
            uval = np.tile(a, (b["n"], 1)) if a.ndim == 1 else a
        side_of = np.full(b["n"], "", dtype=object)
        for s, m in (("W", np.abs(c[:, 0] - xs[0]) < tol_x), ("E", np.abs(c[:, 0] - xs[-1]) < tol_x),
                     ("S", np.abs(c[:, 1] - ys[0]) < tol_x), ("N", np.abs(c[:, 1] - ys[-1]) < tol_x)):
            side_of[m] = s
        if np.any(side_of == ""):
            raise RefUnsupported(f"patch {b['name']}: faces off the box sides")
        pinfo[b["name"]] = []
        for s in sorted(set(side_of)):
            m = side_of == s
            n = ny if s in "WE" else nx
            if int(m.sum()) != n:
                raise RefUnsupported(f"patch {b['name']} covers part of side {s}; the reference solver needs one patch per side")
            if s in sides:
                raise RefUnsupported(f"side {s} has two patches")
            idx = np.rint(((c[m, 1] - ys[0]) / dy if s in "WE" else (c[m, 0] - xs[0]) / dx) - 0.5).astype(int)
            if ut == "fixedValue":
                kind, val = "dirichlet", np.zeros((n, 3))
                val[idx] = uval[m]
            elif ut == "noSlip":
                kind, val = "dirichlet", np.zeros((n, 3))
            elif ut in ("slip", "symmetry", "symmetryPlane"):
                kind, val = "slip", None
            elif ut in ("zeroGradient", "inletOutlet", "pressureInletOutletVelocity"):
                if pt != "fixedValue":
                    raise RefUnsupported(f"patch {b['name']}: zero-gradient velocity needs a fixed pressure")
                kind, val = "pressure", float(np.asarray(pb["value"], float).ravel()[0])
            else:
                raise RefUnsupported(f"patch {b['name']}: U type {ut} not supported")
            if kind == "dirichlet" and pt == "fixedValue":
                raise RefUnsupported(f"patch {b['name']}: fixed velocity and fixed pressure")
            sides[s] = (kind, val)
            pinfo[b["name"]].append((s, np.nonzero(m)[0], idx))
    if set(sides) != set("WENS"):
        raise RefUnsupported("every side needs a patch")
    U, P, p_cells = _solve(nx, ny, dx, dy, nu, sides, max_iter, tol)
    u, v = U
    # fields in OpenFOAM layout
    Uc = np.zeros((nc, 3))
    Uc[cell.ravel(), 0] = (0.5 * (u[:-1, :] + u[1:, :])).ravel()
    Uc[cell.ravel(), 1] = (0.5 * (v[:, :-1] + v[:, 1:])).ravel()
    pc = np.zeros(nc)
    pc[cell.ravel()] = p_cells.ravel()
    patches = {}
    for b in bnd:
        if b["name"] not in pinfo:
            continue
        name, nf = b["name"], b["n"]
        Ub, pb, phi = np.zeros((nf, 3)), np.zeros(nf), np.zeros(nf)
        for s, rows, k in pinfo[name]:
            kind, val = sides[s]
            if s in "WE":
                i_face, i_cell, sign = (0, 0, -1.0) if s == "W" else (nx, nx - 1, 1.0)
                Ub[rows, 0] = u[i_face, k]
                Ub[rows, 1] = val[k, 1] if kind == "dirichlet" else 0.5 * (v[i_cell, k] + v[i_cell, k + 1])
                phi[rows] = sign * Ub[rows, 0] * dy * dz
                pcell = p_cells[i_cell, k]
            else:
                j_face, j_cell, sign = (0, 0, -1.0) if s == "S" else (ny, ny - 1, 1.0)
                Ub[rows, 1] = v[k, j_face]
                Ub[rows, 0] = val[k, 0] if kind == "dirichlet" else 0.5 * (u[k, j_cell] + u[k + 1, j_cell])
                phi[rows] = sign * Ub[rows, 1] * dx * dz
                pcell = p_cells[k, j_cell]
            pb[rows] = val if kind == "pressure" else pcell
        patches[name] = {"U": Ub, "p": pb, "phi": phi, "wss": None, "nut": None, "yPlus": None}
    for b in bnd:
        if b["type"] == "empty":
            patches[b["name"]] = {"U": None, "p": None, "phi": None}
    return {"U": Uc, "p": pc, "patches": patches, "source": "reference solver", "iterations": P["iterations"],
            "converged": P["converged"], "change": P["change"]}


def _solve(nx, ny, dx, dy, nu, sides, max_iter, tol):
    """Picard iteration of the MAC equations. Returns (u, v), info, p (nx, ny)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    nu_ = (nx + 1) * ny
    nv_ = nx * (ny + 1)
    npp = nx * ny
    iu = lambda i, j: i * ny + j                     # noqa: E731
    iv = lambda i, j: nu_ + i * (ny + 1) + j         # noqa: E731
    ip = lambda i, j: nu_ + nv_ + i * ny + j         # noqa: E731
    N = nu_ + nv_ + npp
    u = np.zeros((nx + 1, ny))
    v = np.zeros((nx, ny + 1))
    # boundary normal velocities
    for s, (kind, val) in sides.items():
        if kind == "dirichlet":
            if s == "W":
                u[0, :] = val[:, 0]
            elif s == "E":
                u[nx, :] = val[:, 0]
            elif s == "S":
                v[:, 0] = val[:, 1]
            else:
                v[:, ny] = val[:, 1]

    def tang(side, node):
        """Tangential velocity prescribed on a dirichlet side at a node position along it (node 0..n)."""
        val = sides[side][1]
        comp = 1 if side in "WE" else 0
        n = len(val)
        a = val[max(node - 1, 0), comp]
        b = val[min(node, n - 1), comp]
        return 0.5 * (a + b)

    def ghost(side, inner_val_index, node):
        """(alpha, beta, kind) for a ghost tangential velocity beyond `side`: ghost = alpha * inner + beta."""
        kind = sides[side][0]
        if kind == "dirichlet":
            return -1.0, 2.0 * tang(side, node)
        return 1.0, 0.0

    pfix = {s: val for s, (kind, val) in sides.items() if kind == "pressure"}
    change, it = np.inf, 0
    p = np.zeros((nx, ny))
    for it in range(1, max_iter + 1):
        rows, cols, vals = [], [], []
        rhs = np.zeros(N)

        def add(r, c, a):
            rows.append(r)
            cols.append(c)
            vals.append(a)

        # ---- u equations
        for i in range(nx + 1):
            for j in range(ny):
                r = iu(i, j)
                bside = "W" if i == 0 else "E" if i == nx else None
                if bside and sides[bside][0] != "pressure":
                    add(r, r, 1.0)
                    rhs[r] = u[i, j]
                    continue
                # advecting velocities (old iterate)
                ue = 0.5 * (u[i, j] + u[i + 1, j]) if i < nx else u[i, j]
                uw = 0.5 * (u[i - 1, j] + u[i, j]) if i > 0 else u[i, j]

                def vat(ii, jj):
                    if 0 <= ii < nx:
                        return v[ii, jj]
                    side = "W" if ii < 0 else "E"
                    inner = v[0 if ii < 0 else nx - 1, jj]
                    a, b = ghost(side, None, jj)
                    return a * inner + b
                vn = 0.5 * (vat(i - 1, j + 1) + vat(i, j + 1))
                vs = 0.5 * (vat(i - 1, j) + vat(i, j))
                Fe, Fw, Fn, Fs = ue * dy, uw * dy, vn * dx, vs * dx
                aE = Fe / 2 - nu * dy / dx
                aW = -Fw / 2 - nu * dy / dx
                aN = Fn / 2 - nu * dx / dy
                aS = -Fs / 2 - nu * dx / dy
                aP = (Fe - Fw + Fn - Fs) / 2 + 2 * nu * (dy / dx + dx / dy)
                b = 0.0
                # east / west neighbours (zero-gradient beyond a pressure side)
                if i < nx:
                    add(r, iu(i + 1, j), aE)
                else:
                    aP += aE
                if i > 0:
                    add(r, iu(i - 1, j), aW)
                else:
                    aP += aW
                # north / south (ghost tangential velocities)
                if j < ny - 1:
                    add(r, iu(i, j + 1), aN)
                else:
                    a, bb = ghost("N", None, i)
                    aP += aN * a
                    b -= aN * bb
                if j > 0:
                    add(r, iu(i, j - 1), aS)
                else:
                    a, bb = ghost("S", None, i)
                    aP += aS * a
                    b -= aS * bb
                add(r, r, aP)
                # pressure gradient (p_i - p_{i-1}) dy, ghost p = 2 p_fix - p_inner beyond a pressure side
                if i < nx:
                    add(r, ip(i, j), dy)
                else:
                    b -= 2 * pfix["E"] * dy
                    add(r, ip(nx - 1, j), -dy)
                if i > 0:
                    add(r, ip(i - 1, j), -dy)
                else:
                    b += 2 * pfix["W"] * dy
                    add(r, ip(0, j), -dy)
                rhs[r] = b
        # ---- v equations
        for i in range(nx):
            for j in range(ny + 1):
                r = iv(i, j)
                bside = "S" if j == 0 else "N" if j == ny else None
                if bside and sides[bside][0] != "pressure":
                    add(r, r, 1.0)
                    rhs[r] = v[i, j]
                    continue
                vn = 0.5 * (v[i, j] + v[i, j + 1]) if j < ny else v[i, j]
                vs = 0.5 * (v[i, j - 1] + v[i, j]) if j > 0 else v[i, j]

                def uat(ii, jj):
                    if 0 <= jj < ny:
                        return u[ii, jj]
                    side = "S" if jj < 0 else "N"
                    inner = u[ii, 0 if jj < 0 else ny - 1]
                    a, b = ghost(side, None, ii)
                    return a * inner + b
                ue = 0.5 * (uat(i + 1, j - 1) + uat(i + 1, j))
                uw = 0.5 * (uat(i, j - 1) + uat(i, j))
                Fn, Fs, Fe, Fw = vn * dx, vs * dx, ue * dy, uw * dy
                aN = Fn / 2 - nu * dx / dy
                aS = -Fs / 2 - nu * dx / dy
                aE = Fe / 2 - nu * dy / dx
                aW = -Fw / 2 - nu * dy / dx
                aP = (Fe - Fw + Fn - Fs) / 2 + 2 * nu * (dy / dx + dx / dy)
                b = 0.0
                if j < ny:
                    add(r, iv(i, j + 1), aN)
                else:
                    aP += aN
                if j > 0:
                    add(r, iv(i, j - 1), aS)
                else:
                    aP += aS
                if i < nx - 1:
                    add(r, iv(i + 1, j), aE)
                else:
                    a, bb = ghost("E", None, j)
                    aP += aE * a
                    b -= aE * bb
                if i > 0:
                    add(r, iv(i - 1, j), aW)
                else:
                    a, bb = ghost("W", None, j)
                    aP += aW * a
                    b -= aW * bb
                add(r, r, aP)
                if j < ny:
                    add(r, ip(i, j), dx)
                else:
                    b -= 2 * pfix["N"] * dx
                    add(r, ip(i, ny - 1), -dx)
                if j > 0:
                    add(r, ip(i, j - 1), -dx)
                else:
                    b += 2 * pfix["S"] * dx
                    add(r, ip(i, 0), -dx)
                rhs[r] = b
        # ---- continuity
        for i in range(nx):
            for j in range(ny):
                r = ip(i, j)
                if not pfix and i == 0 and j == 0:
                    add(r, r, 1.0)
                    continue
                add(r, iu(i + 1, j), dy)
                add(r, iu(i, j), -dy)
                add(r, iv(i, j + 1), dx)
                add(r, iv(i, j), -dx)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        x = spl.spsolve(A.tocsc(), rhs)
        un = x[:nu_].reshape(nx + 1, ny)
        vn_ = x[nu_:nu_ + nv_].reshape(nx, ny + 1)
        p = x[nu_ + nv_:].reshape(nx, ny)
        scale = max(np.abs(un).max(), np.abs(vn_).max(), 1e-30)
        change = max(np.abs(un - u).max(), np.abs(vn_ - v).max()) / scale
        u, v = un, vn_
        if change < tol:
            break
    return (u, v), {"iterations": it, "converged": bool(change < tol), "change": float(change)}, p
