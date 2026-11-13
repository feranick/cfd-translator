"""transfer.py — a request run on a cluster cfdt's servers cannot reach (NERSC Perlmutter, MIT Engaging), brought
back for study_status, window averages, render_images and comparisons (other/supercomputers/clusters.md).
Version: 2026.11.13.8

    pack(request, state=False, out=None, studies=None, init=False)
                                           on the cluster: a .tgz of the request's results, without the case's fields
    import_request(path)                   on a cfdt machine: the .tgz (or the folder) as a request under CFDT_OUT
    init_check(ref, fields='last')         can '<request>/<study>' of this machine be started from (init_from)?

init=True (2026.11.13.6) packs what solve.init_from needs from finished studies of another machine (the router copies
it when a study's init_from names a run there): case/system, case/constant (polyMesh and the dictionaries; no
surfaces) and the latest reconstructed time directory, besides the results. Imported, the request answers
init_from '<imported request>/<study>' on this machine (import gives the refs: 'init_from').

The archive holds, per study: <name>.json (the result: reliable, unreliable_because, averages, revolution averages,
drift...), images/, mesh.json, job.json, job.log, transient.json, state.json (the plan: window averages need it),
case/postProcessing/ (the monitors' histories: window averages, frequency) and the case's logs (log.foamRun and
other large logs only their last LOG_TAIL_MB). state=True adds state.npz (the mesh and the final or mean fields:
render_images and reprocess's results without the case; about 100 B a cell). The time directories, processor*
and polyMesh stay on the cluster.

Imported requests carry 'imported' in request.json (where from, when); paths in their JSON files point at the new
folder; studies that had no result when packed get one with state 'not_finished' (pack again when they finish).
"""
from __future__ import annotations

import io
import json
import shutil
import socket
import tarfile
import tempfile
import time
from pathlib import Path

from . import __version__

LOG_TAIL_MB = 2.0
MANIFEST = "cfdt_pack.json"
_KEEP_FILES = ("{name}.json", "mesh.json", "job.json", "job.log", "transient.json", "state.json")
_INIT_SKIP = ("triSurface", "geometry", "extendedFeatureEdgeMesh")     # constant/: surfaces the run does not need


def init_check(ref: str, fields: str = "last") -> dict:
    """Whether '<request>/<study>' (this machine) can be started from: finished, a reconstructed time to take."""
    from .transient import init_problems, source_folder, source_times
    src = source_folder(ref)
    probs = init_problems({"solve": {"init_from": ref, "init_fields": fields}})
    out = {"ref": ref, "folder": str(src), "ok": not probs, "problems": probs}
    res = src / f"{src.name}.json"
    out["finished"] = res.is_file()
    if res.is_file():
        try:
            r = json.loads(res.read_text())
            out.update(result_ok=bool(r.get("ok")), reliable=r.get("reliable"))
        except ValueError:
            out["finished"] = False
    t = source_times(src)
    if t:
        tdir = t[-1][1]
        # when the fields last changed (a reprocess rewrites the result, not the fields): the router copies a run
        # again only when its fields are newer than the copy (2026.11.13.8)
        mt = [tdir.stat().st_mtime] + [f.stat().st_mtime for f in tdir.iterdir() if f.is_file()]
        out.update(time=t[-1][0], init_bytes=_init_bytes(src, tdir), fields_mtime=max(mt))
    if not out["finished"]:
        out["ok"] = False
        out["problems"] = probs + [f"{src.name} has no result yet: wait until it finishes"]
    return out


def _init_files(d: Path, tdir: Path):
    """(path, arcname relative to the study folder) of what init_from takes from study folder d."""
    c = d / "case"
    for sub in ("system", "constant"):
        for f in sorted((c / sub).rglob("*")) if (c / sub).is_dir() else []:
            rel = f.relative_to(c)
            if f.is_file() and not f.is_symlink() and not (sub == "constant" and len(rel.parts) > 1
                                                            and rel.parts[1] in _INIT_SKIP):
                yield f, f"case/{rel.as_posix()}"
    for f in sorted(tdir.rglob("*")):
        if f.is_file() and not f.is_symlink():
            yield f, f"case/{tdir.relative_to(c).as_posix()}/{f.relative_to(tdir).as_posix()}"


def _init_bytes(d: Path, tdir: Path) -> int:
    return sum(f.stat().st_size for f, _ in _init_files(d, tdir))


def _out_root() -> Path:
    from .api import out_root
    return Path(out_root())


def pack(request: str, state: bool = False, out: str | None = None, studies: list | None = None,
         init: bool = False) -> dict:
    """A .tgz of a finished (or partly finished) request, to copy to a cfdt machine and import there. studies: only
    these (the archive's request.json lists only them). init: also what solve.init_from takes from each (finished
    studies only)."""
    from .transient import source_times
    folder = Path(request).expanduser().resolve()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    meta = json.loads((folder / "request.json").read_text())
    names = list(meta.get("names", []))
    if studies:
        bad = [n for n in studies if n not in names]
        if bad:
            return {"error": f"not in {folder.name}: {bad}", "studies": names}
        names = [n for n in names if n in studies]
    tdirs = {}
    if init:
        probs = []
        for n in names:
            c = init_check(str(folder / n))
            if not c["ok"]:
                probs += [f"{n}: {p}" for p in c["problems"]]
            else:
                tdirs[n] = source_times(folder / n)[-1][1]
        if probs:
            return {"error": "cannot pack for init_from: " + "; ".join(probs)}
    dest = Path(out).expanduser() if out else folder.parent / f"{folder.name}.cfdt.tgz"
    done, missing, size_note = [], [], []
    manifest = {"request": str(folder), "name": folder.name, "host": socket.gethostname(), "packed": time.time(),
                "translator": __version__, "state": bool(state), "init": bool(init), "studies": {}}
    tail = int(LOG_TAIL_MB * 1024 * 1024)
    # fields of millions of cells: fast compression (binary fields hardly compress further)
    with tarfile.open(dest, "w:gz", compresslevel=1 if init else 9) as tar:
        def add(p: Path, arc: str):
            if p.is_file():
                tar.add(str(p), arcname=arc, recursive=False)

        def add_bytes(data: bytes, arc: str):
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime = len(data), int(time.time())
            tar.addfile(ti, io.BytesIO(data))
        if studies:
            sub = dict(meta, names=names, studies={k: v for k, v in (meta.get("studies") or {}).items() if k in names})
            add_bytes(json.dumps(sub, default=float).encode(), f"{folder.name}/request.json")
        else:
            add(folder / "request.json", f"{folder.name}/request.json")
        for f in ("studies.json", "results.json"):
            add(folder / f, f"{folder.name}/{f}")
        for n in names:
            d = folder / n
            if not d.is_dir():
                continue
            has = (d / f"{n}.json").is_file()
            (done if has else missing).append(n)
            manifest["studies"][n] = {"result": has}
            for pat in _KEEP_FILES:
                add(d / pat.format(name=n), f"{folder.name}/{n}/{pat.format(name=n)}")
            if state:
                add(d / "state.npz", f"{folder.name}/{n}/state.npz")
            if (d / "images").is_dir():
                tar.add(str(d / "images"), arcname=f"{folder.name}/{n}/images")
            c = d / "case"
            if (c / "postProcessing").is_dir():
                tar.add(str(c / "postProcessing"), arcname=f"{folder.name}/{n}/case/postProcessing")
            for lg in sorted(c.glob("log.*")) if c.is_dir() else []:
                if lg.stat().st_size <= tail:
                    add(lg, f"{folder.name}/{n}/case/{lg.name}")
                    continue
                with open(lg, "rb") as fh:          # a long log: its end (progress, the last residuals, the error)
                    fh.seek(lg.stat().st_size - tail)
                    data = b"(cfdt pack: the first part of this log was left out)\n" + fh.read()
                ti = tarfile.TarInfo(f"{folder.name}/{n}/case/{lg.name}")
                ti.size, ti.mtime = len(data), int(lg.stat().st_mtime)
                tar.addfile(ti, io.BytesIO(data))
                size_note.append(lg.name)
            if n in tdirs:
                nb = 0
                for f, arc in _init_files(d, tdirs[n]):
                    if not arc.startswith("case/postProcessing/"):
                        add(f, f"{folder.name}/{n}/{arc}")
                        nb += f.stat().st_size
                manifest["studies"][n]["init"] = {"time": source_times(d)[-1][0], "bytes": nb}
        data = json.dumps(manifest, indent=1).encode()
        ti = tarfile.TarInfo(f"{folder.name}/{MANIFEST}")
        ti.size, ti.mtime = len(data), int(time.time())
        tar.addfile(ti, io.BytesIO(data))
    out_ = {"archive": str(dest), "bytes": dest.stat().st_size, "studies_with_results": done,
            "studies_without_results": missing,
            "next": f"copy {dest.name} to a cfdt machine and import it (import_results, or `cfdt import {dest.name}`)"}
    if size_note:
        out_["logs_cut"] = f"{len(set(size_note))} long log(s) kept to their last {LOG_TAIL_MB:g} MB"
    if missing:
        out_["note"] = "studies without a result yet are imported as 'not_finished': pack again when they finish"
    if init:
        out_["init"] = {n: manifest["studies"][n]["init"] for n in tdirs}
    return out_


def _safe_members(tar: tarfile.TarFile, top: str):
    for m in tar.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != top or m.issym() or m.islnk():
            raise ValueError(f"unexpected entry {m.name!r} in the archive")
        yield m


def import_request(path: str, name: str | None = None) -> dict:
    """A packed request (.tgz from pack, or its unpacked folder) as a request under CFDT_OUT."""
    src = Path(path).expanduser()
    if not src.exists():
        return {"error": f"no file or folder {src} on this machine (copy it here first; the router copies local files)"}
    root = _out_root()
    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cfdt_import_", dir=str(root)))
    try:
        if src.is_dir():                     # a request folder copied back whole: packed here first
            if not (src / "request.json").is_file():
                return {"error": f"{src} holds no request.json"}
            names_ = json.loads((src / "request.json").read_text()).get("names") or []
            p = pack(str(src), state=any((src / n / "state.npz").is_file() for n in names_), out=str(tmp / "dir.tgz"))
            if "error" in p:
                return p
            src_arc = Path(p["archive"])
        else:
            src_arc = src
        try:
            with tarfile.open(src_arc, "r:*") as tar:
                tops = {Path(m.name).parts[0] for m in tar.getmembers() if Path(m.name).parts}
                if len(tops) != 1:
                    return {"error": f"{src.name}: expected one request folder, found {sorted(tops)}"}
                top = tops.pop()
                tar.extractall(tmp, members=list(_safe_members(tar, top)))
        except (tarfile.TarError, ValueError) as exc:
            return {"error": f"{src.name}: not a cfdt pack ({exc})"}
        got = tmp / top
        if not (got / "request.json").is_file():
            return {"error": f"{src.name}: no request.json in {top}"}
        man = {}
        if (got / MANIFEST).is_file():
            man = json.loads((got / MANIFEST).read_text())
        new = name or top
        dest = root / new
        k = 1
        while dest.exists():
            k += 1
            dest = root / f"{new}_import{k}"
        old = man.get("request")
        meta = json.loads((got / "request.json").read_text())
        names = list(meta.get("names", []))
        # SLURM job ids belong to the cluster: no local queue lookups for these studies
        for n in names:
            jf = got / n / "job.json"
            if jf.is_file():
                jf.rename(got / n / "job.cluster.json")
        shutil.move(str(got), str(dest))
        # the cluster's paths in the JSON files point here now
        if old:
            for f in dest.rglob("*.json"):
                try:
                    t = f.read_text()
                except (OSError, UnicodeDecodeError):
                    continue
                if old in t:
                    f.write_text(t.replace(old, str(dest)))
        meta = json.loads((dest / "request.json").read_text())
        meta["imported"] = {"from": str(src), "host": man.get("host"), "path": old, "packed": man.get("packed"),
                            "at": time.time(), "translator": man.get("translator"), "state": man.get("state"),
                            "init": bool(man.get("init"))}
        (dest / "request.json").write_text(json.dumps(meta, default=float))
        unfinished = []
        for n in names:
            if (dest / n).is_dir() and not (dest / n / f"{n}.json").is_file():
                unfinished.append(n)
                (dest / n / f"{n}.json").write_text(json.dumps({
                    "name": n, "ok": False, "state": "not_finished",
                    "why": f"imported from {man.get('host') or 'the cluster'} before it finished: pack it again there "
                           "when it is done, and import that"}, indent=1))
        out = {"request": str(dest), "imported_from": man.get("host"), "studies": names, "not_finished": unfinished,
               "fields": bool(man.get("state")),
               **({"init_from": {n: f"{dest.name}/{n}" for n, v in (man.get("studies") or {}).items()
                                 if isinstance(v, dict) and v.get("init")}} if man.get("init") else {}),
               "next": f"study_status(request='{dest}') for the results; window=[t0, t1] for other averages"
                       + ("; render_images works on its studies" if man.get("state") else
                          "; render_images needs the fields (pack with --state)")}
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
