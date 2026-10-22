"""foamio.py — tolerant readers of OpenFOAM files and logs.
Version: 2026.10.21.2

    read_foam(text) -> dict              any ASCII dictionary or field file (lists as NumPy arrays)
    read_list(text) -> np.ndarray        polyMesh lists (points, owner, neighbour)
    read_faces(text) -> list[list[int]]  polyMesh faces
    read_boundary(text) -> list[dict]    polyMesh boundary
    read_mesh_arrays(dir) -> dict        a whole polyMesh, ASCII or binary (any face size)
    parse_log(text) -> dict              residuals, iterations, convergence, fatal errors
    parse_checkmesh(text) -> dict        checkMesh verdict and quality numbers
    latest_time(case) -> Path | None     the last time directory
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_TOKEN = re.compile(r'"[^"]*"|[{}();]|[^\s{}();"]+')
_NONUNI = re.compile(r"nonuniform\s+List<(\w+)>\s*(\d+)\s*\(")
_CLOSE = re.compile(r"\)\s*\)")


def _matching(text: str, start: int) -> int:
    """Index of the ')' that closes the '(' at text[start]."""
    b = np.frombuffer(text[start:].encode("latin-1", "replace"), np.uint8)
    depth = np.cumsum((b == 40).astype(np.int64) - (b == 41).astype(np.int64))
    hit = np.nonzero(depth == 0)[0]
    if not len(hit):
        raise ValueError("unbalanced parentheses")
    return start + int(hit[0])


def _numbers(s: str) -> np.ndarray:
    return np.array(s.replace("(", " ").replace(")", " ").split(), float)


def read_foam(text: str) -> dict:
    text = _COMMENT.sub(" ", text)
    arrays = []
    out, pos = [], 0
    for m in _NONUNI.finditer(text):
        if m.start() < pos:
            continue
        typ, n = m.group(1), int(m.group(2))
        op = m.end() - 1
        width = {"scalar": 1, "vector": 3, "symmTensor": 6, "tensor": 9}.get(typ, 1)
        # the closing parenthesis without scanning the rest of the file: scalars have no inner
        # parentheses; other types end at the first ')' followed by ')'
        if width == 1 or n == 0:
            cl = text.find(")", op + 1)
        else:
            mm = _CLOSE.search(text, op + 1)
            cl = mm.end() - 1 if mm else -1
        a = _numbers(text[op + 1:cl]) if cl > op else np.zeros(0)
        if a.size != n * width:
            cl = _matching(text, op)
            a = _numbers(text[op + 1:cl])
        if a.size != n * width:
            raise ValueError(f"list of {n} {typ} has {a.size // width} entries")
        a = a.reshape(n, width) if width > 1 else a
        out.append(text[pos:m.start()])
        out.append(f" __ARR{len(arrays)}__ ")
        arrays.append(a)
        pos = cl + 1
    out.append(text[pos:])
    toks = _TOKEN.findall("".join(out))
    d, _ = _parse_dict(toks, 0, arrays)
    return d


def _value(toks: list, arrays: list):
    if len(toks) == 1:
        t = toks[0]
        if t.startswith("__ARR"):
            return arrays[int(t[5:-2])]
        try:
            return float(t)
        except ValueError:
            return t.strip('"')
    if toks and toks[0] == "uniform":
        rest = toks[1:]
        if len(rest) == 1:
            return float(rest[0])
        return np.array([float(x) for x in rest if x not in ("(", ")")])
    if toks and toks[0] == "nonuniform" and len(toks) == 2 and toks[1].startswith("__ARR"):
        return arrays[int(toks[1][5:-2])]
    if toks and toks[0] == "nonuniform" and len(toks) >= 3:      # empty or inline list: nonuniform List<T> 0()
        nums = [x for x in toks[2:] if x not in ("(", ")")]
        m = re.match(r"(\d+)", toks[2])
        return np.array([float(x) for x in nums[1:] if re.match(r"^[-+.\deE]+$", x)]) if m else toks
    return " ".join(toks)


def _parse_dict(toks: list, i: int, arrays: list):
    d = {}
    n = len(toks)
    while i < n and toks[i] != "}":
        key = toks[i].strip('"')
        i += 1
        if i < n and toks[i] == "{":
            sub, i = _parse_dict(toks, i + 1, arrays)
            d[key] = sub
            i += 1
            continue
        val, depth = [], 0
        while i < n and not (toks[i] == ";" and depth == 0):
            if toks[i] == "(":
                depth += 1
            elif toks[i] == ")":
                depth -= 1
            if toks[i] == "}" and depth == 0:
                break
            val.append(toks[i])
            i += 1
        i += 1 if i < n and toks[i] == ";" else 0
        if key.startswith("#"):
            continue
        d[key] = _value(val, arrays)
    return d, i


def _strip_header(text: str) -> str:
    text = _COMMENT.sub(" ", text)
    m = re.search(r"FoamFile\s*\{.*?\}", text, re.S)
    return text[m.end():] if m else text


def read_list(text: str) -> np.ndarray:
    """A polyMesh list file (points, owner, neighbour): numbers after the count."""
    body = _strip_header(text)
    m = re.search(r"(\d+)\s*\(", body)
    if not m:
        raise ValueError("no list found")
    n = int(m.group(1))
    op = m.end() - 1
    cl = _matching(body, op)
    a = _numbers(body[op + 1:cl])
    return a.reshape(n, -1) if n and len(a) != n else a


def read_faces(text: str) -> list:
    body = _strip_header(text)
    m = re.search(r"(\d+)\s*\(", body)
    op = m.end() - 1
    cl = _matching(body, op)
    return [list(map(int, f.split())) for f in re.findall(r"\d+\s*\(([^)]*)\)", body[op + 1:cl])]


def read_boundary(text: str) -> list:
    body = _strip_header(text)
    out = []
    for m in re.finditer(r"(\w+)\s*\{([^}]*)\}", body):
        e = dict(re.findall(r"(\w+)\s+([^;]+);", m.group(2)))
        out.append({"name": m.group(1), "type": e.get("type", "patch").strip(),
                     "n": int(e["nFaces"]), "start": int(e["startFace"])})
    return out


# ---- polyMesh files, ASCII or binary (snappyHexMesh writes binary: fast to read) -------------------
_HDR = re.compile(rb"FoamFile\s*\{(.*?)\}", re.S)
_LIST = re.compile(rb"(\d+)\s*\(")


def _mesh_file(path: Path) -> tuple[dict, bytes, int]:
    """(header entries, file bytes, offset after the header) of a polyMesh file."""
    raw = Path(path).read_bytes()
    m = _HDR.search(raw)
    if not m:
        raise ValueError(f"{path}: no FoamFile header")
    hdr = {k.decode(): v.decode().strip().strip('"') for k, v in re.findall(rb"(\w+)\s+([^;]+);", m.group(1))}
    return hdr, raw, m.end()


def _skip_comments(raw: bytes, pos: int) -> int:
    """Position of the next token after whitespace and // comments (ASCII parts of a file)."""
    while True:
        while pos < len(raw) and raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 2] == b"//":
            nl = raw.find(b"\n", pos)
            pos = len(raw) if nl < 0 else nl + 1
        elif raw[pos:pos + 2] == b"/*":
            e = raw.find(b"*/", pos)
            pos = len(raw) if e < 0 else e + 2
        else:
            return pos


def _next_list(raw: bytes, pos: int, binary: bool, dtype, width: int = 1):
    """The list starting at or after pos: (array, position after its ')'). Binary lists are 'N(' + raw
    bytes + ')'; ASCII lists are numbers (vectors in parentheses)."""
    pos = _skip_comments(raw, pos)
    m = _LIST.match(raw, pos)
    if not m:
        raise ValueError(f"no list at byte {pos}")
    n = int(m.group(1))
    op = m.end() - 1
    if binary:
        nb = n * width * np.dtype(dtype).itemsize
        if np.dtype(dtype).kind == "i" and n and raw[op + 1 + nb:op + 2 + nb] != b")":
            dtype = "<i8" if np.dtype(dtype).itemsize == 4 else "<i4"      # a 64-bit-label build (no 'arch' in v14)
            nb = n * width * np.dtype(dtype).itemsize
        a = np.frombuffer(raw, dtype=dtype, count=n * width, offset=op + 1) if n else np.zeros(0, dtype)
        end = op + 1 + nb
        end = raw.index(b")", end) + 1
        return (a.reshape(n, width) if width > 1 else a).copy(), end
    # the closing parenthesis (vectors and faces nest one level)
    b = np.frombuffer(raw, np.uint8, count=len(raw) - op, offset=op)
    d = np.cumsum((b == 40).astype(np.int32) - (b == 41).astype(np.int32))
    cl = op + int(np.nonzero(d == 0)[0][0])
    body = raw[op + 1:cl].translate(bytes.maketrans(b"()", b"  "))
    a = np.array(body.split(), dtype=dtype)
    return (a.reshape(n, -1) if (width > 1 and n) else a), cl + 1


def _arch(hdr: dict) -> tuple:
    a = hdr.get("arch", "LSB;label=32;scalar=64")
    lab = int(re.search(r"label=(\d+)", a).group(1)) if "label=" in a else 32
    sca = int(re.search(r"scalar=(\d+)", a).group(1)) if "scalar=" in a else 64
    return ({32: "<i4", 64: "<i8"}[lab], {32: "<f4", 64: "<f8"}[sca])


def read_mesh_arrays(d: Path) -> dict:
    """constant/polyMesh as arrays: points (P, 3) in the file's units, faces (F, W) padded with -1,
    owner, neighbour, ncells, boundary [{name, type, n, start}]."""
    d = Path(d)
    out = {}
    for name, width in (("points", 3), ("owner", 1), ("neighbour", 1)):
        hdr, raw, pos = _mesh_file(d / name)
        lab, sca = _arch(hdr)
        binary = hdr.get("format", "ascii") == "binary"
        dt = sca if name == "points" else lab
        if not binary:
            dt = float if name == "points" else np.int64
        a, _ = _next_list(raw, pos, binary, dt, width)
        out[name] = np.asarray(a, float if name == "points" else np.int64)
        if name == "owner":
            m = re.search(r"nCells:\s*(\d+)", hdr.get("note", ""))
            out["ncells"] = int(m.group(1)) if m else None
    nf = len(out["owner"])
    hdr, raw, pos = _mesh_file(d / "faces")
    lab, _ = _arch(hdr)
    binary = hdr.get("format", "ascii") == "binary"
    first, end = _next_list(raw, pos, binary, lab if binary else np.int64)
    first = np.asarray(first, np.int64)
    if "Compact" in hdr.get("class", "") and len(first) in (nf, nf + 1) and _LIST.match(raw, _skip_comments(raw, end)):
        data, _ = _next_list(raw, end, binary, lab if binary else np.int64)
        data = np.asarray(data, np.int64)
        off = first if len(first) == nf + 1 else np.concatenate([[0], first])
    else:                                      # ASCII faceList: n(a b c ...) per face, flattened as n, a, b, c...
        t = first
        sizes, starts, k = np.empty(nf, np.int64), np.empty(nf, np.int64), 0
        for i in range(nf):
            sizes[i] = t[k]
            starts[i] = k + 1
            k += int(t[k]) + 1
        idx = np.concatenate([np.arange(s, s + z) for s, z in zip(starts, sizes)]) if nf else np.zeros(0, np.int64)
        data = t[idx]
        off = np.concatenate([[0], np.cumsum(sizes)])
    sizes = np.diff(off)
    W = int(sizes.max()) if nf else 3
    F = -np.ones((nf, max(W, 3)), np.int64)
    col = np.arange(len(data)) - np.repeat(off[:-1], sizes)
    F[np.repeat(np.arange(nf), sizes), col] = data
    out["faces"] = F
    if out["ncells"] is None:
        out["ncells"] = int(max(out["owner"].max(initial=-1), out["neighbour"].max(initial=-1)) + 1)
    out["boundary"] = read_boundary((d / "boundary").read_text(errors="replace"))
    return out


def latest_time(case: Path):
    best, bt = None, -1.0
    for p in Path(case).iterdir():
        if p.is_dir():
            try:
                t = float(p.name)
            except ValueError:
                continue
            if t > bt and t > 0:
                best, bt = p, t
    return best


_RES = re.compile(r"Solving for (\w+), Initial residual = ([^,\s]+), Final residual = ([^,\s]+), No Iterations (\d+)")
_TIME = re.compile(r"^Time = ([-+0-9.eE]+)", re.M)


def parse_log(text: str) -> dict:
    """Iterations, first initial residual of each field in each iteration (last and first values),
    convergence, continuity errors, fatal errors and warnings."""
    steps = [m.start() for m in _TIME.finditer(text)]
    hist: dict[str, list] = {}
    bounds = steps + [len(text)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seen = set()
        for m in _RES.finditer(text, a, b):
            f = m.group(1)
            if f in seen:
                continue
            seen.add(f)
            try:
                hist.setdefault(f, []).append(float(m.group(2)))
            except ValueError:
                pass
    out = {"iterations": len(steps), "residuals": {f: v[-1] for f, v in hist.items()},
           "first_residuals": {f: v[0] for f, v in hist.items()},
           "residual_history": {f: v[:: max(1, len(v) // 50)] for f, v in hist.items()}}
    m = re.search(r"(\w+) solution converged in\s+(\S+)\s+iterations", text)
    out["converged"] = bool(m)
    if m:
        out["converged_at"] = m.group(2)
    c = re.findall(r"continuity errors : sum local = ([^,\s]+), global = ([^,\s]+), cumulative = ([^,\s]+)", text)
    if c:
        out["continuity"] = {"sum_local": float(c[-1][0]), "global": float(c[-1][1])}
    k = text.find("FOAM FATAL")
    if k >= 0:
        out["fatal"] = "\n".join(l for l in text[k:k + 3000].splitlines()[:25] if l.strip())
    out["ended"] = bool(re.search(r"^End\s*$", text, re.M))
    out["warnings"] = len(re.findall(r"--> FOAM Warning", text))
    return out


def parse_checkmesh(text: str) -> dict:
    out = {"ok": "Mesh OK." in text}
    m = re.search(r"Failed (\d+) mesh checks", text)
    if m:
        out["failed_checks"] = int(m.group(1))
    for key, pat in (("max_non_orthogonality", r"non-orthogonality Max: ([^\s]+)"),
                     ("max_skewness", r"Max skewness = ([-+0-9.eE]+)"),
                     ("min_volume", r"Min volume = ([-+]?[0-9][0-9.eE+-]*?)\.\s+Max volume"),
                     ("max_aspect_ratio", r"Max aspect ratio = ([^\s]+)"),
                     ("cells", r"^\s*cells:\s+(\d+)")):
        mm = re.search(pat, text, re.M)
        if mm:
            try:
                out[key] = float(mm.group(1))
            except ValueError:
                pass
    out["errors"] = re.findall(r"\*\*\*[^\n]*", text)[:10]
    return out
