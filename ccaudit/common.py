"""
ccaudit.common -- plumbing shared by every module.

Design rules enforced here:

* every write is atomic (tmp file + os.replace), so a session killed at the
  platform wall-clock limit never leaves a half-written JSON behind;
* every path stored inside a JSON is RELATIVE to that JSON, so an output
  folder keeps working after the platform remounts it read-only under a
  different prefix;
* the DEV/TEST split is a pure function of the identity pair, so it is stable
  across sessions, machines and dataset versions without storing anything;
* the bootstrap resamples CLIPS, never frames, because two frames of one clip
  are near-duplicates and frame-level resampling would understate variance.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# ids and names
# --------------------------------------------------------------------------

def stable_id(*parts: Any) -> str:
    """Deterministic short id.  Used for sample ids and every cache key."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


_SAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def safe_name(s: str) -> str:
    """Filesystem-safe version of a detector/tag name."""
    return _SAFE.sub("_", str(s)).strip("_") or "x"


# --------------------------------------------------------------------------
# JSON io
# --------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(path: str, obj: Any, indent: Optional[int] = None) -> str:
    """Atomically write `obj` as JSON to `path`.  Returns the path."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# relative-path index
# --------------------------------------------------------------------------

_PATH_FIELDS = ("real", "fake", "lab")


def save_index(path: str, records: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    """
    Write parsed/index.json.  Path fields of every record are stored relative
    to the directory containing `path`.
    """
    base = os.path.dirname(os.path.abspath(path))
    out: List[Dict[str, Any]] = []
    for rec in records:
        r = dict(rec)
        for f in _PATH_FIELDS:
            if f in r and r[f]:
                p = r[f]
                if os.path.isabs(p):
                    r[f] = os.path.relpath(p, base)
        out.append(r)
    return save_json(path, {"meta": dict(meta), "records": out})


def load_index(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read an index.json and resolve every path field to an absolute path
    against the index's own directory.  Returns (records, meta).  Each
    record receives a private `_index_dir` key holding that directory.
    """
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    blob = load_json(path)
    if isinstance(blob, list):            # tolerate a bare list of records
        recs, meta = blob, {}
    else:
        recs, meta = blob.get("records", []), blob.get("meta", {})
    out = []
    for rec in recs:
        r = dict(rec)
        for f in _PATH_FIELDS:
            if f in r and r[f]:
                r[f] = resolve_path(r[f], base)
        r["_index_dir"] = base
        out.append(r)
    return out, meta


def resolve_path(p: str, base: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def merge_indices(paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load several index.json files (e.g. two shards of module 2) and merge them,
    de-duplicating by sample_id.  Meta of the first is kept, with the shard
    list recorded.
    """
    recs: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}
    srcs: List[str] = []
    for p in paths:
        r, m = load_index(p)
        if not meta:
            meta = dict(m)
        srcs.append(os.path.abspath(p))
        for rec in r:
            recs.setdefault(rec["sample_id"], rec)
    meta["merged_from"] = srcs
    return sorted(recs.values(), key=lambda r: r["sample_id"]), meta


# --------------------------------------------------------------------------
# fixed DEV / TEST split
# --------------------------------------------------------------------------

DEV_FRACTION = 0.30
SPLIT_SALT = "ccaudit-split-v1"

_FFPP_PAIR = re.compile(r"^(\d{3})_(\d{3})$")
_CELEB_FAKE = re.compile(r"^id(\d+)_id(\d+)_(\d{4})$")
_CELEB_REAL = re.compile(r"^id(\d+)_(\d{4})$")


def split_key(pair_id: str) -> str:
    """
    Canonical key whose hash decides the split.  Designed so that no identity
    can leak between DEV and TEST.

    FF++     "033_097" and "097_033" -> "033_097"  (unordered identity pair)
    Celeb-DF "id0_id16_0000"          -> "id0_0000" (the TARGET clip, which is
             the authentic clip the fake was built on; the authentic clip
             "id0_0000" maps to itself)
    anything else -> the pair_id unchanged
    """
    m = _FFPP_PAIR.match(pair_id)
    if m:
        a, b = m.group(1), m.group(2)
        return "_".join(sorted((a, b)))
    m = _CELEB_FAKE.match(pair_id)
    if m:
        return f"id{int(m.group(1))}_{m.group(3)}"
    m = _CELEB_REAL.match(pair_id)
    if m:
        return f"id{int(m.group(1))}_{m.group(2)}"
    return pair_id


def split_of(pair_id: str) -> str:
    """'dev' for 30 % of identity pairs, 'test' for the rest.  Deterministic."""
    h = hashlib.blake2b(
        (SPLIT_SALT + "|" + split_key(pair_id)).encode("utf-8"), digest_size=8
    ).hexdigest()
    bucket = int(h, 16) % 10_000
    return "dev" if bucket < int(DEV_FRACTION * 10_000) else "test"


def filter_split(records: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    if split in ("", "all", None):
        return list(records)
    if split not in ("dev", "test"):
        raise ValueError(f"split must be dev|test|all, got {split!r}")
    return [r for r in records if split_of(r["pair_id"]) == split]


# --------------------------------------------------------------------------
# deterministic subsetting and sharding
# --------------------------------------------------------------------------

def parse_shard(spec: Optional[str]) -> Tuple[int, int]:
    """'2/5' -> (2, 5).  None/'' -> (0, 1)."""
    if not spec:
        return 0, 1
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", str(spec).strip())
    if not m:
        raise ValueError(f"bad shard spec {spec!r}, expected 'i/n'")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not (0 <= i < n):
        raise ValueError(f"bad shard spec {spec!r}")
    return i, n


def shard(items: Sequence[Any], spec: Optional[str], key=None) -> List[Any]:
    """
    Deterministic sharding by a hash of the item key rather than by position,
    so that adding samples later does not reshuffle which shard owns which
    sample (that would invalidate a partially filled cache).
    """
    i, n = parse_shard(spec)
    if n == 1:
        return list(items)
    keyf = key or (lambda x: x.get("sample_id") if isinstance(x, dict) else x)
    out = []
    for it in items:
        h = int(hashlib.blake2b(str(keyf(it)).encode(), digest_size=8).hexdigest(), 16)
        if h % n == i:
            out.append(it)
    return out


def limit_samples(
    records: Sequence[Dict[str, Any]], n: int, seed: int, key: str = "sample_id"
) -> List[Dict[str, Any]]:
    """
    Deterministic random subset of size n.  Implemented as "sort by a seeded
    hash of the key, take the first n" so that the chosen subset does not
    depend on the order of `records` nor on how many records were passed in.
    Every shard, detector and session therefore audits the same samples.
    """
    if not n or n <= 0 or n >= len(records):
        return list(records)
    def rank(r):
        return hashlib.blake2b(
            f"{seed}|{r[key]}".encode(), digest_size=8
        ).hexdigest()
    return sorted(records, key=rank)[:n]


# --------------------------------------------------------------------------
# image / JPEG path
# --------------------------------------------------------------------------

def jpeg_roundtrip(img: np.ndarray, q: int) -> np.ndarray:
    """
    Encode and decode through JPEG at quality q.  q <= 0 disables it (returns
    a copy).  This is the only way any module applies compression, so the
    original and every counterfactual go through an identical number of JPEG
    passes.
    """
    import cv2

    if q is None or q <= 0:
        return img.copy()
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError("JPEG decode failed")
    return out


def imwrite_jpeg(path: str, img: np.ndarray, q: int) -> None:
    """Write a JPEG at an explicit quality, atomically."""
    import cv2

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf.tobytes())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def imwrite_png(path: str, img: np.ndarray) -> None:
    import cv2

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf.tobytes())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def imread(path: str, flags: int = 1) -> np.ndarray:
    import cv2

    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"could not read image {path}")
    return img


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    groups: Optional[Sequence[Any]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI.  If `groups` is given the bootstrap is a cluster
    bootstrap: whole groups (clips) are resampled with replacement and all
    their values move together.  Returns (point, lo, hi).
    """
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")

    stat = (lambda a: float(np.mean(a))) if statistic == "mean" else (
        lambda a: float(np.median(a))
    )
    point = stat(v)
    if v.size == 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        boots = np.mean(v[idx], axis=1) if statistic == "mean" else np.median(
            v[idx], axis=1
        )
    else:
        g = np.asarray(list(groups))[: len(values)]
        g = g[np.isfinite(np.asarray(list(values), dtype=np.float64))]
        uniq, inv = np.unique(g, return_inverse=True)
        by_group = [np.where(inv == i)[0] for i in range(len(uniq))]
        if len(uniq) == 1:
            return point, point, point
        boots = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            sel = np.concatenate([by_group[p] for p in pick])
            boots[b] = stat(v[sel])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_p_two_sided(
    values: Sequence[float],
    groups: Optional[Sequence[Any]] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """
    Two-sided bootstrap p-value for H0: mean == 0, computed as
    2*min(P(boot<=0), P(boot>=0)) with a +1/(n+1) correction so p is never 0.
    With `groups`, a cluster bootstrap is used as in `bootstrap_ci`.
    """
    v_all = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(v_all)
    v = v_all[finite]
    if v.size < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        boots = np.mean(v[idx], axis=1)
    else:
        # Apply the same finite mask to the groups so that they stay aligned
        # with the retained values.
        g = np.asarray(list(groups))[: v_all.size]
        g = g[finite]
        uniq, inv = np.unique(g, return_inverse=True)
        by_group = [np.where(inv == i)[0] for i in range(len(uniq))]
        if len(uniq) < 2:
            return 1.0
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            sel = np.concatenate([by_group[p] for p in pick])
            boots[b] = float(np.mean(v[sel]))
    lo = (np.sum(boots <= 0) + 1) / (n_boot + 1)
    hi = (np.sum(boots >= 0) + 1) / (n_boot + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm step-down.  Returns a reject/keep flag per input position."""
    p = list(pvals)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho without scipy (average ranks, then Pearson)."""
    a = np.asarray(list(x), float)
    b = np.asarray(list(y), float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    return pearson(_rankdata(a), _rankdata(b))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(list(x), float)
    b = np.asarray(list(y), float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # Ties receive the average rank.
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Kendall tau-b, O(n^2); n is at most the vocabulary size.  `tx` counts
    pairs tied in x only and `ty` pairs tied in y only; pairs tied in both
    enter neither term, which matches the tau-b definition.
    """
    a = np.asarray(list(x), float)
    b = np.asarray(list(y), float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 2:
        return float("nan")
    conc = disc = tx = ty = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        s = np.sign(da) * np.sign(db)
        conc += int(np.sum(s > 0))
        disc += int(np.sum(s < 0))
        tx += int(np.sum((da == 0) & (db != 0)))
        ty += int(np.sum((db == 0) & (da != 0)))
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / den) if den > 0 else float("nan")


def auc_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    ROC AUC via the rank (Mann-Whitney) formula, ties averaged.  labels: 1 for
    positive (fake), 0 for negative (real).
    """
    s = np.asarray(list(scores), float)
    y = np.asarray(list(labels), int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n0 == 0 or n1 == 0:
        return float("nan")
    r = _rankdata(s)
    return float((np.sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def entropy_norm(p: Sequence[float]) -> float:
    """Shannon entropy of a distribution, normalised to [0,1] by log(K)."""
    a = np.asarray(list(p), float)
    a = a[np.isfinite(a)]
    a = np.clip(a, 0, None)
    ssum = a.sum()
    if ssum <= 0 or a.size < 2:
        return float("nan")
    a = a / ssum
    nz = a[a > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(a.size))


# --------------------------------------------------------------------------
# disk, scratch, time
# --------------------------------------------------------------------------

def scratch_dir(sub: str = "") -> str:
    """
    A large, session-local scratch directory.  On Kaggle, /kaggle/temp is not
    counted against the output size cap and is erased when the session ends.
    Everything large and disposable goes here: HF weights, video frames,
    materialised condition images.
    """
    for base in ("/kaggle/temp", "/kaggle/tmp", "/tmp"):
        if os.path.isdir(os.path.dirname(base)) or os.path.isdir(base):
            root = os.path.join(base, "ccaudit")
            try:
                os.makedirs(os.path.join(root, sub) if sub else root, exist_ok=True)
                return os.path.join(root, sub) if sub else root
            except OSError:
                continue
    root = os.path.join(tempfile.gettempdir(), "ccaudit", sub)
    os.makedirs(root, exist_ok=True)
    return root


def disk_free_gb(path: str = "/kaggle/working") -> float:
    try:
        st = os.statvfs(path if os.path.isdir(path) else "/")
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return float("nan")


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


@dataclass
class Budget:
    """
    Hard wall-clock budget.  Every long loop checks `expired` and exits
    cleanly after checkpointing, so that a session is never killed mid-write
    by the platform time limit.
    """
    minutes: float
    t0: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.t0:
            self.t0 = time.time()

    @property
    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60.0

    @property
    def remaining_min(self) -> float:
        return float("inf") if self.minutes <= 0 else self.minutes - self.elapsed_min

    @property
    def expired(self) -> bool:
        return self.minutes > 0 and self.elapsed_min >= self.minutes

    def report(self) -> str:
        if self.minutes <= 0:
            return f"{self.label} elapsed {self.elapsed_min:.1f} min (no budget)"
        return (
            f"{self.label} elapsed {self.elapsed_min:.1f} / {self.minutes:.0f} min"
            f" ({self.remaining_min:.1f} left)"
        )


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def code_hash() -> str:
    """Hash of every .py file in the package; embedded in every output."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(os.listdir(pkg)):
        if name.endswith(".py"):
            with open(os.path.join(pkg, name), "rb") as fh:
                h.update(name.encode())
                h.update(fh.read())
    return h.hexdigest()


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def gpu_info() -> List[Dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append({"name": parts[0], "memory": parts[1],
                         "driver": parts[2] if len(parts) > 2 else ""})
    return gpus


def provenance(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything needed to reproduce a run, embedded in every output file."""
    from . import regions as _regions

    info: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "code_hash": code_hash(),
        "vocab": _regions.active_vocab(),
        "packages": {
            p: _pkg_version(p)
            for p in ("numpy", "opencv-python", "opencv-python-headless",
                      "mediapipe", "torch", "transformers", "accelerate",
                      "timm", "peft", "bitsandbytes", "qwen-vl-utils",
                      "simple-lama-inpainting")
        },
        "gpus": gpu_info(),
        "cwd": os.getcwd(),
    }
    if extra:
        info.update(extra)
    return info


def write_provenance(out_dir: str, name: str, extra: Optional[Dict] = None) -> str:
    return save_json(
        os.path.join(out_dir, f"provenance_{safe_name(name)}.json"),
        provenance(extra), indent=2,
    )


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def chunked(it: Iterable[Any], n: int) -> Iterable[List[Any]]:
    buf: List[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def banner(title: str, width: int = 72) -> str:
    return "\n" + "=" * width + f"\n{title}\n" + "=" * width


def copy_tree_into(src: str, dst: str) -> None:
    """Copy src/* into dst (dst may exist).  Used to stage read-only inputs."""
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s, d = os.path.join(src, item), os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
