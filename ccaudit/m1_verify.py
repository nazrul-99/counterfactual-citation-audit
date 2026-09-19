"""
ccaudit.m1_verify -- Module 1, the PAIRING GATE.

The audit rests on one assumption: every fake clip has a frame-aligned
authentic twin, so that fake frame i and real frame i depict the same
temporal instant with the same camera, lighting and background.  If that is
false, the counterfactual "region k is not manipulated" is not a
counterfactual at all but a different moment in time.  This module is the
gate that refuses to let the rest of the pipeline run until the assumption
has been checked on the actual video files.

How the check works
-------------------
For a sample of pairs, a few frames at identical indices are decoded from
the fake and the real clip and the BORDER RING of the frame -- the outer
margin, which no face manipulation touches -- is compared.  If the two clips
are the same moment, the ring is identical up to compression noise.

Four failure modes that occur on public mirrors are handled here:

1. INEXACT SEEKING.  cv2.CAP_PROP_POS_FRAMES on H.264 lands on the nearest
   keyframe, and the real and fake clips can have different GOP structures,
   so the same requested index returns different actual frames.  Seeking is
   therefore never used; frames are decoded sequentially with grab() and
   retrieve() is called only at the wanted indices (`read_frames_at`).
2. RESOLUTION MISMATCH.  Some mirrors re-encode the manipulated videos at a
   different resolution.  Cropping to the common top-left corner would
   compare different parts of the scene, so the fake frame is RESIZED to the
   real frame's size instead.
3. RESAMPLING.  A resized clip has different high-frequency content, so a
   raw-pixel NCC can fail even on a perfectly aligned pair.  The comparison
   is therefore multi-scale: the metric is evaluated on progressively
   low-pass filtered versions and the best scale is reported (`scale_used`).
4. FRAME OFFSET.  A few clips in some mirrors are off by one or two frames.
   A small offset search is run; a pair that aligns at offset o is kept, the
   offset is written into pairs.json, and Module 2 applies it when sampling.

Gate criterion (dual):

    aligned  <=>  (NCC > 0.95 and MAE < 0.05) or (NCC > 0.80 and MAE < 0.06)

The pair set PASSES if alignment_rate >= --min-alignment (default 0.90).
`--min-alignment` is a documented knob rather than an escape hatch: lowering
it is a decision that must be recorded alongside the results, and the report
JSON stores the value that was used.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C

# --------------------------------------------------------------------------
# gate constants
# --------------------------------------------------------------------------
NCC_STRICT = 0.95
MAE_STRICT = 0.05
NCC_LOOSE = 0.80
MAE_LOOSE = 0.06

DEFAULT_MIN_ALIGNMENT = 0.90
DEFAULT_MIN_RESOLUTION = 0.95

BORDER_FRAC = 0.15          # outer ring width, fraction of each dimension
SCALES = (0, 1.0, 2.0)      # gaussian sigma of the low-pass ladder
OFFSET_SEARCH = (0, -1, 1, -2, 2)

FFPP_METHODS = [
    "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
    "FaceShifter", "DeepFakeDetection",
]
_NON_METHOD_DIRS = {
    "videos", "c23", "c40", "raw", "manipulated_sequences",
    "original_sequences", "youtube", "actors", "sequences",
}

_RE_FFPP_FAKE = re.compile(r"^(\d{3})_(\d{3})$")
_RE_FFPP_REAL = re.compile(r"^(\d{3})$")
_RE_CELEB_FAKE = re.compile(r"^id(\d+)_id(\d+)_(\d{4})$")
_RE_CELEB_REAL = re.compile(r"^(id\d+)_(\d{4})$")

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


# --------------------------------------------------------------------------
# video access
# --------------------------------------------------------------------------

def video_meta(path: str) -> Dict[str, Any]:
    """(n_frames, width, height, fps).  n_frames is the container's count,
    which can be slightly optimistic; callers must tolerate a short read."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return {"ok": False, "n_frames": 0, "w": 0, "h": 0, "fps": 0.0}
    meta = {
        "ok": True,
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    }
    cap.release()
    return meta


def read_frames_at(path: str, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """
    Decode the frames at the given 0-based indices EXACTLY.

    CAP_PROP_POS_FRAMES is deliberately not used: on H.264 it seeks to a
    keyframe and OpenCV's compensation is unreliable when the two clips of a
    pair have different GOP structures, which silently compares frame 100 of
    one clip with frame 97 of the other.  Sequential grab() is cheap (no
    colour conversion, no copy) and exact.
    """
    import cv2

    want = sorted({int(i) for i in indices if i >= 0})
    out: Dict[int, np.ndarray] = {}
    if not want:
        return out
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return out
    try:
        target = 0
        last = want[-1]
        for idx in range(last + 1):
            ok = cap.grab()
            if not ok:
                break
            if target < len(want) and idx == want[target]:
                ok2, frame = cap.retrieve()
                if ok2 and frame is not None:
                    out[idx] = frame
                target += 1
                if target >= len(want):
                    break
    finally:
        cap.release()
    return out


def true_frame_count(path: str, cap_at: int = 100000) -> int:
    """Exact frame count by decoding.  Only used when the container count is
    missing or obviously wrong; grab() makes it fast enough."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0
    n = 0
    while n < cap_at and cap.grab():
        n += 1
    cap.release()
    return n


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def border_mask(h: int, w: int, frac: float = BORDER_FRAC) -> np.ndarray:
    """Boolean mask of the outer ring (everything outside the central box)."""
    m = np.ones((h, w), bool)
    by, bx = int(round(frac * h)), int(round(frac * w))
    by = min(max(by, 1), h // 2 - 1) if h > 4 else 1
    bx = min(max(bx, 1), w // 2 - 1) if w > 4 else 1
    m[by:h - by, bx:w - bx] = False
    return m


def _gray(img: np.ndarray) -> np.ndarray:
    import cv2

    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _match_size(fake: np.ndarray, real: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Bring the fake frame to the real frame's size.  The frame is resized,
    never cropped: a top-left crop compares different parts of the scene and
    produces false MISALIGNED verdicts.
    """
    import cv2

    if fake.shape[:2] == real.shape[:2]:
        return fake, False
    th, tw = real.shape[:2]
    interp = cv2.INTER_AREA if (fake.shape[0] > th) else cv2.INTER_LINEAR
    return cv2.resize(fake, (tw, th), interpolation=interp), True


def _ncc_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if x.size < 16:
        return float("nan"), float("nan")
    mae = float(np.mean(np.abs(x - y)) / 255.0)
    sx, sy = x.std(), y.std()
    if sx < 1e-6 or sy < 1e-6:
        # A flat border (letterboxing, black bars) leaves NCC undefined.
        # Report NCC as 1.0 when both are flat and agree, else 0.
        ncc = 1.0 if mae < 0.01 else 0.0
    else:
        ncc = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
    return ncc, mae


def compare_frames(
    fake: np.ndarray, real: np.ndarray, border_frac: float = BORDER_FRAC
) -> Dict[str, Any]:
    """
    Multi-scale border comparison of one frame pair.

    Returns the best (highest-NCC) scale's metrics plus diagnostics:
      ncc, mae            on the border ring at the chosen scale
      scale_used          gaussian sigma of that scale (0 = raw pixels)
      resized             whether the fake had to be resized
      center_mae          MAE inside the central box (should be > 0 for a
                          genuine fake/real pair; ~0 means the two files are
                          the same video)
    """
    import cv2

    fake2, resized = _match_size(fake, real)
    gf, gr = _gray(fake2), _gray(real)
    h, w = gr.shape[:2]
    ring = border_mask(h, w, border_frac)
    centre = ~ring

    best: Optional[Dict[str, Any]] = None
    for sigma in SCALES:
        if sigma <= 0:
            af, ar = gf, gr
        else:
            k = int(2 * round(3 * sigma) + 1)
            af = cv2.GaussianBlur(gf, (k, k), sigma)
            ar = cv2.GaussianBlur(gr, (k, k), sigma)
        ncc, mae = _ncc_mae(af, ar, ring)
        cand = {"ncc": ncc, "mae": mae, "scale_used": float(sigma)}
        if best is None or (np.isfinite(ncc) and ncc > best["ncc"]):
            best = cand
    assert best is not None
    best["resized"] = bool(resized)
    best["center_mae"] = float(np.mean(np.abs(
        gf[centre].astype(np.float64) - gr[centre].astype(np.float64))) / 255.0)
    return best


def is_aligned(ncc: float, mae: float) -> bool:
    if not (np.isfinite(ncc) and np.isfinite(mae)):
        return False
    return (ncc > NCC_STRICT and mae < MAE_STRICT) or (
        ncc > NCC_LOOSE and mae < MAE_LOOSE
    )


def probe_pair(
    fake_path: str,
    real_path: str,
    n_probes: int = 3,
    border_frac: float = BORDER_FRAC,
    offsets: Sequence[int] = OFFSET_SEARCH,
) -> Dict[str, Any]:
    """
    Probe one pair.  Decodes `n_probes` frames spread over the usable range
    (the range both clips actually have) at identical indices, compares them,
    and if they do not align tries a small frame offset.

    Returns a diagnostic record; `aligned` is the verdict for this pair.
    """
    mf, mr = video_meta(fake_path), video_meta(real_path)
    rec: Dict[str, Any] = {
        "fake": fake_path, "real": real_path,
        "fake_meta": mf, "real_meta": mr,
        "aligned": False, "frame_offset": 0, "error": "",
        "resolution_match": bool(mf.get("w") == mr.get("w") and mf.get("h") == mr.get("h")),
    }
    if not mf["ok"] or not mr["ok"]:
        rec["error"] = "unreadable video"
        return rec

    n = min(mf["n_frames"], mr["n_frames"])
    if n <= 0:
        n = min(true_frame_count(fake_path), true_frame_count(real_path))
        rec["fake_meta"]["n_frames"] = rec["real_meta"]["n_frames"] = n
    if n < 5:
        rec["error"] = f"too few frames (n={n})"
        return rec
    rec["n_usable"] = int(n)

    # Probe indices must lie inside BOTH clips and leave room for the offset
    # search.  Indexing past min(len) on clips of unequal length would
    # silently break temporal alignment.
    pad = max(abs(min(offsets)), abs(max(offsets))) + 1
    lo, hi = pad, max(pad + 1, n - pad - 1)
    if hi <= lo:
        idxs = [n // 2]
    else:
        idxs = [int(round(lo + (hi - lo) * (i + 1) / (n_probes + 1)))
                for i in range(n_probes)]
        idxs = sorted({min(max(i, lo), hi) for i in idxs})

    best_overall: Optional[Dict[str, Any]] = None
    for off in offsets:
        f_idx = idxs
        r_idx = [i + off for i in idxs]
        if min(r_idx) < 0 or max(r_idx) >= n:
            continue
        fframes = read_frames_at(fake_path, f_idx)
        rframes = read_frames_at(real_path, r_idx)
        per_frame: List[Dict[str, Any]] = []
        for fi, ri in zip(f_idx, r_idx):
            if fi not in fframes or ri not in rframes:
                continue
            m = compare_frames(fframes[fi], rframes[ri], border_frac)
            m["fake_index"], m["real_index"] = int(fi), int(ri)
            m["aligned"] = is_aligned(m["ncc"], m["mae"])
            per_frame.append(m)
        if not per_frame:
            continue
        frac = float(np.mean([p["aligned"] for p in per_frame]))
        summary = {
            "offset": int(off),
            "frac_aligned": frac,
            "ncc": float(np.median([p["ncc"] for p in per_frame])),
            "mae": float(np.median([p["mae"] for p in per_frame])),
            "center_mae": float(np.median([p["center_mae"] for p in per_frame])),
            "scale_used": float(np.median([p["scale_used"] for p in per_frame])),
            "resized": bool(any(p["resized"] for p in per_frame)),
            "frames": per_frame,
        }
        if best_overall is None or (
            summary["frac_aligned"], summary["ncc"]
        ) > (best_overall["frac_aligned"], best_overall["ncc"]):
            best_overall = summary
        if frac >= 1.0 - 1e-9:
            break           # perfect at this offset; stop searching

    if best_overall is None:
        rec["error"] = "no frame could be decoded at matching indices"
        return rec

    rec.update({
        "aligned": best_overall["frac_aligned"] >= 0.5,
        "frac_frames_aligned": best_overall["frac_aligned"],
        "frame_offset": best_overall["offset"],
        "ncc": best_overall["ncc"],
        "mae": best_overall["mae"],
        "center_mae": best_overall["center_mae"],
        "scale_used": best_overall["scale_used"],
        "resized": best_overall["resized"],
        "frames": best_overall["frames"],
    })
    return rec


# --------------------------------------------------------------------------
# layout discovery and pairing
# --------------------------------------------------------------------------

def _method_of(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)[:-1]
    for p in reversed(parts):
        if p in FFPP_METHODS:
            return p
        if p.lower() not in _NON_METHOD_DIRS and not p.startswith("."):
            return p
    return "unknown"


def _is_original(path: str, root: str) -> bool:
    rel = os.path.relpath(path, root).lower()
    return ("original" in rel.split(os.sep)[0]) or ("original_sequences" in rel)


def scan_videos(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in files:
            if f.lower().endswith(VIDEO_EXT):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def build_pairs_ffpp(root: str, methods: Sequence[str] = ()) -> Tuple[List[Dict], Dict]:
    """
    FaceForensics++ pairing rule.

    A manipulated clip is named <target>_<source>.mp4.  The TARGET video
    supplies every pixel outside the swapped face and every frame timing, so
    original/<target>.mp4 is the frame-aligned authentic twin.  Pairing to
    the source would be wrong, since it is a different scene entirely.
    """
    vids = scan_videos(root)
    reals: Dict[str, str] = {}
    fakes: List[Tuple[str, str, str, str]] = []   # (pair_id, target, method, path)
    for v in vids:
        stem = os.path.splitext(os.path.basename(v))[0]
        if _is_original(v, root) and _RE_FFPP_REAL.match(stem):
            reals.setdefault(stem, v)
            continue
        m = _RE_FFPP_FAKE.match(stem)
        if m and not _is_original(v, root):
            fakes.append((stem, m.group(1), _method_of(v, root), v))

    wanted = {m.strip() for m in methods if m.strip()} if methods else None
    pairs: List[Dict] = []
    missing: List[str] = []
    for pair_id, target, method, fpath in fakes:
        if wanted and method not in wanted:
            continue
        rpath = reals.get(target)
        if not rpath:
            missing.append(f"{method}/{pair_id} (no original/{target})")
            continue
        pairs.append({
            "pair_id": pair_id,
            "method": method,
            "fake": os.path.relpath(fpath, root),
            "real": os.path.relpath(rpath, root),
            "frame_offset": 0,
        })
    pairs.sort(key=lambda p: (p["method"], p["pair_id"]))
    stats = {
        "layout": "ffpp",
        "n_videos_seen": len(vids),
        "n_originals": len(reals),
        "n_fakes_seen": len(fakes),
        "n_pairs": len(pairs),
        "methods_found": sorted({p["method"] for p in pairs}),
        "unpaired_examples": missing[:20],
        "n_unpaired": len(missing),
    }
    return pairs, stats


def build_pairs_celebdf(root: str) -> Tuple[List[Dict], Dict]:
    """
    Celeb-DF-v2 pairing rule.

    Celeb-synthesis/idX_idY_SSSS.mp4 is built on the TARGET clip
    Celeb-real/idX_SSSS.mp4 with the same frames.
    """
    vids = scan_videos(root)
    reals: Dict[str, str] = {}
    fakes: List[Tuple[str, str, str]] = []
    for v in vids:
        stem = os.path.splitext(os.path.basename(v))[0]
        parent = os.path.basename(os.path.dirname(v))
        if parent in ("Celeb-real", "YouTube-real"):
            reals.setdefault(stem, v)
        elif parent == "Celeb-synthesis":
            m = _RE_CELEB_FAKE.match(stem)
            if m:
                fakes.append((stem, f"id{int(m.group(1))}_{m.group(3)}", v))
    pairs, missing = [], []
    for pair_id, target, fpath in fakes:
        rpath = reals.get(target)
        if not rpath:
            missing.append(f"{pair_id} (no {target})")
            continue
        pairs.append({
            "pair_id": pair_id, "method": "CelebDF",
            "fake": os.path.relpath(fpath, root),
            "real": os.path.relpath(rpath, root),
            "frame_offset": 0,
        })
    pairs.sort(key=lambda p: p["pair_id"])
    return pairs, {
        "layout": "celebdf", "n_videos_seen": len(vids),
        "n_originals": len(reals), "n_fakes_seen": len(fakes),
        "n_pairs": len(pairs), "methods_found": ["CelebDF"],
        "unpaired_examples": missing[:20], "n_unpaired": len(missing),
    }


def detect_layout(root: str) -> str:
    names = set()
    for dirpath, dirnames, _f in os.walk(root):
        names.update(dirnames)
        if len(names) > 400:
            break
    if "Celeb-synthesis" in names:
        return "celebdf"
    return "ffpp"


# --------------------------------------------------------------------------
# pairs.json io
# --------------------------------------------------------------------------

def save_pairs(path: str, pairs: Sequence[Dict], meta: Dict) -> str:
    return C.save_json(path, {"meta": dict(meta), "pairs": list(pairs)}, indent=1)


def load_pairs(path: str, data_root: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    """
    Load pairs.json and resolve clip paths.  `data_root` overrides the root
    recorded at write time; it should always be passed on Kaggle, because the
    dataset is mounted at a different path in every notebook.
    """
    blob = C.load_json(path)
    pairs = list(blob.get("pairs", []))
    meta = dict(blob.get("meta", {}))
    root = data_root or meta.get("root") or ""
    out = []
    n_missing = 0
    for p in pairs:
        q = dict(p)
        q["fake_path"] = C.resolve_path(q["fake"], root)
        q["real_path"] = C.resolve_path(q["real"], root)
        if not (os.path.exists(q["fake_path"]) and os.path.exists(q["real_path"])):
            n_missing += 1
        out.append(q)
    meta["n_missing_files"] = n_missing
    meta["resolved_root"] = root
    return out, meta


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def verify(
    root: str,
    methods: Sequence[str] = (),
    sample: int = 20,
    probes: int = 3,
    seed: int = 0,
    min_alignment: float = DEFAULT_MIN_ALIGNMENT,
    min_resolution: float = DEFAULT_MIN_RESOLUTION,
    border_frac: float = BORDER_FRAC,
    layout: str = "auto",
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """Run the gate.  Returns (pairs, report)."""
    t0 = time.time()
    lay = detect_layout(root) if layout == "auto" else layout

    if verbose:
        print(C.banner("PHASE 1 -- layout discovery"))
        print(f"root    : {root}")
        print(f"layout  : {lay}" + ("  (auto-detected)" if layout == "auto" else ""))

    if lay == "celebdf":
        pairs, stats = build_pairs_celebdf(root)
    else:
        pairs, stats = build_pairs_ffpp(root, methods)

    if verbose:
        print(C.banner("PHASE 2 -- pairing"))
        for k in ("n_videos_seen", "n_originals", "n_fakes_seen", "n_pairs",
                  "n_unpaired"):
            print(f"  {k:16s} {stats[k]}")
        print(f"  methods          {stats['methods_found']}")
        if stats["n_unpaired"]:
            print(f"  unpaired examples: {stats['unpaired_examples'][:5]}")

    if not pairs:
        report = {
            "verdict": "FAILED", "reason": "no pairs could be built",
            "root": root, "layout": lay, **stats,
            "min_alignment": min_alignment, "provenance": C.provenance(),
        }
        if verbose:
            print(C.banner("RESULT: PAIRING GATE FAILED -- no pairs"))
        return pairs, report

    # ---- sample and probe -------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    if sample and sample < len(pairs):
        # Stratify by method so that every manipulation is represented.
        by_method: Dict[str, List[int]] = {}
        for i, p in enumerate(pairs):
            by_method.setdefault(p["method"], []).append(i)
        take: List[int] = []
        per = max(1, sample // max(1, len(by_method)))
        for m, ids in sorted(by_method.items()):
            pick = rng.permutation(ids)[:per]
            take.extend(int(i) for i in pick)
        rest = [i for i in idx if i not in set(take)]
        if len(take) < sample and rest:
            extra = rng.permutation(rest)[: sample - len(take)]
            take.extend(int(i) for i in extra)
        idx = np.array(sorted(take[:sample]))

    if verbose:
        print(C.banner(f"PHASE 3 -- probing {len(idx)} pairs x {probes} frames"))

    results: List[Dict] = []
    for n, i in enumerate(idx):
        p = pairs[int(i)]
        fpath = os.path.join(root, p["fake"])
        rpath = os.path.join(root, p["real"])
        r = probe_pair(fpath, rpath, n_probes=probes, border_frac=border_frac)
        r["pair_id"] = p["pair_id"]
        r["method"] = p["method"]
        results.append(r)
        pairs[int(i)]["frame_offset"] = int(r.get("frame_offset", 0))
        if verbose:
            flag = "OK  " if r["aligned"] else "FAIL"
            print(f"  [{n+1:3d}/{len(idx)}] {flag} {r['method']:<16s} {r['pair_id']:<10s} "
                  f"ncc={r.get('ncc', float('nan')):.4f} mae={r.get('mae', float('nan')):.4f} "
                  f"off={r.get('frame_offset',0):+d} scale={r.get('scale_used',0):.0f} "
                  f"{'RESIZED ' if r.get('resized') else ''}{r.get('error','')}",
                  flush=True)

    # ---- aggregate --------------------------------------------------------
    n_probed = len(results)
    n_aligned = sum(1 for r in results if r["aligned"])
    n_resmatch = sum(1 for r in results if r["resolution_match"])
    alignment_rate = n_aligned / n_probed if n_probed else 0.0
    resolution_rate = n_resmatch / n_probed if n_probed else 0.0
    n_offset = sum(1 for r in results if r.get("frame_offset", 0) != 0 and r["aligned"])
    n_identical = sum(1 for r in results if r.get("center_mae", 1.0) < 0.002)

    if verbose:
        print(C.banner("PHASE 4 -- aggregate"))
        print(f"  probed pairs      {n_probed}")
        print(f"  aligned           {n_aligned}  -> alignment_rate = {alignment_rate:.3f}"
              f"   (need >= {min_alignment:.2f})")
        print(f"  resolution match  {n_resmatch}  -> resolution_rate = {resolution_rate:.3f}"
              f"   (need >= {min_resolution:.2f}; a mismatch is handled by resizing)")
        print(f"  rescued by offset {n_offset}")
        if n_identical:
            print(f"  !! {n_identical} pairs have a near-zero CENTRE difference: the "
                  f"'fake' and 'real' files may be the same video")

    passed = (alignment_rate >= min_alignment) and (n_probed > 0)

    # ---- PHASE 5: diagnostic block explaining a failed gate ---------------
    if verbose and not passed:
        print(C.banner("PHASE 5 -- diagnosis"))
        bad = [r for r in results if not r["aligned"]]
        print(f"{len(bad)} of {n_probed} probed pairs failed the criterion")
        print(f"criterion: (ncc>{NCC_STRICT} and mae<{MAE_STRICT}) or "
              f"(ncc>{NCC_LOOSE} and mae<{MAE_LOOSE})\n")
        hdr = (f"{'pair':<12}{'method':<16}{'ncc':>8}{'mae':>8}{'cmae':>8}"
               f"{'off':>5}{'scl':>5}{'rsz':>5}  note")
        print(hdr)
        print("-" * len(hdr))
        for r in bad[:40]:
            fm, rm = r.get("fake_meta", {}), r.get("real_meta", {})
            note = r.get("error", "")
            if not note:
                if not r["resolution_match"]:
                    note = f"{fm.get('w')}x{fm.get('h')} vs {rm.get('w')}x{rm.get('h')}"
                if abs(fm.get("n_frames", 0) - rm.get("n_frames", 0)) > 2:
                    note += f" nframes {fm.get('n_frames')} vs {rm.get('n_frames')}"
                if r.get("center_mae", 1.0) < 0.002:
                    note += " CENTRE IDENTICAL (same file?)"
                if r.get("ncc", 0) > NCC_LOOSE and r.get("mae", 1) >= MAE_LOOSE:
                    note += " high-ncc/high-mae -> global brightness or codec shift"
                if r.get("ncc", 0) <= NCC_LOOSE and r.get("mae", 1) < MAE_STRICT:
                    note += " low-ncc/low-mae -> near-flat border, check border_frac"
            print(f"{r['pair_id']:<12}{r['method']:<16}"
                  f"{r.get('ncc', float('nan')):>8.4f}{r.get('mae', float('nan')):>8.4f}"
                  f"{r.get('center_mae', float('nan')):>8.4f}"
                  f"{r.get('frame_offset', 0):>5d}{r.get('scale_used', 0):>5.0f}"
                  f"{'Y' if r.get('resized') else 'N':>5}  {note}")
        print("\nsuggested remedies, in order:")
        print("  1. If most failures are one METHOD, exclude it with --methods and "
              "record the exclusion with the results.")
        print("  2. If 'CENTRE IDENTICAL' appears, the mirror's manipulated folder "
              "contains copies of the originals -- the mirror is unusable.")
        print("  3. If nframes differ a lot, the mirror re-encoded at a different "
              "frame rate. Temporal alignment is then NOT recoverable by an "
              "integer offset; use another mirror.")
        print("  4. Only if the failures look like borderline codec noise (ncc>0.8, "
              "mae 0.06-0.08) should --min-alignment be relaxed, and the "
              "relaxed value must be reported as a deviation.")

    verdict = "PASSED" if passed else "FAILED"
    report = {
        "verdict": verdict,
        "root": root, "layout": lay,
        "alignment_rate": alignment_rate,
        "resolution_rate": resolution_rate,
        "n_probed": n_probed, "n_aligned": n_aligned,
        "n_rescued_by_offset": n_offset,
        "n_centre_identical": n_identical,
        "min_alignment": min_alignment,
        "min_resolution": min_resolution,
        "criterion": {
            "ncc_strict": NCC_STRICT, "mae_strict": MAE_STRICT,
            "ncc_loose": NCC_LOOSE, "mae_loose": MAE_LOOSE,
            "border_frac": border_frac, "scales": list(SCALES),
            "offset_search": list(OFFSET_SEARCH),
        },
        "per_pair": results,
        "elapsed_sec": time.time() - t0,
        **stats,
        "provenance": C.provenance(),
    }

    if verbose:
        if passed:
            print(C.banner("RESULT: PAIRING VERIFIED."))
            print(f"  alignment_rate  {alignment_rate:.3f}")
            print(f"  resolution_rate {resolution_rate:.3f}")
            print(f"  {len(pairs)} pairs written.")
        else:
            print(C.banner("RESULT: PAIRING GATE FAILED -- see PHASE 5 above"))
    return pairs, report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ccaudit.m1_verify",
        description="Module 1: verify that every fake clip has a frame-aligned "
                    "authentic twin, and emit pairs.json.",
    )
    ap.add_argument("--root", required=True, help="dataset root (contains original/)")
    ap.add_argument("--methods", default="", help="comma-separated subset, '' = all")
    ap.add_argument("--layout", default="auto", choices=["auto", "ffpp", "celebdf"])
    ap.add_argument("--sample", type=int, default=20, help="pairs to probe")
    ap.add_argument("--probes", type=int, default=3, help="frames per probed pair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-alignment", type=float, default=DEFAULT_MIN_ALIGNMENT)
    ap.add_argument("--min-resolution", type=float, default=DEFAULT_MIN_RESOLUTION)
    ap.add_argument("--border-frac", type=float, default=BORDER_FRAC)
    ap.add_argument("--json", default="", help="write the full report here")
    ap.add_argument("--emit-pairs", default="", help="write pairs.json here")
    ap.add_argument("--fail-hard", action="store_true",
                    help="exit(2) when the gate fails (notebooks use this)")
    a = ap.parse_args(argv)

    methods = [m for m in a.methods.split(",") if m.strip()]
    pairs, report = verify(
        root=a.root, methods=methods, sample=a.sample, probes=a.probes,
        seed=a.seed, min_alignment=a.min_alignment,
        min_resolution=a.min_resolution, border_frac=a.border_frac,
        layout=a.layout,
    )
    if a.json:
        C.save_json(a.json, report, indent=1)
        print(f"report -> {a.json}")
    if a.emit_pairs:
        meta = {k: report[k] for k in (
            "verdict", "root", "layout", "alignment_rate", "resolution_rate",
            "n_probed", "n_aligned", "min_alignment", "criterion",
            "methods_found", "n_pairs",
        ) if k in report}
        meta["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["code_hash"] = C.code_hash()
        save_pairs(a.emit_pairs, pairs, meta)
        print(f"pairs  -> {a.emit_pairs}  ({len(pairs)} pairs)")
    if report["verdict"] != "PASSED":
        return 2 if a.fail_hard else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
