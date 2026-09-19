"""
ccaudit.m2_parse -- Module 2, the PARSER.

Turns pairs.json into `parsed/index.json` plus, per sample, three files:

    <method>/<pair_id>/f<frame>_real.jpg    authentic crop
    <method>/<pair_id>/f<frame>_fake.jpg    manipulated crop, SAME crop box
    <method>/<pair_id>/f<frame>_lab.png     region label map, SAME crop box

Invariants enforced here
------------------------
1. TEMPORAL ALIGNMENT.  The fake frame index i and the real frame index
   i + frame_offset are decoded in a single sequential pass each (never by
   seeking), and every index is inside min(len_fake, len_real).  Sampling
   indices that exist in one clip but not the other would silently break the
   paired-frame assumption; this is impossible by construction because the
   index list is built from the usable range and checked against both clips
   before anything is written.

2. ONE JPEG PASS.  The decoded frame is cropped and then encoded exactly once
   at --jpeg-q.  Real and fake follow an identical path, so the compression
   history of the two conditions is the same.  The label map is PNG
   (lossless; a JPEG label map is meaningless).

3. SHARED CROP BOX.  The crop box is computed from the landmarks of the REAL
   frame and applied unchanged to the fake frame and to the label map, so the
   spatial support is identical across conditions.  Boxes that run off the
   frame are padded by replication rather than clamped, because clamping
   would change the aspect ratio and therefore the geometry.

4. NO FALLBACK PARSER.  If mediapipe is unavailable or the landmarker cannot
   be built, this module fails.  A purely geometric parser produces boxes
   that do not sit on the face, and a silent fallback would let such label
   maps propagate downstream unnoticed.

Quality fields written per sample
---------------------------------
parser_iou       IoU between the face oval parsed from the real frame and the
                 one parsed from the fake frame.  Low -> the two frames
                 disagree about where the face is, so a shared crop/label map
                 is not trustworthy.  Gated by --min-iou.
manip_coverage   fraction of the changed pixels (|fake-real|) that fall inside
                 a named region.  Low -> the manipulation is somewhere the
                 vocabulary cannot describe.  Gated by --min-coverage.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C
from . import regions as R
from .m1_verify import load_pairs, read_frames_at, video_meta

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER_NAME = "face_landmarker.task"


# --------------------------------------------------------------------------
# mediapipe bring-up
# --------------------------------------------------------------------------

def _patch_tf_doc_controls() -> bool:
    """
    Kaggle's image ships a TensorFlow whose `tensorflow.tools.docs` package is
    absent, while mediapipe's `__init__` imports `doc_controls` from it.  The
    result is `ImportError: cannot import name 'doc_controls'`, which is not a
    mediapipe bug.  A stub module whose decorators are the identity is
    registered in sys.modules beforehand; this is what doc_controls does at
    runtime anyway (it only annotates for the documentation generator).
    """
    import importlib
    import types

    try:
        importlib.import_module("tensorflow.tools.docs.doc_controls")
        return False
    except Exception:
        pass

    def _identity(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(fn):
            return fn
        return deco

    mod = types.ModuleType("tensorflow.tools.docs.doc_controls")
    for name in ("do_not_generate_docs", "do_not_doc_inheritable",
                 "do_not_doc_in_subclasses", "for_subclass_implementers",
                 "doc_private", "doc_in_current_and_subclasses",
                 "inheritable_header", "header", "decorate_all_class_attributes"):
        setattr(mod, name, _identity)

    try:
        tf = importlib.import_module("tensorflow")
    except Exception:
        tf = types.ModuleType("tensorflow")
        sys.modules["tensorflow"] = tf
    tools = sys.modules.get("tensorflow.tools") or types.ModuleType("tensorflow.tools")
    docs = sys.modules.get("tensorflow.tools.docs") or types.ModuleType("tensorflow.tools.docs")
    sys.modules["tensorflow.tools"] = tools
    sys.modules["tensorflow.tools.docs"] = docs
    sys.modules["tensorflow.tools.docs.doc_controls"] = mod
    setattr(tools, "docs", docs)
    setattr(docs, "doc_controls", mod)
    try:
        setattr(tf, "tools", tools)
    except Exception:
        pass
    return True


def _import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError as exc:
        if "doc_controls" not in str(exc) and "tensorflow" not in str(exc):
            raise
        _patch_tf_doc_controls()
        import mediapipe as mp      # retry with the stub in place
        print("[m2] applied tensorflow.tools.docs.doc_controls shim", flush=True)
        return mp


def locate_landmarker(explicit: str = "") -> str:
    """
    Find face_landmarker.task, in order:
      1. --mp-model
      2. anywhere under /kaggle/input (upload it as a small dataset if offline)
      3. the scratch cache
      4. download it into scratch (needs Internet access)
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--mp-model {explicit} does not exist")
        return explicit
    for root in ("/kaggle/input",):
        if os.path.isdir(root):
            for dirpath, dirnames, files in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                if LANDMARKER_NAME in files:
                    return os.path.join(dirpath, LANDMARKER_NAME)
    cache = os.path.join(C.scratch_dir("mediapipe"), LANDMARKER_NAME)
    if os.path.exists(cache) and os.path.getsize(cache) > 1_000_000:
        return cache
    print(f"[m2] downloading {LANDMARKER_NAME} -> {cache}", flush=True)
    tmp = cache + ".part"
    urllib.request.urlretrieve(LANDMARKER_URL, tmp)
    os.replace(tmp, cache)
    if os.path.getsize(cache) < 1_000_000:
        raise RuntimeError("downloaded face_landmarker.task looks truncated")
    return cache


class FaceParser:
    """Thin wrapper over the mediapipe Tasks API FaceLandmarker."""

    def __init__(self, model_path: str, min_conf: float = 0.4):
        mp = _import_mediapipe()
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        with open(model_path, "rb") as fh:
            buf = fh.read()
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=buf),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=float(min_conf),
            min_face_presence_confidence=float(min_conf),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)
        print("[m2] mediapipe backend ready (tasks API)", flush=True)

    def landmarks(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """(468+,2) pixel coordinates of the single detected face, or None."""
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect(image)
        if not getattr(res, "face_landmarks", None):
            return None
        h, w = bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]],
                       dtype=np.float32)
        return pts if pts.shape[0] >= 468 else None

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def crop_box_from_landmarks(
    pts: np.ndarray, shape: Sequence[int], margin: float
) -> Tuple[int, int, int, int]:
    """
    Square box around the face, expanded by `margin` (fraction of the larger
    face dimension).  May extend outside the frame; the caller pads.
    """
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
    side = max(side, 16.0)
    half = side / 2.0
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def crop_pad(img: np.ndarray, box: Tuple[int, int, int, int],
             nearest: bool = False) -> np.ndarray:
    """Crop `box` from `img`, replicating the border where the box runs out."""
    import cv2

    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    if pl or pt or pr or pb:
        border = cv2.BORDER_CONSTANT if nearest else cv2.BORDER_REPLICATE
        value = int(R.BACKGROUND) if nearest else 0
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, border, value=value)
        x0, y0, x1, y1 = x0 + pl, y0 + pt, x1 + pl, y1 + pt
    return img[y0:y1, x0:x1]


def resize_to(img: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    import cv2

    if size <= 0 or (img.shape[0] == size and img.shape[1] == size):
        return img
    interp = cv2.INTER_NEAREST if nearest else (
        cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    )
    return cv2.resize(img, (size, size), interpolation=interp)


def face_oval_mask(pts: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    import cv2

    m = np.zeros((int(shape[0]), int(shape[1])), np.uint8)
    cv2.fillPoly(m, [pts[R.FACE_OVAL].astype(np.int32).reshape(-1, 1, 2)], 1)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    return inter / union if union > 0 else 0.0


def manipulation_mask(real: np.ndarray, fake: np.ndarray,
                      sigma: float = 2.0,
                      bg_mask: Optional[np.ndarray] = None,
                      min_bg_frac: float = 0.05
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Locate the changed pixels.  Returns (binary mask, continuous diff map).

    This is the canonical definition, imported by m10_localization and by the
    control detectors' calibration, so that "where the manipulation is" means
    the same thing everywhere in the codebase.

    Threshold: max(8, mu + 2*sd).  The statistics are taken over BACKGROUND
    pixels when a background mask is supplied and covers at least
    `min_bg_frac` of the image.  Background is where nothing should have
    changed, so it estimates the compression noise floor rather than being
    inflated by the manipulation itself.  On a tight face crop there may be
    almost no background, in which case the statistics fall back to the
    whole image.
    """
    import cv2

    d = cv2.absdiff(real, fake)
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    d = cv2.GaussianBlur(d, (0, 0), sigma) if sigma > 0 else d
    ref = d
    if bg_mask is not None:
        bg = bg_mask.astype(bool)
        if bg.sum() >= min_bg_frac * bg.size:
            ref = d[bg]
    thr = max(8.0, float(ref.mean() + 2.0 * ref.std()))
    return (d >= thr).astype(np.uint8), d


# --------------------------------------------------------------------------
# per-clip work
# --------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _worker_init(model_path: str, min_conf: float, vocab: str) -> None:
    R.set_vocab(vocab)
    _WORKER["parser"] = FaceParser(model_path, min_conf) if vocab == "face8" else None
    _WORKER["vocab"] = vocab


def frame_indices(n_usable: int, k: int, edge_frac: float = 0.1) -> List[int]:
    """
    k indices spread over the middle of the usable range.  The usable range is
    the number of frames BOTH clips have; nothing outside it is ever returned.
    """
    if n_usable <= 0 or k <= 0:
        return []
    lo = int(round(edge_frac * n_usable))
    hi = int(round((1.0 - edge_frac) * n_usable)) - 1
    if hi <= lo:
        lo, hi = 0, max(0, n_usable - 1)
    if k == 1:
        return [int((lo + hi) // 2)]
    step = (hi - lo) / float(k - 1)
    idx = sorted({int(round(lo + i * step)) for i in range(k)})
    return [i for i in idx if 0 <= i < n_usable]


def parse_clip(
    pair: Dict[str, Any],
    out_dir: str,
    frames_per_clip: int,
    crop_size: int,
    crop_margin: float,
    jpeg_q: int,
    min_iou: float,
    min_coverage: float,
    min_region_px: int,
    vocab: str,
) -> Dict[str, Any]:
    """
    Parse one (fake, real) clip pair.  Returns
    {"records": [...], "drops": {reason: n}, "pair_id": ..., "error": str}.
    Runs inside a worker process; all image IO happens here.
    """
    parser: Optional[FaceParser] = _WORKER.get("parser")
    out: Dict[str, Any] = {"pair_id": pair["pair_id"], "records": [],
                           "drops": {}, "error": ""}

    def drop(reason: str, n: int = 1) -> None:
        out["drops"][reason] = out["drops"].get(reason, 0) + n

    fpath, rpath = pair["fake_path"], pair["real_path"]
    if not (os.path.exists(fpath) and os.path.exists(rpath)):
        out["error"] = "missing file"
        drop("missing_file")
        return out

    mf, mr = video_meta(fpath), video_meta(rpath)
    if not (mf["ok"] and mr["ok"]):
        out["error"] = "unreadable"
        drop("unreadable")
        return out

    off = int(pair.get("frame_offset", 0))
    # Usable range: indices i such that i is in the fake and i+off is in the
    # real clip.
    n_usable = min(mf["n_frames"], mr["n_frames"] - off if off > 0 else mr["n_frames"])
    if off < 0:
        n_usable = min(mf["n_frames"] + off, mr["n_frames"])
    if n_usable < 5:
        out["error"] = f"too few usable frames ({n_usable})"
        drop("too_short")
        return out

    f_idx = frame_indices(n_usable, frames_per_clip)
    r_idx = [i + off for i in f_idx]
    if not f_idx or min(r_idx) < 0 or max(r_idx) >= mr["n_frames"] \
            or max(f_idx) >= mf["n_frames"]:
        out["error"] = "index out of range after offset"
        drop("index_range")
        return out

    fframes = read_frames_at(fpath, f_idx)
    rframes = read_frames_at(rpath, r_idx)

    method_dir = os.path.join(out_dir, C.safe_name(pair["method"]),
                              C.safe_name(pair["pair_id"]))

    for fi, ri in zip(f_idx, r_idx):
        if fi not in fframes or ri not in rframes:
            drop("decode_failed")
            continue
        fake_full, real_full = fframes[fi], rframes[ri]
        if fake_full.shape[:2] != real_full.shape[:2]:
            import cv2
            th, tw = real_full.shape[:2]
            fake_full = cv2.resize(
                fake_full, (tw, th),
                interpolation=cv2.INTER_AREA if fake_full.shape[0] > th
                else cv2.INTER_LINEAR)

        if vocab == "grid9":
            box = (0, 0, real_full.shape[1], real_full.shape[0])
            real_c = resize_to(real_full, crop_size)
            fake_c = resize_to(fake_full, crop_size)
            lab_c = R.build_grid9_labels(real_c.shape[:2])
            p_iou = 1.0
        else:
            assert parser is not None, "face parser not initialised"
            pts_r = parser.landmarks(real_full)
            if pts_r is None:
                drop("no_face_real")
                continue
            pts_f = parser.landmarks(fake_full)
            if pts_f is None:
                drop("no_face_fake")
                continue

            p_iou = iou(face_oval_mask(pts_r, real_full.shape[:2]).astype(bool),
                        face_oval_mask(pts_f, fake_full.shape[:2]).astype(bool))
            if p_iou < min_iou:
                drop("low_parser_iou")
                continue

            try:
                lab_full = R.build_face8_labels(pts_r, real_full.shape[:2])
            except Exception as exc:                       # degenerate geometry
                drop("label_error")
                out["error"] = str(exc)[:120]
                continue

            if crop_size > 0:
                box = crop_box_from_landmarks(pts_r, real_full.shape[:2], crop_margin)
                real_c = resize_to(crop_pad(real_full, box), crop_size)
                fake_c = resize_to(crop_pad(fake_full, box), crop_size)
                lab_c = resize_to(crop_pad(lab_full, box, nearest=True),
                                  crop_size, nearest=True)
            else:
                box = (0, 0, real_full.shape[1], real_full.shape[0])
                real_c, fake_c, lab_c = real_full, fake_full, lab_full

        # Quality gate: the vocabulary must cover where the pixels changed.
        mgt, _d = manipulation_mask(real_c, fake_c,
                                    bg_mask=(lab_c == R.BACKGROUND))
        n_changed = int(mgt.sum())
        if n_changed < 16:
            drop("no_manipulation_detected")
            continue
        named = (lab_c != R.BACKGROUND)
        coverage = float(np.count_nonzero(mgt.astype(bool) & named)) / float(n_changed)
        if coverage < min_coverage:
            drop("low_manip_coverage")
            continue

        areas = R.region_areas(lab_c, vocab)
        present = [r for r in R.get_vocab(vocab) if areas[r] >= min_region_px]
        if len(present) < 2:
            drop("too_few_regions")
            continue

        sid = C.stable_id(pair["method"], pair["pair_id"], fi, "v1")
        stem = os.path.join(method_dir, f"f{fi:06d}")
        C.imwrite_jpeg(stem + "_real.jpg", real_c, jpeg_q)
        C.imwrite_jpeg(stem + "_fake.jpg", fake_c, jpeg_q)
        C.imwrite_png(stem + "_lab.png", lab_c)

        out["records"].append({
            "sample_id": sid,
            "pair_id": pair["pair_id"],
            "method": pair["method"],
            "frame": int(fi),
            "real_frame": int(ri),
            "frame_offset": off,
            "real": stem + "_real.jpg",
            "fake": stem + "_fake.jpg",
            "lab": stem + "_lab.png",
            "parser_iou": round(float(p_iou), 4),
            "manip_coverage": round(float(coverage), 4),
            "changed_frac": round(n_changed / float(mgt.size), 5),
            "backend": "grid" if vocab == "grid9" else "mediapipe",
            "vocab": vocab,
            "crop_box": [int(v) for v in box],
            "crop_size": int(crop_size),
            "jpeg_q": int(jpeg_q),
            "areas": {k: int(v) for k, v in areas.items()},
            "present_regions": present,
            "split": C.split_of(pair["pair_id"]),
        })
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(
    pairs_path: str,
    out_dir: str,
    data_root: str = "",
    frames_per_clip: int = 2,
    crop_size: int = 384,
    crop_margin: float = 0.35,
    jpeg_q: int = 90,
    min_iou: float = 0.70,
    min_coverage: float = 0.50,
    min_region_px: int = 64,
    methods: Sequence[str] = (),
    max_pairs: int = 0,
    seed: int = 0,
    shard_spec: str = "",
    workers: int = 4,
    resume: bool = True,
    resume_from: Sequence[str] = (),
    save_every: int = 50,
    time_budget_min: float = 0.0,
    mp_model: str = "",
    min_conf: float = 0.4,
    vocab: str = "face8",
) -> Dict[str, Any]:
    """
    Parse every pair in `pairs_path` into `out_dir`.  `save_every` <= 0
    disables intermediate checkpoints of index.json; the index is always
    written at the end.
    """
    import concurrent.futures as cf
    import multiprocessing as mp_lib

    R.set_vocab(vocab)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.json")
    stats_path = os.path.join(out_dir, "parse_stats.json")
    budget = C.Budget(time_budget_min, label="m2")

    pairs, pmeta = load_pairs(pairs_path, data_root or None)
    if pmeta.get("verdict") not in (None, "", "PASSED"):
        print(f"[m2] WARNING: pairs.json carries verdict "
              f"{pmeta.get('verdict')!r} -- Module 1 did not pass.", flush=True)
    if pmeta.get("n_missing_files"):
        print(f"[m2] WARNING: {pmeta['n_missing_files']} pairs point at files "
              f"that do not exist. Pass --data-root <mounted dataset>.", flush=True)

    if methods:
        want = {m.strip() for m in methods if m.strip()}
        pairs = [p for p in pairs if p["method"] in want]
    pairs.sort(key=lambda p: (p["method"], p["pair_id"]))
    if max_pairs and max_pairs < len(pairs):
        rng = np.random.default_rng(seed)
        # Deterministic subset, stratified by method.
        by: Dict[str, List[Dict]] = {}
        for p in pairs:
            by.setdefault(p["method"], []).append(p)
        per = max(1, max_pairs // max(1, len(by)))
        picked: List[Dict] = []
        for m in sorted(by):
            arr = by[m]
            take = rng.permutation(len(arr))[:per]
            picked.extend(arr[int(i)] for i in take)
        if len(picked) < max_pairs:
            have = {id(p) for p in picked}
            picked.extend([p for p in pairs if id(p) not in have][: max_pairs - len(picked)])
        pairs = sorted(picked, key=lambda p: (p["method"], p["pair_id"]))[:max_pairs]

    pairs = C.shard(pairs, shard_spec, key=lambda p: f"{p['method']}/{p['pair_id']}")

    # ---- resume ----------------------------------------------------------
    done_pairs: set = set()
    records: List[Dict[str, Any]] = []
    drops: Dict[str, int] = {}
    sources = list(resume_from) + ([index_path] if resume else [])
    for src in sources:
        if not src or not os.path.exists(src):
            continue
        try:
            recs, _m = C.load_index(src)
        except Exception as exc:
            print(f"[m2] could not read {src}: {exc}")
            continue
        same_dir = os.path.dirname(os.path.abspath(src)) == os.path.abspath(out_dir)
        for r in recs:
            key = f"{r['method']}/{r['pair_id']}"
            if not same_dir:
                # The crops live in a read-only input; load_index has already
                # resolved their paths to absolute ones, which are kept.
                pass
            done_pairs.add(key)
            # Private keys added by load_index (e.g. `_index_dir`) are not
            # part of the on-disk record schema and are stripped.
            records.append({k: v for k, v in r.items() if not str(k).startswith("_")})
        print(f"[m2] resume: {len(recs)} records from {src}", flush=True)
    if done_pairs:
        before = len(pairs)
        pairs = [p for p in pairs if f"{p['method']}/{p['pair_id']}" not in done_pairs]
        print(f"[m2] resume: skipping {before - len(pairs)} finished clips, "
              f"{len(pairs)} to go", flush=True)

    meta = {
        "vocab": vocab, "regions": R.get_vocab(vocab),
        "frames_per_clip": frames_per_clip, "crop_size": crop_size,
        "crop_margin": crop_margin, "jpeg_q": jpeg_q, "min_iou": min_iou,
        "min_coverage": min_coverage, "min_region_px": min_region_px,
        "backend": "grid" if vocab == "grid9" else "mediapipe",
        "pairs_json": os.path.abspath(pairs_path),
        "data_root": data_root, "shard": shard_spec or "0/1",
        "code_hash": C.code_hash(),
    }

    if not pairs:
        print("[m2] nothing to do (all clips already parsed)")
        C.save_index(index_path, records, meta)
        return {"n_records": len(records), "n_pairs_done": len(done_pairs),
                "drops": drops, "stopped_early": False}

    model_path = "" if vocab == "grid9" else locate_landmarker(mp_model)
    if model_path:
        print(f"[m2] landmarker: {model_path}", flush=True)
        meta["mp_model"] = model_path

    print(f"[m2] parsing {len(pairs)} clips with {workers} worker(s), "
          f"{frames_per_clip} frame(s)/clip, crop={crop_size}, q={jpeg_q}",
          flush=True)

    t0 = time.time()
    n_done = 0
    stopped_early = False
    errors: List[str] = []

    def absorb(res: Dict[str, Any]) -> None:
        nonlocal n_done
        records.extend(res["records"])
        for k, v in res["drops"].items():
            drops[k] = drops.get(k, 0) + v
        if res.get("error"):
            errors.append(f"{res['pair_id']}: {res['error']}")
        n_done += 1

    kw = dict(out_dir=out_dir, frames_per_clip=frames_per_clip,
              crop_size=crop_size, crop_margin=crop_margin, jpeg_q=jpeg_q,
              min_iou=min_iou, min_coverage=min_coverage,
              min_region_px=min_region_px, vocab=vocab)

    if workers <= 1:
        _worker_init(model_path, min_conf, vocab)
        for p in pairs:
            if budget.expired:
                stopped_early = True
                break
            absorb(parse_clip(p, **kw))
            if save_every > 0 and n_done % save_every == 0:
                C.save_index(index_path, records, meta)
                print(f"[m2] {n_done}/{len(pairs)} clips, {len(records)} samples, "
                      f"{budget.report()}", flush=True)
    else:
        ctx = mp_lib.get_context("spawn")
        with cf.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
            initializer=_worker_init, initargs=(model_path, min_conf, vocab),
        ) as ex:
            pending = {}
            it = iter(pairs)
            # Keep the pool fed but bounded, so that cancelling on budget is
            # cheap.
            for _ in range(min(workers * 3, len(pairs))):
                try:
                    p = next(it)
                except StopIteration:
                    break
                pending[ex.submit(parse_clip, p, **kw)] = p
            while pending:
                done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    p = pending.pop(fut)
                    try:
                        absorb(fut.result())
                    except Exception as exc:
                        errors.append(f"{p['pair_id']}: {type(exc).__name__}: {exc}")
                        drops["worker_exception"] = drops.get("worker_exception", 0) + 1
                        n_done += 1
                    if save_every > 0 and n_done % save_every == 0:
                        C.save_index(index_path, records, meta)
                        print(f"[m2] {n_done}/{len(pairs)} clips, "
                              f"{len(records)} samples, {budget.report()}",
                              flush=True)
                    if budget.expired:
                        stopped_early = True
                        continue
                    try:
                        q = next(it)
                    except StopIteration:
                        continue
                    pending[ex.submit(parse_clip, q, **kw)] = q
                if stopped_early and not pending:
                    break

    # De-duplicate by sample_id (a resumed run can see the same sample twice).
    uniq: Dict[str, Dict] = {}
    for r in records:
        uniq[r["sample_id"]] = r
    records = sorted(uniq.values(), key=lambda r: (r["method"], r["pair_id"], r["frame"]))

    meta["stopped_early"] = stopped_early
    C.save_index(index_path, records, meta)

    n_clips = len({r["pair_id"] + "/" + r["method"] for r in records})
    stats = {
        "n_records": len(records),
        "n_clips_with_output": n_clips,
        "n_clips_attempted_this_run": n_done,
        "n_clips_resumed": len(done_pairs),
        "drops": drops,
        "n_dropped_total": int(sum(drops.values())),
        "errors_sample": errors[:40],
        "n_errors": len(errors),
        "elapsed_min": (time.time() - t0) / 60.0,
        "stopped_early": stopped_early,
        "parser_iou_mean": float(np.mean([r["parser_iou"] for r in records]))
        if records else float("nan"),
        "manip_coverage_mean": float(np.mean([r["manip_coverage"] for r in records]))
        if records else float("nan"),
        "split_counts": {
            s: sum(1 for r in records if r.get("split") == s) for s in ("dev", "test")
        },
        "method_counts": {
            m: sum(1 for r in records if r["method"] == m)
            for m in sorted({r["method"] for r in records})
        },
        "meta": meta,
    }
    C.save_json(stats_path, stats, indent=1)
    C.save_json(os.path.join(out_dir, "provenance.json"), C.provenance(), indent=1)

    print(C.banner("Module 2 summary"))
    print(f"  samples written        {len(records)}")
    print(f"  clips with output      {n_clips}")
    print(f"  parser IoU mean        {stats['parser_iou_mean']:.3f}")
    print(f"  manip coverage mean    {stats['manip_coverage_mean']:.3f}")
    print(f"  dropped                {stats['n_dropped_total']}  {drops}")
    print(f"  split                  {stats['split_counts']}")
    print(f"  methods                {stats['method_counts']}")
    print(f"  elapsed                {stats['elapsed_min']:.1f} min")
    if stopped_early:
        print("  STOPPED EARLY on time budget -- index checkpointed; re-run with "
              "--resume-from <this index.json> to continue")
    if errors:
        print(f"  first errors: {errors[:3]}")
    return stats


# --------------------------------------------------------------------------
# overlay figure (visual check of the parse)
# --------------------------------------------------------------------------

def overlay_figure(index_path: str, out_png: str, n: int = 4, seed: int = 0) -> str:
    """
    Build a contact sheet: real crop | fake crop | label map colourised |
    label map over the real crop.  If the regions do not sit on the face, the
    parse is wrong and nothing downstream is meaningful.
    """
    import cv2

    recs, meta = C.load_index(index_path)
    if not recs:
        raise RuntimeError("empty index")
    vocab = meta.get("vocab", "face8")
    R.set_vocab(vocab)
    rng = np.random.default_rng(seed)
    pick = [recs[int(i)] for i in rng.permutation(len(recs))[:n]]

    palette = np.array([
        [230, 80, 80], [80, 120, 240], [80, 200, 120], [200, 80, 220],
        [230, 200, 60], [60, 200, 230], [150, 110, 60], [240, 140, 40],
        [120, 120, 120],
    ], np.uint8)

    rows = []
    for r in pick:
        real = C.imread(r["real"])
        fake = C.imread(r["fake"])
        lab = C.imread(r["lab"], 0)
        col = np.zeros((*lab.shape, 3), np.uint8)
        for i in range(R.num_regions(vocab)):
            col[lab == i] = palette[i % len(palette)]
        blend = cv2.addWeighted(real, 0.6, col, 0.4, 0)
        h = real.shape[0]
        tile = np.concatenate([real, fake, col, blend], axis=1)
        cv2.putText(tile, f"{r['method']} {r['pair_id']} f{r['frame']} "
                          f"iou={r['parser_iou']:.2f} cov={r['manip_coverage']:.2f}",
                    (6, max(14, h // 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, h / 900), (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(tile)
    sheet = np.concatenate(rows, axis=0)
    legend = " | ".join(f"{i}:{name}" for i, name in enumerate(R.get_vocab(vocab)))
    print("label colours -> " + legend)
    C.imwrite_png(out_png, sheet)
    return out_png


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ccaudit.m2_parse",
        description="Module 2: parse paired clips into face crops + region "
                    "label maps (mediapipe Tasks API; no fallback parser).")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True, help="parsed/ output directory")
    ap.add_argument("--data-root", default="", help="where the videos are mounted now")
    ap.add_argument("--frames-per-clip", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=384, help="0 = full frames")
    ap.add_argument("--crop-margin", type=float, default=0.35)
    ap.add_argument("--jpeg-q", type=int, default=90)
    ap.add_argument("--min-iou", type=float, default=0.70)
    ap.add_argument("--min-coverage", type=float, default=0.50)
    ap.add_argument("--min-region-px", type=int, default=64)
    ap.add_argument("--methods", default="")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--resume-from", default="", help="comma-separated index.json paths")
    ap.add_argument("--save-every", type=int, default=50,
                    help="checkpoint index.json every N clips; 0 = only at the end")
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--mp-model", default="")
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--vocab", default="face8", choices=["face8", "grid9"])
    ap.add_argument("--overlay", default="", help="also write an overlay PNG here")
    a = ap.parse_args(argv)

    stats = run(
        pairs_path=a.pairs, out_dir=a.out, data_root=a.data_root,
        frames_per_clip=a.frames_per_clip, crop_size=a.crop_size,
        crop_margin=a.crop_margin, jpeg_q=a.jpeg_q, min_iou=a.min_iou,
        min_coverage=a.min_coverage, min_region_px=a.min_region_px,
        methods=[m for m in a.methods.split(",") if m.strip()],
        max_pairs=a.max_pairs, seed=a.seed, shard_spec=a.shard,
        workers=a.workers, resume=not a.no_resume,
        resume_from=[p for p in a.resume_from.split(",") if p.strip()],
        save_every=a.save_every, time_budget_min=a.time_budget_min,
        mp_model=a.mp_model, min_conf=a.min_conf, vocab=a.vocab,
    )
    if a.overlay and stats["n_records"]:
        overlay_figure(os.path.join(a.out, "index.json"), a.overlay)
        print(f"overlay -> {a.overlay}")
    return 0 if stats["n_records"] else 1


if __name__ == "__main__":
    sys.exit(main())
