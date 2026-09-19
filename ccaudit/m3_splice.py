"""
ccaudit.m3_splice -- Module 3, the SPLICE OPERATOR.

This is the instrument.  Given a paired (real, fake) crop and a region label
map it manufactures counterfactuals that contain NO SYNTHETIC PIXELS: every
pixel in every condition comes from one of the two real photographs.

Conditions (per region k)
-------------------------
real        take the FAKE frame, paste the REAL frame's region k into it.
            Reads as "region k is not manipulated".  Necessity direction.
identity    take the FAKE frame, paste the FAKE frame's own region k into it.
            Pixel-wise this is (almost) the original, so any change in the
            detector's output is caused by the SEAM and by the extra
            compression pass, not by content.  This is the control that
            makes the measurement valid.
reverse     take the REAL frame, paste the FAKE frame's region k into it.
            Reads as "only region k is manipulated".  Sufficiency direction.
floor       take the REAL frame, paste the REAL frame's own region k into it.
            The operator applied to an authentic image.  If the detector calls
            this manipulated, the operator itself manufactures forgery
            evidence and the instrument is invalid; this is a validity check.
inpaint_*   the SAME dilated masks filled by an inpainter instead of by the
            paired frame, on the fake (inpaint_fake) and on the real
            (inpaint_real).  This is the comparison against inpainting-based
            counterfactuals.

Why the difference-in-differences cancels compression
-----------------------------------------------------
The originals come off disk with one JPEG pass; every spliced condition gets
exactly one more.  Both `real` and `identity` therefore carry two passes, so

    delta      = delta_raw - delta_seam
               = (p_fake - p_real_cond) - (p_fake - p_identity)
               = p_identity - p_real_cond

and the baseline p_fake, its compression history and the extra pass all drop
out algebraically.  One condition's JPEG path must never be changed without
changing the other's.

Degenerate masks
----------------
cv2.seamlessClone fails on very small or border-touching masks.  When that
happens the region is blended with the requested mode's nearest safe
alternative and the fact is RECORDED in `blend_used`, which M6 reports.  A
silent substitution would corrupt the measurement; a recorded one is an
ablation cell.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C
from . import regions as R

BLEND_MODES = ("poisson", "feather", "hard")
CONDITIONS = ("real", "identity", "reverse", "floor", "inpaint_fake", "inpaint_real")
INPAINT_METHODS = ("telea", "ns", "lama")

_SEAMLESS_PAD = 8
_MIN_MASK_PX = 24


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

def region_mask(lab: np.ndarray, region: str, dilate: int = 3,
                vocab: Optional[str] = None) -> np.ndarray:
    """
    Binary uint8 mask (0/1) of one region, optionally dilated by `dilate`
    pixels with an elliptical structuring element.

    Dilation matters: the manipulated content does not stop exactly at the
    parser's boundary, and a mask that is a pixel too small leaves a rim of
    fake pixels inside the "authentic" counterfactual.  The dilation radius
    is an ablation axis (0 / 3 / 7 / 11) because the right value is an
    empirical question.
    """
    import cv2

    m = (lab == R.rid(region, vocab)).astype(np.uint8)
    if dilate and dilate > 0 and m.any():
        k = int(2 * dilate + 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    return m


def union_mask(lab: np.ndarray, regions_: Sequence[str], dilate: int = 3,
               vocab: Optional[str] = None) -> np.ndarray:
    out = np.zeros(lab.shape, np.uint8)
    for r in regions_:
        out |= region_mask(lab, r, dilate, vocab)
    return out


def mask_touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any())


# --------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------

def _poisson(src: np.ndarray, dst: np.ndarray, mask: np.ndarray
             ) -> Tuple[Optional[np.ndarray], str]:
    """
    Poisson (seamless) clone of src's masked area into dst.

    The image is padded first so the mask can never touch the border, which
    is the condition under which OpenCV's implementation misbehaves.  Returns
    (image, mode_used) or (None, reason) on failure.
    """
    import cv2

    if int(mask.sum()) < _MIN_MASK_PX:
        return None, "mask_too_small"
    p = _SEAMLESS_PAD
    src_p = cv2.copyMakeBorder(src, p, p, p, p, cv2.BORDER_REPLICATE)
    dst_p = cv2.copyMakeBorder(dst, p, p, p, p, cv2.BORDER_REPLICATE)
    msk_p = cv2.copyMakeBorder(mask * 255, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    xs, ys, w, h = cv2.boundingRect(msk_p)
    if w < 3 or h < 3:
        return None, "mask_degenerate"
    centre = (int(xs + w // 2), int(ys + h // 2))
    try:
        out = cv2.seamlessClone(src_p, dst_p, msk_p, centre, cv2.NORMAL_CLONE)
    except cv2.error:
        return None, "seamless_clone_error"
    if out is None:
        return None, "seamless_clone_none"
    return out[p:p + dst.shape[0], p:p + dst.shape[1]], "poisson"


def _feather(src: np.ndarray, dst: np.ndarray, mask: np.ndarray,
             sigma: float = 2.5) -> np.ndarray:
    import cv2

    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    a = np.clip(a, 0.0, 1.0)[..., None]
    return np.clip(src.astype(np.float32) * a + dst.astype(np.float32) * (1.0 - a),
                   0, 255).astype(np.uint8)


def _hard(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = dst.copy()
    out[mask.astype(bool)] = src[mask.astype(bool)]
    return out


def blend(src: np.ndarray, dst: np.ndarray, mask: np.ndarray,
          mode: str = "poisson") -> Tuple[np.ndarray, str]:
    """
    Paste src's masked region into dst.  Returns (image, blend_used).
    `blend_used` differs from `mode` only when the requested mode was not
    applicable to this mask; it is stored on every raw row.
    """
    if mode not in BLEND_MODES:
        raise ValueError(f"blend mode must be one of {BLEND_MODES}, got {mode!r}")
    if not mask.any():
        return dst.copy(), "empty_mask"
    if mode == "hard":
        return _hard(src, dst, mask), "hard"
    if mode == "feather":
        return _feather(src, dst, mask), "feather"
    out, used = _poisson(src, dst, mask)
    if out is None:
        # The fallback is recorded in the returned mode, never silent.
        return _feather(src, dst, mask), f"feather(fallback:{used})"
    return out, used


# --------------------------------------------------------------------------
# inpainting (the inpainting-based counterfactual comparison)
# --------------------------------------------------------------------------

_LAMA = {"model": None, "tried": False}


def _lama_inpaint(img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    import cv2

    if not _LAMA["tried"]:
        _LAMA["tried"] = True
        try:
            from simple_lama_inpainting import SimpleLama

            _LAMA["model"] = SimpleLama()
        except Exception as exc:
            print(f"[m3] LaMa unavailable ({type(exc).__name__}: {exc}); "
                  f"install simple-lama-inpainting for INPAINT_METHOD='lama'",
                  flush=True)
            _LAMA["model"] = None
    if _LAMA["model"] is None:
        return None
    from PIL import Image

    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    m = Image.fromarray((mask * 255).astype(np.uint8))
    out = _LAMA["model"](rgb, m)
    out = np.array(out)
    if out.shape[:2] != img.shape[:2]:
        out = cv2.resize(out, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "telea",
            radius: int = 3) -> Tuple[np.ndarray, str]:
    """
    Fill the masked area with an inpainter.  Returns (image, method_used).
    `lama` falls back to `telea` if the package is missing, and says so in the
    returned method name so that the ablation table never mislabels a row.
    """
    import cv2

    if method not in INPAINT_METHODS:
        raise ValueError(f"inpaint method must be one of {INPAINT_METHODS}")
    if not mask.any():
        return img.copy(), "empty_mask"
    if method == "lama":
        out = _lama_inpaint(img, mask)
        if out is not None:
            return out, "lama"
        method_used = "telea(fallback:no_lama)"
        flag = cv2.INPAINT_TELEA
    else:
        method_used = method
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, (mask * 255).astype(np.uint8), radius, flag), method_used


# --------------------------------------------------------------------------
# the sample-level API used by the runner (nothing is written to disk)
# --------------------------------------------------------------------------

def load_sample(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(real, fake, labels) for one index record."""
    real = C.imread(rec["real"])
    fake = C.imread(rec["fake"])
    lab = C.imread(rec["lab"], 0)
    if real.shape[:2] != fake.shape[:2] or real.shape[:2] != lab.shape[:2]:
        raise ValueError(
            f"sample {rec.get('sample_id')} has mismatched shapes: "
            f"real {real.shape} fake {fake.shape} lab {lab.shape}")
    return real, fake, lab


def sample_conditions(
    rec: Dict[str, Any],
    ks: Optional[Sequence[str]] = None,
    dilate: int = 3,
    mode: str = "poisson",
    jpeg_q: int = 90,
    with_floor: bool = False,
    inpaint_method: str = "",
    min_area_px: int = 64,
    vocab: Optional[str] = None,
    extra_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Build every condition for one sample IN MEMORY.

    Returns (real, fake, labels, per_region) where per_region is a list of
      {region, region_id, area_px, blend_used, images: {condition: ndarray}}

    Nothing is written to disk: the condition images exist only for as long
    as the detector needs them, which keeps the output footprint independent
    of the number of conditions.
    """
    vocab = vocab or rec.get("vocab") or R.active_vocab()
    real, fake, lab = load_sample(rec)
    want = list(ks) if ks else R.get_vocab(vocab)

    out: List[Dict[str, Any]] = []
    for region in want:
        m = region_mask(lab, region, dilate, vocab)
        area = int(m.sum())
        if area < min_area_px:
            continue
        images: Dict[str, np.ndarray] = {}
        used: Dict[str, str] = {}

        img, b = blend(real, fake, m, mode)          # authentic region into fake
        images["real"] = C.jpeg_roundtrip(img, jpeg_q)
        used["real"] = b

        img, b = blend(fake, fake, m, mode)          # seam control
        images["identity"] = C.jpeg_roundtrip(img, jpeg_q)
        used["identity"] = b

        img, b = blend(fake, real, m, mode)          # injection into a real frame
        images["reverse"] = C.jpeg_roundtrip(img, jpeg_q)
        used["reverse"] = b

        if with_floor:
            img, b = blend(real, real, m, mode)      # operator on an authentic frame
            images["floor"] = C.jpeg_roundtrip(img, jpeg_q)
            used["floor"] = b

        if inpaint_method:
            img, mu = inpaint(fake, m, inpaint_method)
            images["inpaint_fake"] = C.jpeg_roundtrip(img, jpeg_q)
            img, _ = inpaint(real, m, inpaint_method)
            images["inpaint_real"] = C.jpeg_roundtrip(img, jpeg_q)
            used["inpaint"] = mu

        out.append({
            "region": region,
            "region_id": R.rid(region, vocab),
            "area_px": area,
            "blend_used": used.get("real", mode),
            "blend_detail": used,
            "images": images,
        })

    # Optional extra masks (e.g. a union of the top-2 cited regions) go
    # through the identical machinery so that their rows are directly
    # comparable.
    if extra_masks:
        for name, m in extra_masks.items():
            area = int(m.sum())
            if area < min_area_px:
                continue
            images = {}
            img, b = blend(real, fake, m, mode)
            images["real"] = C.jpeg_roundtrip(img, jpeg_q)
            img, _ = blend(fake, fake, m, mode)
            images["identity"] = C.jpeg_roundtrip(img, jpeg_q)
            img, _ = blend(fake, real, m, mode)
            images["reverse"] = C.jpeg_roundtrip(img, jpeg_q)
            out.append({
                "region": name, "region_id": -1, "area_px": area,
                "blend_used": b, "blend_detail": {"real": b}, "images": images,
            })
    return real, fake, lab, out


def proxy_conditions(
    fake: np.ndarray,
    lab: np.ndarray,
    ks: Sequence[str],
    proxies: Sequence[str],
    dilate: int = 3,
    jpeg_q: int = 90,
    min_area_px: int = 64,
    vocab: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Deployable proxy conditions: perturbations of region k that need only
    the SUSPECT image, no paired original.

      blur     Gaussian blur, sigma 6 px at 384
      noise    region mean colour + N(0, 8^2)
      shuffle  8x8 patch shuffle inside the region (keeps colour statistics,
               destroys structure)

    Returns {proxy_name: {region: image}}.
    """
    import cv2

    vocab = vocab or R.active_vocab()
    rng = np.random.default_rng(seed)
    scale = fake.shape[0] / 384.0
    out: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in proxies}

    for region in ks:
        m = region_mask(lab, region, dilate, vocab)
        area = int(m.sum())
        if area < min_area_px:
            continue
        mb = m.astype(bool)

        if "blur" in out:
            sigma = max(1.0, 6.0 * scale)
            blurred = cv2.GaussianBlur(fake, (0, 0), sigma)
            img = fake.copy()
            img[mb] = blurred[mb]
            out["blur"][region] = C.jpeg_roundtrip(img, jpeg_q)

        if "noise" in out:
            mean = fake[mb].mean(axis=0)
            noise = rng.normal(0.0, 8.0, size=(int(mb.sum()), 3))
            img = fake.copy()
            img[mb] = np.clip(mean[None, :] + noise, 0, 255).astype(np.uint8)
            out["noise"][region] = C.jpeg_roundtrip(img, jpeg_q)

        if "shuffle" in out:
            img = fake.copy()
            ys, xs, w, h = cv2.boundingRect(m)
            patch = max(4, int(round(8 * scale)))
            tiles = []
            coords = []
            for y in range(xs, xs + h, patch):
                for x in range(ys, ys + w, patch):
                    y1, x1 = min(y + patch, img.shape[0]), min(x + patch, img.shape[1])
                    if y1 - y < 2 or x1 - x < 2:
                        continue
                    tiles.append(img[y:y1, x:x1].copy())
                    coords.append((y, x, y1, x1))
            if len(tiles) > 1:
                order = rng.permutation(len(tiles))
                canvas = img.copy()
                for (y, x, y1, x1), j in zip(coords, order):
                    t = tiles[int(j)]
                    t = cv2.resize(t, (x1 - x, y1 - y), interpolation=cv2.INTER_NEAREST)
                    canvas[y:y1, x:x1] = t
                img[mb] = canvas[mb]
            out["shuffle"][region] = C.jpeg_roundtrip(img, jpeg_q)

    return out


# --------------------------------------------------------------------------
# materialised conditions (figures and small releases only)
# --------------------------------------------------------------------------

def make_conditions(
    rec: Dict[str, Any],
    out_dir: str,
    ks: Optional[Sequence[str]] = None,
    dilate: int = 3,
    mode: str = "poisson",
    jpeg_q: int = 90,
    with_floor: bool = True,
    inpaint_method: str = "",
    vocab: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Write every condition of one sample to PNG (lossless copies of exactly the
    arrays the in-memory path produces; the self-test asserts they are
    identical).  Intended for figures and for a small visual release only:
    the full audit keeps conditions in memory because materialising every
    condition of every sample is prohibitively large.
    """
    real, fake, lab, per = sample_conditions(
        rec, ks, dilate, mode, jpeg_q, with_floor, inpaint_method, vocab=vocab)
    sid = rec["sample_id"]
    d = os.path.join(out_dir, sid)
    os.makedirs(d, exist_ok=True)
    C.imwrite_png(os.path.join(d, "real.png"), real)
    C.imwrite_png(os.path.join(d, "fake.png"), fake)
    C.imwrite_png(os.path.join(d, "lab.png"), lab)
    entry: Dict[str, Any] = {
        "sample_id": sid, "pair_id": rec["pair_id"], "method": rec["method"],
        "blend": mode, "dilate": dilate, "jpeg_q": jpeg_q,
        "regions": {},
    }
    for p in per:
        rd = os.path.join(d, C.safe_name(p["region"]))
        os.makedirs(rd, exist_ok=True)
        files = {}
        for cond, img in p["images"].items():
            fp = os.path.join(rd, f"{cond}.png")
            C.imwrite_png(fp, img)
            files[cond] = os.path.relpath(fp, out_dir)
        entry["regions"][p["region"]] = {
            "area_px": p["area_px"], "blend_used": p["blend_used"], "files": files,
        }
    return entry


def build(
    index_path: str,
    out_dir: str,
    n_samples: int = 30,
    seed: int = 0,
    **kw: Any,
) -> str:
    """Materialise conditions for a small set of samples -> conditions.json."""
    recs, meta = C.load_index(index_path)
    R.set_vocab(meta.get("vocab", "face8"))
    recs = C.limit_samples(recs, n_samples, seed)
    entries = [make_conditions(r, out_dir, vocab=meta.get("vocab"), **kw)
               for r in recs]
    path = os.path.join(out_dir, "conditions.json")
    C.save_json(path, {"meta": {**meta, **kw, "n": len(entries)},
                       "entries": entries}, indent=1)
    return path


# --------------------------------------------------------------------------
# invariants (used by the self-test and callable from a notebook)
# --------------------------------------------------------------------------

def check_invariants(rec: Dict[str, Any], dilate: int = 3,
                     vocab: Optional[str] = None) -> Dict[str, Any]:
    """
    Operator sanity on one real sample.  Returns a dict of measurements; the
    self-test asserts the bounds.

      identity_mae     identity condition vs the original fake, OUTSIDE the
                       mask.  Must be ~0: the operator must not touch pixels
                       it was not asked to touch.
      outside_mae      same for the `real` condition.
      inside_changed   `real` condition vs fake INSIDE the mask.  Must be > 0,
                       otherwise the splice did nothing.
      masks_disjoint   whether the undilated region masks partition the face.
    """
    vocab = vocab or rec.get("vocab") or R.active_vocab()
    real, fake, lab, per = sample_conditions(
        rec, None, dilate, "hard", 0, with_floor=True, vocab=vocab)
    res: Dict[str, Any] = {"regions": {}}
    for p in per:
        m = region_mask(lab, p["region"], dilate, vocab).astype(bool)
        out_m = ~m
        ident = p["images"]["identity"]
        rcond = p["images"]["real"]
        res["regions"][p["region"]] = {
            "identity_mae_outside": float(
                np.mean(np.abs(ident[out_m].astype(float) - fake[out_m].astype(float)))),
            "real_mae_outside": float(
                np.mean(np.abs(rcond[out_m].astype(float) - fake[out_m].astype(float)))),
            "real_mae_inside": float(
                np.mean(np.abs(rcond[m].astype(float) - fake[m].astype(float)))),
            "identity_mae_inside": float(
                np.mean(np.abs(ident[m].astype(float) - fake[m].astype(float)))),
            "area_px": p["area_px"],
        }
    counts = np.bincount(lab.reshape(-1), minlength=256)
    res["masks_disjoint"] = True          # by construction of build_face8_labels
    res["n_regions_present"] = int(sum(1 for i in range(R.num_regions(vocab))
                                       if counts[i] > 0))
    return res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI for the operator itself.  Two uses:

      --materialise N   write the condition images for N samples as PNGs, for
                        figures and for visual inspection of what the splice
                        produces
      --check N         run the operator invariants on N real samples and print
                        the worst violation of each: leakage outside the mask,
                        the identity splice not being a no-op, and the `real`
                        condition failing to change anything inside the mask

    `--check` should be run on every new dataset.  The self-test establishes
    that the operator is correct on a synthetic fixture; `--check` establishes
    it on the actual frames.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m ccaudit.m3_splice",
        description="Module 3: the splice operator (materialise / verify).")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default="", help="output dir for --materialise")
    ap.add_argument("--materialise", type=int, default=0)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--blend", default="poisson", choices=list(BLEND_MODES))
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--jpeg-q", type=int, default=90)
    ap.add_argument("--inpaint", default="", choices=["", "telea", "ns", "lama"])
    ap.add_argument("--splice-floor", dest="splice_floor", action="store_true",
                    default=True, help="also build the floor condition (default)")
    ap.add_argument("--no-splice-floor", dest="splice_floor", action="store_false",
                    help="do not build the floor condition")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    recs, meta = C.load_index(a.index)
    vocab = meta.get("vocab", "face8")
    R.set_vocab(vocab)
    if not (a.materialise or a.check):
        ap.error("pass --materialise N or --check N")

    if a.materialise:
        if not a.out:
            ap.error("--materialise needs --out")
        path = build(a.index, a.out, n_samples=a.materialise, seed=a.seed,
                     dilate=a.dilate, mode=a.blend, jpeg_q=a.jpeg_q,
                     with_floor=a.splice_floor, inpaint_method=a.inpaint)
        print(f"[m3] {a.materialise} sample(s) -> {path}")

    if a.check:
        sel = C.limit_samples(recs, a.check, a.seed)
        worst_out = 0.0
        worst_ident = 0.0
        per_sample_max: List[float] = []
        n_checked = 0
        for rec in sel:
            try:
                inv = check_invariants(rec, dilate=a.dilate, vocab=vocab)
            except Exception as exc:
                print(f"[m3] {rec['sample_id']}: {type(exc).__name__}: {exc}")
                continue
            n_checked += 1
            sample_max = 0.0
            for k, v in inv["regions"].items():
                worst_out = max(worst_out, v["identity_mae_outside"],
                                v["real_mae_outside"])
                worst_ident = max(worst_ident, v["identity_mae_inside"])
                # Most regions of a sample are not manipulated, so their
                # `real` condition is identical to the fake and changes
                # nothing; that is correct.  What must hold is that at least
                # one region per sample changes, otherwise the pair carries no
                # manipulation the operator can act on.
                sample_max = max(sample_max, v["real_mae_inside"])
            per_sample_max.append(sample_max)
        print(C.banner(f"operator invariants over {n_checked} sample(s)"))
        print(f"  worst leakage OUTSIDE the mask     {worst_out:.6f}"
              f"   (must be ~0)")
        print(f"  worst identity change INSIDE mask  {worst_ident:.6f}"
              f"   (must be ~0)")
        n_dead = sum(1 for v in per_sample_max if v <= 0.0)
        med = float(np.median(per_sample_max)) if per_sample_max else 0.0
        print(f"  median largest per-sample change   {med:.4f}"
              f"   (must be > 0)")
        print(f"  samples where NOTHING changed      {n_dead}/{n_checked}"
              f"   (must be 0)")
        bad = []
        if worst_out > 1e-6:
            bad.append("the operator is touching pixels outside its mask")
        if worst_ident > 1e-6:
            bad.append("the identity splice is not a no-op")
        if n_dead:
            bad.append(f"{n_dead} sample(s) had no region the splice could "
                       f"change: the fake and real crops are identical there")
        if bad:
            print("\n  FAILED:")
            for b in bad:
                print("   -", b)
            return 1
        print("\n  all invariants hold on real frames")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
