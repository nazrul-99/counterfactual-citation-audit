"""
ccaudit.m5_runner -- Module 5, the audit runner.

For every (sample, region) the runner produces one raw row containing every
number the metrics module needs.  It is the only module that calls detectors.
Four properties make long, interruptible sessions practical:

In memory       Conditions are built by m3.sample_conditions while scoring and
                discarded immediately; no counterfactual image is written to
                disk.

Content cache   Every model call is keyed by
                    stable_id(detector, sample_id, region, condition, blend,
                              dilate, jpeg_q)
                so a re-run never repeats a call it has already made, and
                caches from different sessions merge safely: the key depends
                on what was asked, never on when or in which shard.  For VLM
                detectors the detector component of the key carries the input
                resolution and quantisation when they differ from the
                defaults (see `cache_name`).

Skip without    If every key a sample needs is already cached, the sample is
re-splicing     rebuilt from the cache without decoding or splicing anything,
                which makes resuming cheap rather than merely possible.

Time budget     The loop checks a hard wall-clock budget, flushes the cache and
                the raw rows, and exits cleanly.  Re-running continues.

Outputs per detector: raw_<stem>.json (rows), cache_<stem>.json (content
cache) and runstats_<stem>.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import common as C
from . import m3_splice as M3
from . import m4_detectors as M4
from . import regions as R


# --------------------------------------------------------------------------
# cache keys
# --------------------------------------------------------------------------

DEFAULT_MAX_PIXELS = 384 * 384
DEFAULT_QUANT = "fp16"


def cache_name(det) -> str:
    """
    The detector component of every cache key.

    Layout: `<det.name>` for detectors without a resolution or quantisation
    setting, and `<det.name>[:px<max_pixels>][:q<quant>]` for VLM detectors,
    where each suffix is appended only when the setting differs from the
    default (384*384 pixels, fp16).  Keys written with default settings are
    therefore identical to keys without the suffixes, so existing cache files
    remain valid, while runs at another resolution or quantisation never
    reuse each other's predictions.
    """
    name = str(det.name)
    mp = getattr(det, "max_pixels", None)
    if mp is not None and int(mp) != DEFAULT_MAX_PIXELS:
        name += f":px{int(mp)}"
    q = getattr(det, "quant", None)
    if q is not None and str(q) != DEFAULT_QUANT:
        name += f":q{q}"
    return name


def cond_key(det: str, sid: str, region: str, cond: str, blend: str,
             dilate: int, jpeg_q: int) -> str:
    """Key of one condition prediction; `det` is the `cache_name` string."""
    return C.stable_id(det, sid, region, cond, blend, dilate, jpeg_q)


def orig_key(det: str, sid: str, what: str, variant: int = 0) -> str:
    """Key of a per-sample quantity (original predictions, citations, text);
    `det` is the `cache_name` string."""
    if variant:
        return C.stable_id(det, sid, what, "v", variant)
    return C.stable_id(det, sid, what)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

class RunConfig:
    def __init__(self, **kw: Any):
        self.blend: str = kw.get("blend", "poisson")
        self.dilate: int = int(kw.get("dilate", 3))
        self.jpeg_q: int = int(kw.get("jpeg_q", 90))
        self.regions: List[str] = list(kw.get("regions") or [])
        self.splice_floor: bool = bool(kw.get("splice_floor", False))
        self.inpaint: str = kw.get("inpaint", "") or ""
        self.prompt_variants: int = int(kw.get("prompt_variants", 0))
        self.no_cite: bool = bool(kw.get("no_cite", False))
        self.proxies: List[str] = list(kw.get("proxies") or [])
        # A proxy's floor (its effect on an authentic frame) determines
        # whether the proxy is usable at all, so it is computed by default
        # whenever proxies are requested.
        self.proxy_floor: bool = bool(kw.get("proxy_floor", True))
        self.free_text: bool = bool(kw.get("free_text", False))
        self.pairs_topk: int = int(kw.get("pairs_topk", 0))
        self.cite_mode: str = kw.get("cite_mode", "text")
        self.encoder_probe: bool = bool(kw.get("encoder_probe", False))
        self.attribution: bool = bool(kw.get("attribution", False))
        self.tag: str = kw.get("tag", "main")
        self.min_area_px: int = int(kw.get("min_area_px", 64))
        self.vocab: str = kw.get("vocab", "face8")

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------
# one sample
# --------------------------------------------------------------------------

def _needed_keys(det_name: str, rec: Dict[str, Any], ks: Sequence[str],
                 cfg: RunConfig) -> List[str]:
    """Every cache key this sample will require.  Used to decide whether the
    sample can be rebuilt without decoding any image."""
    sid = rec["sample_id"]
    keys = [orig_key(det_name, sid, "orig_fake"),
            orig_key(det_name, sid, "orig_real")]
    if not cfg.no_cite:
        keys.append(orig_key(det_name, sid, "cite_orig"))
    for v in range(1, cfg.prompt_variants):
        keys.append(orig_key(det_name, sid, "orig_fake", v))
        if not cfg.no_cite:
            keys.append(orig_key(det_name, sid, "cite_orig", v))
    conds = ["real", "identity", "reverse"]
    if cfg.splice_floor:
        conds.append("floor")
    if cfg.inpaint:
        conds += ["inpaint_fake", "inpaint_real"]
    for k in ks:
        for c in conds:
            keys.append(cond_key(det_name, sid, k, c, cfg.blend, cfg.dilate,
                                 cfg.jpeg_q))
        if not cfg.no_cite:
            keys.append(cond_key(det_name, sid, k, "cite_reverse", cfg.blend,
                                 cfg.dilate, cfg.jpeg_q))
        for p in cfg.proxies:
            keys.append(cond_key(det_name, sid, k, f"proxy_{p}", cfg.blend,
                                 cfg.dilate, cfg.jpeg_q))
            if cfg.proxy_floor:
                keys.append(cond_key(det_name, sid, k, f"proxyfloor_{p}",
                                     cfg.blend, cfg.dilate, cfg.jpeg_q))
        if cfg.cite_mode in ("overlay", "both"):
            keys.append(cond_key(det_name, sid, k, "overlay", cfg.blend,
                                 cfg.dilate, cfg.jpeg_q))
        if cfg.encoder_probe:
            keys.append(cond_key(det_name, sid, k, "enc_shift", cfg.blend,
                                 cfg.dilate, cfg.jpeg_q))
    if cfg.free_text:
        keys.append(orig_key(det_name, sid, "explain"))
    if cfg.attribution:
        keys.append(orig_key(det_name, sid, "cite_attr"))
    if cfg.pairs_topk >= 2:
        keys.append(orig_key(det_name, sid, "union_top2"))
    return keys


def _enc_shift(det, img_a: np.ndarray, img_b: np.ndarray,
               labels: np.ndarray, region: str) -> Dict[str, float]:
    """
    Encoder blind-spot probe.

    S_k = 1 - cos( mean visual tokens covering region k under x^f ,
                   same tokens under the counterfactual x~f_k )
    plus the same quantity for the tokens outside k as a control, which is
    near zero except at the seam.  A small S_k together with an unchanged
    verdict indicates that the encoder representation does not reflect the
    evidence in region k.

    Cost: this implementation runs two additional vision-encoder passes per
    region rather than reusing the runner's existing forward passes.  Only
    the vision tower is re-run (the hook fires at the projector), so it is
    much cheaper than a full generate, but --encoder-probe should be budgeted
    as a surcharge of roughly 30-50% on the run.  Results are cached like
    every other call, so the cost is paid once.
    """
    import cv2

    ta, grid = det.encoder_features(img_a)
    tb, _ = det.encoder_features(img_b)
    n = min(ta.shape[0], tb.shape[0])
    ta, tb = ta[:n], tb[:n]
    gh, gw = grid
    cells = np.zeros((gh, gw), np.float32)
    m = (labels == R.rid(region)).astype(np.float32)
    small = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
    cells[:] = small
    sel = (cells.reshape(-1) >= 0.5)[:n]
    if sel.sum() < 1:
        sel = (cells.reshape(-1) > 0)[:n]

    def cos_shift(mask: np.ndarray) -> float:
        if mask.sum() < 1:
            return float("nan")
        a = ta[mask].mean(axis=0)
        b = tb[mask].mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return float("nan")
        return float(1.0 - float(np.dot(a, b) / (na * nb)))

    return {"in": cos_shift(sel), "out": cos_shift(~sel)}


def run_sample(det, rec: Dict[str, Any], cfg: RunConfig,
               cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score one sample with one detector.  Returns its raw rows.

    Cache keys use `cache_name(det)`; the `detector` field of each row uses
    `det.name`, so row files are unaffected by the key suffixes."""
    sid = rec["sample_id"]
    vocab = rec.get("vocab", cfg.vocab)
    present = rec.get("present_regions") or R.get_vocab(vocab)
    ks = [k for k in present if (not cfg.regions or k in cfg.regions)]
    if not ks:
        return []

    dname = cache_name(det)
    keys = _needed_keys(dname, rec, ks, cfg)
    fully_cached = all(k in cache for k in keys)

    real = fake = lab = None
    per: List[Dict[str, Any]] = []
    if not fully_cached:
        real, fake, lab, per = M3.sample_conditions(
            rec, ks, cfg.dilate, cfg.blend, cfg.jpeg_q,
            with_floor=cfg.splice_floor, inpaint_method=cfg.inpaint,
            min_area_px=cfg.min_area_px, vocab=vocab)
        ks = [p["region"] for p in per]      # regions that produced masks
        if not ks:
            return []

    def call(key: str, fn) -> Any:
        if key in cache:
            return cache[key]
        val = fn()
        cache[key] = val
        return val

    # ---- originals -------------------------------------------------------
    p_orig_fake = call(orig_key(dname, sid, "orig_fake"),
                       lambda: det.predict(fake, lab, 0))
    p_orig_real = call(orig_key(dname, sid, "orig_real"),
                       lambda: det.predict(real, lab, 0))
    cite_orig = ({} if cfg.no_cite else
                 call(orig_key(dname, sid, "cite_orig"),
                      lambda: det.cite(fake, lab, 0)))

    p_var: List[float] = []
    cite_var: List[Dict[str, float]] = []
    for v in range(1, cfg.prompt_variants):
        p_var.append(call(orig_key(dname, sid, "orig_fake", v),
                          lambda v=v: det.predict(fake, lab, v)))
        if not cfg.no_cite:
            cite_var.append(call(orig_key(dname, sid, "cite_orig", v),
                                 lambda v=v: det.cite(fake, lab, v)))

    explanation = (call(orig_key(dname, sid, "explain"),
                        lambda: det.explain(fake))
                   if cfg.free_text and hasattr(det, "explain") else None)

    cite_attr = (call(orig_key(dname, sid, "cite_attr"),
                      lambda: det.attribute(fake, lab))
                 if cfg.attribution and hasattr(det, "attribute") else None)

    # ---- proxies (need only the suspect image) ---------------------------
    proxy_imgs: Dict[str, Dict[str, np.ndarray]] = {}
    pixel_proxies = [p for p in cfg.proxies if p in ("blur", "noise", "shuffle")]
    proxy_imgs_real: Dict[str, Dict[str, np.ndarray]] = {}
    if pixel_proxies and not fully_cached:
        proxy_imgs = M3.proxy_conditions(
            fake, lab, ks, pixel_proxies, cfg.dilate, cfg.jpeg_q,
            min_area_px=cfg.min_area_px, vocab=vocab)
        if cfg.proxy_floor:
            # The same perturbation applied to the authentic frame measures
            # how much manipulation evidence the proxy introduces by itself.
            proxy_imgs_real = M3.proxy_conditions(
                real, lab, ks, pixel_proxies, cfg.dilate, cfg.jpeg_q,
                min_area_px=cfg.min_area_px, vocab=vocab)

    # ---- per-region ------------------------------------------------------
    by_region = {p["region"]: p for p in per}
    rows: List[Dict[str, Any]] = []
    for k in ks:
        p_ = by_region.get(k)
        imgs = p_["images"] if p_ else {}

        def cond(name: str, fallback=None):
            key = cond_key(dname, sid, k, name, cfg.blend, cfg.dilate,
                           cfg.jpeg_q)
            if key in cache:
                return cache[key]
            if name not in imgs:
                return fallback
            val = det.predict(imgs[name], lab, 0)
            cache[key] = val
            return val

        row: Dict[str, Any] = {
            "sample_id": sid,
            "pair_id": rec["pair_id"],
            "method": rec["method"],
            "frame": rec.get("frame"),
            "split": rec.get("split") or C.split_of(rec["pair_id"]),
            "region": k,
            "region_id": R.rid(k, vocab),
            "detector": det.name,
            "tag": cfg.tag,
            "area_px": int(p_["area_px"]) if p_ else None,
            "blend": cfg.blend,
            "dilate": cfg.dilate,
            "jpeg_q": cfg.jpeg_q,
            "blend_used": p_["blend_used"] if p_ else cfg.blend,
            "p_orig_fake": p_orig_fake,
            "p_orig_real": p_orig_real,
            "p_real": cond("real"),
            "p_identity": cond("identity"),
            "p_reverse": cond("reverse"),
            "cite_orig": cite_orig,
        }
        if cfg.splice_floor:
            row["p_floor"] = cond("floor")
        if cfg.inpaint:
            row["inpaint"] = cfg.inpaint
            row["p_inpaint_fake"] = cond("inpaint_fake")
            row["p_inpaint_real"] = cond("inpaint_real")
        if not cfg.no_cite:
            key = cond_key(dname, sid, k, "cite_reverse", cfg.blend,
                           cfg.dilate, cfg.jpeg_q)
            if key in cache:
                row["cite_reverse"] = cache[key]
            elif "reverse" in imgs:
                cache[key] = det.cite(imgs["reverse"], lab, 0)
                row["cite_reverse"] = cache[key]
        if p_var:
            row["p_orig_fake_variants"] = p_var
        if cite_var:
            row["cite_orig_variants"] = cite_var
        if explanation is not None:
            row["explanation_text"] = explanation
        if cite_attr is not None:
            row["cite_attr"] = cite_attr

        # proxies
        for pname in cfg.proxies:
            key = cond_key(dname, sid, k, f"proxy_{pname}", cfg.blend,
                           cfg.dilate, cfg.jpeg_q)
            if key in cache:
                row[f"p_{pname}"] = cache[key]
            elif pname == "selfcheck" and hasattr(det, "p_ignore"):
                cache[key] = det.p_ignore(fake, k)
                row[f"p_{pname}"] = cache[key]
            elif pname in proxy_imgs and k in proxy_imgs[pname]:
                cache[key] = det.predict(proxy_imgs[pname][k], lab, 0)
                row[f"p_{pname}"] = cache[key]

            if not cfg.proxy_floor:
                continue
            fkey = cond_key(dname, sid, k, f"proxyfloor_{pname}", cfg.blend,
                            cfg.dilate, cfg.jpeg_q)
            if fkey in cache:
                row[f"p_{pname}_real"] = cache[fkey]
            elif pname == "selfcheck" and hasattr(det, "p_ignore") and real is not None:
                cache[fkey] = det.p_ignore(real, k)
                row[f"p_{pname}_real"] = cache[fkey]
            elif pname in proxy_imgs_real and k in proxy_imgs_real[pname]:
                cache[fkey] = det.predict(proxy_imgs_real[pname][k], lab, 0)
                row[f"p_{pname}_real"] = cache[fkey]

        # overlay-elicited citation
        if cfg.cite_mode in ("overlay", "both") and hasattr(det, "p_overlay"):
            key = cond_key(dname, sid, k, "overlay", cfg.blend, cfg.dilate,
                           cfg.jpeg_q)
            if key in cache:
                row["p_mark"] = cache[key]
            elif fake is not None:
                marked = M4.draw_region_outline(fake, lab, k, vocab=vocab)
                cache[key] = det.p_overlay(marked)
                row["p_mark"] = cache[key]

        # encoder blind-spot probe
        if cfg.encoder_probe and hasattr(det, "encoder_features"):
            key = cond_key(dname, sid, k, "enc_shift", cfg.blend,
                           cfg.dilate, cfg.jpeg_q)
            if key in cache:
                s = cache[key]
            elif "real" in imgs:
                s = _enc_shift(det, fake, imgs["real"], lab, k)
                cache[key] = s
            else:
                s = None
            if s:
                row["enc_shift_in"] = s.get("in")
                row["enc_shift_out"] = s.get("out")

        rows.append(row)

    # ---- two-region union (compositional test) ---------------------------
    if cfg.pairs_topk >= 2 and cite_orig and rows:
        key = orig_key(dname, sid, "union_top2")
        if key in cache:
            u = cache[key]
        elif lab is not None:
            top2 = sorted(ks, key=lambda k: cite_orig.get(k, 0.0), reverse=True)[:2]
            m = M3.union_mask(lab, top2, cfg.dilate, vocab)
            img, _b = M3.blend(real, fake, m, cfg.blend)
            u = {"regions": top2,
                 "p_union_real": det.predict(C.jpeg_roundtrip(img, cfg.jpeg_q),
                                             lab, 0)}
            img, _b = M3.blend(fake, fake, m, cfg.blend)
            u["p_union_identity"] = det.predict(
                C.jpeg_roundtrip(img, cfg.jpeg_q), lab, 0)
            cache[key] = u
        else:
            u = None
        if u:
            for row in rows:
                row["cite_union_regions"] = u["regions"]
                row["p_union_real"] = u["p_union_real"]
                row["p_union_identity"] = u["p_union_identity"]

    # The overlay citation is a distribution over regions, attached to every row.
    if cfg.cite_mode in ("overlay", "both"):
        marks = {r["region"]: r.get("p_mark") for r in rows
                 if r.get("p_mark") is not None}
        if marks:
            tot = sum(marks.values()) or 1.0
            dist = {k: v / tot for k, v in marks.items()}
            for row in rows:
                row["cite_overlay"] = dist
    return rows


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _out_stem(det_name: str, tag: str, shard_spec: str) -> str:
    stem = C.safe_name(det_name)
    if tag:
        stem += f"_{C.safe_name(tag)}"
    i, n = C.parse_shard(shard_spec)
    if n > 1:
        stem += f"_s{i}of{n}"
    return stem


def load_caches(det_name: str, tag: str, dirs: Sequence[str]) -> Dict[str, Any]:
    """
    Seed the cache from every cache_*.json for this detector under the given
    directories.  Keys are content-addressed, so overlapping runs merge safely
    and de-duplicate themselves.
    """
    import fnmatch

    cache: Dict[str, Any] = {}
    pattern = f"cache_{C.safe_name(det_name)}*"
    n_files = 0
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dd, files in os.walk(d):
            for f in files:
                if f.endswith(".json") and fnmatch.fnmatch(f[:-5], pattern):
                    try:
                        blob = C.load_json(os.path.join(root, f))
                    except Exception:
                        continue
                    entries = blob.get("cache", blob) if isinstance(blob, dict) else {}
                    if isinstance(entries, dict):
                        cache.update(entries)
                        n_files += 1
    if n_files:
        print(f"[m5] seeded cache with {len(cache)} entries from {n_files} file(s)",
              flush=True)
    return cache


def run_multi(
    records: Sequence[Dict[str, Any]],
    detector_specs: Sequence[str],
    out_dir: str,
    cfg: RunConfig,
    device: str = "cpu",
    quant: str = "fp16",
    max_pixels: int = 384 * 384,
    shard_spec: str = "",
    save_every: int = 25,
    time_budget_min: float = 0.0,
    resume_from: Sequence[str] = (),
    calibrate_n: int = 32,
    seed: int = 0,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    R.set_vocab(cfg.vocab)
    budget = C.Budget(time_budget_min, label="m5")
    stats: Dict[str, Any] = {"detectors": {}, "stopped_early": False}

    all_records = list(records)
    todo = C.shard(all_records, shard_spec)
    print(f"[m5] {len(todo)} samples in this shard "
          f"(of {len(all_records)}), detectors={list(detector_specs)}",
          flush=True)

    for spec in detector_specs:
        if budget.expired:
            stats["stopped_early"] = True
            print("[m5] time budget reached before starting "
                  f"{spec!r} -- re-run to continue", flush=True)
            break

        det = M4.build_detector(spec, device=device, quant=quant,
                                max_pixels=max_pixels)
        if getattr(det, "needs_calibration", False):
            det.calibrate(all_records, n=calibrate_n, seed=seed, vocab=cfg.vocab)

        stem = _out_stem(det.name, cfg.tag, shard_spec)
        raw_path = os.path.join(out_dir, f"raw_{stem}.json")
        cache_path = os.path.join(out_dir, f"cache_{stem}.json")

        cache = load_caches(det.name, cfg.tag,
                            list(resume_from) + [out_dir])
        rows: List[Dict[str, Any]] = []
        done_samples: set = set()
        if os.path.exists(raw_path):
            try:
                prev = C.load_json(raw_path).get("rows", [])
                rows.extend(prev)
                done_samples = {r["sample_id"] for r in prev}
                print(f"[m5] resume: {len(prev)} rows / {len(done_samples)} "
                      f"samples already in {os.path.basename(raw_path)}",
                      flush=True)
            except Exception as exc:
                print(f"[m5] could not read {raw_path}: {exc}")

        pending = [r for r in todo if r["sample_id"] not in done_samples]
        meta = {
            "detector": det.name, "spec": spec, "info": det.info(),
            **cfg.as_dict(), "shard": shard_spec or "0/1",
            "n_samples_target": len(todo), "code_hash": C.code_hash(),
        }

        t0 = time.time()
        n_new = 0
        errors: List[str] = []
        for rec in pending:
            if budget.expired:
                stats["stopped_early"] = True
                break
            try:
                new_rows = run_sample(det, rec, cfg, cache)
                rows.extend(new_rows)
                n_new += 1
            except Exception as exc:
                errors.append(f"{rec['sample_id']}: {type(exc).__name__}: {exc}")
                if len(errors) <= 3:
                    print(f"[m5] ERROR on {rec['sample_id']}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            if save_every > 0 and n_new and n_new % save_every == 0:
                C.save_json(raw_path, {"meta": meta, "rows": rows})
                C.save_json(cache_path, {"meta": meta, "cache": cache})
                rate = n_new / max(1e-9, (time.time() - t0) / 60.0)
                print(f"[m5] {det.name}: {n_new}/{len(pending)} new samples "
                      f"({rate:.1f}/min), {len(rows)} rows, {budget.report()}",
                      flush=True)

        C.save_json(raw_path, {"meta": meta, "rows": rows})
        C.save_json(cache_path, {"meta": meta, "cache": cache})
        C.write_provenance(out_dir, stem, {"detector": det.name, "spec": spec,
                                           "config": cfg.as_dict()})
        n_samples = len({r["sample_id"] for r in rows})
        d_stats = {
            "rows": len(rows), "samples": n_samples,
            "new_samples_this_run": n_new,
            "target_samples": len(todo),
            "complete": n_samples >= len(todo),
            "cache_entries": len(cache),
            "elapsed_min": (time.time() - t0) / 60.0,
            "n_errors": len(errors), "errors_sample": errors[:20],
            "raw": raw_path, "cache": cache_path,
        }
        stats["detectors"][det.name] = d_stats
        C.save_json(os.path.join(out_dir, f"runstats_{stem}.json"),
                    {**d_stats, "meta": meta}, indent=1)
        print(f"[m5] {det.name}: {n_samples}/{len(todo)} samples, "
              f"{len(rows)} rows, {len(cache)} cache entries, "
              f"{d_stats['elapsed_min']:.1f} min"
              + ("  COMPLETE" if d_stats["complete"] else "  INCOMPLETE"),
              flush=True)
        if errors:
            print(f"[m5] {len(errors)} sample errors; first: {errors[0]}")
        det.close()

    if stats["stopped_early"]:
        print("[m5] STOPPED EARLY on time budget -- rows and cache are "
              "checkpointed. Pass this output directory to --resume-from "
              "and re-run to continue.",
              flush=True)
    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ccaudit.m5_runner",
        description="Module 5: run the counterfactual audit.")
    ap.add_argument("--index", required=True,
                    help="parsed/index.json (comma-separated to merge shards)")
    ap.add_argument("--detector", required=True, help="comma-separated specs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--blend", default="poisson", choices=list(M3.BLEND_MODES))
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--jpeg-q", type=int, default=90, help="0 = no re-encode")
    ap.add_argument("--regions", default="", help="restrict to these regions")
    ap.add_argument("--splice-floor", action="store_true")
    ap.add_argument("--inpaint", default="", choices=["", "telea", "ns", "lama"])
    ap.add_argument("--prompt-variants", type=int, default=0)
    ap.add_argument("--no-cite", action="store_true")
    ap.add_argument("--proxy", default="",
                    help="blur,noise,shuffle,selfcheck (deployable proxies)")
    ap.add_argument("--no-proxy-floor", action="store_true",
                    help="skip the proxy floor (not recommended: the floor "
                         "on authentic frames is what decides whether a "
                         "proxy is usable)")
    ap.add_argument("--free-text", action="store_true")
    ap.add_argument("--pairs-topk", type=int, default=0)
    ap.add_argument("--cite-mode", default="text",
                    choices=["text", "overlay", "both"])
    ap.add_argument("--encoder-probe", action="store_true")
    ap.add_argument("--attribution", action="store_true")
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--methods", default="")
    ap.add_argument("--shard", default="")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default="fp16",
                    choices=["auto", "4bit", "fp16", "bf16"])
    ap.add_argument("--max-pixels", type=int, default=384 * 384)
    ap.add_argument("--calibrate-n", type=int, default=32)
    ap.add_argument("--vocab", default="face8", choices=["face8", "grid9"])
    a = ap.parse_args(argv)

    idx_paths = [p for p in a.index.split(",") if p.strip()]
    if len(idx_paths) > 1:
        records, meta = C.merge_indices(idx_paths)
    else:
        records, meta = C.load_index(idx_paths[0])
    vocab = meta.get("vocab", a.vocab)
    R.set_vocab(vocab)

    if a.methods:
        want = {m.strip() for m in a.methods.split(",") if m.strip()}
        records = [r for r in records if r["method"] in want]
    records = C.filter_split(records, a.split)
    # The subset is taken before sharding so that every shard audits the
    # same subset.
    if a.limit_samples:
        records = C.limit_samples(records, a.limit_samples, a.seed)
    print(f"[m5] {len(records)} samples after split={a.split} "
          f"limit={a.limit_samples or 'none'}", flush=True)

    cfg = RunConfig(
        blend=a.blend, dilate=a.dilate, jpeg_q=a.jpeg_q,
        regions=[r for r in a.regions.split(",") if r.strip()],
        splice_floor=a.splice_floor, inpaint=a.inpaint,
        prompt_variants=a.prompt_variants, no_cite=a.no_cite,
        proxies=[p for p in a.proxy.split(",") if p.strip()],
        proxy_floor=not a.no_proxy_floor,
        free_text=a.free_text, pairs_topk=a.pairs_topk,
        cite_mode=a.cite_mode, encoder_probe=a.encoder_probe,
        attribution=a.attribution, tag=a.tag, vocab=vocab,
    )
    stats = run_multi(
        records, [d for d in a.detector.split(",") if d.strip()], a.out, cfg,
        device=a.device, quant=a.quant, max_pixels=a.max_pixels,
        shard_spec=a.shard, save_every=a.save_every,
        time_budget_min=a.time_budget_min,
        resume_from=[p for p in a.resume_from.split(",") if p.strip()],
        calibrate_n=a.calibrate_n, seed=a.seed,
    )
    return 2 if stats.get("stopped_early") else 0


if __name__ == "__main__":
    sys.exit(main())
