"""
ccaudit.m6_metrics -- Module 6, the metrics.

Input: raw_*.json row files written by m5.  Output: metrics.json (or
metrics_coarse.json), optionally metrics_by_method.json.

Core derivations:

    delta_raw (k)  = p(x^f) - p(x~real_k)      effect of removing region k's
                                               manipulation, seam included
    delta_seam(k)  = p(x^f) - p(x~identity_k)  effect of the seam and of the
                                               extra compression pass alone
    delta     (k)  = delta_raw - delta_seam    the seam-corrected causal effect
                   = p(x~identity_k) - p(x~real_k)

    FS (faithfulness score), per sample:
                   = delta(cited) - mean_{k != cited} delta(k)
    CR (citation recall), reverse direction:
                   over rows whose reverse condition flipped the verdict
                   (p_reverse >= tau), the frequency with which the model
                   cites the region that was injected, compared against the
                   prior, i.e. the accuracy of always guessing the most
                   common injected region.  A constant citation can reach a
                   high CR but cannot exceed its own prior, which is why
                   CR - prior rather than CR is the reverse criterion.

    faithful  <=>  FS_lo > 0  AND  CR_lo > CR_prior

Design decisions
----------------
1. Coarsening is applied exactly once, inside _coarsen(), to a normalised
   per-sample structure, never to rows that have already been coarsened.
   Mapping a citation distribution twice silently destroys most of its mass.
2. When several fine regions collapse into one coarse region, every measured
   quantity is averaged over the members that are present rather than taken
   from one representative member.
3. The resampling unit is the clip (`pair_id`), never the frame: frames of
   one clip are near-duplicates and bootstrapping over frames would
   understate every interval.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C
from . import regions as R

DEFAULT_TAUS = (0.3, 0.5, 0.7)
TAU_MAIN = 0.5
N_BOOT = 2000


# --------------------------------------------------------------------------
# loading and grouping raw files
# --------------------------------------------------------------------------

def group_raw_files(dirs: Sequence[str], pattern: str = "raw_*.json"
                    ) -> Dict[Tuple[str, str], List[str]]:
    """
    Find every raw_*.json under the given directories and group by
    (detector, tag).  Shards and separate sessions of the same run land in the
    same group and are merged; rows are de-duplicated by (sample_id, region).
    """
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dd, files in os.walk(d):
            for f in files:
                if not fnmatch.fnmatch(f, pattern):
                    continue
                path = os.path.join(root, f)
                try:
                    meta = C.load_json(path).get("meta", {})
                except Exception:
                    continue
                key = (meta.get("detector", "unknown"), meta.get("tag", ""))
                groups[key].append(path)
    return dict(groups)


def load_rows(paths: Sequence[str]) -> Tuple[List[Dict], Dict]:
    """Merge raw files, de-duplicating by (sample_id, region, detector, tag)."""
    seen: Dict[Tuple, Dict] = {}
    meta: Dict[str, Any] = {}
    for p in paths:
        blob = C.load_json(p)
        if not meta:
            meta = dict(blob.get("meta", {}))
        for r in blob.get("rows", []):
            seen[(r.get("sample_id"), r.get("region"), r.get("detector"),
                  r.get("tag"))] = r
    return list(seen.values()), meta


# --------------------------------------------------------------------------
# rows -> per-sample structure
# --------------------------------------------------------------------------

def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def samples_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalise raw rows into one record per sample.  Every downstream metric
    reads this structure, so coarsening, per-method splits and proxies all
    operate on one representation rather than on ad-hoc row filters.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = r["sample_id"]
        s = out.get(sid)
        if s is None:
            s = out[sid] = {
                "sample_id": sid,
                "pair_id": r.get("pair_id"),
                "method": r.get("method"),
                "split": r.get("split"),
                "detector": r.get("detector"),
                "tag": r.get("tag"),
                "p_orig_fake": _f(r.get("p_orig_fake")),
                "p_orig_real": _f(r.get("p_orig_real")),
                "cite_orig": dict(r.get("cite_orig") or {}),
                "cite_attr": dict(r.get("cite_attr") or {}) or None,
                "cite_overlay": dict(r.get("cite_overlay") or {}) or None,
                "cite_variants": [dict(c) for c in (r.get("cite_orig_variants") or [])],
                "p_variants": list(r.get("p_orig_fake_variants") or []),
                "explanation_text": r.get("explanation_text"),
                "union": ({"regions": r.get("cite_union_regions"),
                           "p_real": _f(r.get("p_union_real")),
                           "p_identity": _f(r.get("p_union_identity"))}
                          if r.get("cite_union_regions") else None),
                "regions": {},
            }
        pf = _f(r.get("p_orig_fake"))
        entry: Dict[str, Any] = {
            "p_real": _f(r.get("p_real")),
            "p_identity": _f(r.get("p_identity")),
            "p_reverse": _f(r.get("p_reverse")),
            "area_px": r.get("area_px"),
            "blend_used": r.get("blend_used"),
            "cite_reverse": dict(r.get("cite_reverse") or {}),
        }
        entry["delta_raw"] = pf - entry["p_real"]
        entry["delta_seam"] = pf - entry["p_identity"]
        entry["delta"] = entry["delta_raw"] - entry["delta_seam"]
        if r.get("p_floor") is not None:
            entry["p_floor"] = _f(r.get("p_floor"))
        if r.get("p_inpaint_fake") is not None:
            entry["p_inpaint_fake"] = _f(r.get("p_inpaint_fake"))
            entry["p_inpaint_real"] = _f(r.get("p_inpaint_real"))
            entry["delta_inpaint"] = pf - entry["p_inpaint_fake"]
        for pname in ("blur", "noise", "shuffle", "selfcheck"):
            if r.get(f"p_{pname}") is not None:
                entry[f"p_{pname}"] = _f(r.get(f"p_{pname}"))
                entry[f"delta_{pname}"] = pf - entry[f"p_{pname}"]
            if r.get(f"p_{pname}_real") is not None:
                entry[f"p_{pname}_real"] = _f(r.get(f"p_{pname}_real"))
        if r.get("enc_shift_in") is not None:
            entry["enc_shift_in"] = _f(r.get("enc_shift_in"))
            entry["enc_shift_out"] = _f(r.get("enc_shift_out"))
        if r.get("p_mark") is not None:
            entry["p_mark"] = _f(r.get("p_mark"))
        out[sid]["regions"][r["region"]] = entry
    return out


def _renorm(d: Dict[str, float], keys: Sequence[str]) -> Dict[str, float]:
    vals = np.array([max(0.0, _f(d.get(k, 0.0))) for k in keys])
    vals = np.nan_to_num(vals, nan=0.0)
    s = vals.sum()
    if s <= 0:
        vals = np.ones(len(keys))
        s = float(len(keys))
    return {k: float(v / s) for k, v in zip(keys, vals)}


def _coarsen(samples: Dict[str, Dict[str, Any]], vocab: Optional[str] = None
             ) -> Dict[str, Dict[str, Any]]:
    """
    Collapse the fine vocabulary into the coarse one, exactly once.

    Citation distributions are summed over the members of a coarse region (a
    probability over a partition).  Every measured effect (delta, p_*, shifts)
    is averaged over the members that are present, never inherited from one
    representative.
    """
    cmap = R.coarse_map(vocab)
    out: Dict[str, Dict[str, Any]] = {}
    for sid, s in samples.items():
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for fine, e in s["regions"].items():
            if fine in cmap:
                groups[cmap[fine]].append(e)
        if not groups:
            continue
        new_regions: Dict[str, Any] = {}
        numeric = ("p_real", "p_identity", "p_reverse", "delta_raw",
                   "delta_seam", "delta", "p_floor", "p_inpaint_fake",
                   "p_inpaint_real", "delta_inpaint", "enc_shift_in",
                   "enc_shift_out", "p_mark",
                   "p_blur", "p_noise", "p_shuffle", "p_selfcheck",
                   "p_blur_real", "p_noise_real", "p_shuffle_real",
                   "p_selfcheck_real",
                   "delta_blur", "delta_noise", "delta_shuffle",
                   "delta_selfcheck")
        for g, members in groups.items():
            e: Dict[str, Any] = {}
            for f in numeric:
                vals = [m[f] for m in members if f in m and np.isfinite(_f(m[f]))]
                if vals:
                    e[f] = float(np.mean(vals))
            areas = [m.get("area_px") or 0 for m in members]
            e["area_px"] = int(sum(areas))
            e["n_fine"] = len(members)
            e["blend_used"] = members[0].get("blend_used")
            # Reverse citations are distributions and are summed into the
            # coarse vocabulary once.
            rev_keys = R.coarse_vocab(vocab)
            acc = {k: 0.0 for k in rev_keys}
            n_rev = 0
            for m in members:
                cr = m.get("cite_reverse") or {}
                if cr:
                    n_rev += 1
                    for fine, p in cr.items():
                        if fine in cmap:
                            acc[cmap[fine]] += _f(p)
            if n_rev:
                tot = sum(acc.values()) or 1.0
                e["cite_reverse"] = {k: v / tot for k, v in acc.items()}
            else:
                e["cite_reverse"] = {}
            new_regions[g] = e

        def coarsen_dist(d: Optional[Dict[str, float]]):
            if not d:
                return d
            acc = {k: 0.0 for k in R.coarse_vocab(vocab)}
            for fine, p in d.items():
                if fine in cmap:
                    acc[cmap[fine]] += _f(p)
            tot = sum(acc.values()) or 1.0
            return {k: v / tot for k, v in acc.items()}

        ns = dict(s)
        ns["regions"] = new_regions
        ns["cite_orig"] = coarsen_dist(s["cite_orig"]) or {}
        ns["cite_attr"] = coarsen_dist(s.get("cite_attr"))
        ns["cite_overlay"] = coarsen_dist(s.get("cite_overlay"))
        ns["cite_variants"] = [coarsen_dist(c) or {} for c in s["cite_variants"]]
        out[sid] = ns
    return out


# --------------------------------------------------------------------------
# per-sample quantities
# --------------------------------------------------------------------------

def cited_region(sample: Dict[str, Any], which: str = "cite_orig"
                 ) -> Optional[str]:
    """argmax of a citation distribution, restricted to regions this sample
    has rows for (a citation for a region that is not in the crop cannot be
    tested and is not scored)."""
    ks = list(sample["regions"].keys())
    if not ks:
        return None
    d = sample.get(which) or {}
    if not d:
        return None
    dd = _renorm(d, ks)
    return max(dd, key=dd.get)


def fs_of(sample: Dict[str, Any], cited: Optional[str] = None,
          delta_field: str = "delta") -> float:
    """FS for one sample: effect at the cited region minus the mean effect
    elsewhere."""
    ks = list(sample["regions"].keys())
    cited = cited or cited_region(sample)
    if cited is None or cited not in sample["regions"] or len(ks) < 2:
        return float("nan")
    d = {k: _f(sample["regions"][k].get(delta_field)) for k in ks}
    others = [v for k, v in d.items() if k != cited and np.isfinite(v)]
    if not others or not np.isfinite(d[cited]):
        return float("nan")
    return float(d[cited] - np.mean(others))


def rank_of_cited(sample: Dict[str, Any], cited: Optional[str] = None,
                  delta_field: str = "delta") -> float:
    """1 = the cited region has the largest effect."""
    ks = list(sample["regions"].keys())
    cited = cited or cited_region(sample)
    if cited is None or cited not in ks:
        return float("nan")
    vals = [(k, _f(sample["regions"][k].get(delta_field))) for k in ks]
    vals = [(k, v) for k, v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    order = sorted(vals, key=lambda kv: -kv[1])
    for i, (k, _v) in enumerate(order):
        if k == cited:
            return float(i + 1)
    return float("nan")


# --------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------

def analyse(
    rows: Sequence[Dict[str, Any]],
    seed: int = 0,
    coarse: bool = False,
    taus: Sequence[float] = DEFAULT_TAUS,
    tau: float = TAU_MAIN,
    vocab: Optional[str] = None,
    n_boot: int = N_BOOT,
    localization: Optional[Dict[str, Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the full result dict for one detector x tag."""
    meta = dict(meta or {})
    samples = samples_from_rows(rows)
    if coarse:
        samples = _coarsen(samples, vocab)
        region_names = R.coarse_vocab(vocab)
    else:
        region_names = R.get_vocab(vocab)
    if not samples:
        return {"error": "no samples", "n_rows": len(rows)}

    sids = sorted(samples)
    groups = [samples[s]["pair_id"] for s in sids]

    res: Dict[str, Any] = {
        "detector": meta.get("detector") or rows[0].get("detector"),
        "tag": meta.get("tag") or rows[0].get("tag"),
        "blend": meta.get("blend") or rows[0].get("blend"),
        "dilate": meta.get("dilate", rows[0].get("dilate")),
        "jpeg_q": meta.get("jpeg_q", rows[0].get("jpeg_q")),
        "granularity": "coarse" if coarse else "fine",
        "split": sorted({samples[s]["split"] for s in sids if samples[s]["split"]}),
        "methods": sorted({samples[s]["method"] for s in sids}),
        "n_rows": len(rows),
        "n_samples": len(sids),
        "n_clips": len(set(groups)),
        "regions": region_names,
        "tau": tau,
    }

    # ---- AUC on the originals -------------------------------------------
    pf = [samples[s]["p_orig_fake"] for s in sids]
    pr = [samples[s]["p_orig_real"] for s in sids]
    res["AUC"] = C.auc_score(pf + pr, [1] * len(pf) + [0] * len(pr))
    res["p_orig_fake_mean"] = float(np.nanmean(pf))
    res["p_orig_real_mean"] = float(np.nanmean(pr))

    # ---- forward: FS -----------------------------------------------------
    cited = {s: cited_region(samples[s]) for s in sids}
    fs_vals, fs_groups, rank_vals = [], [], []
    for s in sids:
        v = fs_of(samples[s], cited[s])
        if np.isfinite(v):
            fs_vals.append(v)
            fs_groups.append(samples[s]["pair_id"])
            rank_vals.append(rank_of_cited(samples[s], cited[s]))
    fs, fs_lo, fs_hi = C.bootstrap_ci(fs_vals, fs_groups, n_boot, seed=seed)
    res.update({
        "FS": fs, "FS_lo": fs_lo, "FS_hi": fs_hi,
        "FS_n": len(fs_vals),
        "FS_p": C.bootstrap_p_two_sided(fs_vals, fs_groups, n_boot, seed=seed),
        "rank_cited": float(np.nanmean(rank_vals)) if rank_vals else float("nan"),
        "cited_distribution": {
            k: v / max(1, len([c for c in cited.values() if c]))
            for k, v in Counter(c for c in cited.values() if c).items()},
    })

    # effect sizes
    d_cited, d_all = [], []
    for s in sids:
        ks = list(samples[s]["regions"])
        for k in ks:
            v = _f(samples[s]["regions"][k].get("delta"))
            if np.isfinite(v):
                d_all.append(v)
                if k == cited[s]:
                    d_cited.append(v)
    res["delta_cited"] = float(np.mean(d_cited)) if d_cited else float("nan")
    res["delta_mean_all"] = float(np.mean(d_all)) if d_all else float("nan")
    if len(d_cited) > 1 and len(d_all) > 1:
        pooled = np.sqrt((np.var(d_cited) + np.var(d_all)) / 2.0)
        res["cohen_d_cited_vs_all"] = (
            float((np.mean(d_cited) - np.mean(d_all)) / pooled)
            if pooled > 1e-12 else float("nan"))

    # ---- seam and floor --------------------------------------------------
    seam = [abs(_f(samples[s]["regions"][k].get("delta_seam")))
            for s in sids for k in samples[s]["regions"]]
    seam = [v for v in seam if np.isfinite(v)]
    res["seam_abs"] = float(np.mean(seam)) if seam else float("nan")
    res["seam_abs_sd"] = float(np.std(seam)) if seam else float("nan")
    if res.get("seam_abs_sd") and res["seam_abs_sd"] > 1e-12:
        res["FS_in_seam_sd"] = float(fs / res["seam_abs_sd"])

    floor = [_f(samples[s]["regions"][k].get("p_floor"))
             for s in sids for k in samples[s]["regions"]
             if "p_floor" in samples[s]["regions"][k]]
    floor = [v for v in floor if np.isfinite(v)]
    if floor:
        res["floor_p_mean"] = float(np.mean(floor))
        res["floor_frac_flagged"] = float(np.mean([v >= tau for v in floor]))

    # ---- reverse: CR vs prior -------------------------------------------
    def cr_at(t: float) -> Dict[str, Any]:
        hits, hit_groups, injected = [], [], []
        for s in sids:
            for k, e in samples[s]["regions"].items():
                p_rev = _f(e.get("p_reverse"))
                if not np.isfinite(p_rev) or p_rev < t:
                    continue
                cr_dist = e.get("cite_reverse") or {}
                if not cr_dist:
                    continue
                ks = list(samples[s]["regions"].keys())
                pick = max(_renorm(cr_dist, ks), key=lambda kk: _renorm(cr_dist, ks)[kk])
                hits.append(1.0 if pick == k else 0.0)
                hit_groups.append(samples[s]["pair_id"])
                injected.append(k)
        if not hits:
            return {"CR": float("nan"), "CR_lo": float("nan"),
                    "CR_hi": float("nan"), "CR_prior": float("nan"),
                    "CR_minus_prior": float("nan"), "n_reverse_flipped": 0}
        cr, lo, hi = C.bootstrap_ci(hits, hit_groups, n_boot, seed=seed)
        cnt = Counter(injected)
        prior = max(cnt.values()) / len(injected)
        return {"CR": cr, "CR_lo": lo, "CR_hi": hi, "CR_prior": float(prior),
                "CR_minus_prior": float(cr - prior),
                "n_reverse_flipped": len(hits),
                "injected_distribution": {k: v / len(injected)
                                          for k, v in cnt.items()}}

    main_cr = cr_at(tau)
    res.update(main_cr)
    res["CR_by_tau"] = {f"{t:g}": cr_at(t) for t in taus}

    # ---- verdict ---------------------------------------------------------
    res["faithful_forward"] = bool(np.isfinite(fs_lo) and fs_lo > 0)
    res["faithful_reverse"] = bool(
        np.isfinite(main_cr.get("CR_lo", float("nan")))
        and np.isfinite(main_cr.get("CR_prior", float("nan")))
        and main_cr["CR_lo"] > main_cr["CR_prior"])
    res["faithful"] = bool(res["faithful_forward"] and res["faithful_reverse"])

    # ---- per region ------------------------------------------------------
    per_region: Dict[str, Any] = {}
    pvals, names = [], []
    for k in region_names:
        vals, grp, raws, seams, floors = [], [], [], [], []
        for s in sids:
            e = samples[s]["regions"].get(k)
            if not e:
                continue
            v = _f(e.get("delta"))
            if np.isfinite(v):
                vals.append(v)
                grp.append(samples[s]["pair_id"])
                raws.append(_f(e.get("delta_raw")))
                seams.append(_f(e.get("delta_seam")))
                if "p_floor" in e:
                    floors.append(_f(e["p_floor"]))
        if not vals:
            continue
        d, lo, hi = C.bootstrap_ci(vals, grp, n_boot, seed=seed)
        p = C.bootstrap_p_two_sided(vals, grp, n_boot, seed=seed)
        per_region[k] = {
            "n": len(vals), "delta": d, "delta_lo": lo, "delta_hi": hi,
            "delta_raw": float(np.nanmean(raws)),
            "delta_seam": float(np.nanmean(seams)),
            "floor_p": float(np.nanmean(floors)) if floors else None,
            "p_boot": p,
            "cited_frac": float(np.mean([cited[s] == k for s in sids])),
        }
        pvals.append(p)
        names.append(k)
    for k, sig in zip(names, C.holm_bonferroni(pvals)):
        per_region[k]["significant_holm"] = bool(sig)
    res["per_region"] = per_region

    # ---- confusion: cited region vs region with the largest effect --------
    conf: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in sids:
        c = cited[s]
        if not c:
            continue
        ks = list(samples[s]["regions"])
        vals = {k: _f(samples[s]["regions"][k].get("delta")) for k in ks}
        vals = {k: v for k, v in vals.items() if np.isfinite(v)}
        if not vals:
            continue
        conf[c][max(vals, key=vals.get)] += 1
    res["confusion"] = {k: dict(v) for k, v in conf.items()}

    # ---- citation entropy ------------------------------------------------
    ents, ent_fs = [], []
    for s in sids:
        ks = list(samples[s]["regions"])
        d = _renorm(samples[s]["cite_orig"], ks) if samples[s]["cite_orig"] else None
        if not d:
            continue
        e = C.entropy_norm(list(d.values()))
        if np.isfinite(e):
            ents.append(e)
            ent_fs.append(fs_of(samples[s], cited[s]))
    res["citation_entropy"] = {
        "mean": float(np.mean(ents)) if ents else float("nan"),
        "sd": float(np.std(ents)) if ents else float("nan"),
        "corr_with_FS": C.spearman(ents, ent_fs) if len(ents) > 3 else float("nan"),
        "n": len(ents),
    }

    # ---- optional blocks -------------------------------------------------
    inp = inpaint_comparison(samples, sids, cited, n_boot, seed)
    if inp:
        res["inpaint"] = inp
    ps = prompt_stability(samples, sids, cited)
    if ps:
        res["prompt_stability"] = ps
    px = proxy_validation(samples, sids, cited, n_boot, seed)
    if px:
        res["proxies"] = px
    ag = citation_agreement(samples, sids)
    if ag:
        res["citation_agreement"] = ag
    bs = blind_spot(samples, sids, cited)
    if bs:
        res["blind_spot"] = bs
    un = union_additivity(samples, sids)
    if un:
        res["union"] = un
    if localization:
        loc = localization_crosstab(samples, sids, cited, localization,
                                    n_boot, seed)
        if loc:
            res["localization"] = loc

    return res


# --------------------------------------------------------------------------
# optional analysis blocks
# --------------------------------------------------------------------------

def inpaint_comparison(samples, sids, cited, n_boot=N_BOOT, seed=0
                       ) -> Optional[Dict[str, Any]]:
    """
    Compare the inpainting counterfactual with the paired-frame one.  Three
    quantities are reported: correlation of the effects, sign disagreement,
    and whether inpainting an authentic frame raises the manipulation
    probability, i.e. introduces forgery evidence by itself.  The last is the
    decisive validity criterion.
    """
    d_inp, d_raw, d_true = [], [], []
    rise, rise_groups = [], []
    for s in sids:
        for k, e in samples[s]["regions"].items():
            if "delta_inpaint" not in e:
                continue
            di, dr, dt = (_f(e["delta_inpaint"]), _f(e.get("delta_raw")),
                          _f(e.get("delta")))
            if np.isfinite(di) and np.isfinite(dt):
                d_inp.append(di)
                d_raw.append(dr)
                d_true.append(dt)
        pr = samples[s]["p_orig_real"]
        vals = [_f(e.get("p_inpaint_real")) for e in samples[s]["regions"].values()
                if "p_inpaint_real" in e]
        vals = [v for v in vals if np.isfinite(v)]
        if vals and np.isfinite(pr):
            rise.append(float(np.mean(vals)) - pr)
            rise_groups.append(samples[s]["pair_id"])
    if not d_inp:
        return None
    disagree = [1.0 if (np.sign(a) != np.sign(b)) else 0.0
                for a, b in zip(d_inp, d_true) if abs(b) > 0.01]
    r, lo, hi = (C.bootstrap_ci(rise, rise_groups, n_boot, seed=seed)
                 if rise else (float("nan"),) * 3)
    fs_inp = []
    for s in sids:
        v = fs_of(samples[s], cited[s], delta_field="delta_inpaint")
        if np.isfinite(v):
            fs_inp.append(v)
    return {
        "n_pairs": len(d_inp),
        "corr_inpaint_vs_raw": C.pearson(d_inp, d_raw),
        "corr_inpaint_vs_delta": C.pearson(d_inp, d_true),
        "spearman_inpaint_vs_delta": C.spearman(d_inp, d_true),
        "sign_disagreement": float(np.mean(disagree)) if disagree else float("nan"),
        "n_sign_tested": len(disagree),
        "p_rise_on_real": r, "p_rise_on_real_lo": lo, "p_rise_on_real_hi": hi,
        "FS_inpaint": float(np.mean(fs_inp)) if fs_inp else float("nan"),
        "manufactures_evidence": bool(np.isfinite(lo) and lo > 0),
    }


def prompt_stability(samples, sids, cited) -> Optional[Dict[str, Any]]:
    """Stability of the cited region and of p(manipulated) across prompt
    paraphrases."""
    prim_agree, pair_agree, p_sd, fs_by_variant = [], [], [], defaultdict(list)
    any_v = False
    for s in sids:
        variants = samples[s]["cite_variants"]
        if not variants:
            continue
        any_v = True
        ks = list(samples[s]["regions"])
        prim = cited[s]
        picks = []
        for c in variants:
            d = _renorm(c, ks)
            picks.append(max(d, key=d.get))
        if prim:
            prim_agree.extend([1.0 if p == prim else 0.0 for p in picks])
        allp = ([prim] if prim else []) + picks
        if len(allp) > 1:
            n = 0
            agree = 0
            for i in range(len(allp)):
                for j in range(i + 1, len(allp)):
                    n += 1
                    agree += 1 if allp[i] == allp[j] else 0
            pair_agree.append(agree / n)
        pv = [samples[s]["p_orig_fake"]] + list(samples[s]["p_variants"])
        pv = [v for v in pv if np.isfinite(_f(v))]
        if len(pv) > 1:
            p_sd.append(float(np.std(pv)))
        for i, pick in enumerate(picks):
            v = fs_of(samples[s], pick)
            if np.isfinite(v):
                fs_by_variant[i + 1].append(v)
    if not any_v:
        return None
    return {
        "argmax_agreement_with_primary":
            float(np.mean(prim_agree)) if prim_agree else float("nan"),
        "mean_pairwise_agreement":
            float(np.mean(pair_agree)) if pair_agree else float("nan"),
        "p_fake_sd_across_wordings":
            float(np.mean(p_sd)) if p_sd else float("nan"),
        "FS_by_variant": {str(k): float(np.mean(v))
                          for k, v in sorted(fs_by_variant.items())},
        "n_samples": len(pair_agree),
    }


def proxy_validation(samples, sids, cited, n_boot=N_BOOT, seed=0
                     ) -> Optional[Dict[str, Any]]:
    """
    Deployable-proxy validation.  For each proxy: whether FS computed from the
    proxy tracks the paired-frame FS, whether it ranks regions the same way,
    and its own floor (how much it inflates p on authentic frames by itself).
    """
    out: Dict[str, Any] = {}
    for pname in ("blur", "noise", "shuffle", "selfcheck", "inpaint"):
        field = f"delta_{pname}" if pname != "inpaint" else "delta_inpaint"
        fs_true, fs_proxy, groups, taus_ = [], [], [], []
        for s in sids:
            has = any(field in e for e in samples[s]["regions"].values())
            if not has:
                continue
            ft = fs_of(samples[s], cited[s])
            fp = fs_of(samples[s], cited[s], delta_field=field)
            if np.isfinite(ft) and np.isfinite(fp):
                fs_true.append(ft)
                fs_proxy.append(fp)
                groups.append(samples[s]["pair_id"])
            ks = list(samples[s]["regions"])
            a = [_f(samples[s]["regions"][k].get("delta")) for k in ks]
            b = [_f(samples[s]["regions"][k].get(field)) for k in ks]
            if sum(np.isfinite(a)) > 2 and sum(np.isfinite(b)) > 2:
                taus_.append(C.kendall_tau(a, b))
        if len(fs_true) < 4:
            continue
        # Floor: the effect of the perturbation on an authentic frame.  A
        # proxy that correlates with FS but inflates p on authentic frames
        # introduces the evidence it is meant to measure.
        rises, rise_groups = [], []
        for s in sids:
            pr = samples[s]["p_orig_real"]
            vals = [_f(e.get(f"p_{pname}_real"))
                    for e in samples[s]["regions"].values()
                    if f"p_{pname}_real" in e]
            vals = [v for v in vals if np.isfinite(v)]
            if vals and np.isfinite(pr):
                rises.append(float(np.mean(vals)) - pr)
                rise_groups.append(samples[s]["pair_id"])
        fl, fl_lo, fl_hi = (C.bootstrap_ci(rises, rise_groups, n_boot, seed=seed)
                            if len(rises) >= 4 else (float("nan"),) * 3)
        out[pname] = {
            "n": len(fs_true),
            "corr_fs": C.spearman(fs_proxy, fs_true),
            "kendall_delta": float(np.nanmean(taus_)) if taus_ else float("nan"),
            "FS_proxy": float(np.mean(fs_proxy)),
            "FS_true": float(np.mean(fs_true)),
            "floor_p_rise": fl, "floor_p_rise_lo": fl_lo, "floor_p_rise_hi": fl_hi,
            "contaminating": bool(np.isfinite(fl_lo) and fl_lo > 0),
        }
    return out or None


def citation_agreement(samples, sids) -> Optional[Dict[str, Any]]:
    """Agreement between stated, gradient-attributed, overlay-elicited and
    causal (largest-effect) regions."""
    have_attr = any(samples[s].get("cite_attr") for s in sids)
    have_ovl = any(samples[s].get("cite_overlay") for s in sids)
    if not (have_attr or have_ovl):
        return None
    rows = {"stated_vs_causal": [], "attr_vs_causal": [], "stated_vs_attr": [],
            "overlay_vs_causal": [], "stated_vs_overlay": []}
    venn = Counter()
    for s in sids:
        ks = list(samples[s]["regions"])
        if len(ks) < 2:
            continue
        deltas = {k: _f(samples[s]["regions"][k].get("delta")) for k in ks}
        deltas = {k: v for k, v in deltas.items() if np.isfinite(v)}
        if not deltas:
            continue
        causal = max(deltas, key=deltas.get)
        stated = cited_region(samples[s])
        attr = (max(_renorm(samples[s]["cite_attr"], ks),
                    key=lambda k: _renorm(samples[s]["cite_attr"], ks)[k])
                if samples[s].get("cite_attr") else None)
        ovl = (max(_renorm(samples[s]["cite_overlay"], ks),
                   key=lambda k: _renorm(samples[s]["cite_overlay"], ks)[k])
               if samples[s].get("cite_overlay") else None)
        if stated:
            rows["stated_vs_causal"].append(1.0 if stated == causal else 0.0)
        if attr:
            rows["attr_vs_causal"].append(1.0 if attr == causal else 0.0)
        if stated and attr:
            rows["stated_vs_attr"].append(1.0 if stated == attr else 0.0)
        if ovl:
            rows["overlay_vs_causal"].append(1.0 if ovl == causal else 0.0)
        if stated and ovl:
            rows["stated_vs_overlay"].append(1.0 if stated == ovl else 0.0)
        # Three-set membership over samples, for the Venn diagram:
        #   S = the stated citation is the causal region
        #   A = the attributed (gradient) citation is the causal region
        #   O = the overlay-elicited citation is the causal region
        # Only samples that have all three elicitations enter, so the cells
        # of the diagram are comparable.
        if stated and attr and ovl:
            venn[(stated == causal, attr == causal, ovl == causal)] += 1
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in rows.items()}
    out["n"] = max(len(v) for v in rows.values())
    # Keys such as "SAO", "S.O", "...": S stated, A attributed, O overlay,
    # each letter present when that elicitation matched the causal region.
    out["venn"] = {
        "".join(c if flag else "." for c, flag in zip("SAO", k)): v
        for k, v in venn.items()}
    out["venn_n"] = int(sum(venn.values()))
    return out


def blind_spot(samples, sids, cited) -> Optional[Dict[str, Any]]:
    """
    Encoder blind-spot decomposition.  Splits (sample, region) cells into
    encoder blindness (the visual tokens did not move under the counterfactual)
    and language-side confabulation (they moved, but the verdict did not).
    """
    shifts, deltas = [], []
    for s in sids:
        for e in samples[s]["regions"].values():
            if "enc_shift_in" in e:
                shifts.append(_f(e["enc_shift_in"]))
                deltas.append(_f(e.get("delta")))
    ok = [(a, b) for a, b in zip(shifts, deltas)
          if np.isfinite(a) and np.isfinite(b)]
    if len(ok) < 10:
        return None
    sh = np.array([a for a, _ in ok])
    de = np.array([b for _, b in ok])
    s_lo = float(np.percentile(sh, 10))
    d_small = float(np.percentile(np.abs(de), 50))
    blind = float(np.mean((sh <= s_lo) & (np.abs(de) <= d_small)))
    confab = float(np.mean((sh > s_lo) & (np.abs(de) <= d_small)))
    faith = float(np.mean((sh > s_lo) & (np.abs(de) > d_small)))
    out_ctrl = [_f(e.get("enc_shift_out")) for s in sids
                for e in samples[s]["regions"].values() if "enc_shift_out" in e]
    out_ctrl = [v for v in out_ctrl if np.isfinite(v)]
    return {
        "n": len(ok),
        "encoder_blind_frac": blind,
        "language_confabulation_frac": confab,
        "faithful_frac": faith,
        "spearman_shift_vs_delta": C.spearman(sh, de),
        "shift_p10": s_lo,
        "shift_outside_mean": float(np.mean(out_ctrl)) if out_ctrl else float("nan"),
    }


def union_additivity(samples, sids) -> Optional[Dict[str, Any]]:
    """Compositional test: compare delta(k1 u k2) with delta(k1) + delta(k2)."""
    obs, add = [], []
    for s in sids:
        u = samples[s].get("union")
        if not u or not u.get("regions"):
            continue
        d_union = _f(u["p_identity"]) - _f(u["p_real"])
        parts = [_f(samples[s]["regions"][k].get("delta"))
                 for k in u["regions"] if k in samples[s]["regions"]]
        parts = [p for p in parts if np.isfinite(p)]
        if np.isfinite(d_union) and len(parts) == len(u["regions"]):
            obs.append(d_union)
            add.append(float(sum(parts)))
    if len(obs) < 4:
        return None
    return {
        "n": len(obs),
        "delta_union_observed": float(np.mean(obs)),
        "delta_sum_of_parts": float(np.mean(add)),
        "superadditivity": float(np.mean(np.array(obs) - np.array(add))),
        "corr": C.pearson(obs, add),
    }


def localization_crosstab(samples, sids, cited, loc, n_boot=N_BOOT, seed=0
                          ) -> Optional[Dict[str, Any]]:
    """
    Localization-versus-faithfulness cross-tabulation: whether the cited region
    is the one where pixels changed (correct) and whether it is the one that
    drove the verdict (causal).  The four cells separate the two properties.
    """
    cc, sc, quad = [], [], Counter()
    fs_vals = []
    for s in sids:
        rec = loc.get(s)
        c = cited[s]
        if not rec or not c:
            continue
        f = rec.get("f") or {}
        ks = list(samples[s]["regions"])
        f = {k: _f(v) for k, v in f.items() if k in ks}
        if not f or max(f.values()) <= 0:
            continue
        gt = max(f, key=f.get)
        correct = 1.0 if c == gt else 0.0
        cc.append(correct)
        sc.append(float(sum(_renorm(samples[s]["cite_orig"], ks)[k] * f.get(k, 0.0)
                            for k in ks) / max(f.values())))
        rk = rank_of_cited(samples[s], c)
        fsv = fs_of(samples[s], c)
        fs_vals.append(fsv)
        causal = bool(np.isfinite(fsv) and fsv > 0 and np.isfinite(rk) and rk <= 2)
        quad[("cc1" if correct else "cc0") +
             ("_faithful" if causal else "_unfaithful")] += 1
    if not cc:
        return None
    return {
        "n": len(cc),
        "CC": float(np.mean(cc)),
        "SC": float(np.mean(sc)),
        "corr_SC_FS": C.spearman(sc, fs_vals),
        "crosstab": {k: quad.get(k, 0) for k in
                     ("cc0_faithful", "cc1_faithful",
                      "cc0_unfaithful", "cc1_unfaithful")},
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def detector_level_holm(results: List[Dict[str, Any]], alpha: float = 0.05
                        ) -> List[Dict[str, Any]]:
    """
    Holm-Bonferroni correction across the detector zoo.

    Each detector's faithfulness verdict is a separate hypothesis test, so
    reporting every detector that clears alpha = 0.05 inflates the family-wise
    error rate.  The per-region Holm correction inside `analyse` corrects
    within a detector, not across detectors.

    The family contains one result per detector: the result tagged "main"
    when one exists, otherwise the first result (in input order) for that
    detector.  Ablation tags of the same detector are therefore not counted as
    additional hypotheses.

    Adds to every result:
        FS_significant_holm_detectors   FS_p survives the correction (None for
                                        results outside the family)
        faithful_holm                   the corrected verdict for family
                                        members; the uncorrected `faithful`
                                        for results outside the family
        holm_family_size                number of detectors in the family

    The uncorrected `faithful` is kept so the difference is visible.
    """
    by_det: Dict[Any, Dict[str, Any]] = {}
    for r in results:
        if not np.isfinite(_f(r.get("FS_p"))):
            continue
        det = r.get("detector")
        cur = by_det.get(det)
        if cur is None or (r.get("tag") == "main" and cur.get("tag") != "main"):
            by_det[det] = r
    fam = list(by_det.values())
    if not fam:
        return results
    flags = C.holm_bonferroni([_f(r["FS_p"]) for r in fam], alpha)
    for r, sig in zip(fam, flags):
        r["FS_significant_holm_detectors"] = bool(sig)
        r["holm_family_size"] = len(fam)
        r["faithful_holm"] = bool(
            sig and r.get("faithful_forward") and r.get("faithful_reverse"))
    for r in results:
        r.setdefault("FS_significant_holm_detectors", None)
        r.setdefault("faithful_holm", r.get("faithful"))
        r.setdefault("holm_family_size", len(fam))
    return results


def additivity(groups: Dict[Tuple[str, str], List[str]], seed: int = 0,
               n_boot: int = N_BOOT, vocab: Optional[str] = None
               ) -> Dict[str, Any]:
    """
    Additivity of seam and content effects.

    The seam correction delta = delta_raw - delta_seam is an unbiased estimate
    of the content effect only if the seam and the content contribute
    additively.  Dilation radius varies the seam while leaving the content
    largely unchanged, so under additivity

        delta_raw(r_hi) - delta_raw(r_lo)  ~=  delta_seam(r_hi) - delta_seam(r_lo)

    and the residual between the two estimates the bias in delta.

    Requires runs at two or more dilation radii, identified by tags of the
    form dilate_<r>.

    Interpretation caveat.  The test assumes dilation varies the seam while
    leaving the content alone, which holds only once the mask already covers
    the whole manipulated area.  At r = 0 the mask is the bare parsed region,
    which can be smaller than the manipulation, so growing it adds content as
    well as seam and the residual is large for a reason unrelated to
    additivity.  The two cases are distinguished as follows:

      * content-coverage artifact: the interaction sits in delta_raw while
        delta_seam barely moves, and `delta` itself grows with r.  Comparing
        two radii that both cover the manipulation (e.g. 7 vs 11) shows
        whether the residual collapses.
      * non-additivity: delta_seam moves substantially with r and the residual
        persists at radii where `delta` has plateaued.  In that case the
        subtraction is biased.

    `per_radius[...]["seam_share"]` (mean |delta_seam| / mean |delta_raw|)
    indicates which case applies.
    """
    import re as _re

    by_det: Dict[str, Dict[int, List[str]]] = defaultdict(dict)
    for (det, tag), paths in groups.items():
        m = _re.match(r"^dilate_(\d+)$", str(tag))
        if m:
            by_det[det][int(m.group(1))] = paths

    out: Dict[str, Any] = {}
    for det, by_r in sorted(by_det.items()):
        radii = sorted(by_r)
        if len(radii) < 2:
            continue
        lo, hi = radii[0], radii[-1]
        per_r: Dict[int, Dict[str, Any]] = {}
        cell: Dict[int, Dict[Tuple[str, str], Tuple[float, float]]] = {}
        for r in radii:
            rows, _m = load_rows(by_r[r])
            samples = samples_from_rows(rows)
            raws, seams, deltas = [], [], []
            cell[r] = {}
            for sid, s in samples.items():
                for k, e in s["regions"].items():
                    a, b = _f(e.get("delta_raw")), _f(e.get("delta_seam"))
                    if np.isfinite(a) and np.isfinite(b):
                        raws.append(a)
                        seams.append(b)
                        deltas.append(_f(e.get("delta")))
                        cell[r][(sid, k)] = (a, b)
            per_r[r] = {
                "n": len(raws),
                "delta_raw": float(np.mean(raws)) if raws else float("nan"),
                "delta_seam": float(np.mean(seams)) if seams else float("nan"),
                "delta": float(np.mean(deltas)) if deltas else float("nan"),
                "seam_share": (float(np.mean(np.abs(seams)))
                               / max(1e-9, float(np.mean(np.abs(raws)))))
                if raws else float("nan"),
            }

        # Paired residual over the (sample, region) cells present at both radii.
        common = set(cell[lo]) & set(cell[hi])
        resid, groups_ = [], []
        for key in sorted(common):
            d_raw = cell[hi][key][0] - cell[lo][key][0]
            d_seam = cell[hi][key][1] - cell[lo][key][1]
            resid.append(d_raw - d_seam)
            groups_.append(key[0])
        res, res_lo, res_hi = (C.bootstrap_ci(resid, groups_, n_boot, seed=seed)
                               if len(resid) >= 4 else (float("nan"),) * 3)
        out[det] = {
            "radii": radii, "r_lo": lo, "r_hi": hi,
            "per_radius": {str(k): v for k, v in per_r.items()},
            "n_paired_cells": len(resid),
            "interaction_delta_raw": per_r[hi]["delta_raw"] - per_r[lo]["delta_raw"],
            "interaction_delta_seam": per_r[hi]["delta_seam"] - per_r[lo]["delta_seam"],
            "residual": res, "residual_lo": res_lo, "residual_hi": res_hi,
            "additive": bool(np.isfinite(res_lo) and res_lo <= 0 <= res_hi),
        }
    return out


def analyse_all(
    raw_dirs: Sequence[str],
    out_dir: str,
    seed: int = 0,
    coarse: bool = False,
    by_method: bool = False,
    split: str = "all",
    taus: Sequence[float] = DEFAULT_TAUS,
    tau: float = TAU_MAIN,
    vocab: str = "face8",
    n_boot: int = N_BOOT,
    localization_path: str = "",
) -> Dict[str, Any]:
    R.set_vocab(vocab)
    os.makedirs(out_dir, exist_ok=True)
    groups = group_raw_files(raw_dirs)
    if not groups:
        raise FileNotFoundError(f"no raw_*.json found under {list(raw_dirs)}")

    loc: Optional[Dict[str, Dict]] = None
    if localization_path and os.path.exists(localization_path):
        blob = C.load_json(localization_path)
        loc = {r["sample_id"]: r for r in blob.get("records", [])}
        print(f"[m6] localization: {len(loc)} records")

    results: List[Dict[str, Any]] = []
    by_method_results: List[Dict[str, Any]] = []
    for (det, tag), paths in sorted(groups.items()):
        rows, meta = load_rows(paths)
        if split != "all":
            rows = [r for r in rows if r.get("split") == split]
        if not rows:
            continue
        res = analyse(rows, seed=seed, coarse=coarse, taus=taus, tau=tau,
                      vocab=vocab, n_boot=n_boot, localization=loc, meta=meta)
        res["source_files"] = [os.path.basename(p) for p in paths]
        results.append(res)
        print(f"[m6] {det} [{tag}] {res.get('granularity')}: "
              f"n={res.get('n_samples')} AUC={res.get('AUC'):.3f} "
              f"FS={res.get('FS'):.4f} [{res.get('FS_lo'):.4f},{res.get('FS_hi'):.4f}] "
              f"CR-prior={res.get('CR_minus_prior', float('nan')):.3f} "
              f"-> {'FAITHFUL' if res.get('faithful') else ('fwd only' if res.get('faithful_forward') else 'UNFAITHFUL')}",
              flush=True)

        if by_method:
            for m in sorted({r.get("method") for r in rows}):
                mrows = [r for r in rows if r.get("method") == m]
                if len(mrows) < 8:
                    continue
                mr = analyse(mrows, seed=seed, coarse=coarse, taus=taus,
                             tau=tau, vocab=vocab, n_boot=n_boot,
                             localization=loc, meta=meta)
                mr["method_slice"] = m
                by_method_results.append(mr)

    # Holm across the detector zoo, then the operator's additivity check.
    detector_level_holm(results)
    add = additivity(groups, seed=seed, n_boot=n_boot, vocab=vocab)
    if add:
        for det, blk in add.items():
            print(f"[m6] additivity {det}: residual={blk['residual']:+.4f} "
                  f"[{blk['residual_lo']:+.4f},{blk['residual_hi']:+.4f}] "
                  f"-> {'additive' if blk['additive'] else 'NON-ADDITIVE'}")

    n_corrected = sum(1 for r in results
                      if r.get("faithful") and not r.get("faithful_holm"))
    if n_corrected:
        print(f"[m6] {n_corrected} detector(s) lose the faithful verdict after "
              f"Holm correction across the zoo; faithful_holm is the "
              f"corrected verdict.")

    name = "metrics_coarse.json" if coarse else "metrics.json"
    payload = {"results": results, "additivity": add,
               "provenance": C.provenance(),
               "settings": {"seed": seed, "coarse": coarse, "split": split,
                            "tau": tau, "taus": list(taus), "vocab": vocab,
                            "n_boot": n_boot}}
    C.save_json(os.path.join(out_dir, name), payload, indent=1)
    print(f"[m6] -> {os.path.join(out_dir, name)}")
    if by_method_results:
        C.save_json(os.path.join(out_dir, "metrics_by_method.json"),
                    {"results": by_method_results,
                     "provenance": C.provenance()}, indent=1)
        print(f"[m6] -> {os.path.join(out_dir, 'metrics_by_method.json')}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m6_metrics")
    ap.add_argument("--raw", required=True, help="comma-separated run directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--by", default="", choices=["", "method"])
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--tau", type=float, default=TAU_MAIN)
    ap.add_argument("--taus", default="0.3,0.5,0.7")
    ap.add_argument("--vocab", default="face8", choices=["face8", "grid9"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--localization", default="")
    a = ap.parse_args(argv)

    analyse_all(
        [d for d in a.raw.split(",") if d.strip()], a.out, seed=a.seed,
        coarse=a.coarse, by_method=(a.by == "method"), split=a.split,
        taus=[float(t) for t in a.taus.split(",") if t.strip()], tau=a.tau,
        vocab=a.vocab, n_boot=a.n_boot, localization_path=a.localization,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
