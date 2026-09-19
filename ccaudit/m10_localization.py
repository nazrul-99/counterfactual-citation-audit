"""
ccaudit.m10_localization -- Module 10, ground-truth localization.

The paired frames make the manipulated area observable without annotation:
D = |fake - real| is the manipulation itself.  This gives citation
correctness (does the detector point where the pixels changed) to set beside
citation faithfulness (does the cited region drive the verdict), and their
cross-tabulation isolates citations that point at the right region without
that region driving the verdict.

Per sample the module writes:

    f[k]        manipulation fraction of region k: |M_gt n k| / |k|
    m[k]        manipulation intensity in region k: mean(D over k)
    iou[k]      |M_gt n k| / |M_gt u k|
    gt_region   argmax_k f[k]
    gt_dist     f normalised to a distribution

Design decisions
----------------
* gt_region uses the fraction, not the intensity.  Intensity favours whichever
  region happens to contain the most strongly changed pixels, while fraction
  asks how much of the region was touched, which is what a citation claims.
  Fraction also prevents large regions from winning by size alone; `skin`
  covers roughly half the crop and would otherwise dominate every sample.
* Results are aggregated per manipulation method, because methods differ in
  which regions they alter.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import common as C
from . import regions as R
from .m2_parse import manipulation_mask


def localize_sample(rec: Dict[str, Any], sigma: float = 2.0,
                    vocab: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Ground-truth localization for one index record.  Returns None when the
    images cannot be read, their shapes disagree, or fewer than 16 pixels
    changed.
    """
    vocab = vocab or rec.get("vocab") or R.active_vocab()
    try:
        real = C.imread(rec["real"])
        fake = C.imread(rec["fake"])
        lab = C.imread(rec["lab"], 0)
    except Exception:
        return None
    if real.shape[:2] != fake.shape[:2] or lab.shape[:2] != real.shape[:2]:
        return None

    bg = (lab == R.BACKGROUND)
    mgt, d = manipulation_mask(real, fake, sigma=sigma, bg_mask=bg)
    mb = mgt.astype(bool)
    n_changed = int(mb.sum())
    if n_changed < 16:
        return None

    names = R.get_vocab(vocab)
    f: Dict[str, float] = {}
    m: Dict[str, float] = {}
    iou: Dict[str, float] = {}
    for nm in names:
        reg = (lab == R.rid(nm, vocab))
        area = int(reg.sum())
        if area == 0:
            f[nm] = 0.0
            m[nm] = 0.0
            iou[nm] = 0.0
            continue
        inter = float((reg & mb).sum())
        f[nm] = inter / area
        m[nm] = float(d[reg].mean())
        union = float((reg | mb).sum())
        iou[nm] = inter / union if union > 0 else 0.0

    tot = sum(f.values())
    gt_dist = ({k: v / tot for k, v in f.items()} if tot > 0
               else {k: 1.0 / len(names) for k in names})
    gt_region = max(f, key=f.get) if tot > 0 else None

    return {
        "sample_id": rec["sample_id"],
        "pair_id": rec["pair_id"],
        "method": rec["method"],
        "split": rec.get("split") or C.split_of(rec["pair_id"]),
        "gt_region": gt_region,
        "gt_dist": {k: round(v, 5) for k, v in gt_dist.items()},
        "f": {k: round(v, 5) for k, v in f.items()},
        "m": {k: round(v, 3) for k, v in m.items()},
        "iou": {k: round(v, 5) for k, v in iou.items()},
        "area_changed_frac": round(n_changed / float(mb.size), 5),
        "changed_inside_vocab_frac": round(
            float((mb & (lab != R.BACKGROUND)).sum()) / n_changed, 5),
    }


def run(index_path: str, out_dir: str, split: str = "all",
        limit: int = 0, seed: int = 0, sigma: float = 2.0,
        time_budget_min: float = 0.0) -> Dict[str, Any]:
    """Localize every record of the requested split and write
    `localization.json` to `out_dir`.  Returns the written payload."""
    os.makedirs(out_dir, exist_ok=True)
    records, meta = C.load_index(index_path)
    vocab = meta.get("vocab", "face8")
    R.set_vocab(vocab)
    records = C.filter_split(records, split)
    if limit:
        records = C.limit_samples(records, limit, seed)

    budget = C.Budget(time_budget_min, label="m10")
    t0 = time.time()
    out: List[Dict[str, Any]] = []
    skipped = 0
    for i, rec in enumerate(records):
        if budget.expired:
            print(f"[m10] STOPPED EARLY on time budget at {i}/{len(records)}")
            break
        r = localize_sample(rec, sigma, vocab)
        if r is None:
            skipped += 1
            continue
        out.append(r)
        if (i + 1) % 500 == 0:
            print(f"[m10] {i+1}/{len(records)}", flush=True)

    gt_counts: Dict[str, int] = {}
    for r in out:
        if r["gt_region"]:
            gt_counts[r["gt_region"]] = gt_counts.get(r["gt_region"], 0) + 1
    by_method: Dict[str, Dict[str, int]] = {}
    for r in out:
        d = by_method.setdefault(r["method"], {})
        if r["gt_region"]:
            d[r["gt_region"]] = d.get(r["gt_region"], 0) + 1

    payload = {
        "meta": {
            "index": os.path.abspath(index_path), "vocab": vocab,
            "split": split, "sigma": sigma, "n": len(out), "n_skipped": skipped,
            "gt_region_counts": gt_counts,
            "gt_region_counts_by_method": by_method,
            "mean_changed_frac": float(np.mean(
                [r["area_changed_frac"] for r in out])) if out else float("nan"),
            "mean_changed_inside_vocab": float(np.mean(
                [r["changed_inside_vocab_frac"] for r in out])) if out else float("nan"),
            "elapsed_min": (time.time() - t0) / 60.0,
            "code_hash": C.code_hash(),
        },
        "records": out,
    }
    path = os.path.join(out_dir, "localization.json")
    C.save_json(path, payload, indent=None)

    print(C.banner("Module 10 summary"))
    print(f"  samples localized     {len(out)}  (skipped {skipped})")
    print(f"  mean changed area     {payload['meta']['mean_changed_frac']:.3f}")
    print(f"  changed inside vocab  {payload['meta']['mean_changed_inside_vocab']:.3f}")
    print(f"  gt_region counts      {gt_counts}")
    for meth, d in sorted(by_method.items()):
        print(f"    {meth:<18s} {d}")
    print(f"  -> {path}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m10_localization")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    a = ap.parse_args(argv)
    run(a.index, a.out, a.split, a.limit, a.seed, a.sigma, a.time_budget_min)
    return 0


if __name__ == "__main__":
    sys.exit(main())
