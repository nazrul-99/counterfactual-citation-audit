"""
ccaudit.m11_proxy -- Module 11, deployable proxy validation.

FS requires the authentic twin frame, which is unavailable at deployment.
This module asks which perturbation of the suspect image alone best predicts
the true FS, and what that perturbation does to an authentic frame by itself.

The proxy conditions are produced by m5 (`--proxy blur,noise,shuffle,
selfcheck`), which also measures each proxy's floor.  This module ranks them.

Three validations:

  (a) sample-level Spearman between FS computed from the proxy and true FS;
  (b) region-ranking agreement, Kendall tau between Delta_proxy and Delta,
      averaged over samples;
  (c) decision agreement across the detector zoo: whether replacing FS by
      FS_proxy would reach the same faithful / not-faithful verdict for each
      detector.  A proxy can correlate well within samples and still change
      the per-detector conclusion.

And the disqualifier:

  (d) floor: the mean rise in p on an authentic frame when the proxy is
      applied.  A proxy with a positive floor whose CI excludes zero
      manufactures the evidence it claims to measure.  Proxies are ranked by
      correlation only among those that pass the floor check; the rest are
      reported as disqualified.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import common as C
from . import m6_metrics as M6
from . import regions as R

PROXY_NAMES = ("blur", "noise", "shuffle", "selfcheck", "inpaint")


def run(raw_dirs: Sequence[str], out_dir: str, vocab: str = "face8",
        seed: int = 0, n_boot: int = 1000, split: str = "all"
        ) -> Dict[str, Any]:
    """Validate and rank the proxies found in `raw_dirs`; write
    `proxies.json` to `out_dir` and return its content."""
    R.set_vocab(vocab)
    os.makedirs(out_dir, exist_ok=True)

    per_detector: List[Dict[str, Any]] = []
    # fs_proxy_by_det[proxy][detector] = FS_proxy, used for decision agreement.
    fs_true_by_det: Dict[str, float] = {}
    fs_proxy_by_det: Dict[str, Dict[str, float]] = {p: {} for p in PROXY_NAMES}

    for (det, tag), paths in sorted(M6.group_raw_files(raw_dirs).items()):
        rows, _m = M6.load_rows(paths)
        if split != "all":
            rows = [r for r in rows if r.get("split") == split]
        if not rows:
            continue
        samples = M6.samples_from_rows(rows)
        sids = sorted(samples)
        cited = {s: M6.cited_region(samples[s]) for s in sids}
        blocks = M6.proxy_validation(samples, sids, cited, n_boot, seed)
        if not blocks:
            continue
        entry = {"detector": det, "tag": tag, "n_samples": len(sids),
                 "proxies": blocks}
        per_detector.append(entry)

        fs_vals = [M6.fs_of(samples[s], cited[s]) for s in sids]
        fs_vals = [v for v in fs_vals if np.isfinite(v)]
        if fs_vals:
            fs_true_by_det[f"{det}|{tag}"] = float(np.mean(fs_vals))
        for pname, b in blocks.items():
            fs_proxy_by_det.setdefault(pname, {})[f"{det}|{tag}"] = b["FS_proxy"]

        for pname, b in sorted(blocks.items()):
            flag = "CONTAMINATING" if b.get("contaminating") else "clean"
            print(f"[m11] {det} [{tag}] {pname:10s} rho(FS)="
                  f"{b['corr_fs']:+.3f} tau={b['kendall_delta']:+.3f} "
                  f"floor={b.get('floor_p_rise', float('nan')):+.4f} {flag}",
                  flush=True)

    if not per_detector:
        print("[m11] no proxy rows found; run m5 with "
              "--proxy blur,noise,shuffle[,selfcheck]")
        path = os.path.join(out_dir, "proxies.json")
        C.save_json(path, {"results": [], "provenance": C.provenance()}, indent=1)
        return {"results": []}

    # ---- (c) decision agreement across the detector zoo ------------------
    dets = sorted(fs_true_by_det)
    summary: Dict[str, Any] = {}
    for pname in PROXY_NAMES:
        corrs, floors, taus, contam = [], [], [], []
        for e in per_detector:
            b = e["proxies"].get(pname)
            if not b:
                continue
            corrs.append(b["corr_fs"])
            taus.append(b["kendall_delta"])
            if np.isfinite(b.get("floor_p_rise", float("nan"))):
                floors.append(b["floor_p_rise"])
            contam.append(bool(b.get("contaminating")))
        if not corrs:
            continue                      # this proxy was never run
        entry: Dict[str, Any] = {
            "n_detectors": len(corrs),
            "mean_corr_fs": float(np.nanmean(corrs)),
            "mean_kendall": float(np.nanmean(taus)) if taus else float("nan"),
            "mean_floor_rise": float(np.nanmean(floors)) if floors else float("nan"),
            "contaminating_in_any_detector": bool(any(contam)),
        }
        # Decision agreement needs at least two detectors.  With fewer, only
        # that field is marked unavailable, so (a), (b) and the floor are
        # still reported for a single-detector run.
        got = fs_proxy_by_det.get(pname) or {}
        common = [d for d in dets if d in got]
        if len(common) >= 2:
            true_v = [fs_true_by_det[d] for d in common]
            prox_v = [got[d] for d in common]
            entry["rank_corr_over_detectors"] = C.spearman(prox_v, true_v)
            entry["verdict_sign_agreement"] = float(np.mean(
                [1.0 if np.sign(a) == np.sign(b) else 0.0
                 for a, b in zip(true_v, prox_v)]))
        else:
            entry["rank_corr_over_detectors"] = float("nan")
            entry["verdict_sign_agreement"] = float("nan")
            entry["note"] = ("decision agreement needs >= 2 detectors; "
                             f"only {len(common)} available")
        summary[pname] = entry

    # ---- ranking by correlation among proxies that pass the floor check
    ranked = sorted(
        [(k, v) for k, v in summary.items()],
        key=lambda kv: (kv[1]["contaminating_in_any_detector"],
                        -(kv[1]["mean_corr_fs"]
                          if np.isfinite(kv[1]["mean_corr_fs"]) else -9)))
    for i, (k, v) in enumerate(ranked, start=1):
        v["rank"] = i
        v["eligible"] = not v["contaminating_in_any_detector"]

    out = {"results": per_detector, "summary": summary,
           "ranking": [k for k, _v in ranked],
           "best_eligible": next((k for k, v in ranked if v["eligible"]), None),
           "provenance": C.provenance()}
    path = os.path.join(out_dir, "proxies.json")
    C.save_json(path, out, indent=1)

    print(C.banner("Module 11 summary -- proxy ranking"))
    print(f"  {'proxy':<12}{'rho(FS)':>10}{'kendall':>10}{'floor':>10}"
          f"{'sign agr':>10}  status")
    for k, v in ranked:
        print(f"  {k:<12}{v['mean_corr_fs']:>10.3f}{v['mean_kendall']:>10.3f}"
              f"{v['mean_floor_rise']:>10.4f}{v['verdict_sign_agreement']:>10.2f}"
              f"  {'eligible' if v['eligible'] else 'DISQUALIFIED (floor)'}")
    print(f"\n  best eligible proxy: {out['best_eligible']}")
    if out["best_eligible"] is None:
        print("  NOTE: every proxy contaminates authentic frames; no "
              "perturbation-based proxy is deployable on this data.")
    print(f"  -> {path}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m11_proxy")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default="face8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    a = ap.parse_args(argv)
    run([d for d in a.raw.split(",") if d.strip()], a.out, a.vocab, a.seed,
        a.n_boot, a.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
