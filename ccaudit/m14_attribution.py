"""
ccaudit.m14_attribution -- Module 14, stated versus attributed versus causal.

Compares three answers to "which region is responsible", obtained from the
same model on the same image:

  STATED      the letter it picks from the citation menu        (m4.cite)
  ATTRIBUTED  where the gradient of the fake-logit puts mass on
              the visual tokens                                 (m4.attribute)
  OVERLAY     which outlined region it says contains the forgery
              when the region is drawn on the pixels            (m4.p_overlay)
  CAUSAL      argmax of the measured Delta                      (the audit)

The joint agreement counts quantify the dissociation between what the model
states, what its gradients attribute, and what causally drives its verdict,
paralleling the stated-versus-actual reasoning comparisons made on the text
side of the chain-of-thought faithfulness literature.

This module also reports the encoder blind-spot decomposition: of the
citations that are not causal, how many are cases where the visual tokens
covering that region never moved under the counterfactual.  Such cases are
invisible to the encoder and could not be corrected by aligning the language
side alone.

The hooks and gradients live in m4; the runner collects the fields; this
module is the analysis and the CLI that writes attribution.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Optional, Sequence

import numpy as np

from . import common as C
from . import m6_metrics as M6
from . import regions as R


def three_way(rows: Sequence[Dict[str, Any]], vocab: Optional[str] = None
              ) -> Optional[Dict[str, Any]]:
    """
    Joint counts of stated/attributed/overlay agreement with the causal
    region, overall and per method, plus the mean rank of the stated and
    attributed regions.  Returns None when no row carries `cite_attr` or
    `cite_overlay`.
    """
    samples = M6.samples_from_rows(rows)
    have = any(s.get("cite_attr") or s.get("cite_overlay")
               for s in samples.values())
    if not have:
        return None

    counts: Counter = Counter()
    per_method: Dict[str, Counter] = defaultdict(Counter)
    n = 0
    rank_attr, rank_stated = [], []

    for sid, s in samples.items():
        ks = list(s["regions"])
        if len(ks) < 2:
            continue
        deltas = {k: M6._f(s["regions"][k].get("delta")) for k in ks}
        deltas = {k: v for k, v in deltas.items() if np.isfinite(v)}
        if not deltas:
            continue
        causal = max(deltas, key=deltas.get)
        stated = M6.cited_region(s)
        attr = (max(M6._renorm(s["cite_attr"], ks).items(),
                    key=lambda kv: kv[1])[0] if s.get("cite_attr") else None)
        ovl = (max(M6._renorm(s["cite_overlay"], ks).items(),
                   key=lambda kv: kv[1])[0] if s.get("cite_overlay") else None)
        if stated is None:
            continue
        n += 1
        key = (f"stated{'=' if stated == causal else '!='}causal",
               f"attr{'=' if attr == causal else '!='}causal" if attr else "attr:n/a",
               f"overlay{'=' if ovl == causal else '!='}causal" if ovl else "overlay:n/a")
        counts[key] += 1
        per_method[s["method"]][key] += 1
        rank_stated.append(M6.rank_of_cited(s, stated))
        if attr:
            rank_attr.append(M6.rank_of_cited(s, attr))

    if not n:
        return None
    return {
        "n": n,
        "joint_counts": {" & ".join(k): v for k, v in counts.most_common()},
        "by_method": {m: {" & ".join(k): v for k, v in c.most_common()}
                      for m, c in per_method.items()},
        "mean_rank_stated": float(np.nanmean(rank_stated)) if rank_stated else float("nan"),
        "mean_rank_attributed": float(np.nanmean(rank_attr)) if rank_attr else float("nan"),
    }


def run(raw_dirs: Sequence[str], out_dir: str, vocab: str = "face8",
        seed: int = 0, n_boot: int = 1000) -> Dict[str, Any]:
    """Analyse every run under `raw_dirs` and write `attribution.json`."""
    R.set_vocab(vocab)
    os.makedirs(out_dir, exist_ok=True)
    groups = M6.group_raw_files(raw_dirs)
    out: Dict[str, Any] = {"results": [], "provenance": C.provenance()}
    for (det, tag), paths in sorted(groups.items()):
        rows, _m = M6.load_rows(paths)
        samples = M6.samples_from_rows(rows)
        sids = sorted(samples)
        cited = {s: M6.cited_region(samples[s]) for s in sids}
        entry: Dict[str, Any] = {"detector": det, "tag": tag,
                                 "n_samples": len(sids)}
        ag = M6.citation_agreement(samples, sids)
        if ag:
            entry["agreement"] = ag
        tw = three_way(rows, vocab)
        if tw:
            entry["three_way"] = tw
        bs = M6.blind_spot(samples, sids, cited)
        if bs:
            entry["blind_spot"] = bs
        if len(entry) > 3:
            out["results"].append(entry)
            a = entry.get("agreement", {})
            b = entry.get("blind_spot", {})
            print(f"[m14] {det} [{tag}]: stated=causal "
                  f"{a.get('stated_vs_causal', float('nan')):.3f}, "
                  f"attr=causal {a.get('attr_vs_causal', float('nan')):.3f}, "
                  f"stated=attr {a.get('stated_vs_attr', float('nan')):.3f}"
                  + (f" | encoder-blind {b.get('encoder_blind_frac', float('nan')):.3f}"
                     f" confab {b.get('language_confabulation_frac', float('nan')):.3f}"
                     if b else ""), flush=True)
    if not out["results"]:
        print("[m14] no rows carry cite_attr / cite_overlay / enc_shift_in; "
              "run m5 with --attribution --cite-mode both --encoder-probe")
    path = os.path.join(out_dir, "attribution.json")
    C.save_json(path, out, indent=1)
    print(f"[m14] -> {path}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m14_attribution")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default="face8")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run([d for d in a.raw.split(",") if d.strip()], a.out, a.vocab, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
