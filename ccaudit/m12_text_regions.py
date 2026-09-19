"""
ccaudit.m12_text_regions -- Module 12, free text to regions.

Maps a detector's free-text explanation onto the region vocabulary with a
fixed lexicon, so that FS can be computed for what the model states in its
own words and not only for the option letter it selects from a menu.  This
addresses the objection that the closed citation vocabulary is artificial.

Design decisions
----------------
* The lexicon is fixed and published (see LEXICON below) and is not tuned
  per detector.
* Longer phrases are matched before shorter ones, so "corner of the eye" does
  not also fire "ear" via substring overlap.  Matching is on word boundaries,
  never bare substrings: "ear" must not match "eyebrow" or "beard".
* Laterality: "left"/"right" within a few words of an eye term routes to that
  eye; an unqualified eye term splits its mass 50/50, because the model has
  not committed to a side and resolving it would invent precision.
* "left" and "right" are image sides, matching the convention in regions.py.
* Sentences naming no region are counted rather than dropped: `frac_no_region`
  is a reported quantity, since a detector that mostly says nothing locatable
  is a result rather than missing data.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C
from . import m6_metrics as M6
from . import regions as R

# region -> trigger phrases.  Order inside a list does not matter; globally,
# longer phrases are tried first (see _compile).
LEXICON: Dict[str, List[str]] = {
    "mouth": ["mouth", "lips", "lip", "teeth", "tooth", "smile", "smiling",
              "mouth region", "lip line", "philtrum", "vermilion"],
    "nose": ["nose", "nostril", "nostrils", "nasal", "bridge of the nose",
             "nose bridge", "nasal bridge", "tip of the nose"],
    "_eye": ["eye", "eyes", "eyelid", "eyelids", "iris", "irises", "pupil",
             "pupils", "eyebrow", "eyebrows", "brow", "brows", "eye region",
             "eye socket", "under-eye", "eyelash", "eyelashes", "gaze"],
    "skin": ["skin", "cheek", "cheeks", "forehead", "complexion", "texture",
             "skin texture", "skin tone", "pores", "cheekbone", "cheekbones",
             "temple", "temples", "blemish", "wrinkles"],
    "jaw_boundary": ["jaw", "jawline", "jaw line", "chin", "boundary",
                     "face outline", "outline of the face", "face border",
                     "edge of the face", "facial contour", "contour",
                     "blending boundary", "face edge", "mandible"],
    "hair": ["hair", "hairline", "hair line", "fringe", "bangs", "scalp"],
    "ears": ["ear", "ears", "earlobe", "earlobes", "auricle"],
}

LEFT_WORDS = ("left", "left-hand", "viewer's left", "image left")
RIGHT_WORDS = ("right", "right-hand", "viewer's right", "image right")

# Number of words before an eye term that are searched for a laterality cue.
LATERALITY_WINDOW = 4


def _compile() -> List[Tuple[re.Pattern, str]]:
    """Longest phrase first, matched on word boundaries."""
    pairs: List[Tuple[str, str]] = []
    for region, phrases in LEXICON.items():
        for p in phrases:
            pairs.append((p, region))
    pairs.sort(key=lambda t: -len(t[0]))
    return [(re.compile(r"\b" + re.escape(p) + r"\b", re.I), r) for p, r in pairs]


_PATTERNS = _compile()


def text_to_regions(text: str, vocab: Optional[str] = None
                    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Map an explanation sentence to a region distribution.

    Returns (distribution, diagnostics).  The distribution is uniform and
    `diagnostics["named"]` is False when nothing matched; the caller decides
    how to treat that case, and `analyse_text` counts it.
    """
    names = R.get_vocab(vocab)
    hits: Counter = Counter()
    if not text or not str(text).strip():
        return ({n: 1.0 / len(names) for n in names},
                {"named": False, "matches": [], "reason": "empty"})

    s = str(text)
    low = s.lower()
    consumed = [False] * len(low)
    matches: List[Tuple[str, str]] = []

    for pat, region in _PATTERNS:
        for m in pat.finditer(low):
            a, b = m.span()
            if any(consumed[a:b]):
                continue            # already claimed by a longer phrase
            for i in range(a, b):
                consumed[i] = True
            matches.append((m.group(0), region))

            if region != "_eye":
                hits[region] += 1.0
                continue
            # Laterality: search a few words back for left/right.
            before = low[:a].split()[-LATERALITY_WINDOW:]
            said_left = any(w.strip(",.;:") in LEFT_WORDS for w in before)
            said_right = any(w.strip(",.;:") in RIGHT_WORDS for w in before)
            if said_left and not said_right:
                hits["left_eye"] += 1.0
            elif said_right and not said_left:
                hits["right_eye"] += 1.0
            else:
                hits["left_eye"] += 0.5
                hits["right_eye"] += 0.5

    hits = Counter({k: v for k, v in hits.items() if k in names})
    if not hits:
        return ({n: 1.0 / len(names) for n in names},
                {"named": False, "matches": matches, "reason": "no_region_term"})
    tot = sum(hits.values())
    dist = {n: float(hits.get(n, 0.0) / tot) for n in names}
    return dist, {"named": True, "matches": matches,
                  "argmax": max(dist, key=dist.get)}


def _argmax_fair(dist: Dict[str, float], salt: str) -> str:
    """
    argmax with ties broken by a hash of the sample id rather than by dict
    order.  An unqualified "the eyes" splits 50/50 between left_eye and
    right_eye; always resolving that to left_eye would introduce a systematic
    bias into the text-versus-letter agreement statistic.  Hashing spreads
    the ties evenly across the dataset and is reproducible.
    """
    if not dist:
        return ""
    top = max(dist.values())
    tied = sorted(k for k, v in dist.items() if v >= top - 1e-12)
    if len(tied) == 1:
        return tied[0]
    h = int(C.stable_id("m12-tie", salt), 16)
    return tied[h % len(tied)]


def analyse_text(rows: Sequence[Dict[str, Any]], vocab: Optional[str] = None
                 ) -> Optional[Dict[str, Any]]:
    """
    Agreement between the free-text citation and the constrained one, plus FS
    recomputed from the text-derived citation.  Returns None when no row
    carries `explanation_text`.
    """
    samples = M6.samples_from_rows(rows)
    agree, fs_text, fs_letter, no_region = [], [], [], []
    per_region_named: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    for sid, s in samples.items():
        txt = s.get("explanation_text")
        if txt is None:
            continue
        dist, diag = text_to_regions(txt, vocab)
        no_region.append(0.0 if diag["named"] else 1.0)
        ks = list(s["regions"])
        if not ks:
            continue
        letter = M6.cited_region(s)
        if diag["named"]:
            d = M6._renorm(dist, ks)
            pick = _argmax_fair(d, sid)
            per_region_named[pick] += 1
            if letter:
                agree.append(1.0 if pick == letter else 0.0)
            v = M6.fs_of(s, pick)
            if np.isfinite(v):
                fs_text.append(v)
        if letter:
            v = M6.fs_of(s, letter)
            if np.isfinite(v):
                fs_letter.append(v)
        if len(examples) < 12:
            examples.append({"sample_id": sid, "text": str(txt)[:240],
                             "text_argmax": diag.get("argmax"),
                             "letter": letter, "matches": diag["matches"][:6]})

    if not no_region:
        return None
    return {
        "n": len(no_region),
        "frac_no_region": float(np.mean(no_region)),
        "agreement_text_vs_letter": float(np.mean(agree)) if agree else float("nan"),
        "n_agreement": len(agree),
        "FS_text": float(np.mean(fs_text)) if fs_text else float("nan"),
        "FS_letter": float(np.mean(fs_letter)) if fs_letter else float("nan"),
        "text_argmax_distribution": {k: v / max(1, sum(per_region_named.values()))
                                     for k, v in per_region_named.items()},
        "examples": examples,
    }


def run(raw_dirs: Sequence[str], out_dir: str, vocab: str = "face8"
        ) -> Dict[str, Any]:
    """Analyse every run under `raw_dirs` and write `text_regions.json`."""
    R.set_vocab(vocab)
    os.makedirs(out_dir, exist_ok=True)
    groups = M6.group_raw_files(raw_dirs)
    out: Dict[str, Any] = {"results": [], "lexicon": LEXICON,
                           "provenance": C.provenance()}
    for (det, tag), paths in sorted(groups.items()):
        rows, _m = M6.load_rows(paths)
        r = analyse_text(rows, vocab)
        if not r:
            continue
        r["detector"], r["tag"] = det, tag
        out["results"].append(r)
        print(f"[m12] {det} [{tag}]: n={r['n']} "
              f"agreement={r['agreement_text_vs_letter']:.3f} "
              f"FS_text={r['FS_text']:.4f} vs FS_letter={r['FS_letter']:.4f} "
              f"no_region={r['frac_no_region']:.3f}", flush=True)
    if not out["results"]:
        print("[m12] no rows carry explanation_text; run m5 with --free-text")
    path = os.path.join(out_dir, "text_regions.json")
    C.save_json(path, out, indent=1)
    print(f"[m12] -> {path}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m12_text_regions")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default="face8")
    ap.add_argument("--try-text", default="", help="map one sentence and exit")
    a = ap.parse_args(argv)
    R.set_vocab(a.vocab)
    if a.try_text:
        d, diag = text_to_regions(a.try_text)
        print({k: round(v, 3) for k, v in d.items() if v > 0})
        print(diag)
        return 0
    run([d for d in a.raw.split(",") if d.strip()], a.out, a.vocab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
