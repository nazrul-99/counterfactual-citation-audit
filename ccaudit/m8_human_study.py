"""
ccaudit.m8_human_study -- Module 8, human-study materials.

Builds the materials for a human plausibility study and scores the returned
annotations.  The study collects human judgements without any exposure to the
detector's numbers, and a key file allows the judgements to be joined back to
the detector's per-region effects afterwards, giving a cross-tabulation of
human-visible difference against detector verdict movement.

Each item shows the original fake frame and its counterfactual (the cited
region replaced by authentic pixels) side by side in random order, with the
cited region named in the caption, and asks two questions:

    Q1  which image looks more manipulated?           (A / B / cannot tell)
    Q2  does the named region look manipulated in the
        image you chose?                              (yes / no / cannot tell)

Q1 is the plausibility probe: if the counterfactual removed visible forgery
evidence, annotators should prefer the original as "more manipulated".
Joining Q1 against the detector's hidden Delta yields the cross-tabulation of
items where humans see the difference but the verdict did not move, and the
converse.

Files written
-------------
    items/NNN.png       the side-by-side pair, captioned, order randomised
    annotations.csv     one row per (annotator, item), blank answers
    INSTRUCTIONS.txt    the text given to annotators
    key.json            true left/right order, cited region, detector, Delta,
                        FS and sample_id; must not be given to annotators

`key.json` is the only file that contains the answers.  The folder is
distributed to annotators without it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import common as C
from . import m3_splice as M3
from . import m6_metrics as M6
from . import regions as R

INSTRUCTIONS = """\
Human study -- instructions for annotators
==========================================

You will see {n} items.  Each item shows TWO images of the same face, side by
side, labelled A (left) and B (right).  One of them is a manipulated
("deepfake") frame.  The other is the same frame with ONE facial region
replaced by the corresponding region from the authentic video.  You are NOT
told which is which, and the order is random for every item.

Above each pair a facial region is named (for example "mouth").

For each item answer two questions in annotations.csv:

  Q1  Which image looks MORE manipulated?
      Write  A  or  B  or  ?  (if you cannot tell)

  Q2  In the image you chose, does the NAMED REGION look manipulated?
      Write  yes  or  no  or  ?  (if you cannot tell)

Please:
  * work through the items in order, and do not go back to change earlier
    answers after seeing later ones;
  * judge only what you can see -- do not try to guess the study's purpose;
  * take a break every 25 items or so; fatigue shows up in this kind of task;
  * answering "?" is a valid and useful response.  Please do not guess.

There are no right answers that we will tell you about, and your individual
answers will not be identified in any report.
"""


def _caption(img: np.ndarray, text: str, height: int = 34) -> np.ndarray:
    import cv2

    bar = np.full((height, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, int(height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, img.shape[1] / 1400), (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _side_by_side(left: np.ndarray, right: np.ndarray, region: str,
                  item_id: int) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    gap = np.full((h, 16, 3), 24, np.uint8)
    l = _caption(left, "A")
    r = _caption(right, "B")
    h2 = max(l.shape[0], r.shape[0])
    if l.shape[0] < h2:
        l = np.pad(l, ((0, h2 - l.shape[0]), (0, 0), (0, 0)))
    if r.shape[0] < h2:
        r = np.pad(r, ((0, h2 - r.shape[0]), (0, 0), (0, 0)))
    gap = np.full((h2, 16, 3), 24, np.uint8)
    pair = np.concatenate([l, gap, r], axis=1)
    return _caption(pair, f"item {item_id:03d}   named region: "
                          f"{R.prompt_name(region)}", height=40)


def build(
    index_path: str,
    raw_dirs: Sequence[str],
    out_dir: str,
    n_items: int = 100,
    detector: str = "",
    tag: str = "",
    split: str = "test",
    n_annotators: int = 3,
    blend: str = "poisson",
    dilate: int = 3,
    jpeg_q: int = 90,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Build the study materials under `out_dir` from an index and one or more
    audit run directories.  Returns a summary dict with `n_items`, `detector`
    and `out_dir`.
    """
    if not index_path:
        raise ValueError("--index is required to build the study materials")
    if not raw_dirs:
        raise ValueError("--raw is required to build the study materials")
    os.makedirs(out_dir, exist_ok=True)
    items_dir = os.path.join(out_dir, "items")
    os.makedirs(items_dir, exist_ok=True)

    records, meta = C.load_index(index_path)
    vocab = meta.get("vocab", "face8")
    R.set_vocab(vocab)
    by_id = {r["sample_id"]: r for r in records}

    groups = M6.group_raw_files(raw_dirs)
    if not groups:
        raise FileNotFoundError(f"no raw_*.json under {list(raw_dirs)}")
    # Use the requested detector; otherwise the matching run with most rows.
    chosen = None
    for (det, t), paths in sorted(groups.items()):
        if detector and detector not in det:
            continue
        if tag and t != tag:
            continue
        rows, m = M6.load_rows(paths)
        if chosen is None or len(rows) > len(chosen[1]):
            chosen = ((det, t), rows, m)
    if chosen is None:
        raise ValueError(f"no run matched detector={detector!r} tag={tag!r}")
    (det_name, det_tag), rows, _m = chosen
    print(f"[m8] using {det_name} [{det_tag}] with {len(rows)} rows")

    samples = M6.samples_from_rows(rows)
    usable = []
    for sid, s in samples.items():
        if sid not in by_id:
            continue
        if split != "all" and s.get("split") != split:
            continue
        cited = M6.cited_region(s)
        if not cited or cited not in s["regions"]:
            continue
        usable.append((sid, cited, s))
    if not usable:
        raise RuntimeError("no usable samples (check --split and that the run "
                           "carries citations)")
    usable.sort(key=lambda t: t[0])
    pick = C.limit_samples([{"sample_id": sid} for sid, _c, _s in usable],
                           n_items, seed)
    keep = {p["sample_id"] for p in pick}
    usable = [u for u in usable if u[0] in keep]
    print(f"[m8] building {len(usable)} items from split={split}")

    rng = np.random.default_rng(seed)
    key: List[Dict[str, Any]] = []
    for i, (sid, cited, s) in enumerate(usable, start=1):
        rec = by_id[sid]
        _real, fake, lab, per = M3.sample_conditions(
            rec, [cited], dilate, blend, jpeg_q, with_floor=False, vocab=vocab)
        if not per:
            continue
        counterfactual = per[0]["images"]["real"]
        swap = bool(rng.integers(0, 2))
        left, right = ((counterfactual, fake) if swap else (fake, counterfactual))
        sheet = _side_by_side(left, right, cited, i)
        path = os.path.join(items_dir, f"{i:03d}.png")
        C.imwrite_png(path, sheet)

        e = s["regions"][cited]
        key.append({
            "item": i, "file": os.path.relpath(path, out_dir),
            "sample_id": sid, "pair_id": s["pair_id"], "method": s["method"],
            "cited_region": cited,
            "original_fake_side": "B" if swap else "A",
            "counterfactual_side": "A" if swap else "B",
            "detector": det_name, "tag": det_tag,
            "delta": e.get("delta"), "delta_raw": e.get("delta_raw"),
            "delta_seam": e.get("delta_seam"),
            "p_orig_fake": s["p_orig_fake"],
            "FS": M6.fs_of(s, cited),
            "rank_cited": M6.rank_of_cited(s, cited),
        })

    # Annotation sheet: blank, one row per (annotator, item).
    csv_path = os.path.join(out_dir, "annotations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["annotator", "item", "file",
                    "q1_more_manipulated_A_B_?", "q2_named_region_yes_no_?",
                    "notes"])
        for a in range(1, n_annotators + 1):
            for k in key:
                w.writerow([f"annotator_{a}", k["item"], k["file"], "", "", ""])

    with open(os.path.join(out_dir, "INSTRUCTIONS.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(INSTRUCTIONS.format(n=len(key)))

    C.save_json(os.path.join(out_dir, "key.json"), {
        "meta": {"detector": det_name, "tag": det_tag, "split": split,
                 "n_items": len(key), "n_annotators": n_annotators,
                 "blend": blend, "dilate": dilate, "jpeg_q": jpeg_q,
                 "seed": seed, "code_hash": C.code_hash(),
                 "WARNING": "contains the answers -- do not give to annotators"},
        "items": key,
    }, indent=1)

    print(C.banner("Module 8 summary"))
    print(f"  items          {len(key)}  -> {items_dir}")
    print(f"  annotations    {csv_path}  ({n_annotators} annotators)")
    print(f"  key            {os.path.join(out_dir, 'key.json')}  "
          f"** keep away from annotators **")
    side = sum(1 for k in key if k["original_fake_side"] == "A")
    print(f"  randomisation  original fake on side A in {side}/{len(key)} items")
    return {"n_items": len(key), "detector": det_name, "out_dir": out_dir}


def score(out_dir: str, filled_csv: str = "") -> Dict[str, Any]:
    """
    Join returned annotations against key.json and produce the cross-tab of
    human-visible difference (majority Q1 answer equals the true original
    side) against detector verdict movement (|Delta| > 0.01).
    """
    key_blob = C.load_json(os.path.join(out_dir, "key.json"))
    key = {k["item"]: k for k in key_blob["items"]}
    path = filled_csv or os.path.join(out_dir, "annotations.csv")
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    per_item: Dict[int, List[str]] = {}
    for r in rows:
        try:
            it = int(r["item"])
        except (KeyError, ValueError):
            continue
        ans = (r.get("q1_more_manipulated_A_B_?") or "").strip().upper()
        if ans in ("A", "B"):
            per_item.setdefault(it, []).append(ans)

    tab = {"human_saw_detector_moved": 0, "human_saw_detector_still": 0,
           "human_blind_detector_moved": 0, "human_blind_detector_still": 0}
    agree, n_scored = [], 0
    for it, answers in per_item.items():
        k = key.get(it)
        if not k or not answers:
            continue
        majority = max(set(answers), key=answers.count)
        human_correct = (majority == k["original_fake_side"])
        agree.append(1.0 if human_correct else 0.0)
        moved = abs(_safe(k.get("delta"))) > 0.01
        tab["human_saw_detector_moved" if human_correct and moved else
            "human_saw_detector_still" if human_correct else
            "human_blind_detector_moved" if moved else
            "human_blind_detector_still"] += 1
        n_scored += 1

    out = {
        "n_items_scored": n_scored,
        "human_accuracy": float(np.mean(agree)) if agree else float("nan"),
        "crosstab": tab,
        "detector": key_blob["meta"]["detector"],
        "note": "human_saw_detector_still = plausible to humans but the "
                "verdict did not move (plausible-but-not-causal citations)",
    }
    C.save_json(os.path.join(out_dir, "human_results.json"), out, indent=1)
    print(f"[m8] scored {n_scored} items, human accuracy "
          f"{out['human_accuracy']:.3f}, crosstab {tab}")
    return out


def _safe(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m8_human_study")
    ap.add_argument("--index", default="", help="required unless --score")
    ap.add_argument("--raw", default="",
                    help="comma-separated run dirs; required unless --score")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--detector", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="test", choices=["all", "dev", "test"])
    ap.add_argument("--annotators", type=int, default=3)
    ap.add_argument("--blend", default="poisson")
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--jpeg-q", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", default="", help="score a filled-in CSV instead")
    a = ap.parse_args(argv)

    if a.score:
        score(a.out, a.score)
        return 0
    if not a.index or not a.raw:
        ap.error("--index and --raw are required unless --score is given")
    build(a.index, [d for d in a.raw.split(",") if d.strip()], a.out,
          n_items=a.n_items, detector=a.detector, tag=a.tag, split=a.split,
          n_annotators=a.annotators, blend=a.blend, dilate=a.dilate,
          jpeg_q=a.jpeg_q, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
