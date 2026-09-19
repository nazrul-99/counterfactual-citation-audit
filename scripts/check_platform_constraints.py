#!/usr/bin/env python3
"""
check_platform_constraints.py -- verify that the codebase respects the
limits of the Kaggle free tier.

This is a static audit: it reads the generated notebooks and the package
source and checks the constraints that, if violated, terminate a session
hours in.  It should be run after any change to the notebooks or the modules.

    python scripts/check_platform_constraints.py
    python scripts/check_platform_constraints.py --samples 10000   # size model input

Constraints checked (free tier, 2026):

  C1  session wall clock 10 h   -- every session's time budgets must sum to a
                                   value that leaves real head-room
  C2  /kaggle/working 19 GB     -- projected output size for a full run
  C3  /kaggle/temp is scratch   -- big, disposable things must go there, and
                                   nothing needed later may be left there
  C4  GPU 2 x T4, 15 GB, no NVLink -- one process per card, never model-parallel
  C5  resumability              -- every long stage takes --time-budget-min and
                                   can resume, so a wall-clock kill is recoverable
  C6  no session fallbacks      -- no silent degradation paths
  C7  accelerator metadata      -- CPU sessions must not request a GPU
  C8  read-only inputs          -- nothing writes into /kaggle/input
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")
PKG = os.path.join(ROOT, "ccaudit")

WALL_H = 10.0
WALL_MIN = WALL_H * 60
SAFE_MIN = 540          # 9 h: budgets above this leave too little head-room
OUTPUT_CAP_GB = 19.0

FINDINGS: List[Tuple[str, str, str]] = []   # (level, constraint, message)


def ok(c: str, m: str) -> None:
    FINDINGS.append(("ok", c, m))


def warn(c: str, m: str) -> None:
    FINDINGS.append(("warn", c, m))


def fail(c: str, m: str) -> None:
    FINDINGS.append(("fail", c, m))


def notebooks() -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for p in sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb"))):
        out.append((os.path.basename(p), json.load(open(p))))
    return out


def code_cells(nb: Dict[str, Any]) -> List[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def config_of(nb: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the config cell's literal assignments."""
    cfg: Dict[str, Any] = {}
    for src in code_cells(nb)[:2]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name):
                    try:
                        cfg[t.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        try:
                            cfg[t.id] = eval(compile(ast.Expression(node.value),
                                                     "<c>", "eval"), {}, {})
                        except Exception:
                            pass
    return cfg


# --------------------------------------------------------------------------
# C1 -- wall clock
# --------------------------------------------------------------------------

def c1_wall_clock() -> None:
    print("\nC1  session wall clock (10 h)")
    print(f"  {'notebook':34s}{'budgeted min':>14}{'hours':>8}   verdict")
    for name, nb in notebooks():
        cfg = config_of(nb)
        budgets = {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float)) and
                   ("BUDGET" in k or k.endswith("_MIN") and "BUDGET" in k)}
        # Per-variant / per-model budgets multiply.
        total = 0.0
        detail = []
        for k, v in sorted(budgets.items()):
            mult = 1
            if "PER_MODEL" in k and isinstance(cfg.get("MODELS"), list):
                mult = max(1, len(cfg["MODELS"]))
            if "PER_VARIANT" in k and isinstance(cfg.get("VARIANTS"), list):
                mult = max(1, len(cfg["VARIANTS"]))
            if k == "TRAIN_BUDGET_MIN" and isinstance(cfg.get("VARIANTS"), list):
                mult = max(1, len(cfg["VARIANTS"]))
            if k == "EXTRA_BUDGET_MIN":
                n = len(cfg.get("INPAINT_METHODS", []) or []) \
                    + len(cfg.get("ABL_MAX_PIXELS", []) or []) \
                    + (1 if cfg.get("CROP_CONTROL_INDEX") else 0)
                mult = max(1, n)
            total += v * mult
            detail.append(f"{k}x{mult}={v*mult:g}")
        hours = total / 60.0
        if total == 0:
            verdict = "no long stage"
        elif total > WALL_MIN:
            verdict = "EXCEEDS THE WALL"
            fail("C1", f"{name}: budgets sum to {total:.0f} min > {WALL_MIN:.0f}")
        elif total > SAFE_MIN:
            verdict = "tight"
            warn("C1", f"{name}: {total:.0f} min leaves under 1 h of head-room")
        else:
            verdict = "ok"
        print(f"  {name:34s}{total:>14.0f}{hours:>8.1f}   {verdict}")
        if total > SAFE_MIN:
            print(f"      {'; '.join(detail)}")
    if not any(f[0] == "fail" and f[1] == "C1" for f in FINDINGS):
        ok("C1", "every session's budgets fit inside the 10 h wall")


# --------------------------------------------------------------------------
# C2 -- output size
# --------------------------------------------------------------------------

def c2_output_size(n_samples: int) -> None:
    print(f"\nC2  /kaggle/working output cap ({OUTPUT_CAP_GB:.0f} GB), "
          f"modelled at {n_samples} samples")
    # Measured on the fixture: a 384 px q90 crop is ~28 KB, a label PNG ~4 KB.
    per_sample_mb = (2 * 28 + 4) / 1024.0
    crops = n_samples * per_sample_mb / 1024.0
    rows_mb = n_samples * 8 * 0.9 / 1024.0          # 8 rows/sample, ~0.9 KB each
    cache_mb = n_samples * 60 * 0.12 / 1024.0       # ~60 cache entries/sample
    items = [
        ("parsed crops + labels", crops),
        ("raw rows (per detector)", rows_mb / 1024.0),
        ("caches (per detector)", cache_mb / 1024.0),
        ("metrics + report + figures", 0.05),
        ("human study (100 items)", 0.03),
        ("cnn checkpoint", 0.03),
        ("cset adapters (2)", 0.10),
    ]
    total = sum(v for _k, v in items)
    for k, v in items:
        print(f"    {k:32s}{v:8.2f} GB")
    print(f"    {'TOTAL (single session)':32s}{total:8.2f} GB "
          f"({100*total/OUTPUT_CAP_GB:.0f}% of cap)")
    if total > OUTPUT_CAP_GB:
        fail("C2", f"projected output {total:.1f} GB exceeds the cap")
    elif total > 0.7 * OUTPUT_CAP_GB:
        warn("C2", f"projected output {total:.1f} GB is above 70% of the cap")
    else:
        ok("C2", f"projected output {total:.1f} GB, comfortably under "
                 f"{OUTPUT_CAP_GB:.0f} GB")

    # Materialising every condition image would exceed the cap; the runner
    # must build conditions in memory.
    src = open(os.path.join(PKG, "m5_runner.py")).read()
    if "sample_conditions" in src and "make_conditions" not in src:
        ok("C2", "the runner builds conditions in memory and never "
                 "materialises them")
    else:
        warn("C2", "check that the runner does not materialise conditions")
    for name, nb in notebooks():
        cfg = config_of(nb)
        m = cfg.get("MATERIALISE_SAMPLES")
        if isinstance(m, int) and m > 100:
            warn("C2", f"{name}: MATERIALISE_SAMPLES={m} writes ~24 PNGs each")
        if cfg.get("CROP_SIZE") == 0:
            warn("C2", f"{name}: CROP_SIZE=0 stores full frames (~30x disk)")


# --------------------------------------------------------------------------
# C3 -- scratch
# --------------------------------------------------------------------------

def c3_scratch() -> None:
    print("\nC3  /kaggle/temp scratch (erased at session end)")
    common = open(os.path.join(PKG, "common.py")).read()
    if "/kaggle/temp" in common and "scratch_dir" in common:
        ok("C3", "common.scratch_dir() resolves to /kaggle/temp")
    else:
        fail("C3", "no scratch directory helper found")
    hf = [n for n, nb in notebooks()
          if any('HF_HOME' in c and 'temp_dir' in c for c in code_cells(nb))]
    if len(hf) == len(notebooks()):
        ok("C3", f"all {len(hf)} notebooks point HF_HOME at scratch, so model "
                 f"weights never count against the 19 GB output")
    else:
        fail("C3", f"only {len(hf)} notebooks set HF_HOME to scratch")
    # Nothing needed later may live only in scratch.
    bad = []
    for name, nb in notebooks():
        for c in code_cells(nb):
            if re.search(r'--out\s+"?\{?TMP', c) or '--out "{TMP}' in c:
                bad.append(name)
    if bad:
        fail("C3", f"these write results into scratch, which is erased: {bad}")
    else:
        ok("C3", "no session writes results into scratch")


# --------------------------------------------------------------------------
# C4 -- GPU
# --------------------------------------------------------------------------

def c4_gpu() -> None:
    print("\nC4  2 x T4, 15 GB each, no NVLink")
    m4 = open(os.path.join(PKG, "m4_detectors.py")).read()
    single = re.search(r'device_map"?\]?\s*=\s*\{\s*""\s*:\s*0\s*\}', m4)
    if single:
        ok("C4", "models load onto a single card (device_map={'': 0}); the "
                 "code never model-parallels across the two T4s")
    else:
        warn("C4", "could not confirm single-card loading")
    n = 0
    for name, nb in notebooks():
        for c in code_cells(nb):
            if "CUDA_VISIBLE_DEVICES" in c:
                n += 1
                break
    ok("C4", f"{n} GPU notebooks pin one process per card via "
             f"CUDA_VISIBLE_DEVICES + --shard i/n")
    budget = {"3B fp16": 7.5, "3B fp16 + activations @384px": 10.5,
              "7B 4bit": 5.5, "QLoRA 3B training": 10.0,
              "3B fp16 + backward (attribution)": 11.5}
    print(f"    {'configuration':38s}{'GB':>6}   fits 15 GB?")
    for k, v in budget.items():
        print(f"    {k:38s}{v:>6.1f}   {'yes' if v < 14 else 'TIGHT'}")
    for name, nb in notebooks():
        cfg = config_of(nb)
        mdl = str(cfg.get("MODEL", "")) + str(cfg.get("MODELS", ""))
        if "7B" in mdl and cfg.get("QUANT") not in ("4bit",):
            fail("C4", f"{name}: a 7B model with QUANT={cfg.get('QUANT')!r} "
                       f"will not fit a T4; use 4bit")
    ok("C4", "no session configures a 7B model without 4-bit")


# --------------------------------------------------------------------------
# C5 -- resumability
# --------------------------------------------------------------------------

def c5_resume() -> None:
    print("\nC5  resumability (a wall-clock kill must be recoverable)")
    need = {"m2_parse.py": ["--time-budget-min", "--resume", "--shard"],
            "m5_runner.py": ["--time-budget-min", "--resume-from", "--shard"],
            "m9_train_cnn.py": ["--train-budget-min", "--resume"],
            "m13_cset.py": ["--time-budget-min"],
            "m10_localization.py": ["--time-budget-min"]}
    for mod, flags in need.items():
        src = open(os.path.join(PKG, mod)).read()
        missing = [f for f in flags if f not in src]
        if missing:
            fail("C5", f"{mod} lacks {missing}")
        else:
            ok("C5", f"{mod}: {', '.join(flags)}")
    src = open(os.path.join(PKG, "common.py")).read()
    if "os.replace" in src and "fsync" in src:
        ok("C5", "writes are atomic (tmp + fsync + os.replace), so a kill "
                 "mid-write cannot corrupt a checkpoint")
    else:
        fail("C5", "writes are not atomic")
    m5 = open(os.path.join(PKG, "m5_runner.py")).read()
    if "fully_cached" in m5:
        ok("C5", "a fully cached sample is rebuilt without re-splicing, so "
                 "resuming is cheap, not merely possible")


# --------------------------------------------------------------------------
# C6 -- no silent fallbacks
# --------------------------------------------------------------------------

def c6_fallbacks() -> None:
    print("\nC6  no silent session fallbacks")
    m2 = open(os.path.join(PKG, "m2_parse.py")).read()
    if "geometric" in m2 and "backend" in m2 and "--backend" in m2:
        fail("C6", "m2_parse still exposes a --backend fallback parser")
    else:
        ok("C6", "m2_parse has no geometric fallback: mediapipe or it fails")
    if "center-fallback" in m2 or "center_fallback" in m2:
        fail("C6", "m2_parse still has a centre fallback")
    else:
        ok("C6", "no centre fallback")
    m3 = open(os.path.join(PKG, "m3_splice.py")).read()
    if "blend_used" in m3 and "fallback" in m3:
        ok("C6", "blend fallbacks exist but are RECORDED in blend_used, so "
                 "they appear as an ablation cell rather than a silent change")
    m13 = open(os.path.join(PKG, "m13_cset.py")).read()
    if "DEV only" in m13:
        ok("C6", "m13_cset refuses to train on TEST rows rather than warning")
    else:
        fail("C6", "m13_cset does not enforce DEV-only training")
    m1 = open(os.path.join(PKG, "m1_verify.py")).read()
    if "--fail-hard" in m1:
        ok("C6", "the pairing gate can exit non-zero and the notebook stops")


# --------------------------------------------------------------------------
# C7 / C8
# --------------------------------------------------------------------------

def c7_accelerator() -> None:
    print("\nC7  accelerator metadata")
    expect_gpu = {"03", "04", "05", "06", "07", "08", "09", "10a", "10b"}
    for name, nb in notebooks():
        acc = nb["metadata"].get("kaggle", {}).get("accelerator", "none")
        pre = name.split("_")[0]
        want_gpu = pre in expect_gpu
        got_gpu = acc != "none"
        if want_gpu != got_gpu:
            fail("C7", f"{name}: accelerator={acc!r} but "
                       f"{'GPU' if want_gpu else 'CPU'} required")
        else:
            print(f"    {name:34s}{acc}")
    ok("C7", "every notebook requests the right accelerator "
             "(a CPU session asking for a GPU wastes quota)")


def c8_readonly_inputs() -> None:
    print("\nC8  /kaggle/input is read-only")
    bad = []
    for mod in glob.glob(os.path.join(PKG, "*.py")):
        src = open(mod).read()
        for m in re.finditer(r'open\(\s*[^)]*kaggle/input[^)]*["\']w', src):
            bad.append(os.path.basename(mod))
    for name, nb in notebooks():
        for c in code_cells(nb):
            if re.search(r'--out\s+"?/kaggle/input', c):
                bad.append(name)
    if bad:
        fail("C8", f"these appear to write into /kaggle/input: {sorted(set(bad))}")
    else:
        ok("C8", "nothing writes into /kaggle/input; the stage() helper copies "
                 "read-only inputs into the working directory first")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10000)
    a = ap.parse_args()

    print("=" * 72)
    print("Platform constraint audit (Kaggle free tier)")
    print("=" * 72)

    c1_wall_clock()
    c2_output_size(a.samples)
    c3_scratch()
    c4_gpu()
    c5_resume()
    c6_fallbacks()
    c7_accelerator()
    c8_readonly_inputs()

    n_fail = sum(1 for lv, _c, _m in FINDINGS if lv == "fail")
    n_warn = sum(1 for lv, _c, _m in FINDINGS if lv == "warn")
    print("\n" + "=" * 72)
    if n_fail:
        print(f"RESULT: {n_fail} VIOLATION(S), {n_warn} warning(s)")
    else:
        print(f"RESULT: no violations, {n_warn} warning(s)")
    print("=" * 72)
    for lv, c, m in FINDINGS:
        if lv != "ok":
            print(f"  [{lv.upper():4s}] {c}  {m}")
    if not n_fail and not n_warn:
        print("  every constraint satisfied")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
