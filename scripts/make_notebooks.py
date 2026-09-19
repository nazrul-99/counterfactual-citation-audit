#!/usr/bin/env python3
"""
make_notebooks.py -- generator for the session notebooks.

Every .ipynb in notebooks/ is produced from this file; edits made directly to
a notebook are overwritten the next time the generator runs.  To change a
session, edit the corresponding function here and re-run:

    python scripts/make_notebooks.py            # writes notebooks/*.ipynb
    python scripts/make_notebooks.py --out DIR  # writes elsewhere

Session map.  Each session fits inside a 10 h wall-clock limit with margin,
and each session's output is published as a dataset that the later sessions
attach as input:

  S1   CPU   pairing gate + parse                 -> cca-s1-parsed
  S2   CPU   controls + ablations + localization  -> cca-s2-controls
  S3   GPU   VLM smoke matrix                     -> cca-s3-smoke
  S4   GPU   CNN train + audit                    -> cca-s4-cnn
  S5   GPU   VLM main audit (one model per run)   -> cca-s5-vlm-<model>
  S6   GPU   VLM extras: inpaint, tokens, crop    -> cca-s6-extras
  S7   GPU   proxies + free text                  -> cca-s7-proxy
  S8   GPU   attribution + encoder probe          -> cca-s8-attr
  S9   GPU   CSET: DEV audit + QLoRA training     -> cca-s9-cset
  S10  GPU   CSET re-audit (a: base+causal, b: mask) -> cca-s10{a,b}-cset
  S11  CPU   final metrics, report, human study

Design decisions:
  * The CSET re-audit is the only stage whose budget approaches the wall, so
    it is split by variant into S10a and S10b, each an independent session.
  * Input datasets are referred to by placeholders (`<your-code-dataset>`,
    `<your-ff-mirror>`, `<your-celebdf-mirror>`); the notebooks locate every
    input by file name under /kaggle/input, so the mount name is irrelevant.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "notebooks")

# --------------------------------------------------------------------------
# notebook plumbing
# --------------------------------------------------------------------------

def md(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> Dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: List[Dict[str, Any]], accelerator: str = "none",
             internet: bool = True) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": accelerator,
                       "dataSources": [],
                       "isInternetEnabled": internet,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def write(out_dir: str, name: str, nb: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    return path


# --------------------------------------------------------------------------
# shared cells
# --------------------------------------------------------------------------

BOOT = '''
# ---------------------------------------------------------------- BOOT
# Locates the code dataset wherever it is mounted, puts it on sys.path,
# prints the attached inputs, and records provenance.  The search is by
# file name, so the dataset's mount name does not matter.
import os, sys, subprocess, json, time

def _find_code():
    for root in ("/kaggle/input", "."):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if os.path.basename(dirpath) == "ccaudit" and "kaggle_utils.py" in files:
                return os.path.dirname(dirpath)
    raise FileNotFoundError(
        "Could not find the ccaudit package.\\n"
        "Add Input -> your code dataset (<your-code-dataset>), and check that "
        "the preview shows ccaudit/kaggle_utils.py at the top level.")

CODE = _find_code()
if CODE not in sys.path:
    sys.path.insert(0, CODE)
# Child processes do not inherit sys.path.  Every `python -m ccaudit.<module>`
# below runs as a subprocess, so the code directory must be on PYTHONPATH.
os.environ["PYTHONPATH"] = CODE + os.pathsep + os.environ.get("PYTHONPATH", "")
from ccaudit import kaggle_utils as KU
from ccaudit import common as C

OUT = KU.work_dir("audit")
TMP = KU.temp_dir()
os.environ["HF_HOME"] = KU.temp_dir("hf")          # model weights stay out of /kaggle/working
os.environ["TOKENIZERS_PARALLELISM"] = "false"
KU.session_header(SESSION, OUT)
print("code:", CODE)
'''

SELFTEST = '''
# ------------------------------------------------------- SELF TEST (always)
# The self-test suite runs in under a minute and needs no dataset.  Each check
# corresponds to a failure mode that would produce plausible-looking but
# incorrect numbers, so a failure here invalidates everything that follows.
rc = KU.sh(f"{sys.executable} {CODE}/scripts/selftest.py", check=False)
if rc != 0:
    raise SystemExit("SELF TEST FAILED -- inspect the failures above before proceeding.")
'''

FOOTER = '''
# ----------------------------------------------------------- WRAP UP
KU.disk_report()
print(C.banner("NEXT STEP"))
print(NEXT_STEP)
'''


def session_cells(session: str, title: str, intro: str, config: str,
                  body: List[Any], next_step: str, extra_boot: str = "",
                  ) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = [md(f"# {title}\n\n{intro}")]
    cells.append(code(f'SESSION = "{session}"\n\n' + config.strip("\n")))
    cells.append(code(BOOT + extra_boot))
    cells.append(code(SELFTEST))
    for b in body:
        cells.append(b if isinstance(b, dict) else code(b))
    cells.append(code(f'NEXT_STEP = """{next_step.strip()}"""\n' + FOOTER))
    return cells


# --------------------------------------------------------------------------
# S1 -- gate + parse
# --------------------------------------------------------------------------

def s1() -> Dict[str, Any]:
    intro = """
**CPU, 1-3 h.** Verifies that every manipulated clip has a frame-aligned authentic
counterpart, then parses the paired clips into 384 px face crops with region label maps.

Inputs: the code dataset (`<your-code-dataset>`) and a FaceForensics++ mirror
(`<your-ff-mirror>`) or a Celeb-DF mirror (`<your-celebdf-mirror>`).
Accelerator **None**, Internet **On** (mediapipe and the landmarker model are downloaded).

The output of this session is the dataset that every later session reads.
"""
    config = '''
# ============================== CONFIG ==============================
DATA            = ""     # "" = auto-detect the folder containing original/
LAYOUT          = "auto" # auto | ffpp | celebdf
METHODS         = ""     # "" = all manipulation methods; e.g. "Deepfakes,Face2Face"
MAX_PAIRS       = 0      # 0 = all clips; 300 for a short trial run (~15 min)
FRAMES_PER_CLIP = 2      # 1-2 recommended: more frames per clip is pseudo-replication
CROP_SIZE       = 384    # 0 = full frames (30x disk, 5x slower; only for the crop control in Session 6)
CROP_MARGIN     = 0.35
JPEG_Q          = 90
WORKERS         = 4
SHARD           = ""     # "0/2" and "1/2" in two notebooks to halve the wall time
VOCAB           = "face8"

# Module 1 gate
GATE_SAMPLE     = 40     # clips to probe
GATE_PROBES     = 3      # frames per probed clip
MIN_ALIGNMENT   = 0.90   # any change to this threshold must be reported with the results

# Quality gates for Module 2
MIN_IOU         = 0.70   # real-vs-fake parse agreement
MIN_COVERAGE    = 0.50   # fraction of changed pixels the vocabulary can name

# 8 h for parsing leaves about 2 h of head-room under the 10 h wall
TIME_BUDGET_MIN = 480
RESUME_FROM     = ""     # "/kaggle/input/<mount>/audit/parsed/index.json"
'''
    body = [
        '''
# ------------------------------------------------------- dependencies
KU.pip_install("mediapipe")
import importlib
from ccaudit import m2_parse as M2
importlib.reload(M2)
print("landmarker:", M2.locate_landmarker(""))
''',
        '''
# ------------------------------------------------------- locate the data
DATA = DATA or KU.find_dataset_root() or (KU.find_celebdf_root() or "")
if not DATA:
    raise SystemExit(
        "Could not find the dataset. Add Input -> your FF++ mirror "
        "(<your-ff-mirror>) or Celeb-DF mirror (<your-celebdf-mirror>), or set "
        "DATA to the folder that directly contains original/ and the "
        "manipulation folders.")
print("DATA =", DATA)
print("contains:", sorted(os.listdir(DATA))[:12])
''',
        '''
# ============================ MODULE 1: PAIRING GATE ============================
# Every downstream computation assumes that fake frame i and real frame i show
# the same instant.  This is verified on the actual video before any further
# cost is incurred.
PAIRS = f"{OUT}/pairs.json"
rc = KU.sh(
    f'{sys.executable} -m ccaudit.m1_verify --root "{DATA}" '
    f'--layout {LAYOUT} --methods "{METHODS}" --sample {GATE_SAMPLE} '
    f'--probes {GATE_PROBES} --min-alignment {MIN_ALIGNMENT} '
    f'--json "{OUT}/m1_report.json" --emit-pairs "{PAIRS}" --fail-hard',
    check=False, log=f"{OUT}/logs/m1.log")
if rc != 0:
    raise SystemExit(
        "PAIRING GATE FAILED. The PHASE 5 block above lists the diagnosis in "
        "order. Nothing should be parsed or audited until the gate passes: a "
        "failed gate means the counterfactual pairing does not hold.")
''',
        '''
# ============================ MODULE 2: PARSE ============================
# The parser uses the mediapipe Tasks API only.  There is deliberately no
# geometric fallback, because a fallback can silently produce boxes that do
# not sit on the face.
resume = f' --resume-from "{RESUME_FROM}"' if RESUME_FROM else ""
shard  = f' --shard {SHARD}' if SHARD else ""
KU.sh(
    f'{sys.executable} -m ccaudit.m2_parse --pairs "{PAIRS}" '
    f'--out "{OUT}/parsed" --data-root "{DATA}" '
    f'--frames-per-clip {FRAMES_PER_CLIP} --crop-size {CROP_SIZE} '
    f'--crop-margin {CROP_MARGIN} --jpeg-q {JPEG_Q} --min-iou {MIN_IOU} '
    f'--min-coverage {MIN_COVERAGE} --methods "{METHODS}" '
    f'--max-pairs {MAX_PAIRS} --workers {WORKERS} --vocab {VOCAB} '
    f'--time-budget-min {TIME_BUDGET_MIN}{shard}{resume} '
    f'--overlay "{OUT}/overlay.png"',
    check=False, log=f"{OUT}/logs/m2.log")
''',
        '''
# ------------------------------------- VISUAL CHECK OF THE PARSE
# If the coloured regions do not sit on the face, the parse is wrong and every
# downstream number is meaningless.  The log should contain
#   "mediapipe backend ready (tasks API)"
try:
    from IPython.display import Image, display
    if os.path.exists(f"{OUT}/overlay.png"):
        display(Image(filename=f"{OUT}/overlay.png"))
    else:
        print("no overlay written -- module 2 produced no samples")
except Exception as exc:                 # a display failure must not abort a long run
    print(f"(could not display the overlay: {exc}); the file is at "
          f"{OUT}/overlay.png -- open it from the Output panel)")
''',
        '''
# ------------------------------------------------------- summary
stats = C.load_json(f"{OUT}/parsed/parse_stats.json", {})
for k in ("n_records", "n_clips_with_output", "parser_iou_mean",
          "manip_coverage_mean", "n_dropped_total", "split_counts",
          "method_counts", "stopped_early"):
    print(f"  {k:24s} {stats.get(k)}")
print("  drops:", stats.get("drops"))
if stats.get("stopped_early"):
    print("\\n!! Module 2 stopped on its time budget. Save this version, add "
          "its output as an input, set RESUME_FROM to its parsed/index.json "
          "and run again. Finished clips are skipped.")
'''
    ]
    next_step = """
1. Save Version -> Save & Run All (Commit) if not already done.
2. Open the finished version -> Output tab -> New Dataset. Name it
   `cca-s1-parsed`. That dataset is the input to every later session.
3. Continue with Session 2 (02_cpu_controls.ipynb).

Before moving on, confirm:
  * Module 1 printed RESULT: PAIRING VERIFIED
  * the overlay picture shows regions sitting on the face
  * parser IoU mean is above ~0.7 and coverage above ~0.5
  * `samples written` is roughly 2 x the number of clips
"""
    return notebook(session_cells("S1 gate+parse", "Session 1 - pairing gate + parsing",
                                  intro, config, body, next_step), "none", True)


# --------------------------------------------------------------------------
# S2 -- controls + ablations + localization
# --------------------------------------------------------------------------

def s2() -> Dict[str, Any]:
    intro = """
**CPU, 2-4 h.** Runs the four synthetic control detectors, the operator ablations,
and ground-truth localization.

This session validates the instrument. The controls are constructed with known
faithfulness, so their ordering on **real frames** is a check on the masks and the
data: the audit proceeds to the GPU sessions only if that ordering holds.

Inputs: `<your-code-dataset>` and `cca-s1-parsed`. Accelerator **None**.
"""
    config = '''
# ============================== CONFIG ==============================
CONTROLS    = "adaptive_oracle,fixed_oracle:mouth,confabulator,dummy"
N_PROC      = 4          # one shard per CPU core
BLEND       = "poisson"
DILATE      = 3
JPEG_Q      = 90
INPAINT     = "telea"    # inpainting pipeline check ("" = off)
SPLICE_FLOOR = True      # operator validity check on authentic frames; keep on

# operator ablations. 0 samples = skip.
ABL_SAMPLES = 600
ABL_BLENDS  = ["feather", "hard"]
ABL_DILATE  = [0, 7, 11]
ABL_NO_REENCODE = True

MAIN_BUDGET_MIN = 300    # controls stop cleanly at 5 h
ABL_BUDGET_MIN  = 150    # ablations get a further 2.5 h -> 7.5 h total
MATERIALISE_SAMPLES = 30 # PNG condition images for figures
SEED = 0
'''
    body = [
        '''
# ------------------------------------------------------- locate session 1
INDEX = KU.find_parsed_index()
if not INDEX:
    raise SystemExit(
        "Session 1 output not found. Add Input -> your `cca-s1-parsed` dataset "
        "(or Your Work -> Notebooks -> the session 1 notebook).")
print("INDEX =", INDEX)
recs, meta = C.load_index(INDEX)
print(f"{len(recs)} samples, vocab={meta.get('vocab')}, "
      f"crop={meta.get('crop_size')}")
VOCAB = meta.get("vocab", "face8")
''',
        '''
# ==================== CONTROL DETECTORS (sharded over cores) ====================
floor = " --splice-floor" if SPLICE_FLOOR else ""
inp   = f" --inpaint {INPAINT}" if INPAINT else ""
cmds, logs = [], []
for i in range(N_PROC):
    logs.append(f"{OUT}/logs/controls_{i}.log")
    cmds.append(
        f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
        f'--detector "{CONTROLS}" --out "{OUT}/run_controls" --tag main '
        f'--blend {BLEND} --dilate {DILATE} --jpeg-q {JPEG_Q} '
        f'--shard {i}/{N_PROC} --device cpu --vocab {VOCAB} '
        f'--time-budget-min {MAIN_BUDGET_MIN}{floor}{inp}')
KU.run_parallel(cmds, logs=logs, check=False)
KU.tail(logs[0], 12)
''',
        '''
# ==================== OPERATOR ABLATIONS ====================
# Each cell of the grid is a separate tag, so Session 11 reports them as rows.
if ABL_SAMPLES:
    jobs = []
    for b in ABL_BLENDS:
        jobs.append((f"blend_{b}", f"--blend {b} --dilate {DILATE} --jpeg-q {JPEG_Q}"))
    for d in ABL_DILATE:
        jobs.append((f"dilate_{d}", f"--blend {BLEND} --dilate {d} --jpeg-q {JPEG_Q}"))
    if ABL_NO_REENCODE:
        jobs.append(("no_reencode", f"--blend {BLEND} --dilate {DILATE} --jpeg-q 0"))
    per = max(1, ABL_BUDGET_MIN // max(1, len(jobs)))
    for tag, flags in jobs:
        cmds, logs = [], []
        for i in range(N_PROC):
            logs.append(f"{OUT}/logs/abl_{tag}_{i}.log")
            cmds.append(
                f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
                f'--detector "{CONTROLS}" --out "{OUT}/run_ablation" '
                f'--tag {tag} {flags} --limit-samples {ABL_SAMPLES} '
                f'--seed {SEED} --shard {i}/{N_PROC} --device cpu '
                f'--vocab {VOCAB} --time-budget-min {per}')
        print(C.banner(f"ablation: {tag}"))
        KU.run_parallel(cmds, logs=logs, check=False)
else:
    print("ablations skipped (ABL_SAMPLES = 0)")
''',
        '''
# ==================== GROUND-TRUTH LOCALIZATION (m10) ====================
KU.sh(f'{sys.executable} -m ccaudit.m10_localization --index "{INDEX}" '
      f'--out "{OUT}/loc"', check=False, log=f"{OUT}/logs/m10.log")
''',
        '''
# ==================== METRICS + REPORT ====================
raw = f"{OUT}/run_controls,{OUT}/run_ablation"
for extra, out in ((f'--localization "{OUT}/loc/localization.json" --by method',
                    f"{OUT}/metrics"),
                   ("--coarse", f"{OUT}/metrics")):
    KU.sh(f'{sys.executable} -m ccaudit.m6_metrics --raw "{raw}" '
          f'--out "{out}" --vocab {VOCAB} {extra}', check=False,
          log=f"{OUT}/logs/m6.log")
KU.sh(f'{sys.executable} -m ccaudit.m7_report '
      f'--metrics "{OUT}/metrics/metrics.json" '
      f'--coarse "{OUT}/metrics/metrics_coarse.json" '
      f'--by-method "{OUT}/metrics/metrics_by_method.json" '
      f'--raw "{OUT}/run_controls" --out "{OUT}/report_controls.html" '
      f'--figures-dir "{OUT}/figures" '
      f'--title "Counterfactual citation audit - control detectors"', check=False)
''',
        '''
# ---------------- INSTRUMENT CHECK ----------------
# The controls are constructed so that adaptive_oracle is faithful, the
# confabulator detects well but cites incorrectly, and fixed_oracle does not
# exceed adaptive_oracle on reverse citation.  The audit proceeds to the GPU
# sessions only if that ordering holds on real frames.
res = C.load_json(f"{OUT}/metrics/metrics.json", {}).get("results", [])
by = {r["detector"]: r for r in res}
print(f"{'detector':24s}{'AUC':>8}{'FS':>10}{'CR-prior':>10}  verdict")
for d, r in sorted(by.items()):
    v = ("FAITHFUL" if r.get("faithful") else
         "fwd only" if r.get("faithful_forward") else "UNFAITHFUL")
    print(f"{d:24s}{r.get('AUC',float('nan')):>8.3f}"
          f"{r.get('FS',float('nan')):>10.4f}"
          f"{r.get('CR_minus_prior',float('nan')):>10.3f}  {v}")

problems = []
ad = by.get("adaptive_oracle")
cf = by.get("confabulator")
fx = next((v for k, v in by.items() if k.startswith("fixed_oracle")), None)
if ad and not ad.get("faithful_forward"):
    problems.append("adaptive_oracle is not faithful in the forward direction")
if cf and abs(cf.get("FS", 0)) > 0.5 * abs((ad or {}).get("FS", 1)):
    problems.append("confabulator's FS is not clearly below adaptive_oracle's")
if cf and cf.get("AUC", 0) < 0.7:
    problems.append("confabulator's AUC is low: by construction it should detect "
                    "well and explain badly")
if fx and fx.get("CR_minus_prior", 0) > (ad or {}).get("CR_minus_prior", 1):
    problems.append("fixed_oracle exceeds adaptive_oracle on reverse citation")
for r in res:
    fp = r.get("floor_p_mean")
    if fp is not None and fp == fp and fp > 0.5:
        problems.append(f"{r['detector']}: floor p = {fp:.2f}; the operator "
                        f"may be introducing forgery evidence on authentic frames")
print()
if problems:
    print("!! INSTRUMENT PROBLEMS -- resolve these before the GPU sessions:")
    for p in problems:
        print("   -", p)
else:
    print("Instrument validation passed. Proceed to the GPU sessions.")
''',
        '''
# ---------------- condition images for figures ----------------
if MATERIALISE_SAMPLES:
    from ccaudit import m3_splice as M3
    from ccaudit import regions as R
    R.set_vocab(VOCAB)
    p = M3.build(INDEX, f"{OUT}/cond_sample", n_samples=MATERIALISE_SAMPLES,
                 dilate=DILATE, mode=BLEND, jpeg_q=JPEG_Q, with_floor=True,
                 inpaint_method=INPAINT or "")
    print("condition images ->", p)

try:
    from IPython.display import HTML, display
    if os.path.exists(f"{OUT}/report_controls.html"):
        display(HTML(open(f"{OUT}/report_controls.html").read()))
except Exception as exc:
    print(f"(inline report unavailable: {exc}); download "
          f"{OUT}/report_controls.html from the Output panel")
'''
    ]
    next_step = """
1. Save Version -> Save & Run All.
2. Output tab -> New Dataset -> `cca-s2-controls`.
3. If the instrument check above reported problems, resolve them first. The
   usual causes are a bad parse (inspect Session 1's overlay) or a mirror
   whose "fake" clips are not actually paired with the originals.
4. Otherwise continue with Session 3 (03_gpu_vlm_smoke.ipynb) to measure VLM
   throughput before committing GPU time.
"""
    return notebook(session_cells("S2 controls", "Session 2 - control detectors + ablations",
                                  intro, config, body, next_step), "none", True)


# --------------------------------------------------------------------------
# S3 -- VLM smoke matrix
# --------------------------------------------------------------------------

def s3() -> Dict[str, Any]:
    intro = """
**GPU T4 x2, ~1 h.** Loads each candidate VLM once and checks the four properties that
must hold before any GPU time is committed: probabilities in range, citations normalised
over the vocabulary, determinism, and measured throughput.

The samples-per-GPU-hour figure printed here is the basis for choosing `VLM_SAMPLES`
in Session 5.
"""
    config = '''
# ============================== CONFIG ==============================
MODELS = [
    "qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct",
    "hfvlm:Qwen/Qwen2-VL-2B-Instruct",
    "hfvlm:HuggingFaceTB/SmolVLM-Instruct",
    # "qwen25vl:Qwen/Qwen2.5-VL-7B-Instruct",   # needs QUANT="4bit"
]
QUANT      = "fp16"      # auto | fp16 | bf16 | 4bit
MAX_PIXELS = 384 * 384   # ~190 visual tokens at a 384 px crop
N_SMOKE    = 4           # real crops per model
'''
    body = [
        '''
KU.pip_install("transformers accelerate qwen-vl-utils")
if QUANT == "4bit":
    KU.pip_install("bitsandbytes")
KU.gpu_report()
''',
        '''
INDEX = KU.find_parsed_index()
print("INDEX =", INDEX or "(none: the smoke test will use random images)")
''',
        '''
# ==================== SMOKE MATRIX ====================
idx = f' --index "{INDEX}"' if INDEX else ""
results = []
for spec in MODELS:
    print(C.banner(spec))
    KU.sh(f'{sys.executable} -m ccaudit.m4_detectors --smoke-vlm "{spec}" '
          f'--quant {QUANT} --max-pixels {MAX_PIXELS} --n {N_SMOKE}{idx} '
          f'--out "{OUT}/smoke/{C.safe_name(spec)}.json"',
          check=False, log=f"{OUT}/logs/smoke.log")
''',
        '''
# ==================== BUDGET TABLE ====================
import glob
rows = []
for p in sorted(glob.glob(f"{OUT}/smoke/*.json")):
    for r in C.load_json(p).get("results", []):
        rows.append(r)
C.save_json(f"{OUT}/smoke/smoke_matrix.json", {"results": rows}, indent=1)

print(f"{'model':46s}{'load s':>8}{'s/call':>9}{'samp/GPU-h':>12}  status")
for r in rows:
    st = "OK" if r.get("ok") else "FAILED: " + "; ".join(r.get("errors", []))[:60]
    print(f"{r['spec'][:45]:46s}{r.get('load_sec',0):>8.0f}"
          f"{r.get('mean_sec_per_call',float('nan')):>9.3f}"
          f"{r.get('samples_per_gpu_hour',float('nan')):>12.0f}  {st}")

print("\\nSizing Session 5 (2 GPUs, 43 calls/sample with PROMPT_VARIANTS=5):")
for r in rows:
    sph = r.get("samples_per_gpu_hour") or float("nan")
    if sph == sph and sph > 0:
        for hrs in (2, 4, 7):
            print(f"  {r['spec'][:40]:42s} {hrs} GPU-h on 2 cards "
                  f"-> ~{2*hrs*sph*35/43:.0f} samples")
        break
'''
    ]
    next_step = """
1. Any model marked FAILED is excluded from the model set; record the reason.
2. Choose VLM_SAMPLES for Session 5 from the sizing table (600 is the default
   main size; 300 for the 7B model).
3. Output tab -> New Dataset -> `cca-s3-smoke` (small; keeps the throughput
   numbers with the rest of the provenance).
4. Continue with Session 4 (CNN) or directly with Session 5 (VLM main audit).
"""
    return notebook(session_cells("S3 VLM smoke", "Session 3 - VLM smoke matrix",
                                  intro, config, body, next_step), "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S4 -- CNN train + audit
# --------------------------------------------------------------------------

def s4() -> Dict[str, Any]:
    intro = """
**GPU T4, ~2-3 h.** Trains a conventional classifier on the **DEV** split of the same
crops, explains it with Grad-CAM projected onto the region vocabulary, and audits it
on **TEST** with the identical FS/CR metric.

This provides a detector with high detection AUC and a saliency-based explanation
against which the language-based explanations can be compared.
"""
    config = '''
# ============================== CONFIG ==============================
ARCH             = "efficientnet_b0"   # or "legacy_xception"
EPOCHS           = 4
BATCH_SIZE       = 32
LR               = 3e-4
INPUT_SIZE       = 256
TRAIN_BUDGET_MIN = 120

AUDIT_SAMPLES    = 0      # 0 = all TEST samples (a CNN call is ~10 ms)
AUDIT_BUDGET_MIN = 300
SPLICE_FLOOR     = True
INPAINT          = "telea"
CKPT             = ""     # "" = train; or the path of an existing checkpoint
'''
    body = [
        '''
KU.pip_install("timm")
KU.gpu_report()
INDEX = KU.find_parsed_index()
if not INDEX:
    raise SystemExit("Add Input -> `cca-s1-parsed`.")
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")
print("INDEX =", INDEX, "| samples:", len(recs))
''',
        '''
# ==================== TRAIN ON DEV ====================
# Re-uses a checkpoint from an attached earlier run if one is present.
found = [p for p in KU.find_checkpoints("cnn_*.pt")]
CKPT = CKPT or (found[0] if found else "")
if CKPT:
    print("resuming from", CKPT)
KU.sh(f'{sys.executable} -m ccaudit.m9_train_cnn --index "{INDEX}" '
      f'--out "{OUT}/cnn" --arch {ARCH} --epochs {EPOCHS} '
      f'--batch-size {BATCH_SIZE} --lr {LR} --input-size {INPUT_SIZE} '
      f'--train-budget-min {TRAIN_BUDGET_MIN}'
      + (f' --resume "{CKPT}"' if CKPT else ""),
      check=False, log=f"{OUT}/logs/m9.log")
log = C.load_json(f"{OUT}/cnn/train_log.json", {})
print("best val AUC:", log.get("best_val_auc"))
if (log.get("best_val_auc") or 0) < 0.9:
    print("!! val AUC below 0.9. Add epochs or try ARCH='legacy_xception' "
          "before auditing: a faithfulness measurement on a weak detector is "
          "of limited value.")
''',
        '''
# ==================== AUDIT ON TEST ====================
ckpt = f"{OUT}/cnn/cnn_{C.safe_name(ARCH)}.pt"
floor = " --splice-floor" if SPLICE_FLOOR else ""
inp   = f" --inpaint {INPAINT}" if INPAINT else ""
lim   = f" --limit-samples {AUDIT_SAMPLES}" if AUDIT_SAMPLES else ""
KU.sh(f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
      f'--detector "cnn:{ckpt}" --out "{OUT}/run_cnn" --tag main '
      f'--split test{lim} --device cuda --vocab {VOCAB} '
      f'--time-budget-min {AUDIT_BUDGET_MIN}{floor}{inp}',
      check=False, log=f"{OUT}/logs/cnn_audit.log")
''',
        '''
# ==================== INTERIM METRICS ====================
KU.sh(f'{sys.executable} -m ccaudit.m6_metrics --raw "{OUT}/run_cnn" '
      f'--out "{OUT}/metrics_cnn" --vocab {VOCAB}', check=False)
for r in C.load_json(f"{OUT}/metrics_cnn/metrics.json", {}).get("results", []):
    print(f"{r['detector']}: AUC={r.get('AUC'):.3f} FS={r.get('FS'):.4f} "
          f"CR-prior={r.get('CR_minus_prior'):.3f} "
          f"faithful={r.get('faithful')}")
'''
    ]
    next_step = """
1. Output tab -> New Dataset -> `cca-s4-cnn`.
2. Attach it to Session 11; the CNN appears as `cnn-<arch>` in every table.
3. If this notebook is re-run with `cca-s4-cnn` attached, the checkpoint is
   picked up automatically and training resumes instead of restarting.
"""
    return notebook(session_cells("S4 CNN", "Session 4 - CNN + Grad-CAM baseline",
                                  intro, config, body, next_step), "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S5 -- VLM main audit
# --------------------------------------------------------------------------

def s5() -> Dict[str, Any]:
    intro = """
**GPU T4 x2, 2-3 h per model, repeated until complete.** The main audit.

One process per card (`CUDA_VISIBLE_DEVICES`, shards `0/2` and `1/2`). Every model
call is cached by content, so a re-run continues rather than restarts; finished
samples are rebuilt from the cache without re-splicing.

**Run this notebook once per model.** Keep `TAG`, `SEED` and `VLM_SAMPLES` constant
across resumed runs of the same model, otherwise the cache does not line up.
"""
    config = '''
# ============================== CONFIG ==============================
MODEL        = "qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct"
QUANT        = "fp16"      # "4bit" for the 7B model
MAX_PIXELS   = 384 * 384
TAG          = "main"      # keep constant across resumed runs
VLM_SAMPLES  = 600         # keep constant across resumed runs
SEED         = 0           # keep constant across resumed runs
SPLIT        = "test"      # every reported number comes from TEST

PROMPT_VARIANTS = 5        # prompt-paraphrase stability; 0 = off
SPLICE_FLOOR    = True
USE_BOTH_GPUS   = True
TIME_BUDGET_MIN = 420      # 7 h; the runner stops cleanly and flushes its cache
'''
    body = [
        '''
KU.pip_install("transformers accelerate qwen-vl-utils")
if QUANT == "4bit":
    KU.pip_install("bitsandbytes")
KU.gpu_report()
INDEX = KU.find_parsed_index()
if not INDEX:
    raise SystemExit("Add Input -> `cca-s1-parsed`.")
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")

# Seed the cache from every attached previous run of this model.
prev = [d for d in KU.find_run_dirs() if "run_vlm" in d]
RESUME = ",".join(prev)
print("INDEX  =", INDEX)
print("resume =", RESUME or "(nothing attached; this is the first run)")
''',
        '''
# ==================== MAIN AUDIT (one process per GPU) ====================
n_gpu = KU.n_gpus() if USE_BOTH_GPUS else 1
n_gpu = max(1, n_gpu)
floor = " --splice-floor" if SPLICE_FLOOR else ""
res   = f' --resume-from "{RESUME}"' if RESUME else ""
cmds, envs, logs = [], [], []
for i in range(n_gpu):
    logs.append(f"{OUT}/logs/vlm/proc_{i}.log")
    envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
    cmds.append(
        f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
        f'--detector "{MODEL}" --out "{OUT}/run_vlm" --tag {TAG} '
        f'--split {SPLIT} --limit-samples {VLM_SAMPLES} --seed {SEED} '
        f'--shard {i}/{n_gpu} --device cuda --quant {QUANT} '
        f'--max-pixels {MAX_PIXELS} --prompt-variants {PROMPT_VARIANTS} '
        f'--vocab {VOCAB} --time-budget-min {TIME_BUDGET_MIN}{floor}{res}')
KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)
''',
        '''
# ==================== PROGRESS ====================
import glob
done = set()
for p in glob.glob(f"{OUT}/run_vlm/raw_*.json"):
    for r in C.load_json(p).get("rows", []):
        done.add(r["sample_id"])
print(f"VLM samples complete: {len(done)} / {VLM_SAMPLES}")
if len(done) < VLM_SAMPLES:
    print("\\nNOT FINISHED. Make a dataset from this output (Output tab -> New "
          "Dataset, e.g. `cca-s5-vlm`), attach it to this notebook, and Save & "
          "Run All again. Finished samples are rebuilt from the cache in "
          "seconds; only the remainder costs GPU time.")
else:
    print("\\nComplete. Make the dataset and continue.")
for l in logs:
    KU.tail(l, 6)
''',
        '''
# ==================== INTERIM METRICS ====================
KU.sh(f'{sys.executable} -m ccaudit.m6_metrics --raw "{OUT}/run_vlm" '
      f'--out "{OUT}/metrics_vlm" --split {SPLIT} --vocab {VOCAB}', check=False)
for r in C.load_json(f"{OUT}/metrics_vlm/metrics.json", {}).get("results", []):
    print(f"{r['detector']}: n={r.get('n_samples')} AUC={r.get('AUC'):.3f} "
          f"FS={r.get('FS'):.4f} [{r.get('FS_lo'):.4f},{r.get('FS_hi'):.4f}] "
          f"CR-prior={r.get('CR_minus_prior'):.3f} "
          f"faithful={r.get('faithful')}")
'''
    ]
    next_step = """
1. Output tab -> New Dataset -> `cca-s5-vlm-<model>` (a few MB: cache + rows).
2. If `VLM samples complete` is below VLM_SAMPLES, attach that dataset back to
   this notebook and Save & Run All again. Repeat until complete.
3. Then change MODEL and repeat for the next model in the set.
4. Attach every `cca-s5-vlm-*` dataset to Session 11; overlapping runs are
   de-duplicated by content key, so attaching more than needed is safe.
"""
    return notebook(session_cells("S5 VLM main", "Session 5 - VLM main audit",
                                  intro, config, body, next_step), "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S6 -- extras
# --------------------------------------------------------------------------

def s6() -> Dict[str, Any]:
    intro = """
**GPU T4 x2, ~3 h.** Three additional conditions, each run on a small subset under its
own tag so that Session 11 reports them as separate rows:

* **inpainting comparison** -- the counterfactual is produced by inpainting region k
  instead of splicing the authentic pixels, and the two operators are compared;
* **visual-token budget** -- whether faithfulness changes as the encoder sees fewer tokens;
* **crop control** -- whether removing background context changes the verdict.

Run this only after Session 5 has reached its sample target for this model.
"""
    config = '''
# ============================== CONFIG ==============================
MODEL        = "qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct"
QUANT        = "fp16"
MAX_PIXELS   = 384 * 384
SEED         = 0
SPLIT        = "test"

INPAINT_METHODS = ["telea", "lama"]   # [] = skip; lama installs a package
INPAINT_SAMPLES = 300

ABL_MAX_PIXELS  = [256 * 256]         # [] = skip
ABL_SAMPLES     = 150

CROP_CONTROL_INDEX = ""   # index.json from a CROP_SIZE=0 run of Session 1
CROP_SAMPLES       = 200

EXTRA_BUDGET_MIN = 150    # per extra
'''
    body = [
        '''
KU.pip_install("transformers accelerate qwen-vl-utils")
if "lama" in INPAINT_METHODS:
    KU.pip_install("simple-lama-inpainting")
KU.gpu_report()
INDEX = KU.find_parsed_index()
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")
RESUME = ",".join(d for d in KU.find_run_dirs() if "run_" in d)
n_gpu = max(1, KU.n_gpus())
print("INDEX =", INDEX, "| GPUs:", n_gpu)

def fan_out(tag, flags, samples, budget, out_sub="run_extras", index=None):
    cmds, envs, logs = [], [], []
    for i in range(n_gpu):
        logs.append(f"{OUT}/logs/{tag}_{i}.log")
        envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
        cmds.append(
            f'{sys.executable} -m ccaudit.m5_runner '
            f'--index "{index or INDEX}" --detector "{MODEL}" '
            f'--out "{OUT}/{out_sub}" --tag {tag} --split {SPLIT} '
            f'--limit-samples {samples} --seed {SEED} --shard {i}/{n_gpu} '
            f'--device cuda --quant {QUANT} --max-pixels {MAX_PIXELS} '
            f'--vocab {VOCAB} --time-budget-min {budget} {flags}'
            + (f' --resume-from "{RESUME}"' if RESUME else ""))
    print(C.banner(tag))
    KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)
''',
        '''
# ==================== INPAINTING COMPARISON ====================
for m in INPAINT_METHODS:
    fan_out(f"inpaint_{m}", f"--inpaint {m} --splice-floor",
            INPAINT_SAMPLES, EXTRA_BUDGET_MIN)
''',
        '''
# ==================== VISUAL-TOKEN BUDGET ====================
# The runner appends ":px<max_pixels>" to the cache key only when MAX_PIXELS
# differs from the default (384*384), so a non-default budget never reuses the
# default run's predictions.  Each budget still carries its own tag so that
# its rows are reported separately.
for px in ABL_MAX_PIXELS:
    cmds, envs, logs = [], [], []
    for i in range(n_gpu):
        logs.append(f"{OUT}/logs/px{px}_{i}.log")
        envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
        cmds.append(
            f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
            f'--detector "{MODEL}" --out "{OUT}/run_extras" --tag px{px} '
            f'--split {SPLIT} --limit-samples {ABL_SAMPLES} --seed {SEED} '
            f'--shard {i}/{n_gpu} --device cuda --quant {QUANT} '
            f'--max-pixels {px} --vocab {VOCAB} '
            f'--time-budget-min {EXTRA_BUDGET_MIN}')
    print(C.banner(f"token budget {px}"))
    KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)
''',
        '''
# ==================== CROP CONTROL ====================
# Requires a second Session 1 run with CROP_SIZE = 0 on a subset.
if CROP_CONTROL_INDEX:
    fan_out("fullframe", "", CROP_SAMPLES, EXTRA_BUDGET_MIN,
            index=CROP_CONTROL_INDEX)
else:
    print("crop control skipped. To run it: re-run Session 1 with "
          "CROP_SIZE = 0 and MAX_PAIRS = 200, make a dataset, attach it here "
          "and set CROP_CONTROL_INDEX to its parsed/index.json.")
''',
        '''
KU.sh(f'{sys.executable} -m ccaudit.m6_metrics --raw "{OUT}/run_extras" '
      f'--out "{OUT}/metrics_extras" --split {SPLIT} --vocab {VOCAB}',
      check=False)
for r in C.load_json(f"{OUT}/metrics_extras/metrics.json", {}).get("results", []):
    ip = r.get("inpaint") or {}
    print(f"{r['detector']} [{r.get('tag')}]: FS={r.get('FS'):.4f}"
          + (f"  | inpaint: corr={ip.get('corr_inpaint_vs_delta'):.3f} "
             f"sign-disagree={ip.get('sign_disagreement'):.3f} "
             f"p-rise-on-REAL={ip.get('p_rise_on_real'):.4f} "
             f"manufactures_evidence={ip.get('manufactures_evidence')}"
             if ip else ""))
'''
    ]
    next_step = """
1. Output tab -> New Dataset -> `cca-s6-extras`.
2. In the inpainting table, **p rise on REAL** measures the inpainter's effect
   on authentic frames. A confidence interval that excludes zero indicates
   that the inpainting operator itself introduces forgery evidence, which
   disqualifies it as a counterfactual operator; this is recorded as a
   property of the operator.
"""
    return notebook(session_cells("S6 extras", "Session 6 - inpainting, token budget, crop control",
                                  intro, config, body, next_step), "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S7 -- proxies + free text
# --------------------------------------------------------------------------

def s7() -> Dict[str, Any]:
    intro = """
**GPU T4 x2, ~3 h.** Two analyses that need only the suspect image.

* **Deployable proxies**: blur / noise / shuffle / selfcheck perturbations of region k,
  validated against the true FS. Each proxy's **floor** -- its effect on an *authentic*
  frame -- is measured as well, and a proxy with a positive floor is disqualified
  regardless of its correlation.
* **Free-text explanations**: the model explains itself in a sentence, which is mapped
  onto the same vocabulary so that FS can be recomputed for the stated explanation.
"""
    config = '''
# ============================== CONFIG ==============================
MODEL      = "qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct"
QUANT      = "fp16"
MAX_PIXELS = 384 * 384
SPLIT      = "test"
SEED       = 0

PROXIES        = "blur,noise,shuffle,selfcheck"
PROXY_SAMPLES  = 600
PROXY_BUDGET   = 210

FREE_TEXT_SAMPLES = 300
FREE_TEXT_BUDGET  = 120
'''
    body = [
        '''
KU.pip_install("transformers accelerate qwen-vl-utils")
KU.gpu_report()
INDEX = KU.find_parsed_index()
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")
RESUME = ",".join(d for d in KU.find_run_dirs() if "run_" in d)
n_gpu = max(1, KU.n_gpus())
''',
        '''
# ==================== PROXY CONDITIONS ====================
cmds, envs, logs = [], [], []
for i in range(n_gpu):
    logs.append(f"{OUT}/logs/proxy_{i}.log")
    envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
    cmds.append(
        f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
        f'--detector "{MODEL}" --out "{OUT}/run_proxy" --tag proxy '
        f'--split {SPLIT} --limit-samples {PROXY_SAMPLES} --seed {SEED} '
        f'--shard {i}/{n_gpu} --device cuda --quant {QUANT} '
        f'--max-pixels {MAX_PIXELS} --proxy "{PROXIES}" --splice-floor '
        f'--vocab {VOCAB} --time-budget-min {PROXY_BUDGET}'
        + (f' --resume-from "{RESUME}"' if RESUME else ""))
KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)
''',
        '''
# ==================== FREE TEXT ====================
cmds, envs, logs = [], [], []
for i in range(n_gpu):
    logs.append(f"{OUT}/logs/text_{i}.log")
    envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
    cmds.append(
        f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
        f'--detector "{MODEL}" --out "{OUT}/run_text" --tag freetext '
        f'--split {SPLIT} --limit-samples {FREE_TEXT_SAMPLES} --seed {SEED} '
        f'--shard {i}/{n_gpu} --device cuda --quant {QUANT} '
        f'--max-pixels {MAX_PIXELS} --free-text --vocab {VOCAB} '
        f'--time-budget-min {FREE_TEXT_BUDGET}')
KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)
''',
        '''
# ==================== ANALYSE ====================
KU.sh(f'{sys.executable} -m ccaudit.m11_proxy --raw "{OUT}/run_proxy" '
      f'--out "{OUT}/proxy" --vocab {VOCAB} --split {SPLIT}', check=False)
KU.sh(f'{sys.executable} -m ccaudit.m12_text_regions --raw "{OUT}/run_text" '
      f'--out "{OUT}/text" --vocab {VOCAB}', check=False)
'''
    ]
    next_step = """
1. Output tab -> New Dataset -> `cca-s7-proxy`.
2. In the proxy ranking, a deployable proxy must have both a high correlation
   with the true FS **and** a floor whose confidence interval includes zero.
   If every proxy is disqualified by its floor, that is recorded as the
   outcome of the analysis: it is the same defect that disqualifies an
   inpainting operator.
"""
    return notebook(session_cells("S7 proxy", "Session 7 - deployable proxies + free text",
                                  intro, config, body, next_step), "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S8 -- attribution + encoder probe
# --------------------------------------------------------------------------

def s8() -> Dict[str, Any]:
    intro = """
**GPU T4 x2, ~2.5-3.5 h.** The mechanism analyses.

* **stated vs attributed vs causal** -- the region the model cites, where its gradients
  place mass on the visual tokens, and which region moves the verdict;
* **overlay-elicited citation** -- the citation prompt refers to a region outlined on
  the pixels;
* **encoder blind-spot probe** -- whether the visual tokens covering region k move
  under the counterfactual at all; if they do not, no language component could
  have cited that region faithfully.

**Cost note:** `--encoder-probe` runs two extra vision-tower passes per region, which
adds roughly 30-50% to the run time. The passes are cached, so the cost is paid once.
"""
    config = '''
# ============================== CONFIG ==============================
MODELS = ["qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct",
          "hfvlm:Qwen/Qwen2-VL-2B-Instruct"]
QUANT      = "fp16"
MAX_PIXELS = 384 * 384
SPLIT      = "test"
SEED       = 0
SAMPLES    = 600
CITE_MODE  = "both"      # text | overlay | both
ATTRIBUTION   = True
ENCODER_PROBE = True
BUDGET_PER_MODEL_MIN = 210
'''
    body = [
        '''
KU.pip_install("transformers accelerate qwen-vl-utils")
KU.gpu_report()
INDEX = KU.find_parsed_index()
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")
RESUME = ",".join(d for d in KU.find_run_dirs() if "run_" in d)
n_gpu = max(1, KU.n_gpus())
''',
        '''
# ==================== ATTRIBUTION + PROBE ====================
flags = f"--cite-mode {CITE_MODE}"
if ATTRIBUTION:
    flags += " --attribution"
if ENCODER_PROBE:
    flags += " --encoder-probe"
for model in MODELS:
    cmds, envs, logs = [], [], []
    for i in range(n_gpu):
        tag = "attr"
        logs.append(f"{OUT}/logs/attr_{C.safe_name(model)}_{i}.log")
        envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
        cmds.append(
            f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
            f'--detector "{model}" --out "{OUT}/run_attr" --tag {tag} '
            f'--split {SPLIT} --limit-samples {SAMPLES} --seed {SEED} '
            f'--shard {i}/{n_gpu} --device cuda --quant {QUANT} '
            f'--max-pixels {MAX_PIXELS} {flags} --vocab {VOCAB} '
            f'--time-budget-min {BUDGET_PER_MODEL_MIN}'
            + (f' --resume-from "{RESUME}"' if RESUME else ""))
    print(C.banner(model))
    KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)
''',
        '''
# ==================== ANALYSE ====================
KU.sh(f'{sys.executable} -m ccaudit.m14_attribution --raw "{OUT}/run_attr" '
      f'--out "{OUT}/attr" --vocab {VOCAB}', check=False)
blob = C.load_json(f"{OUT}/attr/attribution.json", {})
for e in blob.get("results", []):
    a, b = e.get("agreement", {}), e.get("blind_spot", {})
    print(f"\\n{e['detector']}:")
    print(f"  stated = causal      {a.get('stated_vs_causal')}")
    print(f"  attributed = causal  {a.get('attr_vs_causal')}")
    print(f"  stated = attributed  {a.get('stated_vs_attr')}")
    if b:
        print(f"  encoder-blind        {b.get('encoder_blind_frac')}")
        print(f"  language confab      {b.get('language_confabulation_frac')}")
        print(f"  faithful             {b.get('faithful_frac')}")
'''
    ]
    next_step = """
1. Output tab -> New Dataset -> `cca-s8-attr`.
2. The three agreement rates (stated vs causal, attributed vs causal, stated vs
   attributed) are reported per model; they need not coincide, and the
   decomposition into encoder-blind and language-confabulation fractions
   locates the source of any disagreement.
3. If `attribute()` raised "could not locate the visual projector" for a model,
   that model has no gradient citation; report it as not supported rather
   than dropping the model.
"""
    return notebook(session_cells("S8 attribution", "Session 8 - attribution + encoder probe",
                                  intro, config, body, next_step), "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S9 -- CSET training
# --------------------------------------------------------------------------

def s9() -> Dict[str, Any]:
    intro = """
**GPU T4, ~5 h.** Causally-Supervised Explanation Tuning (CSET).

Three steps: audit the base model on **DEV** to measure Delta, build two label sets,
and QLoRA-tune one adapter per label set.

* `causal` -- the citation target is argmax Delta (the region that moves the verdict)
* `mask` -- the citation target is the region where the pixels changed (the standard supervision)

The verdict target is the true label in both, so detection is trained identically and
only the citation supervision differs. **DEV only** -- the module refuses to build a
training set containing TEST rows.
"""
    config = '''
# ============================== CONFIG ==============================
BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
DETECTOR    = "qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct"
QUANT       = "fp16"
MAX_PIXELS  = 384 * 384
SEED        = 0

DEV_SAMPLES     = 3000     # DEV audit size; 0 = all DEV
DEV_BUDGET_MIN  = 150

VARIANTS    = ["causal", "mask"]
MIN_DELTA   = 0.02         # drop DEV samples with no causal signal
LORA_R      = 16
LR          = 1e-4
EPOCHS      = 1
BATCH_SIZE  = 4
GRAD_ACCUM  = 4
TRAIN_BUDGET_MIN = 150     # per variant -> 150 + 2x150 = 7.5 h total
'''
    body = [
        '''
KU.pip_install("transformers accelerate qwen-vl-utils peft bitsandbytes")
KU.gpu_report()
INDEX = KU.find_parsed_index()
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")
n_gpu = max(1, KU.n_gpus())
print("DEV samples in index:", len(C.filter_split(recs, "dev")))
''',
        '''
# ==================== 1. AUDIT THE BASE MODEL ON DEV ====================
RESUME = ",".join(d for d in KU.find_run_dirs() if "run_dev" in d)
cmds, envs, logs = [], [], []
for i in range(n_gpu):
    logs.append(f"{OUT}/logs/dev_{i}.log")
    envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
    cmds.append(
        f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
        f'--detector "{DETECTOR}" --out "{OUT}/run_dev" --tag dev '
        f'--split dev --limit-samples {DEV_SAMPLES} --seed {SEED} '
        f'--shard {i}/{n_gpu} --device cuda --quant {QUANT} '
        f'--max-pixels {MAX_PIXELS} --vocab {VOCAB} '
        f'--time-budget-min {DEV_BUDGET_MIN}'
        + (f' --resume-from "{RESUME}"' if RESUME else ""))
KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)

# Ground truth for the `mask` variant, on DEV.
KU.sh(f'{sys.executable} -m ccaudit.m10_localization --index "{INDEX}" '
      f'--out "{OUT}/loc_dev" --split dev', check=False)
''',
        '''
# ==================== 2. INSPECT THE TRAINING SETS BEFORE TRAINING ====================
# --dry-run is inexpensive and detects an empty or lopsided label set before
# training time is spent on it.
for v in VARIANTS:
    loc = f' --localization "{OUT}/loc_dev/localization.json"' if v == "mask" else ""
    KU.sh(f'{sys.executable} -m ccaudit.m13_cset --raw "{OUT}/run_dev" '
          f'--index "{INDEX}" --out "{OUT}/cset" --variant {v} '
          f'--min-delta {MIN_DELTA} --vocab {VOCAB}{loc} --dry-run',
          check=False)
''',
        '''
# ==================== 3. QLoRA TRAINING ====================
for v in VARIANTS:
    loc = f' --localization "{OUT}/loc_dev/localization.json"' if v == "mask" else ""
    print(C.banner(f"CSET training: {v}"))
    KU.sh(f'{sys.executable} -m ccaudit.m13_cset --raw "{OUT}/run_dev" '
          f'--index "{INDEX}" --out "{OUT}/cset" --variant {v} '
          f'--base-model "{BASE_MODEL}" --min-delta {MIN_DELTA} '
          f'--lora-r {LORA_R} --lr {LR} --epochs {EPOCHS} '
          f'--batch-size {BATCH_SIZE} --grad-accum {GRAD_ACCUM} '
          f'--max-pixels {MAX_PIXELS} --seed {SEED} --vocab {VOCAB}{loc} '
          f'--time-budget-min {TRAIN_BUDGET_MIN}',
          check=False, log=f"{OUT}/logs/cset_{v}.log")
    log = C.load_json(f"{OUT}/cset/cset_{v}_log.json", {})
    print(f"  final loss {log.get('final_loss')}  steps {log.get('steps')}  "
          f"adapter {log.get('adapter_dir')}")
'''
    ]
    next_step = """
1. Output tab -> New Dataset -> `cca-s9-cset` (adapters are ~50 MB each).
2. Continue with Sessions 10a and 10b to audit base / cset-causal / cset-mask
   on TEST.
3. The two variants differ only in citation supervision, so the comparison
   isolates the effect of the supervision target; both outcomes are reported
   as measured, without re-tuning.
"""
    return notebook(session_cells("S9 CSET", "Session 9 - CSET training",
                                  intro, config, body, next_step), "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S10a / S10b -- CSET re-audit
# --------------------------------------------------------------------------

def s10(part: str, variants: List[str], intro_extra: str) -> Dict[str, Any]:
    intro = f"""
**GPU T4 x2, ~3.5 h.** Re-audit on TEST with the identical command used for the base
model, so that the comparison is like for like.

{intro_extra}

The full re-audit would take about 7 h, which is close to the 10 h wall once loading
and resume overhead are counted, so it is split across **10a** and **10b**. Each half
is an independent session with its own output dataset.
"""
    config = f'''
# ============================== CONFIG ==============================
BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
VARIANTS    = {variants!r}   # "" = the untuned base model
QUANT       = "fp16"
MAX_PIXELS  = 384 * 384
SPLIT       = "test"
SAMPLES     = 600
SEED        = 0
PROMPT_VARIANTS = 5
BUDGET_PER_VARIANT_MIN = 150
'''
    body = [
        '''
KU.pip_install("transformers accelerate qwen-vl-utils peft bitsandbytes")
KU.gpu_report()
INDEX = KU.find_parsed_index()
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")
n_gpu = max(1, KU.n_gpus())

# Locate the adapters produced by Session 9.
import glob
ADAPTERS = {}
for root in ("/kaggle/input", KU.work_dir()):
    for p in glob.glob(f"{root}/**/cset/*/adapter_config.json", recursive=True):
        ADAPTERS[os.path.basename(os.path.dirname(p))] = os.path.dirname(p)
print("adapters found:", ADAPTERS or "(none -- attach `cca-s9-cset`)")
# Seed the cache from earlier re-audit runs and from the Session 5 main run:
# the base-model command here is identical to Session 5's (same detector,
# split, samples, seed and settings), and cache keys are content-addressed,
# so an attached `cca-s5-vlm-*` dataset lets the base re-audit rebuild from
# cache instead of re-running the model.
RESUME = ",".join(d for d in KU.find_run_dirs()
                  if "run_cset" in d or "run_vlm" in d)
''',
        '''
# ==================== RE-AUDIT ====================
for v in VARIANTS:
    if v:
        adir = ADAPTERS.get(v)
        if not adir:
            print(f"skipping {v}: no adapter attached")
            continue
        det, tag = f"qwen25vl:{BASE_MODEL}:lora={adir}", f"cset_{v}"
    else:
        det, tag = f"qwen25vl:{BASE_MODEL}", "cset_base"
    cmds, envs, logs = [], [], []
    for i in range(n_gpu):
        logs.append(f"{OUT}/logs/{tag}_{i}.log")
        envs.append({"CUDA_VISIBLE_DEVICES": str(i)})
        cmds.append(
            f'{sys.executable} -m ccaudit.m5_runner --index "{INDEX}" '
            f'--detector "{det}" --out "{OUT}/run_cset" --tag {tag} '
            f'--split {SPLIT} --limit-samples {SAMPLES} --seed {SEED} '
            f'--shard {i}/{n_gpu} --device cuda --quant {QUANT} '
            f'--max-pixels {MAX_PIXELS} --prompt-variants {PROMPT_VARIANTS} '
            f'--splice-floor --vocab {VOCAB} '
            f'--time-budget-min {BUDGET_PER_VARIANT_MIN}'
            + (f' --resume-from "{RESUME}"' if RESUME else ""))
    print(C.banner(tag))
    KU.run_parallel(cmds, envs=envs, logs=logs, check=False, poll_sec=60)
''',
        '''
# ==================== INTERIM COMPARISON ====================
KU.sh(f'{sys.executable} -m ccaudit.m6_metrics --raw "{OUT}/run_cset" '
      f'--out "{OUT}/metrics_cset" --split {SPLIT} --vocab {VOCAB}', check=False)
print(f"{'variant':16s}{'AUC':>8}{'FS':>10}{'CR-prior':>10}")
for r in C.load_json(f"{OUT}/metrics_cset/metrics.json", {}).get("results", []):
    print(f"{str(r.get('tag')):16s}{r.get('AUC',float('nan')):>8.3f}"
          f"{r.get('FS',float('nan')):>10.4f}"
          f"{r.get('CR_minus_prior',float('nan')):>10.3f}")
print("\\nA drop in AUC of more than 0.01 relative to the base model is "
      "reported as a detection trade-off alongside any change in FS or CR.")
'''
    ]
    next_step = f"""
1. Output tab -> New Dataset -> `cca-s10{part}-cset`.
2. Run the other half ({'10b' if part == 'a' else '10a'}) if not already done.
3. Attach both to Session 11 for the final table.
"""
    return notebook(session_cells(f"S10{part} CSET audit",
                                  f"Session 10{part} - CSET re-audit",
                                  intro, config, body, next_step),
                    "nvidiaTeslaT4", True)


# --------------------------------------------------------------------------
# S11 -- final
# --------------------------------------------------------------------------

def s11() -> Dict[str, Any]:
    intro = """
**CPU, minutes.** Merges the outputs of every earlier session into the final tables,
figures and report.

Attach **every** output dataset: `cca-s1-parsed`, `cca-s2-controls`, `cca-s4-cnn`,
every `cca-s5-vlm-*`, `cca-s6-extras`, `cca-s7-proxy`, `cca-s8-attr`,
`cca-s10a-cset`, `cca-s10b-cset`. Overlapping runs are de-duplicated by content key,
so attaching more than needed is safe; attaching too few silently drops rows.
"""
    config = '''
# ============================== CONFIG ==============================
SPLIT        = "test"    # every reported number comes from TEST
TAU          = 0.5
TAUS         = "0.3,0.5,0.7"
N_BOOT       = 2000
SEED         = 0
HUMAN_STUDY_N = 100
HUMAN_DETECTOR = "qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct"  # the main VLM run of Session 5
ANNOTATORS   = 3
'''
    body = [
        '''
# ------------------------------------------------- find everything
INDEX = KU.find_parsed_index()
if not INDEX:
    raise SystemExit("Add Input -> `cca-s1-parsed`.")
recs, meta = C.load_index(INDEX)
VOCAB = meta.get("vocab", "face8")

RUN_DIRS = KU.find_run_dirs()
print(f"INDEX = {INDEX}\\n{len(recs)} samples, vocab={VOCAB}\\n")
print("run directories found:")
for d in RUN_DIRS:
    print("  ", d)
if not RUN_DIRS:
    raise SystemExit("No raw_*.json found. Attach the session outputs.")
RAW = ",".join(RUN_DIRS)
''',
        '''
# ------------------------------------------------- localization on TEST
KU.sh(f'{sys.executable} -m ccaudit.m10_localization --index "{INDEX}" '
      f'--out "{OUT}/loc" --split {SPLIT}', check=False,
      log=f"{OUT}/logs/m10.log")
LOC = f"{OUT}/loc/localization.json"
''',
        '''
# ------------------------------------------------- metrics
common = (f'--raw "{RAW}" --out "{OUT}/metrics" --split {SPLIT} --tau {TAU} '
          f'--taus {TAUS} --n-boot {N_BOOT} --seed {SEED} --vocab {VOCAB}')
KU.sh(f'{sys.executable} -m ccaudit.m6_metrics {common} '
      f'--localization "{LOC}" --by method', check=False,
      log=f"{OUT}/logs/m6.log")
KU.sh(f'{sys.executable} -m ccaudit.m6_metrics {common} --coarse',
      check=False, log=f"{OUT}/logs/m6.log")
''',
        '''
# ------------------------------------------------- side analyses
KU.sh(f'{sys.executable} -m ccaudit.m11_proxy --raw "{RAW}" '
      f'--out "{OUT}/proxy" --split {SPLIT} --vocab {VOCAB}', check=False)
KU.sh(f'{sys.executable} -m ccaudit.m12_text_regions --raw "{RAW}" '
      f'--out "{OUT}/text" --vocab {VOCAB}', check=False)
KU.sh(f'{sys.executable} -m ccaudit.m14_attribution --raw "{RAW}" '
      f'--out "{OUT}/attr" --vocab {VOCAB}', check=False)
''',
        '''
# ------------------------------------------------- report + csv
KU.sh(f'{sys.executable} -m ccaudit.m7_report '
      f'--metrics "{OUT}/metrics/metrics.json" '
      f'--coarse "{OUT}/metrics/metrics_coarse.json" '
      f'--by-method "{OUT}/metrics/metrics_by_method.json" '
      f'--raw "{RAW}" --out "{OUT}/report.html" '
      f'--csv "{OUT}/results_table.csv" --figures-dir "{OUT}/figures" '
      f'--title "Counterfactual citation audit - final report ({SPLIT})"', check=False)
''',
        '''
# ------------------------------------------------- human study materials
# HUMAN_DETECTOR selects the run whose items are shown to annotators; it is
# set explicitly in the config cell so that the study is tied to a named run.
det = f' --detector "{HUMAN_DETECTOR}"' if HUMAN_DETECTOR else ""
KU.sh(f'{sys.executable} -m ccaudit.m8_human_study --index "{INDEX}" '
      f'--raw "{RAW}" --out "{OUT}/human_study" --n-items {HUMAN_STUDY_N} '
      f'--split {SPLIT} --annotators {ANNOTATORS}{det}', check=False)
print("\\nDownload audit/human_study/ and distribute everything EXCEPT key.json.")
''',
        '''
# ------------------------------------------------- final table
res = C.load_json(f"{OUT}/metrics/metrics.json", {}).get("results", [])
print(f"{'detector':30s}{'tag':12s}{'n':>6}{'AUC':>8}{'FS':>10}"
      f"{'CR-prior':>10}  verdict")
for r in sorted(res, key=lambda x: (str(x.get('detector')), str(x.get('tag')))):
    v = ("FAITHFUL" if r.get("faithful") else
         "fwd only" if r.get("faithful_forward") else "UNFAITHFUL")
    print(f"{str(r.get('detector'))[:29]:30s}{str(r.get('tag'))[:11]:12s}"
          f"{r.get('n_samples',0):>6}{r.get('AUC',float('nan')):>8.3f}"
          f"{r.get('FS',float('nan')):>10.4f}"
          f"{r.get('CR_minus_prior',float('nan')):>10.3f}  {v}")

try:
    from IPython.display import HTML, display
    display(HTML(open(f"{OUT}/report.html").read()))
except Exception as exc:
    print(f"(inline report unavailable: {exc}); download "
          f"{OUT}/report.html from the Output panel")
'''
    ]
    next_step = """
Download from the Output panel:
  audit/report.html          self-contained, no external files
  audit/results_table.csv    the main results table
  audit/human_study/         distribute WITHOUT key.json
  audit/metrics/*.json       the complete per-metric output

Before reporting, check:
  * the control ordering still holds in section 2 of the report
  * floor p is low for every detector (otherwise the operator is suspect)
  * any detector whose AUC dropped after tuning is reported as a trade-off
"""
    return notebook(session_cells("S11 final", "Session 11 - final metrics, report, human study",
                                  intro, config, body, next_step), "none", False)


# --------------------------------------------------------------------------

def build_all() -> List[Any]:
    return [
        ("01_cpu_gate_and_parse.ipynb", s1()),
        ("02_cpu_controls.ipynb", s2()),
        ("03_gpu_vlm_smoke.ipynb", s3()),
        ("04_gpu_cnn_baseline.ipynb", s4()),
        ("05_gpu_vlm_main.ipynb", s5()),
        ("06_gpu_vlm_extras.ipynb", s6()),
        ("07_gpu_proxy_and_text.ipynb", s7()),
        ("08_gpu_attribution.ipynb", s8()),
        ("09_gpu_cset_train.ipynb", s9()),
        ("10a_gpu_cset_audit.ipynb",
         s10("a", ["", "causal"],
             "**10a** audits the untuned base model and the `causal` adapter.")),
        ("10b_gpu_cset_audit.ipynb",
         s10("b", ["mask"],
             "**10b** audits the `mask` adapter.")),
        ("11_cpu_final_report.ipynb", s11()),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the session notebooks from this file.")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR,
                    help="directory to write the .ipynb files into "
                         "(default: <repo>/notebooks/)")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    built = build_all()
    for name, nb in built:
        write(out_dir, name, nb)
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"  {name:32s} {len(nb['cells']):2d} cells ({n_code} code)")
    print(f"\n{len(built)} notebooks -> {out_dir}")


if __name__ == "__main__":
    main()
