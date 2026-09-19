# Running the audit

Operating guide for the eleven notebook sessions on a free-tier Kaggle-style
environment, and for the local CPU path. Each session's output becomes the
next session's input dataset, so no single run has to fit the whole project
into one wall-clock window or into the output size cap.

| Session | Accelerator | Duration | Content | Output dataset |
|---|---|---|---|---|
| S1 | CPU | 1-3 h | pairing gate + parse | `cca-s1-parsed` |
| S2 | CPU | 2-4 h | controls + ablations + localization | `cca-s2-controls` |
| S3 | GPU | ~1 h | VLM smoke matrix | `cca-s3-smoke` |
| S4 | GPU | 2-3 h | CNN train + audit | `cca-s4-cnn` |
| S5 | GPU | 2-3 h per model | VLM main audit (one run per model) | `cca-s5-vlm-<model>` |
| S6 | GPU | ~3 h | inpainting, token budget, crop control | `cca-s6-extras` |
| S7 | GPU | ~3 h | proxies + free text | `cca-s7-proxy` |
| S8 | GPU | ~3 h | attribution + encoder probe | `cca-s8-attr` |
| S9 | GPU | ~5 h | CSET: DEV audit + QLoRA training | `cca-s9-cset` |
| S10a | GPU | ~3.5 h | CSET re-audit: base + causal | `cca-s10a-cset` |
| S10b | GPU | ~3.5 h | CSET re-audit: mask | `cca-s10b-cset` |
| S11 | CPU | minutes | final metrics + report | `report.html` |

No session is budgeted above about 7.5 h of compute against the 10 h wall.
The CSET re-audit would take about 7 h in one session, which leaves too
little margin once model loading and resume overhead are counted, so it is
split into 10a and 10b.

---

## 1. Platform limits and how the code handles them

| Limit | Value (free tier) | What the code does |
|---|---|---|
| Session wall clock | ~10 h | every long step takes `--time-budget-min`; each session's budgets sum to at most 7.5 h; unfinished work is checkpointed and resumes |
| GPU quota | 30 h per week, T4 x2 or P100 | the VLM stage is budgeted in samples, not in "all data" |
| `/kaggle/working` | ~19 GB, kept as the output | only JSON, crops and reports are written here |
| `/kaggle/temp` | ~55 GB, erased at session end | model weights (`HF_HOME`) and scratch |
| RAM | ~30 GB | sufficient |
| Internet | off by default; requires phone verification | needed for `pip install`, the landmarker model and model weights |

The two GPUs have 15 GB each and no NVLink, so they are treated as two
independent cards. The code never model-parallelises; it runs one process
per card with `CUDA_VISIBLE_DEVICES=i` and `--shard i/2`.

Further platform facts:

* `/kaggle/input/...` is read-only. The notebooks copy anything they need to
  extend into `/kaggle/working`.
* Long runs should be started with **Save Version -> Save & Run All
  (Commit)**. The notebook then runs in the background and survives closing
  the browser; an interactive session left open can be lost.
* A notebook cannot attach its own output. Each session therefore ends by
  naming the dataset to create from its Output tab.
* No mount path is ever typed. Every notebook discovers its inputs by file
  name under `/kaggle/input`.

---

## 2. Upload the code as a dataset (once)

1. Zip the repository so that `ccaudit/` is at the top level:
   ```
   counterfactual-citation-audit.zip
     ├── ccaudit/           the package
     ├── notebooks/         the 12 session notebooks
     ├── scripts/           selftest.py, make_notebooks.py, check_platform_constraints.py, run_pipeline.sh
     ├── tests/
     └── docs/
   ```
2. **Datasets -> + New Dataset**, drag the zip in, give it a title (referred
   to below as `<your-code-dataset>`), and click **Create**.
3. Check that the dataset preview shows `ccaudit/kaggle_utils.py`. That is
   the marker file the auto-discovery looks for. An extra folder level from
   the zip is tolerated, but a dataset without that file is reported
   explicitly by the boot cell.
4. When the code changes: open the dataset, **New Version**, upload the new
   zip.

---

## 3. Enable Internet (once per notebook)

Right panel -> **Session options** -> **Internet** -> **On**. If the switch
is greyed out, verify the phone number under **Settings -> Phone
verification**.

Without Internet, `pip install mediapipe` fails and Session 1 cannot parse.
There is no geometric fallback parser: the code fails loudly rather than
produce boxes that do not sit on the face.

Offline alternative for the landmarker only: upload `face_landmarker.task`
(3.7 MB, from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`)
as a small dataset and attach it; `locate_landmarker()` finds it under
`/kaggle/input`. `mediapipe` itself must still be installable, so Internet
remains necessary.

---

## 4. Session 1: gate + parse (CPU)

**Create it**

1. **Code -> + New Notebook -> File -> Import Notebook**, upload
   `notebooks/01_cpu_gate_and_parse.ipynb`.
2. **+ Add Input** -> `<your-code-dataset>`.
3. **+ Add Input** -> the FaceForensics++ mirror (`<your-ff-mirror>`) or the
   Celeb-DF mirror (`<your-celebdf-mirror>`).
4. Session options: **Accelerator None**, **Internet On**.

**Configure.** The defaults are set for a full run. A first pass with
`MAX_PAIRS = 300` interactively (about 15 min) shows the gate verdict and
the overlay picture; then set `MAX_PAIRS = 0` and Save & Run All.

**Operational checks, in order**

1. `RESULT: N/N CHECKS PASSED` from the self-test.
2. `RESULT: PAIRING VERIFIED.` with `alignment_rate >= MIN_ALIGNMENT`
   (default 0.90).
3. `mediapipe backend ready (tasks API)` in the log.
4. The overlay picture: eyes, nose, mouth, skin, jaw, hair and ears must sit
   on the face. This is the single most informative output of the session.
5. `samples written` of about 2 x clips; `parser_iou_mean` and
   `manip_coverage_mean` above their gates (`MIN_IOU = 0.70`,
   `MIN_COVERAGE = 0.50`).
6. Output well under 19 GB (of the order of 1 GB for 10k crops at 384 px).

**Pairing gate (decision gate 1).** If the gate fails, read PHASE 5 in the
log. It prints a per-pair diagnostic table and a ranked list of remedies. In
order: exclude a bad method with `METHODS`; check for `CENTRE IDENTICAL`
(the mirror's manipulated folder contains copies of the originals, so the
mirror is unusable); check for a large `nframes` mismatch (the mirror was
re-encoded at a different frame rate, which an integer offset cannot fix).
In either of the last two cases the remedy is to change mirrors. Only if the
failures resemble borderline codec noise should `MIN_ALIGNMENT` be lowered,
and a lowered value must be recorded with the results.

**If the session is killed mid-parse.** The parser checkpoints every 50
clips. Save Version, make a dataset from that version's output, attach it,
set `RESUME_FROM = "/kaggle/input/<mount>/audit/parsed/index.json"`, and
re-run. Finished clips are skipped.

**To halve the wall time.** Run two copies with `SHARD = "0/2"` and
`"1/2"`, make two datasets, and attach both everywhere downstream;
`index.json` files merge.

**Operator check on parsed frames.** The self-test proves the splice
operator on a synthetic fixture. To prove it on the parsed frames, run from
a notebook cell:

```
python -m ccaudit.m3_splice --index audit/parsed/index.json --check 50
```

It must print `all invariants hold on real frames`. If it reports samples
where nothing changed, the fake and real crops are identical there and the
pairing is suspect even though the gate passed.

**Then:** Output tab -> **New Dataset** -> `cca-s1-parsed`.

---

## 5. Session 2: controls + ablations (CPU)

Import `02_cpu_controls.ipynb`. Inputs: `<your-code-dataset>` and
`cca-s1-parsed`. Accelerator **None**.

**Instrument validation (decision gate 2).** This session decides whether to
commit GPU quota. The four control detectors have known faithfulness by
construction, so the metric must reproduce their ordering on the parsed
frames:

| control | constructed as | check |
|---|---|---|
| `adaptive_oracle` | faithful | `FS > 0` and `CR > prior` |
| `fixed_oracle[mouth]` | forward only | `FS > 0`, `CR - prior` not above `adaptive_oracle`'s |
| `confabulator` | discriminative verdict, unfaithful citation | `FS` well below `adaptive_oracle`'s, with AUC above 0.7 |
| `dummy` | no signal | `FS` about 0 and AUC about 0.5 |
| floor p | all detectors | low (the operator does not manufacture forgery evidence on authentic frames) |

The notebook checks this and prints either `Instrument validation passed.
Proceed to the GPU sessions.` or a list of problems. If the ordering is
wrong, the masks or the data are the cause; inspect Session 1's overlay
before anything else.

**Then:** Output tab -> **New Dataset** -> `cca-s2-controls`.

---

## 6. Session 3: VLM smoke matrix (GPU, ~1 h)

Import `03_gpu_vlm_smoke.ipynb`. Accelerator **GPU T4 x2** (not P100: no
usable fp16 tensor cores and no int4 kernels). Internet **On**.

The session loads each candidate model once and checks: `p` in `[0, 1]`;
citations normalised over the vocabulary; determinism across two identical
calls (fp16 jitter under 1e-4 is tolerated and printed, anything larger
fails); and measured throughput.

The notebook then prints a sizing table: how many samples each model
processes in 2, 4 and 7 GPU-hours on two cards. `VLM_SAMPLES` for Session 5
is chosen from it. A model that fails here is excluded from the model set,
and the reason is recorded.

**Cost model.** Per sample: 2 original verdicts + 8 regions x 3 conditions
+ 1 citation + 8 reverse citations = 35 calls, +8 with `SPLICE_FLOOR`, +8
with `PROMPT_VARIANTS = 5`, giving 43. With a 3B model in fp16 on a T4 at
`MAX_PIXELS = 384 * 384` (about 190 visual tokens) a call takes roughly
0.4-0.7 s, so the two cards together process about 300-450 samples per
hour. A 7B model in 4-bit is 3-4 times slower.

---

## 7. Sessions 4-10: the audits

Each follows the same shape: import the notebook, attach
`<your-code-dataset>` + `cca-s1-parsed` + whatever earlier outputs it names,
set the configuration cell, Save & Run All, then make a dataset from the
output.

| Session | Notebook | Key configuration | Additional inputs |
|---|---|---|---|
| S4 CNN | `04_gpu_cnn_baseline.ipynb` | `ARCH`, `EPOCHS`, `TRAIN_BUDGET_MIN`, `CKPT` | `cca-s4-cnn` from an earlier run reuses its checkpoint |
| S5 VLM main | `05_gpu_vlm_main.ipynb` | `MODEL`, `VLM_SAMPLES`, `TAG`, `SEED`, `QUANT` | previous `cca-s5-vlm-<model>` to resume |
| S6 extras | `06_gpu_vlm_extras.ipynb` | `INPAINT_METHODS`, `ABL_MAX_PIXELS`, `CROP_CONTROL_INDEX` | `cca-s5-vlm-<model>` (cache) |
| S7 proxies | `07_gpu_proxy_and_text.ipynb` | `PROXIES`, `PROXY_SAMPLES`, `FREE_TEXT_SAMPLES` | `cca-s5-vlm-<model>` (cache) |
| S8 attribution | `08_gpu_attribution.ipynb` | `MODELS`, `CITE_MODE`, `ATTRIBUTION`, `ENCODER_PROBE` | `cca-s5-vlm-<model>` (cache) |
| S9 CSET train | `09_gpu_cset_train.ipynb` | `VARIANTS`, `MIN_DELTA`, `DEV_SAMPLES` | none beyond `cca-s1-parsed` |
| S10a / S10b CSET audit | `10a_gpu_cset_audit.ipynb`, `10b_gpu_cset_audit.ipynb` | `VARIANTS`, `SAMPLES` | `cca-s9-cset` (adapters); `cca-s5-vlm-<model>` lets the base re-audit rebuild from cache |

**Resuming Session 5.** Run it once per model. If it prints `VLM samples
complete: X / VLM_SAMPLES` with X short of the target:

1. Output tab -> **New Dataset** -> `cca-s5-vlm-<model>`;
2. **+ Add Input** -> that dataset;
3. **Save & Run All** again.

The runner finds every `cache_*.json` under `/kaggle/input`, seeds its cache,
and skips finished samples without re-splicing them; a finished sample is
rebuilt from the cache in milliseconds. Repeat until complete.

`TAG`, `SEED`, `VLM_SAMPLES`, `MODEL`, `QUANT` and `MAX_PIXELS` must stay
constant across resumed runs. The cache key includes the detector name (with
resolution and quantisation suffixes when they differ from the defaults),
and the sample subset is chosen by a seeded hash. Changing any of them
starts the run over, which is the cause of a `VLM samples complete` count
that does not grow between runs.

**Cost note for S8.** `--encoder-probe` runs two extra vision-tower passes
per region; budget a 30-50% surcharge. The passes are cached, so the cost is
paid once.

**S9 is DEV-only.** The CSET module refuses to build a training set
containing TEST rows. Its `--dry-run` output (printed before training) shows
the label sets; check that they are neither empty nor lopsided before
committing hours of GPU time.

---

## 8. Session 11: final report (CPU, minutes)

Import `11_cpu_final_report.ipynb`. Attach **every** dataset made so far:
`cca-s1-parsed`, `cca-s2-controls`, `cca-s4-cnn`, every `cca-s5-vlm-*`,
`cca-s6-extras`, `cca-s7-proxy`, `cca-s8-attr`, `cca-s10a-cset`,
`cca-s10b-cset`. Accelerator **None**. Run All.

The session merges every `raw_*.json` it finds; computes the primary table
at fine and coarse granularity with cluster-bootstrap CIs over clips, one
row per manipulation method, the inpainting comparison, prompt stability,
proxies, citation agreement, the blind-spot decomposition, citation entropy
and the tau sweep; writes `report.html` and `results_table.csv`; and builds
the human-study materials.

Overlapping runs are de-duplicated by content key, so attaching too many
datasets is safe; attaching too few silently drops rows. Check the printed
list of run directories against the sessions that were run.

---

## 9. Reading the output

```
audit/
├── m1_report.json           gate verdict + per-pair alignment table
├── pairs.json               resolved (fake, real) clip pairs + frame offsets
├── overlay.png              contact sheet of the parse
├── parsed/
│   ├── index.json           one record per sample; paths relative to this file
│   ├── parse_stats.json     drops by reason
│   └── <method>/<pair>/f<frame>_{real,fake}.jpg + _lab.png
├── run_controls/            raw_<det>[_<tag>][_sXofY].json, cache_*, runstats_*, provenance_*
├── run_ablation/ run_vlm/ run_cnn/ run_extras/ run_proxy/ run_text/ run_attr/ run_dev/ run_cset/
├── loc/localization.json    ground-truth manipulation per region (loc_dev/ for the DEV audit)
├── smoke/smoke_matrix.json  VLM smoke results and throughput
├── cnn/cnn_<arch>.pt        checkpoint (trained on DEV), train_log.json
├── cset/<variant>/          LoRA adapters
├── metrics/metrics.json, metrics_coarse.json, metrics_by_method.json
├── proxy/, text/, attr/attribution.json
├── logs/                    pip_freeze.txt, gpu.txt, per-command logs
├── report.html, results_table.csv, figures/*.svg
├── human_study/             items/*.png, annotations.csv, INSTRUCTIONS.txt, key.json
└── cond_sample/             PNG condition images for figures
```

**Two verdict columns.** The primary table's `verdict` is uncorrected.
Section 15 of the report gives the Holm-corrected verdict across the
detector zoo (`faithful_holm`), which is the one to report; a detector that
is faithful only before correction is not claimed as faithful. Section 16
reports the additivity residual: if its CI excludes zero at radii that both
cover the manipulation, the subtraction `delta_raw - delta_seam` is biased,
which is a finding about the operator rather than about any detector.

**Column meanings.** **AUC**: fake-versus-real separation on the originals.
**FS**: seam-corrected effect at the cited region minus the mean effect
elsewhere. **rank**: position of the cited region by effect size (1 is
best). **CR**: reverse citation recall, against **prior**, the best constant
guess. **|seam|**: identity-splice effect. **floor p**: manipulation
probability on identity-spliced authentic frames, which must stay low or the
operator is manufacturing evidence. Faithful requires FS and CR - prior both
positive with CIs excluding zero.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not find the ccaudit package` / `... ccaudit code under /kaggle/input` | code dataset not attached, or an old version | Add Input -> `<your-code-dataset>`; check the preview shows `ccaudit/kaggle_utils.py` |
| `ModuleNotFoundError: No module named 'ccaudit'` inside a subprocess | `PYTHONPATH` not set | the boot cell and `kaggle_utils.sh` inject it; a module called by hand must be run from a cell, not a terminal |
| `Could not find the dataset` | mirror not attached, or an unusual layout | Add Input -> the mirror, or set `DATA` by hand to the folder that directly contains `original/` and the manipulation folders |
| `pip install mediapipe` fails | Internet off | Session options -> Internet -> On |
| `face_landmarker.task could not be downloaded` | Internet off | turn it on, or upload the file as a dataset |
| `ImportError ... tensorflow ... doc_controls` | TF/mediapipe clash on the image | handled by a shim; if it persists, `pip install -U mediapipe` |
| Overlay boxes not on the face | the parse is wrong | check the log for `mediapipe backend ready (tasks API)`; there is no fallback parser, so the landmarker itself failed |
| `no_face_real` / `no_face_fake` drops very high in `parse_stats.json` | tight crops or profile views | occurs to a degree; record the drop rate |
| `WARNING: N pairs point at files that do not exist` | dataset mounted elsewhere than when Module 1 ran | the notebook passes `--data-root`; add it when running by hand |
| `CUDA out of memory` | 7B without 4-bit, or `MAX_PIXELS` too high | `QUANT = "4bit"`, or a 3B model, or `MAX_PIXELS = 320 * 320` |
| smoke test `NON-DETERMINISTIC` | kernel non-determinism above 1e-4 | record it; try `fp16` instead of `4bit` |
| `STOPPED EARLY on time budget` | intended | make a dataset from the output, attach it, re-run |
| VLM progress does not grow | `TAG`, `SEED`, `VLM_SAMPLES`, `MODEL`, `QUANT` or `MAX_PIXELS` changed | keep them constant across resumed runs |
| `/kaggle/working` above 19 GB | `CROP_SIZE = 0`, or too many materialised samples | keep crops on; materialise at most 100; large files go to `/kaggle/temp` |
| The platform refuses to attach a notebook to itself | platform rule | make a dataset from the Output tab |
| `could not locate the visual projector` | that model has no hookable projector | it has no gradient citation; record it as unsupported rather than dropping the model |

---

## 11. Running without the platform

The local CPU path covers the gate, the parse, the control detectors,
localization, metrics and the report. The GPU stages are per-model and live
in the notebooks; each module is also a CLI and can be run by hand on any
machine with a CUDA device.

Dependencies: `numpy`, `opencv-python`, `mediapipe` for the CPU path;
`torch`, `transformers`, `accelerate`, `qwen-vl-utils`, `peft`,
`bitsandbytes` and `timm` for the GPU stages; `simple-lama-inpainting` for
the `lama` inpainter.

```bash
python scripts/selftest.py                # synthetic fixture, under a minute
python scripts/check_platform_constraints.py   # static audit of notebook budgets and paths

python -m ccaudit.m1_verify --root /data/FF++_C23 --emit-pairs work/pairs.json --fail-hard
python -m ccaudit.m2_parse  --pairs work/pairs.json --out work/parsed --data-root /data/FF++_C23
python -m ccaudit.m5_runner --index work/parsed/index.json \
    --detector adaptive_oracle,fixed_oracle:mouth,confabulator,dummy \
    --out work/run_controls --tag main --splice-floor --inpaint telea --device cpu
python -m ccaudit.m10_localization --index work/parsed/index.json --out work/loc
python -m ccaudit.m6_metrics --raw work/run_controls --out work/metrics \
    --localization work/loc/localization.json --by method
python -m ccaudit.m7_report --metrics work/metrics/metrics.json --out work/report.html --figures-dir work/figures

# or, in one step (self-test, gate, parse, sharded controls, localization, metrics, report):
./scripts/run_pipeline.sh /data/FF++_C23 ./work [MAX_PAIRS]
```

`scripts/run_pipeline.sh` runs the control detectors with one shard per CPU
core and writes `report.html` under the work directory. Section 2 of that
report shows the control detectors; their ordering must hold before any GPU
stage is run. `tests/make_fixture.py <out_dir>` builds a synthetic
FF++-shaped mirror, including a deliberately misaligned method that the
pairing gate must reject, so the whole CPU path can be exercised without the
real dataset.

Every module prints `--help`.

---

## 12. Before reporting results

* Confirm the control ordering holds in section 2 of the final report.
* Confirm `floor p` is low for every detector.
* Apply the three operational gates: if the pairing gate fails, change
  mirrors; if the instrument validation fails, fix the parse before spending
  GPU quota; if a model fails the smoke matrix, exclude it and record why.
* State the protocol alongside the numbers: face crops (384 px, margin 0.35,
  computed on the real frame and shared by all conditions); frames per clip
  with clips as the bootstrap unit; the DEV/TEST split rule and seed; the
  prompt paraphrase set; operator settings (Poisson, r = 3, q = 90) and the
  ablation grid; tau = 0.5 with the sweep; the sample size and its
  GPU-budget rationale; any change to `MIN_ALIGNMENT`.
* **Data licence.** Public mirrors of FaceForensics++ and Celeb-DF are
  third-party redistributions. Cite the original datasets, state which
  mirror was used, and do not release crops or counterfactual images
  without checking the datasets' terms. The code, `index.json`, the caches
  and the metrics reproduce everything from a licensed copy.
