# Code architecture

Reference for the `ccaudit` package: what the pipeline computes, how the
modules fit together, and the interfaces (CLI flags, JSON records, dictionary
keys) that other modules and the session notebooks depend on.

---

## 1. What the pipeline computes

Face-swap benchmarks manipulate a *target* video, so every fake frame has a
frame-aligned authentic twin. The counterfactual "region *k* is not
manipulated" is therefore obtainable by splicing the authentic region back
into the fake frame, without synthesising any pixels. From the detector's
manipulation probability `p(.)` on the fake frame `x^f` and on the spliced
conditions:

```
delta_raw (k) = p(x^f) - p(x~real_k)        removing region k's manipulation
delta_seam(k) = p(x^f) - p(x~identity_k)    the seam and the extra JPEG pass alone
delta     (k) = delta_raw - delta_seam  =  p(x~identity_k) - p(x~real_k)

FS (per sample) = delta(cited) - mean_{k != cited} delta(k)
CR              = over rows where the reverse condition flipped the verdict
                  (p_reverse >= tau), the frequency with which the model cites
                  the injected region, compared against the prior (always
                  guessing the most common injected region)

faithful  <=>  FS_lo > 0  AND  CR_lo > CR_prior
```

Because the difference-in-differences cancels `p(x^f)`, the baseline's
compression history and the extra JPEG pass drop out algebraically. This
holds only if both conditions follow the same JPEG path; one condition's
compression path must never be changed without changing the other's.

---

## 2. Package layout

```
ccaudit/
  regions.py            vocabulary registry + landmark -> region geometry
  common.py             atomic IO, relative-path index, split, bootstrap, budgets
  kaggle_utils.py       input auto-discovery, subprocess runner, disk/GPU reporting
  m1_verify.py          pairing gate                    -> pairs.json
  m2_parse.py           mediapipe parsing + crops       -> parsed/index.json
  m3_splice.py          the splice operator             (library, used in memory)
  m4_detectors.py       detector zoo + prompts
  m5_runner.py          resumable sharded audit         -> raw_*.json, cache_*.json
  m6_metrics.py         all metrics                     -> metrics.json
  m7_report.py          self-contained HTML + SVG       -> report.html
  m8_human_study.py     annotator materials
  m9_train_cnn.py       CNN baseline on the DEV split
  m10_localization.py   ground-truth manipulation per region
  m11_proxy.py          deployable proxy ranking
  m12_text_regions.py   free text -> region distribution
  m13_cset.py           causally supervised explanation tuning (QLoRA)
  m14_attribution.py    stated vs attributed vs causal, blind spots
notebooks/              12 session notebooks, generated
scripts/
  make_notebooks.py     source of truth for notebooks/*.ipynb
  selftest.py           pipeline self-test on a synthetic fixture
  check_platform_constraints.py   static audit of session budgets and paths
  run_pipeline.sh       local CPU pipeline (gate -> parse -> controls -> report)
tests/
  make_fixture.py       synthetic FF++-shaped mirror
  fake_parser.py        mediapipe test double
docs/                   this file, RUNNING.md, PROPOSAL.md
```

Every module is a CLI: `python -m ccaudit.<module> --help`. The scripts insert
the repository root into `sys.path`, so they run from any working directory.

`scripts/make_notebooks.py` is the source of truth for the notebooks. An edit
made directly to an `.ipynb` file is reverted the next time the generator
runs, so all notebook changes go into the generator.

---

## 3. `regions.py`

```python
FACE8 = [left_eye, right_eye, nose, mouth, skin, jaw_boundary, hair, ears]  # K = 8
GRID9 = [r0c0 ... r2c2]                                    # 3 x 3 grid, second domain
BACKGROUND = 255
```

The active vocabulary is process-global: `set_vocab(name)`, `active_vocab()`,
`get_vocab(name=None)`, `num_regions()`, `rid(region)`, `region_of(id)`,
`coarse_map()`, `coarse_vocab()`, `prompt_name(region)`, `letters(n)`. Every
CLI entry point calls `set_vocab()` once from its `--vocab` flag, and the
vocabulary name is written into every output JSON.

Coarse map (face8): `{left_eye, right_eye} -> eyes; nose -> nose;
mouth -> mouth; {skin, jaw_boundary, hair, ears} -> rest`.

Eye naming is by image side, not anatomy: `left_eye` is the eye on the left
of the picture (the subject's right). A VLM describing an image means the eye
it sees on the left. The parser assigns the two eye hulls by x-centroid at
runtime, so the convention holds regardless of which mediapipe index set is
which.

`build_face8_labels(pts, shape)` paints a disjoint label map from 468+
landmarks. Regions are painted in increasing priority so disjointness is
structural: hair, ears, skin, jaw_boundary, nose, eyes, mouth. Hair is a
band above the forehead arc; ears are boxes flanking the face oval between
eye and mouth level; jaw_boundary is the inner ring of the oval.
`build_grid9_labels(shape)` tiles the image. `region_areas(lab)` and
`present_regions(lab, min_px)` read a label map back.

---

## 4. `common.py`

Four rules are enforced here rather than in each module:

* **Atomic writes** (`save_json`, `imwrite_jpeg`, `imwrite_png`): temporary
  file plus `os.replace`, so a session killed at the wall-clock limit never
  leaves a half-written JSON.
* **Relative paths in indices** (`save_index` / `load_index` / `resolve_path`
  / `merge_indices`): the fields `real`, `fake`, `lab` are stored relative to
  the index file, so an output folder keeps working after the platform
  remounts it read-only under another prefix.
* **Deterministic split**: `split_of(pair_id)` returns `dev` (30%) or `test`
  (70%), hashed from `split_key(pair_id)`, which collapses `033_097` and
  `097_033` to one key (FF++) and `idX_idY_SSSS` to its target clip
  `idX_SSSS` (Celeb-DF). No identity can appear in both splits and nothing is
  stored. `filter_split(records, split)` applies it.
* **Cluster bootstrap**: `bootstrap_ci(values, groups=pair_ids, n_boot,
  alpha, seed, statistic)` resamples clips, never frames, because two frames
  of one clip are near-duplicates. `bootstrap_p_two_sided` is the matching
  test.

Also: `stable_id(*parts)` (every cache key), `safe_name(s)`,
`shard(items, "i/n")` (by content hash, so adding samples does not reshuffle
existing shard membership), `limit_samples(records, n, seed)` (subset
independent of input order, so every shard and detector audits the same
samples), `holm_bonferroni`, `auc_score`, `spearman`, `pearson`,
`kendall_tau`, `entropy_norm`, `jpeg_roundtrip(img, q)` (the only compression
path; `q <= 0` disables it), `Budget(minutes)` (hard wall-clock; `expired`,
`report()`), `provenance()` / `code_hash()` / `write_provenance()`, and
`scratch_dir()` (a large session-local directory: `/kaggle/temp` when
present, otherwise `/kaggle/tmp`, `/tmp` or the system temporary directory).

---

## 5. `kaggle_utils.py`

Auto-discovery by file name, never by mount path: `find_code_dir()` looks for
the marker file `ccaudit/kaggle_utils.py`; `find_parsed_index()`,
`find_all_parsed_indices()`, `find_run_dirs(prefix="raw_")`,
`find_caches()`, `find_dataset_root()`, `find_celebdf_root()`,
`find_landmarker_task()` and `find_checkpoints()` search `/kaggle/input`.

`sh(cmd, check, env, log, echo, cwd)` and `run_parallel(cmds, envs, logs)`
launch child processes. Both inject `PYTHONPATH` through `_child_env()`: a
subprocess does not inherit the parent's `sys.path`, and on the platform the
code lives under `/kaggle/input/<slug>/`, so `python -m ccaudit.<module>`
would otherwise fail with `ModuleNotFoundError`. The notebooks' boot cell sets
the same variable.

`session_header(name, out_dir)` prints the attached inputs, GPUs and disk,
and writes `pip freeze` and `nvidia-smi` output into `<out_dir>/logs/`.
`work_dir(sub)` and `temp_dir(sub)` resolve the output and scratch roots.

Constants: `OUTPUT_CAP_GB = 19`, `SESSION_WALL_H = 10`.

---

## 6. `m1_verify.py` (the pairing gate)

Pairing rules. FF++: `<target>_<source>.mp4` is built on the target video, so
the twin is `original/<target>.mp4`. Celeb-DF: `idX_idY_SSSS` pairs with
`Celeb-real/idX_SSSS`. The layout is auto-detected (`--layout auto | ffpp |
celebdf`); both the nested `manipulated_sequences/<M>/c23/videos/` layout and
flat mirror layouts are accepted.

Verification decodes `--probes` frames at identical indices from both clips
and compares the border ring (outer `--border-frac`, default 15%, which no
face manipulation touches). Four failure modes are handled structurally:

1. **Inexact seeking.** `CAP_PROP_POS_FRAMES` lands on a keyframe and the two
   clips can have different GOP structures, so the same requested index
   returns different frames. `read_frames_at()` never seeks; it decodes
   sequentially with `grab()` and calls `retrieve()` only at the wanted
   indices.
2. **Resolution mismatch.** The fake frame is resized to the real frame's
   size. Cropping to a common top-left corner would compare different parts
   of the scene.
3. **Resampling.** A resized clip has different high-frequency content, so
   the metric is evaluated on a low-pass ladder (sigma = 0, 1, 2) and the best
   scale is reported as `scale_used`.
4. **Frame offset.** A small offset search (0, +-1, +-2); a pair rescued at
   offset *o* has that offset written into `pairs.json`, and Module 2 applies
   it.

Gate: `aligned <=> (NCC > 0.95 and MAE < 0.05) or (NCC > 0.80 and MAE < 0.06)`.
A flat border (letterboxing) makes NCC undefined and is handled explicitly.
The pair set passes when `alignment_rate >= --min-alignment` (default 0.90).
`--min-alignment` is a recorded setting rather than an escape hatch; the
value used is written into the report JSON.

On failure, PHASE 5 of the log prints a per-pair diagnostic table and a
ranked list of remedies, including detection of `CENTRE IDENTICAL` (the
mirror's manipulated folder contains copies of the originals) and a large
frame-count mismatch (a frame-rate change, which an integer offset cannot
recover).

CLI: `--root`, `--methods`, `--layout`, `--sample`, `--probes`, `--seed`,
`--min-alignment`, `--min-resolution`, `--border-frac`, `--json`,
`--emit-pairs`, `--fail-hard`.

Output `pairs.json`: `{meta, pairs: [{pair_id, method, fake, real,
frame_offset}]}` with paths relative to `meta.root`; `load_pairs(path,
data_root=...)` re-resolves them and records `n_missing_files`.

---

## 7. `m2_parse.py` (the parser)

The parser uses the mediapipe Tasks API only; there is no geometric fallback.
A geometric parser produces boxes that do not sit on the face, and a silent
fallback would let such label maps propagate downstream unnoticed, so the
module fails instead.

`_patch_tf_doc_controls()` stubs `tensorflow.tools.docs.doc_controls`, whose
absence on some images makes mediapipe's import fail in a way that resembles
a mediapipe bug. `locate_landmarker(explicit)` searches `--mp-model`, then
`/kaggle/input`, then the scratch cache, then downloads the model.

Per clip: the usable range is the set of frames both clips have (with the
offset applied); indices are spread over the middle 80%; the index list is
built from the usable range and asserted against both clips, so sampling a
frame that exists in one clip and not the other is impossible by
construction.

Per frame: landmarks on the real frame give the crop box and the label map;
the same box is applied to the fake frame and to the labels. Boxes running
off the frame are padded by replication, not clamped, because clamping
changes the aspect ratio and therefore the geometry. One JPEG encode at
`--jpeg-q`; labels are PNG.

Quality fields, both gated:

* `parser_iou`: IoU between the face oval parsed from the real frame and
  from the fake one. Low values mean the two frames disagree about where the
  face is (`--min-iou`).
* `manip_coverage`: fraction of changed pixels falling inside a named region.
  Low values mean the manipulation is somewhere the vocabulary cannot
  describe (`--min-coverage`).

`manipulation_mask(real, fake, sigma, bg_mask, min_bg_frac)` is the canonical
definition of "where the pixels changed", imported by `m10_localization` and
by the control calibration so the phrase means the same thing everywhere.
Threshold `max(8, mu + 2 sd)` over background pixels when a background mask
covers at least 5% of the image, otherwise over the whole difference map.

`overlay_figure(index_path, out_png, n, seed)` builds the contact sheet used
to inspect the parse (`--overlay`).

CLI: `--pairs`, `--out`, `--data-root`, `--frames-per-clip`, `--crop-size`
(0 = full frames), `--crop-margin`, `--jpeg-q`, `--min-iou`,
`--min-coverage`, `--min-region-px`, `--methods`, `--max-pairs`, `--seed`,
`--shard`, `--workers`, `--no-resume`, `--resume-from` (comma-separated
`index.json` paths), `--save-every`, `--time-budget-min`, `--mp-model`,
`--min-conf`, `--vocab`, `--overlay`.

### The `index.json` record

```json
{"sample_id": "8f67b717044586e4", "pair_id": "033_097", "method": "Deepfakes",
 "frame": 123, "real_frame": 123, "frame_offset": 0,
 "real": "Deepfakes/033_097/f000123_real.jpg", "fake": "..._fake.jpg", "lab": "..._lab.png",
 "parser_iou": 0.83, "manip_coverage": 0.91, "changed_frac": 0.04,
 "backend": "mediapipe", "vocab": "face8", "crop_box": [x0, y0, x1, y1],
 "crop_size": 384, "jpeg_q": 90, "areas": {"left_eye": 1234, "...": 0},
 "present_regions": ["left_eye", "..."], "split": "test"}
```

`sample_id` is `stable_id(method, pair_id, frame, "v1")`. The index file
also carries `meta` (vocabulary, region list, crop settings, quality
thresholds, backend), and `parse_stats.json` records drops by reason.

---

## 8. `m3_splice.py` (the operator)

Conditions per region *k*:

| condition | construction | reads as |
|---|---|---|
| `real` | authentic region *k* into the fake | "region *k* is not manipulated" (necessity) |
| `identity` | the fake's own region *k* into the fake | seam and extra JPEG pass only (the control) |
| `reverse` | the fake's region *k* into the real | "only region *k* is manipulated" (sufficiency) |
| `floor` | the real's own region *k* into the real | the operator applied to an authentic image (validity check) |
| `inpaint_fake` / `inpaint_real` | the same masks filled by an inpainter | the inpainting-based counterfactual, for comparison |

`region_mask(lab, region, dilate)` and `union_mask(lab, regions, dilate)`
build masks. `blend(src, dst, mask, mode)` returns `(image, blend_used)`,
with modes `poisson | feather | hard`. Poisson pads before `seamlessClone`
so the mask can never touch the border. When a mask is too small or
degenerate, the blend falls back to the nearest applicable mode and records
the fact in `blend_used`, which lands on every raw row. A silent
substitution would corrupt the measurement; a recorded one is an ablation
cell.

`sample_conditions(rec, ks, dilate, mode, jpeg_q, with_floor, inpaint_method,
min_area_px, vocab, extra_masks)` builds every condition in memory and
returns `(real, fake, labels, per_region)`, where each `per_region` entry is
`{region, region_id, area_px, blend_used, images: {condition: ndarray}}`.
Nothing is written to disk, so the output footprint is independent of the
number of conditions. `make_conditions()` and `build()` materialise
condition PNGs for figures only; the self-test asserts that in-memory and
materialised conditions are bit-identical.

`proxy_conditions(fake, lab, ks, proxies, dilate, jpeg_q, ...)` produces the
blur / noise / shuffle perturbations that need only the suspect image.

`inpaint(img, mask, method)` supports `telea`, `ns` and `lama`.

CLI: `--index`, `--out`, `--materialise N` (write condition PNGs for
figures), `--check N` (run the operator invariants on N parsed samples:
leakage outside the mask, the identity splice being a no-op, and at least one
region per sample changing), `--blend`, `--dilate`, `--jpeg-q`, `--inpaint`,
`--splice-floor` / `--no-splice-floor`, `--seed`. The self-test proves the
operator on a synthetic fixture; `--check` proves it on the parsed frames.

---

## 9. `m4_detectors.py` (the zoo)

Uniform interface:

```python
predict(img, labels=None, variant=0) -> float          # p(manipulated), in [0, 1]
cite(img, labels=None, variant=0)    -> {region: p}    # sums to 1 over the vocabulary
.name  .explainable  .needs_calibration  .close()  .info()
# optional, used only when the matching runner flag is set:
explain(img) -> str                          # --free-text
p_ignore(img, region) -> float               # --proxy selfcheck
p_overlay(img) -> float                      # --cite-mode overlay | both
attribute(img, labels=None, variant=0) -> {region: p}   # --attribution
encoder_features(img) -> (tokens, grid_hw)   # --encoder-probe
```

**Prompts.** `VERDICT_PROMPTS` and `CITE_PROMPTS` hold five paraphrases
each. All paraphrases keep identical option letters so the scored token ids
never change; only the wording varies, which is what the prompt-stability
ablation requires. The citation menu (`cite_options()`) always lists all
regions in vocabulary order with fixed letters, so token ids are constant
across samples as well. `IGNORE_PROMPT`, `OVERLAY_PROMPT` and
`EXPLAIN_PROMPT` serve the optional read-outs.

**The four controls.** Synthetic detectors whose faithfulness is known by
construction, so that a metric which misranks them is known to be broken.

Shared evidence: `region_contrast(img, labels)` = (mean artifact energy in a
thin ring just outside region *k* minus the mean inside *k*) / image artifact
sd. The contrast is local rather than global: a within-image z-score across
regions is mean-zero by construction and discards the level that separates
real from fake. Calibration (`calibrate(records, n=--calibrate-n, seed,
vocab)`), from a few paired samples:

1. **Per-region baseline** from authentic frames. `skin` ringed by textured
   hair has a different natural contrast from `hair`; without standardising
   per region, `max_k` measures anatomy rather than forgery.
2. **Sign** (whether the artifact rises or falls when a region becomes
   synthetic, which is dataset dependent) and **offset** (the midpoint
   between the control's own statistic on authentic and on manipulated
   frames, so that p of about 0.5 sits between the two populations).

| control | verdict statistic | citation | constructed as |
|---|---|---|---|
| `adaptive_oracle` | `max_k evidence` | softmax over evidence | faithful in both directions |
| `fixed_oracle[R]` | `evidence[R]` | always R | faithful forward only |
| `confabulator` | `max_k evidence` | proportional to region area squared | discriminative verdict, unfaithful citation |
| `dummy` | image hash near 0.5 | uniform | no signal |

`fixed_oracle` is faithful in the forward direction, but a constant citation
cannot exceed its own prior, so `CR - prior` is about zero. This dissociation
is what separates the forward and reverse tests.

**VLMs.** `_LogitVLM` never generates for the verdict or the citation: one
forward pass, read the logits of the option letters, softmax over exactly
those. The read-out is deterministic, needs no parsing and yields a
probability. `QwenVLDetector` (`qwen25vl:`) and `HFVLMDetector` (`hfvlm:`,
via `AutoModelForImageTextToText`). The loader is version-aware about
`dtype=` versus `torch_dtype=`; `quant` is one of `fp16 | bf16 | 4bit |
auto`; an optional `:lora=<dir>` suffix loads a PEFT adapter and appends
`+lora:<name>` to `.name`. `max_pixels` and `quant` are stored on the
instance and enter the cache key (Section 10).

`CNNGradCAMDetector` (`cnn:<ckpt>` or `cnn:arch=...`): the decision never
sees the masks; only the projection of the Grad-CAM heat map onto the closed
vocabulary uses them.

`build_detector(spec, device, quant, max_pixels)` spec grammar:

```
dummy | confabulator | adaptive_oracle | fixed_oracle:mouth
qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct[:lora=/path]
hfvlm:HuggingFaceTB/SmolVLM-Instruct
cnn:/path/to/cnn_efficientnet_b0.pt | cnn:arch=efficientnet_b0
```

`smoke_vlm(spec, device, quant, max_pixels, n, index)` checks range,
normalisation, determinism (fp16 jitter under 1e-4 is tolerated) and
throughput before any GPU time is committed.

CLI: `--smoke-vlm`, `--device`, `--quant`, `--max-pixels`, `--index`, `--n`,
`--out`.

---

## 10. `m5_runner.py` (the audit)

Four properties make long, interruptible sessions practical:

* **In memory**: conditions are built while scoring and discarded
  immediately.
* **Content cache**: every model call is keyed by
  `stable_id(det, sample_id, region, condition, blend, dilate, jpeg_q)`,
  where `det` is `cache_name(det)`: `<det.name>` for detectors without a
  resolution or quantisation setting, and
  `<det.name>[:px<max_pixels>][:q<quant>]` for VLM detectors, with each
  suffix appended only when the setting differs from the defaults
  (384 x 384 pixels, `fp16`). Keys written with default settings are
  therefore identical to keys without the suffixes, while runs at another
  resolution or quantisation never reuse each other's predictions.
  Per-sample quantities use `orig_key(det, sid, what[, variant])` with
  `what` in `orig_fake | orig_real | cite_orig | explain | cite_attr |
  union_top2`; prompt variants append `("v", variant)`. Per-region
  quantities use `cond_key(det, sid, region, cond, blend, dilate, jpeg_q)`
  with `cond` in `real | identity | reverse | floor | inpaint_fake |
  inpaint_real | cite_reverse | proxy_<p> | proxyfloor_<p> | overlay |
  enc_shift`. The key depends on what was asked, never on when or in which
  shard, so caches from different sessions merge and de-duplicate
  themselves. The `detector` field of each row is `det.name` without the
  suffixes.
* **Skip without re-splicing**: `_needed_keys()` enumerates every key a
  sample needs; if all are cached, the sample is rebuilt from the cache
  without decoding or splicing anything.
* **Time budget**: the loop checks the wall clock, flushes rows and cache,
  and exits with status 2 so that a re-run continues.

CLI: `--index` (comma-separated to merge shards), `--detector a,b,c`,
`--out`, `--tag`, `--blend`, `--dilate`, `--jpeg-q` (0 = no re-encode),
`--regions`, `--splice-floor`, `--inpaint`, `--prompt-variants`, `--no-cite`,
`--proxy blur,noise,shuffle,selfcheck`, `--no-proxy-floor`, `--free-text`,
`--pairs-topk`, `--cite-mode {text,overlay,both}`, `--encoder-probe`,
`--attribution`, `--limit-samples`, `--seed`, `--split {all,dev,test}`,
`--methods`, `--shard i/n`, `--save-every`, `--time-budget-min`,
`--resume-from` (comma-separated directories whose `cache_*.json` seed the
cache), `--device`, `--quant {auto,4bit,fp16,bf16}`, `--max-pixels`,
`--calibrate-n`, `--vocab {face8,grid9}`.

Subsetting (`--methods`, `--split`, `--limit-samples`) happens before
sharding, so every shard audits the same subset. Control detectors with
`needs_calibration` are calibrated on `--calibrate-n` records of the full
index before the shard is processed.

Outputs per detector, with stem `<det>[_<tag>][_s<i>of<n>]`:
`raw_<stem>.json` (`{meta, rows}`), `cache_<stem>.json` (`{meta, cache}`),
`runstats_<stem>.json` and `provenance_<stem>.json`.

### Raw row schema

```
sample_id, pair_id, method, frame, split, region, region_id, detector, tag,
area_px, blend, dilate, jpeg_q, blend_used,
p_orig_fake, p_orig_real, p_real, p_identity, p_reverse,
cite_orig{region: p}, cite_reverse{region: p},
[p_floor], [inpaint, p_inpaint_fake, p_inpaint_real],
[p_orig_fake_variants[V], cite_orig_variants[V]],
[p_blur, p_noise, p_shuffle, p_selfcheck] and their p_<proxy>_real floor twins,
[p_mark, cite_overlay{region: p}], [cite_attr{region: p}],
[enc_shift_in, enc_shift_out],
[explanation_text], [cite_union_regions, p_union_real, p_union_identity]
```

Bracketed fields are present only when the matching flag was set. New
per-region conditions follow the `p_<name>` pattern; new per-sample fields
go on every row of that sample.

---

## 11. `m6_metrics.py`

Rows are normalised by `samples_from_rows()` into one record per sample
(`{sample_id, pair_id, method, split, detector, tag, p_orig_fake,
p_orig_real, cite_orig, cite_attr, cite_overlay, cite_variants, p_variants,
explanation_text, union, regions: {k: {p_real, p_identity, p_reverse,
delta_raw, delta_seam, delta, area_px, blend_used, cite_reverse, ...}}}`).
Every metric reads that structure, so coarsening, per-method slices and
proxies all operate on one representation rather than on ad-hoc row filters.

Two errors are prevented structurally:

* **Double mapping.** `_coarsen()` is applied exactly once, to the normalised
  structure, never to already-coarsened rows. Mapping a citation distribution
  twice destroys most of its mass.
* **Representative inheritance.** Citation distributions are summed over the
  members of a coarse region (a probability over a partition); every measured
  effect is averaged over the members that are present.

`analyse(rows, seed, coarse, taus, tau, vocab, n_boot, localization, meta)`
returns, per detector x tag:

```
detector, tag, blend, dilate, jpeg_q, granularity, split, methods,
n_rows, n_samples, n_clips, regions, tau,
AUC, p_orig_fake_mean, p_orig_real_mean,
FS, FS_lo, FS_hi, FS_n, FS_p, rank_cited, cited_distribution,
delta_cited, delta_mean_all, cohen_d_cited_vs_all,
seam_abs, seam_abs_sd, FS_in_seam_sd,
floor_p_mean, floor_frac_flagged,
CR, CR_lo, CR_hi, CR_prior, CR_minus_prior, n_reverse_flipped,
injected_distribution, CR_by_tau{tau: {...}},
faithful_forward, faithful_reverse, faithful,
per_region{k: {n, delta, delta_lo, delta_hi, delta_raw, delta_seam, floor_p,
               p_boot, cited_frac, significant_holm}},
confusion{cited: {largest_effect: n}}, citation_entropy{mean, sd, corr_with_FS, n}
```

plus the optional blocks `inpaint`, `prompt_stability`, `proxies`,
`citation_agreement`, `blind_spot`, `union` and `localization`, each present
only when the rows carry the corresponding fields. `analyse_all` adds
`source_files` to every result.

`detector_level_holm(results)` applies Holm-Bonferroni across the detector
zoo. Each detector's verdict is a separate hypothesis, so reporting every
detector that clears alpha = 0.05 inflates the family-wise error rate. The
family contains one result per detector (the one tagged `main` when it
exists). It adds `FS_significant_holm_detectors`, `holm_family_size` and
`faithful_holm`; the uncorrected `faithful` is kept so the difference is
visible.

`additivity(groups)` tests the assumption behind the seam correction: that
`delta = delta_raw - delta_seam` is unbiased, which requires seam and content
to act additively. Dilation radius varies the seam, so
`delta_raw(r_hi) - delta_raw(r_lo)` should equal
`delta_seam(r_hi) - delta_seam(r_lo)`; the paired within-cell residual
estimates the bias. It requires runs at two or more radii, identified by
tags of the form `dilate_<r>`, and returns per detector `residual`,
`residual_lo`, `residual_hi`, `additive` and `per_radius[...]["seam_share"]`.
Caveat: at small radii the mask can be smaller than the manipulation, so
dilation adds content as well as seam and a large residual says nothing
about additivity. Radii that both cover the manipulation are the ones to
compare, together with `seam_share`.

`group_raw_files(dirs)` groups every `raw_*.json` by `(detector, tag)`;
shards and separate sessions merge, and `load_rows` de-duplicates by
`(sample_id, region, detector, tag)`.

`analyse_all(...)` writes `metrics.json` (or `metrics_coarse.json` with
`--coarse`) as `{results, additivity, provenance, settings}` and, with
`--by method`, `metrics_by_method.json`.

CLI: `--raw` (comma-separated run directories), `--out`, `--seed`,
`--coarse`, `--by method`, `--split`, `--tau`, `--taus`, `--vocab`,
`--n-boot`, `--localization`.

---

## 12. Analysis modules

**`m10_localization`**: `D = |fake - real|` gives the manipulation exactly.
Per region: `f[k]` = `|M ∩ k| / |k|`, `m[k]` = mean intensity, `iou[k]`;
`gt_region = argmax f`, `gt_dist` = `f` normalised. The fraction is used
rather than the intensity: fraction asks how much of the region was touched,
which is what a citation claims, and prevents `skin` from winning by size.
CLI: `--index`, `--out`, `--split`, `--limit`, `--seed`, `--sigma`,
`--time-budget-min`. Output `loc/localization.json` (`{records: [...]}`).

**`m11_proxy`**: ranks blur / noise / shuffle / selfcheck by (a) Spearman
correlation with true FS, (b) Kendall tau on the region ranking, (c) decision
agreement across the detector zoo, and disqualifies any proxy whose floor
(mean p rise on an authentic frame) has a CI excluding zero: a proxy that
correlates well but manufactures evidence is the same defect that
disqualifies an inpainting counterfactual. CLI: `--raw`, `--out`, `--vocab`,
`--seed`, `--n-boot`, `--split`.

**`m12_text_regions`**: a fixed lexicon; longest phrases matched first; word
boundaries only (so "beard" does not fire `ears`); laterality within a few
words routes an eye term, an unqualified eye term splits 50/50 with ties
broken by a hash of the sample id so the agreement statistic is unbiased.
Reports `frac_no_region` rather than dropping unlocatable sentences. CLI:
`--raw`, `--out`, `--vocab`, `--try-text` (map one sentence and exit).

**`m13_cset`**: builds (image, prompt, answer) triples from a DEV audit and
refuses rows whose split is not `dev`. Two variants: `causal` (target =
argmax delta, dropping samples with max |delta| below `--min-delta`) and
`mask` (target = `gt_region` from `--localization`). Verdict targets are the
true labels in both, so detection trains identically. QLoRA on the language
model only, vision tower frozen; the loss is taken on the answer token alone,
with everything before it masked to -100. `--dry-run` builds and reports the
dataset without a GPU. CLI: `--raw`, `--index`, `--out`, `--variant`,
`--base-model`, `--localization`, `--min-delta`, `--lora-r`, `--lora-alpha`,
`--lr`, `--epochs`, `--batch-size`, `--grad-accum`, `--max-pixels`, `--seed`,
`--device`, `--time-budget-min`, `--vocab`, `--detector-filter`, `--dry-run`.

**`m14_attribution`**: stated / attributed / overlay / causal agreement, the
joint counts per method, and the blind-spot decomposition
(`encoder_blind_frac`, `language_confabulation_frac`, `faithful_frac`).
CLI: `--raw`, `--out`, `--vocab`, `--seed`. Output `attribution.json`.

**`m7_report`**: a self-contained HTML file with no external CSS, no
JavaScript and no image files; every figure is an inline SVG. Sections: the
primary table, instrument validation (checks the control ordering and prints
PASSED or FAILED), figures, inpainting comparison, prompt stability, stated
vs attributed vs causal citation, blind-spot decomposition, compositional
interventions, deployable proxies, citation entropy, reverse-threshold
sweep, coarse granularity, per manipulation method, confusion matrix,
multiplicity across the detector zoo (uncorrected vs Holm-corrected verdict
per detector), additivity of seam and content effects, and provenance.
Writes `results_table.csv` beside the report and, with `--figures-dir`, each
figure as a standalone `.svg`. CLI: `--metrics`, `--coarse`, `--by-method`,
`--raw` (run directories, for the inpainting scatter), `--out`, `--csv`,
`--title`, `--figures-dir`.

**`m8_human_study`**: side-by-side items in randomised order with the cited
region named, a blank annotation CSV for N annotators, `INSTRUCTIONS.txt`,
and `key.json` containing the answers and the detector's delta. The folder
is distributed without `key.json`. `--score <csv>` joins returned
annotations into the plausible-versus-causal cross-tabulation. CLI:
`--index`, `--raw`, `--out`, `--n-items`, `--detector`, `--tag`, `--split`,
`--annotators`, `--blend`, `--dilate`, `--jpeg-q`, `--seed`, `--score`.

**`m9_train_cnn`**: DEV only; train/val split by clip, not by sample;
classes balanced by construction (each sample gives one real and one fake
crop); checkpoint `{arch, input_size, state_dict, provenance, val_auc,
epochs_done}` written as `cnn_<arch>.pt` with `train_log.json`. External
weights are used by saving them in the same shape. CLI: `--index`, `--out`,
`--arch`, `--epochs`, `--batch-size`, `--lr`, `--weight-decay`,
`--input-size`, `--val-frac`, `--workers`, `--seed`, `--device`, `--no-amp`,
`--no-pretrained`, `--train-budget-min`, `--resume`, `--max-samples`.

---

## 13. Notebook conventions

The twelve notebooks in `notebooks/` are generated by
`scripts/make_notebooks.py` and share one structure:

* A `SESSION` name and a configuration cell of upper-case variables at the
  top; every later cell reads only those variables.
* A boot cell that locates the code by searching for
  `ccaudit/kaggle_utils.py` under `/kaggle/input`, inserts the containing
  directory into `sys.path` and `PYTHONPATH`, sets `HF_HOME` to the scratch
  directory, and calls `kaggle_utils.session_header()`.
* A self-test cell that runs `scripts/selftest.py` before any long step.
* Inputs discovered by file name (`find_parsed_index`, `find_run_dirs`,
  `find_checkpoints`), never by mount path.
* Every long step launched as `python -m ccaudit.<module>` through
  `kaggle_utils.sh` or `run_parallel`, with `--time-budget-min` and a log
  file under `audit/logs/`.
* Outputs written under `audit/` in `/kaggle/working`; large disposable
  files (model weights, materialised frames) under the scratch directory.
* A closing `NEXT_STEP` banner naming the output dataset to create.

`scripts/check_platform_constraints.py` reads the generated notebooks and
the package source and checks that the budgets of each session sum to a
value below the wall clock, that the projected output fits the cap, that
scratch is used for large files, that CPU sessions request no accelerator,
and that nothing writes into `/kaggle/input`.

---

## 14. Conventions for new modules

* CLI via `python -m ccaudit.<module> --help`.
* Read and write JSON through `common.load_index` / `save_index` /
  `save_json`, with relative paths.
* Support `--split`, `--limit-samples --seed`, `--shard`,
  `--time-budget-min`, `--resume-from` and `--vocab` wherever they apply.
* Cache keys via `stable_id`, keyed by what was asked.
* Never write large binaries to the output directory; use
  `common.scratch_dir()`.
* Add one self-test check per module, and make it fail loudly rather than
  degrade.
* Record, never silently substitute: any fallback goes into an output field.
