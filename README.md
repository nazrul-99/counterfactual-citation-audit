# counterfactual-citation-audit

Interventional faithfulness auditing for explainable deepfake detectors.

An explainable forgery detector returns a verdict together with a *citation*: the facial region it names as the site of the manipulation, whether as a saliency map, a predicted mask, or a sentence. This package measures whether the cited region is the region that caused the verdict. It does so with counterfactual images that contain no synthetic pixels, built from the frame-aligned authentic twin that every manipulated frame in a face-swap benchmark already has.

The full research proposal is in [`docs/PROPOSAL.md`](docs/PROPOSAL.md). This file explains what the code does and how it is organised.

## The measurement

For a fake frame $x^f$, its authentic twin $x^r$, and a facial region $k$, three images are constructed by splicing one frame's region into the other:

| Condition | Construction | What it isolates |
|---|---|---|
| real | region $k$ of $x^f$ replaced by the same region of $x^r$ | region $k$'s manipulation removed |
| identity | region $k$ of $x^f$ replaced by itself | the splice seam alone |
| reverse | region $k$ of $x^r$ replaced by the same region of $x^f$ | region $k$'s manipulation injected into an authentic frame |

The detector's manipulation probability $p(\cdot)$ is read on each. The artifact-free causal effect of region $k$ is the difference in differences

$$\Delta(k) = p(\text{identity}_k) - p(\text{real}_k),$$

which cancels the baseline $p(x^f)$ and, with it, the compression history shared by both conditions. Two metrics follow:

- **Faithfulness score.** $\mathrm{FS} = \Delta(\text{cited}) - \operatorname{mean}_{k \neq \text{cited}} \Delta(k)$. A detector whose citation is unrelated to its computation has $\mathrm{FS} \approx 0$ whatever its detection accuracy.
- **Reverse citation recall.** Among (sample, region) pairs where injecting region $k$'s manipulation flips the verdict to fake, the fraction in which the detector cites $k$, reported against the best-constant-guess prior.

A detector is called faithful when the lower confidence bound of FS exceeds zero and the lower bound of reverse recall exceeds the prior. Confidence intervals come from a cluster bootstrap over clips; Holm–Bonferroni is applied across regions and across detectors.

Before any real detector is audited, the same pipeline is run on four synthetic detectors whose faithfulness is known by construction (an adaptive oracle, a fixed-region oracle, a confabulator that detects well but cites at random, and a dummy). The audit proceeds only if the metrics order them as constructed.

## What the package does beyond the primary audit

| Capability | Module | Purpose |
|---|---|---|
| Ground-truth localization | `m10_localization` | Where the pixels changed, from $\lvert x^f - x^r \rvert$; cross-tabulates citation *correctness* against citation *faithfulness* |
| Inpainting comparison | `m3_splice`, `m6_metrics` | The same masks filled by an inpainter, with the inpainter's effect on authentic frames reported as a validity check |
| Deployable proxies | `m11_proxy` | Single-image perturbations (blur, noise, shuffle, inpaint, self-check prompt) validated against the paired reference, each with its floor on authentic frames |
| Free-text agreement | `m12_text_regions` | Lexicon mapping from a model's prose explanation to regions, compared with its constrained citation |
| Gradient attribution | `m14_attribution`, `m4_detectors.attribute` | Gradient-times-input on visual tokens at the connector, pooled per region; three-way agreement between stated, attributed, and causal citations |
| Encoder sensitivity | `m5_runner --encoder-probe`, `m6_metrics` | Cosine shift of the visual tokens covering region $k$ under the counterfactual; classifies unfaithful citations as encoder-blind or language-side |
| Explanation tuning | `m13_cset` | Low-rank adapter tuning of a VLM's citation under causal supervision ($\arg\max_k \Delta(k)$) or mask supervision, trained on the development split only |
| Operator validity | `m6_metrics.additivity`, `union_additivity` | Additivity of seam and content effects across dilation radii; super-additivity of two-region unions |
| Human study | `m8_human_study` | Annotator materials and scoring for a preference-versus-faithfulness cross-tabulation |

## Repository layout

```
ccaudit/                    the pipeline, one module per stage
  regions.py                region vocabularies and landmark-to-region geometry
  common.py                 atomic I/O, relative-path index, split, bootstrap, budgets
  kaggle_utils.py           input discovery and subprocess helpers for notebook sessions
  m1_verify.py              pairing gate: is the mirror frame-aligned?          -> pairs.json
  m2_parse.py               face parsing and cropping                          -> parsed/index.json
  m3_splice.py              the splice operator and its conditions (in memory)
  m4_detectors.py           detector zoo: synthetic controls, VLMs, CNN
  m5_runner.py              resumable, sharded, time-budgeted audit             -> raw_*.json
  m6_metrics.py             all metrics                                        -> metrics.json
  m7_report.py              self-contained HTML report with inline SVG figures -> report.html
  m8_human_study.py  m9_train_cnn.py  m10_localization.py  m11_proxy.py
  m12_text_regions.py  m13_cset.py  m14_attribution.py
notebooks/                  eleven session notebooks for a free-tier GPU platform (generated)
scripts/
  selftest.py               metric and operator checks on a synthetic fixture; no dataset, no GPU
  check_platform_constraints.py   static audit of session budgets, memory, and disk limits
  make_notebooks.py         source of truth for notebooks/
  run_pipeline.sh           local CPU pipeline
tests/                      fixture generator and a landmarker test double
docs/
  PROPOSAL.md               research proposal
  ARCHITECTURE.md           module interfaces, schemas, cache-key layout
  RUNNING.md                operating guide for the session notebooks and the local path
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/selftest.py                    # exercises the operator and every metric on a synthetic fixture
python scripts/check_platform_constraints.py  # static check of the session budgets
```

The self-test needs no dataset, no GPU, and no network. Each check corresponds to a way of producing plausible-looking numbers that are wrong: the splice touching pixels outside its mask, the coarse vocabulary double-counting a citation, the bootstrap resampling frames rather than clips, an identity leaking across the development/test split, a cache key that depends on when a call was made rather than on what was asked.

To run on data, follow [`docs/RUNNING.md`](docs/RUNNING.md). The first session is a pairing gate that checks whether a given benchmark mirror is frame-aligned and exits non-zero if it is not; nothing downstream is meaningful without it.

## Requirements

CPU stages need `numpy`, `opencv-python-headless`, `mediapipe`, and `tqdm`. GPU stages additionally need `torch`, `timm`, `transformers`, `accelerate`, and `qwen-vl-utils`; adapter tuning and 4-bit inference need `peft` and `bitsandbytes`; the learned inpainter needs `simple-lama-inpainting`. See `requirements.txt`.

## Data

The audit requires a face-swap benchmark in which each manipulated clip is aligned frame-for-frame with its target clip (FaceForensics++ and Celeb-DF-v2 satisfy this). Benchmark data are licensed separately and are not distributed here. The package writes an index, per-call caches, and metric files that reproduce every number from a licensed copy; it does not write counterfactual images to disk.

## Licence

Code is released under the MIT licence (see `LICENSE`). Benchmark data and pretrained model weights are subject to their own terms.
