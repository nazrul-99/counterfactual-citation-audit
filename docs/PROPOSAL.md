# Research Proposal

## Does the Cited Region Cause the Verdict? An Artifact-Free Interventional Audit of Region Explanations in Deepfake Detection

*Ongoing project. This document describes the research question, the proposed instrument, the experimental programme, and the analysis protocol. It contains no experimental results.*

---

## Abstract

Explainable deepfake detectors increasingly justify a verdict by naming a facial region: a convolutional detector highlights the mouth with a saliency map, and a vision-language model (VLM) writes that "the boundary of the jaw shows blending artifacts." Current evaluation asks whether such an explanation is *plausible* (does it read well to a human or to a language-model judge) or *accurate about the image* (does the named region overlap the forgery mask). Neither test asks whether the named region is the region that *drove* the verdict. A detector can be right about where the manipulation is and still base its decision on something else entirely.

This project proposes an interventional audit that measures exactly that. Face-swap benchmarks are constructed by editing a source video, so every manipulated frame has a frame-aligned authentic twin. The counterfactual "region *k* was not manipulated" can therefore be realised by splicing the authentic pixels of region *k* back into the fake frame. No generative model fills the hole, so the counterfactual introduces no synthetic evidence of its own; the residual effect of the splice seam is measured separately by an identity splice that replaces the region with itself. From the change in the detector's output under these interventions we define two quantities: a *faithfulness score* that compares the causal effect of the cited region with that of the uncited regions, and a *reverse citation recall* that injects manipulated pixels into an authentic frame and asks whether the detector cites the injected region. The instrument is validated on synthetic detectors with known faithfulness before it is applied to real ones.

The programme has three layers. The first establishes the instrument and its validity conditions. The second applies it to a detector zoo (a trained CNN with gradient-based saliency, and several open-weight VLMs) and cross-tabulates faithfulness against localization accuracy, against inpainting-based counterfactuals, against gradient attribution on visual tokens, and against the sensitivity of the vision encoder itself. The third asks whether faithfulness can be *trained*: an explanation-tuning procedure supervises the cited region with the region of largest measured causal effect rather than with the forgery mask, and is compared with mask supervision at matched detection accuracy. The full programme is sized for a free-tier cloud GPU allocation and a pre-registered statistical protocol.

---

## 1. Motivation

### 1.1 The practical question

A deepfake detector deployed in a moderation pipeline, a newsroom, or a courtroom is rarely trusted on its verdict alone. It is trusted, or not, on its explanation. The explanations in current use take one of two forms: a spatial map over the image (saliency, attention, or a predicted forgery mask), or, for detectors built on VLMs, a natural-language sentence that names a facial region and describes the artifact. In both forms the explanation makes an implicit causal claim: *this region is why I say the image is fake*.

The claim is testable, but it is not tested. The evaluation literature for explainable forgery detection scores explanations for text similarity to a reference, for agreement with a language-model judge, or for overlap between a predicted region and the ground-truth forgery mask. Each of these measures whether the explanation is *plausible* or *correct about the image*. None measures whether the explanation is *faithful* to the detector's own decision process. A detector that always cites the mouth on a benchmark where most manipulations touch the mouth will score well on every existing metric while providing no information about what it computed.

### 1.2 Why the obvious tool is unsafe here

The standard way to test a causal claim about an input region is to remove that region and observe the change in output. For natural images this is done by masking, blurring, or inpainting the region. In forgery detection every one of these removals is itself a forgery. Masking introduces a sharp boundary; blurring introduces a frequency discontinuity; inpainting introduces the output of a generative model, which is precisely the class of evidence a deepfake detector is trained to find. A detector that responds to the removal operation cannot be distinguished from a detector that responds to the removed content. This is the imputation-artifact problem identified in the attribution-evaluation literature, and it is more severe for forgery detectors than for ordinary classifiers because the artifact and the signal are the same kind of thing.

### 1.3 The observation that makes the problem tractable

Face-swap benchmarks are built by manipulating a target video, and the target video is shipped alongside the manipulated one. Every fake frame therefore has an authentic frame that is aligned to it pixel for pixel outside the manipulated area. This gives an oracle answer to the question "what would this region look like if it had not been manipulated": it would look like the same region of the authentic twin. Splicing that region back in produces a counterfactual with no synthetic pixels. The only foreign content is the splice seam, and the seam can be measured on its own by splicing a region onto an identical copy of itself.

The same paired structure yields two further quantities for free. The difference image between the fake frame and its twin reveals where the pixels actually changed, so citation *correctness* can be measured without a separate mask annotation. And the manipulation can be moved in the other direction, from the fake frame into the authentic one, which permits a *sufficiency* test that asks whether injecting the evidence causes the detector to cite it.

---

## 2. Position With Respect to Prior Work

The proposal sits at the intersection of four lines of work and departs from each in a specific way.

**Explainable forgery detection.** A growing family of detectors produces a verdict together with a textual or spatial explanation, and several benchmarks now evaluate those explanations. The evaluation targets are plausibility (text-overlap and judge-model scores) and localization (mask overlap). This work adopts the same detectors and the same region-level framing but adds the missing test: whether the explanation is causally responsible for the verdict. The framing here is deliberately compatible with mask-producing detectors, whose predicted masks are mapped to a region distribution by overlap so that they enter the same analysis.

**Faithfulness of model self-explanations.** Work on language models has shown that stated reasoning can be unfaithful to the computation that produced an answer, and has developed counterfactual and perturbation tests to detect this. Those tests operate on text inputs. This proposal transfers the counterfactual logic to the vision side, where the intervention must be physically valid and where a closed region vocabulary makes the counterfactual well-defined.

**Attribution evaluation by removal.** Remove-and-retrain and deletion-curve protocols evaluate pixel attributions by removing the attributed pixels and measuring the change in output. The most careful of these protocols address imputation artifacts by using a smooth imputation and by retraining. The present proposal treats the paired frame as an *oracle imputation*, which sidesteps the artifact rather than mitigating it, and adds an identity control that measures the residual seam effect directly. That protocol is the closest methodological ancestor of this work and the natural anchor for its related-work discussion.

**Analysis of what deepfake detectors compute.** Prior analyses have asked which source or target content a CNN matches, and whether identity information leaks into the decision. Those studies analyse detectors; this work analyses detectors' *explanations*, and extends the analysis to VLMs, whose explanations are produced by a language model that may or may not be reading the visual evidence.

A separate strand, faithfulness-oriented training, supervises attention or explanations toward human-provided masks. The tuning procedure proposed in Section 4.6 differs in the supervision signal: it uses the measured causal effect rather than the mask, and the comparison between the two is one of the experiments.

---

## 3. Research Questions and Hypotheses

The programme is organised around five questions. Each is stated with the hypothesis it tests and the observation that would refute it.

**RQ1 — Can region-level faithfulness be measured without artifacts?**
*Hypothesis.* The paired-frame splice, with the identity control subtracted, produces a causal-effect estimate whose seam component is small relative to its content component, and whose sign and ordering are recovered on synthetic detectors with known faithfulness.
*Refutation.* The identity splice moves the detector as much as the real splice, or the synthetic detectors are not ordered as constructed.

**RQ2 — Are inpainting-based counterfactuals valid for forgery detection?**
*Hypothesis.* Filling the same regions with an inpainter raises manipulation confidence on *authentic* frames, and the inpainting-based effect estimate disagrees in sign with the artifact-free estimate on a non-trivial fraction of samples.
*Refutation.* Inpainting-based and splice-based estimates agree, and the inpainter's effect on authentic frames is indistinguishable from zero.

**RQ3 — Is citation correctness the same thing as citation faithfulness?**
*Hypothesis.* A substantial fraction of correct citations (the cited region is the manipulated region) are not causal (removing the manipulation from the cited region does not change the verdict more than removing it elsewhere). The cross-tabulation of the two properties has a populated "correct but not causal" cell.
*Refutation.* Correctness and faithfulness coincide at the sample level.

**RQ4 — Where does unfaithfulness arise in a VLM: in the encoder or in the language model?**
*Hypothesis.* Unfaithful citations decompose into two mechanistically distinct cases: those where the visual tokens covering the cited region do not move under the counterfactual (the encoder is blind to the evidence), and those where the tokens move but the verdict does not (the language model does not use them). Stated citations, gradient-attributed citations, and causal citations will disagree with one another in a detector-dependent way.
*Refutation.* Token shift and causal effect are tightly coupled, and the three citation types coincide.

**RQ5 — Can faithfulness be trained without sacrificing detection accuracy?**
*Hypothesis.* Supervising a VLM's citation with the region of largest measured causal effect improves faithfulness at matched detection accuracy, whereas supervising with the forgery mask improves correctness without improving faithfulness.
*Refutation.* Both supervision signals move faithfulness equally, or the causal supervision lowers detection accuracy by more than a pre-specified margin.

A sixth, subsidiary question asks whether a *deployable* proxy for faithfulness exists, one that needs no paired original at test time. It is subsidiary because it can only be answered once the artifact-free reference is available to validate the proxy against.

---

## 4. Method

### 4.1 Setting and notation

A sample consists of a fake frame $x^f$, its frame-aligned authentic twin $x^r$, and a region label map that partitions the face into $K$ disjoint regions. The default vocabulary has eight regions: left eye, right eye, nose, mouth, skin, jaw boundary, hair, and ears. Laterality is by image side rather than anatomy, because a model describing a picture refers to the eye it sees on the left. A coarser four-region vocabulary (eyes, nose, mouth, rest) is obtained by merging, and a nine-cell grid vocabulary is available for domains without facial structure.

A detector exposes a manipulation probability $p(x) \in [0, 1]$ and a citation distribution $c(x) \in \Delta^{K}$ over regions. For a CNN the citation is the per-region mass of a gradient-based saliency map. For a VLM the citation is the softmax over the logits of the region letters in a constrained multiple-choice prompt, and the probability is the softmax over the letters of a real/fake prompt. Free-text explanations are also collected and mapped to regions by a fixed lexicon, so that the constrained citation can be checked against what the model says in prose.

### 4.2 The splice operator and its conditions

For region $k$ with mask $M_k$ (optionally dilated by radius $r$), define the splice of source $s$ into destination $d$ as

$$\mathrm{splice}(d, s, M_k) = \mathrm{blend}(d,\ s,\ M_k),$$

where blend is Poisson blending by default, with feathered and hard alternatives kept for ablation. Both outputs pass through the same JPEG round trip so that compression history is matched. Three conditions are built for every (sample, region):

| Condition | Construction | Meaning |
|---|---|---|
| real | $\tilde{x}^{f}_{k} = \mathrm{splice}(x^f, x^r, M_k)$ | region $k$ of the fake is replaced by its authentic twin |
| identity | $\tilde{x}^{f}_{k,\mathrm{id}} = \mathrm{splice}(x^f, x^f, M_k)$ | region $k$ is replaced by itself; only the seam remains |
| reverse | $\tilde{x}^{r}_{k} = \mathrm{splice}(x^r, x^f, M_k)$ | region $k$ of the authentic frame receives the manipulated pixels |

A fourth condition, the *floor*, splices the authentic frame onto itself and measures how much the operator alone moves the detector on an image with no manipulation anywhere.

### 4.3 Effect estimates and metrics

The raw effect of removing region $k$'s manipulation is $\Delta_{\mathrm{raw}}(k) = p(x^f) - p(\tilde{x}^{f}_{k})$. The seam effect is $\Delta_{\mathrm{seam}}(k) = p(x^f) - p(\tilde{x}^{f}_{k,\mathrm{id}})$. Their difference cancels $p(x^f)$ algebraically and therefore cancels the baseline's compression history as well:

$$\Delta(k) = \Delta_{\mathrm{raw}}(k) - \Delta_{\mathrm{seam}}(k) = p(\tilde{x}^{f}_{k,\mathrm{id}}) - p(\tilde{x}^{f}_{k}).$$

This is the artifact-free causal effect of region $k$'s manipulation on the verdict. The difference-in-differences form assumes that seam and content effects are additive; the assumption is tested rather than taken for granted (Section 5.2).

Two metrics follow. The **faithfulness score** of a sample is the causal effect of the cited region relative to the others,

$$\mathrm{FS} = \Delta(k^\star) - \frac{1}{K-1}\sum_{k \neq k^\star} \Delta(k), \qquad k^\star = \arg\max_k c(x^f)_k,$$

so that a detector whose citation is unrelated to its computation has $\mathrm{FS} \approx 0$ regardless of its detection accuracy. The **reverse citation recall** is measured on the reverse condition: over the (sample, region) pairs in which injecting region $k$'s manipulation into the authentic frame flips the verdict to fake ($p(\tilde{x}^{r}_{k}) \ge \tau$), it is the fraction in which the detector cites $k$. Because a detector that always cites the most common region achieves a non-zero recall by chance, the recall is reported against that best-constant-guess prior. A detector is called faithful when the lower confidence bound of FS exceeds zero *and* the lower bound of the reverse recall exceeds the prior, at $\tau = 0.5$, with $\tau \in \{0.3, 0.7\}$ reported as a sensitivity check.

### 4.4 Validation on detectors with known faithfulness

Before any real detector is audited, the instrument is run on four synthetic detectors constructed so that the correct answer is known. An *adaptive oracle* reads the manipulation mask and cites the most manipulated region; it should be faithful in both directions. A *fixed oracle* reads the mask but always cites one fixed region; it should pass the forward test only when that region happens to be the manipulated one, and should fail the reverse test, which is the dissociation the reverse metric exists to detect. A *confabulator* detects accurately but cites a region drawn independently of its computation; it must show high detection accuracy with $\mathrm{FS} \approx 0$, which is the single most important check because it demonstrates that FS is not a re-measurement of accuracy. A *dummy* neither detects nor cites. The synthetic detectors read region evidence through a local ring contrast with per-region baselines calibrated on authentic frames; a within-image z-score was found in preliminary implementation to be mean-zero across regions by construction and was replaced.

### 4.5 Citation correctness from the paired difference

The difference image $D = |x^f - x^r|$, smoothed and thresholded against the background statistics, gives a binary manipulation mask without annotation. Per region, the manipulation *fraction* $f_k$ (share of the region's pixels that changed) determines the ground-truth region $\arg\max_k f_k$; the fraction rather than the mean intensity is used so that large regions do not dominate. Citation correctness is then the indicator that the cited region is the ground-truth region, with a soft variant that weights the citation distribution by $f_k$. The cross-tabulation of correctness against per-sample faithfulness is the analysis that answers RQ3.

### 4.6 Extensions that turn the audit into an analysis of mechanism

**Attributed citation.** For a VLM, a single forward and backward pass of the verdict prompt yields the gradient of the fake-letter logit with respect to the visual-token hidden states at the connector output. Scoring each token by gradient-times-input, mapping tokens to pixels through the patch grid, and summing over each region's mask gives the model's *attributional* citation. The three-way comparison of what the model *says*, what its *gradients* point at, and what *causes* its verdict (the arg-max of $\Delta$) is the analysis for the first half of RQ4.

**Overlay-elicited citation.** As a second elicitation channel, the outline of region $k$ is drawn on the frame and the model is asked whether the manipulation lies inside the outlined region. The resulting per-region distribution is both an alternative citation and a candidate deployable proxy, since it needs no original.

**Encoder sensitivity.** During the forward passes already made for $p(x^f)$ and $p(\tilde{x}^{f}_{k})$, the connector output is captured and the cosine shift of the tokens covering region $k$ is computed, with the shift of tokens outside $k$ as a control. Combining the shift with $\Delta(k)$ classifies each cited region as encoder-blind (no shift, no effect), language-side (shift without effect), or faithful (shift with effect). This is the analysis for the second half of RQ4. Implementation note: capturing features for the counterfactual requires additional vision-tower passes per region, so the probe is not free; it is cached so the cost is paid once.

**Deployable proxies.** Five single-image perturbations of region $k$ are evaluated as substitutes for the paired counterfactual: Gaussian blur, mean-plus-noise replacement, patch shuffle, an inpainter, and, for VLMs, a self-check prompt that asks the model to ignore the region. Each proxy is scored by its sample-level rank correlation with FS, its per-sample region-ranking agreement with $\Delta$, and its decision agreement across the detector zoo. Each proxy's floor, its effect on authentic frames, is reported alongside; a proxy whose floor confidence interval excludes zero is disqualified regardless of its correlation, because it manufactures the evidence it claims to measure.

**Causally-supervised explanation tuning.** A VLM is fine-tuned with low-rank adapters on the language model only, vision tower frozen, with loss on the answer token alone, under two supervision variants that differ only in the citation target: the *causal* variant uses $\arg\max_k \Delta(k)$ from the base model's own audit on the development split, discarding samples with no measurable causal signal; the *mask* variant uses the ground-truth region from the paired difference. Verdict targets are identical in both. The tuned models are audited with the identical protocol on the held-out test split, and detection accuracy is required not to fall by more than a pre-specified margin.

---

## 5. Experimental Programme

### 5.1 Data and detectors

The primary benchmark is a face-swap dataset with five manipulation methods and a moderate compression level, in which every manipulated clip is aligned to its target clip. A second face-swap dataset with a different generation pipeline provides a cross-dataset check for the CNN and for the tuned VLMs. A higher-compression variant of the primary benchmark, if available, provides a robustness check. Frames are parsed with a face landmarker into the eight-region label map and cropped to a fixed face box; a crop-versus-full-frame control is run on a subset because cropping removes context that a detector may use.

The detector zoo comprises a CNN trained on the development split with gradient-based saliency for citation, and open-weight VLMs from at least three model families, spanning roughly two to seven billion parameters, run in half precision or four-bit quantisation as memory allows. Closed-weight models are excluded on principle: they expose no logits, so $\Delta$ is undefined for them, and this limitation will be stated.

### 5.2 Experiments

The experiments are grouped by the research question they serve. Each is listed with the detectors and data it uses.

*Instrument validation (RQ1).* Synthetic detectors on all frames; the four-way ordering of Section 4.4 must hold before any GPU budget is committed. Operator ablations over blend mode, dilation radius, and JPEG re-encoding, on the synthetic detectors and one VLM. An additivity test that compares the interaction term of dilation radius with and without the identity control, interpretable only at radii that already cover the manipulation. A two-region union intervention that tests super-additivity of the top-two cited regions.

*Inpainting invalidity (RQ2).* The same masks filled by a classical and a learned inpainter, on the CNN and one VLM; the quantities of interest are the inpainter's effect on authentic frames and the sign agreement with the splice-based estimate.

*Primary audit and dissociation (RQ3).* Detection accuracy, FS, reverse recall, seam magnitude, and floor for every detector; per manipulation method; at both vocabulary granularities; the correctness-by-faithfulness cross-tabulation; prompt-paraphrase stability over five verdict and five citation prompts frozen before any test-split number is computed; a visual-token-budget ablation; free-text-to-constrained agreement.

*Mechanism (RQ4).* Stated, attributed, and causal citations with their pairwise agreement; overlay-elicited citation; encoder-shift decomposition per detector and per region.

*Training (RQ5).* Base, causal-tuned, and mask-tuned variants of one VLM audited under the identical command on the test split and on the second dataset; detection accuracy, FS, reverse recall, correctness, and prompt stability side by side.

*Proxies.* All five proxies on the CNN and one VLM, with the validation criteria of Section 4.6.

*Human study.* A small annotator study on test-split items asks which explanation a person would trust and cross-tabulates preference against measured faithfulness, framing the question as preference versus faithfulness rather than as a plausibility score.

A stretch experiment applies the grid vocabulary to a paired original/edited image dataset outside the face domain, to show that the instrument is not specific to faces. It will be dropped if no suitable paired dataset with edit masks is available.

### 5.3 Statistical protocol

The protocol is fixed before test-split numbers are computed.

The unit of resampling is the clip, not the frame, because frames from one clip are not independent; confidence intervals are 95% percentile intervals from 2,000 cluster-bootstrap replicates. Multiple comparisons are corrected with Holm–Bonferroni across regions for per-region claims and across detectors for faithfulness verdicts; the uncorrected verdict is retained in the report only for diagnosis. Effect sizes are reported alongside significance: FS in units of the seam standard deviation, and a standardised difference between the cited and uncited effects.

The development/test split is 30/70 by hash of the unordered identity pair, so that no identity appears on both sides. Prompts are frozen constants in the released code. The tuning procedure trains on the development split only, enforced at dataset-build time. Human-study items are drawn from the test split. Sample sizes are pre-specified per detector.

Failure criteria are also pre-specified. If the pairing gate fails on a data mirror (the fake folder holds copies of the originals, or frame counts disagree), the mirror is replaced. If the seam effect is as large as the raw effect for every region, the result is reported as a negative finding about the operator. If fewer than three detectors run, the study is reported as a case study. If causal tuning lowers detection accuracy by more than the margin, it is reported as a trade-off.

---

## 6. Resources, Timeline, and Feasibility

### 6.1 Compute

The programme is sized for a free-tier cloud environment providing two 15 GB GPUs with a weekly quota of roughly thirty GPU-hours, sessions of at most twelve hours, and a 20 GB persistent output limit. The pipeline is organised into eleven session notebooks, each budgeted below eight hours, whose outputs chain as datasets; every long-running command carries a time budget and resumes from its own cache. Counterfactual images are built in memory and never written to disk, which keeps projected output under one gigabyte. One process is run per GPU, sharded by sample, and no model is split across the two cards.

Rough budget by stage: instrument validation and ablations on CPU; VLM smoke testing, one GPU-hour, before any quota is committed; CNN training and audit, about two hours; the primary VLM audit, two to six hours per model depending on size; inpainting, token-budget, and crop controls, about three hours; proxies and free text, about three hours; attribution and encoder probe, about three hours with the probe surcharge; adapter training and re-audit, about twelve hours; cross-dataset re-audit, about two hours. The total is in the region of forty to forty-five GPU-hours across six to seven weeks, which fits the quota with a buffer for re-runs.

### 6.2 Timeline

| Weeks | Activity |
|---|---|
| 1 | Pairing gate and parsing on the primary dataset; synthetic-detector validation; VLM smoke matrix; CNN training |
| 2–3 | Primary VLM audits; inpainting, token-budget, and crop controls; proxies; free text; attribution and encoder probe |
| 4 | Second dataset parsed and validated; development-split audit of the base VLM; adapter training under both supervision variants |
| 5 | Re-audit of base and tuned models on both datasets; localization, text mapping, metrics, figures; human-study materials |
| 6 | Buffer: re-runs, additional seeds, compression variant, stretch domain |
| 7 | Human study |
| 8–9 | Analysis and writing |

### 6.3 Software status

A pipeline implementing every component above exists as a Python package with seventeen modules, a self-test suite that exercises the metric code end to end on a synthetic fixture without a dataset or a GPU, a static checker for the platform's resource limits, and a generator for the session notebooks. Components that require a GPU or the real dataset have been implemented and reviewed but not executed; the first three sessions are designed as gates that check them cheaply before quota is committed.

---

## 7. Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| The data mirror is not frame-aligned | The counterfactual construction is invalid | A pairing gate with border-correlation and centre-identity checks runs first and exits non-zero on failure; the mirror is replaced |
| The splice seam dominates the effect | FS measures the operator, not the detector | Identity control subtracted per region; blend, dilation, and re-encode ablations; floor reported; pre-registered as a reportable negative |
| Seam and content effects are not additive | The difference-in-differences is biased | Additivity test at covering radii; residual reported with its interaction term |
| Small open VLMs behave differently from large closed ones | Findings do not generalise | Three model families across sizes; stated as a scope limitation; closed models excluded on principle because $\Delta$ is undefined without logits |
| The closed region vocabulary is artificial | Findings are specific to the prompt format | Free-text explanations collected and mapped by lexicon; agreement reported; granularity ablation; overlay elicitation as a second channel |
| Face crops remove context | Detectors are audited on a different input than they would see | Crop-versus-full-frame control on a subset |
| Gradient attribution exceeds GPU memory for the larger model | The three-way citation comparison is incomplete | Parameters frozen and gradient restricted to inputs; the larger model attempted last and skipped if it fails |
| A single benchmark family | Scope objection | Second face dataset; compression variant; stretch domain with a grid vocabulary |

---

## 8. Expected Contributions and Outputs

If the hypotheses hold, the project contributes, in order of confidence:

1. A validated, artifact-free instrument for measuring whether a region explanation is causally responsible for a forgery verdict, together with the conditions under which it is valid and the synthetic detectors that certify it.
2. Evidence on whether inpainting-based counterfactuals, the standard tool, are admissible for forgery detectors, and a remedy if they are not.
3. A separation between citation correctness and citation faithfulness that can be measured on any paired benchmark without annotation.
4. A mechanistic decomposition of unfaithful citations in VLMs into encoder blindness and language-side disregard, and a three-way comparison of stated, attributed, and causal citations.
5. A comparison of causal and mask supervision for explanation tuning, at matched detection accuracy.
6. A validated deployable proxy for faithfulness, if any candidate survives its floor check.

Outputs will be the code, the parsed index, the per-call caches, and the metric files, which reproduce every number from a licensed copy of the benchmark. Crops and counterfactual images derived from the benchmark will not be released until the benchmark's terms have been checked.

---

## 9. Ethical Considerations and Limitations

The benchmark data consist of manipulated videos of consenting subjects released for research under a licence that restricts redistribution; the project uses a mirror for compute-platform convenience and will cite the original dataset, state the mirror, and release no derived images without checking the terms. The human study collects preference judgements on already-public frames and no personal data.

The instrument has limitations that are inherent to its construction and will be stated as such. It requires a paired original, which restricts it to benchmark settings; the proxy experiment is the attempt to relax this. The closed vocabulary constrains what a citation can be; free-text agreement and the overlay channel are the checks. Poisson blending suppresses low-frequency content at the seam, so evidence that lives at low spatial frequencies may be under-measured. Only open-weight models of modest size can be audited. And the difference-in-differences assumes additivity of seam and content, which is tested but cannot be guaranteed for every detector.

---

## References (indicative)

The related-work discussion in Section 2 draws on the following lines of work; a full bibliography will accompany the manuscript.

- Explainable forgery detection with textual and mask outputs, and the benchmarks that evaluate them for plausibility and localization.
- Faithfulness tests for language-model self-explanations and chain-of-thought.
- Remove-and-retrain and deletion-curve evaluation of pixel attributions, including the analysis of imputation artifacts and the noisy-linear-imputation remedy.
- Analyses of what deepfake detectors match between source and target, and of identity leakage.
- Region-prompting for vision-language models by visual markers and by alpha channels.
- Diagnostic studies of vision-encoder blind spots in multimodal models.
- Supervised-attention and explanation-regularisation training.
