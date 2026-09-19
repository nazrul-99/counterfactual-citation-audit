"""
ccaudit.m4_detectors -- the detector zoo.

Every detector exposes the same read-out, so the audit code never branches on
which model it is talking to:

    predict(img, labels=None, variant=0) -> float in [0,1]   p(manipulated)
    cite(img, labels=None, variant=0)    -> {region: prob}    sums to 1
    .name  .explainable  .close()

Optional capabilities, used only when the corresponding runner flag is set:

    explain(img, max_new_tokens=60)      -> str     free-text explanation
    p_ignore(img, region)                -> float   verdict when asked to ignore a region
    p_overlay(img)                       -> float   verdict for a frame with one region outlined
    attribute(img, labels=None, variant=0) -> {region: p}  gradient citation
    encoder_features(img)                -> (tokens, grid_hw)   visual tokens at the connector

Synthetic controls
------------------
Four synthetic detectors are provided whose faithfulness is known by
construction, so that the metrics can be validated against them:

    adaptive_oracle       verdict and citation share the same evidence
    fixed_oracle[region]  verdict driven by one region, always cited
    confabulator          verdict uses the evidence, citation uses region area
    dummy                 no signal, uniform citation

The three non-dummy controls share one evidence function: the local
high-frequency energy of the image, contrasted between each region and a
thin ring around it.  Within-image comparison lets the controls operate on
any dataset without a training set: a synthesised region is smoother (or
sharper) than the authentic pixels beside it, and the only question asked is
which region is the anomalous one in a given picture.

The sign of that anomaly is dataset dependent, so the controls calibrate it
once from a few paired samples by measuring the artifact in the region that
actually changed.  The calibration is recorded in provenance and printed.
Using paired frames is legitimate for a synthetic control and is what makes
its ground-truth faithfulness knowable; a real detector never receives this
information.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C
from . import regions as R

EPS = 1e-8


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
# Five paraphrases each.  The option letters are identical across paraphrases
# so the scored token ids never change; only the wording varies, which is
# what the prompt-stability ablation requires.

VERDICT_PROMPTS: List[str] = [
    "Look at this face image. Has it been digitally manipulated?\n"
    "A. No, it is authentic\nB. Yes, it has been manipulated\n"
    "Answer with a single letter.",

    "Is the face in this photograph real or synthetically altered?\n"
    "A. No, it is authentic\nB. Yes, it has been manipulated\n"
    "Answer with a single letter.",

    "You are a forensic analyst. Judge whether this facial image shows signs "
    "of digital forgery.\nA. No, it is authentic\nB. Yes, it has been "
    "manipulated\nAnswer with a single letter.",

    "Consider the image. Do any parts of this face appear to have been "
    "generated or edited?\nA. No, it is authentic\nB. Yes, it has been "
    "manipulated\nAnswer with a single letter.",

    "Determine the authenticity of the pictured face.\n"
    "A. No, it is authentic\nB. Yes, it has been manipulated\n"
    "Answer with a single letter.",
]

CITE_PROMPTS: List[str] = [
    "This face image has been digitally manipulated. Which facial region "
    "shows the manipulation?\n{options}\nAnswer with a single letter.",

    "Something in this face has been edited. Which region is it?\n"
    "{options}\nAnswer with a single letter.",

    "You are a forensic analyst examining a forged face. Name the region that "
    "carries the evidence of forgery.\n{options}\nAnswer with a single letter.",

    "Where in this face is the manipulation located?\n"
    "{options}\nAnswer with a single letter.",

    "The image contains a manipulated facial region. Identify which one.\n"
    "{options}\nAnswer with a single letter.",
]

IGNORE_PROMPT = (
    "Ignoring the {region} completely, has this face been manipulated?\n"
    "A. No, it is authentic\nB. Yes, it has been manipulated\n"
    "Answer with a single letter."
)

OVERLAY_PROMPT = (
    "A region of this face is outlined in red. Is the digital manipulation "
    "located inside the outlined region?\nA. No\nB. Yes\n"
    "Answer with a single letter."
)

EXPLAIN_PROMPT = (
    "Explain in one sentence which facial region shows the manipulation and why."
)


def cite_options(vocab: Optional[str] = None) -> str:
    """The citation menu.  All regions are always listed, in vocabulary order,
    with fixed letters, so the scored token ids are identical for every sample,
    every prompt paraphrase and every detector."""
    names = R.get_vocab(vocab)
    return "\n".join(f"{L}. {R.prompt_name(n, vocab)}"
                     for L, n in zip(R.letters(len(names)), names))


def verdict_prompt(variant: int = 0) -> str:
    return VERDICT_PROMPTS[int(variant) % len(VERDICT_PROMPTS)]


def cite_prompt(variant: int = 0, vocab: Optional[str] = None) -> str:
    tpl = CITE_PROMPTS[int(variant) % len(CITE_PROMPTS)]
    return tpl.format(options=cite_options(vocab))


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------

class Detector:
    name: str = "detector"
    explainable: bool = True
    needs_calibration: bool = False

    def predict(self, img: np.ndarray, labels: Optional[np.ndarray] = None,
                variant: int = 0) -> float:
        raise NotImplementedError

    def cite(self, img: np.ndarray, labels: Optional[np.ndarray] = None,
             variant: int = 0) -> Dict[str, float]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "explainable": self.explainable,
                "class": type(self).__name__}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _normalise(d: Dict[str, float], vocab: Optional[str] = None
                   ) -> Dict[str, float]:
        names = R.get_vocab(vocab)
        vals = np.array([max(0.0, float(d.get(n, 0.0))) for n in names])
        s = vals.sum()
        if s <= 0:
            vals = np.ones(len(names))
            s = vals.sum()
        return {n: float(v / s) for n, v in zip(names, vals)}


# --------------------------------------------------------------------------
# the shared evidence function of the controls
# --------------------------------------------------------------------------

def artifact_map(img: np.ndarray, sigma: float = 1.5, pool: float = 2.5
                 ) -> np.ndarray:
    """Local high-frequency energy: |I - blur(I)|, then locally pooled."""
    import cv2

    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    hf = np.abs(g - cv2.GaussianBlur(g, (0, 0), sigma))
    return cv2.GaussianBlur(hf, (0, 0), pool) if pool > 0 else hf


def region_contrast(img: np.ndarray, labels: np.ndarray,
                    vocab: Optional[str] = None, min_px: int = 32,
                    ring_px: int = 6) -> Dict[str, float]:
    """
    Per-region local artifact contrast:

        contrast_k = ( mean artifact in a thin ring just outside region k
                       - mean artifact inside region k ) / image artifact std

    The contrast is local rather than global.  A synthesised region is
    smoother than the authentic pixels immediately around it, and that
    relation survives global brightness, sharpness or compression changes.
    A within-image z-score across regions would not, because it is mean-zero
    across regions by construction and therefore discards the level that
    separates real from fake.

    Splicing the authentic pixels back into region k raises the inside term
    and lowers the contrast, which is what makes the controls' forward
    behaviour predictable.
    """
    import cv2

    a = artifact_map(img)
    scale = float(a.std()) or 1.0
    k = int(2 * ring_px + 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    names = R.get_vocab(vocab)
    out: Dict[str, float] = {}
    for n in names:
        m = (labels == R.rid(n, vocab)).astype(np.uint8)
        if int(m.sum()) < min_px:
            out[n] = 0.0
            continue
        ring = cv2.dilate(m, kern, 1).astype(bool) & (~m.astype(bool))
        if int(ring.sum()) < min_px:
            out[n] = 0.0
            continue
        out[n] = float((a[ring].mean() - a[m.astype(bool)].mean()) / scale)
    return out


class _EvidenceControl(Detector):
    """
    Common machinery for adaptive_oracle / fixed_oracle / confabulator.

    Calibration proceeds in two stages, from a few paired samples of the index:

      1. Per-region baseline.  The natural local contrast of `skin` (smooth,
         ringed by textured hair) differs from that of `hair`, so comparing
         raw contrasts across regions measures anatomy rather than forgery.
         The mean and standard deviation of each region's contrast are
         estimated on authentic frames and used to standardise, so that
         evidence_k measures how unusual region k is for that region.
      2. Sign and offset.  The sign records which way the artifact moves when
         a region becomes synthetic (smoother or sharper; dataset dependent).
         The offset is placed midway between the detector's own statistic on
         authentic and on manipulated frames, so p ~ 0.5 sits between the two
         populations and the AUC is meaningful.

    A real detector never receives this calibration.  A control does, and that
    is what makes its ground-truth faithfulness knowable.
    """

    needs_calibration = True

    def __init__(self, gain: float = 2.0, cite_gain: float = 2.0):
        self.sign = 1.0
        self.offset = 0.0
        self.gain = float(gain)
        self.cite_gain = float(cite_gain)
        self.mu: Dict[str, float] = {}
        self.sd: Dict[str, float] = {}
        self.calibration: Dict[str, Any] = {"calibrated": False}

    # -- the statistic each control turns region evidence into -----------
    def _statistic(self, ev: Dict[str, float]) -> float:
        return max(ev.values()) if ev else 0.0

    # -- calibration -----------------------------------------------------
    def calibrate(self, records: Sequence[Dict[str, Any]], n: int = 32,
                  seed: int = 0, vocab: Optional[str] = None) -> Dict[str, Any]:
        from .m2_parse import manipulation_mask

        recs = C.limit_samples(list(records), n, seed)
        names = R.get_vocab(vocab)
        c_real: List[Dict[str, float]] = []
        c_fake: List[Dict[str, float]] = []
        shift_at_gt: List[float] = []

        for rec in recs:
            try:
                real = C.imread(rec["real"])
                fake = C.imread(rec["fake"])
                lab = C.imread(rec["lab"], 0)
            except Exception:
                continue
            mgt, _ = manipulation_mask(real, fake)
            if mgt.sum() < 16:
                continue
            frac = {}
            for nm in names:
                m = lab == R.rid(nm, vocab)
                area = int(m.sum())
                frac[nm] = float((mgt.astype(bool) & m).sum()) / area if area else 0.0
            gt = max(frac, key=frac.get)
            cr = region_contrast(real, lab, vocab)
            cf = region_contrast(fake, lab, vocab)
            c_real.append(cr)
            c_fake.append(cf)
            if frac[gt] > 0:
                shift_at_gt.append(cf[gt] - cr[gt])

        if len(c_real) < 4:
            self.calibration = {"calibrated": False, "n_used": len(c_real)}
            print(f"[m4] {self.name}: calibration SKIPPED "
                  f"(only {len(c_real)} usable samples)", flush=True)
            return self.calibration

        # Stage 1: per-region baseline from authentic frames.
        for nm in names:
            vals = np.array([c[nm] for c in c_real], float)
            vals = vals[np.isfinite(vals)]
            self.mu[nm] = float(vals.mean()) if vals.size else 0.0
            sd = float(vals.std()) if vals.size else 1.0
            self.sd[nm] = sd if sd > 1e-3 else 1.0

        # Stage 2: sign, then offset from this control's own statistic.
        self.sign = 1.0 if float(np.mean(shift_at_gt or [1.0])) > 0 else -1.0
        st_real = [self._statistic(self._standardise(c)) for c in c_real]
        st_fake = [self._statistic(self._standardise(c)) for c in c_fake]
        self.offset = float((np.mean(st_real) + np.mean(st_fake)) / 2.0)

        self.calibration = {
            "calibrated": True, "n_used": len(c_real),
            "mean_shift_at_gt_region": float(np.mean(shift_at_gt or [0.0])),
            "sign": self.sign, "offset": self.offset,
            "statistic_real_mean": float(np.mean(st_real)),
            "statistic_fake_mean": float(np.mean(st_fake)),
            "separation": float(np.mean(st_fake) - np.mean(st_real)),
            "region_mu": dict(self.mu), "region_sd": dict(self.sd),
        }
        print(f"[m4] {self.name}: sign={self.sign:+.0f} offset={self.offset:.3f} "
              f"separation={self.calibration['separation']:+.3f} "
              f"(n={len(c_real)})", flush=True)
        return self.calibration

    # -- evidence --------------------------------------------------------
    def _standardise(self, contrast: Dict[str, float]) -> Dict[str, float]:
        return {k: self.sign * (v - self.mu.get(k, 0.0)) / self.sd.get(k, 1.0)
                for k, v in contrast.items()}

    def _evidence(self, img: np.ndarray, labels: np.ndarray,
                  vocab: Optional[str] = None) -> Dict[str, float]:
        return self._standardise(region_contrast(img, labels, vocab))

    def _p_from(self, value: float) -> float:
        return float(1.0 / (1.0 + math.exp(
            -self.gain * (value - self.offset))))

    def info(self) -> Dict[str, Any]:
        return {**super().info(), "calibration": self.calibration}


class AdaptiveOracle(_EvidenceControl):
    """
    Faithful by construction: the verdict is driven by the most anomalous
    region and the citation is a softmax over the same per-region evidence.
    Removing the cited region's evidence therefore lowers p, and injecting a
    region's evidence into an authentic frame moves the citation to that
    region.
    """
    name = "adaptive_oracle"

    def predict(self, img, labels=None, variant=0) -> float:
        if labels is None:
            return 0.5
        return self._p_from(self._statistic(self._evidence(img, labels)))

    def cite(self, img, labels=None, variant=0) -> Dict[str, float]:
        if labels is None:
            return self._normalise({})
        ev = self._evidence(img, labels)
        names = R.get_vocab()
        w = np.array([ev[n] for n in names]) * self.cite_gain
        w = np.exp(w - w.max())
        return {n: float(v) for n, v in zip(names, w / w.sum())}


class FixedOracle(_EvidenceControl):
    """
    Forward-only by construction: the verdict is driven by a single fixed
    region and the citation always names that region.  Removing the cited
    region's evidence lowers p, but the citation carries no information across
    samples, so reverse citation recall cannot exceed the prior.  This control
    separates the forward and reverse tests.
    """

    def __init__(self, region: str = "mouth", **kw):
        super().__init__(**kw)
        if region not in R.get_vocab():
            raise ValueError(f"fixed_oracle region {region!r} not in vocabulary")
        self.region = region
        self.name = f"fixed_oracle[{region}]"

    def _statistic(self, ev: Dict[str, float]) -> float:
        """A single region drives the verdict."""
        return float(ev.get(self.region, 0.0))

    def predict(self, img, labels=None, variant=0) -> float:
        if labels is None:
            return 0.5
        return self._p_from(self._statistic(self._evidence(img, labels)))

    def cite(self, img, labels=None, variant=0) -> Dict[str, float]:
        names = R.get_vocab()
        eps = 0.02 / max(1, len(names) - 1)
        return {n: (0.98 if n == self.region else eps) for n in names}


class Confabulator(_EvidenceControl):
    """
    Unfaithful with a discriminative verdict: the verdict uses the artifact
    evidence, but the citation is a function of region area alone and always
    points at the largest region, which is unrelated to what drove the
    verdict.  A faithfulness score that does not stay near zero for this
    control is measuring something other than causal responsibility.
    """
    name = "confabulator"

    def predict(self, img, labels=None, variant=0) -> float:
        if labels is None:
            return 0.5
        return self._p_from(self._statistic(self._evidence(img, labels)))

    def cite(self, img, labels=None, variant=0) -> Dict[str, float]:
        if labels is None:
            return self._normalise({})
        names = R.get_vocab()
        areas = R.region_areas(labels)
        w = np.array([float(areas[n]) ** 2 for n in names])
        if w.sum() <= 0:
            w = np.ones(len(names))
        return {n: float(v) for n, v in zip(names, w / w.sum())}


class Dummy(Detector):
    """No signal: a content-hashed verdict near 0.5 and a uniform citation."""
    name = "dummy"

    def predict(self, img, labels=None, variant=0) -> float:
        h = hashlib.blake2b(np.ascontiguousarray(img).tobytes()[:4096],
                            digest_size=4).hexdigest()
        return 0.45 + 0.10 * (int(h, 16) % 1000) / 1000.0

    def cite(self, img, labels=None, variant=0) -> Dict[str, float]:
        names = R.get_vocab()
        return {n: 1.0 / len(names) for n in names}


# --------------------------------------------------------------------------
# VLM detectors
# --------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def _dtype_kwarg(dtype) -> Dict[str, Any]:
    """transformers >= 4.56 renamed torch_dtype to dtype."""
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        if (major, minor) >= (4, 56):
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


class _LogitVLM(Detector):
    """
    Shared implementation of the single-letter logit read-out.

    Neither the verdict nor the citation is obtained by generation.  One
    forward pass gives the logits of the next token; the logits of the option
    letters are read and a softmax is taken over exactly those.  The read-out
    is deterministic, needs no sampling, and yields a probability rather than
    a string that would have to be parsed.
    """

    explainable = True

    def __init__(self, model_id: str, device: str = "cuda", quant: str = "fp16",
                 max_pixels: int = 384 * 384, min_pixels: int = 64 * 64,
                 lora: str = ""):
        torch = _torch()
        from transformers import AutoProcessor

        self.model_id = model_id
        self.device = device
        self.quant = quant
        self.max_pixels = int(max_pixels)
        self.name = f"{self._prefix}:{model_id.split('/')[-1]}"

        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=int(min_pixels), max_pixels=int(max_pixels),
            trust_remote_code=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        load_kw: Dict[str, Any] = {"trust_remote_code": True}
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            load_kw["device_map"] = {"": 0}
        else:
            dt = {"fp16": torch.float16, "bf16": torch.bfloat16,
                  "auto": "auto"}.get(quant, torch.float16)
            load_kw.update(_dtype_kwarg(dt))
            load_kw["device_map"] = {"": 0} if device.startswith("cuda") else None

        self.model = self._load_model(model_id, load_kw)
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
            self.name += f"+lora:{os.path.basename(lora.rstrip('/'))}"
            print(f"[m4] LoRA adapter loaded from {lora}", flush=True)
        self.model.eval()
        if device.startswith("cuda") and quant != "4bit":
            self.model.to(device)

        self._letter_ids = self._build_letter_ids()
        self._hook_store: Dict[str, Any] = {}
        self._hook_handle = None

    # -- subclass hooks --------------------------------------------------
    _prefix = "vlm"

    def _load_model(self, model_id: str, kw: Dict[str, Any]):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, **kw)

    # -- letters ---------------------------------------------------------
    def _build_letter_ids(self) -> Dict[str, List[int]]:
        tok = self.tokenizer
        out: Dict[str, List[int]] = {}
        for L in R.letters(26):
            ids: List[int] = []
            for form in (L, " " + L):
                try:
                    enc = tok.encode(form, add_special_tokens=False)
                except Exception:
                    continue
                if len(enc) == 1:
                    ids.append(int(enc[0]))
            if not ids:                      # multi-token letter: take the first
                enc = tok.encode(L, add_special_tokens=False)
                if enc:
                    ids.append(int(enc[0]))
            out[L] = sorted(set(ids))
        return out

    def _letter_scores(self, logits, letters_: Sequence[str]) -> np.ndarray:
        """Max logit over the token forms of each letter."""
        vals = []
        for L in letters_:
            ids = self._letter_ids.get(L, [])
            vals.append(float(max(logits[i].item() for i in ids)) if ids else -1e9)
        return np.array(vals, dtype=np.float64)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # -- forward ---------------------------------------------------------
    def _to_pil(self, img: np.ndarray):
        import cv2
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _build_inputs(self, img: np.ndarray, prompt: str):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self._to_pil(img)],
                                return_tensors="pt")
        return {k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

    def _next_logits(self, img: np.ndarray, prompt: str):
        torch = _torch()
        inputs = self._build_inputs(img, prompt)
        kw = {}
        try:                       # avoid materialising the whole logit tensor
            import inspect
            if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
                kw["logits_to_keep"] = 1
        except Exception:
            pass
        with torch.inference_mode():
            out = self.model(**inputs, **kw)
        return out.logits[0, -1, :].float()

    # -- API -------------------------------------------------------------
    def predict(self, img, labels=None, variant=0) -> float:
        logits = self._next_logits(img, verdict_prompt(variant))
        s = self._letter_scores(logits, ["A", "B"])
        return float(self._softmax(s)[1])          # B = manipulated

    def cite(self, img, labels=None, variant=0) -> Dict[str, float]:
        names = R.get_vocab()
        logits = self._next_logits(img, cite_prompt(variant))
        s = self._letter_scores(logits, R.letters(len(names)))
        p = self._softmax(s)
        return {n: float(v) for n, v in zip(names, p)}

    def p_ignore(self, img: np.ndarray, region: str) -> float:
        """Self-check proxy: p(manipulated) when the model is instructed to
        ignore `region`, i.e. its own report of the counterfactual."""
        logits = self._next_logits(img, IGNORE_PROMPT.format(
            region=R.prompt_name(region)))
        return float(self._softmax(self._letter_scores(logits, ["A", "B"]))[1])

    def p_overlay(self, img: np.ndarray) -> float:
        logits = self._next_logits(img, OVERLAY_PROMPT)
        return float(self._softmax(self._letter_scores(logits, ["A", "B"]))[1])

    def explain(self, img: np.ndarray, max_new_tokens: int = 60) -> str:
        torch = _torch()
        inputs = self._build_inputs(img, EXPLAIN_PROMPT)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(ids[0, n_in:], skip_special_tokens=True).strip()

    # -- visual-token internals (gradient citation, encoder probe) --------
    def _visual_module(self):
        for attr in ("visual", "vision_tower", "vision_model", "model.visual"):
            obj = self.model
            ok = True
            for part in attr.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                merger = getattr(obj, "merger", None)
                return merger if merger is not None else obj
        raise RuntimeError(
            f"{self.name}: could not locate the visual projector; "
            f"--encoder-probe / attribute() are unavailable for this model")

    def _capture(self, module, inputs, output):
        self._hook_store["feat"] = output

    def encoder_features(self, img: np.ndarray
                         ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Visual tokens at the projector output, plus the (h, w) token grid."""
        torch = _torch()
        mod = self._visual_module()
        self._hook_store.clear()
        h = mod.register_forward_hook(self._capture)
        try:
            inputs = self._build_inputs(img, verdict_prompt(0))
            with torch.inference_mode():
                self.model(**inputs)
            feat = self._hook_store.get("feat")
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            arr = feat.detach().float().cpu().numpy().reshape(-1, feat.shape[-1])
            grid = self._token_grid(inputs, arr.shape[0])
        finally:
            h.remove()
        return arr, grid

    def _token_grid(self, inputs, n_tokens: int) -> Tuple[int, int]:
        thw = inputs.get("image_grid_thw")
        if thw is not None:
            t, gh, gw = [int(v) for v in np.array(thw.cpu()).reshape(-1)[:3]]
            merge = getattr(getattr(self.model, "config", None),
                            "vision_config", None)
            m = int(getattr(merge, "spatial_merge_size", 2) or 2)
            return max(1, gh // m), max(1, gw // m)
        side = int(round(math.sqrt(max(1, n_tokens))))
        return side, side

    def attribute(self, img: np.ndarray, labels: Optional[np.ndarray] = None,
                  variant: int = 0) -> Dict[str, float]:
        """
        Gradient-times-input attribution of the manipulated-option logit onto
        the visual tokens at the projector output, pooled per region.

        All model parameters are frozen for the duration of the call and
        `pixel_values` is marked as requiring grad, so the backward pass
        allocates gradients only along the input path.  This also makes the
        projector output differentiable for 4-bit or LoRA-only models, whose
        parameters may not require grad themselves.  The scored logit is the
        token form of the letter B with the highest logit, matching the set
        of forms used by `_letter_scores`.
        """
        if labels is None:
            raise ValueError("attribute() needs the label map")
        mod = self._visual_module()
        store: Dict[str, Any] = {}

        def hook(_m, _i, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if not o.requires_grad:
                raise RuntimeError(
                    f"{self.name}: the visual projector output does not require "
                    "grad, so gradient attribution cannot be computed; the "
                    "image input is not connected to the projector by a "
                    "differentiable path in this model")
            o.retain_grad()
            store["feat"] = o
            return out

        b_ids = self._letter_ids.get("B", [])
        if not b_ids:
            raise RuntimeError(f"{self.name}: no token id for option letter B")

        frozen = [p for p in self.model.parameters() if p.requires_grad]
        for p in frozen:
            p.requires_grad_(False)
        h = mod.register_forward_hook(hook)
        try:
            inputs = self._build_inputs(img, verdict_prompt(variant))
            pv = inputs.get("pixel_values")
            if pv is not None and hasattr(pv, "requires_grad_"):
                inputs["pixel_values"] = pv.detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            out = self.model(**inputs)
            last = out.logits[0, -1]
            b_id = max(b_ids, key=lambda i: float(last[i].item()))
            logit_b = last[b_id]
            logit_b.backward()
            feat = store["feat"]
            g = feat.grad
            if g is None:
                raise RuntimeError("no gradient reached the visual projector")
            score = (g * feat).abs().sum(-1).detach().float().cpu().numpy().reshape(-1)
            grid = self._token_grid(inputs, score.size)
        finally:
            h.remove()
            self.model.zero_grad(set_to_none=True)
            for p in frozen:
                p.requires_grad_(True)
        return tokens_to_regions(score, grid, labels)

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def info(self) -> Dict[str, Any]:
        return {**super().info(), "model_id": self.model_id,
                "quant": self.quant, "max_pixels": self.max_pixels}


class QwenVLDetector(_LogitVLM):
    _prefix = "qwen25vl"

    def _load_model(self, model_id: str, kw: Dict[str, Any]):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as M
        return M.from_pretrained(model_id, **kw)


class HFVLMDetector(_LogitVLM):
    """Any model that AutoModelForImageTextToText can load with the same
    chat-template + single-letter read-out (Qwen2-VL-2B, SmolVLM, LLaVA-OV)."""
    _prefix = "hfvlm"


def tokens_to_regions(token_scores: np.ndarray, grid_hw: Tuple[int, int],
                      labels: np.ndarray, vocab: Optional[str] = None
                      ) -> Dict[str, float]:
    """
    Map a per-visual-token score onto the region vocabulary by resizing the
    token grid to the label map and summing inside each region.  Returns a
    normalised distribution.
    """
    import cv2

    gh, gw = grid_hw
    n = int(gh * gw)
    s = np.asarray(token_scores, np.float32).reshape(-1)
    if s.size < n:
        s = np.pad(s, (0, n - s.size))
    grid = s[:n].reshape(gh, gw)
    up = cv2.resize(grid, (labels.shape[1], labels.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    names = R.get_vocab(vocab)
    mass = {}
    for nm in names:
        m = labels == R.rid(nm, vocab)
        mass[nm] = float(up[m].sum()) if m.any() else 0.0
    tot = sum(mass.values())
    if tot <= 0:
        return {nm: 1.0 / len(names) for nm in names}
    return {nm: v / tot for nm, v in mass.items()}


def draw_region_outline(img: np.ndarray, labels: np.ndarray, region: str,
                        thickness: int = 3, vocab: Optional[str] = None
                        ) -> np.ndarray:
    """Visual marker in the ViP-LLaVA style: a red contour around region k,
    with no fill."""
    import cv2

    m = (labels == R.rid(region, vocab)).astype(np.uint8)
    out = img.copy()
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), thickness)
    return out


# --------------------------------------------------------------------------
# CNN + Grad-CAM
# --------------------------------------------------------------------------

class CNNGradCAMDetector(Detector):
    """
    A conventional classifier explained by Grad-CAM projected onto the region
    vocabulary.  The decision never sees the masks; only the projection of the
    heat map onto the closed vocabulary uses them, as a saliency-based
    explainable detector would be evaluated.
    """

    explainable = True

    def __init__(self, ckpt: str = "", arch: str = "efficientnet_b0",
                 device: str = "cuda", input_size: int = 256):
        import timm
        import torch

        self.device = device
        if ckpt and os.path.exists(ckpt):
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            arch = blob.get("arch", arch)
            input_size = int(blob.get("input_size", input_size))
            self.model = timm.create_model(arch, pretrained=False, num_classes=2)
            self.model.load_state_dict(blob["state_dict"])
            self.trained = True
            self.provenance = blob.get("provenance", {})
        else:
            self.model = timm.create_model(arch, pretrained=False, num_classes=2)
            self.trained = False
            self.provenance = {}
            print(f"[m4] WARNING: CNN {arch} is UNTRAINED (no checkpoint at "
                  f"{ckpt!r}). Useful only for smoke tests.", flush=True)
        self.arch = arch
        self.input_size = input_size
        self.name = f"cnn-{arch}"
        self.model.eval().to(device)
        self._feat: Dict[str, Any] = {}
        self._target = self._find_target_layer()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def _find_target_layer(self):
        mods = [m for m in self.model.modules()
                if m.__class__.__name__ in ("Conv2d", "Conv2dSame")]
        if not mods:
            raise RuntimeError("no conv layer found for Grad-CAM")
        return mods[-1]

    def _tensor(self, img: np.ndarray):
        import cv2
        import torch

        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if x.shape[0] != self.input_size:
            x = cv2.resize(x, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_AREA)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)

    def predict(self, img, labels=None, variant=0) -> float:
        import torch

        with torch.inference_mode():
            logits = self.model(self._tensor(img))
            return float(torch.softmax(logits, dim=-1)[0, 1].item())

    def _gradcam(self, img: np.ndarray) -> np.ndarray:
        import torch

        acts: Dict[str, Any] = {}

        def fwd(_m, _i, o):
            o.retain_grad()
            acts["a"] = o

        h = self._target.register_forward_hook(fwd)
        try:
            x = self._tensor(img)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            logits[0, 1].backward()
            a = acts["a"]
            w = a.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * a).sum(dim=1, keepdim=True))
            cam = cam[0, 0].detach().float().cpu().numpy()
        finally:
            h.remove()
            self.model.zero_grad(set_to_none=True)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def cite(self, img, labels=None, variant=0) -> Dict[str, float]:
        import cv2

        if labels is None:
            return self._normalise({})
        cam = self._gradcam(img)
        up = cv2.resize(cam, (labels.shape[1], labels.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
        names = R.get_vocab()
        mass = {}
        for n in names:
            m = labels == R.rid(n)
            mass[n] = float(up[m].mean()) if m.any() else 0.0
        return self._normalise(mass)

    def attribute(self, img, labels=None, variant=0) -> Dict[str, float]:
        return self.cite(img, labels, variant)

    def info(self) -> Dict[str, Any]:
        return {**super().info(), "arch": self.arch, "trained": self.trained,
                "input_size": self.input_size,
                "train_provenance": self.provenance}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

REGISTRY = {
    "dummy": Dummy,
    "confabulator": Confabulator,
    "adaptive_oracle": AdaptiveOracle,
    "fixed_oracle": FixedOracle,
    "qwen25vl": QwenVLDetector,
    "hfvlm": HFVLMDetector,
    "cnn": CNNGradCAMDetector,
}

CONTROL_NAMES = ("dummy", "confabulator", "adaptive_oracle", "fixed_oracle")


def build_detector(spec: str, device: str = "cuda", quant: str = "fp16",
                   max_pixels: int = 384 * 384) -> Detector:
    """
    Build a detector from a spec string:

        dummy
        confabulator
        adaptive_oracle
        fixed_oracle:mouth
        qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct
        qwen25vl:Qwen/Qwen2.5-VL-3B-Instruct:lora=/path/to/cset/causal
        hfvlm:HuggingFaceTB/SmolVLM-Instruct
        cnn:/path/to/cnn_efficientnet_b0.pt
        cnn:arch=efficientnet_b0
    """
    spec = spec.strip()
    head, _, rest = spec.partition(":")
    head = head.strip()
    if head not in REGISTRY:
        raise ValueError(f"unknown detector {head!r}; have {sorted(REGISTRY)}")

    if head in ("dummy", "confabulator", "adaptive_oracle"):
        return REGISTRY[head]()
    if head == "fixed_oracle":
        return FixedOracle(rest.strip() or "mouth")
    if head == "cnn":
        arg = rest.strip()
        if arg.startswith("arch="):
            return CNNGradCAMDetector(arch=arg[5:], device=device)
        return CNNGradCAMDetector(ckpt=arg, device=device)

    # VLMs: the model id may be followed by :lora=<dir>.
    lora = ""
    model_id = rest.strip()
    if ":lora=" in model_id:
        model_id, _, lora = model_id.partition(":lora=")
    cls = QwenVLDetector if head == "qwen25vl" else HFVLMDetector
    return cls(model_id.strip(), device=device, quant=quant,
               max_pixels=max_pixels, lora=lora.strip())


def build_detectors(specs: Sequence[str], **kw) -> List[Detector]:
    return [build_detector(s, **kw) for s in specs if s.strip()]


# --------------------------------------------------------------------------
# VLM smoke test
# --------------------------------------------------------------------------

def smoke_vlm(spec: str, device: str = "cuda", quant: str = "fp16",
              max_pixels: int = 384 * 384, n: int = 4,
              index: str = "") -> Dict[str, Any]:
    """
    Load the model and check the invariants a full run relies on: p in [0,1];
    citations normalised over the vocabulary; determinism across two identical
    calls; and measured throughput.
    """
    import time as _t

    res: Dict[str, Any] = {"spec": spec, "ok": False, "errors": []}
    t0 = _t.time()
    det = build_detector(spec, device=device, quant=quant, max_pixels=max_pixels)
    res["load_sec"] = _t.time() - t0
    res["info"] = det.info()

    if index:
        recs, meta = C.load_index(index)
        R.set_vocab(meta.get("vocab", "face8"))
        recs = C.limit_samples(recs, n, 0)
        imgs = [(C.imread(r["fake"]), C.imread(r["lab"], 0)) for r in recs]
    else:
        rng = np.random.default_rng(0)
        imgs = [(rng.integers(0, 255, (384, 384, 3), dtype=np.uint8),
                 np.zeros((384, 384), np.uint8)) for _ in range(n)]

    ps, cites, times = [], [], []
    for img, lab in imgs:
        t = _t.time()
        p = det.predict(img)
        c = det.cite(img, lab)
        times.append(_t.time() - t)
        ps.append(p)
        cites.append(c)
        if not (0.0 <= p <= 1.0):
            res["errors"].append(f"p out of range: {p}")
        s = sum(c.values())
        if abs(s - 1.0) > 1e-4:
            res["errors"].append(f"citation not normalised: sum={s}")
        if set(c) != set(R.get_vocab()):
            res["errors"].append("citation keys do not match the vocabulary")

    p1 = det.predict(imgs[0][0])
    p2 = det.predict(imgs[0][0])
    res["determinism_delta"] = abs(p1 - p2)
    if res["determinism_delta"] > 1e-4:
        res["errors"].append(
            f"NON-DETERMINISTIC: two identical calls differ by "
            f"{res['determinism_delta']:.2e}")

    sec_per_call = float(np.mean(times)) / 2.0
    res.update({
        "p_values": ps,
        "mean_sec_per_call": sec_per_call,
        "samples_per_gpu_hour": (3600.0 / (35 * sec_per_call))
        if sec_per_call > 0 else float("nan"),
        "cite_example": cites[0],
        "ok": not res["errors"],
    })
    det.close()
    print(f"[smoke] {spec}: load {res['load_sec']:.0f}s, "
          f"{sec_per_call:.3f} s/call, "
          f"{res['samples_per_gpu_hour']:.0f} samples/GPU-hour, "
          f"determinism {res['determinism_delta']:.2e}, "
          f"{'OK' if res['ok'] else 'FAILED: ' + '; '.join(res['errors'])}",
          flush=True)
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m ccaudit.m4_detectors")
    ap.add_argument("--smoke-vlm", default="", help="detector spec(s), comma separated")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default="fp16")
    ap.add_argument("--max-pixels", type=int, default=384 * 384)
    ap.add_argument("--index", default="", help="use real crops for the smoke test")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="", help="write the smoke report here")
    a = ap.parse_args(argv)

    if not a.smoke_vlm:
        print("nothing to do; pass --smoke-vlm <spec>")
        return 0
    out = []
    for spec in a.smoke_vlm.split(","):
        spec = spec.strip()
        if not spec:
            continue
        try:
            out.append(smoke_vlm(spec, a.device, a.quant, a.max_pixels, a.n, a.index))
        except Exception as exc:
            print(f"[smoke] {spec}: EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            out.append({"spec": spec, "ok": False,
                        "errors": [f"{type(exc).__name__}: {exc}"]})
    if a.out:
        C.save_json(a.out, {"results": out, "provenance": C.provenance()}, indent=1)
        print(f"smoke report -> {a.out}")
    return 0 if all(r.get("ok") for r in out) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
