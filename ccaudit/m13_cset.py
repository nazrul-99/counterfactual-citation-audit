"""
ccaudit.m13_cset -- Module 13, causally-supervised explanation tuning.

Fine-tunes a VLM's citation head with a LoRA adapter under one of two
supervision targets, so that the effect of the target on citation
faithfulness can be measured at equal detection accuracy.

Two variants, trained identically except for the citation target:

    causal   target letter = argmax_k Delta(k) from a DEV audit,
             dropping samples whose max |Delta| < --min-delta (there is no
             causal signal to teach in those samples, and training on noise
             would teach the model to invent one)
    mask     target letter = gt_region from m10_localization
             (mask-based supervision, as used by forgery VLMs trained on
             manipulation masks)

The verdict target is the true label in both variants, so detection is
trained identically and any AUC difference is a side effect to be reported,
not a confound in the comparison.

Design decisions
----------------
* DEV only.  Training data comes from a `--split dev` audit.  The module
  refuses to build a dataset from rows whose split is not dev, so the TEST
  split cannot leak into training.
* Loss on the answer token alone.  Everything before it (the chat template,
  the image tokens, the prompt) is masked out with -100.  Letting the loss
  run over the prompt would mostly teach the model to reproduce the prompt
  and would shrink the difference between the two variants.
* LoRA on the language model only; the vision tower stays frozen, so the two
  variants differ purely in what the language side is taught to say.
* Examples are processed one at a time with gradient accumulation over
  `grad_accum` examples per optimizer step.  `batch_size` is recorded in the
  hyper-parameters and reserved for a batched encoder; it does not affect
  the optimisation schedule.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C
from . import m4_detectors as M4
from . import m6_metrics as M6
from . import regions as R

VARIANTS = ("causal", "mask")


# --------------------------------------------------------------------------
# dataset construction (pure python; unit-testable without torch)
# --------------------------------------------------------------------------

def build_examples(
    raw_dirs: Sequence[str],
    index_path: str,
    variant: str,
    localization_path: str = "",
    min_delta: float = 0.02,
    vocab: str = "face8",
    detector_filter: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Turn a DEV audit into (image, prompt, answer) triples.

    Each usable sample yields three examples:
      * a verdict example on the fake crop   -> answer B
      * a verdict example on the real crop   -> answer A
      * a citation example on the fake crop  -> answer = target letter
    The two verdict examples keep the detection head balanced.

    Raises RuntimeError if any row belongs to a split other than dev.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    R.set_vocab(vocab)
    names = R.get_vocab(vocab)
    letters = R.letters(len(names))
    letter_of = {n: L for n, L in zip(names, letters)}

    records, meta = C.load_index(index_path)
    by_id = {r["sample_id"]: r for r in records}

    loc: Dict[str, Dict[str, Any]] = {}
    if variant == "mask":
        if not localization_path or not os.path.exists(localization_path):
            raise FileNotFoundError(
                "variant 'mask' needs --localization loc/localization.json "
                "(run m10_localization on the DEV split first)")
        loc = {r["sample_id"]: r
               for r in C.load_json(localization_path).get("records", [])}

    groups = M6.group_raw_files(raw_dirs)
    rows: List[Dict[str, Any]] = []
    for (det, tag), paths in sorted(groups.items()):
        if detector_filter and detector_filter not in det:
            continue
        rr, _m = M6.load_rows(paths)
        rows.extend(rr)
    if not rows:
        raise FileNotFoundError(f"no raw_*.json under {list(raw_dirs)}")

    non_dev = {r.get("split") for r in rows} - {"dev"}
    if non_dev:
        raise RuntimeError(
            f"training rows contain splits {sorted(non_dev)}. CSET must be "
            f"trained on DEV only -- re-run the audit with --split dev.")

    samples = M6.samples_from_rows(rows)
    examples: List[Dict[str, Any]] = []
    stats = Counter()
    target_hist = Counter()

    for sid, s in sorted(samples.items()):
        rec = by_id.get(sid)
        if rec is None:
            stats["no_index_record"] += 1
            continue
        ks = list(s["regions"])
        if len(ks) < 2:
            stats["too_few_regions"] += 1
            continue

        if variant == "causal":
            deltas = {k: M6._f(s["regions"][k].get("delta")) for k in ks}
            deltas = {k: v for k, v in deltas.items() if np.isfinite(v)}
            if not deltas:
                stats["no_delta"] += 1
                continue
            target = max(deltas, key=deltas.get)
            if abs(deltas[target]) < min_delta:
                stats["below_min_delta"] += 1
                continue
        else:
            lrec = loc.get(sid)
            if not lrec or not lrec.get("gt_region"):
                stats["no_gt_region"] += 1
                continue
            target = lrec["gt_region"]
            if target not in ks:
                stats["gt_region_absent"] += 1
                continue

        target_hist[target] += 1
        stats["kept"] += 1
        examples.append({"image": rec["fake"], "prompt": M4.verdict_prompt(0),
                         "answer": "B", "kind": "verdict", "sample_id": sid})
        examples.append({"image": rec["real"], "prompt": M4.verdict_prompt(0),
                         "answer": "A", "kind": "verdict", "sample_id": sid})
        examples.append({"image": rec["fake"], "prompt": M4.cite_prompt(0, vocab),
                         "answer": letter_of[target], "kind": "cite",
                         "sample_id": sid, "target_region": target})

    info = {
        "variant": variant, "n_samples_kept": stats["kept"],
        "n_examples": len(examples), "drops": dict(stats),
        "target_distribution": dict(target_hist),
        "min_delta": min_delta, "vocab": vocab,
        "index": os.path.abspath(index_path),
    }
    print(f"[m13] {variant}: {stats['kept']} samples -> {len(examples)} "
          f"examples; targets {dict(target_hist)}; drops {dict(stats)}",
          flush=True)
    if stats["kept"] < 50:
        print(f"[m13] WARNING: only {stats['kept']} usable samples; check "
              f"--split dev and that the DEV audit covered all regions.",
              flush=True)
    return examples, info


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(
    raw_dirs: Sequence[str],
    index_path: str,
    out_dir: str,
    variant: str = "causal",
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    localization_path: str = "",
    min_delta: float = 0.02,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lr: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_pixels: int = 384 * 384,
    seed: int = 0,
    device: str = "cuda",
    time_budget_min: float = 90.0,
    vocab: str = "face8",
    save_every: int = 200,
    detector_filter: str = "",
) -> Dict[str, Any]:
    """
    Train one variant's LoRA adapter and write it, together with
    `cset_<variant>_log.json`, under `out_dir`.  Returns the log dict.

    `batch_size` is recorded but not used for optimisation: examples are
    encoded and back-propagated one at a time and the optimizer steps every
    `grad_accum` examples.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig

    os.makedirs(out_dir, exist_ok=True)
    budget = C.Budget(time_budget_min, label=f"m13[{variant}]")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    examples, info = build_examples(raw_dirs, index_path, variant,
                                    localization_path, min_delta, vocab,
                                    detector_filter)
    if not examples:
        raise RuntimeError("no training examples were built")
    order = rng.permutation(len(examples))
    examples = [examples[int(i)] for i in order]

    processor = AutoProcessor.from_pretrained(
        base_model, min_pixels=64 * 64, max_pixels=int(max_pixels),
        trust_remote_code=True)
    tok = processor.tokenizer

    qconf = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as MODEL
    except ImportError:
        from transformers import AutoModelForImageTextToText as MODEL
    model = MODEL.from_pretrained(base_model, quantization_config=qconf,
                                  device_map={"": 0}, trust_remote_code=True)
    try:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    except Exception as exc:
        print(f"[m13] prepare_model_for_kbit_training unavailable: {exc}")

    # LoRA on the language model only; the vision tower stays frozen.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lconf = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                       lora_dropout=lora_dropout, bias="none",
                       task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lconf)
    for n, p in model.named_parameters():
        if "visual" in n or "vision_tower" in n:
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[m13] trainable parameters: {n_train/1e6:.1f} M", flush=True)
    model.train()

    def encode(ex: Dict[str, Any]):
        """Build input_ids and labels with the loss on the answer token only."""
        import cv2

        img = C.imread(ex["image"])
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": ex["prompt"]}]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        enc = processor(text=[text], images=[pil], return_tensors="pt")
        ans_ids = tok.encode(ex["answer"], add_special_tokens=False)
        if len(ans_ids) != 1:
            ans_ids = ans_ids[:1]
        ans = torch.tensor([ans_ids], dtype=enc["input_ids"].dtype)
        input_ids = torch.cat([enc["input_ids"], ans], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[0, -1] = ans[0, 0]
        out = {k: v for k, v in enc.items() if k != "input_ids"}
        out["input_ids"] = input_ids
        out["labels"] = labels
        if "attention_mask" in out:
            out["attention_mask"] = torch.cat(
                [out["attention_mask"], torch.ones_like(ans)], dim=1)
        return out

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    # One optimizer step per `grad_accum` examples (see the loop below), so
    # the schedule length is examples * epochs / grad_accum.
    total = max(1, (len(examples) * epochs) // max(1, grad_accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=total,
                                                pct_start=0.1)
    adapter_dir = os.path.join(out_dir, variant)
    os.makedirs(adapter_dir, exist_ok=True)

    t0 = time.time()
    step, losses, stopped = 0, [], False
    sched_exhausted = False
    history: List[Dict[str, Any]] = []
    for ep in range(epochs):
        for i, ex in enumerate(examples):
            if budget.expired:
                stopped = True
                break
            try:
                batch = encode(ex)
            except Exception as exc:
                print(f"[m13] skipping {ex['sample_id']}: "
                      f"{type(exc).__name__}: {exc}")
                continue
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            losses.append(float(out.loss.item()))
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if not sched_exhausted:
                    try:
                        sched.step()
                    except ValueError as exc:
                        # The schedule is sized for the full run; it can only
                        # run out if skipped examples shift the step count.
                        # Report it once and keep the final learning rate.
                        sched_exhausted = True
                        print(f"[m13] scheduler exhausted at step {step + 1}/"
                              f"{total} ({exc}); continuing at the current "
                              f"learning rate", flush=True)
                step += 1
                if step % 20 == 0:
                    print(f"[m13] {variant} epoch {ep+1} step {step}/{total} "
                          f"loss {np.mean(losses[-200:]):.4f}  "
                          f"{budget.report()}", flush=True)
                if step % save_every == 0:
                    model.save_pretrained(adapter_dir)
            if (i + 1) % 500 == 0:
                history.append({"epoch": ep + 1, "example": i + 1,
                                "loss": float(np.mean(losses[-500:]))})
        if stopped:
            break

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    log = {
        "variant": variant, "adapter_dir": adapter_dir,
        "base_model": base_model, "dataset": info,
        "final_loss": float(np.mean(losses[-200:])) if losses else float("nan"),
        "steps": step, "stopped_early": stopped,
        "elapsed_min": (time.time() - t0) / 60.0,
        "trainable_params": n_train, "history": history,
        "hyperparams": {"lora_r": lora_r, "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout, "lr": lr,
                        "epochs": epochs, "batch_size": batch_size,
                        "grad_accum": grad_accum, "max_pixels": max_pixels,
                        "seed": seed},
        "provenance": C.provenance(),
    }
    C.save_json(os.path.join(out_dir, f"cset_{variant}_log.json"), log, indent=1)
    print(C.banner(f"Module 13 summary [{variant}]"))
    print(f"  adapter        {adapter_dir}")
    print(f"  final loss     {log['final_loss']:.4f}")
    print(f"  steps          {step}")
    print(f"  elapsed        {log['elapsed_min']:.1f} min"
          + ("  (STOPPED EARLY on budget)" if stopped else ""))
    print(f"\n  audit it with:\n    --detector "
          f"qwen25vl:{base_model}:lora={adapter_dir}")
    return log


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m13_cset")
    ap.add_argument("--raw", required=True, help="DEV audit run dirs")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="causal", choices=list(VARIANTS))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--localization", default="")
    ap.add_argument("--min-delta", type=float, default=0.02)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=384 * 384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-budget-min", type=float, default=90.0)
    ap.add_argument("--vocab", default="face8")
    ap.add_argument("--detector-filter", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the dataset, then exit (no GPU)")
    a = ap.parse_args(argv)

    if a.dry_run:
        _ex, info = build_examples(
            [d for d in a.raw.split(",") if d.strip()], a.index, a.variant,
            a.localization, a.min_delta, a.vocab, a.detector_filter)
        C.save_json(os.path.join(a.out, f"cset_{a.variant}_dataset.json"),
                    info, indent=1)
        return 0

    train([d for d in a.raw.split(",") if d.strip()], a.index, a.out,
          variant=a.variant, base_model=a.base_model,
          localization_path=a.localization, min_delta=a.min_delta,
          lora_r=a.lora_r, lora_alpha=a.lora_alpha, lr=a.lr, epochs=a.epochs,
          batch_size=a.batch_size, grad_accum=a.grad_accum,
          max_pixels=a.max_pixels, seed=a.seed, device=a.device,
          time_budget_min=a.time_budget_min, vocab=a.vocab,
          detector_filter=a.detector_filter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
