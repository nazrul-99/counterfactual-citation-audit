"""
ccaudit.m9_train_cnn -- Module 9, the CNN baseline.

Trains a conventional timm classifier on the DEV split of the same face crops
the VLMs are audited on.  This provides a detector with high detection AUC and
a saliency-based explanation (via m4.CNNGradCAMDetector) to set against the
language-based citations of the VLMs.

Design decisions
----------------
* DEV only.  Training never sees a TEST clip.  The split is a hash of the
  identity pair, so this holds without any bookkeeping.
* Clip-disjoint validation.  The train/val split inside DEV is by pair_id,
  not by sample.  Two frames of one clip are near-duplicates; splitting by
  sample would put a clip in both halves and inflate validation AUC.
* Balanced by construction.  Every sample contributes its real crop (label 0)
  and its fake crop (label 1), so the classes are exactly balanced and AUC is
  interpretable without reweighting.
* Time budget.  Training checks the wall clock between batches and stops
  cleanly, saving the best checkpoint so far.

Checkpoint format (consumed by m4.CNNGradCAMDetector):

    {"arch": str, "input_size": int, "state_dict": ..., "provenance": {...},
     "val_auc": float, "epochs_done": int}

External weights can be used by saving them in this same shape.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C
from . import regions as R

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_items(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, int, str]]:
    """(path, label, pair_id) for every crop.  Real = 0, fake = 1."""
    items: List[Tuple[str, int, str]] = []
    for r in records:
        items.append((r["real"], 0, r["pair_id"]))
        items.append((r["fake"], 1, r["pair_id"]))
    return items


def clip_disjoint_split(items: Sequence[Tuple[str, int, str]],
                        val_frac: float = 0.2, seed: int = 0
                        ) -> Tuple[List, List]:
    """Split by pair_id so no clip appears in both halves."""
    import hashlib

    clips = sorted({p for _f, _l, p in items})
    val_clips = set()
    for c in clips:
        h = int(hashlib.blake2b(f"{seed}|cnnval|{c}".encode(),
                                digest_size=8).hexdigest(), 16)
        if (h % 10_000) < int(val_frac * 10_000):
            val_clips.add(c)
    train = [it for it in items if it[2] not in val_clips]
    val = [it for it in items if it[2] in val_clips]
    return train, val


class CropDataset:
    """Minimal torch Dataset.  Torch is imported lazily so the data logic can
    be imported and unit-tested without torch installed."""

    def __init__(self, items: Sequence[Tuple[str, int, str]], input_size: int,
                 train: bool, seed: int = 0):
        self.items = list(items)
        self.input_size = int(input_size)
        self.train = bool(train)
        self.seed = int(seed)
        self._rng: Optional[np.random.Generator] = None
        self._rng_pid: int = -1

    def __len__(self) -> int:
        return len(self.items)

    def _get_rng(self) -> np.random.Generator:
        """
        Per-process augmentation generator.  A DataLoader worker receives a
        copy of this object, so a single generator created in the constructor
        would give every worker the same augmentation stream.  The generator
        is therefore created lazily in the process that uses it and seeded
        from the base seed and the worker seed (which torch draws afresh for
        every iterator), so streams differ across workers and across epochs.
        """
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            salt = 0
            try:
                import torch
                info = torch.utils.data.get_worker_info()
                if info is not None:
                    salt = int(info.seed) % (2 ** 32)
            except Exception:
                salt = 0
            self._rng = np.random.default_rng([self.seed, salt])
            self._rng_pid = pid
        return self._rng

    def _augment(self, img: np.ndarray) -> np.ndarray:
        rng = self._get_rng()
        if rng.random() < 0.5:
            img = img[:, ::-1]
        # Mild photometric jitter only; stronger augmentation could erase the
        # forgery artefacts the classifier is meant to learn.
        if rng.random() < 0.3:
            img = np.clip(img.astype(np.float32)
                          * rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        if rng.random() < 0.2:
            q = int(rng.integers(60, 95))
            img = C.jpeg_roundtrip(img, q)
        return np.ascontiguousarray(img)

    def __getitem__(self, i: int):
        import cv2
        import torch

        path, label, _pair = self.items[i]
        img = C.imread(path)
        if self.train:
            img = self._augment(img)
        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            interp = (cv2.INTER_AREA if img.shape[0] > self.input_size
                      else cv2.INTER_LINEAR)
            img = cv2.resize(img, (self.input_size, self.input_size),
                             interpolation=interp)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), int(label)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(
    index_path: str,
    out_dir: str,
    arch: str = "efficientnet_b0",
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    input_size: int = 256,
    val_frac: float = 0.2,
    workers: int = 2,
    seed: int = 0,
    device: str = "cuda",
    amp: bool = True,
    train_budget_min: float = 60.0,
    pretrained: bool = True,
    resume: str = "",
    max_samples: int = 0,
) -> Dict[str, Any]:
    """
    Train the classifier on the DEV split and write the best checkpoint and
    `train_log.json` to `out_dir`.  Returns the log dict.
    """
    import timm
    import torch
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)
    budget = C.Budget(train_budget_min, label="m9")
    torch.manual_seed(seed)
    np.random.seed(seed)

    records, meta = C.load_index(index_path)
    R.set_vocab(meta.get("vocab", "face8"))
    dev = C.filter_split(records, "dev")
    if not dev:
        raise RuntimeError(
            "no DEV samples in the index. The CNN must be trained on DEV only; "
            "check that the index carries pair_ids the split rule understands.")
    if max_samples:
        dev = C.limit_samples(dev, max_samples, seed)

    items = build_items(dev)
    tr_items, va_items = clip_disjoint_split(items, val_frac, seed)
    n_tr_clips = len({p for _f, _l, p in tr_items})
    n_va_clips = len({p for _f, _l, p in va_items})
    print(f"[m9] DEV samples {len(dev)} -> {len(items)} crops; "
          f"train {len(tr_items)} ({n_tr_clips} clips) / "
          f"val {len(va_items)} ({n_va_clips} clips)", flush=True)
    if not va_items:
        raise RuntimeError("validation split is empty; lower --val-frac or "
                           "check that DEV has more than one clip")

    model = timm.create_model(arch, pretrained=pretrained, num_classes=2)
    start_epoch, best_auc = 0, -1.0
    if resume and os.path.exists(resume):
        blob = torch.load(resume, map_location="cpu", weights_only=False)
        if blob.get("arch") == arch:
            model.load_state_dict(blob["state_dict"])
            start_epoch = int(blob.get("epochs_done", 0))
            best_auc = float(blob.get("val_auc", -1.0))
            print(f"[m9] resumed from {resume}: {start_epoch} epochs, "
                  f"val AUC {best_auc:.4f}", flush=True)
        else:
            print(f"[m9] checkpoint arch {blob.get('arch')!r} != {arch!r}; "
                  f"ignoring", flush=True)
    model.to(device)

    tr_loader = DataLoader(
        CropDataset(tr_items, input_size, True, seed), batch_size=batch_size,
        shuffle=True, num_workers=workers, pin_memory=device.startswith("cuda"),
        drop_last=len(tr_items) > batch_size)
    va_loader = DataLoader(
        CropDataset(va_items, input_size, False, seed), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=device.startswith("cuda"))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, len(tr_loader)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.25)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = bool(amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(out_dir, f"cnn_{C.safe_name(arch)}.pt")
    history: List[Dict[str, Any]] = []
    stopped_early = False

    def evaluate() -> Tuple[float, float]:
        model.eval()
        scores, labels, losses = [], [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    losses.append(float(crit(logits, yb).item()))
                p = torch.softmax(logits.float(), dim=-1)[:, 1]
                scores.extend(p.cpu().numpy().tolist())
                labels.extend(yb.cpu().numpy().tolist())
        return C.auc_score(scores, labels), float(np.mean(losses or [np.nan]))

    t0 = time.time()
    for ep in range(start_epoch, epochs):
        model.train()
        run_loss, n_seen = 0.0, 0
        for step, (xb, yb) in enumerate(tr_loader):
            if budget.expired:
                stopped_early = True
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            try:
                sched.step()
            except ValueError:
                pass                      # total_steps exhausted after a resume
            run_loss += float(loss.item()) * xb.size(0)
            n_seen += int(xb.size(0))
            if step % 50 == 0:
                print(f"[m9] epoch {ep+1}/{epochs} step {step}/{len(tr_loader)} "
                      f"loss {run_loss/max(1,n_seen):.4f}  {budget.report()}",
                      flush=True)

        val_auc, val_loss = evaluate()
        history.append({"epoch": ep + 1, "train_loss": run_loss / max(1, n_seen),
                        "val_auc": val_auc, "val_loss": val_loss,
                        "elapsed_min": (time.time() - t0) / 60.0})
        print(f"[m9] epoch {ep+1}: train {run_loss/max(1,n_seen):.4f} "
              f"val_loss {val_loss:.4f} val AUC {val_auc:.4f}"
              + ("  * best" if val_auc > best_auc else ""), flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "arch": arch, "input_size": input_size,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_auc": best_auc, "epochs_done": ep + 1,
                "provenance": C.provenance({
                    "index": os.path.abspath(index_path), "split": "dev",
                    "epochs": epochs, "lr": lr, "batch_size": batch_size,
                    "input_size": input_size, "val_frac": val_frac,
                    "seed": seed, "pretrained": pretrained,
                    "n_train_crops": len(tr_items), "n_val_crops": len(va_items),
                    "n_train_clips": n_tr_clips, "n_val_clips": n_va_clips,
                }),
            }, ckpt_path)
            print(f"[m9] checkpoint -> {ckpt_path}", flush=True)
        if stopped_early:
            print("[m9] STOPPED EARLY on time budget", flush=True)
            break

    log = {
        "arch": arch, "best_val_auc": best_auc, "history": history,
        "checkpoint": ckpt_path, "stopped_early": stopped_early,
        "n_dev_samples": len(dev), "elapsed_min": (time.time() - t0) / 60.0,
        "provenance": C.provenance(),
    }
    C.save_json(os.path.join(out_dir, "train_log.json"), log, indent=1)

    print(C.banner("Module 9 summary"))
    print(f"  best val AUC   {best_auc:.4f}"
          + ("   (below 0.9: consider more epochs or --arch legacy_xception "
             "before running the downstream GPU stages)"
             if best_auc < 0.9 else ""))
    print(f"  checkpoint     {ckpt_path}")
    print(f"  elapsed        {log['elapsed_min']:.1f} min")
    return log


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccaudit.m9_train_cnn")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--train-budget-min", type=float, default=60.0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-samples", type=int, default=0)
    a = ap.parse_args(argv)

    train(a.index, a.out, arch=a.arch, epochs=a.epochs,
          batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
          input_size=a.input_size, val_frac=a.val_frac, workers=a.workers,
          seed=a.seed, device=a.device, amp=not a.no_amp,
          train_budget_min=a.train_budget_min,
          pretrained=not a.no_pretrained, resume=a.resume,
          max_samples=a.max_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
