"""
Build a synthetic FF++-shaped mirror so the CPU pipeline can be exercised
without the real dataset.  Produces:

  <root>/original/<tid>.mp4                 authentic clip
  <root>/Deepfakes/<tid>_<sid>.mp4          same clip, centre region altered
  <root>/Face2Face/<tid>_<sid>.mp4          same clip, mouth area altered
  <root>/Broken/<tid>_<sid>.mp4             deliberately misaligned (offset and
                                            shifted background); the pairing
                                            gate must reject it

The "faces" are crude but have the structure the geometric checks rely on: a
moving background (so the border is informative, not flat), an oval face, and
distinct eye/nose/mouth blobs.

Usage:
    python tests/make_fixture.py <out_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def _bg(t: int, h: int, w: int, rng) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 90 + 60 * np.sin((xx + 3 * t) / 23.0)
    img[..., 1] = 90 + 60 * np.sin((yy + 2 * t) / 31.0)
    img[..., 2] = 90 + 60 * np.sin((xx + yy + 5 * t) / 17.0)
    img += rng.normal(0, 9.0, img.shape)
    return img


def _face(img: np.ndarray, t: int, variant: str) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w // 2 + int(3 * np.sin(t / 9.0)), h // 2
    fw, fh = int(w * 0.22), int(h * 0.30)
    out = img.copy()
    cv2.ellipse(out, (cx, cy), (fw, fh), 0, 0, 360, (165, 178, 196), -1)
    ey = cy - fh // 3
    ex = fw // 2
    eye_col = (40, 40, 40) if variant != "swap" else (30, 60, 120)
    cv2.circle(out, (cx - ex, ey), max(3, fw // 7), eye_col, -1)
    cv2.circle(out, (cx + ex, ey), max(3, fw // 7), eye_col, -1)
    nose_col = (150, 130, 120) if variant != "swap" else (120, 150, 140)
    cv2.ellipse(out, (cx, cy), (max(3, fw // 6), max(4, fh // 6)), 0, 0, 360,
                nose_col, -1)
    my = cy + int(fh * 0.45)
    mw = fw // 2 + (4 if variant == "mouth" else 0)
    mouth_col = (60, 40, 130) if variant != "mouth" else (40, 30, 190)
    cv2.ellipse(out, (cx, my), (mw, max(3, fh // 9)), 0, 0, 360, mouth_col, -1)
    cv2.ellipse(out, (cx, cy - fh), (int(fw * 1.15), int(fh * 0.55)), 0, 180, 360,
                (50, 45, 45), -1)                                   # hair
    if variant in ("swap", "mouth"):
        # The characteristic artefact of a face swap: the synthesised area is
        # smoother than the surrounding authentic pixels.  Restricted to the
        # region the variant manipulates, so the ground truth is unambiguous.
        m = np.zeros(out.shape[:2], np.uint8)
        if variant == "swap":
            cv2.ellipse(m, (cx, cy - fh // 4), (int(fw * 0.8), int(fh * 0.5)),
                        0, 0, 360, 255, -1)
        else:
            cv2.ellipse(m, (cx, my), (int(mw * 1.6), int(fh * 0.30)),
                        0, 0, 360, 255, -1)
        sm = cv2.GaussianBlur(out, (0, 0), 2.4)
        out = np.where(m[..., None] > 0, sm, out)
    return out


def write_clip(path: str, n: int, h: int, w: int, variant: str, seed: int,
               offset: int = 0, shift_bg: bool = False, fps: int = 25) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cannot open writer for {path}")
    for i in range(n):
        t = i + offset
        frame = _bg(t + (7 if shift_bg else 0), h, w, rng)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = _face(frame, t, variant)
        vw.write(frame)
    vw.release()


def build(root: str, n_ids: int = 4, n_frames: int = 60, h: int = 240,
          w: int = 320) -> str:
    os.makedirs(root, exist_ok=True)
    ids = [f"{i:03d}" for i in range(n_ids)]
    for k, tid in enumerate(ids):
        write_clip(os.path.join(root, "original", f"{tid}.mp4"),
                   n_frames, h, w, "real", seed=100 + k)
    for k, tid in enumerate(ids):
        sid = ids[(k + 1) % len(ids)]
        write_clip(os.path.join(root, "Deepfakes", f"{tid}_{sid}.mp4"),
                   n_frames, h, w, "swap", seed=100 + k)
        write_clip(os.path.join(root, "Face2Face", f"{tid}_{sid}.mp4"),
                   n_frames, h, w, "mouth", seed=100 + k)
    # One deliberately broken method: different background phase and a
    # frame offset.
    write_clip(os.path.join(root, "Broken", f"{ids[0]}_{ids[1]}.mp4"),
               n_frames, h, w, "swap", seed=100, offset=17, shift_bg=True)
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python tests/make_fixture.py",
        description="Build a synthetic FF++-shaped fixture directory.")
    ap.add_argument("out", help="output directory for the fixture")
    ap.add_argument("--n-ids", type=int, default=4)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    a = ap.parse_args(argv)
    print(build(a.out, a.n_ids, a.n_frames, a.height, a.width))
    return 0


if __name__ == "__main__":
    sys.exit(main())
