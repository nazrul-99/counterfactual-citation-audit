"""
A test double for mediapipe's FaceLandmarker.

This is not a fallback parser and is never importable from the pipeline: it
lives in tests/ and has to be installed explicitly.  Its only purpose is to
let the CPU pipeline be exercised end to end in an environment where the
face_landmarker.task file cannot be downloaded.

It finds the synthetic fixture's skin-coloured ellipse and emits a 478-point
array in which exactly the indices the region builder uses (FACE_OVAL, the two
eye/brow sets, LIPS_OUTER, NOSE_POINTS and the level landmarks) sit where they
should; the remaining indices are filled inside the oval so the bounding box,
and therefore the crop box, behave like the real landmarker's output.
"""
from __future__ import annotations

import numpy as np

from ccaudit import regions as R


def _ellipse_points(cx, cy, rx, ry, n, start_deg=-90.0, sweep_deg=360.0):
    ang = np.deg2rad(start_deg + np.linspace(0.0, sweep_deg, n, endpoint=False))
    return np.stack([cx + rx * np.cos(ang), cy + ry * np.sin(ang)], axis=1)


def detect_face_ellipse(bgr: np.ndarray):
    """Locate the fixture's skin ellipse: (cx, cy, rx, ry) or None."""
    import cv2

    b, g, r = bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    skin = ((r > 140) & (r < 235) & (g > 120) & (g < 215) & (b > 110) & (b < 205)
            & (r >= g) & (g >= b) & ((r - b) > 12) & ((r - b) < 70))
    m = (skin.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0)


def synth_landmarks(cx, cy, rx, ry) -> np.ndarray:
    """Synthesise a 478-point landmark array for a face oval centred at
    (cx, cy) with radii (rx, ry)."""
    pts = np.zeros((478, 2), np.float32)
    # Every index defaults to a ring well inside the oval so the bounding box
    # is the oval itself.
    filler = _ellipse_points(cx, cy, rx * 0.55, ry * 0.55, 478)
    pts[:] = filler

    oval = _ellipse_points(cx, cy, rx, ry, len(R.FACE_OVAL))
    pts[R.FACE_OVAL] = oval

    eye_dx, eye_dy = rx * 0.45, ry * 0.32
    er_x, er_y = rx * 0.20, ry * 0.12
    pts[R.EYE_A] = _ellipse_points(cx - eye_dx, cy - eye_dy, er_x, er_y, len(R.EYE_A))
    pts[R.EYE_B] = _ellipse_points(cx + eye_dx, cy - eye_dy, er_x, er_y, len(R.EYE_B))
    pts[R.BROW_A] = _ellipse_points(cx - eye_dx, cy - eye_dy - ry * 0.16,
                                    er_x * 1.1, er_y * 0.4, len(R.BROW_A))
    pts[R.BROW_B] = _ellipse_points(cx + eye_dx, cy - eye_dy - ry * 0.16,
                                    er_x * 1.1, er_y * 0.4, len(R.BROW_B))
    pts[R.LIPS_OUTER] = _ellipse_points(cx, cy + ry * 0.45, rx * 0.42, ry * 0.13,
                                        len(R.LIPS_OUTER))
    pts[R.NOSE_POINTS] = _ellipse_points(cx, cy + ry * 0.02, rx * 0.20, ry * 0.22,
                                         len(R.NOSE_POINTS))
    # Level landmarks used for the hair band and the ear boxes.
    pts[33] = [cx - eye_dx - er_x, cy - eye_dy]
    pts[263] = [cx + eye_dx + er_x, cy - eye_dy]
    pts[61] = [cx - rx * 0.42, cy + ry * 0.45]
    pts[291] = [cx + rx * 0.42, cy + ry * 0.45]
    return pts


class FakeFaceParser:
    def __init__(self, model_path: str = "", min_conf: float = 0.4):
        print("[m2] TEST DOUBLE parser active (not mediapipe)", flush=True)

    def landmarks(self, bgr: np.ndarray):
        e = detect_face_ellipse(bgr)
        if e is None:
            return None
        return synth_landmarks(*e)

    def close(self):
        pass


def install():
    """Monkeypatch m2_parse so worker processes use the double."""
    from ccaudit import m2_parse

    m2_parse.FaceParser = FakeFaceParser
    m2_parse.locate_landmarker = lambda explicit="": "TEST_DOUBLE"
    return True
