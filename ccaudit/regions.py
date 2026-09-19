"""
ccaudit.regions -- the closed region vocabulary.

A *vocabulary* is an ordered list of region names.  The parser (m2) writes a
single-channel uint8 label map whose values are region ids 0..K-1, with 255
reserved for background.  Every other module reads the vocabulary through this
module and never hard-codes region names.

Two vocabularies ship:

  face8   the eight facial parts used throughout the audit (default)
  grid9   a 3x3 spatial grid, used for the non-face second domain to show
          that the instrument does not depend on facial structure

Naming convention for the eyes
------------------------------
`left_eye` is the eye that appears on the LEFT HALF OF THE IMAGE, i.e. the
subject's anatomical *right* eye.  This is deliberate: a VLM asked to describe
an image says "the left eye" meaning the one it sees on the left.  The parser
assigns the two eye hulls by comparing their x-centroids at runtime, so this
holds regardless of which mediapipe index set is which.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

BACKGROUND = 255

# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------

FACE8: List[str] = [
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
    "skin",
    "jaw_boundary",
    "hair",
    "ears",
]

GRID9: List[str] = [f"r{r}c{c}" for r in range(3) for c in range(3)]

_VOCABS: Dict[str, List[str]] = {"face8": FACE8, "grid9": GRID9}

# Coarse granularity maps.  Every fine region must appear exactly once as a
# key.
_COARSE: Dict[str, Dict[str, str]] = {
    "face8": {
        "left_eye": "eyes",
        "right_eye": "eyes",
        "nose": "nose",
        "mouth": "mouth",
        "skin": "rest",
        "jaw_boundary": "rest",
        "hair": "rest",
        "ears": "rest",
    },
    "grid9": {
        "r0c0": "top", "r0c1": "top", "r0c2": "top",
        "r1c0": "middle", "r1c1": "middle", "r1c2": "middle",
        "r2c0": "bottom", "r2c1": "bottom", "r2c2": "bottom",
    },
}

# Human-readable names used inside prompts.  They are kept short and
# unambiguous because they are the only part of the vocabulary the model sees.
_PROMPT_NAMES: Dict[str, Dict[str, str]] = {
    "face8": {
        "left_eye": "left eye",
        "right_eye": "right eye",
        "nose": "nose",
        "mouth": "mouth",
        "skin": "cheeks and forehead skin",
        "jaw_boundary": "jawline and face outline",
        "hair": "hair",
        "ears": "ears",
    },
    "grid9": {n: n.replace("r", "row ").replace("c", ", column ") for n in GRID9},
}

# The active vocabulary is process-global.  Every CLI entry point calls
# set_vocab() once from its --vocab flag before doing anything else; the value
# is also written into every output JSON so downstream modules can check it.
_ACTIVE = "face8"


def set_vocab(name: str) -> None:
    if name not in _VOCABS:
        raise ValueError(f"unknown vocabulary {name!r}; have {sorted(_VOCABS)}")
    global _ACTIVE
    _ACTIVE = name


def active_vocab() -> str:
    return _ACTIVE


def get_vocab(name: str | None = None) -> List[str]:
    return list(_VOCABS[name or _ACTIVE])


def num_regions(name: str | None = None) -> int:
    return len(_VOCABS[name or _ACTIVE])


def rid(region: str, name: str | None = None) -> int:
    return _VOCABS[name or _ACTIVE].index(region)


def region_of(region_id: int, name: str | None = None) -> str:
    return _VOCABS[name or _ACTIVE][region_id]


def coarse_map(name: str | None = None) -> Dict[str, str]:
    return dict(_COARSE[name or _ACTIVE])


def coarse_vocab(name: str | None = None) -> List[str]:
    """Coarse region names in a stable order (order of first appearance)."""
    seen: List[str] = []
    for fine in get_vocab(name):
        c = _COARSE[name or _ACTIVE][fine]
        if c not in seen:
            seen.append(c)
    return seen


def prompt_name(region: str, name: str | None = None) -> str:
    return _PROMPT_NAMES[name or _ACTIVE][region]


def letters(n: int) -> List[str]:
    """Option letters A, B, C, ... used in the citation menu."""
    if n > 26:
        raise ValueError("vocabulary too large for single-letter options")
    return [chr(ord("A") + i) for i in range(n)]


# --------------------------------------------------------------------------
# mediapipe FaceMesh landmark index sets (468/478-point canonical model)
# --------------------------------------------------------------------------
# Ordered ring, usable directly as a polygon.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

# mediapipe's own naming is subject-anatomical and is not relied upon: the two
# eye hulls are assigned to left_eye/right_eye by x-centroid at runtime.
EYE_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_B = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROW_A = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_B = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# Outer lip ring, ordered.
LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185,
]

# Nose: dorsum, tip and alae.  Used as a point cloud -> convex hull.
NOSE_POINTS = [
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2,          # bridge down to sub-nasal
    98, 97, 326, 327,                               # nostril floor
    129, 358, 49, 279, 64, 294,                     # alae / wings
    115, 344, 220, 440, 45, 275, 141, 370,          # sides of the dorsum
]

# Landmarks whose y-level brackets the ears (eye corner .. mouth corner).
EAR_LEVEL_TOP = [33, 263]
EAR_LEVEL_BOTTOM = [61, 291]


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of an (N,2) int32 point array, returned as an (M,1,2) contour."""
    import cv2

    if len(pts) < 3:
        return np.zeros((0, 1, 2), np.int32)
    return cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))


def build_face8_labels(
    pts: np.ndarray,
    shape: Sequence[int],
    jaw_ring_frac: float = 0.045,
    hair_up_frac: float = 0.55,
    ear_out_frac: float = 0.13,
) -> np.ndarray:
    """
    Paint a face8 label map from mediapipe landmark pixel coordinates.

    Parameters
    ----------
    pts   : (L, 2) float/int array of landmark pixel coordinates in the frame
            whose size is `shape`.  L must be >= 468.
    shape : (H, W) of the output label map.
    jaw_ring_frac : width of the jaw/outline ring as a fraction of face height.
    hair_up_frac  : how far above the forehead arc the hair band extends,
            as a fraction of face height.
    ear_out_frac  : how far outside the face oval the ear boxes extend,
            as a fraction of face width.

    Returns
    -------
    uint8 (H, W) label map, values in 0..7 and 255 for background.

    Regions are painted in increasing priority so the result is disjoint by
    construction:  hair, ears, skin, jaw_boundary, nose, eyes, mouth.
    """
    import cv2

    if pts.shape[0] < 468:
        raise ValueError(f"expected >=468 landmarks, got {pts.shape[0]}")
    H, W = int(shape[0]), int(shape[1])
    lab = np.full((H, W), BACKGROUND, np.uint8)
    P = np.asarray(pts, np.float32)

    oval = P[FACE_OVAL].astype(np.int32)
    face_w = float(oval[:, 0].max() - oval[:, 0].min())
    face_h = float(oval[:, 1].max() - oval[:, 1].min())
    if face_w < 8 or face_h < 8:
        raise ValueError("degenerate face geometry")

    oval_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(oval_mask, [oval.reshape(-1, 1, 2)], 1)

    # ---- hair: a band above the forehead arc ------------------------------
    # The forehead arc is the part of the oval above the eye line.  The oval
    # points above that line are extended outward and upward and the polygon
    # is closed.  Any part of it that falls inside the oval is overwritten by
    # skin later, so only the part above the head survives.
    eye_y = float(P[[33, 263], 1].mean())
    top_pts = oval[oval[:, 1] <= eye_y]
    if len(top_pts) >= 3:
        cx = float(oval[:, 0].mean())
        up = max(4.0, hair_up_frac * face_h)
        out = ear_out_frac * face_w
        order = np.argsort(top_pts[:, 0])
        arc = top_pts[order].astype(np.float32)
        # Widen the arc horizontally away from the centre, then lift it.
        widened = arc.copy()
        widened[:, 0] += np.sign(widened[:, 0] - cx) * out
        widened[:, 1] -= up
        poly = np.concatenate([arc, widened[::-1]], axis=0).astype(np.int32)
        cv2.fillPoly(lab, [poly.reshape(-1, 1, 2)], int(rid("hair", "face8")))

    # ---- ears: boxes flanking the oval between eye and mouth level --------
    y0 = float(P[EAR_LEVEL_TOP, 1].mean())
    y1 = float(P[EAR_LEVEL_BOTTOM, 1].mean())
    if y1 > y0:
        band = oval[(oval[:, 1] >= y0) & (oval[:, 1] <= y1)]
        if len(band) >= 2:
            out = max(3.0, ear_out_frac * face_w)
            xl = float(band[:, 0].min())
            xr = float(band[:, 0].max())
            eid = int(rid("ears", "face8"))
            cv2.rectangle(lab, (int(xl - out), int(y0)), (int(xl), int(y1)), eid, -1)
            cv2.rectangle(lab, (int(xr), int(y0)), (int(xr + out), int(y1)), eid, -1)

    # ---- skin: everything inside the oval ---------------------------------
    lab[oval_mask > 0] = int(rid("skin", "face8"))

    # ---- jaw_boundary: inner ring of the oval -----------------------------
    ring_px = max(2, int(round(jaw_ring_frac * face_h)))
    k = 2 * ring_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(oval_mask, kernel, iterations=1)
    ring = (oval_mask > 0) & (inner == 0)
    lab[ring] = int(rid("jaw_boundary", "face8"))

    # ---- nose -------------------------------------------------------------
    nose_hull = _hull(P[NOSE_POINTS].astype(np.int32))
    if len(nose_hull):
        cv2.fillPoly(lab, [nose_hull], int(rid("nose", "face8")))

    # ---- eyes (contour + brow, assigned by x-centroid) --------------------
    hull_a = _hull(P[EYE_A + BROW_A].astype(np.int32))
    hull_b = _hull(P[EYE_B + BROW_B].astype(np.int32))
    cx_a = float(P[EYE_A, 0].mean())
    cx_b = float(P[EYE_B, 0].mean())
    if cx_a <= cx_b:
        left_hull, right_hull = hull_a, hull_b
    else:
        left_hull, right_hull = hull_b, hull_a
    if len(left_hull):
        cv2.fillPoly(lab, [left_hull], int(rid("left_eye", "face8")))
    if len(right_hull):
        cv2.fillPoly(lab, [right_hull], int(rid("right_eye", "face8")))

    # ---- mouth ------------------------------------------------------------
    lips = P[LIPS_OUTER].astype(np.int32).reshape(-1, 1, 2)
    mouth_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mouth_mask, [lips], 1)
    # Dilate slightly so that the vermilion border falls inside the region.
    mk = max(3, int(round(0.012 * face_h)) * 2 + 1)
    mouth_mask = cv2.dilate(
        mouth_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk)), 1
    )
    lab[mouth_mask > 0] = int(rid("mouth", "face8"))

    return lab


def build_grid9_labels(shape: Sequence[int]) -> np.ndarray:
    """3x3 grid label map covering the whole image (no background)."""
    H, W = int(shape[0]), int(shape[1])
    lab = np.zeros((H, W), np.uint8)
    ys = [0, H // 3, 2 * H // 3, H]
    xs = [0, W // 3, 2 * W // 3, W]
    for r in range(3):
        for c in range(3):
            lab[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = rid(f"r{r}c{c}", "grid9")
    return lab


def region_areas(lab: np.ndarray, name: str | None = None) -> Dict[str, int]:
    """Pixel count per region in a label map."""
    vocab = get_vocab(name)
    counts = np.bincount(lab.reshape(-1), minlength=256)
    return {r: int(counts[i]) for i, r in enumerate(vocab)}


def present_regions(lab: np.ndarray, min_px: int, name: str | None = None) -> List[str]:
    """Regions with at least `min_px` pixels, in vocabulary order."""
    areas = region_areas(lab, name)
    return [r for r in get_vocab(name) if areas[r] >= min_px]
