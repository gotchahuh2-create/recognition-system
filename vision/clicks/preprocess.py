"""Prepare image crops, for both building the gallery and searching it.

The same function prepares catalog images, their variations and query
crops. Any difference between how gallery and query images are prepared
quietly lowers accuracy, so the steps have a version number
(``PREPROCESS_VERSION``) that is stored with every descriptor.

    1. crop tightly to the board (catalog images: trim the white margin),
    2. pad to a square with a neutral colour. Never stretch, because a
       Click's shape and its S/M/L size are useful clues,
    3. optionally apply CLAHE to the lightness channel (evens out glare),
    4. resize to the model's input size, normalise with ImageNet mean/std,
       CHW float32.

Uses only NumPy/OpenCV, so the app doesn't need PyTorch or the Hugging Face processor.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PAD_COLOR = (124, 116, 104)  # BGR of the ImageNet mean, so it becomes ~0 after normalising
PREPROCESS_REV = 1  # increase this whenever any step below changes


@dataclass(frozen=True)
class PrepConfig:
    size: int = 224  # model input size (a multiple of 14 for DINOv2)
    clahe: bool = False

    @property
    def version(self) -> str:
        return f"pp{PREPROCESS_REV}-padsq-{self.size}" + ("-clahe" if self.clahe else "")


def trim_margin(img: np.ndarray, thresh: int = 235, pad: int = 2) -> np.ndarray:
    """Cut the white or very light studio background off a catalog image.

    Thresholds the image and keeps the bounding box of everything that
    isn't background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = (gray < thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pts = cv2.findNonZero(mask)
    if pts is None:
        return img
    x, y, w, h = cv2.boundingRect(pts)
    if w * h < 0.05 * gray.size:  # almost empty: keep the original
        return img
    H, W = gray.shape[:2]
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    return img[y0:y1, x0:x1]


def expand_box(box, frac: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Grow box (x, y, bw, bh) by ``frac`` on every side, clipped to the image."""
    x, y, bw, bh = (float(v) for v in box)
    dx, dy = bw * frac, bh * frac
    x0, y0 = max(int(round(x - dx)), 0), max(int(round(y - dy)), 0)
    x1, y1 = min(int(round(x + bw + dx)), w), min(int(round(y + bh + dy)), h)
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def crop_box(img: np.ndarray, box, expand: float = 0.08) -> np.ndarray:
    """Crop a Click box (x, y, w, h in image pixels), made 5-10 % bigger."""
    H, W = img.shape[:2]
    x, y, w, h = expand_box(box, expand, W, H)
    return img[y:y + h, x:x + w]


def pad_square(img: np.ndarray, color=PAD_COLOR) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=color)


def clahe_l(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def prepare(img: np.ndarray, cfg: PrepConfig) -> np.ndarray:
    """Turn a BGR crop into a float32 CHW tensor (3, size, size), normalised with ImageNet
    values.
    """
    img = pad_square(to_bgr(img))
    if cfg.clahe:
        img = clahe_l(img)
    interp = cv2.INTER_AREA if img.shape[0] > cfg.size else cv2.INTER_CUBIC
    img = cv2.resize(img, (cfg.size, cfg.size), interpolation=interp)
    rgb = img[..., ::-1].astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()


def prepare_batch(imgs, cfg: PrepConfig) -> np.ndarray:
    return np.stack([prepare(i, cfg) for i in imgs]).astype(np.float32)


# ---- Size class (S / M / L Click boards are 28.6, 42.9 or 57.15 mm long x 25.4 mm) ----
SIZE_ASPECT = {"S": 28.6 / 25.4, "M": 42.9 / 25.4, "L": 57.15 / 25.4}


def size_class(w: float, h: float) -> str | None:
    """Guess S/M/L from a tight crop's aspect ratio (works at any rotation)."""
    if min(w, h) <= 0:
        return None
    r = max(w, h) / min(w, h)
    return min(SIZE_ASPECT, key=lambda k: abs(np.log(r / SIZE_ASPECT[k])))
