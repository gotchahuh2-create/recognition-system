"""Image clean-up for small or blurry text, to help OCR read it.

Simple OpenCV building blocks:

* Enlarge before OCR, either with bicubic scaling or a super-resolution
  network (ESPCN / FSRCNN / EDSR via ``cv2.dnn_superres``; this needs
  ``opencv-contrib-python-headless`` and the model files from
  ``scripts/get_sr_models.py``). Tesseract treats letters smaller than
  about 20 px as noise, so we scale crops until the letters are about
  ``TARGET_GLYPH_PX`` tall.
* Denoise with non-local means, with the strength based on how noisy the
  image is.
* Deblur with an unsharp mask, plus a few Richardson-Lucy steps for
  out-of-focus images.
* Contrast: stretch the levels, apply CLAHE, and brighten dark photos.
* Threshold to black and white: Otsu (one threshold for the whole image)
  and Sauvola (a local threshold that handles uneven glare).

All functions take single-channel uint8 images unless noted otherwise.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(os.environ.get("VISION_MODELS",
                                 Path(__file__).resolve().parent.parent / "models"))

# Letter height (px) we aim for. Tesseract works best with capital letters
# around 30-50 px tall. Neural recognisers resize lines to 32/48 px anyway.
TARGET_GLYPH_PX = 44
MAX_UPSCALE = 6.0  # enlarge at most 6x


# ---- Measurements ----
def noise_sigma(gray: np.ndarray) -> float:
    """Estimate how noisy the image is (Immerkaer's method). Returns the noise standard
    deviation.
    """
    g = gray.astype(np.float64)
    if min(g.shape) < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    conv = cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REFLECT)
    h, w = g.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def text_is_light(gray: np.ndarray) -> bool:
    """Is the text lighter than the background?

    Printed silkscreen and laser markings usually are. We split the image
    with Otsu's threshold and assume the smaller part is the text.
    """
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (b > 0).mean() < 0.5


# ---- Brightness and contrast ----
def stretch(gray: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Stretch contrast using percentiles, so a few glare pixels don't throw it off."""
    a, b = np.percentile(gray, (lo, hi))
    if b - a < 1:
        return gray
    return np.clip((gray.astype(np.float32) - a) * (255.0 / (b - a)), 0, 255).astype(np.uint8)


def gamma_lift(img: np.ndarray, target_mean: float = 110.0, max_gamma: float = 3.0
               ) -> tuple[np.ndarray, float]:
    """Brighten a dark photo with a gamma curve so its average ends up near target_mean.

    Works on colour or gray images. Returns (image, gamma); a gamma of 1.0
    means nothing changed.
    """
    v = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    m = float(v.mean())
    if m <= 1 or m >= target_mean:
        return img, 1.0
    # rough estimate: mean(x^(1/g)) is about mean(x)^(1/g)
    gamma = min(max_gamma, np.log(m / 255.0) / np.log(target_mean / 255.0))
    lut = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(img, lut), float(gamma)


def denoise(img: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Remove noise with non-local means, using a strength that matches the measured noise."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = noise_sigma(gray) if sigma is None else sigma
    if s < 2.0:
        return img
    h = float(np.clip(1.1 * s, 3, 25))
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h, 7, 15)
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 15)


def unsharp(gray: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def richardson_lucy(gray: np.ndarray, sigma: float = 1.6, iters: int = 8) -> np.ndarray:
    """Sharpen with Richardson-Lucy deconvolution, assuming a Gaussian blur.

    A cheap stand-in for a learned deblurring model. Only use it on small
    crops: each iteration costs about two blurs.
    """
    img = gray.astype(np.float32) / 255.0 + 1e-3
    est = img.copy()
    for _ in range(iters):
        conv = cv2.GaussianBlur(est, (0, 0), sigma) + 1e-6
        est *= cv2.GaussianBlur(img / conv, (0, 0), sigma)
    return np.clip(est * 255.0, 0, 255).astype(np.uint8)


# ---- Black-and-white thresholds ----
def otsu(gray: np.ndarray, dark_text: bool = True) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if dark_text and (b > 0).mean() < 0.5:
        b = 255 - b
    return b


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.25, R: float = 128.0,
            dark_text: bool = True) -> np.ndarray:
    """Sauvola local threshold, built from box filters (so scikit-image isn't needed).

    Expects dark text on a light background when ``dark_text`` is True.
    """
    window |= 1
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1 + k * (std / R - 1))
    b = np.where(g > thr, 255, 0).astype(np.uint8)
    if dark_text and (b > 0).mean() < 0.5:
        b = 255 - b
    return b


# ---- Enlarging / super-resolution ----
SR_MODELS = {  # name -> (file name, scales it supports)
    "espcn": ("ESPCN", (2, 3, 4)),
    "fsrcnn": ("FSRCNN", (2, 3, 4)),
    "edsr": ("EDSR", (2, 3, 4)),
    "lapsrn": ("LapSRN", (2, 4, 8)),
}


def available_upscalers() -> list[str]:
    out = ["bicubic", "lanczos"]
    if not hasattr(cv2, "dnn_superres"):
        return out
    for name, (stem, scales) in SR_MODELS.items():
        if any((MODELS_DIR / f"{stem}_x{s}.pb").is_file() for s in scales):
            out.append(name)
    return out


@lru_cache(maxsize=8)
def _sr_engine(name: str, scale: int):
    stem, _ = SR_MODELS[name]
    path = MODELS_DIR / f"{stem}_x{scale}.pb"
    if not path.is_file() or not hasattr(cv2, "dnn_superres"):
        return None
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(path))
    sr.setModel(name, scale)
    return sr


def upscale(img: np.ndarray, factor: float, method: str = "bicubic") -> np.ndarray:
    """Enlarge the image by ``factor``.

    Super-resolution models run at their closest built-in scale and are then
    resized to the exact factor. If a model is unknown or missing we use
    bicubic instead, so the pipeline never breaks.
    """
    if factor <= 1.01:
        return img
    if method in SR_MODELS:
        _, scales = SR_MODELS[method]
        s = min(scales, key=lambda x: (abs(x - factor), -x))
        sr = _sr_engine(method, s)
        if sr is not None:
            src = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out = sr.upsample(src)
            if img.ndim == 2:
                out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            target = (round(w * factor), round(h * factor))
            if out.shape[1] != target[0]:
                out = cv2.resize(out, target, interpolation=cv2.INTER_CUBIC)
            return out
        method = "bicubic"
    interp = cv2.INTER_LANCZOS4 if method == "lanczos" else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def scale_for_glyph(glyph_px: float, target: float = TARGET_GLYPH_PX,
                    max_up: float = MAX_UPSCALE) -> float:
    if glyph_px <= 0:
        return 1.0
    return float(np.clip(target / glyph_px, 1.0, max_up))


# ---- Cleaning up a text crop before OCR ----
CROP_VARIANTS = ("enhanced", "otsu", "sauvola", "deblur")


def enhance_crop(crop: np.ndarray, glyph_px: float, upscaler: str = "bicubic",
                 variants=CROP_VARIANTS, target_glyph: float = TARGET_GLYPH_PX
                 ) -> list[tuple[str, np.ndarray]]:
    """Turn one small text crop into several versions ready for OCR, all dark text on white.

    Steps, in order: denoise at the original size (noise is easier to
    handle before scaling smears it), enlarge, boost contrast, make the text
    dark, then sharpen or threshold.
    """
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return []
    gray = denoise(gray)
    f = scale_for_glyph(glyph_px, target_glyph)
    big = upscale(gray, f, upscaler)
    big = stretch(big)
    if text_is_light(big):
        big = 255 - big  # Tesseract expects dark text on a light background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(big)
    out: list[tuple[str, np.ndarray]] = []
    pad = max(8, int(0.4 * target_glyph))
    for v in variants:
        if v == "enhanced":
            img = unsharp(clahe, sigma=max(1.0, f * 0.5), amount=0.8)
        elif v == "otsu":
            img = otsu(clahe)
        elif v == "sauvola":
            img = sauvola(clahe, window=int(target_glyph * 1.5) | 1)
        elif v == "deblur":
            # A blur of about 2 px at the target letter size fixes mild focus or motion
            # blur without halo artifacts. Very blurry crops are left for the "not
            # sure" rule to handle.
            img = stretch(richardson_lucy(clahe, sigma=2.0, iters=10))
        else:
            raise ValueError(f"unknown crop variant {v!r}")
        # white margin, because Tesseract needs some empty space around a line
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        out.append((v, img))
    return out
