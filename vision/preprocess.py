"""Clean up a photo before OCR (OpenCV only; fast and always gives the same result).

The basics: resize to a fixed long side, boost contrast with CLAHE, and
optionally fix perspective from 4 corners the user clicks. On top of that,
because OCR needs it:

* a quality check (blur, glare, exposure), so the UI can ask for a new
  photo instead of returning a confident wrong answer,
* straightening small tilts using the board edges and text lines,
* removing glare (shiny spots on the solder mask),
* image variants for OCR (silkscreen is usually light text on a dark PCB),
* overlapping tiles, so tiny chip markings can be cut out and read.

Every geometric change is tracked in a 3x3 matrix ``M`` (original to
processed), so anything found later can be mapped back onto the user's
original photo.

On purpose, this module does NOT try to find sockets or Click boards with
OpenCV tricks. That's a job for a trained detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import cv2
import numpy as np

# ---- Settings and results ----
@dataclass
class PreprocessConfig:
    long_side: int = 2000  # OCR needs pixels; later steps can scale down further
    upscale_small: bool = True  # enlarge small photos (e.g. chip close-ups) up to long_side
    max_upscale: float = 2.5
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    deskew: bool = True
    max_skew_deg: float = 30.0  # only fix small tilts; OCR handles 90 degree turns
    reduce_glare: bool = True
    glare_inpaint_max_frac: float = 0.12  # filling in large areas does more harm than good
    # Extra clean-up steps, all OFF by default. On the synthetic test set they
    # made the whole-photo pass worse on dark photos (denoising and gamma soften
    # the thin laser-engraved strokes: MCU 6/10 -> 4/10), and the crop pass does
    # these steps per crop anyway (see DECISIONS.md). They're kept as options to
    # try on real photos. "auto" means only when the photo needs it, True means
    # always.
    denoise: bool | str = False  # denoise when the measured noise is above NOISE_MAX
    lift_dark: bool = False  # brighten under-exposed photos (before CLAHE)
    sharpen: bool | str = False  # sharpen when the photo is flagged as blurry
    # straighten the board outline if found (off: fails on busy desks)
    auto_perspective: bool = False


@dataclass
class QualityReport:
    width: int
    height: int
    blur_var: float  # sharpness (variance of Laplacian) on a copy 1000 px long
    glare_frac: float  # share of pixels that are glare
    brightness: float  # average brightness, 0..255
    contrast: float  # standard deviation of gray
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class Preprocessed:
    image: np.ndarray  # BGR, resized, straightened, glare reduced, CLAHE applied
    gray: np.ndarray              # gray version of ``image``
    M: np.ndarray  # 3x3 transform from original to processed
    scale: float
    skew_deg: float
    quality: QualityReport
    steps: list[str] = field(default_factory=list)

    def to_original(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 points from the processed image back to the original photo."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        Minv = np.linalg.inv(self.M)
        hom = np.hstack([pts, np.ones((len(pts), 1))]) @ Minv.T
        return hom[:, :2] / hom[:, 2:3]


# Starting thresholds. Tune them on real photos.
BLUR_MIN = 60.0
GLARE_MAX = 0.08
DARK_MAX = 60.0
BRIGHT_MIN = 225.0
MIN_LONG_SIDE = 800
NOISE_MAX = 5.0  # noise level (Immerkaer) on the working image


# ---- Quality checks ----
def glare_mask(img: np.ndarray, v_min: int = 235, s_max: int = 45) -> np.ndarray:
    """Find glare: pixels that are very bright and almost colourless."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).astype(np.uint8) * 255
    # remove isolated pixels (white silkscreen lines are thin, glare comes in blobs)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def assess_quality(img: np.ndarray) -> QualityReport:
    h, w = img.shape[:2]
    small = resize_long_side(img, 1000, allow_upscale=False)[0]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gm = glare_mask(small)
    glare = float((gm > 0).mean())
    v = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].mean())
    q = QualityReport(w, h, blur, glare, v, float(gray.std()))
    if max(w, h) < MIN_LONG_SIDE:
        q.warnings.append(f"low resolution ({w}x{h}); move closer or use the main camera")
    if blur < BLUR_MIN:
        q.warnings.append("photo looks blurry; hold still or tap to focus")
    if glare > GLARE_MAX:
        q.warnings.append(f"strong glare on {glare:.0%} of the photo; tilt the board or light")
    if v < DARK_MAX:
        q.warnings.append("photo is too dark")
    elif v > BRIGHT_MIN:
        q.warnings.append("photo is overexposed")
    return q


# ---- Geometry ----
def resize_long_side(img: np.ndarray, target: int, allow_upscale: bool = True,
                     max_upscale: float = 2.5) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = target / max(h, w)
    if s > 1 and not allow_upscale:
        return img, 1.0
    s = min(s, max_upscale)
    if abs(s - 1) < 1e-3:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp), s


def estimate_skew(gray: np.ndarray, max_abs_deg: float = 45.0) -> float:
    """Find the main line angle in the image, between -45 and 45 degrees.

    PCBs are full of straight lines (board edges, headers, traces, text)
    that run along the board's axes. So the most common line angle (modulo
    90 degrees) is how much the board is tilted. Longer lines count more.
    Returns 0 if there is no clear winner.
    """
    g = resize_long_side(gray, 1000, allow_upscale=False)[0]
    edges = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 150)
    min_len = max(g.shape) // 12
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                            minLineLength=min_len, maxLineGap=6)
    if lines is None or len(lines) < 4:
        return 0.0
    # works with both OpenCV 4 and 5 output shapes
    x1, y1, x2, y2 = lines.reshape(-1, 4).T.astype(np.float64)
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    ang = (ang + 45) % 90 - 45  # fold into [-45, 45)
    length = np.hypot(x2 - x1, y2 - y1)
    hist, bins = np.histogram(ang, bins=180, range=(-45, 45), weights=length)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")  # smooth the histogram (0.5 degree bins)
    k = int(hist.argmax())
    if hist[k] < 0.15 * length.sum():  # no clear main direction
        return 0.0
    near = np.abs(ang - (bins[k] + bins[k + 1]) / 2) < 1.5
    angle = float(np.average(ang[near], weights=length[near])) if near.any() else 0.0
    return angle if abs(angle) <= max_abs_deg else 0.0


def rotate_bound(img: np.ndarray, angle_deg: float,
                 border=cv2.BORDER_REPLICATE) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by ``angle_deg`` (counter-clockwise, like OpenCV) without cutting anything off.

    Returns (image, 3x3 matrix).
    """
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    A = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(A[0, 0]), abs(A[0, 1])
    nw, nh = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    A[0, 2] += nw / 2 - c[0]
    A[1, 2] += nh / 2 - c[1]
    out = cv2.warpAffine(img, A, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=border)
    return out, np.vstack([A, [0, 0, 1]])


def rotate90(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate by k * 90 degrees clockwise (k is 0..3)."""
    k %= 4
    if k == 0:
        return img
    return cv2.rotate(img, [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                            cv2.ROTATE_90_COUNTERCLOCKWISE][k])


def order_quad(pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Sort 4 points as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]],
                    dtype=np.float32)


def four_point_warp(img: np.ndarray, pts: Sequence[Sequence[float]]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Fix perspective using the 4 corners the user clicked (in any order).

    Returns (warped image, 3x3 transform from original to warped).
    """
    tl, tr, br, bl = q = order_quad(pts)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC), H


def find_board_quad(img: np.ndarray, min_area_frac: float = 0.15) -> np.ndarray | None:
    """Try to find the board outline: the biggest convex 4-cornered shape.

    Works when the PCB stands out from the desk. Returns 4x2 points in
    ``img`` coordinates, or None. Only use the result as a suggestion the
    user can accept, because it fails on busy desks.
    """
    small, s = resize_long_side(img, 800, allow_upscale=False)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(g, 40, 120), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area_frac * small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < area_min:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad(approx.reshape(4, 2) / s)
    return None


# ---- Brightness and contrast ----
def apply_clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Apply CLAHE to the lightness channel only, so colours (green/black PCB, LEDs) are kept."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def suppress_glare(img: np.ndarray, max_frac: float = 0.12) -> tuple[np.ndarray, float]:
    """Fill in glare spots.

    Skipped when there is a lot of glare: filling a big area invents
    texture that OCR then reads as letters.
    """
    m = glare_mask(img)
    frac = float((m > 0).mean())
    if frac == 0 or frac > max_frac:
        return img, frac
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA), frac


# ---- Main entry point ----
def preprocess(img: np.ndarray, cfg: PreprocessConfig | None = None,
               quad: Sequence[Sequence[float]] | None = None) -> Preprocessed:
    """Run all the preprocessing steps.

    ``quad``: optional 4 corners (in original photo coordinates) from the
    UI, to fix the perspective. If given, we skip straightening, because
    the warp already did that.
    """
    cfg = cfg or PreprocessConfig()
    quality = assess_quality(img)
    steps: list[str] = []
    M = np.eye(3)

    out = img
    if quad is None and cfg.auto_perspective:
        q = find_board_quad(img)
        if q is not None:
            quad = q
            steps.append("auto-quad")
    if quad is not None:
        out, H = four_point_warp(out, quad)
        M = H @ M
        steps.append("perspective")

    # Brightness/noise fixes run at the original size: noise is easier to remove
    # before scaling smears it, and denoising gets slower with more pixels.
    from . import enhance  # imported here; enhance doesn't depend on this module
    if cfg.lift_dark and quality.brightness < DARK_MAX * 1.5:
        out, g = enhance.gamma_lift(out)
        if g > 1.0:
            steps.append(f"gamma {g:.2f}")
    if cfg.denoise:
        small = resize_long_side(out, 1000, allow_upscale=False)[0]
        sigma = enhance.noise_sigma(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if cfg.denoise != "auto" or sigma > NOISE_MAX:
            out = enhance.denoise(out, sigma=max(sigma, 3.0))
            steps.append(f"nlm-denoise σ={sigma:.1f}")

    out, s = resize_long_side(out, cfg.long_side, allow_upscale=cfg.upscale_small,
                              max_upscale=cfg.max_upscale)
    M = np.diag([s, s, 1.0]) @ M
    steps.append(f"resize x{s:.2f}")

    if cfg.reduce_glare:
        out, gfrac = suppress_glare(out, cfg.glare_inpaint_max_frac)
        if 0 < gfrac <= cfg.glare_inpaint_max_frac:
            steps.append(f"glare-inpaint {gfrac:.1%}")

    skew = 0.0
    if cfg.deskew and quad is None:
        skew = estimate_skew(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cfg.max_skew_deg)
        if abs(skew) >= 0.5:
            out, R = rotate_bound(out, skew)  # rotating by +skew undoes the tilt
            M = R @ M
            steps.append(f"deskew {skew:+.1f}°")

    out = apply_clahe(out, cfg.clahe_clip, cfg.clahe_tile)
    steps.append("clahe")
    if cfg.sharpen and (cfg.sharpen != "auto" or quality.blur_var < BLUR_MIN):
        out = cv2.addWeighted(out, 1.8, cv2.GaussianBlur(out, (0, 0), 2.0), -0.8, 0)
        steps.append("unsharp")
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return Preprocessed(out, gray, M, s, skew, quality, steps)


# ---- OCR helpers ----
def ocr_variants(gray: np.ndarray, which: Sequence[str] = ("gray", "inverted")
                 ) -> list[tuple[str, np.ndarray]]:
    """Make image versions that help classic OCR engines.

    Silkscreen is light text on a dark board, and chip markings are light
    engraving on dark plastic. Tesseract expects dark text on light, so we
    also make an "inverted" version. Neural OCR (PaddleOCR / RapidOCR)
    usually only needs "gray".
    """
    out: list[tuple[str, np.ndarray]] = []
    for name in which:
        if name == "gray":
            out.append((name, gray))
        elif name == "inverted":
            out.append((name, 255 - gray))
        elif name == "otsu":
            g = cv2.GaussianBlur(gray, (3, 3), 0)
            _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # make the text dark: if most pixels are dark, the text must be the light part
            if (b > 0).mean() < 0.5:
                b = 255 - b
            out.append((name, b))
        elif name == "adaptive":
            b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
            if (b > 0).mean() < 0.5:
                b = 255 - b
            out.append((name, b))
        else:
            raise ValueError(f"unknown variant {name!r}")
    return out


def iter_tiles(img: np.ndarray, tile: int = 1200, overlap: float = 0.2
               ) -> Iterator[tuple[int, int, np.ndarray]]:
    """Split the image into overlapping tiles, yielding (x0, y0, crop).

    The overlap must be longer than the longest word, so no part number gets
    cut in every tile.
    """
    h, w = img.shape[:2]
    if max(h, w) <= tile:
        yield 0, 0, img
        return
    step = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for x in xs:
            yield x, y, img[y:y + tile, x:x + tile]


def crop(img: np.ndarray, box: Sequence[float], pad: float = 0.1) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a box (x, y, w, h) with some extra padding. Returns the crop and its offset."""
    x, y, w, h = box
    px, py = w * pad, h * pad
    x0, y0 = max(0, int(x - px)), max(0, int(y - py))
    x1 = min(img.shape[1], int(x + w + px))
    y1 = min(img.shape[0], int(y + h + py))
    return img[y0:y1, x0:x1], (x0, y0)
