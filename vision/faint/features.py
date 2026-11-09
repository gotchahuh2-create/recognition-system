"""Steps 2-3: colour values for every pixel in RGB, HSV and LAB.

Faint laser marking differs from the chip surface by only a few levels of
lightness, and often by a slight change in gloss, saturation or tint. No
single channel shows it reliably, so every pixel gets values in three
colour spaces:

    RGB   R, G, B
    HSV   V, S, S*cos(H), S*sin(H)   (hue as a direction, because hue goes in a
                                      circle and means nothing when S is ~0)
    LAB   L, a, b

First, uneven lighting is removed from each channel (``flatten``): we
estimate the background with a blur that ignores pixels that already stand
out (the text itself, reflections), and subtract it. What's left (the
*residual*) is about 0 on clean chip surface everywhere, whatever the
lighting.

The residuals are then normalised using the chip's spread (median / MAD),
so the three spaces can be compared, and lightly smoothed at the size of a
letter stroke (``stroke_sigma``). This makes faint strokes stand out while
noise averages away. The unsmoothed residuals are kept too, and the
threshold step uses both.

Reflections (very bright, colourless, large) are marked in ``glare`` and
left out of all statistics, but their pixels aren't deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SPACES = ("rgb", "hsv", "lab")
CHANNELS = {"rgb": ("R", "G", "B"), "hsv": ("V", "S", "Hc", "Hs"), "lab": ("L", "a", "b")}


@dataclass
class PixelFeatures:
    raw: dict[str, np.ndarray]            # space -> HxWxC float32, original values
    resid: dict[str, np.ndarray]  # space -> HxWxC float32, lighting removed, normalised
    smooth: dict[str, np.ndarray]  # space -> HxWxC, resid smoothed at stroke size
    background: dict[str, np.ndarray]  # space -> HxWxC estimated surface (for debugging)
    stats_mask: np.ndarray  # bool: chip pixels used for statistics
    glare: np.ndarray  # bool: reflections
    glyph_px: float
    info: dict = field(default_factory=dict)

    def stack(self, which: str = "smooth", spaces=SPACES) -> np.ndarray:
        src = self.smooth if which == "smooth" else self.resid
        return np.concatenate([src[s] for s in spaces], axis=2)


def color_spaces(bgr: np.ndarray) -> dict[str, np.ndarray]:
    f = bgr.astype(np.float32) / 255.0
    rgb = f[:, :, ::-1] * 255.0
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)  # H 0..360, S and V 0..1
    h = np.deg2rad(hsv[:, :, 0])
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    hsv4 = np.dstack([v * 255.0, s * 255.0, s * np.cos(h) * 255.0, s * np.sin(h) * 255.0])
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)  # L 0..100, a/b about -127..127
    lab = np.dstack([lab[:, :, 0] * 2.55, lab[:, :, 1], lab[:, :, 2]])
    return {"rgb": rgb.astype(np.float32), "hsv": hsv4.astype(np.float32),
            "lab": lab.astype(np.float32)}


def glare_mask(bgr: np.ndarray, body: np.ndarray, glyph: float) -> np.ndarray:
    """Find glare: pixels much brighter than the chip, colourless, and bigger than a letter
    stroke.

    The size check stops bright text from being mistaken for glare.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    med = float(np.median(v[body])) if body.any() else float(np.median(v))
    m = ((v > max(med + 90, 170)) & (hsv[:, :, 1] < 70)).astype(np.uint8)
    k = max(3, int(glyph * 0.6) | 1)
    core = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return cv2.dilate(core, np.ones((5, 5), np.uint8)).astype(bool)


def _norm_conv(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(x * w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return num / np.maximum(den, 1e-4)


def flatten(ch: np.ndarray, valid: np.ndarray, sigma: float, iters: int = 2,
            k: float = 2.0) -> np.ndarray:
    """Estimate the background of one channel, ignoring outliers.

    A large ``sigma`` (about 2-3 letters) keeps the letters themselves out
    of the background.
    """
    w = valid.astype(np.float32)
    # work at a lower resolution: the background is smooth, and big blurs are slow
    f = max(1, int(sigma // 6))
    small = cv2.resize(ch, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA) if f > 1 else ch
    ws = cv2.resize(w, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) if f > 1 else w
    s = sigma / f
    bg = _norm_conv(small, ws, s)
    for _ in range(iters):
        r = small - bg
        mad = np.median(np.abs(r[ws > 0.5])) * 1.4826 + 1e-3 if (ws > 0.5).any() else 1.0
        wr = ws * (np.abs(r) < k * mad)
        bg = _norm_conv(small, wr.astype(np.float32), s)
    if f > 1:
        bg = cv2.resize(bg, (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_CUBIC)
    return bg.astype(np.float32)


def flatten_median(ch: np.ndarray, valid: np.ndarray, win: float) -> np.ndarray:
    """Estimate the background as the median over a window of about 1.6 letters.

    Strokes are a minority of any such window (even for bold, dense text),
    so they disappear, while broad things like lighting gradients,
    reflections and the pin-1 dimple are followed. Done in 8 bits after
    stretching each channel over the chip (the photo is 8-bit anyway), at
    half resolution for speed, then smoothed.
    """
    v = ch[valid] if valid.any() else ch.ravel()
    lo, hi = np.percentile(v, [0.2, 99.8])
    hi = max(hi, lo + 1e-3)
    u = np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    half = cv2.resize(u, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    k = max(3, int(win / 2) | 1)
    med = cv2.medianBlur(half, min(k, 255))
    med = cv2.GaussianBlur(med.astype(np.float32), (0, 0), max(1.0, k / 6))
    bg = cv2.resize(med, (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_CUBIC)
    return (bg * (hi - lo) / 255.0 + lo).astype(np.float32)


def robust_z(x: np.ndarray, mask: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Normalise as (x - median) / MAD over ``mask``.

    ``floor`` sets a minimum spread. In 8-bit / JPEG images the colour
    channels are often constant over whole blocks, so their MAD is about 0,
    and without a minimum those channels would blow up to huge values.
    """
    v = x[mask] if mask.any() else x.ravel()
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return ((x - med) / max(mad, floor)).astype(np.float32)


def compute(bgr: np.ndarray, body: np.ndarray, glyph_px: float, denoise: bool = True,
            bg_glyphs: float = 1.6, bg_mode: str = "median", saver=None) -> PixelFeatures:
    """Compute all colour values for the chip.

    bg_mode: "median" (the default; follows glare) or "gauss" (smoother,
    but leaves glare in the residual; kept for comparison).
    """
    g = max(glyph_px, 4.0)
    src = bgr
    if denoise:
        # Light, edge-preserving smoothing well below the stroke width: reduces camera
        # noise but keeps faint strokes. (Not non-local means: it erases faint strokes.)
        d = max(3, int(g / 10) | 1)
        src = cv2.bilateralFilter(bgr, d, 12, d)
    glare = glare_mask(bgr, body, g)
    stats = body & ~glare
    spaces = color_spaces(src)
    raw, resid, smooth, bgs = {}, {}, {}, {}
    stroke_sigma = max(0.8, g / 9.0)  # about half a stroke width
    # Colour channels: JPEG stores colour at half resolution in 8x8 blocks, so
    # smooth these at block size first, so block edges don't look like strokes.
    chroma = {("hsv", 1), ("hsv", 2), ("hsv", 3), ("lab", 1), ("lab", 2)}
    chroma_sigma = max(1.0, g / 5.0)
    for name, arr in spaces.items():
        rs, ss, bb = [], [], []
        for c in range(arr.shape[2]):
            ch = arr[:, :, c]
            if (name, c) in chroma:
                ch = cv2.GaussianBlur(ch, (0, 0), chroma_sigma)
            bg = (flatten_median(ch, body, bg_glyphs * g) if bg_mode == "median"
                  else flatten(ch, stats, sigma=bg_glyphs * g))
            r = robust_z(ch - bg, stats, floor=1.0)  # at least 1 level of an 8-bit channel
            rs.append(r)
            ss.append(cv2.GaussianBlur(r, (0, 0), stroke_sigma))
            bb.append(bg)
        raw[name] = arr
        resid[name] = np.dstack(rs)
        sm = np.dstack(ss)
        # normalise again after smoothing: noise shrinks, but the text mostly stays
        smooth[name] = np.dstack([robust_z(sm[:, :, c], stats, floor=0.05)
                                  for c in range(sm.shape[2])])
        bgs[name] = np.dstack(bb)
    pf = PixelFeatures(raw, resid, smooth, bgs, stats, glare, g,
                       {"stroke_sigma": round(stroke_sigma, 2), "glare_frac": round(float(glare[body].mean()) if body.any() else 0.0, 4)})
    if saver is not None and saver.enabled:
        saver.save("2a_denoised", src)
        saver.save("2b_glare_mask", glare)
        saver.save("2c_bg_L", bgs["lab"][:, :, 0])
        for name in SPACES:
            saver.save(f"3_{name}_resid_{CHANNELS[name][0]}", resid[name][:, :, 0])
        saver.save("3_lab_smooth_L", smooth["lab"][:, :, 0])
        saver.save("3_hsv_smooth_S", smooth["hsv"][:, :, 1])
        saver.save("3_lab_smooth_ab", np.hypot(smooth["lab"][:, :, 1], smooth["lab"][:, :, 2]))
    return pf
