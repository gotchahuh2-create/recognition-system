"""Make realistic variations of catalog photos for the reference gallery.

Each catalog photo gets 8-20 altered copies that look more like phone photos:

* perspective warp up to about 20 degrees, rotation by 0/90/180/270 plus
  up to 10 degrees either way;
* motion and focus blur, noise, JPEG quality 40-90;
* colour temperature and brightness changes, and a bright spot to imitate
  glare on the solder mask;
* random crops keeping 70-100 % of the board, and patches that imitate pin
  headers and cables covering parts of it;
* pasting onto backgrounds like a desk, an ESD mat or a dev board (either
  generated, or your own photos via ``backgrounds=``).

This is written with OpenCV/NumPy instead of Albumentations: the same
operations, repeatable with a seed, no extra package, and no breakage from
API changes between Albumentations 1.x and 2.x.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# ---- Backgrounds ----
def _bg_desk(rng, s):
    base = np.array([rng.uniform(60, 120), rng.uniform(90, 150), rng.uniform(130, 200)])
    img = np.ones((s, s, 3), np.float32) * base
    grain = cv2.resize(rng.normal(0, 18, (s // 8 + 1, 2)).astype(np.float32), (s, s))
    img += grain[..., None]
    return img


def _bg_mat(rng, s):
    palettes = [(110, 120, 40), (60, 90, 50), (90, 90, 90), (40, 40, 40)]
    img = np.ones((s, s, 3), np.float32) * np.array(palettes[rng.integers(len(palettes))])
    step = int(rng.uniform(0.05, 0.12) * s)
    for i in range(0, s, max(step, 4)):
        cv2.line(img, (i, 0), (i, s), (200, 200, 200), 1)
        cv2.line(img, (0, i), (s, i), (200, 200, 200), 1)
    return img


def _bg_devboard(rng, s):
    pcb = np.array([(40, 110, 30), (30, 30, 30), (120, 60, 20), (60, 20, 90)][rng.integers(4)])
    img = np.ones((s, s, 3), np.float32) * pcb
    for _ in range(int(rng.integers(8, 30))):
        x, y = rng.integers(0, s, 2)
        w, h = rng.integers(s // 40 + 2, s // 6 + 3, 2)
        col = [(180, 180, 190), (20, 20, 20), (60, 160, 200), (230, 230, 230)][rng.integers(4)]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, -1)
    return img


_BGS = (_bg_desk, _bg_mat, _bg_devboard)


def _load_backgrounds(paths) -> list[np.ndarray]:
    out = []
    for p in paths or []:
        p = Path(p)
        files = sorted(p.glob("*")) if p.is_dir() else [p]
        for f in files:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR) \
                if f.is_file() else None
            if img is not None:
                out.append(img)
    return out


# ---- Colour and lighting helpers ----
def add_glare(img: np.ndarray, rng) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    c = (int(rng.uniform(0.15, 0.85) * w), int(rng.uniform(0.15, 0.85) * h))
    axes = (int(rng.uniform(0.08, 0.3) * w), int(rng.uniform(0.04, 0.15) * h))
    cv2.ellipse(mask, c, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(axes) * 0.35 + 1)
    a = rng.uniform(0.4, 0.9) * mask[..., None]
    return np.clip(img * (1 - a) + 255 * a, 0, 255)


def colour_shift(img: np.ndarray, rng) -> np.ndarray:
    t = rng.uniform(-0.15, 0.15)  # warmer (+) or cooler (-)
    gain = np.array([1 - t, 1.0, 1 + t], np.float32)  # BGR
    img = img * gain * rng.uniform(0.7, 1.25) + rng.uniform(-20, 20)
    m = img.mean()
    return np.clip((img - m) * rng.uniform(0.8, 1.2) + m, 0, 255)


def motion_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, M, (length, length))
    return k / max(k.sum(), 1e-6)


def blur(img: np.ndarray, rng, scale: float) -> np.ndarray:
    kind = rng.integers(3)
    if kind == 0:
        return img
    if kind == 1:  # motion blur
        L = int(max(3, rng.uniform(0.004, 0.02) * scale)) | 1
        return cv2.filter2D(img, -1, motion_kernel(L, rng.uniform(0, 180)))
    r = max(1, int(rng.uniform(0.002, 0.008) * scale))  # out-of-focus blur (disc)
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1, -1)
    return cv2.filter2D(img, -1, k / k.sum())


def occlude(img: np.ndarray, rng, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    for _ in range(int(rng.integers(0, 3))):
        bw, bh = (x1 - x0), (y1 - y0)
        horiz = rng.random() < 0.5
        w = int(bw * (rng.uniform(0.3, 0.9) if horiz else rng.uniform(0.05, 0.15)))
        h = int(bh * (rng.uniform(0.05, 0.15) if horiz else rng.uniform(0.3, 0.9)))
        x = int(rng.uniform(x0, max(x1 - w, x0 + 1)))
        y = int(rng.uniform(y0, max(y1 - h, y0 + 1)))
        col = [(20, 20, 20), (40, 40, 40), (200, 200, 200), (30, 30, 160)][rng.integers(4)]
        cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
    return img


# ---- Main ----
class Augmenter:
    def __init__(self, seed: int = 0, backgrounds: Sequence[str | Path] = (),
                 max_side: int = 640):
        self.rng = np.random.default_rng(seed)
        self.bg_images = _load_backgrounds(backgrounds)
        self.max_side = max_side

    def _background(self, s: int) -> np.ndarray:
        rng = self.rng
        if self.bg_images and rng.random() < 0.7:
            bg = self.bg_images[rng.integers(len(self.bg_images))]
            bh, bw = bg.shape[:2]
            c = int(min(bh, bw) * rng.uniform(0.5, 1.0))
            y, x = rng.integers(0, bh - c + 1), rng.integers(0, bw - c + 1)
            return cv2.resize(bg[y:y + c, x:x + c], (s, s)).astype(np.float32)
        return _BGS[rng.integers(len(_BGS))](rng, s)

    def __call__(self, board: np.ndarray) -> np.ndarray:
        """Make one altered copy of a tightly cropped catalog board image (BGR)."""
        rng = self.rng
        h, w = board.shape[:2]
        f = min(1.0, self.max_side / max(h, w))
        if f < 1:
            board = cv2.resize(board, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
            h, w = board.shape[:2]
        src = board.astype(np.float32)
        if rng.random() < 0.5:
            src = add_glare(src, rng)

        # geometry: rotate by k*90 +/- 10 degrees, plus random perspective (up to ~20 degree
        # tilt)
        S = int(max(h, w) * 1.5)
        ang = np.deg2rad(90 * rng.integers(4) + rng.uniform(-10, 10))
        fit = rng.uniform(0.62, 0.8) * S / max(h, w)
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        dst = corners @ R.T * fit + S / 2
        jitter = rng.uniform(-1, 1, (4, 2)) * rng.uniform(0, 0.12) * max(h, w) * fit
        dst = (dst + jitter).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
        warped = cv2.warpPerspective(src, Hm, (S, S), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.ones((h, w), np.float32), Hm, (S, S))[..., None]
        canvas = self._background(S) * (1 - mask) + warped * mask

        # crop tightly like a query box (+3-10 %), sometimes cutting off part of the board
        x0, y0 = dst.min(0)
        x1, y1 = dst.max(0)
        bw, bh = x1 - x0, y1 - y0
        e = rng.uniform(0.03, 0.10)
        cx0, cy0, cx1, cy1 = x0 - e * bw, y0 - e * bh, x1 + e * bw, y1 + e * bh
        if rng.random() < 0.4:   # keep 70-100 % of the board
            side = rng.integers(4)
            cut = rng.uniform(0.0, 0.3)
            if side == 0:
                cx0 += cut * bw
            elif side == 1:
                cx1 -= cut * bw
            elif side == 2:
                cy0 += cut * bh
            else:
                cy1 -= cut * bh
        canvas = occlude(canvas, rng, (x0, y0, x1, y1))
        cx0, cy0 = int(max(cx0, 0)), int(max(cy0, 0))
        cx1, cy1 = int(min(cx1, S)), int(min(cy1, S))
        out = canvas[cy0:cy1, cx0:cx1]

        out = colour_shift(out, rng)
        out = blur(out, rng, max(out.shape[:2]))
        out = out + rng.normal(0, rng.uniform(0, 8), out.shape)
        out = np.clip(out, 0, 255).astype(np.uint8)
        q = int(rng.uniform(40, 90))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out

    def many(self, board: np.ndarray, n: int) -> list[np.ndarray]:
        return [self(board) for _ in range(n)]
