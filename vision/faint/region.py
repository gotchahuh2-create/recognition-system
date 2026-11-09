"""Step 1: find the dark area in the middle (the MCU) and straighten it.

Two detectors; the best score wins:

``blob``    The card step's detector (``card_reader.find_chip``): dark,
            smooth, square blobs covering 2-45 % of the image. Works well
            on a crop of a whole MCU card.
``central`` For close-ups, where the chip fills most of the picture: Otsu
            threshold on the smoothed lightness, with dark blobs scored by
            size, squareness and how close they are to the centre. Allows
            up to 95 % of the image.

If neither finds the chip, the whole image is used (method ``whole``) so
the later steps can still run.

The straightened chip is a square, enlarged so the text is big enough to
work with (the chip side is about 18 text lines tall, as in card_reader).
``body_mask`` marks the pixels of the chip's top surface, leaving out the
pins, the padding around it and a thin edge (the bevelled edges reflect a
lot of light).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np



def order_quad(pts) -> np.ndarray:
    """Sort the corners clockwise, starting near the top-left, by angle around the centre.

    Unlike the sum/difference trick, this never merges two corners when the
    chip is rotated 45 degrees (a diamond shape).
    """
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    c = p.mean(0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]  # clockwise in image coordinates (y points down)
    # start with the corner closest to -135 degrees (the top-left direction)
    a = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    k = int(np.argmin(np.abs(np.angle(np.exp(1j * (a + 3 * np.pi / 4))))))
    return np.roll(p, -k, axis=0)

TARGET_SIDE = 900  # straightened chip side (px), same as card_reader.CHIP_TARGET_SIDE
MAX_UP = 6.0
PAD = 0.04  # padding around the chip before straightening
RIM = 0.045  # edge left out of the body mask (bevels reflect light)
GLYPHS_PER_SIDE = 18.0  # guess: a line of text is about 1/18 of the chip side


@dataclass
class ChipCrop:
    image: np.ndarray  # the straightened chip, BGR uint8, S x S
    body_mask: np.ndarray        # bool S x S
    quad: np.ndarray  # 4x2 chip corners in the INPUT image
    H: np.ndarray  # transform from the input to the straightened image
    method: str
    score: float
    glyph_px: float  # expected letter height in ``image`` px

    def as_dict(self) -> dict:
        return {"method": self.method, "score": round(self.score, 3),
                "quad": np.round(self.quad, 1).tolist(), "side": int(self.image.shape[0]),
                "glyph_px": round(self.glyph_px, 1)}


def _flat_mask(img: np.ndarray) -> np.ndarray:
    """Find smooth (low-detail) pixels.

    In a chip close-up, the chip's top is the biggest smooth area; the pins,
    traces, pads and printing around it are not. On a black board the board
    is as dark as the chip, so darkness alone can't tell them apart, but
    smoothness can. Faint text and soft glare are smooth too, so they stay
    inside the chip area.
    """
    H, W = img.shape[:2]
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    k = max(5, int(0.025 * max(H, W)) | 1)
    mu = cv2.blur(L, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(L * L, (k, k)) - mu * mu, 0))
    # threshold relative to the smoothest part of the image centre
    c = sd[H // 4:3 * H // 4, W // 4:3 * W // 4]
    t = max(4.0, 2.2 * float(np.percentile(c, 30)))
    return (sd < t).astype(np.uint8) * 255


def _central_candidates(img: np.ndarray) -> list[tuple[np.ndarray, float]]:
    H, W = img.shape[:2]
    k = max(5, int(0.02 * max(H, W)) | 1)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    Ls = cv2.GaussianBlur(L, (k, k), 0)
    _, dark = cv2.threshold(Ls, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks = [_flat_mask(img), dark]
    out = []
    cx, cy, diag = W / 2, H / 2, np.hypot(W, H) / 2
    for i, m in enumerate(masks):
        # close up strokes and glare edges inside the chip, and remove thin traces
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        for q, s in _score_contours(m, H, W, cx, cy, diag):
            out.append((q, s * (1.0 if i == 0 else 0.85)))
    return out


def _score_contours(m, H, W, cx, cy, diag):
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        a = cv2.contourArea(c)
        if not (0.02 * H * W <= a <= 0.95 * H * W):
            continue
        r = cv2.minAreaRect(c)
        rw, rh = r[1]
        fill = a / max(rw * rh, 1.0)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if fill < 0.7 or aspect > 1.5:
            continue
        cent = 1.0 - min(1.0, np.hypot(r[0][0] - cx, r[0][1] - cy) / diag)
        score = (0.35 * min(1.0, a / (0.15 * H * W)) + 0.25 * (fill - 0.7) / 0.3
                 + 0.2 * (1.5 - aspect) / 0.5 + 0.2 * cent)
        out.append((order_quad(cv2.boxPoints(r)), float(score)))
    return out


def locate_chip(img: np.ndarray, use_blob: bool = True) -> tuple[np.ndarray, str, float]:
    """Find the best chip outline in ``img``. Returns (corners, detector name, score)."""
    cands: list[tuple[np.ndarray, str, float]] = []
    if use_blob:
        from ..card_reader import find_chip
        for c in find_chip(img)[:3]:
            cands.append((c.quad, "blob", c.score))
    for q, s in _central_candidates(img)[:5]:
        cands.append((q, "central", s))
    if not cands:
        H, W = img.shape[:2]
        q = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
        return q, "whole", 0.0
    q, m, s = max(cands, key=lambda c: c[2])
    return q, m, s


def rectify(img: np.ndarray, quad: np.ndarray, side: int = TARGET_SIDE,
            pad: float = PAD, method: str = "given", score: float = 1.0,
            upscaler: str = "bicubic") -> ChipCrop:
    """Straighten the chip and enlarge it.

    upscaler: "bicubic" warps straight to the final size. A super-resolution
    model ("espcn" / "fsrcnn" / "edsr", see ``vision.enhance``) first warps
    at the chip's ORIGINAL size (only rotating, no scaling) and lets the
    model enlarge it; step 2 then works on that output. If the model is
    missing, ``enhance.upscale`` falls back to bicubic.
    """
    q = order_quad(quad).astype(np.float64)
    c = q.mean(0)
    qp = c + (q - c) * (1 + 2 * pad)
    native = max(np.linalg.norm(qp[1] - qp[0]), np.linalg.norm(qp[3] - qp[0]), 8.0)
    S = int(np.clip(side, native, native * MAX_UP))
    dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]], np.float32)
    Hm = cv2.getPerspectiveTransform(qp.astype(np.float32), dst)
    if upscaler not in ("bicubic", None) and S > native * 1.2:
        from .. import enhance
        n = int(round(native))
        dn = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], np.float32)
        Hn = cv2.getPerspectiveTransform(qp.astype(np.float32), dn)
        small = cv2.warpPerspective(img, Hn, (n, n), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        out = enhance.upscale(small, S / n, upscaler)
        out = cv2.resize(out, (S, S), interpolation=cv2.INTER_CUBIC) if out.shape[0] != S else out
    else:
        out = cv2.warpPerspective(img, Hm, (S, S), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    m = int(round(S * (pad / (1 + 2 * pad) + RIM)))
    body = np.zeros((S, S), bool)
    body[m:S - m, m:S - m] = True
    glyph = (S - 2 * m) / GLYPHS_PER_SIDE
    return ChipCrop(out, body, q.astype(np.float32), Hm, method, score, float(glyph))


def extract_chip(img: np.ndarray, quad: np.ndarray | None = None, side: int = TARGET_SIDE,
                 saver=None) -> ChipCrop:
    """Step 1. Passing ``quad`` skips detection (when the card step already found the chip)."""
    if quad is None:
        quad, method, score = locate_chip(img)
    else:
        method, score = "given", 1.0
    cc = rectify(img, quad, side, method=method, score=score)
    if saver is not None and saver.enabled:
        vis = img.copy()
        cv2.polylines(vis, [np.round(cc.quad).astype(np.int32)], True, (255, 0, 255),
                      max(2, img.shape[1] // 300))
        saver.save(f"1a_located_{method}", vis)
        saver.save("1b_rectified_chip", cc.image)
        bm = cc.image.copy()
        bm[~cc.body_mask] //= 4
        saver.save("1c_body_mask", bm)
    return cc
