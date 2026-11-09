"""Shape helpers and *careful* clean-up (step 6).

The marking is the faintest thing on the chip, so the usual clean-up
("erode the mask, drop everything small") deletes it first. Instead:

* Nothing is removed by erosion. We only drop blobs that can't be part of
  a letter: specks smaller than about 1/12 of a letter in both directions,
  and big solid blobs (glare patches, the pin-1 dimple, edge bevels) that
  are bigger than a letter in BOTH directions and mostly filled.
* Broken strokes are joined again with a small closing, so weak pixels
  next to strong ones survive.
* Hysteresis keeps weak pixels that touch strong ones (like Canny edge
  detection). This is the main way the black-and-white variants keep
  useful faint pixels.

``stroke_score`` rates how much a black-and-white mask looks like text. The
cluster and threshold steps use it to pick which group / level is the text.
"""

from __future__ import annotations

import cv2
import numpy as np


def _cc(mask: np.ndarray):
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)


def stroke_score(mask: np.ndarray, body: np.ndarray, g: float) -> float:
    """Rate how text-like a mask is, from 0 to 1.

    It's the share of the mask's area in letter-sized blobs, times a factor
    for how much of the chip is covered (text usually covers about 1-25 %).
    """
    m = mask & body
    tot = int(m.sum())
    nb = max(int(body.sum()), 1)
    cov = tot / nb
    if tot == 0 or cov > 0.5:
        return 0.0
    n, lab, st, cen = _cc(m)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < max(4, (g / 12) ** 2):
            continue
        # One letter, or several merged into a word (smoothing joins letters):
        # the short side is about a letter tall, the long side up to a full line.
        if 0.3 * g <= min(w, h) <= 1.8 * g and max(w, h) <= 14 * g and max(w, h) >= 0.3 * g:
            fill = a / max(w * h, 1)
            if 0.12 <= fill <= 0.9:
                boxes.append((x, y, w, h, a))
    prior = 1.0 if 0.005 <= cov <= 0.3 else (cov / 0.005 if cov < 0.005 else max(0.0, (0.5 - cov) / 0.2))
    if len(boxes) < 3:
        return float(0.2 * sum(b[4] for b in boxes) / tot * prior)
    return float(prior * max(_chained_area(boxes, g), _chained_area(
        [(y, x, h, w, a) for x, y, w, h, a in boxes], g)) / tot)


def _chained_area(boxes, g: float) -> float:
    """Area of letter-sized blobs that sit in a row of 3 or more along x.

    They must be on the same line (centres within 0.35 of the height), of
    similar height, with small gaps. Random noise rarely lines up like this;
    printed text always does.
    """
    B = np.asarray(boxes, np.float64)
    x, y, w, h, a = B.T
    cy = y + h / 2
    order = np.argsort(x)
    parent = list(range(len(B)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            gap = x[j] - (x[i] + w[i])
            if gap > 0.9 * g:
                break
            if (abs(cy[i] - cy[j]) < 0.35 * max(h[i], h[j])
                    and 0.6 < h[i] / max(h[j], 1) < 1.65 and gap > -0.3 * g):
                parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(B))])
    _, inv, cnt = np.unique(roots, return_inverse=True, return_counts=True)
    # a merged word along x (at least 2.5 letters long, letter-height tall) counts as a row
    # by itself
    word = (w >= 2.5 * h) & (h >= 0.4 * g)
    return float(a[(cnt[inv] >= 3) | word].sum())


def gentle_clean(mask: np.ndarray, body: np.ndarray, g: float, close: bool = True) -> np.ndarray:
    m = (mask & body).astype(np.uint8)
    if close:
        k = max(3, int(g / 16) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m &= body.astype(np.uint8)
    n, lab, st, _ = _cc(m)
    keep = np.zeros(n, bool)
    speck = max(3, (g / 12) ** 2)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < speck and max(w, h) < g / 6:
            continue
        if w > 2.2 * g and h > 2.2 * g and a / max(w * h, 1) > 0.55:
            continue  # glare patch, dimple or bevel
        if is_streak(w, h, a, g):
            continue                                  # thin reflection streak
        keep[i] = True
    return keep[lab]


def is_streak(w: float, h: float, a: float, g: float) -> bool:
    """Is this blob a reflection streak?

    A streak is as long as a word but SOLID. A word (even with letters merged
    by smoothing) has gaps and holes, so only about 30-65 % is filled.
    """
    return max(w, h) > 3 * g and a / max(w * h, 1) > 0.72


def streak_mask(score_mask: np.ndarray, g: float) -> np.ndarray:
    """Find solid, line-length blobs (glare streaks) in a "possible text" mask, grown a little.

    Used to blank them out of the gray variants.
    """
    n, lab, st, _ = _cc(score_mask)
    bad = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        bad[i] = is_streak(w, h, a, g)
    m = bad[lab]
    if m.any():
        k = max(3, int(g / 5) | 1)
        m = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    return m


def hysteresis(score: np.ndarray, lo: float, hi: float, body: np.ndarray) -> np.ndarray:
    weak = (score > lo) & body
    strong = (score > hi) & body
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    hit = np.zeros(n, bool)
    hit[np.unique(lab[strong])] = True
    hit[0] = False
    return hit[lab]


def to_ocr(img: np.ndarray, g: float, target_glyph: float = 40.0, pad_glyphs: float = 0.8,
           is_mask: bool = False) -> tuple[np.ndarray, float, int]:
    """Make a dark-text-on-white uint8 image for OCR, with letters about ``target_glyph`` px
    tall.

    ``img`` is a mask (True = text) or a float/uint8 text score (high = text).
    Adds a white margin. Returns (image, scale, pad) so words can be mapped
    back: chip_xy = (ocr_xy - pad) / scale.
    """
    if is_mask or img.dtype == bool:
        u = np.where(img, 0, 255).astype(np.uint8)
    else:
        a = img.astype(np.float32)
        p0, p1 = np.percentile(a, [1, 99.7])
        u = (255 - np.clip((a - p0) * 255.0 / max(p1 - p0, 1e-6), 0, 255)).astype(np.uint8)
    f = float(np.clip(target_glyph / max(g, 1.0), 0.35, 3.0))
    if abs(f - 1) > 0.05:
        u = cv2.resize(u, None, fx=f, fy=f,
                       interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    pad = int(pad_glyphs * target_glyph)
    u = cv2.copyMakeBorder(u, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return u, f, pad


def text_axis(mask: np.ndarray, body: np.ndarray, g: float) -> tuple[str | None, float, float]:
    """Work out which way the text lines run.

    Returns 'h' (read at 0/180 degrees), 'v' (90/270 degrees), or None when
    neither is clear. Uses the same row test as ``stroke_score``, once along
    x and once along y.
    """
    m = mask & body
    n, lab, st, _ = _cc(m)
    boxes = [tuple(st[i]) for i in range(1, n)
             if 0.3 * g <= min(st[i][2], st[i][3]) <= 1.8 * g and max(st[i][2], st[i][3]) <= 14 * g
             and st[i][4] >= max(4, (g / 12) ** 2)]
    if len(boxes) < 2:
        return None, 0.0, 0.0
    h = _chained_area(boxes, g)
    v = _chained_area([(y, x, hh, w, a) for x, y, w, hh, a in boxes], g)
    if max(h, v) <= 0 or max(h, v) < 1.5 * min(h, v):
        return None, float(h), float(v)
    return ("h" if h > v else "v"), float(h), float(v)
