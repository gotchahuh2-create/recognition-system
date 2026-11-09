"""Step 7: the processed images that get sent to OCR.

Gray variants keep every weak pixel as a shade of gray instead of deciding
for it, which gives the OCR engine's own thresholding a second opinion:

``fis-soft``   the Fisher text score (all three colour spaces combined)
``L-flat``     LAB lightness with lighting removed, flipped if needed, with
               CLAHE. The closest to a classic "enhanced gray" image.

Black-and-white variants:

``vote2``      pixels that at least two colour spaces grouped as text
``stage2``     the local second-pass split
``sharp-*``    the unsmoothed Fisher score, kept only where the smoothed
               score found text (detail without the noise)
``<best N>``   the N thresholded masks with the best stroke score
               (``threshold.threshold_all``), skipping near-duplicates

All are dark text on white, with letters about ``target_glyph`` px tall
and a white margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .cluster import ClusterResult
from .features import PixelFeatures
from .preserve import gentle_clean, streak_mask, stroke_score, to_ocr


@dataclass
class Variant:
    name: str
    image: np.ndarray  # image sent to OCR
    scale: float             # chip px -> OCR px
    pad: int
    stroke: float | None = None

    def to_chip(self, poly: np.ndarray) -> np.ndarray:
        return (poly - self.pad) / self.scale


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def build(pf: PixelFeatures, cr: ClusterResult, masks: dict[str, np.ndarray],
          scores: dict[str, float], body: np.ndarray, n_best: int = 3,
          target_glyph: float = 40.0, soft: bool = True, saver=None) -> list[Variant]:
    g = pf.glyph_px
    out: list[Variant] = []

    def add(name, img, is_mask, stroke=None):
        im, f, pad = to_ocr(img, g, target_glyph, is_mask=is_mask)
        out.append(Variant(name, im, f, pad, stroke))

    # Reflection streaks (solid, line-length) are painted over with surface colour
    # in the gray variants. The black-and-white masks drop them in gentle_clean.
    streak = streak_mask((cr.score > 1.5) & body, g)
    body_s = body & ~streak
    if soft:
        add("fis-soft", np.where(body_s, np.clip(cr.score, -1.0, 8.0), -1.0), False)
        # Lightly smoothed (1/20 of a letter) so camera grain doesn't dominate.
        # Which way the text goes (lighter/darker) is decided from lightness itself,
        # not the Fisher map: it's the side whose 2.5-sigma cut forms text lines.
        Ls = pf.smooth["lab"][:, :, 0]
        pol = 1 if stroke_score(Ls > 2.5, body, g) >= stroke_score(Ls < -2.5, body, g) else -1
        L = pol * cv2.GaussianBlur(pf.resid["lab"][:, :, 0], (0, 0), max(0.8, g / 20))
        L = L / max(float(np.std(L[body])) if body.any() else 1.0, 1e-3)
        u = np.clip((L + 2.5) * (255 / 8), 0, 255).astype(np.uint8)
        u = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(u).astype(np.float32)
        add("L-flat", np.where(body_s, u, float(np.median(u[body])) if body.any() else 0), False)
    # Two scales: the smoothed score finds WHERE the text is (not fooled by
    # noise), and the unsmoothed one keeps the stroke detail (M vs N, 6 vs G)
    # that smoothing fills in. Sharp pixels are only kept near the smoothed text.
    zone = cv2.dilate((cr.score > 2.5).astype(np.uint8),
                      np.ones((max(3, int(g / 6)) | 1,) * 2, np.uint8)).astype(bool) & body_s
    sharp_s = cv2.GaussianBlur(cr.sharp, (0, 0), max(0.7, g / 30))
    if soft and zone.any():
        add("sharp-soft", np.where(zone, np.clip(sharp_s, -1.0, 6.0), -1.0), False)
    chosen: list[np.ndarray] = []
    sh = gentle_clean((sharp_s > 1.5) & zone, body, g, close=False)
    if sh.sum() > 20:
        add("sharp-zone", sh, True)
        chosen.append(sh)
    for name, m in (("vote2", gentle_clean(cr.votes >= 2, body, g)),
                    ("stage2", gentle_clean(cr.stage2, body, g))):
        if m.sum() > 20:
            add(name, m, True)
            chosen.append(m)
    for name in sorted(masks, key=lambda n: -scores[n]):
        if sum(v.name not in ("fis-soft", "L-flat", "sharp-soft", "sharp-zone", "vote2", "stage2")
               for v in out) >= n_best:
            break
        m = masks[name]
        if m.sum() <= 20 or scores[name] <= 0.02 or any(_iou(m, c) > 0.85 for c in chosen):
            continue
        add(name, m, True, scores[name])
        chosen.append(m)
    if saver is not None and saver.enabled:
        for v in out:
            saver.save(f"7_variant_{v.name}", v.image)
    return out
