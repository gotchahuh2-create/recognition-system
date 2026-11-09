"""Step 5: try many thresholds on the text-vs-surface maps.

The inputs are maps where high = text: the Fisher score from ``cluster``,
and the lighting-corrected lightness (which doesn't depend on the
clustering, in case that went wrong). Each map gets:

global    several fixed levels, in units of the surface's spread (1.5, 2.5,
          4). Low levels keep faint strokes plus some noise, high levels
          only the clearest strokes. The OCR and fusion steps decide which
          works best.
otsu      Otsu's threshold inside the chip.
hyst      hysteresis: strong pixels (above 3.5) are kept, and weak ones
          (above 1.5) are kept if they touch a strong one. This saves
          faint pixels attached to clear ones.
adaptive  local thresholds, for when the contrast varies (inside glare,
          near the edges): Gaussian mean, Niblack, Sauvola and
          Wolf-Jolion, each at two window sizes (1x and 2.5x the letter
          height).

Every mask then goes through ``preserve.gentle_clean`` and is rated with
``preserve.stroke_score`` (shown in the report, and used to pick the
black-and-white variants).
"""

from __future__ import annotations

import cv2
import numpy as np

from .preserve import gentle_clean, hysteresis, stroke_score

GLOBAL_Z = (1.5, 2.5, 4.0)


def _ink(score: np.ndarray, lo: float = -2.0, hi: float = 10.0) -> np.ndarray:
    """Turn a score map (high = text) into a dark-text-on-light float image, 0..255."""
    return (255.0 - np.clip((score - lo) * 255.0 / (hi - lo), 0, 255)).astype(np.float32)


def _local(u: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    m = cv2.boxFilter(u, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(u * u, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    return m, np.sqrt(np.maximum(m2 - m * m, 0))


def adaptive(u: np.ndarray, body: np.ndarray, win: int) -> dict[str, np.ndarray]:
    """Make dark-text masks from ``u`` (dark text on light, float 0..255)."""
    win = max(3, win | 1)
    m, s = _local(u, win)
    out = {}
    out["gauss"] = u < cv2.GaussianBlur(u, (0, 0), win / 4) - 6
    out["niblack"] = u < m - 0.2 * s
    out["sauvola"] = u < m * (1 + 0.2 * (s / 128.0 - 1))
    M = float(u[body].min()) if body.any() else 0.0
    smax = float(s[body].max()) if body.any() else 1.0
    k = 0.5
    out["wolf"] = u < (1 - k) * m + k * M + k * (s / max(smax, 1e-6)) * (m - M)
    # flat areas (plain surface) must not turn into "text" because of a tiny spread
    flat = s < 4.0
    return {n: (v & body & ~flat) for n, v in out.items()}


def masks_for(score: np.ndarray, body: np.ndarray, g: float, tag: str,
              zs=GLOBAL_Z, windows=(1.0, 2.5)) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for z in zs:
        out[f"{tag}-z{z:g}"] = (score > z) & body
    v = np.clip(score[body], -2, 10)
    if v.size:
        u8 = ((np.clip(score, -2, 10) + 2) * (255 / 12)).astype(np.uint8)
        t, _ = cv2.threshold(u8[body].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out[f"{tag}-otsu"] = (u8 > t) & body
    out[f"{tag}-hyst"] = hysteresis(score, 1.5, 3.5, body)
    u = _ink(score)
    for wf in windows:
        for n, m in adaptive(u, body, int(wf * g)).items():
            out[f"{tag}-{n}{wf:g}"] = m
    return out


def threshold_all(score: np.ndarray, lightness: np.ndarray, polarity: int,
                  body: np.ndarray, g: float, saver=None
                  ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Make all the masks (already cleaned up) and their stroke scores."""
    raw = masks_for(score, body, g, "fis")
    raw.update(masks_for(polarity * lightness, body, g, "L", zs=(2.0, 3.0), windows=(2.5,)))
    masks, scores = {}, {}
    for n, m in raw.items():
        c = gentle_clean(m, body, g)
        masks[n], scores[n] = c, round(stroke_score(c, body, g), 4)
    if saver is not None and saver.enabled:
        for n in sorted(masks, key=lambda n: -scores[n])[:12]:
            saver.save(f"5_{n}_s{scores[n]:.2f}", masks[n])
    return masks, scores
