"""Step 4: separate the marking from the chip surface by grouping pixel colours.

A. For each colour space (RGB, HSV, LAB), group the chip's pixels with
   k-means on their smoothed, lighting-corrected values. The biggest group
   is the chip surface. Of the other groups, the ones whose pixels form
   letter-shaped blobs (``preserve.stroke_score``) are taken as text. This
   gives one text mask per colour space.
B. The three spaces vote (0 to 3) on each pixel. A pixel all three agree on
   is almost certainly text; a pixel with one vote is kept as "possible".
C. Fisher discriminant: using the vote as examples (text: 2+ votes,
   surface: 0 votes and away from text), find the direction in the combined
   10-number colour space that best separates text from surface. Projecting
   every pixel onto it gives one "text score" map, normalised against the
   surface. This is the input to the threshold step, and it's where
   combining the spaces pays off: a tiny change in lightness plus a small
   change in gloss can be a strong signal together.
D. A second, local pass: only inside the (slightly grown) "possible text"
   area, split pixels into two groups using the unsmoothed values. Using
   local instead of chip-wide statistics gives sharper strokes and keeps
   thin parts that the smoothing blurred.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import SPACES, PixelFeatures, robust_z
from .preserve import stroke_score

MAX_SAMPLES = 40000


@dataclass
class ClusterResult:
    space_masks: dict[str, np.ndarray]  # step A
    space_labels: dict[str, np.ndarray]  # step A label maps (for debugging)
    votes: np.ndarray  # step B, uint8 0..3
    score: np.ndarray  # step C, normalised score map (high = text)
    stage2: np.ndarray  # step D, bool
    sharp: np.ndarray  # step C on the unsmoothed values (same direction)
    polarity: int  # +1 if the text is lighter than the surface, -1 if darker
    info: dict = field(default_factory=dict)


def _kmeans(X: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Find group centres from a sample of rows, then give every row the label of its nearest
    centre.
    """
    rng = np.random.default_rng(seed)
    S = X if len(X) <= MAX_SAMPLES else X[rng.choice(len(X), MAX_SAMPLES, replace=False)]
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, _, C = cv2.kmeans(S.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
    return d.argmin(1), C


def _space_mask(pf: PixelFeatures, space: str, k: int, body: np.ndarray, g: float
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    F = pf.smooth[space]
    H, W, C = F.shape
    idx = np.flatnonzero(pf.stats_mask.ravel())
    X = F.reshape(-1, C)[idx]
    lab_s, cent = _kmeans(X, k)
    full = np.full(H * W, -1, np.int32)
    # give every chip pixel (including glare) the label of its nearest centre
    bidx = np.flatnonzero(body.ravel())
    d = ((F.reshape(-1, C)[bidx][:, None, :] - cent[None]) ** 2).sum(2)
    full[bidx] = d.argmin(1)
    full = full.reshape(H, W)
    sizes = np.bincount(lab_s, minlength=k)
    surface = int(sizes.argmax())
    scores = {}
    for c in range(k):
        if c == surface:
            continue
        scores[c] = stroke_score(full == c, body, g)
    mask = np.zeros((H, W), bool)
    chosen = []
    if scores:
        best = max(scores.values())
        if best > 0.05:
            # the first channel of every space is a lightness (R / V / L), which tells us
            # the polarity
            bc = max(scores, key=scores.get)
            sign = np.sign(cent[bc][0] - cent[surface][0]) or 1.0
            for c, s in scores.items():
                if s >= 0.5 * best and np.sign(cent[c][0] - cent[surface][0]) == sign:
                    mask |= full == c
                    chosen.append(c)
    return mask, full, {"sizes": sizes.tolist(), "surface": surface, "chosen": chosen,
                        "stroke_scores": {int(c): round(s, 3) for c, s in scores.items()}}


def _fisher(F: np.ndarray, text: np.ndarray, surf: np.ndarray, ridge: float = 1e-2
            ) -> np.ndarray | None:
    if text.sum() < 30 or surf.sum() < 100:
        return None
    Xt, Xs = F[text], F[surf]
    rng = np.random.default_rng(1)
    if len(Xs) > MAX_SAMPLES:
        Xs = Xs[rng.choice(len(Xs), MAX_SAMPLES, replace=False)]
    if len(Xt) > MAX_SAMPLES:
        Xt = Xt[rng.choice(len(Xt), MAX_SAMPLES, replace=False)]
    Sw = np.cov(Xt, rowvar=False) + np.cov(Xs, rowvar=False)
    Sw += ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    w = np.linalg.solve(Sw, Xt.mean(0) - Xs.mean(0))
    n = np.linalg.norm(w)
    return w / n if n > 0 else None


def cluster(pf: PixelFeatures, body: np.ndarray, k: int = 4, saver=None) -> ClusterResult:
    g = pf.glyph_px
    masks, labels, info = {}, {}, {}
    for s in SPACES:
        masks[s], labels[s], info[s] = _space_mask(pf, s, k, body, g)
    votes = sum(m.astype(np.uint8) for m in masks.values()).astype(np.uint8)

    F = pf.stack("smooth")
    L = pf.smooth["lab"][:, :, 0]
    # Possible starting points: the clustering vote, or plain lightness in either
    # direction. Whichever looks most like text (stroke score) is used. With very
    # faint text the per-space groups are mostly noise, while a simple 2.5-sigma
    # lightness cut already shows letters.
    seeds = {"vote2": (votes >= 2), "vote1": (votes >= 1),
             "L+": L > 2.5, "L-": -L > 2.5}
    seed_scores = {n: stroke_score(m & pf.stats_mask, body, g) for n, m in seeds.items()}
    seed_name = max(seed_scores, key=seed_scores.get)
    text_seed = seeds[seed_name] & pf.stats_mask
    near = cv2.dilate(text_seed.astype(np.uint8), np.ones((int(g) | 1,) * 2, np.uint8)) > 0
    surf_seed = pf.stats_mask & ~near
    info["seeds"] = {"used": seed_name, **{k: round(v, 3) for k, v in seed_scores.items()}}
    w = _fisher(F, text_seed, surf_seed)
    ref = surf_seed if surf_seed.sum() > 100 else pf.stats_mask
    if w is None:
        score, sharp = L.copy(), pf.resid["lab"][:, :, 0].copy()
        info["fisher"] = "fallback: lightness"
    else:
        score = (F @ w).astype(np.float32)
        sharp = (pf.stack("resid") @ w).astype(np.float32)
        info["fisher"] = {"w": np.round(w, 3).tolist()}
    score = robust_z(score, ref, floor=0.05)
    sharp = robust_z(sharp, ref, floor=0.05)
    # flip the map if needed so text = high score. The side that forms letters wins.
    sp = stroke_score((score > 2.5) & body, body, g)
    sn = stroke_score((-score > 2.5) & body, body, g)
    if sn > sp:
        score, sharp = -score, -sharp
    # is the text lighter or darker? (needed for the gray variants)
    hi = (score > 2.5) & body
    polarity = 1 if (not hi.any() or float(np.median(L[hi])) >= 0) else -1
    info["orient"] = {"pos": round(sp, 3), "neg": round(sn, 3)}
    score = np.where(body, score, 0).astype(np.float32)

    # step D: local split into two groups inside the possible-text area
    zone = cv2.dilate(((votes >= 1) | (score > 2.0)).astype(np.uint8),
                      np.ones((max(3, int(0.6 * g)) | 1,) * 2, np.uint8)).astype(bool) & body
    stage2 = np.zeros_like(body)
    if zone.sum() > 200:
        R = np.concatenate([pf.resid["lab"], pf.resid["hsv"][:, :, :2]], 2)
        Xz = np.concatenate([R[zone], score[zone][:, None] * 0.5], 1)
        lz, cz = _kmeans(Xz, 2, seed=2)
        tc = int(np.argmax(cz[:, -1]))  # the group with the higher text score is the text
        tmp = np.zeros(zone.sum(), bool)
        tmp[lz == tc] = True
        stage2[zone] = tmp
        info["stage2_frac"] = round(float(tmp.mean()), 3)
    sharp = np.where(body, sharp, 0).astype(np.float32)
    res = ClusterResult(masks, labels, votes, score, stage2, sharp, polarity, info)
    if saver is not None and saver.enabled:
        for s in SPACES:
            lab = labels[s]
            vis = (lab.astype(np.float32) + 1) * (255.0 / (k + 1))
            saver.save(f"4a_{s}_kmeans{k}", cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_JET))
            saver.save(f"4b_{s}_text_cluster", masks[s])
        saver.save("4c_vote_rgb_hsv_lab", (votes * 85).astype(np.uint8))
        saver.save("4d_fisher_score", np.clip(score, -2, 8))
        saver.save("4e_stage2_local_split", stage2)
        saver.save("4f_fisher_sharp", np.clip(sharp, -2, 6))
    return res
