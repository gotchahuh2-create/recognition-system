"""Find the MCU card on the photo, cut it out and zoom in.

This runs after the board is identified. Instead of reading the card as a
few hundred pixels inside the whole photo, we cut it out and read it on
its own (see ``vision/card_reader.py``). Three separate methods guess where
the card is. Each guess is a 4-corner shape in original photo pixels:

``frame``   Most cards (146 of the 207 in ``data/mcu_cards.json``) have a
            black board with a yellow frame around the edge. That frame is
            the only closed yellow rectangle on a MikroE board; the board's
            other yellow lines are open outlines or thin dividers. To count
            as a card, a shape must be rectangular, mostly yellow along its
            edge, mostly dark inside, and contain the small yellow pin-label
            boxes. On real UNI-DS v8 photos the card scored 0.91 to 0.98 for
            "rectangular", and the board outline only 0.02.
``layout``  Where the board's stored layout (``data/layouts.json``) says the
            MCU card is, if the layout could be placed on the photo.
``text``    A card title or port box that was read on the photo ("MCU CARD
            for STM32", "SiBRAIN for ...", "16-BIT PORT", older logos). This
            only tells us where the card is, not how big it is, so we guess
            the size from the title's letter height (on 8th-gen cards the
            title letters are about 1/27 of the card's side).

When methods agree, the score goes up. For example a frame that contains
the title, or overlaps the layout area, is almost certainly the card.
``crop_card`` then cuts the chosen shape out of the original photo (using
every pixel) and enlarges it until its long side is ``target_long`` px
(at most 4x, using bicubic scaling or a super-resolution model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from . import enhance
from .ocr import OcrToken
from .textmatch import compact

# Starting values, checked on two UNI-DS v8 photos and synthetic cards.
# Tune them on real photos.
YELLOW_LO = (15, 80, 110)  # HSV, OpenCV hue goes 0..180
YELLOW_HI = (40, 255, 255)
MIN_AREA = 0.003                 # card area / photo area
MAX_AREA = 0.6
MIN_RECT = 0.80  # shape area / its bounding rectangle area
MIN_PERIM = 0.55  # share of the frame's edge that is yellow
ASPECT = (0.45, 2.2)
TITLE_TO_SIDE = 27.0  # card side / title letter height (8th gen)
WORK_SIDE = 1600  # search for the frame at this size or smaller
ACCEPT = 0.55  # below this we report no location and the caller falls back

_TITLE_RE = re.compile(r"(C[A4]RD|S[I1L]BR[A4][I1L]N)(?:[0-9OIL]{0,2})F[O0DQ]R")
_PORT_RE = re.compile(r"(16|8|32)[-]?B[I1L]T[-]?P[O0D]RT")


@dataclass
class CardLocation:
    quad: np.ndarray  # 4x2 corners in photo px: top-left, top-right, bottom-right, bottom-left
    method: str  # "frame", "layout", "text", or several joined with "+"
    score: float
    evidence: list[str] = field(default_factory=list)

    @property
    def size_px(self) -> tuple[int, int]:
        q = self.quad
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        return int(round(w)), int(round(h))

    def as_dict(self) -> dict:
        return {"method": self.method, "score": round(self.score, 3),
                "size_px": list(self.size_px), "quad": np.round(self.quad, 1).tolist(),
                "evidence": self.evidence}


@dataclass
class CardCrop:
    image: np.ndarray  # the card, straightened and zoomed (BGR)
    native: np.ndarray  # the card straightened, at its original size
    H: np.ndarray  # transform from the original photo to ``image``
    zoom: float  # zoomed px per original px
    location: CardLocation

    def to_original(self, pts: np.ndarray) -> np.ndarray:
        return _apply(np.linalg.inv(self.H), pts)

    def as_dict(self) -> dict:
        return {"size": [int(self.image.shape[1]), int(self.image.shape[0])],
                "native_size": [int(self.native.shape[1]), int(self.native.shape[0])],
                "zoom": round(self.zoom, 2), **self.location.as_dict()}


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def order_quad(pts) -> np.ndarray:
    """Sort 4 corners as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, np.float64).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    idx = [np.argmin(s), np.argmin(d), np.argmax(s), np.argmax(d)]
    if len(set(idx)) == 4:
        return p[idx]
    # The usual sum/difference trick fails for a square tilted exactly 45
    # degrees (a diagonally soldered QFP): two corners tie and one gets used
    # twice. Instead, sort by angle around the centre, clockwise from top-left.
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    a = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    k = int(np.argmin(np.abs(np.angle(np.exp(1j * (a + 3 * np.pi / 4))))))
    return np.roll(p, -k, axis=0)


def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    ua = cv2.contourArea(a32) + cv2.contourArea(b32) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _inside(q: np.ndarray, pt) -> bool:
    return cv2.pointPolygonTest(q.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def yellow_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _perimeter_cover(mask: np.ndarray, q: np.ndarray, band: int) -> float:
    """Share of points along the shape's edge that have yellow within ``band`` px."""
    dil = cv2.dilate(mask, np.ones((2 * band + 1, 2 * band + 1), np.uint8))
    h, w = mask.shape
    hits = tot = 0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        n = max(8, int(np.linalg.norm(b - a) / 3))
        for t in np.linspace(0, 1, n, endpoint=False):
            x, y = (a + (b - a) * t).astype(int)
            if 0 <= x < w and 0 <= y < h:
                tot += 1
                hits += dil[y, x] > 0
    return hits / tot if tot else 0.0


def _interior_stats(img: np.ndarray, mask: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Return (dark share, yellow share) inside the middle 80 % of the shape."""
    c = q.mean(0)
    inner = (c + (q - c) * 0.8).astype(np.int32)
    m = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(m, inner, 255)
    n = max(int((m > 0).sum()), 1)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = float(((v < 90) & (m > 0)).sum()) / n
    yel = float(((mask > 0) & (m > 0)).sum()) / n
    return dark, yel


def _snap_to_frame(mask: np.ndarray, q: np.ndarray, min_cov: float = 0.85,
                   max_inset: float = 0.16) -> tuple[np.ndarray, float]:
    """Snap each side of ``q`` onto the card's yellow frame.

    A side snaps to the outer edge of a CONTINUOUS yellow line (covering at
    least ``min_cov`` of the middle 70 % of that side). Label boxes are
    yellow too, but they have gaps, so they don't count.

    Why this is needed: sometimes the frame touches a yellow socket outline
    or a row of labels on the board. They merge into one blob, and the
    shape ends up 10 to 30 px bigger than the card. Two cases:
    * the shape is outside every continuous line: use the first line inward;
    * there is a closed outline around the card: every side then shows two
      lines (outline, gap, frame), so use the inner one.

    Returns (new shape, biggest shift as a share of the card size). If any
    side has no continuous line, ``q`` is returned unchanged.
    """
    H, W = mask.shape
    q = np.asarray(q, np.float64)
    cen = q.mean(0)
    sides = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        size = float(np.linalg.norm(q[(i + 2) % 4] - q[(i + 1) % 4]))
        nrm = cen - (a + b) / 2
        nrm /= max(np.linalg.norm(nrm), 1e-9)
        ts = np.linspace(0.15, 0.85, max(8, int(L * 0.7)))
        ds = np.arange(-3.0, max_inset * size, 1.0)
        cov = []
        for d in ds:
            pts = a + (b - a) * ts[:, None] + nrm * d
            x = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            y = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            cov.append(float((mask[y, x] > 0).mean()))
        cov = np.array(cov)
        # find runs of continuous yellow, split where coverage drops below 0.5
        runs, k = [], 0
        while k < len(ds):
            if cov[k] >= min_cov:
                k0 = k
                # step back out to the line's outer edge (with tilt or blur, coverage
                # only reaches min_cov a few px inside the line)
                while k0 > 0 and cov[k0 - 1] >= 0.5 and k - k0 < 4:
                    k0 -= 1
                while k < len(ds) and cov[k] >= 0.5:
                    k += 1
                runs.append(ds[k0])
            else:
                k += 1
        if not runs:
            return q, 0.0
        sides.append((a, b, L, size, nrm, runs))
    pick = 1 if all(len(r) >= 2 for *_, r in sides) else 0
    lines, shift = [], 0.0
    for a, b, L, size, nrm, runs in sides:
        d = runs[pick]
        shift = max(shift, d / max(size, 1.0))
        lines.append((a + nrm * d, (b - a) / max(L, 1e-9)))

    def meet(l1, l2):
        (p1, d1), (p2, d2) = l1, l2
        t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        return p1 + d1 * t[0]
    try:
        out = np.array([meet(lines[(i - 1) % 4], lines[i]) for i in range(4)])
    except np.linalg.LinAlgError:
        return q, 0.0
    return order_quad(out), shift


class CardLocator:
    def __init__(self, accept: float = ACCEPT):
        self.accept = accept

    # ---- Method 1: the yellow frame ----
    def frame_candidates(self, photo: np.ndarray) -> list[CardLocation]:
        h0, w0 = photo.shape[:2]
        s = min(1.0, WORK_SIDE / max(h0, w0))
        img = cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else photo
        h, w = img.shape[:2]
        mask = yellow_mask(img)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out: list[CardLocation] = []
        for i in range(1, n):
            x, y, bw, bh, npx = st[i]
            if bw * bh < MIN_AREA * h * w or bw * bh > 1.6 * MAX_AREA * h * w or min(bw, bh) < 12:
                continue
            if npx > 0.6 * bw * bh:
                continue  # a solid yellow area, not a frame
            # Turn the blob into a solid shape: close small gaps in the frame, fill
            # everything inside it (by flooding the outside), then erode/dilate with a
            # kernel about 6 % of its size to trim board lines or label rows touching
            # the frame from outside.
            pad = 4
            comp = np.zeros((bh + 2 * pad, bw + 2 * pad), np.uint8)
            comp[pad:pad + bh, pad:pad + bw] = (lab[y:y + bh, x:x + bw] == i) * 255
            k1 = max(3, int(0.03 * max(bw, bh)) | 1)
            comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1)))
            flood = comp.copy()
            cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8),
                          (0, 0), 255)
            solid = cv2.bitwise_or(comp, cv2.bitwise_not(flood))
            filled = int((solid > 0).sum())
            if filled < MIN_AREA * h * w or filled > MAX_AREA * h * w:
                continue
            k2 = max(3, int(0.06 * np.sqrt(filled)) | 1)
            solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2)))
            sub, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not sub:
                continue
            shape = max(sub, key=cv2.contourArea) + np.array([x - pad, y - pad])
            hole = False
            a = cv2.contourArea(shape)
            if not (MIN_AREA * h * w <= a <= MAX_AREA * h * w):
                continue
            rect = cv2.minAreaRect(shape)
            rw, rh = rect[1]
            if min(rw, rh) < 12:
                continue
            rectness = a / max(rw * rh, 1.0)
            aspect = max(rw, rh) / max(min(rw, rh), 1.0)
            if rectness < MIN_RECT or aspect > ASPECT[1]:
                continue
            q = order_quad(cv2.boxPoints(rect))
            q, inset = _snap_to_frame(mask, q)
            if inset > 0.02:
                hole = True  # we report the frame's inner line
                rw, rh = (np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0]))
            band = max(3, int(0.04 * min(rw, rh)))
            perim = _perimeter_cover(mask, q, band)
            if perim < MIN_PERIM:
                continue
            dark, yel = _interior_stats(img, mask, q)
            # The inside should be black with small yellow label boxes. A shape that is
            # mostly yellow (a label box or logo) or empty board doesn't count.
            labels = 1.0 if 0.01 <= yel <= 0.35 else 0.3
            score = (0.35 * min(1.0, perim / 0.9) + 0.25 * min(1.0, (rectness - 0.6) / 0.35)
                     + 0.2 * min(1.0, dark / 0.5) + 0.2 * labels)
            out.append(CardLocation(q / s, "frame", float(score),
                                    [f"yellow frame ({'snapped to the continuous line inside the outline' if hole else 'outer edge'}): "
                                     f"perimeter {perim:.0%}, rectangular {rectness:.2f}, "
                                     f"dark inside {dark:.0%}, labels {yel:.1%}"]))
        # A closed board outline can also pass the frame tests. If one candidate
        # contains another, the outer one is the board (or a section), not the card.
        for a in out:
            if any(b is not a and all(_inside(a.quad, p) for p in b.quad) for b in out):
                a.score *= 0.6
                a.evidence.append("encloses another frame (board outline?)")
        # near-duplicates (outer and inner edge of the same frame): keep the larger one
        out.sort(key=lambda c: -cv2.contourArea(c.quad.astype(np.float32)))
        kept: list[CardLocation] = []
        for c in out:
            dup = next((k for k in kept if _poly_iou(k.quad, c.quad) > 0.6), None)
            if dup is None:
                kept.append(c)
            else:
                dup.score = max(dup.score, c.score)
        return kept

    # ---- Method 3: a title read on the photo ----
    @staticmethod
    def title_tokens(tokens: Sequence[OcrToken]) -> list[OcrToken]:
        out = []
        for t in tokens:
            s = compact(t.text)
            if _TITLE_RE.search(s) or _PORT_RE.search(s) or s in ("MCUCARD", "SIBRAIN", "CARD"):
                out.append(t)
        return out

    def text_candidates(self, tokens: Sequence[OcrToken], to_orig) -> list[CardLocation]:
        """Make a square around each card title we read, sized from its letter height."""
        out = []
        for t in self.title_tokens(tokens):
            poly = to_orig(t.poly)
            c = poly.mean(0)
            e = sorted(np.linalg.norm(poly - np.roll(poly, -1, 0), axis=1))
            side = TITLE_TO_SIDE * max(e[0], 4.0) * 1.3  # +30 % because the title isn't centred
            q = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * side / 2 + c
            out.append(CardLocation(q, "text", 0.45, [f"card text '{t.text}' read at "
                                                       f"({c[0]:.0f}, {c[1]:.0f})"]))
        return out

    # ---- Combine all methods ----
    def locate(self, photo: np.ndarray, tokens: Sequence[OcrToken] = (), to_orig=None,
               layout_quad: np.ndarray | None = None) -> list[CardLocation]:
        """Return every likely card location, best first (empty if none scores ``accept``).

        tokens:      OCR words from the board pass. ``to_orig`` converts their
                     positions to original photo pixels (``Preprocessed.to_original``).
        layout_quad: where the board layout puts the MCU card, in original px.
        """
        to_orig = to_orig or (lambda p: np.asarray(p, float))
        frames = self.frame_candidates(photo)
        titles = [to_orig(t.poly).mean(0) for t in self.title_tokens(tokens)]
        for f in frames:
            n = sum(_inside(f.quad, c) for c in titles)
            if n:
                f.score = min(1.0, f.score + 0.25)
                f.method += "+text"
                f.evidence.append(f"contains {n} card title/port-box read(s)")
            if layout_quad is not None:
                iou = _poly_iou(f.quad, order_quad(layout_quad))
                if iou > 0.3:
                    f.score = min(1.0, f.score + 0.2 * iou / 0.7)
                    f.method += "+layout"
                    f.evidence.append(f"overlaps the layout's MCU-card area (IoU {iou:.2f})")
                elif iou < 0.05 and f.score < 0.9:
                    f.score *= 0.7          # a yellow box somewhere else on the board
                    f.evidence.append("outside the layout's MCU-card area")
        cands = list(frames)
        if layout_quad is not None and not any("layout" in f.method for f in frames):
            cands.append(CardLocation(order_quad(layout_quad), "layout",
                                      0.6 + 0.15 * any(_inside(order_quad(layout_quad), c)
                                                       for c in titles),
                                      ["board layout MCU-card area"]))
        if not any("text" in c.method for c in cands):
            cands += self.text_candidates(tokens, to_orig)
        cands.sort(key=lambda c: -c.score)
        return [c for c in cands if c.score >= self.accept or
                (c.method in ("layout", "text") and c.score >= 0.45)]


def crop_card(photo: np.ndarray, loc: CardLocation, target_long: int = 1800,
              max_zoom: float = 4.0, pad: float = 0.02, upscaler: str = "bicubic",
              max_native: int = 2400) -> CardCrop:
    """Cut ``loc`` out of the original photo, straighten it and zoom in.

    We don't know which way up the card is yet (the reader tries all four
    rotations), so the crop keeps the photo's orientation.
    """
    q = order_quad(loc.quad)
    c = q.mean(0)
    q = c + (q - c) * (1 + 2 * pad)
    w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    s = min(1.0, max_native / max(w, h, 1.0))
    W, Hh = max(16, int(round(w * s))), max(16, int(round(h * s)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]], np.float32)
    Hn = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    native = cv2.warpPerspective(photo, Hn, (W, Hh), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    zoom = float(np.clip(target_long / max(W, Hh), 1.0, max_zoom))
    img = enhance.upscale(native, zoom, upscaler) if zoom > 1.01 else native
    zoom_eff = img.shape[1] / W
    Hz = np.diag([zoom_eff, zoom_eff, 1.0]) @ Hn
    return CardCrop(img, native, Hz, zoom_eff, loc)
