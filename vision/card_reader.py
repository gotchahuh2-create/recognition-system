"""Read ALL the text on an MCU card that was found and zoomed in.

The input is a ``card_locate.CardCrop`` (straightened and enlarged so the
card's long side is about 1800 px). We make three passes, each at all four
90-degree rotations. The card can sit in the board any way round, and its
labels run in different directions (MikroE prints the mikroBUS 3 column
horizontally and the other four rows vertically).

``page``    The whole card, read like any page of text.
``labels``  Only the light label boxes (white port-pin boxes and yellow
            signal boxes), with everything else blanked out. This leaves
            just the dark text inside them, without traces, pads or the chip
            getting in the way. The pin maps in the dataset come from here.
``chip``    The MCU itself. We find it as the biggest dark, smooth, square
            blob (QFP chips are often soldered at 45 degrees), cut it out of
            the full-resolution crop, enlarge it so the letters are about
            44 px tall, and clean it up (denoise, contrast, thresholds), the
            same way ``vision/refine.py`` does. With ``chip_mode="faint"``
            the experimental faint-marking method (``vision/faint``) is used
            instead, and ``"both"`` runs both and keeps all the results.

All words are returned in ``CardCrop.image`` coordinates. The matching
itself (title, port box, other text, chip marking, PID label) is done by
``McuRecognizer`` / ``CardRecognizer``, the same as for any list of words.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import enhance
from .card_locate import CardCrop, order_quad, yellow_mask
from .ocr import OcrBackend, OcrToken, get_backend, read_text

ROTATIONS = (0, 90, 180, 270)
CHIP_MIN, CHIP_MAX = 0.02, 0.45      # chip area / card area
CHIP_TARGET_SIDE = 900  # chip side after enlarging (text is about 1/18 of it)
CHIP_VARIANTS = ("enhanced", "otsu", "sauvola")


@dataclass
class ChipRegion:
    quad: np.ndarray  # 4x2 corners in CardCrop.image px
    score: float
    image: np.ndarray | None = None  # the enlarged, straightened chip (for debug and the UI)

    def as_dict(self) -> dict:
        return {"quad": np.round(self.quad, 1).tolist(), "score": round(self.score, 3),
                **({"size": [int(self.image.shape[1]), int(self.image.shape[0])]}
                   if self.image is not None else {})}


@dataclass
class CardReading:
    tokens: list[OcrToken]
    chip: ChipRegion | None = None
    passes: dict = field(default_factory=dict)  # pass name -> words read
    timings_ms: dict = field(default_factory=dict)
    faint: dict | None = None  # vision.faint report (when chip_mode is faint or both)

    def text_lines(self, limit: int = 40) -> list[str]:
        from .ocr import group_lines
        out = []
        for ln in group_lines(self.tokens):
            s = " ".join(t.text for t in ln).strip()
            if len(s) >= 2 and s not in out:
                out.append(s)
        return out[:limit]

    def as_dict(self) -> dict:
        return {"passes": self.passes, "chip": self.chip.as_dict() if self.chip else None,
                "text": self.text_lines(), "timings_ms": self.timings_ms,
                **({"faint": self.faint} if self.faint else {})}


def label_image(img: np.ndarray) -> np.ndarray:
    """Return a gray image that keeps only the label boxes (dark text on white).

    Everything else is made white. A label box is any white (bright, not
    colourful) or yellow area. We close small gaps so the letter strokes
    inside a box don't split it apart.
    """
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    white = ((v > 140) & (hsv[:, :, 1] < 100)).astype(np.uint8) * 255
    light = cv2.bitwise_or(white, yellow_mask(img))
    k = max(3, int(0.012 * max(H, W)) | 1)
    filled = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    filled = cv2.erode(filled, np.ones((3, 3), np.uint8))
    blk = max(15, int(0.03 * max(H, W)) | 1)
    at = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 18)
    return np.where(filled > 0, at, 255).astype(np.uint8)


def find_chip(img: np.ndarray) -> list[ChipRegion]:
    """Find dark, smooth, square blobs (MCU chips), best first."""
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(5, int(0.03 * max(H, W)) | 1)
    mu = cv2.blur(g, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mu * mu, 0))
    # "Dark" is relative to this card. In an over-exposed photo (chip at gray
    # level 102, glare over the whole card) nothing would be below a fixed 90.
    dark = max(90.0, float(np.percentile(mu, 50)) + 15.0)
    m = ((mu < dark) & (sd < 18)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k // 2 | 1, k // 2 | 1), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        a = cv2.contourArea(c)
        if not (CHIP_MIN * H * W <= a <= CHIP_MAX * H * W):
            continue
        r = cv2.minAreaRect(c)
        rw, rh = r[1]
        fill = a / max(rw * rh, 1.0)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if fill < 0.75 and aspect <= 1.35:
            # A sharp, readable marking has lots of texture, so in a close-up the text
            # cuts deep notches into the "dark and smooth" mask. (On one close-up with
            # a 319 px chip, the fill was 0.7499, so the chip was missed entirely.)
            # So we also accept the chip when its outline is a clean square and the
            # notches only cover part of it, and score it with the lower fill value.
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            if ha / max(rw * rh, 1.0) >= 0.9 and a / max(ha, 1.0) >= 0.6:
                fill = 0.75
        if fill < 0.75 or aspect > 1.35:
            continue
        # big + square + solid. QFN, BGA and TQFP chips are all square.
        score = 0.4 * min(1.0, a / (0.08 * H * W)) + 0.3 * (fill - 0.75) / 0.25 + \
            0.3 * (1.35 - aspect) / 0.35
        out.append(ChipRegion(order_quad(cv2.boxPoints(r)), float(score)))
    out.sort(key=lambda c: -c.score)
    return out


def _chip_warp(native: np.ndarray, quad_native: np.ndarray, pad: float = 0.04
               ) -> tuple[np.ndarray, np.ndarray]:
    """Cut the chip out of the full-resolution card crop, straighten and enlarge it.

    Returns (image, transform from the full-resolution crop to that image).
    """
    q = order_quad(quad_native)
    c = q.mean(0)
    q = c + (q - c) * (1 + 2 * pad)
    side = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0]), 8.0)
    S = int(np.clip(CHIP_TARGET_SIDE, side, side * 6))
    dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]], np.float32)
    Hc = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(native, Hc, (S, S), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE), Hc


def _map(tokens: list[OcrToken], H: np.ndarray) -> None:
    for t in tokens:
        hom = np.hstack([t.poly, np.ones((len(t.poly), 1))]) @ H.T
        t.poly = hom[:, :2] / hom[:, 2:3]


CHIP_MODES = ("classic", "faint", "both")


def _back(quad: np.ndarray, side: int) -> np.ndarray:
    """Convert chip-image pixels back to crop.image pixels.

    We deliberately keep the axes straight here. Chips are often soldered
    at 45 degrees, and if we mapped the words through that rotation, the
    stacked marking lines ('STM32F411' / 'RET6') would end up diagonal on
    the card and the line grouping would mix them up. So we just scale the
    chip square into the chip's bounding box.
    """
    x0, y0 = quad.min(0)
    x1, y1 = quad.max(0)
    k = min(x1 - x0, y1 - y0) / side
    return np.array([[k, 0, (x0 + x1 - k * side) / 2],
                     [0, k, (y0 + y1 - k * side) / 2], [0, 0, 1.0]])


def read_card(crop: CardCrop, backend: OcrBackend | str = "auto", labels: bool = True,
              chip: bool = True, upscaler: str = "bicubic", chip_mode: str = "classic",
              faint_cfg=None, debug_dir=None, debug_prefix: str = "card") -> CardReading:
    """Read all the text on a found card.

    chip_mode: "classic" (standard clean-up), "faint" (vision.faint) or "both".
    debug_dir: where the faint method saves its step-by-step images.
    """
    if chip_mode not in CHIP_MODES:
        raise ValueError(f"chip_mode must be one of {CHIP_MODES}")
    be = get_backend(backend) if isinstance(backend, str) else backend
    timings, passes = {}, {}
    t0 = time.perf_counter()
    toks = read_text(crop.image, be, rotations=ROTATIONS)
    for t in toks:
        t.variant = "card-" + t.variant
    passes["page"] = len(toks)
    timings["page"] = round((time.perf_counter() - t0) * 1000)

    if labels:
        t1 = time.perf_counter()
        lab = read_text(label_image(crop.image), be, rotations=ROTATIONS, variants=("gray",))
        for t in lab:
            t.variant = "labels"
        toks += lab
        passes["labels"] = len(lab)
        timings["labels"] = round((time.perf_counter() - t1) * 1000)

    chip_reg = None
    faint_rep = None
    if chip:
        t2 = time.perf_counter()
        chips = find_chip(crop.image)
        if chips:
            chip_reg = chips[0]
            qn = chip_reg.quad / crop.zoom  # zoomed px -> original px
            cimg, Hc = _chip_warp(crop.native, qn)
            chip_reg.image = cimg
            back = _back(chip_reg.quad, cimg.shape[0])
            glyph = cimg.shape[0] / 18.0  # guess: a marking line is about 1/18 of the chip
            n = 0
            if chip_mode in ("classic", "both"):
                # enhance_crop enlarges by f and adds a white margin of `pad`
                f = enhance.scale_for_glyph(glyph)
                pad = max(8, int(0.4 * enhance.TARGET_GLYPH_PX))
                for vname, vimg in enhance.enhance_crop(cimg, glyph, upscaler=upscaler,
                                                        variants=CHIP_VARIANTS):
                    ct = read_text(vimg, be, rotations=ROTATIONS, variants=("gray",))
                    for t in ct:
                        t.poly = (t.poly - pad) / f
                        t.variant = f"chip-{vname}"
                    _map(ct, back)
                    toks += ct
                    n += len(ct)
            if chip_mode in ("faint", "both"):
                from .faint import FaintChipReader
                from .faint.debug import StageSaver
                from .faint.region import rectify
                t3 = time.perf_counter()
                fchip = rectify(crop.native, qn, method="card", upscaler=upscaler)
                fr = FaintChipReader(be, faint_cfg).read_chip(
                    fchip, StageSaver(debug_dir, f"{debug_prefix}_faint"))
                fb = _back(chip_reg.quad, fchip.image.shape[0])
                _map(fr.tokens, fb)  # the variants are already named 'chip-faint-*'
                toks += fr.tokens
                n += len(fr.tokens)
                passes["chip_faint"] = len(fr.tokens)
                timings["chip_faint"] = round((time.perf_counter() - t3) * 1000)
                faint_rep = fr.as_dict()
            passes["chip"] = n
        else:
            passes["chip"] = "no chip body found"
        timings["chip"] = round((time.perf_counter() - t2) * 1000)
    timings["total"] = round((time.perf_counter() - t0) * 1000)
    return CardReading(toks, chip_reg, passes, timings, faint_rep)
