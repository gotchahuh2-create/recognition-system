"""Smart camera acquisition — live quality assessment of preview frames.

The phone streams two images per preview step (``vision/camera/web/phone.html``):

* ``frame``  the whole camera view, downscaled (~960 px long side) — board
             detection, alignment, perspective, exposure, glare, motion;
* ``crop``   a NATIVE-resolution cut of the camera stream around the MCU card
             (requested by the server once the card has been found). Fine
             detail — focus, chip glare, how big the marking will be in the
             full-resolution photo — is invisible in a downscaled frame, so it
             is measured here.

Every metric has a level (good / warn / bad / na). ``CaptureGate`` turns the
stream of reports into one decision: capture only when every blocking metric
has been good for several consecutive frames AND the focus is at its recent
peak. The full-resolution photo is checked again (``analyze_photo``) before it
is handed to the recognition pipeline; a bad still triggers a retake.

Why these metrics (what the OCR stages need, see README 0.8 / DECISIONS):

* the chip body must be >= ~250 px in the PHOTO for its marking to be read
  (marking glyphs ~1/18 of the chip side -> ~14 px); this is predicted from
  the native crop and the camera's still resolution;
* specular glare on the epoxy wipes out laser markings even when the rest of
  the board looks fine, so glare is measured on the chip separately;
* blur is measured with a no-reference re-blur metric (Crete et al. 2007):
  it compares the image with a re-blurred copy, so it is roughly independent
  of how much texture the scene has. Separate horizontal / vertical values
  expose motion blur (one direction much worse);
* perspective: a board seen at an angle has unequal opposite sides; the card
  locator and the layout fit tolerate a few degrees, OCR of small text much
  less (glyphs shrink along the tilt).

Thresholds are [ASSUMPTION] starting values checked on the real UNI-DS v8
photos and synthetic blur / glare; tune them with real captures (every capture
is saved with its metrics, see ``server.py``).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..card_locate import CardLocator, order_quad, yellow_mask
from ..card_reader import find_chip
from ..enhance import noise_sigma
from ..preprocess import find_board_quad, glare_mask

# --------------------------------------------------------------------------- #
# thresholds  [ASSUMPTION] — see module docstring
# --------------------------------------------------------------------------- #
#: chip body side in the full-resolution photo below which the marking is not
#: readable (README 0.8 resolution sweep: ~250 px reads, 200 px does not)
CHIP_PX_MIN = 250
#: comfortable target (glyphs ~17 px)
CHIP_PX_GOOD = 300
#: re-blur metric (0 = sharp, 1 = flat). Values on the chip body, native px
BLUR_CHIP_BAD = 0.45      # synthetic card, chip 372 px: sharp 0.25, sigma 1: 0.36, sigma 2: 0.54
BLUR_CHIP_WARN = 0.37
#: same on the downscaled frame (board region). Loose: downscaling hides blur
BLUR_FRAME_BAD = 0.40
#: |blur_h - blur_v| above this on the chip = directional (motion) blur
MOTION_BLUR_ANISO = 0.12  # 5 px streak: 0.11, 9 px: 0.24; isotropic blur < 0.02
#: share of the chip body covered by specular glare
GLARE_CHIP_BAD = 0.02
GLARE_CHIP_WARN = 0.006
#: share of the board covered by specular glare
GLARE_BOARD_BAD = 0.05
GLARE_BOARD_WARN = 0.015
#: exposure on the board region (gray levels)
DARK_P98 = 110          # brightest silkscreen should reach this
CLIP_FRAC_BAD = 0.06    # share of near-white pixels that are not glare blobs
NOISE_WARN = 6.0        # Immerkaer sigma on the preview frame
#: camera motion between preview frames, frame widths per second
MOTION_BAD = 0.06
MOTION_WARN = 0.03
GYRO_BAD = 12.0         # deg/s from the phone's gyroscope
#: perspective: |log(opposite side ratio)| and corner-angle deviation (deg)
KEYSTONE_BAD = 0.08
KEYSTONE_WARN = 0.045
ANGLE_DEV_BAD = 7.0
#: in-plane rotation (deg, modulo 90) of the board in the guide frame
ROTATION_BAD = 12.0
#: board fill of the guide frame (area share) / centre offset (guide sizes)
FILL_MIN = 0.50
FILL_GOOD = 0.65
CENTER_BAD = 0.10
#: frames in a row that must pass + minimum time span before auto-capture
STABLE_FRAMES = 3
STABLE_MS = 450
#: focus must be within this share of the best focus seen recently
FOCUS_PEAK_SHARE = 0.92
FOCUS_WINDOW_S = 4.0

GOOD, WARN, BAD, NA = "good", "warn", "bad", "na"
_RANK = {GOOD: 0, NA: 0, WARN: 1, BAD: 2}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Metric:
    name: str
    label: str                 # short human text, e.g. "Sharpness"
    value: float | None
    level: str                 # good | warn | bad | na
    text: str = ""             # e.g. "0.21 (sharp)"
    blocking: bool = True      # a bad level prevents auto-capture

    def as_dict(self) -> dict:
        v = self.value
        return {"name": self.name, "label": self.label, "level": self.level, "text": self.text,
                "value": None if v is None else round(float(v), 4), "blocking": self.blocking}


@dataclass
class FrameMeta:
    """What the phone tells us with each preview frame."""
    mode: str = "board"                       # "board" | "mcu"
    guide: tuple = (0.06, 0.06, 0.88, 0.88)   # x, y, w, h in frame units (0..1)
    video_w: int = 0                          # native camera stream size
    video_h: int = 0
    photo_w: int = 0                          # full-resolution still size (0 = unknown)
    photo_h: int = 0
    crop_rect: tuple | None = None            # native crop: x, y, w, h in frame units
    crop_scale: float = 1.0                   # crop px / native px (phone may downscale)
    gyro: float | None = None                 # rotation rate, deg/s
    t_ms: float | None = None                 # phone timestamp

    @classmethod
    def from_dict(cls, d: dict) -> "FrameMeta":
        m = cls()
        for k in ("mode", "video_w", "video_h", "photo_w", "photo_h", "crop_scale", "gyro", "t_ms"):
            if d.get(k) is not None:
                setattr(m, k, d[k])
        if d.get("guide") and len(d["guide"]) == 4:
            m.guide = tuple(float(v) for v in d["guide"])
        if d.get("crop_rect") and len(d["crop_rect"]) == 4:
            m.crop_rect = tuple(float(v) for v in d["crop_rect"])
        m.mode = "mcu" if m.mode == "mcu" else "board"
        m.video_w, m.video_h = int(m.video_w or 0), int(m.video_h or 0)
        m.photo_w, m.photo_h = int(m.photo_w or 0), int(m.photo_h or 0)
        m.crop_scale = float(m.crop_scale or 1.0)
        return m

    @property
    def photo_per_video(self) -> float:
        """Full-resolution still px per native stream px (same field of view
        assumed; phones crop stills 4:3 vs video 16:9 — the long side is
        compared, which is what both keep)."""
        if self.photo_w and self.video_w:
            return max(self.photo_w, self.photo_h) / max(self.video_w, self.video_h, 1)
        return 1.0


@dataclass
class FrameReport:
    metrics: list[Metric]
    instruction: str
    hints: list[str]
    board_quad: list | None = None            # normalized 4x2 (frame units)
    card_quad: list | None = None
    chip_quad: list | None = None
    crop_request: list | None = None          # x, y, w, h (frame units) of the next native crop
    focus_point: list | None = None           # x, y (frame units): where to focus / meter
    chip_px_photo: float | None = None        # predicted chip side in the full-res photo
    score: float = 0.0                        # 0..1 overall
    passing: bool = False                     # every blocking metric good/warn
    ms: float = 0.0

    def as_dict(self) -> dict:
        return {"metrics": [m.as_dict() for m in self.metrics], "instruction": self.instruction,
                "hints": self.hints, "board_quad": self.board_quad, "card_quad": self.card_quad,
                "chip_quad": self.chip_quad, "crop_request": self.crop_request,
                "focus_point": self.focus_point,
                "chip_px_photo": None if self.chip_px_photo is None else round(self.chip_px_photo),
                "score": round(self.score, 3), "passing": self.passing, "ms": round(self.ms, 1)}

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)


# --------------------------------------------------------------------------- #
# low-level measures
# --------------------------------------------------------------------------- #
def reblur(gray: np.ndarray, k: int = 9) -> tuple[float, float]:
    """No-reference blur (Crete, Dolmiere, Ladret, Nicolas 2007).

    Returns (horizontal, vertical) in 0..1: 0 = sharp, 1 = the image does
    not change when blurred further. Robust to scene content because it
    measures how much edge energy a strong low-pass REMOVES."""
    f = gray.astype(np.float32)
    if min(f.shape) < k + 2:
        return 1.0, 1.0
    out = []
    for axis, ks in ((1, (k, 1)), (0, (1, k))):          # horizontal, vertical
        b = cv2.blur(f, ks)
        df = np.abs(np.diff(f, axis=axis))
        db = np.abs(np.diff(b, axis=axis))
        s_f = float(df.sum())
        if s_f < 1e-6:
            out.append(1.0)
            continue
        s_v = float(np.maximum(df - db, 0).sum())
        out.append((s_f - s_v) / s_f)
    return out[0], out[1]


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    """Best 4-corner polygon around the largest component of ``mask``."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.045, 0.06, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_quad(ap.reshape(4, 2).astype(np.float64))
    return order_quad(cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64))


def quad_geometry(q: np.ndarray) -> dict:
    """Side lengths, keystone ratios, corner angles, in-plane rotation."""
    q = np.asarray(q, np.float64)
    top, right = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])
    bottom, left = np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0])
    angs = []
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cosv = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
        angs.append(math.degrees(math.acos(float(np.clip(cosv, -1, 1)))))
    rot = math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]))
    rot = (rot + 45) % 90 - 45
    return {"top": top, "bottom": bottom, "left": left, "right": right,
            "tb": math.log(max(top, 1e-6) / max(bottom, 1e-6)),     # >0: top longer
            "lr": math.log(max(left, 1e-6) / max(right, 1e-6)),     # >0: left longer
            "angle_dev": max(abs(a - 90) for a in angs), "rotation": rot,
            "area": float(cv2.contourArea(q.astype(np.float32)))}


def _regular(q: np.ndarray) -> float:
    g = quad_geometry(q)
    return max(0.0, 1 - g["angle_dev"] / 35.0)


# --------------------------------------------------------------------------- #
# board detection
# --------------------------------------------------------------------------- #
@dataclass
class BoardFind:
    quad: np.ndarray            # px in the analysed image
    method: str
    score: float


def detect_board(img: np.ndarray) -> BoardFind | None:
    """Board outline on a preview frame (any size; works at <= 800 px).

    Three proposals, best regular / plausible one wins, agreement adds score:
    ``dark``   the PCB as the largest dark, low-saturation region (MikroE
               boards are black; desks usually lighter);
    ``yellow`` the hull of the yellow silkscreen (8th-gen boards draw a
               yellow line along the outline and around every section);
    ``edges``  ``preprocess.find_board_quad`` — largest convex 4-corner
               contour (green / blue boards on any desk)."""
    h0, w0 = img.shape[:2]
    s = min(1.0, 640 / max(h0, w0))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    h, w = small.shape[:2]
    area = float(h * w)
    props: list[BoardFind] = []

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (7, 7), 0)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((v < min(t, 110)) & (hsv[..., 1] < 140)).astype(np.uint8) * 255
    k = max(5, int(0.03 * max(h, w)) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    q = quad_from_mask(dark)
    if q is not None:
        a = quad_geometry(q)["area"]
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, q.astype(np.int32), 255)
        fill = float((dark[m > 0] > 0).mean()) if (m > 0).any() else 0.0
        if 0.12 * area <= a <= 0.995 * area and fill > 0.7:
            props.append(BoardFind(q, "dark", 0.5 * _regular(q) + 0.5 * fill))

    ym = yellow_mask(small)
    if (ym > 0).mean() > 0.002:
        pts = cv2.findNonZero(ym)
        hull = cv2.convexHull(pts)
        hm = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hm, hull.reshape(-1, 2), 255)
        q = quad_from_mask(hm)
        if q is not None:
            a = quad_geometry(q)["area"]
            # yellow must run along the hull (an outline), not sit in one blob
            edge = cv2.morphologyEx(hm, cv2.MORPH_GRADIENT, np.ones((9, 9), np.uint8))
            cover = float((cv2.dilate(ym, np.ones((9, 9), np.uint8))[edge > 0] > 0).mean())
            if 0.12 * area <= a <= 0.995 * area and cover > 0.35:
                props.append(BoardFind(q, "yellow", 0.5 * _regular(q) + 0.5 * min(1.0, cover / 0.7)))

    q = find_board_quad(small, min_area_frac=0.12)
    if q is not None:
        props.append(BoardFind(order_quad(q), "edges", 0.55 * _regular(q) + 0.25))

    if not props:
        return None
    # agreement: proposals whose quads overlap strongly support each other
    for p in props:
        others = [o for o in props if o is not p and _iou(o.quad, p.quad) > 0.8]
        if others:
            p.score = min(1.0, p.score + 0.15 * len(others))
            p.method += "+" + "+".join(o.method for o in others)
    best = max(props, key=lambda p: p.score)
    best.quad = best.quad / s
    return best


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    u = cv2.contourArea(a32) + cv2.contourArea(b32) - inter
    return float(inter / u) if u > 0 else 0.0


# --------------------------------------------------------------------------- #
# chip (MCU marking) quality
# --------------------------------------------------------------------------- #
@dataclass
class ChipQuality:
    quad: np.ndarray            # in the analysed crop's px
    side_px: float              # chip body side, analysed px
    blur: float                 # max(h, v) re-blur on the inner body
    aniso: float                # |h - v|
    glare: float                # glare share on the body
    contrast: float             # marking contrast (top-hat p99.5 - p50), gray levels
    exposure: float             # median gray of the body


def chip_quality(card_img: np.ndarray) -> ChipQuality | None:
    """Find the MCU body on a card image (native px) and measure its marking
    area: focus, directional blur, specular glare, marking contrast."""
    chips = find_chip(card_img)
    if not chips:
        return None
    q = order_quad(chips[0].quad)
    side = float(np.mean([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]))
    # rectify the inner 80 % of the body (leads / rim excluded)
    c = q.mean(0)
    qi = (c + (q - c) * 0.8).astype(np.float32)
    S = int(max(16, min(side * 0.8, 600)))
    dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]], np.float32)
    body = cv2.warpPerspective(card_img, cv2.getPerspectiveTransform(qi, dst), (S, S),
                               flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    # blur is measured on an AXIS-ALIGNED square inside the body: chips are
    # often soldered at 45°, and rectifying them would turn horizontal camera
    # motion into a diagonal one that both directions see equally
    theta = math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0])
    half = side / 2 / (abs(math.cos(theta)) + abs(math.sin(theta))) * 0.85
    x0, y0 = (c - half).astype(int)
    x1, y1 = (c + half).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    inner = cv2.cvtColor(card_img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)], cv2.COLOR_BGR2GRAY)
    bh, bv = reblur(inner)
    gm = glare_mask(body, v_min=230, s_max=50) if S >= 24 else np.zeros_like(gray)
    glyph = max(3, int(S / 0.8 / 18))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * glyph + 1, 2 * glyph + 1))
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker)
    contrast = float(np.percentile(top, 99.5) - np.percentile(top, 50))
    return ChipQuality(q, side, max(bh, bv), abs(bh - bv), float((gm > 0).mean()),
                       contrast, float(np.median(gray)))


def center_glare(card_img: np.ndarray) -> float:
    """Glare in the centre of a card image, as a share of the expected chip
    area (8th-gen cards: chip ~0.31 of the card side, centred).

    Used when ``find_chip`` finds no body: even a small reflection breaks its
    dark + low-texture test (synthetic card, 372 px chip: a 30 px highlight is
    enough), and then the recognition pipeline cannot isolate the marking
    either — so any glare there is reported as the reason."""
    h, w = card_img.shape[:2]
    c = card_img[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
    if not c.size:
        return 0.0
    chip_area = (0.31 * (h + w) / 2) ** 2
    return float((glare_mask(c, v_min=230, s_max=50) > 0).sum()) / max(chip_area, 1.0)


#: glare share (of the chip area) that is blamed when no chip body is found
GLARE_HIDES_CHIP = 0.003


# --------------------------------------------------------------------------- #
# frame analyser
# --------------------------------------------------------------------------- #
def _lvl(v: float, warn: float, bad: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        return BAD if v >= bad else WARN if v >= warn else GOOD
    return BAD if v <= bad else WARN if v <= warn else GOOD


def _norm(q: np.ndarray, w: int, h: int) -> list:
    return np.round(np.asarray(q, float) / [w, h], 4).tolist()


class FrameAnalyzer:
    """Stateful: keeps the previous frame (motion), the last card position
    (crop requests) and recent focus values (peak tracking)."""

    def __init__(self):
        self.locator = CardLocator()
        self.prev_gray: np.ndarray | None = None
        self.prev_t: float | None = None
        self.card_norm: np.ndarray | None = None     # last card quad, frame units
        self.card_seen_t = 0.0
        self.focus_hist: deque = deque(maxlen=64)    # (t, chip blur)

    # ------------------------------------------------------------------ main
    def analyze(self, frame: np.ndarray, crop: np.ndarray | None, meta: FrameMeta,
                now: float | None = None) -> FrameReport:
        t0 = time.perf_counter()
        now = time.monotonic() if now is None else now
        h, w = frame.shape[:2]
        gx, gy, gw, gh = meta.guide
        guide = np.array([[gx, gy], [gx + gw, gy], [gx + gw, gy + gh], [gx, gy + gh]]) * [w, h]
        mets: list[Metric] = []
        instr: list[tuple[int, str]] = []        # (priority, text): lower = first
        hints: list[str] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------------- board
        board = detect_board(frame) if meta.mode == "board" else None
        region = np.zeros((h, w), np.uint8)
        if board is not None:
            cv2.fillConvexPoly(region, board.quad.astype(np.int32), 255)
        else:
            cv2.fillConvexPoly(region, guide.astype(np.int32), 255)

        if meta.mode == "board":
            self._board_metrics(board, guide, w, h, mets, instr)

        # ---------------------------------------------------------- card
        frames = self.locator.frame_candidates(frame)
        card = None
        if frames:
            if board is not None:     # the card is ON the board
                inside = [f for f in frames if _inside_frac(f.quad, board.quad) > 0.8]
                frames = inside or frames
            card = max(frames, key=lambda f: f.score)
        if card is not None and card.score >= 0.55:
            self.card_norm = card.quad / [w, h]
            self.card_seen_t = now
        elif self.card_norm is not None and now - self.card_seen_t > 1.5:
            self.card_norm = None

        # ---------------------------------------------------------- exposure / glare
        self._photometric(frame, gray, region, mets, instr, hints)

        # ---------------------------------------------------------- motion
        self._motion(gray, meta, now, mets, instr)

        # ---------------------------------------------------------- sharpness (frame)
        rb = reblur(_masked_crop(gray, region))
        fb = max(rb)
        mets.append(Metric("sharp_frame", "Sharpness (view)", fb,
                           BAD if fb >= BLUR_FRAME_BAD else GOOD,
                           f"{fb:.2f}", blocking=False))

        # ---------------------------------------------------------- MCU marking
        rep = FrameReport(mets, "", hints)
        self._mcu(frame, crop, card, meta, guide, w, h, now, mets, instr, hints, rep)

        if board is not None:
            rep.board_quad = _norm(board.quad, w, h)
        if self.card_norm is not None:
            rep.card_quad = np.round(self.card_norm, 4).tolist()

        bad = [m for m in mets if m.blocking and m.level == BAD]
        rep.passing = not bad
        score = 1.0
        for m in mets:
            if m.level == BAD:
                score -= 0.25 if m.blocking else 0.08
            elif m.level == WARN:
                score -= 0.06
        rep.score = max(0.0, score)
        instr.sort(key=lambda x: x[0])
        if instr:
            rep.instruction = instr[0][1]
            rep.hints = [t for _, t in instr[1:4]] + hints
        else:
            rep.instruction = "Hold still"
        self.prev_gray, self.prev_t = cv2.resize(gray, (320, max(1, int(320 * h / w)))), now
        rep.ms = (time.perf_counter() - t0) * 1000
        return rep

    # ---------------------------------------------------------------- parts
    def _board_metrics(self, board, guide, w, h, mets, instr):
        if board is None:
            mets.append(Metric("board", "Board", None, BAD, "not found"))
            instr.append((0, "Point the camera at the board"))
            return
        q = board.quad
        g = quad_geometry(q)
        mets.append(Metric("board", "Board", board.score, GOOD if board.score >= 0.6 else WARN,
                           f"found ({board.method.split('+')[0]})"))
        # cut off at the frame edge?
        margin = 0.008 * max(w, h)
        touch = ((q[:, 0] < margin) | (q[:, 0] > w - margin) |
                 (q[:, 1] < margin) | (q[:, 1] > h - margin))
        gwid, ghei = guide[1, 0] - guide[0, 0], guide[2, 1] - guide[1, 1]
        bx0, by0 = q.min(0)
        bx1, by1 = q.max(0)
        fill = g["area"] / max(gwid * ghei, 1.0)
        over = max(0.0, guide[0, 0] - bx0, bx1 - guide[1, 0], guide[0, 1] - by0, by1 - guide[2, 1])
        over /= max(gwid, ghei)
        if touch.sum() >= 2:
            mets.append(Metric("framing", "Framing", fill, BAD, "board cut off"))
            instr.append((1, "Move the phone back — the board is cut off"))
        elif over > 0.06:
            mets.append(Metric("framing", "Framing", fill, BAD, "outside the frame"))
            instr.append((1, "Move back a little — fit the board inside the frame"))
        else:
            bw_, bh_ = bx1 - bx0, by1 - by0
            board_land, guide_land = bw_ > bh_ * 1.15, gwid > ghei * 1.15
            lvl = GOOD if fill >= FILL_GOOD else WARN if fill >= FILL_MIN else BAD
            mets.append(Metric("framing", "Framing", fill, lvl, f"fills {fill:.0%} of the frame"))
            if lvl == BAD:
                if board_land != guide_land and min(bw_ / gwid, bh_ / ghei) < 0.75 and \
                        max(bw_ / gwid, bh_ / ghei) > 0.85:
                    instr.append((2, "Turn the phone " + ("sideways (landscape)" if board_land
                                                          else "upright (portrait)")))
                else:
                    instr.append((2, "Move closer to the board"))
        # centre
        c = q.mean(0)
        gc = guide.mean(0)
        dx, dy = (c[0] - gc[0]) / gwid, (c[1] - gc[1]) / ghei
        off = math.hypot(dx, dy)
        mets.append(Metric("center", "Centred", off, BAD if off > CENTER_BAD else GOOD,
                           f"offset {off:.0%}"))
        if off > CENTER_BAD:
            # the board appears right of centre -> move the PHONE right
            if abs(dx) >= abs(dy):
                instr.append((3, f"Move the phone {'right' if dx > 0 else 'left'}"))
            else:
                instr.append((3, f"Move the phone {'down' if dy > 0 else 'up'}"))
        # rotation in the image plane
        rot = g["rotation"]
        mets.append(Metric("rotation", "Rotation", rot, BAD if abs(rot) > ROTATION_BAD else GOOD,
                           f"{rot:+.0f}°"))
        if abs(rot) > ROTATION_BAD:
            # image rotated clockwise (positive atan2 with y down) -> turn the
            # phone clockwise to bring the board edge level
            instr.append((4, f"Rotate the phone slightly {'clockwise' if rot > 0 else 'counter-clockwise'}"))
        # perspective (keystone)
        ks = max(abs(g["tb"]), abs(g["lr"]))
        lvl = BAD if ks > KEYSTONE_BAD or g["angle_dev"] > ANGLE_DEV_BAD else \
            WARN if ks > KEYSTONE_WARN else GOOD
        tilt = math.degrees(math.atan(ks * 2.2))      # rough: 35° FOV phone camera
        mets.append(Metric("perspective", "Perspective", ks, lvl, f"~{tilt:.0f}° tilt"))
        if lvl == BAD:
            if abs(g["tb"]) >= abs(g["lr"]):
                # top side shorter = the board's top is farther from the camera
                # = the phone's top edge is too high above the board
                instr.append((5, "Hold the phone parallel: lower its " +
                              ("top" if g["tb"] < 0 else "bottom") + " edge"))
            else:
                instr.append((5, "Hold the phone parallel: lower its " +
                              ("left" if g["lr"] < 0 else "right") + " edge"))

    def _photometric(self, frame, gray, region, mets, instr, hints):
        sel = region > 0
        if sel.sum() < 100:
            sel = np.ones_like(gray, bool)
        vals = gray[sel]
        p98 = float(np.percentile(vals, 98))
        p50 = float(np.median(vals))
        gm = glare_mask(frame) > 0
        gl = float(gm[sel].mean())
        clip = float(((gray >= 250) & ~gm)[sel].mean())
        if p98 < DARK_P98:
            mets.append(Metric("exposure", "Exposure", p98, BAD, "too dark"))
            instr.append((6, "Too dark — add light (avoid the flash)"))
        elif clip > CLIP_FRAC_BAD or p50 > 200:
            mets.append(Metric("exposure", "Exposure", clip, BAD, "overexposed"))
            instr.append((6, "Too bright — reduce the light or move out of direct light"))
        else:
            mets.append(Metric("exposure", "Exposure", p98, GOOD, f"p98 {p98:.0f}"))
        lvl = _lvl(gl, GLARE_BOARD_WARN, GLARE_BOARD_BAD)
        mets.append(Metric("glare", "Glare (board)", gl, lvl, f"{gl:.1%}", blocking=False))
        if lvl == BAD:
            hints.append("Reflections on the board — tilt the phone slightly or move the lamp")
        small = cv2.resize(gray, (480, max(1, int(480 * gray.shape[0] / gray.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        ns = noise_sigma(small)
        mets.append(Metric("noise", "Noise", ns, WARN if ns > NOISE_WARN else GOOD,
                           f"σ {ns:.1f}", blocking=False))
        if ns > NOISE_WARN:
            hints.append("Image is noisy — more light gives a sharper photo")

    def _motion(self, gray, meta, now, mets, instr):
        speed = None
        if self.prev_gray is not None and self.prev_t is not None and now > self.prev_t:
            cur = cv2.resize(gray, self.prev_gray.shape[::-1])
            win = cv2.createHanningWindow(cur.shape[::-1], cv2.CV_32F)
            (sx, sy), resp = cv2.phaseCorrelate(self.prev_gray.astype(np.float32),
                                                cur.astype(np.float32), win)
            dt = max(now - self.prev_t, 1e-3)
            speed = math.hypot(sx, sy) / cur.shape[1] / dt if resp > 0.05 else None
        gyro = meta.gyro
        lvl = GOOD
        txt = []
        if speed is not None:
            lvl = _lvl(speed, MOTION_WARN, MOTION_BAD)
            txt.append(f"{speed * 100:.1f} %/s")
        if gyro is not None:
            txt.append(f"gyro {gyro:.0f}°/s")
            if gyro >= GYRO_BAD:
                lvl = BAD
        mets.append(Metric("motion", "Steadiness", speed if speed is not None else gyro,
                           lvl if (speed is not None or gyro is not None) else NA,
                           ", ".join(txt) or "—"))
        if lvl == BAD:
            instr.append((8, "Hold still"))

    def _mcu(self, frame, crop, card, meta, guide, w, h, now, mets, instr, hints, rep):
        """MCU-marking metrics from the native crop (or the frame as fallback)."""
        need_mcu = meta.mode == "mcu"
        # where to ask the phone for native pixels next
        if self.card_norm is not None:
            q = self.card_norm
            c = q.mean(0)
            r = q.max(0) - q.min(0)
            x0, y0 = c - r * 0.6
            x1, y1 = c + r * 0.6
            rep.crop_request = [round(float(max(0, x0)), 4), round(float(max(0, y0)), 4),
                                round(float(min(1, x1) - max(0, x0)), 4),
                                round(float(min(1, y1) - max(0, y0)), 4)]
        elif need_mcu:     # no card frame: the centre of the guide is the MCU area
            gx, gy, gw, gh = meta.guide
            rep.crop_request = [round(gx + gw * 0.15, 4), round(gy + gh * 0.15, 4),
                                round(gw * 0.7, 4), round(gh * 0.7, 4)]
        if need_mcu and self.card_norm is None:
            hints.append("Centre the MCU card in the square")

        cq, scale_to_photo, origin, sc = None, None, None, 1.0
        if crop is not None and crop.size and meta.crop_rect is not None:
            cx, cy, cw, ch = meta.crop_rect
            # the crop has a margin of board around the card: rectify the card
            # first (find_chip judges "dark" and size relative to a CARD). The
            # card position comes from the frame analysis, mapped into the crop
            cw_ = None
            if self.card_norm is not None:
                qc = (self.card_norm - [cx, cy]) / [cw, ch] * [crop.shape[1], crop.shape[0]]
                if (qc.min(0) > -0.05 * crop.shape[1]).all() and \
                        (qc.max(0) < 1.05 * np.array(crop.shape[1::-1])).all():
                    cw_ = _warp_quad(crop, qc)
            if cw_ is not None:
                cq = chip_quality(cw_[0])
                if cq is not None:
                    cq.quad = _apply(np.linalg.inv(cw_[2]), cq.quad)
                else:
                    g = center_glare(cw_[0])
                    if g >= GLARE_HIDES_CHIP:
                        mets.append(Metric("chip_glare", "Glare on MCU", g, BAD, f"{g:.1%}"))
                        instr.append((7, "Glare on the MCU — tilt the phone a little or move the lamp"))
                        return
            else:
                cq = chip_quality(crop)
            # crop px -> native stream px -> photo px
            scale_to_photo = meta.photo_per_video / max(meta.crop_scale, 1e-6)
            origin, sc = (cx * w, cy * h), (cw * w) / crop.shape[1]
        elif card is not None:
            # no native crop yet: measure on the preview (coarse)
            cc = _warp_quad(frame, card.quad)
            cq = chip_quality(cc[0]) if cc is not None else None
            if cq is not None:
                # preview px -> native px -> photo px
                scale_to_photo = (max(meta.video_w, meta.video_h) / max(w, h) if meta.video_w else 1.0) \
                    * meta.photo_per_video / cc[1]
                cq.quad = _apply(np.linalg.inv(cc[2]), cq.quad)
                origin, sc = (0.0, 0.0), 1.0
                cq.blur = float("nan")        # preview blur says nothing about the marking

        if cq is None and card is not None:
            cc = _warp_quad(frame, card.quad)
            g = center_glare(cc[0]) if cc is not None else 0.0
            if g >= GLARE_HIDES_CHIP:
                mets.append(Metric("chip_glare", "Glare on MCU", g, BAD, f"{g:.1%}"))
                instr.append((7, "Glare on the MCU — tilt the phone a little or move the lamp"))
                return
        if cq is None and not need_mcu and self.card_norm is not None:
            mets.append(Metric("chip", "MCU found", None, WARN, "card seen, chip body not found",
                               blocking=False))
            hints.append("MCU body not isolated — a close-up of the MCU will follow")
            return
        if cq is None:
            lvl = BAD if need_mcu else NA
            mets.append(Metric("chip", "MCU found", None, lvl,
                               "not found" if (need_mcu or self.card_norm is not None) else "—",
                               blocking=need_mcu))
            if need_mcu:
                instr.append((2, "Centre the MCU card in the square"))
            return
        chip_px = cq.side_px * scale_to_photo
        rep.chip_px_photo = chip_px
        qf = cq.quad * sc + np.array(origin)
        rep.chip_quad = _norm(qf, w, h)
        rep.focus_point = [round(float(qf[:, 0].mean() / w), 4), round(float(qf[:, 1].mean() / h), 4)]
        # predicted size of the marking in the full-resolution photo
        lvl = GOOD if chip_px >= CHIP_PX_GOOD else WARN if chip_px >= CHIP_PX_MIN else BAD
        mets.append(Metric("chip_px", "MCU size in photo", chip_px, lvl,
                           f"{chip_px:.0f} px (need {CHIP_PX_MIN})", blocking=need_mcu))
        if lvl == BAD:
            if need_mcu:
                instr.append((2, "Move closer to the MCU"))
            else:
                hints.append(f"MCU will be {chip_px:.0f} px — a close-up of the MCU will follow")
        if math.isnan(cq.blur):
            # the card is in view but its native pixels have not arrived yet:
            # never capture before the marking's focus has been measured
            mets.append(Metric("focus", "MCU focus", None, BAD, "measuring…"))
            instr.append((9, "Focusing on the MCU…"))
        else:
            self.focus_hist.append((now, cq.blur))
            recent = [b for t, b in self.focus_hist if now - t <= FOCUS_WINDOW_S]
            best = min(recent) if recent else cq.blur
            sharp_now, sharp_best = 1 - cq.blur, 1 - best
            near_peak = sharp_now >= FOCUS_PEAK_SHARE * sharp_best
            lvl = _lvl(cq.blur, BLUR_CHIP_WARN, BLUR_CHIP_BAD)
            if lvl != BAD and not near_peak:
                lvl = BAD
            mets.append(Metric("focus", "MCU focus", cq.blur, lvl,
                               f"{cq.blur:.2f}" + ("" if near_peak else " (not at peak)")))
            if lvl == BAD:
                if cq.aniso > MOTION_BLUR_ANISO:
                    instr.append((8, "Hold still — motion blur on the MCU"))
                else:
                    instr.append((9, "Focusing on the MCU… hold steady (tap the MCU to focus)"))
            if cq.aniso > MOTION_BLUR_ANISO:
                mets.append(Metric("motion_blur", "Motion blur (MCU)", cq.aniso, BAD,
                                   f"{cq.aniso:.2f}"))
        lvl = _lvl(cq.glare, GLARE_CHIP_WARN, GLARE_CHIP_BAD)
        mets.append(Metric("chip_glare", "Glare on MCU", cq.glare, lvl, f"{cq.glare:.1%}"))
        if lvl == BAD:
            instr.append((7, "Glare on the MCU — tilt the phone a little or move the lamp"))
        mets.append(Metric("marking", "Marking contrast", cq.contrast,
                           WARN if cq.contrast < 12 else GOOD, f"{cq.contrast:.0f}", blocking=False))
        if cq.contrast < 12 and chip_px >= CHIP_PX_MIN:
            hints.append("Marking is faint — light from the side makes laser marking visible")


# --------------------------------------------------------------------------- #
def _inside_frac(a: np.ndarray, b: np.ndarray) -> float:
    a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    return float(inter / max(cv2.contourArea(a32), 1e-6))


def _masked_crop(gray: np.ndarray, region: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(region)
    if len(xs) < 100:
        return gray
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def _warp_quad(img: np.ndarray, quad: np.ndarray):
    """Rectify ``quad`` at native size. Returns (image, scale vs the image
    px, image -> warped homography) or None."""
    q = order_quad(quad).astype(np.float32)
    W = int(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    H = int(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
    if min(W, H) < 16:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR), 1.0, M


# --------------------------------------------------------------------------- #
# capture decision
# --------------------------------------------------------------------------- #
class CaptureGate:
    """Auto-capture once quality has held for ``STABLE_FRAMES`` frames and
    ``STABLE_MS``; then waits ``cooldown_s`` before it can fire again."""

    def __init__(self, stable_frames: int = STABLE_FRAMES, stable_ms: float = STABLE_MS,
                 cooldown_s: float = 2.5):
        self.stable_frames, self.stable_ms, self.cooldown_s = stable_frames, stable_ms, cooldown_s
        self.run: list[float] = []
        self.blocked_until = 0.0
        self.armed = True

    def reset(self, cooldown: bool = True) -> None:
        self.run.clear()
        if cooldown:
            self.blocked_until = time.monotonic() + self.cooldown_s

    def update(self, rep: FrameReport, now: float | None = None) -> tuple[bool, float]:
        """(fire now?, progress 0..1)."""
        now = time.monotonic() if now is None else now
        if not rep.passing or not self.armed:
            self.run.clear()
            return False, 0.0
        self.run.append(now)
        n = len(self.run)
        span = (now - self.run[0]) * 1000
        prog = min(1.0, min(n / self.stable_frames, span / self.stable_ms if self.stable_ms else 1))
        if now < self.blocked_until:
            return False, min(prog, 0.95)
        if n >= self.stable_frames and span >= self.stable_ms:
            self.run.clear()
            return True, 1.0
        return False, prog


# --------------------------------------------------------------------------- #
# full-resolution photo check
# --------------------------------------------------------------------------- #
@dataclass
class PhotoCheck:
    ok: bool
    reasons: list[str]
    width: int
    height: int
    board_quad: list | None = None        # px in the photo
    board_score: float = 0.0
    card_quad: list | None = None
    chip_px: float | None = None
    chip_blur: float | None = None
    chip_glare: float | None = None
    chip_contrast: float | None = None
    frame_blur: float | None = None
    needs_closeup: bool = False
    ms: float = 0.0

    def as_dict(self) -> dict:
        def r(v, n=3):
            return None if v is None else round(float(v), n)
        return {"ok": self.ok, "reasons": self.reasons, "width": self.width, "height": self.height,
                "board_quad": self.board_quad, "board_score": r(self.board_score),
                "card_quad": self.card_quad, "chip_px": r(self.chip_px, 0),
                "chip_blur": r(self.chip_blur), "chip_glare": r(self.chip_glare, 4),
                "chip_contrast": r(self.chip_contrast, 1), "frame_blur": r(self.frame_blur),
                "needs_closeup": self.needs_closeup, "ms": round(self.ms)}


def analyze_photo(img: np.ndarray, mode: str = "board") -> PhotoCheck:
    """Check the full-resolution still before recognition.

    Retake when the MCU marking area is blurred or covered by glare (both
    modes), when the board is missing (board mode) or the chip is too small
    (MCU mode). In board mode a too-small chip is NOT a retake: the board
    shot is fine, the MCU needs a separate close-up (``needs_closeup``)."""
    t0 = time.perf_counter()
    H, W = img.shape[:2]
    reasons: list[str] = []
    out = PhotoCheck(True, reasons, W, H)
    s = min(1.0, 1600 / max(H, W))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    if mode == "board":
        b = detect_board(small)
        if b is None:
            out.ok = False
            reasons.append("board not found in the photo")
        else:
            out.board_quad = np.round(b.quad / s, 1).tolist()
            out.board_score = b.score
    gs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    out.frame_blur = max(reblur(gs))
    locs = CardLocator().frame_candidates(img)
    if out.board_quad is not None:
        bq = np.array(out.board_quad)
        locs = [l for l in locs if _inside_frac(l.quad, bq) > 0.8] or locs
    cq = None
    if locs:
        loc = max(locs, key=lambda l: l.score)
        out.card_quad = np.round(loc.quad, 1).tolist()
        warped = _warp_quad(img, loc.quad)
        if warped is not None:
            cq = chip_quality(warped[0])
            if cq is None:
                g = center_glare(warped[0])
                if g >= GLARE_HIDES_CHIP:
                    out.ok = False
                    out.chip_glare = g
                    reasons.append("glare on the MCU hides its body / marking")
    elif mode == "mcu":
        # no card frame (legacy card / plain chip): look in the centre
        cx0, cy0 = int(W * 0.15), int(H * 0.15)
        cq = chip_quality(img[cy0:H - cy0, cx0:W - cx0])
    if cq is not None:
        out.chip_px, out.chip_blur, out.chip_glare = cq.side_px, cq.blur, cq.glare
        out.chip_contrast = cq.contrast
        if cq.blur >= BLUR_CHIP_BAD:
            out.ok = False
            reasons.append(f"MCU marking is blurred ({cq.blur:.2f})"
                           + (" — motion blur" if cq.aniso > MOTION_BLUR_ANISO else ""))
        if cq.glare >= GLARE_CHIP_BAD:
            out.ok = False
            reasons.append(f"glare covers {cq.glare:.0%} of the MCU")
        if cq.side_px < CHIP_PX_MIN:
            if mode == "mcu":
                out.ok = False
                reasons.append(f"MCU is {cq.side_px:.0f} px, needs {CHIP_PX_MIN}: move closer")
            else:
                out.needs_closeup = True
    elif mode == "mcu":
        out.ok = False
        reasons.append("MCU not found in the photo")
    else:
        out.needs_closeup = True          # card not found: a close-up will settle it
    out.ms = (time.perf_counter() - t0) * 1000
    return out
