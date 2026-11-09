"""Board layouts: where the MCU card and the mikroBUS sockets are on each board.

Recognition happens in two steps:

  1. We identify the board by reading its printed name (board_ocr).
  2. The board's stored layout tells us where the MCU card and the mikroBUS
     sockets are. The MCU step then reads ONLY the MCU card area
     (straightened, at the photo's full resolution) instead of the whole
     photo with all its unrelated text. The Click step (DINOv2) gets one
     crop per socket, already labelled with its slot name.

Board coordinates
-----------------
Every layout is stored relative to the board: ``u`` goes left to right and
``v`` top to bottom, both from 0 to 1 across the board outline, with the
board upright (its name reads left to right). ``aspect`` is the board's
width / height.

Placing the layout on a photo (``locate``)
------------------------------------------
When calibrating a board, we save every unique piece of text that OCR
reads on it ("mikroBUS 1", "USB UART", "UNI-DS v8", ...) along with its
position. These are called *anchors*. The photo has already been read by
the board step, so placing the layout costs no extra OCR: words that match
an anchor give us matching points, and a RANSAC fit (a homography if at
least 6 points agree, otherwise a simpler similarity fit with at least 3)
maps the board onto the photo. We don't store anchors inside the MCU card
or socket areas, because that text belongs to whatever card or Click is
plugged in, not to the board.

If there are too few anchors: the board outline (the 4 corners the user
clicked, or ``find_board_quad``) gives the corners, and whatever anchors
were found tell us which corner is which (the board can be rotated by any
multiple of 90 degrees).

Layouts are saved in ``data/layouts.json`` (see ``python -m vision.layout --help``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .ocr import OcrBackend, OcrToken, get_backend, read_text
from .preprocess import (Preprocessed, PreprocessConfig, find_board_quad, preprocess,
                         resize_long_side)
from .textmatch import compact, ocr_ratio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LAYOUTS_FILE = DATA_DIR / "layouts.json"

ISO = 1000.0  # board size used while fitting (same units on both axes)
MIN_ANCHOR_CONF = 0.5
FUZZY_MIN = 0.9  # fuzzy match ratio for anchors of 6+ characters
MCU_PAD = 0.06  # grow the MCU card area by 6 % per side, to allow for placement error
SOCKET_PAD = 0.04


# ---- Data ----
@dataclass
class Anchor:
    text: str  # as read during calibration
    u: float
    v: float
    h: float = 0.0  # letter height, relative to the board (v units)
    seen: int = 1  # how many calibration photos agreed

    @property
    def key(self) -> str:
        return compact(self.text)


@dataclass
class Socket:
    slot: str                # "mikroBUS 1"
    box: list[float]         # u0, v0, u1, v1


@dataclass
class BoardLayout:
    board: str  # board key (from boards.csv)
    aspect: float  # width / height, with the board upright
    mcu: list[float] | None = None  # MCU card area: u0, v0, u1, v1
    sockets: list[Socket] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- JSON ----
    @classmethod
    def from_dict(cls, key: str, d: dict) -> "BoardLayout":
        return cls(key, float(d["aspect"]), d.get("mcu"),
                   [Socket(s["slot"], [float(x) for x in s["box"]]) for s in d.get("sockets", [])],
                   [Anchor(a["text"], float(a["u"]), float(a["v"]), float(a.get("h", 0)),
                           int(a.get("seen", 1))) for a in d.get("anchors", [])],
                   d.get("meta", {}))

    def as_dict(self) -> dict:
        return {"aspect": round(self.aspect, 4),
                "mcu": [round(x, 4) for x in self.mcu] if self.mcu else None,
                "sockets": [{"slot": s.slot, "box": [round(x, 4) for x in s.box]}
                            for s in self.sockets],
                "anchors": [{"text": a.text, "u": round(a.u, 4), "v": round(a.v, 4),
                             "h": round(a.h, 4), "seen": a.seen} for a in self.anchors],
                "meta": self.meta}

    # ---- Helpers ----
    def regions(self) -> list[tuple[str, list[float]]]:
        out = [("MCU card", self.mcu)] if self.mcu else []
        return out + [(s.slot, s.box) for s in self.sockets]

    def anchor_index(self) -> dict[str, Anchor]:
        return {a.key: a for a in self.anchors}


class Layouts(dict):
    """All board layouts (board key -> BoardLayout), loaded from and saved to a JSON file."""

    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Layouts":
        p = Path(path) if path else LAYOUTS_FILE
        out = cls()
        out.path = p
        if p.is_file():
            raw = json.loads(p.read_text("utf-8"))
            for k, v in raw.items():
                if not k.startswith("_"):
                    out[k] = BoardLayout.from_dict(k, v)
        return out

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.path or LAYOUTS_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = {"_doc": "Board layouts for two-stage recognition, board-relative "
                        "0..1 coordinates with the board upright (name reads left->right). "
                        "Edit with: python -m vision.layout --help"}
        body.update({k: v.as_dict() for k, v in sorted(self.items())})
        p.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", "utf-8")
        return p


# ---- Geometry helpers ----
def box_quad(box: Sequence[float], pad: float = 0.0) -> np.ndarray:
    """Turn a box (u0, v0, u1, v1) into 4 corners: top-left, top-right, bottom-right,
    bottom-left.

    ``pad`` grows the box on each side by that fraction of its size.
    """
    u0, v0, u1, v1 = box
    du, dv = (u1 - u0) * pad, (v1 - v0) * pad
    u0, v0, u1, v1 = u0 - du, v0 - dv, u1 + du, v1 + dv
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def _iso(layout: BoardLayout) -> np.ndarray:
    """Convert board uv coordinates to the frame used for fitting."""
    return np.diag([layout.aspect * ISO, ISO, 1.0])


def _plausible(H: np.ndarray, img_shape, min_frac: float = 0.02) -> bool:
    """Check that the board corners land on a sensible shape in the photo.

    It must be convex, not mirrored, and not tiny. (It may go past the
    edge of the photo.)
    """
    q = apply_h(H, box_quad([0, 0, 1, 1]))
    if not np.all(np.isfinite(q)):
        return False
    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return False
    # signed area > 0 means same winding as the board, so not a mirror image
    x, y = q[:, 0], q[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    area_img = img_shape[0] * img_shape[1]
    return signed > min_frac * area_img and signed < 25 * area_img


# ---- Anchors from OCR words ----
def anchor_ok(text: str) -> bool:
    k = compact(text)
    return len(k) >= 3 and any(c.isalpha() for c in k)


def observed_points(tokens: Iterable[OcrToken], min_conf: float = MIN_ANCHOR_CONF
                    ) -> dict[str, tuple[np.ndarray, float, OcrToken]]:
    """Collect unique texts as (centre, letter height, word).

    The same text read several times in the same place is merged. The same
    text found in two different places is ambiguous, so we drop it.
    """
    seen: dict[str, list[tuple[np.ndarray, float, OcrToken]]] = {}
    for t in tokens:
        if t.conf < min_conf or not anchor_ok(t.text):
            continue
        seen.setdefault(compact(t.text), []).append((np.array(t.center), t.height, t))
    out = {}
    for k, items in seen.items():
        c0, h0, _ = items[0]
        if all(np.linalg.norm(c - c0) < 1.5 * max(h0, h, 4.0) for c, h, _ in items):
            best = max(items, key=lambda it: it[2].conf)
            out[k] = (np.mean([c for c, _, _ in items], 0), best[1], best[2])
    return out


def match_anchors(layout: BoardLayout, tokens: Iterable[OcrToken]
                  ) -> list[tuple[Anchor, np.ndarray, OcrToken]]:
    """Return (anchor, centre where it was seen, word) for every anchor found on the photo.

    Exact text matches first. Long anchors may also match fuzzily (allowing
    for OCR mistakes) if the match is unique.
    """
    idx = layout.anchor_index()
    obs = observed_points(tokens)
    pairs, used = [], set()
    for k, (c, _, t) in obs.items():
        if k in idx:
            pairs.append((idx[k], c, t))
            used.add(k)
    for k, (c, _, t) in obs.items():
        if k in idx or len(k) < 6:
            continue
        cands = [a for ak, a in idx.items() if ak not in used and len(ak) >= 6
                 and ocr_ratio(ak, k) >= FUZZY_MIN]
        if len(cands) == 1:
            pairs.append((cands[0], c, t))
            used.add(cands[0].key)
    return pairs


# ---- Placing the layout ----
@dataclass
class Placement:
    """A board layout placed on a photo. ``H`` maps board uv to processed photo pixels."""
    H: np.ndarray
    method: str  # "anchors-homography", "anchors-similarity" or "quad+anchors"
    matched: int  # how many anchors were found in the photo
    inliers: int
    rms_px: float
    anchors_used: list[str] = field(default_factory=list)

    def to_pre(self, uv: np.ndarray) -> np.ndarray:
        return apply_h(self.H, uv)

    def to_original(self, pre: Preprocessed, uv: np.ndarray) -> np.ndarray:
        return pre.to_original(self.to_pre(uv))

    def as_dict(self) -> dict:
        return {"method": self.method, "anchors_matched": self.matched,
                "inliers": self.inliers, "rms_px": round(self.rms_px, 1),
                "anchors_used": self.anchors_used[:20]}


def _fit(src: np.ndarray, dst: np.ndarray, thr: float, kind: str):
    if kind == "homography":
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr, maxIters=4000,
                                     confidence=0.999)
    else:
        A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=thr, maxIters=4000,
                                              confidence=0.999)
        H = None if A is None else np.vstack([A, [0, 0, 1]])
    if H is None or mask is None:
        return None, None
    return H, mask.ravel().astype(bool)


def locate(layout: BoardLayout, tokens: Sequence[OcrToken], pre: Preprocessed,
           quad_pre: np.ndarray | None = None, photo: np.ndarray | None = None
           ) -> Placement | None:
    """Place ``layout`` on the processed photo.

    Returns None when that can't be done reliably. The caller then falls
    back to reading the whole photo.

    tokens:   OCR words in processed-image coordinates (from the board step).
    quad_pre: the board outline in processed coordinates, corners in any
              order (from the 4 corners the user clicked in the UI).
    photo:    the original photo, used to look for the board outline when
              the anchors alone aren't enough.
    """
    pairs = match_anchors(layout, tokens)
    h_img, w_img = pre.image.shape[:2]
    thr = max(6.0, 0.012 * max(h_img, w_img))
    T = _iso(layout)
    best: Placement | None = None
    if len(pairs) >= 3:
        src = np.array([[a.u, a.v] for a, _, _ in pairs]) @ T[:2, :2].T
        dst = np.array([c for _, c, _ in pairs], float)
        for kind, need in (("homography", 6), ("similarity", 3)):
            if len(pairs) < need:
                continue
            Hi, inl = _fit(src.astype(np.float32), dst.astype(np.float32), thr, kind)
            if Hi is None or inl.sum() < need:
                continue
            H = Hi @ T
            if not _plausible(H, pre.image.shape):
                continue
            err = np.linalg.norm(apply_h(Hi, src[inl]) - dst[inl], axis=1)
            best = Placement(H, f"anchors-{kind}", len(pairs), int(inl.sum()),
                             float(np.sqrt(np.mean(err ** 2))),
                             [pairs[i][0].text for i in np.flatnonzero(inl)])
            break
    if best is not None:
        return best

    # ---- Fallback: we know the 4 corners, the anchors tell us which is which ----
    if quad_pre is None and photo is not None:
        q = find_board_quad(photo)
        if q is not None:
            quad_pre = apply_h(pre.M, q)
    if quad_pre is None or not pairs:
        return None
    from .preprocess import order_quad
    # top-left, top-right, bottom-right, bottom-left in the image
    q = order_quad(quad_pre).astype(np.float64)
    board = box_quad([0, 0, 1, 1])
    uv = np.array([[a.u, a.v] for a, _, _ in pairs])
    obs = np.array([c for _, c, _ in pairs], float)
    cands = []
    for k in range(4):  # board rotated by k * 90 degrees
        H = cv2.getPerspectiveTransform(board.astype(np.float32),
                                        np.roll(q, -k, axis=0).astype(np.float32))
        err = np.linalg.norm(apply_h(H, uv) - obs, axis=1)
        cands.append((float(np.median(err)), k, H, err))
    cands.sort(key=lambda c: c[0])
    med, _, H, err = cands[0]
    diag = float(np.hypot(*np.ptp(q, axis=0)))
    # must be clearly better than the other rotations and close to the anchors
    if med > 0.06 * diag or (len(cands) > 1 and cands[1][0] < 1.5 * med):
        return None
    inl = err < 0.06 * diag
    return Placement(H, "quad+anchors", len(pairs), int(inl.sum()),
                     float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else med,
                     [pairs[i][0].text for i in np.flatnonzero(inl)])


# ---- Cropping areas of the board ----
@dataclass
class RoiCrop:
    name: str  # "MCU card" or "mikroBUS 1"
    image: np.ndarray  # straightened crop, board upright, original resolution
    H: np.ndarray  # transform from the original photo to this crop
    quad_orig: np.ndarray          # 4x2 corners in the original photo
    quad_pre: np.ndarray           # 4x2 corners in the processed photo
    visible: float  # how much of the area is inside the photo

    def as_dict(self) -> dict:
        return {"name": self.name, "size": [int(self.image.shape[1]), int(self.image.shape[0])],
                "visible": round(self.visible, 3),
                "quad_orig": np.round(self.quad_orig, 1).tolist(),
                "quad_pre": np.round(self.quad_pre, 1).tolist()}


def crop_roi(photo: np.ndarray, pre: Preprocessed, pl: Placement, name: str,
             box: Sequence[float], pad: float = 0.0, max_side: int = 2400) -> RoiCrop:
    """Cut a board area out of the ORIGINAL photo (using every pixel), straightened and board
    upright.
    """
    uv = box_quad(box, pad)
    q_pre = pl.to_pre(uv)
    q_org = pre.to_original(q_pre)
    w = max(np.linalg.norm(q_org[1] - q_org[0]), np.linalg.norm(q_org[2] - q_org[3]))
    h = max(np.linalg.norm(q_org[3] - q_org[0]), np.linalg.norm(q_org[2] - q_org[1]))
    s = min(1.0, max_side / max(w, h, 1.0))  # never enlarge here; preprocessing does that
    w, h = max(8, int(round(w * s))), max(8, int(round(h * s)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    Hc = cv2.getPerspectiveTransform(q_org.astype(np.float32), dst)
    img = cv2.warpPerspective(photo, Hc, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ph, pw = photo.shape[:2]
    inside = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), Hc, (w, h),
                                 flags=cv2.INTER_NEAREST)
    return RoiCrop(name, img, Hc, q_org, q_pre, float((inside > 0).mean()))


# ---- Calibration ----
def rectify_board(photo: np.ndarray, corners: Sequence[Sequence[float]],
                  long_side: int = 2000) -> tuple[np.ndarray, np.ndarray, float]:
    """Warp the board to an upright rectangle.

    ``corners`` are the board's top-left, top-right, bottom-right and
    bottom-left corners (with the board upright, so its name reads
    normally), in photo pixels. Returns (image, transform, aspect).
    """
    q = np.asarray(corners, np.float64).reshape(4, 2)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    aspect = w / h
    s = long_side / max(w, h)
    W, Hh = int(round(w * s)), int(round(h * s))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]], np.float32)
    H = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(photo, H, (W, Hh), flags=cv2.INTER_CUBIC), H, aspect


def _inside_any(u: float, v: float, boxes: Iterable[Sequence[float]], pad: float = 0.01) -> bool:
    return any(b[0] - pad <= u <= b[2] + pad and b[1] - pad <= v <= b[3] + pad for b in boxes)


def anchors_from_board_image(board_img: np.ndarray, backend: OcrBackend | str = "auto",
                             exclude: Iterable[Sequence[float]] = ()) -> list[Anchor]:
    """Run OCR on an upright, straightened board image; each unique text becomes an anchor."""
    be = get_backend(backend) if isinstance(backend, str) else backend
    pre = preprocess(board_img, PreprocessConfig(long_side=max(board_img.shape[:2]),
                                                 deskew=False, upscale_small=False))
    toks = read_text(pre.image, be, rotations=(0, 90, 180, 270))
    H, W = board_img.shape[:2]
    exclude = list(exclude)
    out = []
    for k, (c, hgt, t) in observed_points(toks).items():
        p = pre.to_original(c[None])[0]
        u, v = p[0] / W, p[1] / H
        if not (0 <= u <= 1 and 0 <= v <= 1) or _inside_any(u, v, exclude):
            continue
        out.append(Anchor(t.text.strip(), float(u), float(v), float(hgt / pre.scale / H)))
    return sorted(out, key=lambda a: (round(a.v, 2), a.u))


def anchors_from_photo(photo: np.ndarray, H_rect: np.ndarray, rect_size: tuple[int, int],
                       backend: OcrBackend | str = "auto",
                       exclude: Iterable[Sequence[float]] = ()) -> list[Anchor]:
    """Collect anchors the same way the pipeline will read them at run time.

    The photo is preprocessed and read exactly as the board step does it,
    and each word is mapped onto the board using the calibration corners.

    Why not just read the straightened board image? Because it's at a
    different scale and angle, OCR splits, joins or misreads labels
    differently there ("CODECRIP" vs "CODEGRIP", "UNI-DSv8" vs "UNI-DS v8"),
    and anchors that only exist in that version never match later. Reading
    the photo itself gives exactly the text the pipeline will see. (With
    only the straightened image, test photos matched just 3-5 anchors.)
    """
    be = get_backend(backend) if isinstance(backend, str) else backend
    pre = preprocess(photo)
    toks = read_text(pre.image, be, rotations=(0, 90, 180, 270))
    W, Hh = rect_size
    exclude = list(exclude)
    out = []
    for k, (c, hgt, t) in observed_points(toks).items():
        p = pre.to_original(c[None])
        r = apply_h(H_rect, p)[0]
        u, v = r[0] / W, r[1] / Hh
        if not (0 <= u <= 1 and 0 <= v <= 1) or _inside_any(u, v, exclude):
            continue
        # letter height: convert original px to straightened px (using the local scale)
        sc = float(np.linalg.norm(apply_h(H_rect, p + [0, 1]) - r))
        out.append(Anchor(t.text.strip(), float(u), float(v),
                          float(hgt / pre.scale * sc / Hh)))
    return out


def merge_anchors(old: list[Anchor], new: list[Anchor], tol: float = 0.03) -> list[Anchor]:
    """Combine anchors from two calibration photos.

    The same text near the same place is averaged and counted. The same
    text far apart is ambiguous, so it's dropped.
    """
    by = {a.key: a for a in old}
    drop = set()
    for a in new:
        b = by.get(a.key)
        if b is None:
            by[a.key] = a
        elif np.hypot(a.u - b.u, a.v - b.v) < tol:
            n = b.seen
            b.u, b.v = (b.u * n + a.u) / (n + 1), (b.v * n + a.v) / (n + 1)
            b.seen += 1
        else:
            drop.add(a.key)
    return [a for k, a in by.items() if k not in drop]


def calibrate(photo: np.ndarray, board: str, corners, mcu=None, sockets=(),
              backend: OcrBackend | str = "auto", existing: BoardLayout | None = None,
              source: str = "") -> BoardLayout:
    """Build a layout (or add to an existing one) from one photo where the board corners are
    known.

    ``mcu`` / ``sockets`` boxes are relative to the board: u0, v0, u1, v1.
    """
    rect, H_rect, aspect = rectify_board(photo, corners)
    boxes = ([mcu] if mcu else []) + [b for _, b in sockets]
    if existing is not None:
        boxes += [b for _, b in existing.regions()]
    anchors = anchors_from_board_image(rect, backend, exclude=boxes)
    # also add the text the board step reads from the photo itself (see anchors_from_photo)
    anchors = merge_anchors(anchors, anchors_from_photo(
        photo, H_rect, (rect.shape[1], rect.shape[0]), backend, exclude=boxes))
    if existing is None:
        lay = BoardLayout(board, aspect, list(mcu) if mcu else None,
                          [Socket(s, list(b)) for s, b in sockets], anchors,
                          {"calibrated_from": [source], "date": str(_dt.date.today())})
    else:
        lay = existing
        lay.aspect = (lay.aspect * len(lay.meta.get("calibrated_from", [1])) + aspect) / \
            (len(lay.meta.get("calibrated_from", [1])) + 1)
        if mcu:
            lay.mcu = list(mcu)
        for s, b in sockets:
            lay.sockets = [x for x in lay.sockets if x.slot != s] + [Socket(s, list(b))]
        lay.anchors = merge_anchors(lay.anchors, anchors)
        lay.meta.setdefault("calibrated_from", []).append(source)
        lay.meta["date"] = str(_dt.date.today())
    return lay


# ---- Drawing (for debugging and the "show" command) ----
def draw(img: np.ndarray, pre: Preprocessed, pl: Placement, layout: BoardLayout,
         matched: Sequence[tuple[Anchor, np.ndarray, OcrToken]] = ()) -> np.ndarray:
    """Draw the board outline, the areas and the anchors on the ORIGINAL photo."""
    out = img.copy()
    lw = max(2, int(max(img.shape[:2]) / 600))
    fs = max(img.shape[:2]) / 1800

    def poly(uv, color, label=None):
        p = pl.to_original(pre, uv).astype(np.int32)
        cv2.polylines(out, [p.reshape(-1, 1, 2)], True, color, lw)
        if label:
            org = tuple(int(x) for x in p.min(0) + [0, -lw * 2])
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw * 3)
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, lw)

    poly(box_quad([0, 0, 1, 1]), (0, 220, 0), layout.board)
    if layout.mcu:
        poly(box_quad(layout.mcu, MCU_PAD), (0, 170, 255), "MCU card")
    for s in layout.sockets:
        poly(box_quad(s.box, SOCKET_PAD), (255, 200, 0), s.slot)
    for a, c, _ in matched:
        p = pre.to_original(np.array([c]))[0].astype(int)
        q = pl.to_original(pre, np.array([[a.u, a.v]]))[0].astype(int)
        cv2.circle(out, tuple(p), lw * 3, (255, 0, 255), -1)
        cv2.line(out, tuple(p), tuple(q), (255, 0, 255), lw)
    return out


# ---- Command line ----
def _floats(s: str, n: int) -> list[float]:
    v = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
    if len(v) != n:
        raise argparse.ArgumentTypeError(f"expected {n} numbers, got {s!r}")
    return v


def _to_uv(box: list[float], units: str, size: tuple[int, int]) -> list[float]:
    """Convert a box to u0, v0, u1, v1.

    Input is x, y, w, h in straightened-image px (units='rect') or already
    u0, v0, u1, v1 (units='uv').
    """
    if units == "uv":
        return box
    W, H = size
    x, y, w, h = box
    return [x / W, y / H, (x + w) / W, (y + h) / H]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m vision.layout",
        description="Board layouts for two-stage recognition.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Workflow for a new board (one good, roughly top-down photo is enough; a second
photo adds anchors):

 1. python -m vision.layout rectify photo.jpg --corners TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy
        corners of the board in the photo, in board-upright order (as the board
        name reads). Writes photo_board.jpg: the board straightened.
        Without --corners the board outline is searched automatically.
 2. Open photo_board.jpg in any image viewer and measure x,y,w,h (pixels) of
    the MCU card area and of every mikroBUS socket (the whole Click footprint).
 3. python -m vision.layout calibrate photo.jpg --board uni-ds-v8 --corners ... \\
        --mcu x,y,w,h --socket "mikroBUS 1=x,y,w,h" --socket "mikroBUS 2=..."
    (boxes in photo_board.jpg pixels; --units uv for board-relative 0..1)
 4. python -m vision.layout show other_photo.jpg   -> other_photo_layout.jpg
""")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rectify", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("photo")
        p.add_argument("--corners", type=lambda s: _floats(s, 8),
                       help="TL,TR,BR,BL board corners in photo px (board upright)")
        p.add_argument("--long-side", type=int, default=2000)
    c = sub.choices["calibrate"]
    c.add_argument("--board", required=True, help="board key from boards.csv")
    c.add_argument("--mcu", type=lambda s: _floats(s, 4))
    c.add_argument("--socket", action="append", default=[], help='"mikroBUS 1=x,y,w,h"')
    c.add_argument("--units", choices=["rect", "uv"], default="rect")
    c.add_argument("--add", action="store_true", help="extend the existing layout")
    c.add_argument("--backend", default="auto")
    c.add_argument("--layouts", default=None, help=f"default {LAYOUTS_FILE}")
    s = sub.add_parser("show")
    s.add_argument("photos", nargs="+")
    s.add_argument("--backend", default="auto")
    s.add_argument("--layouts", default=None)
    s.add_argument("--out", type=Path, default=None, help="folder (default: next to photo)")
    i = sub.add_parser("info")
    i.add_argument("--layouts", default=None)
    a = ap.parse_args(argv)

    from .loader import load_photo
    if a.cmd == "info":
        lays = Layouts.load(a.layouts)
        for k, l in lays.items():
            print(f"{k}: aspect {l.aspect:.3f}, mcu {'yes' if l.mcu else 'no'}, "
                  f"{len(l.sockets)} sockets, {len(l.anchors)} anchors, "
                  f"from {l.meta.get('calibrated_from')}")
        return 0
    if a.cmd in ("rectify", "calibrate"):
        ph = load_photo(a.photo)
        corners = a.corners
        if corners is None:
            q = find_board_quad(ph.image)
            if q is None:
                print("board outline not found; pass --corners", file=sys.stderr)
                return 1
            corners = q.ravel().tolist()
            print(f"auto corners (check the orientation!): {','.join(f'{v:.0f}' for v in corners)}")
        rect, _, aspect = rectify_board(ph.image, np.reshape(corners, (4, 2)), a.long_side)
        if a.cmd == "rectify":
            out = Path(a.photo).with_name(Path(a.photo).stem + "_board.jpg")
            cv2.imwrite(str(out), rect)
            print(f"{out}  {rect.shape[1]}x{rect.shape[0]} px, aspect {aspect:.3f}")
            return 0
        size = (rect.shape[1], rect.shape[0])
        mcu = _to_uv(a.mcu, a.units, size) if a.mcu else None
        socks = []
        for spec in a.socket:
            slot, _, box = spec.partition("=")
            socks.append((slot.strip(), _to_uv(_floats(box, 4), a.units, size)))
        lays = Layouts.load(a.layouts)
        lay = calibrate(ph.image, a.board, np.reshape(corners, (4, 2)), mcu, socks, a.backend,
                        lays.get(a.board) if a.add else None, Path(a.photo).name)
        lays[a.board] = lay
        p = lays.save()
        print(f"{p}: {a.board} aspect {lay.aspect:.3f}, {len(lay.anchors)} anchors, "
              f"mcu={lay.mcu}, sockets={[s.slot for s in lay.sockets]}")
        return 0
    if a.cmd == "show":
        from .board_ocr import BoardRecognizer
        lays = Layouts.load(a.layouts)
        be = get_backend(a.backend)
        rec = BoardRecognizer(backend=be)
        for path in a.photos:
            ph = load_photo(path)
            pre = preprocess(ph.image)
            board, toks = rec.recognise(pre)
            lay = lays.get(board.decision.key) if board.decision else None
            if lay is None:
                print(f"{path}: board {board.decision.name if board.decision else '?'} "
                      f"({board.reason}); no layout")
                continue
            pl = locate(lay, toks, pre, photo=ph.image)
            if pl is None:
                print(f"{path}: {lay.board}: layout could not be placed "
                      f"({len(match_anchors(lay, toks))} anchors found)")
                continue
            img = draw(ph.image, pre, pl, lay, match_anchors(lay, toks))
            folder = a.out or Path(path).parent
            folder.mkdir(parents=True, exist_ok=True)
            out = folder / (Path(path).stem + "_layout.jpg")
            cv2.imwrite(str(out), resize_long_side(img, 2400, allow_upscale=False)[0])
            print(f"{path}: {lay.board} placed by {pl.method}, {pl.inliers}/{pl.matched} "
                  f"anchors, rms {pl.rms_px:.1f}px -> {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
