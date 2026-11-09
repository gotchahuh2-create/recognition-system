"""Socket stage (Phase 4, experimental): is each mikroBUS socket empty or occupied?

Geometry comes first: the sockets are the layout's ``mikroBUS N`` areas,
placed on the photo by ``vision.layout`` from the board's silkscreen anchors
(already computed by the pipeline: ``PhotoResult.rois``). Only then is each
area looked at. A socket is never guessed: the answer is three-valued
(``empty`` / ``occupied`` / ``unknown``) with the features that led to it.

heuristic_v0 (hypothesis H-004, thresholds are [ASSUMPTION] until measured on
labelled sockets):

* The board around a socket is dark solder mask with yellow silkscreen. An empty
  socket area shows only that (plus the black header strips).
* A plugged-in Click covers the area with its own PCB and parts, so a large share
  of the area is neither dark board nor yellow silkscreen ("foreign" pixels).
* A Click with a dark PCB can look like the board; then the yellow silkscreen
  share decides, and anything between the thresholds is ``unknown``.

Socket geometry source of truth: ``data/layouts.json`` (``sockets`` boxes per
board). ``data/sockets.csv`` (centre points) is derived from it by
``derive_sockets_csv`` so the two cannot disagree (ADR-0104).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

# [ASSUMPTION] starting values, from one real UNI-DS v8 capture (5 green Clicks).
BOARD_V_MAX = 80          # HSV value below this = dark board / header plastic
BOARD_S_MAX_GRAY = 60     # low saturation + V < BOARD_GRAY_V_MAX = grey board areas
BOARD_GRAY_V_MAX = 120
YELLOW_H = (15, 40)       # OpenCV hue 0..180
YELLOW_S_MIN, YELLOW_V_MIN = 80, 110
OCCUPIED_FOREIGN = 0.30   # foreign share at or above this -> occupied
EMPTY_FOREIGN = 0.08      # foreign share at or below this ...
EMPTY_YELLOW = 0.01       # ... and visible yellow silkscreen -> empty
MIN_VISIBLE = 0.9         # socket area must be (almost) inside the photo
# heuristic_v1 (E-006), fitted on 30 labelled sockets of 6 UNI-DS v8 photos: [ASSUMPTION] until held out
V1_SAT_OCCUPIED = 0.10    # saturated, non-yellow share (a coloured Click PCB) -> occupied
V1_SAT_EMPTY = 0.03
V1_EDGES_OCCUPIED, V1_STD_OCCUPIED = 0.15, 35.0   # busy centre between the headers (parts)
V1_EDGES_EMPTY, V1_STD_EMPTY = 0.11, 32.0         # quiet centre: bare solder mask
WORK_SIDE = 400           # analyse the rectified crop at this long side


class Occupancy(str, Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


@dataclass
class SocketResult:
    slot: str
    occupancy: Occupancy
    score: float                  # confidence in the occupancy answer, 0..1
    reason: str
    features: dict = field(default_factory=dict)
    click: dict | None = None     # Click recognition, only when occupied (needs a gallery)

    def as_dict(self) -> dict:
        return {"slot": self.slot, "occupancy": self.occupancy.value,
                "score": round(self.score, 3), "reason": self.reason,
                "features": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in self.features.items()},
                **({"click": self.click} if self.click else {})}


def pixel_shares(img: np.ndarray) -> dict:
    """Share of board-like, yellow and foreign pixels in a socket crop (BGR)."""
    s = WORK_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    board = (v < BOARD_V_MAX) | ((sat < BOARD_S_MAX_GRAY) & (v < BOARD_GRAY_V_MAX))
    yellow = (h >= YELLOW_H[0]) & (h <= YELLOW_H[1]) & (sat >= YELLOW_S_MIN) & (v >= YELLOW_V_MIN)
    foreign = ~(board | yellow)
    # edge density: Click parts and text are busy; bare solder mask is not
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 60, 160)
    return {"board": float(board.mean()), "yellow": float(yellow.mean()),
            "foreign": float(foreign.mean()), "edges": float((edges > 0).mean())}


def v1_features(img: np.ndarray) -> dict:
    s = WORK_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow_hue = (h >= 12) & (h <= 40)
    sat_ny = (sat >= 90) & (v >= 60) & ~yellow_hue
    yellow = (h >= YELLOW_H[0]) & (h <= YELLOW_H[1]) & (sat >= YELLOW_S_MIN) & (v >= YELLOW_V_MIN)
    H, W = img.shape[:2]
    c = cv2.cvtColor(img[int(H * .62):int(H * .92), int(W * .3):int(W * .7)], cv2.COLOR_BGR2GRAY)
    return {"sat_nonyellow": float(sat_ny.mean()), "yellow": float(yellow.mean()),
            "centre_edges": float((cv2.Canny(c, 60, 160) > 0).mean()), "centre_std": float(c.std())}


def classify_v1(img: np.ndarray, visible: float = 1.0) -> tuple[Occupancy, float, str, dict]:
    """heuristic_v1: colour of a Click PCB OR a busy centre -> occupied; bare and quiet -> empty."""
    if visible < MIN_VISIBLE:
        return (Occupancy.UNKNOWN, 0.0,
                f"socket area only {visible:.0%} inside the photo", {"visible": visible})
    f = v1_features(img)
    f["visible"] = visible
    busy = f["centre_edges"] >= V1_EDGES_OCCUPIED and f["centre_std"] >= V1_STD_OCCUPIED
    if f["sat_nonyellow"] >= V1_SAT_OCCUPIED or busy:
        why = (f"{f['sat_nonyellow']:.0%} coloured PCB" if f["sat_nonyellow"] >= V1_SAT_OCCUPIED
               else f"parts between the headers (edges {f['centre_edges']:.2f})")
        return Occupancy.OCCUPIED, 0.9, why, f
    quiet = f["centre_edges"] <= V1_EDGES_EMPTY and f["centre_std"] <= V1_STD_EMPTY
    if f["sat_nonyellow"] <= V1_SAT_EMPTY and quiet and f["yellow"] >= EMPTY_YELLOW:
        return Occupancy.EMPTY, 0.8, "bare headers: no coloured PCB, quiet centre, silkscreen visible", f
    return Occupancy.UNKNOWN, 0.0, "neither clearly empty nor clearly occupied (dark Click? shadow?)", f


def classify(img: np.ndarray, visible: float = 1.0) -> tuple[Occupancy, float, str, dict]:
    if visible < MIN_VISIBLE:
        return (Occupancy.UNKNOWN, 0.0,
                f"socket area only {visible:.0%} inside the photo", {"visible": visible})
    f = pixel_shares(img)
    f["visible"] = visible
    if f["foreign"] >= OCCUPIED_FOREIGN:
        conf = min(1.0, 0.6 + (f["foreign"] - OCCUPIED_FOREIGN))
        return Occupancy.OCCUPIED, conf, f"{f['foreign']:.0%} of the area is not board", f
    if f["foreign"] <= EMPTY_FOREIGN and f["yellow"] >= EMPTY_YELLOW:
        conf = min(1.0, 0.6 + (EMPTY_FOREIGN - f["foreign"]) * 4)
        return (Occupancy.EMPTY, conf,
                f"only board colours ({f['foreign']:.0%} other), silkscreen visible", f)
    return (Occupancy.UNKNOWN, 0.0,
            f"between thresholds: {f['foreign']:.0%} not board, {f['yellow']:.1%} silkscreen "
            "(a dark Click or shadow?)", f)


def sockets_from_result(result, impl: str = "heuristic_v0") -> list[SocketResult]:
    """Occupancy for every mikroBUS area the pipeline already cropped (PhotoResult.rois)."""
    fn = classify_v1 if impl == "heuristic_v1" else classify
    out = []
    for roi in getattr(result, "rois", []) or []:
        if not roi.name.lower().startswith("mikrobus"):
            continue
        occ, score, reason, feats = fn(roi.image, roi.visible)
        out.append(SocketResult(roi.name, occ, score, reason, feats))
    return out


def derive_sockets_csv(layouts_path: Path, out_path: Path) -> int:
    """Write data/sockets.csv (board,slot,u,v centres) from layouts.json socket boxes."""
    import json
    data = json.loads(Path(layouts_path).read_text(encoding="utf-8"))
    rows = []
    for key, lay in data.items():
        if key.startswith("_") or not isinstance(lay, dict):
            continue
        for s in lay.get("sockets", []):
            x0, y0, x1, y1 = s["box"]
            rows.append({"board": key, "slot": s["slot"],
                         "u": round((x0 + x1) / 2, 4), "v": round((y0 + y1) / 2, 4)})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["board", "slot", "u", "v"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)
