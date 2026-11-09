"""Make fake Click boards to test the image search code.

    python scripts/make_synthetic_clicks.py syn_clicks/ [--n 12] [--photos 3]

Writes
    syn_clicks/catalog_images/MIKROE-90NN/front.jpg   clean board on white
    syn_clicks/photos/MIKROE-90NN_phoneA_lamp_15deg_01.jpg   "phone photos"
    syn_clicks/clicks.csv                              pid,name,size,group

Each board gets a printed name, a component layout (the same every run),
mikroBUS header rows and an S/M/L outline. Two numbered pairs ("Thermo 2/3",
"Relay 4/5") share the same layout, to imitate Clicks that look almost the
same. Like the OCR test images, this only shows that the code runs; real
catalog images and real photos are the real test. The PIDs are made up
(MIKROE-90xx).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.clicks.augment import Augmenter  # noqa: E402

NAMES = ["Thermo 2", "Thermo 3", "Relay 4", "Relay 5", "Accel", "Buzz", "GNSS",
         "LoRa", "Heart Rate", "OLED C", "RTC", "Weather", "Color", "Proximity",
         "Air Quality", "Stepper"]
SIZES = {"S": (286, 254), "M": (429, 254), "L": (572, 254)}
PCB = [(40, 40, 40), (40, 110, 30)]


def draw_board(name: str, layout_seed: int, size: str) -> np.ndarray:
    rng = np.random.default_rng(layout_seed)
    w, h = SIZES[size]
    b = np.zeros((h, w, 3), np.uint8)
    b[:] = PCB[layout_seed % 2]
    # mikroBUS header rows (top and bottom), gold pads
    for y in (14, h - 14):
        for x in np.linspace(40, w - 40, 8):
            cv2.circle(b, (int(x), y), 7, (60, 180, 220), -1)
            cv2.circle(b, (int(x), y), 3, (20, 20, 20), -1)
    # components
    for _ in range(int(rng.integers(4, 9))):
        cw, ch = rng.integers(20, 90), rng.integers(15, 70)
        x, y = rng.integers(10, w - cw - 10), rng.integers(40, h - ch - 40)
        col = [(25, 25, 25), (200, 200, 200), (30, 90, 160), (150, 150, 160)][rng.integers(4)]
        cv2.rectangle(b, (int(x), int(y)), (int(x + cw), int(y + ch)), col, -1)
    # printed name + "click"
    txt = name.upper()
    fs = min(1.4, (w - 40) / (24 * len(txt) + 1))
    cv2.putText(b, txt, (18, h // 2 + 12), cv2.FONT_HERSHEY_DUPLEX, fs, (245, 245, 245), 2,
                cv2.LINE_AA)
    cv2.putText(b, "click", (18, h // 2 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(b, (2, 2), (w - 3, h - 3), (230, 230, 230), 2)
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--photos", type=int, default=3, help="phone shots per Click")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    cat, ph = a.out / "catalog_images", a.out / "photos"
    cat.mkdir(parents=True, exist_ok=True)
    ph.mkdir(parents=True, exist_ok=True)
    aug = Augmenter(seed=a.seed + 1000)
    rows = []
    for i, name in enumerate(NAMES[:a.n]):
        pid = f"MIKROE-90{i:02d}"
        size = "SML"[i % 3] if i >= 4 else "M"
        series = name.rsplit(" ", 1)[0] if name[-1].isdigit() else None
        layout = {"Thermo": 11, "Relay": 22}.get(series, 100 + i)
        board = draw_board(name, layout, size)
        canvas = np.full((board.shape[0] + 80, board.shape[1] + 80, 3), 255, np.uint8)
        canvas[40:-40, 40:-40] = board
        d = cat / pid
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / "front.jpg"), canvas)
        for j in range(a.photos):
            cv2.imwrite(str(ph / f"{pid}_phoneA_lamp_{15 * j:02d}deg_{j + 1:02d}.jpg"), aug(board))
        rows.append({"pid": pid, "name": f"{name} Click", "size": size,
                     "group": series or ""})
    with open(a.out / "clicks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["pid", "name", "size", "group", "aliases"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} Clicks, {len(rows) * a.photos} photos to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
