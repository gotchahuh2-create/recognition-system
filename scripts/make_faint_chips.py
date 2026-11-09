"""Make fake close-ups of MCU chips with FAINT laser marking, for eval/faint_eval.py.

Each image has a dark board and a dark QFP chip (slightly grainy) with a
marking from the card dataset (the shop photo markings in
``data/mcu_cards.json``, split into lines as printed). The marking's colour
differs from the chip by only ``delta`` levels. On top of that we add the
things that make real chips hard: worn strokes (random gaps), uneven
lighting and dark corners, a reflection across the text, slight rotation
(every other chip at 45 degrees), blur, camera noise and JPEG.

    python scripts/make_faint_chips.py out_dir --n 8 --levels 30,14,8,5

Writes ``out_dir/*.jpg`` and ``out_dir/labels.csv``
(filename,kind,label,delta,glare,lines).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT_MONO, size)
    except OSError:
        return ImageFont.load_default()


def marking_lines(card: dict) -> list[str]:
    obs = card.get("marking")
    if obs and card.get("markings") and " / " in obs and "..." not in obs:
        ls = [x.strip() for x in obs.split(" / ") if "logo" not in x.lower() and x.strip() != "ARM"]
        return ls[:3]
    mcu = card.get("mcu") or ""
    return [mcu[:9], mcu[9:] + "T6", "GH22B VG"] if len(mcu) > 9 else [mcu, "GH22B VG"]


def pick_cards(n: int, seed: int = 0) -> list[dict]:
    data = json.loads((ROOT / "data" / "mcu_cards.json").read_text())
    seen, out = set(), []
    cards = [c for c in data["cards"] if c.get("mcu") and c.get("markings")]
    random.Random(seed).shuffle(cards)
    for c in cards:
        if c["mcu"] in seen:
            continue
        seen.add(c["mcu"])
        out.append(c)
        if len(out) == n:
            break
    return out


def render(lines: list[str], delta: float, rng: random.Random, glare: bool = True,
           angle: float = 0.0, body_px: int = 300, hi: int = 1400,
           gradient: float = 0.35, noise: float = 4.0, defocus: float = 0.9,
           tint: tuple[float, float, float] = (1.0, 1.0, 0.92)) -> np.ndarray:
    nr = np.random.default_rng(rng.randint(0, 2 ** 31))
    W = H = hi
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (28, 42, 26)  # board colour (BGR)
    for _ in range(25):                                         # traces
        y = rng.randint(0, H)
        cv2.line(img, (0, y), (W, y + rng.randint(-200, 200)), (40, 75, 45), rng.randint(4, 12))
    side = int(hi * 0.62)
    cx = cy = hi // 2
    chip = np.zeros((H, W), np.float32)
    x0, y0 = cx - side // 2, cy - side // 2
    pin = int(side * 0.06)
    for k in range(x0 + 20, x0 + side - 20, max(10, side // 40)):   # pins
        cv2.rectangle(img, (k, y0 - pin), (k + side // 90, y0), (170, 170, 168), -1)
        cv2.rectangle(img, (k, y0 + side), (k + side // 90, y0 + side + pin), (170, 170, 168), -1)
        cv2.rectangle(img, (x0 - pin, k), (x0, k + side // 90), (170, 170, 168), -1)
        cv2.rectangle(img, (x0 + side, k), (x0 + side + pin, k + side // 90), (170, 170, 168), -1)
    chip[y0:y0 + side, x0:x0 + side] = 1
    body_col = np.array([33, 32, 31], np.float32)
    grain = cv2.GaussianBlur(nr.normal(0, 2.0, (H, W)).astype(np.float32), (0, 0), 2)
    for c in range(3):
        img[:, :, c] = np.where(chip > 0, body_col[c] + grain, img[:, :, c])
    # pin-1 dimple
    cv2.circle(img, (x0 + int(side * 0.12), y0 + int(side * 0.86)), int(side * 0.04), (24, 24, 24), -1)
    # marking mask
    mk = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mk)
    fs = int(side * 0.085)
    f = _font(fs)
    for i, ln in enumerate(lines):
        d.text((x0 + int(side * 0.12), y0 + int(side * 0.22) + i * int(side * 0.14)), ln, font=f, fill=255)
    m = np.asarray(mk, np.float32) / 255.0
    wear = cv2.GaussianBlur(nr.uniform(0.35, 1.0, (H // 8, W // 8)).astype(np.float32), (0, 0), 1)
    m *= cv2.resize(wear, (W, H), interpolation=cv2.INTER_CUBIC)
    for c in range(3):
        img[:, :, c] += m * delta * tint[c]
    # rotate the whole chip area around the centre
    if angle:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    # lighting: gradient and dark corners
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    grad = 1 + gradient * (((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)) / (W / 2))
    vig = 1 - 0.25 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (W / 2) ** 2)
    img *= (grad * vig)[:, :, None]
    if glare:
        gx = cx + rng.uniform(-0.2, 0.2) * side
        gy = cy + rng.uniform(-0.25, 0.05) * side
        sx, sy = side * rng.uniform(0.12, 0.22), side * rng.uniform(0.05, 0.1)
        blob = np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2))
        img += (blob * rng.uniform(70, 140))[:, :, None]
    # camera: shrink to the photo's resolution, blur, noise, JPEG
    f_out = body_px / side
    img = cv2.resize(np.clip(img, 0, 255), None, fx=f_out, fy=f_out, interpolation=cv2.INTER_AREA)
    if defocus:
        img = cv2.GaussianBlur(img, (0, 0), defocus)
    img += nr.normal(0, noise, img.shape).astype(np.float32)
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=6, help="chips (distinct MCUs) per level")
    ap.add_argument("--levels", default="30,14,8,5", help="marking contrast in gray levels")
    ap.add_argument("--body-px", type=int, default=300, help="chip body side in the output photo")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cards = pick_cards(a.n, a.seed)
    rows = []
    for delta in [float(x) for x in a.levels.split(",")]:
        for i, c in enumerate(cards):
            rng = random.Random(hash((a.seed, delta, i)) & 0xFFFFFFFF)
            lines = marking_lines(c)
            glare = i % 2 == 1
            img = render(lines, delta, rng, glare=glare, angle=45.0 if i % 3 == 2 else rng.uniform(-4, 4),
                         body_px=a.body_px)
            fn = f"faint_d{int(delta):02d}_{i:02d}_{c['mcu']}.jpg"
            cv2.imwrite(str(a.out / fn), img)
            rows.append({"filename": fn, "kind": "mcu", "label": c["mcu"], "delta": delta,
                         "glare": int(glare), "lines": " / ".join(lines)})
    with open(a.out / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
