"""Draw fake 8th-gen MCU cards from the card dataset, plugged into a board-like scene.

Used to test the whole card path: find the card -> crop and zoom -> read
all its text -> match.

The card is drawn from ``data/mcu_cards.json``: its group's title and port
box, every mikroBUS signal/pin label from its pin map (white pin boxes,
yellow signal boxes; MB1/MB2 along the top and MB4/MB5 along the bottom as
vertical text, MB3 as a horizontal column, like the real cards), and a chip
with the card's shop photo marking (or its part number), optionally at 45
degrees. The scene adds board text and port labels around it as
distractions.

This does NOT replace real photos. It shows that the code runs and catches
regressions. Results on real photos are what count.

    python scripts/make_synthetic_cards.py out_dir            # 12 scenes + labels.csv
    python scripts/make_synthetic_cards.py out_dir --n 40 --seed 3
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
sys.path.insert(0, str(ROOT / "scripts"))
from make_synthetic import degrade  # noqa: E402

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_COND = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SIGNALS = ["AN", "RST", "CS", "SCK", "MISO", "MOSI", "PWM", "INT", "RX", "TX", "SCL", "SDA"]
YELLOW = (240, 200, 20)
WHITE = (238, 238, 236)
BLACK = (18, 18, 20)


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _text_img(text: str, font, fg, bg, pad=(6, 3)) -> Image.Image:
    l, t, r, b = font.getbbox(text)
    im = Image.new("RGB", (r - l + 2 * pad[0], b - t + 2 * pad[1]), bg)
    ImageDraw.Draw(im).text((pad[0] - l, pad[1] - t), text, font=font, fill=fg)
    return im


def _box(text: str, font, w: int, h: int, fg, bg, rot: int = 0) -> Image.Image:
    """Draw a filled label box with centred text. rot=90 makes the text read bottom-up."""
    inner = _text_img(text, font, fg, bg, pad=(2, 1))
    bw, bh = (w, h) if rot == 0 else (h, w)
    box = Image.new("RGB", (bw, bh), bg)
    s = min((bw - 4) / inner.width, (bh - 2) / inner.height, 1.0)
    if s < 1.0:
        inner = inner.resize((max(1, int(inner.width * s)), max(1, int(inner.height * s))))
    box.paste(inner, ((bw - inner.width) // 2, (bh - inner.height) // 2))
    return box.rotate(rot, expand=True) if rot else box


def render_card(group: dict, card: dict, side: int = 1600, chip45: bool = True,
                rng: random.Random | None = None) -> np.ndarray:
    """Draw one 8th-gen card, upright, ``side`` px square (the real cards are about square)."""
    rng = rng or random.Random(0)
    W = H = side
    im = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(im)
    fw = max(4, side // 90)
    d.rounded_rectangle((fw, fw, W - fw, H - fw), side // 40, outline=YELLOW, width=fw)
    text = group.get("text") or {}
    pin_map = text.get("pin_map") or {}
    port_pins = text.get("port_pins") or group.get("pins") or []

    def pins_for(mb: str) -> list[str]:
        m = pin_map.get(mb)
        if m:
            return [m.get(s, "") for s in SIGNALS]
        if mb == "MIKROBUS 1" and group.get("pins"):
            p = list(group["pins"]) + [""] * 12
            return p[:12]
        pool = port_pins or ["PA0", "PA1", "PB0", "PB1"]
        return [rng.choice(pool) for _ in SIGNALS]

    f_lab = _font(FONT_COND, int(side * 0.022))
    f_hdr = _font(FONT_BOLD, int(side * 0.02))
    bw, bh = int(side * 0.033), int(side * 0.085)      # one vertical label box
    # -- top rows: MB1 (left), MB2 (right); bottom rows: MB4 (left)... as on the real card
    rows = [("MIKROBUS 1", 0.06, True, False), ("MIKROBUS 2", 0.5, True, True),
            ("MIKROBUS 5", 0.06, False, False), ("MIKROBUS 4", 0.5, False, True)]
    for mb, x0f, top, rev in rows:
        pins = pins_for(mb)
        x = int(x0f * W)
        order = list(range(12))[::-1]
        for k, i in enumerate(order):
            xx = x + k * (bw + 3)
            ys, yp = (int(0.045 * H), int(0.045 * H) + bh + 4) if top else \
                (H - int(0.045 * H) - bh, H - int(0.045 * H) - 2 * bh - 4)
            im.paste(_box(SIGNALS[i], f_lab, bw, bh, BLACK, YELLOW, rot=90), (xx, ys))
            if pins[i]:
                im.paste(_box(pins[i], f_lab, bw, bh, BLACK, WHITE, rot=90), (xx, yp))
        hdr = _text_img(mb, f_hdr, YELLOW, BLACK).rotate(90 if not rev else 270, expand=True)
        im.paste(hdr, (x - hdr.width - 6 if not rev else x + 12 * (bw + 3) + 6,
                       int(0.05 * H) if top else H - int(0.05 * H) - hdr.height))
    # -- MB3: a horizontal column on the right
    pins = pins_for("MIKROBUS 3")
    pw, sw, rh = int(side * 0.085), int(side * 0.08), int(side * 0.036)
    x_p, y0 = W - int(0.05 * W) - pw - sw - 4, int(0.3 * H)
    hdr = _text_img("MIKROBUS 3", f_hdr, YELLOW, BLACK)
    im.paste(hdr, (x_p, y0 - hdr.height - 8))
    for i, s in enumerate(SIGNALS):
        y = y0 + i * (rh + 3)
        if pins[i]:
            im.paste(_box(pins[i], f_lab, pw, rh, BLACK, WHITE), (x_p, y))
        im.paste(_box(s, f_lab, sw, rh, BLACK, YELLOW), (x_p + pw + 4, y))
    # -- title and port box, vertical, on the left (reads bottom-up)
    title = group["title"]
    t_img = _text_img(title, _font(FONT_BOLD, int(side * 0.05)), WHITE, BLACK)
    port = group.get("port_width") or "16-BIT PORT"
    p_img = _text_img(port, _font(FONT_BOLD, int(side * 0.026)), BLACK, WHITE, pad=(8, 4))
    t_img, p_img = t_img.rotate(90, expand=True), p_img.rotate(90, expand=True)
    tx = int(0.07 * W)
    ty = int(0.62 * H) - t_img.height // 2
    if t_img.height > 0.5 * H:
        s = 0.5 * H / t_img.height
        t_img = t_img.resize((int(t_img.width * s), int(t_img.height * s)))
        ty = int(0.62 * H) - t_img.height // 2
    im.paste(t_img, (tx, ty))
    im.paste(p_img, (tx + t_img.width + 8, ty + t_img.height - p_img.height))
    # -- small components as clutter
    f_small = _font(FONT_COND, int(side * 0.014))
    for _ in range(40):
        x, y = rng.randint(int(0.22 * W), int(0.72 * W)), rng.randint(int(0.2 * H), int(0.78 * H))
        d.text((x, y), rng.choice(["C", "R", "U", "Y", "FB"]) + str(rng.randint(1, 40)),
               font=f_small, fill=(200, 200, 200))
    # -- the chip (marking drawn on its own layer, then rotated with the chip)
    cs = int(side * 0.34)
    chip = Image.new("RGBA", (cs + 2 * int(cs * 0.08),) * 2, (0, 0, 0, 0))
    cd = ImageDraw.Draw(chip)
    m = int(cs * 0.08)
    cd.rectangle((0, 0, chip.width - 1, chip.height - 1), fill=(165, 165, 160, 255))
    for k in range(0, chip.width, max(3, cs // 60)):  # row of pins
        cd.line((k, 0, k, m - 2), fill=(30, 30, 30, 255))
        cd.line((k, chip.height - m + 1, k, chip.height), fill=(30, 30, 30, 255))
        cd.line((0, k, m - 2, k), fill=(30, 30, 30, 255))
        cd.line((chip.width - m + 1, k, chip.width, k), fill=(30, 30, 30, 255))
    cd.rectangle((m, m, m + cs, m + cs), fill=(30, 30, 33, 255))
    lines = card.get("_marking_lines") or [card.get("mcu") or "", "GH22B VG", "CHN 406"]
    fm = _font(FONT_MONO, int(cs * 0.075))
    for i, ln in enumerate(lines):
        cd.text((m + int(cs * 0.1), m + int(cs * 0.22) + i * int(cs * 0.12)), ln, font=fm,
                fill=(175, 175, 172, 255))
    cd.ellipse((m + int(cs * 0.08), m + int(cs * 0.8), m + int(cs * 0.15), m + int(cs * 0.87)),
               fill=(55, 55, 58, 255))
    if chip45:
        chip = chip.rotate(45, expand=True, resample=Image.BICUBIC)
    cx, cy = int(0.47 * W), int(0.5 * H)
    im.paste(chip, (cx - chip.width // 2, cy - chip.height // 2), chip)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def marking_lines(card: dict) -> list[str]:
    """Split the shop photo marking into lines the way it's printed ('STM32F405' / 'RGT6')."""
    obs = card.get("marking")
    # only use markings the import kept (WRONG_CHIP photos show a different chip,
    # e.g. the STM32F401RB listing reuses the STM32F205RB photo)
    if obs and card.get("markings") and " / " in obs and "..." not in obs:
        ls = [x for x in obs.split(" / ") if "logo" not in x.lower() and x.strip() != "ARM"]
        return ls[:3]
    mcu = card.get("mcu") or ""
    return [mcu[:9], mcu[9:] + "T6", "GH22B VG"] if len(mcu) > 9 else [mcu, "GH22B VG"]


def render_scene(card_img: np.ndarray, board_title: str = "UNI-DS v8", rot90: int = 0,
                 card_frac: float = 0.3, size=(3000, 2250), seed: int = 0) -> np.ndarray:
    """Draw a board-like background (black, yellow section lines, port labels).

    The card is plugged in at ``card_frac`` of the board's width.
    """
    rng = random.Random(seed)
    W, H = size
    im = Image.new("RGB", (W, H), (190, 190, 186))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((60, 60, W - 60, H - 60), 40, fill=(22, 22, 24), outline=YELLOW, width=8)
    for _ in range(10):                                           # open yellow section lines
        x, y = rng.randint(100, W - 400), rng.randint(100, H - 400)
        d.line((x, y, x + rng.randint(200, 600), y), fill=YELLOW, width=3)
        d.line((x, y, x, y + rng.randint(200, 500)), fill=YELLOW, width=3)
    f = _font(FONT_BOLD, int(H * 0.035))
    d.text((int(W * 0.05), int(H * 0.88)), board_title, font=f, fill=WHITE)
    fs = _font(FONT_COND, int(H * 0.014))
    for i in range(24):  # board port labels (distractions)
        d.text((int(W * 0.62) + (i % 6) * 110, int(H * 0.15) + (i // 6) * 60),
               f"P{'ABCD'[i // 6]}{i % 8}", font=fs, fill=WHITE)
    for k, t in enumerate(["USB-UART", "POWER", "mikroBUS 1", "mikroBUS 2", "CODEGRIP"]):
        d.text((int(W * 0.62), int(H * 0.5) + k * 70), t, font=fs, fill=WHITE)
    cw = int(card_frac * W)
    card = cv2.resize(card_img, (cw, cw), interpolation=cv2.INTER_AREA)
    card = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    if rot90:
        card = card.rotate(90 * rot90, expand=True)
    im.paste(card, (int(W * 0.08), int(H * 0.12)))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def pick_cards(data: dict, n: int, seed: int) -> list[dict]:
    """Pick cards with a chip, preferring 8th-gen groups that have pin maps / markings."""
    groups = data["groups"]
    cards = [c for c in data["cards"] if not c["empty"] and c["mcu"]
             and (groups[c["group"]].get("family"))]
    rng = random.Random(seed)
    rich = [c for c in cards if (groups[c["group"]].get("text") or {}).get("pin_map")]
    rest = [c for c in cards if c not in rich]
    rng.shuffle(rich)
    rng.shuffle(rest)
    return (rich + rest)[:n]


# distance presets: how big the card is in the final photo (for a ~3000 px photo)
PRESETS = {
    "close": dict(card_frac=0.55, out_long=2600, blur=0.6),
    "mid": dict(card_frac=0.3, out_long=3000, blur=0.8),
    "far": dict(card_frac=0.16, out_long=3000, blur=0.8),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--presets", default="close,mid")
    a = ap.parse_args(argv)
    data = json.loads((ROOT / "data" / "mcu_cards.json").read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(pick_cards(data, a.n, a.seed)):
        g = data["groups"][c["group"]]
        c = dict(c, _marking_lines=marking_lines(c))
        card = render_card(g, c, rng=random.Random(i), chip45=i % 2 == 0)
        for p in a.presets.split(","):
            pr = PRESETS[p]
            scene = render_scene(card, rot90=i % 4, card_frac=pr["card_frac"], seed=i)
            img = degrade(scene, seed=i, tilt=(i % 5) - 2, persp=0.02, blur=pr["blur"],
                          out_long=pr["out_long"], noise=3, jpeg=90)
            name = f"card_{p}_{c['key']}.jpg"
            cv2.imwrite(str(a.out / name), img)
            rows.append({"filename": name, "kind": "mcu_card", "label": c["key"],
                         "mcu": c["mcu"], "group": c["group"], "preset": p})
    with open(a.out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scenes -> {a.out}")


if __name__ == "__main__":
    main()
