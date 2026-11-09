"""Draw fake "PCB photos" to quickly test the OCR pipeline without hardware.

This does NOT replace real photos (the real test set must only contain
real photos). It just shows that the code runs and catches regressions.
Each image is a dark board with a printed title, some unrelated text and a
chip with a laser-style marking, then rotated, warped, blurred, and given
glare and noise.

    python scripts/make_synthetic.py out_dir           # 9 basic test scenes
    python scripts/make_synthetic.py out_dir --hard    # also small / blurry / dark chip text

The --hard scenes cover the tough cases (out of focus, motion blur, low
light with noise, heavy JPEG, tiny letters). These are the ones the crop
re-read step (vision/refine.py) is meant to fix.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def render_board(title: str, chip: list[str], size=(2400, 1600), mask=(28, 60, 30),
                 extra=("GND", "3V3", "5V", "RST", "mikroBUS 1", "mikroBUS 2", "USB-C"),
                 chip_font: int = 58) -> np.ndarray:
    W, H = size
    im = Image.new("RGB", (W, H), (200, 200, 195))           # desk
    d = ImageDraw.Draw(im)
    bx0, by0, bx1, by1 = 150, 120, W - 150, H - 120
    d.rounded_rectangle((bx0, by0, bx1, by1), 30, fill=mask)   # PCB
    rng = random.Random(hash(title) & 0xFFFF)
    for _ in range(40):                                        # traces
        x, y = rng.randint(bx0, bx1), rng.randint(by0, by1)
        d.line((x, y, x + rng.randint(-300, 300), y), fill=tuple(c + 18 for c in mask), width=4)
    d.text((bx0 + 80, by0 + 60), title, font=_font(FONT_BOLD, 110), fill=(235, 235, 230))
    f_small = _font(FONT_BOLD, 34)
    for i, t in enumerate(extra):
        d.text((bx0 + 90 + (i % 4) * 420, by1 - 200 + (i // 4) * 70), t, font=f_small,
               fill=(230, 230, 225))
    if chip:                                                   # QFP chip
        cx0, cy0 = W // 2 - 300, H // 2 - 220
        d.rectangle((cx0 - 30, cy0 - 30, cx0 + 630, cy0 + 470), fill=(150, 150, 140))  # pins
        d.rectangle((cx0, cy0, cx0 + 600, cy0 + 440), fill=(32, 32, 34))
        f = _font(FONT_MONO, chip_font)
        for i, line in enumerate(chip):
            d.text((cx0 + 50, cy0 + 70 + i * int(chip_font * 1.55)), line, font=f,
                   fill=(170, 170, 168))
        d.ellipse((cx0 + 30, cy0 + 360, cx0 + 70, cy0 + 400), fill=(55, 55, 58))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def degrade(img: np.ndarray, seed: int, rot90: int = 0, tilt: float = 0.0,
            persp: float = 0.0, blur: float = 0.0, glare: bool = False,
            noise: float = 4.0, out_long: int = 3000, defocus: int = 0,
            motion: int = 0, dark: float = 1.0, jpeg: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if persp:
        d = persp * min(w, h)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    if tilt:
        A = cv2.getRotationMatrix2D((w / 2, h / 2), -tilt, 1.0)   # clockwise tilt
        img = cv2.warpAffine(img, A, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rot90:
        img = cv2.rotate(img, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                               cv2.ROTATE_90_COUNTERCLOCKWISE][rot90 - 1])
    if glare:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0.6, 0.85) * w, rng.uniform(0.2, 0.4) * h
        g = np.exp(-(((xx - cx) / (0.06 * w)) ** 2 + ((yy - cy) / (0.04 * h)) ** 2))
        img = np.clip(img + (g[..., None] * 255), 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    s = out_long / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if defocus:  # out of focus: disc-shaped blur
        k = np.zeros((2 * defocus + 1,) * 2, np.float32)
        cv2.circle(k, (defocus, defocus), defocus, 1, -1)
        img = cv2.filter2D(img, -1, k / k.sum())
    if motion:  # hand shake: line-shaped blur
        k = np.zeros((motion, motion), np.float32)
        k[motion // 2, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((motion / 2, motion / 2), 20, 1),
                           (motion, motion))
        img = cv2.filter2D(img, -1, k / k.sum())
    if dark != 1.0:  # low light: darker, then camera noise
        img = (img.astype(np.float32) * dark)
    img = np.clip(np.asarray(img, np.float32) + rng.normal(0, noise, img.shape), 0, 255
                  ).astype(np.uint8)
    if jpeg:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1],
                           cv2.IMREAD_COLOR)
    return img


SCENES = [
    # filename, kind, label, title, chip lines, degradation
    ("fusion_arm_flat", "board", "fusion-arm-v8", "Fusion for ARM v8",
     ["STM32F407ZGT6", "GH26P 93", "CHN 712"], dict()),
    ("fusion_arm_tilt", "board", "fusion-arm-v8", "Fusion for ARM v8",
     ["STM32F407ZGT6", "GH26P 93"], dict(tilt=7, persp=0.03, blur=1.0)),
    ("easypic_v8_rot90", "board", "easypic-v8", "EasyPIC v8",
     ["PIC18F45K22", "-I/PT 2319"], dict(rot90=1, tilt=3)),
    ("easypic_v7_upside", "board", "easypic-v7", "EasyPIC v7",
     [], dict(rot90=2, glare=True)),
    ("clicker4_glare", "board", "clicker-4-stm32", "Clicker 4 for STM32",
     ["STM32F407VGT6"], dict(glare=True, blur=0.8)),
    ("chip_stm32_closeup", "mcu", "STM32F407ZG", "", ["STM32F407ZGT6", "GH26P 93", "CHN 712"],
     dict(tilt=-12, blur=1.2)),
    ("chip_tm4c_closeup", "mcu", "TM4C129XNCZAD", "", ["TM4C129XNCZAD", "YFC 49 E4"],
     dict(rot90=3)),
    ("card10_closeup", "mcu", "PIC18F65K40", "MCU CARD 10 FOR PIC",
     ["PIC18F65K40", "-I/PT"], dict(tilt=4)),
    ("unknown_board", "board", "none", "Mystery DevKit X", ["XYZ1234ABC"], dict()),
]


# Hard cases (kind=mcu): the chip text is small, blurry, dark or heavily compressed.
# (filename, label, title, chip lines, render kwargs, degradation)
HARD_SCENES = [
    # Chosen so that reading the whole image fails, but a person can still read
    # the marking. The last two are expected to stay unsolved (giving up is fine).
    ("hard_defocus5", "STM32F429ZI", "", ["STM32F429ZIT6", "GH21K 93"],
     dict(chip_font=40), dict(out_long=1400, defocus=5)),
    ("hard_lowlight", "STM32F429ZI", "", ["STM32F429ZIT6", "GH21K 93"],
     dict(chip_font=40), dict(out_long=1400, dark=0.2, noise=14)),
    ("hard_defocus_pic", "PIC18F45K22", "", ["PIC18F45K22", "-I/PT 2319"],
     dict(chip_font=40), dict(out_long=1400, defocus=5)),
    ("hard_lowlight_avr", "ATMEGA2560", "", ["ATMEGA2560-16AU", "2140"],
     dict(chip_font=40), dict(out_long=1400, dark=0.22, noise=12)),
    ("hard_far_small_chip", "STM32F407ZG", "Fusion for ARM v8",
     ["STM32F407ZGT6", "GH26P 93"], dict(chip_font=16), dict(out_long=2400, blur=0.8)),
    ("hard_motion15", "STM32F429ZI", "", ["STM32F429ZIT6", "GH21K 93"],
     dict(chip_font=40), dict(out_long=1400, motion=15)),
    ("hard_defocus7", "STM32F429ZI", "", ["STM32F429ZIT6", "GH21K 93"],
     dict(chip_font=40), dict(out_long=1400, defocus=7)),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--hard", action="store_true", help="also write the hard scenes")
    ap.add_argument("--only-hard", action="store_true", help="write only the hard scenes")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "labels.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "kind", "label", "distance_cm"])
        for i, (fn, kind, label, title, chip, deg) in enumerate(
                [] if a.only_hard else SCENES):
            if kind == "mcu" and not title:
                img = render_board("", chip, size=(1400, 1000), extra=())
            else:
                img = render_board(title, chip)
            img = degrade(img, seed=i, **deg)
            cv2.imwrite(str(a.out / f"{fn}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            wr.writerow([f"{fn}.jpg", kind, label, ""])
        n = 0 if a.only_hard else len(SCENES)
        if a.hard or a.only_hard:
            for i, (fn, label, title, chip, rkw, deg) in enumerate(HARD_SCENES):
                if title:
                    img = render_board(title, chip, **rkw)
                else:
                    img = render_board("", chip, size=(1400, 1000), extra=(), **rkw)
                img = degrade(img, seed=5, **deg)
                cv2.imwrite(str(a.out / f"{fn}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                wr.writerow([f"{fn}.jpg", "mcu", label, ""])
                n += 1
    print(f"wrote {n} images to {a.out}")


if __name__ == "__main__":
    main()
