"""Resolution test: how sharp must a photo be to tell a card's MCU apart from the other cards
in its group?

    python eval/synth_eval.py --out eval/synth_eval.md [--ocr] [--sizes 70,100,140,200]

For each layout group (by default MCU CARD 3 for PIC32 with 4 cards, and
MCU CARD for STM32 with 19 cards) and each card in it, we draw the top of
the chip at high resolution, in a vendor-like layout. We use a DIFFERENT
font than the verifier draws with (DejaVu Sans vs Hershey simplex), so the
verifier isn't tested on its own drawings. The chip is rotated 45 degrees
(as on the card), blurred, shrunk so it's N px wide in the "photo", and
given noise and JPEG compression (quality 70, like a photo sent through a
messenger). Then exactly the card step's code runs: ``region.rectify``
(cubic enlargement) -> faint colour values + clustering ->
``synth.MarkingVerifier`` over the group's cards.

Columns: how often the verifier was right / wrong / gave up, how far the
true card scored above the best other card (negative = a wrong card scored
higher), and with --ocr the faint OCR reader's decision, for comparison.

For scale: in the three real UNI-DS v8 test photos (960x1280) the chip is 71-77 px wide.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.catalog import Catalog  # noqa: E402
from vision.faint import cluster as C  # noqa: E402
from vision.faint import features as F  # noqa: E402
from vision.faint.region import rectify  # noqa: E402
from vision.faint.synth import MarkingVerifier, hypotheses_for  # noqa: E402

FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                   "C:/Windows/Fonts/arialbd.ttf"]


def _font(px: int):
    for f in FONT_CANDIDATES:
        if Path(f).is_file():
            return ImageFont.truetype(f, px)
    return ImageFont.load_default()


def chip_lines(part: str, rnd: random.Random) -> list[tuple[str, float]]:
    """Return the marking lines as (text, relative letter height), top to bottom, like a vendor
    prints them.
    """
    lot = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(7))
    if part.startswith("PIC32"):
        pins = part[-3:]
        return [("PIC32", 1.5), (part[5:-3], 1.0), (f"{pins}-I/PH", 1.0),
                (f"e3 {rnd.randint(18, 25)}{rnd.randint(10, 52)}{lot[:3]}", 1.0)]
    if part.startswith("STM32"):
        return [(part + "T6", 1.0), (lot[:5] + " " + lot[5:], 1.0),
                (f"CHN {rnd.randint(100, 999)}", 1.0)]
    return [(part, 1.0), (lot, 1.0)]


def draw_chip(part: str, rnd: random.Random, side: int = 1000, contrast: float = 16.0
              ) -> np.ndarray:
    """Draw a high-res top view of the chip: dark, with light laser marking and slight shading."""
    body = np.full((side, side, 3), (38, 34, 32), np.float32)
    yy, xx = np.mgrid[0:side, 0:side] / side
    body += (8 * (xx - 0.5) + 5 * (yy - 0.5))[..., None]
    im = Image.new("L", (side, side), 0)
    d = ImageDraw.Draw(im)
    cap0 = side / 22.0  # marking letters are about 1/22 of the chip
    y = side * 0.28
    for text, rel in chip_lines(part, rnd):
        cap = cap0 * rel
        f = _font(int(cap * 1.38))  # font px -> capital height about 0.72 em
        d.text((side * 0.2, y), text, fill=255, font=f)
        y += cap * 1.9
    ink = np.asarray(im, np.float32)[..., None] / 255.0
    return np.clip(body + contrast * ink, 0, 255)


def photo_of_chip(chip: np.ndarray, n_native: int, rnd: random.Random, jpeg_q: int = 70
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate 45 degrees, put on a dark PCB and shrink so the chip is n_native px wide.

    Returns (photo, chip corners in photo px).
    """
    S = chip.shape[0]
    canvas = int(S * 1.8)
    pcb = np.full((canvas, canvas, 3), (25, 22, 20), np.float32)
    o = (canvas - S) // 2
    pcb[o:o + S, o:o + S] = chip
    c = canvas / 2
    M = cv2.getRotationMatrix2D((c, c), 45 + rnd.uniform(-2, 2), 1.0)
    rot = cv2.warpAffine(pcb, M, (canvas, canvas), flags=cv2.INTER_LINEAR, borderValue=(25, 22, 20))
    quad = np.array([[o, o], [o + S, o], [o + S, o + S], [o, o + S]], np.float64)
    quad = np.hstack([quad, np.ones((4, 1))]) @ M.T
    k = n_native / S
    blur = cv2.GaussianBlur(rot, (0, 0), 0.6 / k)  # lens blur: 0.6 original px
    small = cv2.resize(blur, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    small += np.random.default_rng(rnd.randint(0, 1 << 30)).normal(0, 2.0, small.shape)
    small = np.clip(small, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), quad * k


def observe(photo: np.ndarray, quad: np.ndarray):
    chip = rectify(photo, quad, method="card")
    pf = F.compute(chip.image, chip.body_mask, chip.glyph_px)
    cr = C.cluster(pf, chip.body_mask)
    native = np.linalg.norm(quad[1] - quad[0])
    scale = chip.image.shape[0] / (native * (1 + 2 * 0.04))  # same padding as region.PAD
    return chip, pf, cr, scale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default="mcu-card-3-for-pic32,mcu-card-for-stm32")
    ap.add_argument("--sizes", default="70,100,140,200", help="chip body side in photo px")
    ap.add_argument("--truths", type=int, default=4, help="truth cards per group")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--ocr", action="store_true", help="also run the faint OCR reader (slow)")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    cat = Catalog.load()
    ver = MarkingVerifier()
    sizes = [int(s) for s in a.sizes.split(",")]
    rows, raw = [], []
    for gkey in a.groups.split(","):
        cards = [c for c in cat.mcu_cards if c.group == gkey and not c.empty and c.mcus]
        hyps = hypotheses_for(cards)
        rnd = random.Random(7)
        truths = cards[:]
        rnd.shuffle(truths)
        truths = truths[:a.truths]
        for n in sizes:
            ok = bad = ab = ocr_ok = ocr_bad = 0
            margins, t0 = [], time.perf_counter()
            for tc in truths:
                for r in range(a.reps):
                    rr = random.Random(hash((tc.key, n, r)) & 0xFFFF)
                    photo, quad = photo_of_chip(draw_chip(tc.mcus[0], rr), n, rr)
                    chip, pf, cr, scale = observe(photo, quad)
                    res = ver.verify(pf.resid["lab"][:, :, 0], chip.body_mask, cr.polarity,
                                     chip.glyph_px, scale, hyps)
                    other = max(v for k, v in res.scores.items() if k != tc.key) if res.scores else 0
                    m = res.scores.get(tc.key, 0) - other
                    margins.append(m)
                    raw.append((gkey, n, tc.key, res.decision, round(m, 3)))
                    if res.decision is None:
                        ab += 1
                    elif res.decision == tc.key:
                        ok += 1
                    else:
                        bad += 1
                    if a.ocr:
                        from vision.faint.__main__ import _decide
                        from vision.faint.reader import FaintChipReader
                        from vision.mcu_ocr import McuRecognizer
                        rd = FaintChipReader(a.backend)
                        dec, _ = _decide(McuRecognizer(cat, rd.be), rd.read_chip(chip).tokens)
                        if dec:
                            ocr_ok += dec.upper().startswith(tc.mcus[0].upper())
                            ocr_bad += not dec.upper().startswith(tc.mcus[0].upper())
            k = len(margins)
            rows.append((gkey, len(hyps), n, ok, bad, ab, float(np.median(margins)),
                         float(np.min(margins)), f"{ocr_ok}/{ocr_bad}/{k - ocr_ok - ocr_bad}" if a.ocr else "-",
                         (time.perf_counter() - t0) / max(k, 1)))
            print(*rows[-1], flush=True)
    lines = ["# Resolution sweep — marking verifier (synthetic)", "",
             f"Verifier: MIN_SCORE {ver.min_score}, MARGIN {ver.margin}, PSF {ver.psf} native px. "
             "Chip body side = px in the photo (real UNI-DS v8 test photos: 71-77 px).", "",
             "| group | cards | chip px | verifier ✓ | ✗ | – | median margin true-vs-best-other | min margin | OCR ✓/✗/– | s/chip |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for g, nc, n, ok, bad, ab, med, mn, ocr, sec in rows:
        lines.append(f"| {g} | {nc} | {n} | {ok} | {bad} | {ab} | {med:+.3f} | {mn:+.3f} | {ocr} | {sec:.1f} |")
    lines += ["", "Per chip:", "", "| group | px | truth | decision | margin |", "|---|---|---|---|---|"]
    lines += [f"| {g} | {n} | {t} | {d or '–'} | {m:+.3f} |" for g, n, t, d, m in raw]
    txt = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(txt, encoding="utf-8")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
