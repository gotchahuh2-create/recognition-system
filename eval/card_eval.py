"""Test the MCU card step: find the card -> crop and zoom -> read all its text -> match.

Input: a folder with photos and a ``labels.csv``
(``filename,kind,label[,mcu,group,preset]``,
with ``kind=mcu_card`` and ``label`` = the card key, e.g.
``mcu-card-29-for-stm32-stm32f215re``). ``scripts/make_synthetic_cards.py``
creates such a folder. Real photos labelled in the app (Save to dataset,
kind=mcu_card) work the same way.

For every photo it reports where the card was found, and whether the
layout group, the card and the MCU were right, wrong, or left undecided.
``--compare`` also runs the older method on the same photo (reading the
whole photo for the MCU, without finding the card first) so the two can be
compared on exactly the same input.

    python eval/card_eval.py syn_cards/ --backend tesseract
    python eval/card_eval.py syn_cards/ --compare --out eval/card_eval.md
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.card_locate import CardLocator, crop_card, order_quad  # noqa: E402
from vision.card_reader import read_card  # noqa: E402
from vision.catalog import Catalog  # noqa: E402
from vision.mcu_ocr import McuRecognizer  # noqa: E402
from vision.ocr import get_backend  # noqa: E402
from vision.preprocess import Preprocessed, assess_quality, preprocess  # noqa: E402


def outcome(dec, truth) -> str:
    if dec is None:
        return "abstain"
    return "correct" if dec == truth else "WRONG"


def run_card_stage(img, rec, be, locator):
    t0 = time.perf_counter()
    locs = locator.locate(img)
    if not locs:
        return None, None, None, None, time.perf_counter() - t0
    cc = crop_card(img, locs[0])
    rd = read_card(cc, be)
    pre = Preprocessed(cc.image, cv2.cvtColor(cc.image, cv2.COLOR_BGR2GRAY), np.eye(3), 1.0, 0.0,
                       assess_quality(cc.image), [])
    mcu, card = rec.recognise(pre, list(rd.tokens), rotations_read=(0, 90, 180, 270))
    return locs[0], cc, mcu, card, time.perf_counter() - t0


def run_whole_photo(img, rec):
    t0 = time.perf_counter()
    pre = preprocess(img)
    mcu, card = rec.recognise(pre, None, hires=img)
    return mcu, card, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--compare", action="store_true", help="also run the older whole-photo pass, for comparison")
    ap.add_argument("--out", type=Path, default=None, help="markdown report")
    a = ap.parse_args(argv)
    be = get_backend(a.backend)
    cat = Catalog.load()
    rec = McuRecognizer(cat, be)
    locator = CardLocator()
    rows = [r for r in csv.DictReader(open(a.folder / "labels.csv")) if r["kind"] == "mcu_card"]
    res = []
    for r in rows:
        img = cv2.imread(str(a.folder / r["filename"]))
        card_c = cat.card_by_key(r["label"])
        truth_mcu = r.get("mcu") or (card_c.mcu if card_c else None)
        truth_grp = r.get("group") or (card_c.group if card_c else None)
        loc, cc, mcu, card, dt = run_card_stage(img, rec, be, locator)
        row = {"file": r["filename"], "preset": r.get("preset", ""),
               "located": loc is not None,
               "card_px": max(loc.size_px) if loc else 0,
               "group": outcome(card.group["key"] if card and card.group else None, truth_grp),
               "card": outcome(card.decision.key if card and card.decision else None, r["label"]),
               "mcu": outcome(mcu.decision.key if mcu and mcu.decision else None, truth_mcu),
               "s": round(dt, 1),
               "why": (card.reason if card else "card not found")[:110]}
        if a.compare:
            m0, c0, dt0 = run_whole_photo(img, rec)
            row.update({"old_group": outcome(c0.group["key"] if c0.group else None, truth_grp),
                        "old_card": outcome(c0.decision.key if c0.decision else None, r["label"]),
                        "old_mcu": outcome(m0.decision.key if m0.decision else None, truth_mcu),
                        "old_s": round(dt0, 1)})
        res.append(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items() if k != "why"), flush=True)

    by = defaultdict(list)
    for x in res:
        by[x["preset"] or "all"].append(x)
    lines = ["# MCU-card stage evaluation", "",
             f"{len(res)} photos from `{a.folder}`, backend `{be.name}`.", "",
             "| set | n | located | layout ok | card ok / wrong | MCU ok / wrong | median s |"
             + (" 0.5 layout ok | 0.5 card ok / wrong | 0.5 MCU ok / wrong | 0.5 median s |"
                if a.compare else ""),
             "|---|---|---|---|---|---|---|" + ("---|---|---|---|" if a.compare else "")]
    for k in sorted(by) + (["all"] if len(by) > 1 else []):
        xs = res if k == "all" else by[k]

        def c(f, v):
            return sum(x[f] == v for x in xs)
        line = (f"| {k} | {len(xs)} | {sum(x['located'] for x in xs)} | {c('group', 'correct')} | "
                f"{c('card', 'correct')} / {c('card', 'WRONG')} | {c('mcu', 'correct')} / "
                f"{c('mcu', 'WRONG')} | {np.median([x['s'] for x in xs]):.1f} |")
        if a.compare:
            line += (f" {c('old_group', 'correct')} | {c('old_card', 'correct')} / "
                     f"{c('old_card', 'WRONG')} | {c('old_mcu', 'correct')} / {c('old_mcu', 'WRONG')} | "
                     f"{np.median([x['old_s'] for x in xs]):.1f} |")
        lines.append(line)
    lines += ["", "| photo | card px | layout | card | MCU | reason |", "|---|---|---|---|---|---|"]
    lines += [f"| {x['file']} | {x['card_px']} | {x['group']} | {x['card']} | {x['mcu']} | {x['why']} |"
              for x in res]
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        a.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
