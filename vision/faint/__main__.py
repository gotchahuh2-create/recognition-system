"""Run the faint-marking reader on chip photos.

    python -m vision.faint chip.jpg                     # combined lines + MCU decision
    python -m vision.faint photos/ --debug dbg_faint/   # + each step as PNG and an overview
    python -m vision.faint chip.jpg --compare           # + the classic chip pass, same crop
    python -m vision.faint chip.jpg --json              # full report

Input: any photo where the chip is the dark area in the middle (a close-up,
or a zoomed MCU card crop). For photos of a whole board use
``python -m vision photo.jpg --chip-mode faint`` instead; the card step
finds the chip first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from ..catalog import Catalog
from ..mcu_ocr import ACCEPT, MARGIN, McuRecognizer, candidate_strings
from .debug import StageSaver
from .reader import FaintChipReader, FaintConfig
from .region import extract_chip

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _decide(rec: McuRecognizer, toks):
    cands = candidate_strings(toks)
    parts = rec.merge_parts(rec.score_parts(cands), rec.score_markings(cands, rec.catalog.mcu_cards))
    parts.sort(key=lambda c: -c.score)
    if parts and parts[0].score >= ACCEPT and (len(parts) == 1 or parts[0].score - parts[1].score >= MARGIN):
        return parts[0].key, parts[:3]
    return None, parts[:3]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--debug", type=Path, default=None, help="folder for stage images")
    ap.add_argument("--compare", action="store_true", help="also run the classic chip pass")
    ap.add_argument("--ocr", default="crop", choices=["crop", "page"])
    ap.add_argument("--orient", default="geometry", choices=["geometry", "ocr", "all"])
    ap.add_argument("--n-best", type=int, default=3, help="thresholded masks to OCR")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    files = []
    for p in a.inputs:
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXT) if p.is_dir() else [p]
    reader = FaintChipReader(a.backend, FaintConfig(ocr_mode=a.ocr, orient=a.orient, n_best=a.n_best))
    rec = McuRecognizer(Catalog.load(), reader.be)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f}: cannot read", file=sys.stderr)
            continue
        sv = StageSaver(a.debug, f.stem)
        chip = extract_chip(img, saver=sv)
        fr = reader.read_chip(chip, sv)
        dec, top = _decide(rec, fr.tokens)
        out = {"file": str(f), "mcu": dec, "candidates": [[c.key, round(c.score, 3)] for c in top],
               **fr.as_dict()}
        if a.compare:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from eval.faint_eval import classic_tokens
            t0 = time.perf_counter()
            ct = classic_tokens(chip, reader.be, a.ocr)
            cdec, ctop = _decide(rec, ct)
            out["classic"] = {"mcu": cdec, "candidates": [[c.key, round(c.score, 3)] for c in ctop],
                              "seconds": round(time.perf_counter() - t0, 1)}
        if a.json:
            print(json.dumps(out, indent=1, default=str))
        else:
            fused = " | ".join(x.text for x in fr.fused[:4]) or "-"
            line = (f"{f.name}: MCU={dec or '?'}  fused: {fused}  "
                    f"[{chip.method}, rot {fr.rotations}, {fr.timings_ms['total'] / 1000:.1f} s]")
            if a.compare:
                line += f"  | classic MCU={out['classic']['mcu'] or '?'} ({out['classic']['seconds']} s)"
            print(line)
        if sv.enabled:
            print(f"   stages -> {a.debug}/{f.stem}_*.png, sheet {f.stem}_sheet.jpg")


if __name__ == "__main__":
    sys.exit(main())
