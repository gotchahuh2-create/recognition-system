"""Compare two ways of reading faint chip markings: the standard chip pass and the experimental
one.

Both methods get the SAME straightened chip (``vision.faint.region``), so
only the image clean-up (and how many images are read) differs:

``classic``  the card step's chip pass (``card_reader``): denoise, enlarge,
             CLAHE, then 'enhanced' / Otsu / Sauvola images
             (``enhance.enhance_crop``), each read at 0/90/180/270 degrees
``faint``    ``vision.faint``: RGB/HSV/LAB colour values, clustering, many
             thresholds, careful clean-up, variants, OCR, combining results
``both``     the words from both methods together (what ``--chip-mode both`` does)
``faint_nofuse``  for comparison: faint words without the combined lines

Both methods use the same OCR call: ``--ocr page`` = ``read_text``
(Tesseract sparse text, what the card step uses now), ``--ocr crop`` =
``backend.read_crop`` (block mode, only part-number characters, no
dictionaries).

Scoring uses the whole catalog, without narrowing by card title (the chip
on its own). The MCU is decided by ``McuRecognizer.score_parts`` +
``score_markings`` with the pipeline's accept rule (top >= 0.90, margin
>= 0.05): correct / wrong / gave up. "top3" = the right answer is among
the three best candidates. CER = edit distance between the best-matching
read and the printed part number line(s).

    python scripts/make_faint_chips.py syn_faint/ --n 6
    python eval/faint_eval.py syn_faint/ --ocr page,crop --out eval/faint_eval.md --debug dbg_faint/
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import enhance  # noqa: E402
from vision.card_reader import CHIP_VARIANTS, ROTATIONS  # noqa: E402
from vision.catalog import Catalog  # noqa: E402
from vision.faint import FaintChipReader, FaintConfig, extract_chip  # noqa: E402
from vision.faint.debug import StageSaver  # noqa: E402
from vision.faint.reader import read_crop_rotations  # noqa: E402
from vision.mcu_ocr import ACCEPT, MARGIN, McuRecognizer, candidate_strings  # noqa: E402
from vision.ocr import get_backend, group_lines, read_text  # noqa: E402
from vision.textmatch import compact  # noqa: E402

METHODS = ("classic", "faint", "both")
EXTRA = ("faint_nofuse",)  # for comparison: faint words without the combined lines


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_cer(truth: str, toks) -> tuple[float, str]:
    """Return the smallest edit distance from any read string to ``truth``, divided by its
    length.

    Read strings are words, lines and pairs of stacked lines.
    """
    strs = {compact(t.text) for t in toks}
    lines = group_lines(toks)
    ls = ["".join(compact(t.text) for t in ln) for ln in lines]
    strs |= set(ls) | {a + b for a, b in zip(ls, ls[1:])}
    strs.discard("")
    if not strs:
        return 1.0, ""
    s = min(strs, key=lambda s: lev(truth, s))
    return min(1.0, lev(truth, s) / max(len(truth), 1)), s


def classic_tokens(chip, be, ocr: str):
    cimg = chip.image
    glyph = cimg.shape[0] / 18.0
    toks = []
    for vname, vimg in enhance.enhance_crop(cimg, glyph, variants=CHIP_VARIANTS):
        if ocr == "crop":
            ct = read_crop_rotations(vimg, be, ROTATIONS)
        else:
            ct = read_text(vimg, be, rotations=ROTATIONS, variants=("gray",))
        for t in ct:
            t.variant = f"chip-{vname}"
        toks += ct
    return toks


def decide(rec: McuRecognizer, toks):
    cands = candidate_strings(toks)
    parts = rec.merge_parts(rec.score_parts(cands), rec.score_markings(cands, rec.catalog.mcu_cards))
    parts.sort(key=lambda c: -c.score)
    top3 = [c.key for c in parts[:3]]
    if parts and parts[0].score >= ACCEPT and (len(parts) == 1 or parts[0].score - parts[1].score >= MARGIN):
        return parts[0].key, top3, parts[0].score
    return None, top3, parts[0].score if parts else 0.0


def outcome(dec, truth):
    if dec is None:
        return "abstain"
    return "correct" if compact(dec) == compact(truth) else "wrong"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--backend", default="tesseract")
    ap.add_argument("--ocr", default="page,crop", help="OCR call(s) to compare: page, crop")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="markdown report")
    ap.add_argument("--csv", type=Path, default=None, help="per-image rows")
    ap.add_argument("--debug", type=Path, default=None, help="faint-pipeline stage images per photo")
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(open(a.folder / "labels.csv")))
    if a.limit:
        rows = rows[:a.limit]
    be = get_backend(a.backend)
    rec = McuRecognizer(Catalog.load(), be)
    results = []
    for mode in a.ocr.split(","):
        reader = FaintChipReader(be, FaintConfig(ocr_mode=mode))
        for r in rows:
            img = cv2.imread(str(a.folder / r["filename"]))
            truth_line = compact("".join(r["lines"].split(" / ")[:2]))
            chip = extract_chip(img)
            out = {"file": r["filename"], "ocr": mode, "label": r["label"],
                   "delta": float(r.get("delta") or 0), "glare": int(r.get("glare") or 0),
                   "region": chip.method}
            t0 = time.perf_counter()
            ct = classic_tokens(chip, be, mode)
            out["classic_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            sv = StageSaver(a.debug / mode if a.debug else None, Path(r["filename"]).stem)
            fr = reader.read_chip(chip, sv)
            out["faint_s"] = time.perf_counter() - t0
            out["both_s"] = out["classic_s"] + out["faint_s"]
            raw = [t for t in fr.tokens if t.variant != "chip-faint-fused"]
            for m, toks in (("classic", ct), ("faint", fr.tokens), ("both", ct + fr.tokens),
                            ("faint_nofuse", raw)):
                dec, top3, sc = decide(rec, toks)
                cer, s = best_cer(truth_line, toks)
                out[m] = outcome(dec, r["label"])
                out[m + "_dec"] = dec or ""
                out[m + "_top3"] = int(any(compact(x) == compact(r["label"]) for x in top3))
                out[m + "_cer"] = cer
                out[m + "_read"] = s
            out["fused"] = " | ".join(f.text for f in fr.fused[:3])
            print(f"[{mode}] {r['filename']:<38} classic={out['classic']:<8}({out['classic_cer']:.2f}) "
                  f"faint={out['faint']:<8}({out['faint_cer']:.2f}) both={out['both']:<8} "
                  f"{out['classic_s']:.1f}s/{out['faint_s']:.1f}s  {out['faint_read']}", flush=True)
            results.append(out)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
    md = report(results, a)
    print(md)
    if a.out:
        a.out.write_text(md)


def report(results, a) -> str:
    out = ["# Faint chip marking: classic chip pass vs vision.faint", "",
           f"Folder `{a.folder.name}`, backend {a.backend}, {len({r['file'] for r in results})} photos. "
           "Same rectified chip for every method. correct / wrong / abstain = MCU decided by the "
           "pipeline's accept rule over the whole catalogue; top3 = truth among 3 best; "
           "CER = best read vs printed part-number line (lower is better).", ""]
    for mode in sorted({r["ocr"] for r in results}):
        R = [r for r in results if r["ocr"] == mode]
        out += [f"## OCR call: `{mode}`", "",
                "| contrast (levels) | n | classic ✓/✗/– | faint ✓/✗/– | both ✓/✗/– | faint w/o fusion ✓/✗/– | "
                "top-3 c/f/b | mean CER classic | mean CER faint | median s classic | median s faint |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        groups = defaultdict(list)
        for r in R:
            groups[r["delta"]].append(r)
        for key in sorted(groups, reverse=True) + ["all"]:
            G = R if key == "all" else groups[key]

            def cwa(m):
                return "/".join(str(sum(r[m] == o for r in G)) for o in ("correct", "wrong", "abstain"))
            top = "/".join(str(sum(r[m + "_top3"] for r in G)) for m in METHODS)
            out.append(
                f"| {'all' if key == 'all' else f'{key:g}'} | {len(G)} | {cwa('classic')} | {cwa('faint')} | "
                f"{cwa('both')} | {cwa('faint_nofuse') if 'faint_nofuse' in G[0] else 'n/a'} | {top} | {statistics.mean(r['classic_cer'] for r in G):.2f} | "
                f"{statistics.mean(r['faint_cer'] for r in G):.2f} | "
                f"{statistics.median(r['classic_s'] for r in G):.1f} | {statistics.median(r['faint_s'] for r in G):.1f} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
