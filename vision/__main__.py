"""Command line: python -m vision PHOTO_OR_FOLDER [...] [--backend
auto|paddle|rapidocr|tesseract]

Prints one JSON object per photo (JSON Lines), or writes them to --out.
--debug DIR saves the processed image with the OCR boxes drawn on it, so
you can look at why a photo failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .loader import iter_photos
from .ocr import available_backends
from .pipeline import Pipeline
from .enhance import available_upscalers
from .preprocess import PreprocessConfig
from .refine import RefineConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vision", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="photos or folders")
    ap.add_argument("--backend", default="auto",
                    help=f"OCR backend (available here: {', '.join(available_backends()) or 'none'})")
    ap.add_argument("--catalog", default=None, help="data folder with CSVs or products.db")
    ap.add_argument("--long-side", type=int, default=2000)
    ap.add_argument("--no-mcu", action="store_true", help="board only")
    ap.add_argument("--refine", default="auto", choices=["auto", "always", "off"],
                    help="crop-refinement pass for small/blurred markings (default auto)")
    ap.add_argument("--upscaler", default="bicubic",
                    help=f"crop enlargement: {', '.join(available_upscalers())}")
    ap.add_argument("--auto-perspective", action="store_true",
                    help="rectify the board outline automatically when found")
    ap.add_argument("--videos", action="store_true", help="also sample frames from videos")
    ap.add_argument("--out", type=Path, help="write JSON Lines here")
    ap.add_argument("--debug", type=Path, help="folder for annotated debug images")
    ap.add_argument("--summary", action="store_true", help="one human-readable line per photo")
    g0 = ap.add_argument_group("two-stage mode (board layouts, see python -m vision.layout)")
    g0.add_argument("--layouts", default=None, help="layouts.json (default: data/layouts.json)")
    g0.add_argument("--no-layout", action="store_true",
                    help="old behaviour: read the MCU from the whole photo")
    g0.add_argument("--roi-fallback", action="store_true",
                    help="read the whole photo when the MCU-card area gives no answer")
    gc = ap.add_argument_group("MCU card stage (locate -> crop + zoom -> read all text -> match)")
    gc.add_argument("--no-card", action="store_true",
                    help="skip finding and zooming in on the MCU card")
    gc.add_argument("--card-zoom", type=int, default=1800,
                    help="enlarge the card crop until its long side is this many px")
    gc.add_argument("--chip-mode", default="classic", choices=["classic", "faint", "both"],
                    help="chip-marking cleaning: classic (standard), faint (experimental "
                         "vision.faint: colour clustering + many thresholds + fusion), both")
    gc.add_argument("--verify-marking", action="store_true",
                    help="experimental: when the card layout is read but the chip OCR abstains, "
                         "compare the layout's candidate markings through a resolution model "
                         "(vision.faint.synth); decides only with a clear margin")
    gc.add_argument("--faint-debug", type=Path, default=None,
                    help="folder for the faint pipeline's per-stage debug images")
    g = ap.add_argument_group("Clicks (DINOv2 retrieval, see vision/clicks)")
    g.add_argument("--gallery", default=None, help="gallery folder; enables the Click step")
    g.add_argument("--click-model", default="models/dinov2s_reg_224.onnx")
    g.add_argument("--click-mode", default="photo", choices=["photo", "boxes", "off"],
                   help="photo: whole photo is one Click unless --click-box is given")
    g.add_argument("--click-box", action="append", default=None,
                   help="x,y,w,h of one Click in photo pixels (repeat per Click)")
    a = ap.parse_args(argv)

    pipe = Pipeline(a.catalog, a.backend,
                    PreprocessConfig(long_side=a.long_side,
                                     auto_perspective=a.auto_perspective),
                    mcu=not a.no_mcu, refine=a.refine,
                    refine_cfg=RefineConfig(upscaler=a.upscaler),
                    clicks=_click_recognizer(a), click_mode=a.click_mode,
                    layouts=False if a.no_layout else a.layouts,
                    roi_fallback=a.roi_fallback,
                    card_stage=not a.no_card, card_zoom_target=a.card_zoom,
                    upscaler=a.upscaler, chip_mode=a.chip_mode, faint_debug=a.faint_debug,
                    verify_marking=a.verify_marking)
    boxes = [[float(v) for v in b.split(",")] for b in a.click_box] if a.click_box else None
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    if a.debug:
        a.debug.mkdir(parents=True, exist_ok=True)
    n = 0
    for inp in a.inputs:
        for photo in iter_photos(inp, include_videos=a.videos):
            res = pipe.run(photo, click_boxes=boxes)
            d = res.as_dict()
            if a.summary:
                m = d["manifest"]
                b = m["board"]["name"] if m["board"] else f"? ({res.board.reason})"
                mc = m["mcu"]["name"] if m["mcu"] else f"? ({res.mcu.reason})"
                warn = "; ".join(d["quality"]["warnings"])
                if m["mcu_card"]:
                    mc += f" | card={m['mcu_card']['name']}"
                elif m.get("mcu_card_layout"):
                    lay_ = m["mcu_card_layout"]
                    mc += f" | card=? (layout '{lay_['title']}', {lay_['cards']} cards)"
                cl = "".join(f" | click={c.recognition.candidates[0].name if c.outcome == 'accept' else c.outcome + '?'}"
                             for c in res.clicks if c.recognition.candidates) if res.clicks else ""
                lo = (f" | layout={res.placement.method} {res.placement.inliers}/"
                      f"{res.placement.matched}, mcu from {res.mcu_source}"
                      if res.placement else "")
                if res.card_crop is not None:
                    cc = res.card_crop
                    lo += (f" | card at {cc.location.method} {cc.location.size_px[0]}x"
                           f"{cc.location.size_px[1]} px zoom x{cc.zoom:.1f}, mcu from {res.mcu_source}")
                print(f"{photo.name}: board={b} | mcu={mc}{cl}{lo} | {d['timings_ms']['total']} ms"
                      + (f" | WARN {warn}" if warn else ""))
            else:
                print(json.dumps(d, ensure_ascii=False))
            if out:
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
            if a.debug:
                _save_debug(photo, res, a.debug)
            n += 1
    if out:
        out.close()
    if n == 0:
        print("no photos found", file=sys.stderr)
        return 1
    return 0


def _click_recognizer(a):
    if not a.gallery or a.click_mode == "off":
        return None
    from .catalog import Catalog
    from .clicks import ClickRecognizer
    return ClickRecognizer(a.click_model, a.gallery, a.backend, Catalog.load(a.catalog))


def _save_debug(photo, res, folder: Path) -> None:
    img = res.pre.image.copy()  # evidence boxes use these coordinates
    for c in res.mcu.crops:  # re-read crops in magenta
        cv2.polylines(img, [np.asarray(c.region.pts, np.int32).reshape(-1, 1, 2)], True,
                      (255, 0, 255), 2)
    for rec, color in ((res.board, (0, 220, 0)), (res.mcu, (0, 170, 255))):
        for ev in rec.evidence:
            if ev.get("box") is not None:
                pts = np.asarray(ev["box"], np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], True, color, 3)
    fs = max(img.shape[:2]) / 1200
    for i, (label, rec, color) in enumerate((("board", res.board, (0, 255, 0)),
                                             ("mcu", res.mcu, (0, 200, 255)))):
        txt = f"{label}: {rec.decision.name if rec.decision else '? ' + rec.reason}"
        org = (int(15 * fs), int((40 + 40 * i) * fs))
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), int(5 * fs))
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, int(2 * fs))
    stem = Path(photo.name).stem
    for r in res.rois:  # layout areas: cyan for the MCU card, yellow for sockets
        color = (255, 255, 0) if r.name == "MCU card" else (0, 220, 255)
        cv2.polylines(img, [np.asarray(r.quad_pre, np.int32).reshape(-1, 1, 2)], True, color, 3)
        org = tuple(int(v) for v in np.asarray(r.quad_pre).min(0) + [0, -8])
        cv2.putText(img, r.name, org, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, color, int(2 * fs) + 1)
        safe = "".join(ch if ch.isalnum() else "_" for ch in r.name)
        cv2.imwrite(str(folder / f"{stem}_roi_{safe}.jpg"), r.image)
    if res.card_crop is not None:  # found card: magenta outline plus the zoomed crop
        q = res.pre.M @ np.vstack([res.card_crop.location.quad.T, np.ones(4)])
        q = (q[:2] / q[2]).T
        cv2.polylines(img, [q.astype(np.int32).reshape(-1, 1, 2)], True, (180, 0, 255), 4)
        crop = res.card_crop.image.copy()
        if res.card_reading and res.card_reading.chip is not None:
            cv2.polylines(crop, [res.card_reading.chip.quad.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 170, 255), 3)
            if res.card_reading.chip.image is not None:
                cv2.imwrite(str(folder / f"{stem}_card_chip.jpg"), res.card_reading.chip.image)
        cv2.imwrite(str(folder / f"{stem}_card.jpg"), crop)
    cv2.imwrite(str(folder / (stem + "_debug.jpg")), img)

if __name__ == "__main__":
    sys.exit(main())
