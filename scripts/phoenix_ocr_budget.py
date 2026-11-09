"""Efficiency study: how many OCR calls a photo costs, and how much of its time they take.

    python scripts/phoenix_ocr_budget.py [--config v1] > runs/ocr_budget.json

Runs a fixed sample (one photo per kind) through the configured pipeline with
an ``OcrCounter`` around the backend (no profiler, so timings are real) and
reports per photo: total s, OCR calls, OCR wall s (threads overlap, so this
can exceed total on 2 cores), pixels read, and the fixed per-call cost of the
engine. Used by docs/reports/efficiency.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE = [("syn_ocr", "chip_stm32_closeup.jpg"), ("syn_cards", "card_close_mcu-card-29-for-stm32-stm32f215re.jpg"),
          ("syn_cards", "card_far_mcu-card-29-for-stm32-stm32f215re.jpg"),
          ("real3", "uni-ds-v8_stm32f429zi_2.jpg"), ("captures", "20261105_153735_board_01.jpg"),
          ("captures", "20261105_150652_mcu_03.jpg")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    from phoenix.__main__ import _cfg
    from phoenix.datasets import Dataset
    from phoenix.profiling import OcrCounter, tesseract_fixed_cost
    from phoenix.stages import PhoenixPipeline
    from vision.loader import load_photo
    rc = _cfg(a.config)
    pp = PhoenixPipeline(rc)
    cnt = OcrCounter(pp.pipeline.backend)
    rows = []
    for dn, f in SAMPLE:
        it = next(i for i in Dataset.load(dn).items() if i.file == f)
        photo = load_photo(it.path)
        cnt.reset()
        t0 = time.perf_counter()
        res = pp.run(photo)
        total = time.perf_counter() - t0
        rows.append({"photo": f"{dn}/{f}", "size": list(photo.image.shape[1::-1]),
                     "total_s": round(total, 2), "ocr_calls": cnt.calls,
                     "ocr_call_s_sum": round(cnt.seconds, 2),
                     "ocr_mpix": round(cnt.pixels / 1e6, 1),
                     "stages_ms": res.timings_ms})
        print(json.dumps(rows[-1]), file=sys.stderr)
    out = {"config": rc.as_dict(), "backend": pp.pipeline.backend.name,
           "fixed_cost": tesseract_fixed_cost() if pp.pipeline.backend.name == "tesseract" else None,
           "photos": rows}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
