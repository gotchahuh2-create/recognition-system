"""Efficiency instrumentation: where the time goes (feeds docs/reports/efficiency.md).

``summarise(prof_path)`` reads a cProfile dump of a bench run and returns:

* own time grouped into buckets (OCR engine wait, OpenCV / NumPy, fuzzy text
  matching, PHOENIX / Python glue ...);
* the vision functions with the largest cumulative and own time;
* call counts of the OCR entry points (how many OCR passes a photo costs).

``count_ocr_calls`` wraps an OCR backend to count and time every read on
one photo without a profiler (so the timings are not inflated).
"""

from __future__ import annotations

import pstats
import time
from collections import defaultdict
from pathlib import Path

BUCKETS = [
    ("OCR engine (subprocess wait / native)", ("subprocess.py", "selectors.py", "{method 'poll'",
                                                "{built-in method posix.waitpid", "{method 'read' of '_io",
                                                "{built-in method _posixsubprocess", "onnxruntime",
                                                "rapidocr", "paddle", "pytesseract")),
    ("OpenCV (cv2 native)", ("{built-in method cv2.", "cv2.")),
    ("NumPy (native)", ("{method 'reduce' of 'numpy", "numpy/", "{built-in method numpy",
                        "{method 'astype'", "{method 'sort' of 'numpy", "{method 'copy' of 'numpy")),
    ("Fuzzy text matching (vision/textmatch.py)", ("vision/textmatch.py",)),
    ("Faint clustering (vision/faint)", ("vision/faint/",)),
    ("Other vision Python", ("vision/",)),
    ("Image IO (PIL / imread)", ("PIL/", "{built-in method cv2.imread", "imdecode")),
]


def _bucket(key) -> str:
    fn, _, name = key
    s = f"{fn}:{name}"
    for label, pats in BUCKETS:
        if any(p in s for p in pats):
            return label
    return "Python stdlib / other"


def summarise(prof_path: Path, top: int = 25) -> dict:
    st = pstats.Stats(str(prof_path))
    total = st.total_tt
    buckets = defaultdict(float)
    for key, (cc, nc, tt, ct, callers) in st.stats.items():
        buckets[_bucket(key)] += tt
    rows = []
    for key, (cc, nc, tt, ct, callers) in st.stats.items():
        fn, line, name = key
        if "/vision/" in fn or "/phoenix/" in fn:
            short = fn[fn.find("/vision/") + 1:] if "/vision/" in fn else fn[fn.find("/phoenix/") + 1:]
            rows.append({"func": f"{short}:{line}({name})", "calls": nc,
                         "own_s": round(tt, 2), "cum_s": round(ct, 2)})
    ocr = [r for r in rows if r["func"].startswith("vision/ocr.py") and
           any(k in r["func"] for k in ("(read)", "(read_text)", "(read_crop)"))]
    return {"total_s": round(total, 1),
            "buckets": sorted(((k, round(v, 1), round(100 * v / total, 1)) for k, v in buckets.items()),
                              key=lambda x: -x[1]),
            "top_cum": sorted(rows, key=lambda r: -r["cum_s"])[:top],
            "top_own": sorted(rows, key=lambda r: -r["own_s"])[:top],
            "ocr_entry_points": ocr}


class OcrCounter:
    """Wrap an OcrBackend: count + time every ``read`` call (works for any engine)."""

    def __init__(self, backend):
        self.backend = backend
        self.calls, self.seconds, self.pixels = 0, 0.0, 0
        self._orig = backend.read

        def read(img, *a, **kw):
            t0 = time.perf_counter()
            try:
                return self._orig(img, *a, **kw)
            finally:
                self.calls += 1
                self.seconds += time.perf_counter() - t0
                self.pixels += int(img.shape[0] * img.shape[1])
        backend.read = read

    def reset(self):
        self.calls, self.seconds, self.pixels = 0, 0.0, 0

    def restore(self):
        self.backend.read = self._orig


def tesseract_fixed_cost(n: int = 10) -> dict:
    """Seconds one Tesseract call costs before it reads anything (process start + model load)."""
    import numpy as np
    from vision.ocr import TesseractBackend
    be = TesseractBackend()
    blank = np.full((32, 32), 255, np.uint8)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        be.read(blank)
        ts.append(time.perf_counter() - t0)
    crop = blank.copy()
    ts_crop = []
    for _ in range(n):
        t0 = time.perf_counter()
        be.read_crop(crop)
        ts_crop.append(time.perf_counter() - t0)
    return {"page_call_s": round(float(np.median(ts)), 3),
            "crop_call_s": round(float(np.median(ts_crop)), 3)}
