"""Use OCR to re-rank Click search results, plus the text helpers it needs.

For each Click crop: enlarge it (long side about 1000 px), read it at 0 and
180 degrees and keep the more confident read, then normalise the text
(uppercase; remove CLICK, MIKROE, punctuation and the board ID). Then fuzzy
match it against the normalised names of the top 5 search results with
``token_set_ratio``:

    match >= 90  -> move to first place
    75 .. 89     -> +0.10 bonus
    otherwise    -> no change

Numbered series ("Thermo 2" vs "Thermo 3"): the number at the end is
handled separately. If OCR reads a number after the series name, it must
match exactly. If OCR reads no number, OCR gives no bonus and the decision
is left to the image search plus the user's confirmation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except ImportError:  # pragma: no cover
    _rf_fuzz = None

from ..ocr import OcrBackend, effective_rotations, group_lines, read_text
from ..textmatch import normalize

DROP_WORDS = {"CLICK", "MIKROE", "MIKROELEKTRONIKA", "WWW", "COM", "MIKROE.COM",
              "MIKROBUS", "BOARD"}
_ID_BLOCK = re.compile(r"\bMIKROE[\s-]*\d{2,6}\b|\b\d{4,6}\b")
_SERIES = re.compile(r"^(.*?\D)\s*(\d{1,2})$")


def normalize_click_text(s: str) -> str:
    s = normalize(s or "")
    s = _ID_BLOCK.sub(" ", s)  # MIKROE-1234 or a bare board ID
    s = re.sub(r"[^A-Z0-9 ]", " ", s)               # punctuation (keeps digits)
    s = re.sub(r"CLICK\b", " ", s)                  # "THERMO2CLICK" -> "THERMO2"
    toks = [t for t in s.split() if t not in DROP_WORDS]
    # split "THERMO2"-style words so the number is separate
    out = []
    for t in toks:
        m = re.fullmatch(r"([A-Z]{3,})(\d{1,2})", t)
        out += [m.group(1), m.group(2)] if m else [t]
    return " ".join(out)


def series_parts(norm_name: str) -> tuple[str, int | None]:
    m = _SERIES.match(norm_name)
    if m and m.group(1).strip():
        return m.group(1).strip(), int(m.group(2))
    return norm_name, None


# ---- token_set_ratio (RapidFuzz if installed, otherwise our own copy) ----
def _indel_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return 100.0 * 2 * prev[m] / (n + m)


def token_set_ratio(a: str, b: str) -> float:
    if _rf_fuzz is not None:
        return float(_rf_fuzz.token_set_ratio(a, b))
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = " ".join(sorted(ta & tb))
    da, db = " ".join(sorted(ta - tb)), " ".join(sorted(tb - ta))
    if inter and (not da or not db):
        return 100.0
    sa = (inter + " " + da).strip()
    sb = (inter + " " + db).strip()
    return max(_indel_ratio(inter, sa) if inter else 0.0,
               _indel_ratio(inter, sb) if inter else 0.0,
               _indel_ratio(sa, sb))


# ---- OCR of a Click crop ----
@dataclass
class ClickOcr:
    text: str  # as read, in reading order
    norm: str              # normalised
    conf: float
    rotation: int
    ms: float = 0.0


def ocr_click(crop: np.ndarray, backend: OcrBackend, long_side: int = 1000) -> ClickOcr:
    """Enlarge the crop, read it at 0 and 180 degrees, and keep the more confident text."""
    import time
    t0 = time.perf_counter()
    h, w = crop.shape[:2]
    if h > 1.2 * w:
        # Click names run along the long side, so turn a portrait crop sideways.
        # Then 0 and 180 degrees cover the two remaining directions.
        crop = np.ascontiguousarray(np.rot90(crop))
        h, w = w, h
    f = long_side / max(h, w)
    if abs(f - 1) > 0.05:
        crop = cv2.resize(crop, None, fx=f, fy=f,
                          interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    best = ClickOcr("", "", 0.0, 0)
    best_score = -1.0
    for rot in effective_rotations(backend, (0, 180)):
        toks = read_text(crop, backend, rotations=(rot,), min_conf=0.3)
        lines = group_lines(toks)
        text = " | ".join(" ".join(t.text for t in ln) for ln in lines)
        score = sum(t.conf * len(t.text.strip()) for t in toks)
        if score > best_score:
            conf = float(np.mean([t.conf for t in toks])) if toks else 0.0
            best, best_score = ClickOcr(text, normalize_click_text(text), conf, rot), score
    best.ms = (time.perf_counter() - t0) * 1000
    return best


@dataclass
class OcrMatch:
    ratio: float           # 0..100
    number_ok: bool | None  # None if the name has no series number
    ocr_number: int | None


def match_name(ocr_norm: str, name: str) -> OcrMatch:
    """Fuzzy match a name, applying the numbered-series rule."""
    norm_name = normalize_click_text(name)
    if not ocr_norm or not norm_name:
        return OcrMatch(0.0, None, None)
    base, num = series_parts(norm_name)
    m = re.search(r"\b" + re.escape(base) + r"\s*(\d{1,2})\b", ocr_norm)
    ocr_num = int(m.group(1)) if m else None
    r = token_set_ratio(ocr_norm, norm_name)
    if num is None:
        if ocr_num is not None:  # OCR says "THERMO 2", but the candidate is just "THERMO"
            return OcrMatch(0.0, False, ocr_num)
        return OcrMatch(r, None, None)
    if ocr_num is None:  # no number read: OCR doesn't vote
        return OcrMatch(0.0, None, None)
    if ocr_num != num:
        return OcrMatch(0.0, False, ocr_num)
    return OcrMatch(r, True, ocr_num)


def series_base(name: str) -> str:
    return series_parts(normalize_click_text(name))[0]
