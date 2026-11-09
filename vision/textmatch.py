"""Fuzzy text matching for OCR results (no extra libraries needed).

We match two kinds of text:

* Product names ("Fusion for ARM v8"): word by word, where rare words count
  more.
* Part numbers ("STM32F407ZGT6"): letter by letter, using an edit distance
  where letters OCR often confuses (0/O, 1/I/L, 5/S, 8/B, 2/Z, 6/G) cost
  less than a real mistake.

If ``rapidfuzz`` is installed we use it for plain ratios, otherwise pure
Python. The lists are small (tens of boards, hundreds of MCUs), so speed is
fine either way.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable

try:  # optional, makes things faster
    from rapidfuzz.distance import Levenshtein as _RFLev  # type: ignore
except ImportError:  # pragma: no cover
    _RFLev = None

# Pairs of characters OCR mixes up on laser-marked chips and silkscreen.
CONFUSABLE = [
    ("0", "O"), ("0", "D"), ("O", "D"), ("0", "Q"), ("1", "I"), ("1", "L"),
    ("I", "L"), ("1", "T"), ("5", "S"), ("8", "B"), ("2", "Z"), ("6", "G"),
    ("4", "A"), ("7", "T"), ("U", "V"), ("M", "N"), ("H", "N"), ("C", "G"),
    ("E", "F"), ("P", "R"), ("3", "8"), ("9", "G"),
    ("8", "S"),  # blurry laser markings (out of focus or motion)
    ("M", "H"),  # small silkscreen: the middle of the M disappears ('for STH32')
]
_CONF = {frozenset(p) for p in CONFUSABLE}
CONFUSABLE_COST = 0.35


def normalize(s: str) -> str:
    """Upper-case ASCII, make dashes and spaces consistent, drop most punctuation."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.upper().replace("&", " AND ")
    s = re.sub(r"[‐-―_/\\|]", " ", s)
    s = re.sub(r"[^A-Z0-9.+\- ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(s: str) -> str:
    """Keep only letters and digits. Used for part numbers."""
    return re.sub(r"[^A-Z0-9]", "", normalize(s))


def words(s: str) -> list[str]:
    return [w.strip(".-+") for w in normalize(s).split() if w.strip(".-+")]


# ---- Distances ----
def levenshtein(a: str, b: str) -> int:
    if _RFLev is not None:
        return _RFLev.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ratio(a: str, b: str) -> float:
    """Similarity from 0 to 1 (1 minus the normalised edit distance)."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def ocr_distance(a: str, b: str) -> float:
    """Edit distance where swapping two easily confused characters only costs 0.35."""
    n, m = len(a), len(b)
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            if ca == cb:
                sub = 0.0
            elif frozenset((ca, cb)) in _CONF:
                sub = CONFUSABLE_COST
            else:
                sub = 1.0
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
    return prev[m]


def ocr_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return max(0.0, 1.0 - ocr_distance(a, b) / max(len(a), len(b)))


def best_substring_ratio(needle: str, hay: str) -> float:
    """Best match of ``needle`` against any same-length piece of ``hay``.

    Useful when OCR glued a part number to the text next to it.
    """
    if not needle:
        return 0.0
    if len(hay) <= len(needle):
        return ocr_ratio(needle, hay)
    L = len(needle)
    best = 0.0
    for w in (L - 1, L, L + 1):
        if w <= 0 or w > len(hay):
            continue
        for i in range(len(hay) - w + 1):
            r = ocr_ratio(needle, hay[i:i + w])
            if r > best:
                best = r
                if best == 1.0:
                    return 1.0
    return best


# ---- Matching names (dev boards, MCU cards) ----
class NameIndex:
    """Match OCR words against a list of product names made of several words.

    A name's score is the weighted average of how well each of its words
    matched, where rare words weigh more: sum(idf(w) * best_match(w)) /
    sum(idf(w)). So "Fusion", "EasyPIC" or "STM32" decide the match, while
    "for" or "board" barely count. Versions (v7/v8) and numbers must match
    almost exactly, because "EasyPIC v7" and "EasyPIC v8" only differ there.
    """

    STOPWORDS = {"FOR", "AND", "THE", "WITH", "BOARD", "KIT", "DEV"}
    # Multiply the score by this when a long, meaningful word of the name is
    # missing, so "FUSION FOR PIC" doesn't match the line "MCU CARD 10 FOR PIC".
    MISSING_WORD_PENALTY = 0.6

    def __init__(self, entries: Iterable[tuple[str, str]], min_word_sim: float = 0.75):
        """entries: (key, name) pairs. Several names can share a key (aliases)."""
        self.entries = [(k, n, words(n)) for k, n in entries]
        self.min_word_sim = min_word_sim
        # count how many products (not names) use each word, so aliases don't inflate counts
        per_key: dict[str, set[str]] = {}
        for k, _, ws in self.entries:
            per_key.setdefault(k, set()).update(ws)
        df = Counter(w for ws in per_key.values() for w in ws)
        N = max(1, len(per_key))
        self.idf = {w: (0.2 if w in self.STOPWORDS else math.log(1 + N / c))
                    for w, c in df.items()}

    @staticmethod
    def _is_strict(w: str) -> bool:
        # versions, plain numbers and short codes need an (almost) exact match
        return bool(re.fullmatch(r"V?\d+(\.\d+)?|[A-Z]?\d{1,3}[A-Z]?", w)) or len(w) <= 3

    def _word_sim(self, w: str, ocr_words: list[str], ocr_joined: str) -> float:
        best = 0.0
        strict = self._is_strict(w)
        for o in ocr_words:
            if o == w:
                return 1.0
            if strict:
                s = 1.0 if compact(o) == compact(w) else 0.0
            else:
                s = ocr_ratio(w, o)
            best = max(best, s)
        if best < 1.0 and not strict and len(w) >= 5:
            # OCR often glues words together ("FUSIONFOR"), so also search inside the joined
            # text
            best = max(best, best_substring_ratio(w, ocr_joined) - 0.05)
        return best if best >= self.min_word_sim else 0.0

    def score_all(self, ocr_text_words: list[str]) -> list[tuple[str, str, float]]:
        ow = [w for w in ocr_text_words if w]
        joined = "".join(compact(w) for w in ow)
        res = []
        for key, name, ws in self.entries:
            if not ws:
                continue
            num = den = 0.0
            missing_long = False
            for w in ws:
                idf = self.idf.get(w, 1.0)
                sim = self._word_sim(w, ow, joined)
                den += idf
                num += idf * sim
                if sim == 0 and len(w) >= 4 and w not in self.STOPWORDS:
                    missing_long = True
            s = num / den if den else 0.0
            if missing_long:
                s *= self.MISSING_WORD_PENALTY
            if s < 1.0 and len(ws) > 1:
                s = max(s, self._glued_score(ws, joined))
            res.append((key, name, s))
        return res

    def _glued_score(self, ws: list[str], joined: str) -> float:
        """Score a name whose words OCR glued together.

        For example "UNI-DSV8" for "UNI-DS v8": the version word never shows up
        on its own, so the word-by-word score drops (to 0.71). Instead we look
        for the whole name, without spaces, inside the joined line. Strict
        words (versions, numbers) must appear exactly, so "EASYPICV8" never
        counts for "EasyPIC v7".
        """
        cn = "".join(compact(w) for w in ws)
        if len(cn) < 6:
            return 0.0
        if any(compact(w) not in joined for w in ws if self._is_strict(w) and
               w not in self.STOPWORDS):
            return 0.0
        r = best_substring_ratio(cn, joined)
        return r - 0.03 if r >= 0.85 else 0.0  # slightly lower than a clean word match

    def top_k(self, ocr_text_words: list[str], k: int = 3) -> list[tuple[str, str, float]]:
        """Return the top k products, using the best-matching name of each."""
        best: dict[str, tuple[str, str, float]] = {}
        for key, name, s in self.score_all(ocr_text_words):
            if key not in best or s > best[key][2]:
                best[key] = (key, name, s)
        return sorted(best.values(), key=lambda t: -t[2])[:k]
