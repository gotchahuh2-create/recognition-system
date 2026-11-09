"""Step 8: compare the OCR results from all image variants and combine them.

1. Each variant's words are grouped into lines (``ocr.group_lines``) and
   the spaces removed ('STM32F4 07' -> 'STM32F407').
2. Lines from different variants are grouped together if they're in the
   same place on the chip (same rotation, overlapping boxes) or read
   almost the same text. One group = one printed line on the chip.
3. A variant's reliability is how much it agrees with the OTHER variants
   (the average best similarity of its lines to theirs). A variant that
   imagines text in noise agrees with nobody.
4. For each group, vote letter by letter: every line is lined up against
   the most typical line of the group (with difflib), and each letter
   votes with weight = variant reliability x OCR confidence. Inserted and
   missing letters vote too. The result is the agreed line.
5. Each agreed line becomes an ``OcrToken`` (variant ``chip-faint-fused``,
   with confidence = share of variants that support it). These are added
   NEXT TO the raw words, and the catalog matcher sees both. So combining
   can only add a candidate; it can never hide one that a single variant
   read.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..ocr import OcrToken, group_lines
from ..textmatch import compact


@dataclass
class Line:
    variant: str
    text: str  # without spaces
    conf: float
    poly: np.ndarray  # 4x2 box in chip px
    rotation: int


@dataclass
class FusedLine:
    text: str
    support: float  # share of variants that have a line in this group
    members: list[tuple[str, str]] = field(default_factory=list)   # (variant, text)
    poly: np.ndarray | None = None
    rotation: int = 0

    def as_dict(self) -> dict:
        return {"text": self.text, "support": round(self.support, 2),
                "members": [f"{v}: {t}" for v, t in self.members]}


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def _box(toks: list[OcrToken]) -> np.ndarray:
    p = np.vstack([t.poly for t in toks])
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0 = a.min(0); ax1, ay1 = a.max(0)
    bx0, by0 = b.min(0); bx1, by1 = b.max(0)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / small if small > 0 else 0.0


def lines_of(variant: str, tokens: list[OcrToken], min_len: int = 3) -> list[Line]:
    out = []
    for ln in group_lines(tokens):
        s = compact("".join(t.text for t in ln))
        if len(s) >= min_len:
            out.append(Line(variant, s, float(np.mean([t.conf for t in ln])), _box(ln),
                            ln[0].rotation))
    return out


def reliability(lines: list[Line]) -> dict[str, float]:
    byv = defaultdict(list)
    for l in lines:
        byv[l.variant].append(l)
    rel = {}
    for v, ls in byv.items():
        others = [o for o in lines if o.variant != v]
        if not others:
            rel[v] = 0.5
            continue
        rel[v] = float(np.mean([max(sim(l.text, o.text) for o in others) for l in ls]))
    return rel


def _vote(medoid: str, members: list[tuple[str, float]]) -> str:
    """Vote letter by letter, lined up against ``medoid``. members are (text, weight) pairs."""
    n = len(medoid)
    pos = [defaultdict(float) for _ in range(n)]  # letter, or '' for a missing letter
    ins = [defaultdict(float) for _ in range(n + 1)]  # text inserted before position i
    total = sum(w for _, w in members)
    for s, w in members:
        sm = difflib.SequenceMatcher(None, medoid, s, autojunk=False)
        inserted_at = set()
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    pos[i1 + k][s[j1 + k]] += w
            elif op == "replace":
                L = i2 - i1
                seg = s[j1:j2]
                for k in range(L):
                    pos[i1 + k][seg[k] if k < len(seg) else ""] += w
                if len(seg) > L:
                    ins[i2][seg[L:]] += w
                    inserted_at.add(i2)
            elif op == "delete":
                for k in range(i1, i2):
                    pos[k][""] += w
            elif op == "insert":
                ins[i1][s[j1:j2]] += w
                inserted_at.add(i1)
    out = []
    for i in range(n + 1):
        if ins[i]:
            best, wb = max(ins[i].items(), key=lambda kv: kv[1])
            if wb > 0.5 * total:
                out.append(best)
        if i < n and pos[i]:
            c, wc = max(pos[i].items(), key=lambda kv: kv[1])
            out.append(c)
    return "".join(out)


def fuse(per_variant: dict[str, list[OcrToken]], min_support: float = 0.25
         ) -> tuple[list[FusedLine], dict[str, float]]:
    lines = [l for v, toks in per_variant.items() for l in lines_of(v, toks)]
    if not lines:
        return [], {}
    rel = reliability(lines)
    nvar = max(1, len(per_variant))
    # group lines greedily, starting with the most reliable and confident ones
    lines.sort(key=lambda l: -(rel.get(l.variant, 0.5) * (0.3 + l.conf) * len(l.text)))
    clusters: list[list[Line]] = []
    for l in lines:
        home = None
        for c in clusters:
            h = c[0]
            if (h.rotation == l.rotation and _overlap(h.poly, l.poly) > 0.5) or sim(h.text, l.text) >= 0.75:
                home = c
                break
        if home is None:
            clusters.append([l])
        elif all(m.variant != l.variant for m in home):
            home.append(l)
    fused: list[FusedLine] = []
    for c in clusters:
        support = len({m.variant for m in c}) / nvar
        if support < min_support and len(c) < 2:
            continue
        texts = [m.text for m in c]
        medoid = max(texts, key=lambda t: sum(sim(t, o) for o in texts))
        weights = [(m.text, rel.get(m.variant, 0.5) * (0.3 + min(m.conf, 1.0))) for m in c]
        txt = _vote(medoid, weights) if len(c) > 1 else medoid
        fused.append(FusedLine(txt, support, [(m.variant, m.text) for m in c],
                               _box_of(c), c[0].rotation))
    fused.sort(key=lambda f: -f.support)
    return fused, {k: round(v, 3) for k, v in rel.items()}


def _box_of(c: list[Line]) -> np.ndarray:
    p = np.vstack([m.poly for m in c])
    x0, y0 = np.median(p[:, 0].reshape(-1, 4).min(1)), np.median(p[:, 1].reshape(-1, 4).min(1))
    x1, y1 = np.median(p[:, 0].reshape(-1, 4).max(1)), np.median(p[:, 1].reshape(-1, 4).max(1))
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def fused_tokens(fused: list[FusedLine]) -> list[OcrToken]:
    out = []
    for i, f in enumerate(fused):
        out.append(OcrToken(f.text, float(min(0.99, 0.5 + 0.5 * f.support)), f.poly.copy(),
                            f.rotation, "chip-faint-fused", ("fused", i)))
    return out
