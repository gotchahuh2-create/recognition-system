"""Identify the development board by reading the name printed on it.

Dev boards print their name in large letters on the PCB ("Fusion for ARM
v8", "EasyPIC v8", "Clicker 4 for STM32"). We read all the text on the
photo and score every known board name with a fuzzy word match, where rare
words count more than common ones (see textmatch.NameIndex).

The board can be rotated by any multiple of 90 degrees. We score each
rotation separately (mixing upright and upside-down words only adds noise)
and keep the best one. We also return which rotation won, because the Click
step uses it to tell the mikroBUS sockets apart.

If OCR can't decide, you can pass a backup function with ``fallback=``
(for example an image-embedding matcher) and it will be used instead.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from .catalog import Catalog
from .ocr import (OcrBackend, OcrToken, get_backend, group_lines, read_text,
                  rotations_covered)
from .preprocess import Preprocessed
from .results import Candidate, Recognition, decide
from .textmatch import NameIndex, normalize, ocr_ratio, words

# Starting thresholds. Tune them on real board photos (goal: 95 % or better).
ACCEPT = 0.72
MARGIN = 0.08
MIN_TOKEN_CONF = 0.35


class BoardRecognizer:
    def __init__(self, catalog: Catalog | None = None,
                 backend: OcrBackend | str = "auto",
                 accept: float = ACCEPT, margin: float = MARGIN,
                 rotation_passes: Sequence[Sequence[int]] = ((0, 180), (90, 270)),
                 fallback: Callable[[np.ndarray], list[Candidate]] | None = None):
        self.catalog = catalog or Catalog.load()
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.accept, self.margin = accept, margin
        self.rotation_passes = rotation_passes
        self.fallback = fallback
        entries = []
        self._by_key = {}
        for b in self.catalog.boards:
            self._by_key[b.key] = b
            entries.append((b.key, b.name))
            entries += [(b.key, a) for a in b.aliases]
        self.index = NameIndex(entries)

    def score_tokens(self, tokens: Sequence[OcrToken], k: int = 3
                     ) -> tuple[list[Candidate], int | None]:
        """Score board names from text that was already read. Returns (top k, rotation).

        A board name must be found within ONE text line (or a line plus the
        line right below it, for titles printed on two lines). If we matched
        against all the text on the board, "STM32" on the chip or the "2" in
        "mikroBUS 2" could vote for the wrong board.
        """
        by_rot: dict[int, list[OcrToken]] = {}
        for t in tokens:
            if t.conf >= MIN_TOKEN_CONF:
                by_rot.setdefault(t.rotation, []).append(t)
        best: list[Candidate] = []
        best_rot = None
        for rot, toks in by_rot.items():
            scores: dict[str, tuple[float, str, str]] = {}   # key -> (score, alias, line)
            for seg in _segments(group_lines(toks)):
                ws = [w for t in seg for w in words(t.text)]
                if not ws:
                    continue
                for key, alias, s in self.index.score_all(ws):
                    if s > scores.get(key, (0.0,))[0]:
                        scores[key] = (s, alias, " ".join(t.text for t in seg))
            top = sorted(scores.items(), key=lambda kv: -kv[1][0])[:k]
            cands = [Candidate(key, self._by_key[key].name, s, self._by_key[key].pid,
                               {"matched_alias": alias, "line": line, "rotation": rot})
                     for key, (s, alias, line) in top]
            if cands and (not best or cands[0].score > best[0].score):
                best, best_rot = cands, rot
        return best, best_rot

    def _evidence(self, tokens: Sequence[OcrToken], cand: Candidate | None,
                  rot: int | None) -> list[dict]:
        if cand is None:
            return []
        name_words = {w for w in words(cand.detail.get("matched_alias", cand.name))
                      if len(w) > 2}
        ev = []
        for t in tokens:
            if t.rotation != rot:
                continue
            tw = words(t.text)
            if any(ocr_ratio(a, b) >= 0.8 for a in tw for b in name_words):
                ev.append(t.as_dict())
        return ev[:12]

    def recognise(self, pre: Preprocessed, tokens: list[OcrToken] | None = None
                  ) -> tuple[Recognition, list[OcrToken]]:
        """Identify the board in a preprocessed photo.

        Returns the result and every OCR word that was read. The MCU step
        reuses those words so we don't run OCR on the photo twice.
        """
        t0 = time.perf_counter()
        all_tokens: list[OcrToken] = list(tokens or [])
        done_rots = rotations_covered(self.backend, {t.rotation for t in all_tokens})
        cands, rot = self.score_tokens(all_tokens) if all_tokens else ([], None)
        decision, reason = decide(cands, self.accept, self.margin, "board name")

        for rots in self.rotation_passes:
            if decision is not None:
                break
            todo = [r for r in rots if r not in done_rots]
            if not todo:
                continue
            all_tokens += read_text(pre.image, self.backend, rotations=todo)
            done_rots |= rotations_covered(self.backend, todo)
            cands, rot = self.score_tokens(all_tokens)
            decision, reason = decide(cands, self.accept, self.margin, "board name")
        t_ocr = time.perf_counter()

        if decision is None and self.fallback is not None:
            fb = self.fallback(pre.image)
            if fb:
                cands = sorted(cands + fb, key=lambda c: -c.score)[:3]
                d2, r2 = decide(cands, self.accept, self.margin, "board")
                if d2 is not None:
                    decision, reason = d2, r2 + " (embedding fallback)"

        rec = Recognition(
            kind="board", candidates=cands, decision=decision, reason=reason,
            evidence=self._evidence(all_tokens, cands[0] if cands else None, rot),
            raw_reads=_title_like_lines(all_tokens, rot),
            timings_ms={"ocr+match": round((t_ocr - t0) * 1000)},
            rotations_read=sorted(done_rots),
        )
        return rec, all_tokens


def _segments(lines: list[list[OcrToken]]) -> list[list[OcrToken]]:
    """Return each line, plus each line joined with the one right below it.

    Two lines are joined when their letters are about the same height and
    they overlap sideways. This catches titles printed on two lines.
    """
    from .ocr import _reading_frame
    segs = [ln for ln in lines if ln]
    for a, b in zip(lines, lines[1:]):
        if not a or not b or a[0].rotation != b[0].rotation or a[0].variant != b[0].variant:
            continue
        ha = float(np.median([t.height for t in a]))
        hb = float(np.median([t.height for t in b]))
        ua = [_reading_frame(t) for t in a]
        ub = [_reading_frame(t) for t in b]
        dv = np.mean([v for _, v in ub]) - np.mean([v for _, v in ua])
        overlap = min(max(u for u, _ in ua), max(u for u, _ in ub)) - \
            max(min(u for u, _ in ua), min(u for u, _ in ub))
        if 0 < dv < 2.5 * max(ha, hb) and 0.6 < ha / hb < 1.6 and overlap > -2 * ha:
            segs.append(a + b)
    return segs


def _title_like_lines(tokens: Sequence[OcrToken], rot: int | None, n: int = 5) -> list[str]:
    """Return the biggest text lines. Useful for showing the user or logging unknown boards."""
    from .ocr import group_lines
    lines = [ln for ln in group_lines([t for t in tokens if t.rotation == rot]) if ln]
    lines.sort(key=lambda ln: -float(np.median([t.height for t in ln])))
    out = []
    for ln in lines[:n]:
        s = normalize(" ".join(t.text for t in ln))
        if len(s) >= 3 and s not in out:
            out.append(s)
    return out
