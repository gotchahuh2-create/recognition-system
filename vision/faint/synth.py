"""Check chip markings by simulating how each candidate would look in this photo.

Why: when the card title only tells us the layout group (up to 19 cards
share one PCB), the chip marking is the only way to tell the cards apart.
OCR needs readable letters, and enlarging (bicubic or super-resolution)
can't bring back detail the camera or JPEG threw away. But we don't
actually need to know "what does it say". We need to know "which of these
N known markings is it", and that can be answered at a lower resolution
than reading. So we draw each candidate marking, blur and shrink it the way
the camera would (lens blur at the photo's ORIGINAL pixel size, sampling
down to that size, then the same enlargement the chip crop got), and
compare it with the faint reader's text map.

Steps (one call per chip):

1. Text map: the lightness residual from ``vision.faint.features``
   (lighting removed, normalised), flipped so text is positive, and zero
   outside the middle of the chip (the pins are very high-contrast).
2. Line positions: peaks in the row profile of the map, at each 90 degree
   rotation.
3. For each candidate line: draw it in a stroke font (like a laser marker
   uses) at several sizes and widths, simulate the photo, and compare with
   normalised cross-correlation in a band around each line. Keep the best
   over lines, rotations and sizes.
4. Decision: accept only if the best candidate scores high enough AND beats
   every other candidate by ``margin``. Cards in one group share most of
   their marking ('MZ2048EFH' vs 'MZ1024EFE'), so the margin is the real
   test. It was tuned on the synthetic resolution tests
   (``eval/synth_eval.py``), where the markings are drawn in a DIFFERENT
   font than the one used here, so the method isn't tested on its own
   drawings.

This never reads text. It can only choose one of the given candidates, or
give up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

# tuned with eval/synth_eval.py (see DECISIONS.md)
MIN_SCORE = 0.45
MARGIN = 0.06
PSF_NATIVE = 0.7  # camera blur (sigma) in original photo px (phone + JPEG)
# capital letter height as a share of the estimated letter height
CAPS = (0.55, 0.7, 0.85, 1.0, 1.2)


@dataclass
class Hypothesis:
    key: str                  # card key
    name: str
    lines: list[str]          # marking lines that differ between candidates


@dataclass
class VerifyResult:
    scores: dict[str, float]
    decision: str | None
    margin: float
    reason: str
    detail: dict = field(default_factory=dict)

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: -kv[1])

    def as_dict(self) -> dict:
        return {"decision": self.decision, "margin": round(self.margin, 3),
                "reason": self.reason,
                "scores": {k: round(v, 3) for k, v in self.ranked()[:6]}, **self.detail}


# ---- Candidate markings ----
_PIC32 = re.compile(r"^PIC32(M[XZKM][A-Z0-9]*?)(\d{3})?$")


def marking_lines(part: str, observed: list[str] | tuple = ()) -> list[str]:
    """Return the marking line(s) that hold the part number, laid out the way the vendor prints
    them.

    Based on common Microchip / ST layouts. Markings from the shop photos
    in the dataset take priority when we have them.

    PIC32MZ2048EFH144 -> ['MZ2048EFH']  ('PIC32' is the logo line and
    '144-I/PH' the next line; the whole group shares both).
    STM32F429ZI -> ['STM32F429ZI']  (we don't know the 'T6' suffix, so the
    line is matched as a prefix).
    """
    if observed:
        return [o for o in observed if len(o) >= 4][:2]
    p = re.sub(r"[^A-Z0-9]", "", part.upper())
    m = _PIC32.match(p)
    if m:
        return [m.group(1)]
    return [p]


def hypotheses_for(cards) -> list[Hypothesis]:
    out = []
    for c in cards:
        if getattr(c, "empty", False) or not c.mcus:
            continue
        out.append(Hypothesis(c.key, c.name, marking_lines(c.mcus[0], c.markings)))
    return out


# ---- Drawing and simulating the photo ----
def render(s: str, cap: float, sx: float = 1.0) -> np.ndarray:
    """Draw text in a stroke font (ink = 1.0), cropped to the text plus one letter height of
    margin.
    """
    scale = cap / 21.0  # Hershey simplex capitals are ~21 px tall at scale 1
    th = max(1, int(round(cap * 0.14)))
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    img = np.zeros((int(cap * 3), w + int(2 * cap)), np.uint8)
    cv2.putText(img, s, (int(cap), int(cap * 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, th,
                cv2.LINE_AA)
    img = img.astype(np.float32) / 255.0
    if abs(sx - 1.0) > 1e-3:
        img = cv2.resize(img, None, fx=sx, fy=1.0, interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(img > 0.05)
    if len(ys) == 0:
        return img
    m = int(cap * 0.5)
    return img[max(0, ys.min() - m):ys.max() + m + 1, max(0, xs.min() - m):xs.max() + m + 1]


def degrade(img: np.ndarray, f: float, psf: float = PSF_NATIVE) -> np.ndarray:
    """Simulate the photo: how ``img`` would look in a chip crop enlarged ``f`` times from the
    photo.

    Blur at the original size, shrink to the original pixel size, then
    enlarge again with cubic scaling (like ``region.rectify`` /
    ``enhance.upscale``).
    """
    if f <= 1.0:
        return cv2.GaussianBlur(img, (0, 0), max(0.5, psf))
    b = cv2.GaussianBlur(img, (0, 0), psf * f)
    h, w = img.shape[:2]
    small = cv2.resize(b, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


# ---- What the photo shows ----
def text_map(resid_L: np.ndarray, body: np.ndarray, polarity: int, glyph: float) -> np.ndarray:
    """Make a signed text map: positive where the text is, only inside the middle of the chip."""
    k = max(3, int(glyph))
    inner = cv2.erode(body.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    m = resid_L.astype(np.float32) * (1 if polarity >= 0 else -1)
    m[~inner] = 0
    return m


def line_rows(m: np.ndarray, glyph: float, max_lines: int = 6) -> list[int]:
    prof = np.clip(m, 0, None).sum(1)
    w = max(3, int(glyph / 2))
    prof = np.convolve(prof, np.ones(w) / w, "same")
    if prof.max() <= 0:
        return []
    pk = [i for i in range(1, len(prof) - 1)
          if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.3 * prof.max()]
    keep: list[int] = []
    for i in sorted(pk, key=lambda i: -prof[i]):
        if all(abs(i - j) > 0.8 * glyph for j in keep):
            keep.append(i)
    return sorted(keep[:max_lines])


# ---- Matching ----
def render_cells(s: str, cap: float, pitch: float) -> np.ndarray:
    """Draw a line in a fixed-width stroke font.

    Each character is centred in a ``pitch``-wide cell, so candidates of the
    same length line up cell by cell (laser markers use near fixed-width
    fonts).
    """
    p = max(2, int(round(pitch)))
    m = int(round(cap * 0.5))
    out = np.zeros((int(cap * 2.0), m * 2 + p * len(s)), np.float32)
    for i, ch in enumerate(s):
        if ch == " ":
            continue
        g = render(ch, cap)
        ys, xs = np.nonzero(g > 0.05)
        if not len(ys):
            continue
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if g.shape[1] > p:
            g = cv2.resize(g, (p, g.shape[0]), interpolation=cv2.INTER_AREA)
        y0 = int(round(cap * 0.5 + cap - g.shape[0])) if ch not in "gjpqy" else int(round(cap * 0.5))
        y0 = max(0, min(out.shape[0] - g.shape[0], y0))
        x0 = m + i * p + (p - g.shape[1]) // 2
        out[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = np.maximum(
            out[y0:y0 + g.shape[0], x0:x0 + g.shape[1]], g)
    return out


def _valid_ncc(st: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Template matching (TM_CCOEFF_NORMED), but flat windows get -1.

    The zeroed area outside the chip is flat, and OpenCV scores a flat
    window as a perfect match.
    """
    r = cv2.matchTemplate(st, t, cv2.TM_CCOEFF_NORMED)
    h, w = t.shape
    s1 = cv2.boxFilter(st, -1, (w, h), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.boxFilter(st * st, -1, (w, h), normalize=True, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    var = (s2 - s1 * s1)[:r.shape[0], :r.shape[1]]
    ref = float(np.var(st[st != 0])) if np.any(st != 0) else 0.0
    r[(var < 0.05 * ref) | ~np.isfinite(r)] = -1.0
    return r


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


class MarkingVerifier:
    PITCHES = (0.6, 0.72, 0.85, 1.0)  # character width / capital height

    def __init__(self, min_score: float = MIN_SCORE, margin: float = MARGIN,
                 psf: float = PSF_NATIVE, caps=CAPS, pitches=PITCHES):
        self.min_score, self.margin, self.psf = min_score, margin, psf
        self.caps, self.pitches = caps, pitches

    def _align(self, strings: list[str], strips, glyph, f):
        """Find where the shared marking line is and how big, using ALL candidates.

        Returns the best (score, strip index, cap, pitch, x, y).
        """
        best = (-1.0, None)
        for c in self.caps:
            for pr in self.pitches:
                cap = c * glyph
                for s in strings:
                    t = degrade(render_cells(s, cap, pr * cap), f, self.psf)
                    if t.std() < 1e-6:
                        continue
                    for i, st in enumerate(strips):
                        if t.shape[0] > st.shape[0] or t.shape[1] > st.shape[1]:
                            continue
                        r = _valid_ncc(st, t)
                        _, mx, _, loc = cv2.minMaxLoc(r)
                        if mx > best[0]:
                            best = (float(mx), (i, cap, pr * cap, loc[0], loc[1]))
        return best

    def _discriminate(self, strings: list[str], strips, al, f) -> dict[str, float]:
        """Score each candidate only on the characters where the candidates differ.

        Uses the shared position found by ``_align`` (small shifts allowed).
        """
        i, cap, pitch, x, y = al
        st = strips[i]
        L = max(len(s) for s in strings)
        pad = [s.ljust(L) for s in strings]
        diff = [k for k in range(L) if len({s[k] for s in pad}) > 1]
        m = int(round(cap * 0.5))
        p = max(2, int(round(pitch)))
        temps = {s: degrade(render_cells(ps, cap, pitch), f, self.psf) for s, ps in zip(strings, pad)}
        h, w = next(iter(temps.values())).shape
        cols = np.zeros(w, bool)
        for k in diff:
            cols[max(0, m + k * p - p // 4):min(w, m + (k + 1) * p + p // 4)] = True
        if not cols.any():
            return {s: 0.0 for s in strings}
        out = {}
        sh = max(1, int(round(cap * 0.25)))
        for s, t in temps.items():
            best = -1.0
            for dy in range(-sh, sh + 1):
                for dx in range(-sh, sh + 1):
                    yy, xx = y + dy, x + dx
                    if yy < 0 or xx < 0 or yy + h > st.shape[0] or xx + w > st.shape[1]:
                        continue
                    win = st[yy:yy + h, xx:xx + w]
                    best = max(best, _ncc(win[:, cols], t[:, cols]))
            out[s] = best
        return out

    def verify(self, resid_L: np.ndarray, body: np.ndarray, polarity: int, glyph: float,
               native_scale: float, hyps: list[Hypothesis], turns=(0, 90, 180, 270)
               ) -> VerifyResult:
        """resid_L: lightness residual of the straightened chip (features.resid['lab'][...,0]).
        native_scale: chip crop px per original photo px.
        """
        if len(hyps) < 2:
            return VerifyResult({}, None, 0.0, "fewer than two candidate markings")
        base = text_map(resid_L, body, polarity, glyph)
        strips: list[np.ndarray] = []
        rows_info = {}
        for t in turns:
            m = np.ascontiguousarray(np.rot90(base, (t // 90) % 4))
            rows = line_rows(m, glyph)
            rows_info[t] = rows
            for y in rows:
                y0, y1 = max(0, int(y - 1.4 * glyph)), min(m.shape[0], int(y + 1.4 * glyph))
                strips.append(np.ascontiguousarray(m[y0:y1]))
        if not strips:
            return VerifyResult({}, None, 0.0, "no marking lines found on the chip")
        detail = {"lines": {str(k): v for k, v in rows_info.items()},
                  "glyph_native_px": round(glyph / max(native_scale, 1e-6), 2)}
        # the part number line (the first marking line of each candidate)
        line_of = {h.key: h.lines[0] for h in hyps}
        strings = sorted(set(line_of.values()))
        if len(strings) < 2:
            return VerifyResult({}, None, 0.0, "candidates share the same marking", detail)
        align_score, al = self._align(strings, strips, glyph, native_scale)
        detail["line_match"] = round(align_score, 3)
        if al is None or align_score < self.min_score:
            return VerifyResult({}, None, 0.0,
                                f"no line on the chip looks like the part-number line "
                                f"(best {align_score:.2f} < {self.min_score})", detail)
        per = self._discriminate(strings, strips, al, native_scale)
        scores = {h.key: per[line_of[h.key]] for h in hyps}
        ranked = sorted(per.items(), key=lambda kv: -kv[1])
        (s1_str, s1), (_, s2) = ranked[0], ranked[1]
        margin = s1 - s2
        winners = [h for h in hyps if line_of[h.key] == s1_str]
        if margin < self.margin:
            return VerifyResult(scores, None, margin,
                                f"candidate markings indistinguishable at this resolution "
                                f"(differing characters: best {s1:.2f}, margin {margin:.3f} "
                                f"< {self.margin}; marking glyphs ~{detail['glyph_native_px']} px "
                                f"in the photo)", detail)
        if len(winners) > 1:
            return VerifyResult(scores, None, margin,
                                f"marking '{s1_str}' is shared by {len(winners)} cards", detail)
        return VerifyResult(scores, winners[0].key, margin,
                            f"{winners[0].name}: rendered marking '{s1_str}' matches the "
                            f"differing characters best ({s1:.2f}, margin {margin:.3f})", detail)
