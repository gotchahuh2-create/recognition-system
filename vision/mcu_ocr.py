"""Identify the MCU and its MCU card by reading text with OCR.

There are two places to read from (a photo may show either, both or neither):

* The chip marking, e.g. ``STM32F407ZGT6`` or ``PIC18F65K40 -I/PT``. This is
  tiny laser-engraved text, so we need a close-up or a high-resolution
  pass over tiles of the photo.
* The card's printed title, e.g. ``MCU CARD 10 for PIC``. The same title is
  printed on up to 19 different cards (see ``data/mcu_cards.json`` and
  ``vision/card_ocr.py``), so the title only tells us the *layout group*.
  We use it to narrow the part-number search to that group's MCUs, and
  the MCU we find then tells us the card. The title alone only gives the
  MCU when the group has a single card (or when we read a PID label).

Part numbers are matched letter by letter, and common OCR mix-ups (0/O,
1/I, 5/S ...) cost less. The marking may be *longer* than the catalog
entry, because markings include ordering suffixes (package, temperature,
revision). So ``STM32F407ZGT6`` matches the catalog's ``STM32F407ZG`` with
a score of 1.0.

Text that looks like a part number but isn't in the catalog is returned
in ``raw_reads``. That way the user (or whoever maintains the catalog) can
see what was read, rather than getting a wrong forced answer.

The MCU is optional (NECTO picks a default MCU for each board), so this
step gives up easily: a wrong MCU is worse than no MCU.
"""

from __future__ import annotations

import re
import time
from typing import Iterable, Sequence

import numpy as np

from . import package
from .card_ocr import CardEvidence, CardRecognizer
from .catalog import Catalog, McuCard
from .ocr import (OcrBackend, OcrToken, get_backend, group_lines, read_text,
                  rotations_covered)
from .preprocess import Preprocessed
from .refine import RefineConfig, refine
from .results import Candidate, Recognition, decide
from .textmatch import best_substring_ratio, compact, ocr_ratio

# Strict thresholds: a part number that is almost right is usually a
# different chip (F407 vs F405).
ACCEPT = 0.90
MARGIN = 0.05
CARD_ACCEPT = 0.75
MIN_TOKEN_CONF = 0.3

# Vendor prefixes of MCU part numbers. Used to fix misread prefixes and to
# spot text that looks like a part number but isn't in the catalog.
PREFIXES = ["STM32", "STM8", "DSPIC", "PIC", "ATMEGA", "ATTINY", "ATXMEGA", "ATSAM",
            "AT90", "MK", "MKL", "MKV", "TM4C", "LM3S", "LM4F", "LPC", "GD32", "EFM32",
            "NRF", "MSP430", "MSP432", "CEC", "FT9", "R7FA", "S32K", "CY8C", "RP2040",
            "ESP32"]
_PARTLIKE = re.compile(r"^(?=.*\d)(?=(?:.*[A-Z]){2})[A-Z0-9]{6,24}$")


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def fix_prefix(s: str) -> str:
    """Fix a misread vendor prefix: '5TM32F407' -> 'STM32F407'.

    Text that already starts with a known prefix is left alone (otherwise
    'MSP432P401R' would be "fixed" to 'MSP430P401R'). If several prefixes
    are close, the closest one wins.
    """
    if any(s.startswith(p) and len(s) > len(p) for p in PREFIXES):
        return s
    # ST prints many parts without the "STM", e.g. '32F215RET6'
    if re.match(r"^32[FLGHUW][0-9]{3}[A-Z0-9]{2}", s):
        return "STM" + s
    best, best_r = None, 0.75
    for p in sorted(PREFIXES, key=len, reverse=True):
        if len(p) < 3 or len(s) <= len(p):
            continue
        r = ocr_ratio(p, s[:len(p)])
        if r > best_r:
            best, best_r = p, r
    return best + s[len(best):] if best else s


def _stacked(lines: list[list[OcrToken]]) -> list[tuple[str, OcrToken]]:
    """Join chip marking lines that belong together.

    Chip markings are printed on 2-3 short lines ('STM32F405' / 'RGT6',
    'MK64' / 'FN1M0V12'). We join a line with the one below it when they
    have the same orientation, overlap sideways and are close together.
    """
    out = []

    def near(a, b) -> bool:
        ta, tb = a[0], b[0]
        if (ta.rotation, ta.variant) != (tb.rotation, tb.variant):
            return False
        ha = max(np.mean([t.height for t in a]), 1.0)
        ca = np.mean([t.center for t in a], axis=0)
        cb = np.mean([t.center for t in b], axis=0)
        return np.linalg.norm(cb - ca) <= 3.5 * ha

    def txt(line):
        return "".join(compact(t.text) for t in line)

    for i, a in enumerate(lines):
        for j in range(i + 1, min(i + 3, len(lines))):
            b = lines[j]
            if not near(a, b):
                continue
            sa, sb = txt(a), txt(b)
            if 3 <= len(sa) <= 16 and 2 <= len(sb) <= 12:
                out.append((sa + sb, max(a + b, key=lambda t: t.conf)))
                # Microchip PIC32 / dsPIC: the family is on a logo line, the part on the
                # next line, and the third line starts with the pin count
                # ('PIC32' / 'MZ2048EFH' / '144-I/PH' -> PIC32MZ2048EFH144...)
                for c in lines[j + 1:j + 3]:
                    sc = txt(c)
                    if near(b, c) and 2 <= len(sc) <= 12 and len(sa) <= 7:
                        out.append((sa + sb + sc, max(a + b + c, key=lambda t: t.conf)))
    return out


def candidate_strings(tokens: Iterable[OcrToken]) -> dict[str, OcrToken]:
    """Find text that looks like a part number. Returns {text: word it came from}.

    We look at single words, pairs of neighbouring words (markings are often
    split, e.g. 'STM32F407' 'ZGT6') and, for chip close-ups (words with a
    ``variant`` starting with 'chip-', see ``vision/card_reader.py``), also
    two stacked lines.
    """
    out: dict[str, OcrToken] = {}
    # the OCR confidence of re-read crops (vision/refine.py) isn't reliable, so keep them
    # all
    toks = [t for t in tokens if t.conf >= MIN_TOKEN_CONF or t.variant.startswith(("refine", "chip"))]
    chip_lines = group_lines([t for t in toks if t.variant.startswith("chip")])
    for s, tok in _stacked(chip_lines):
        s = fix_prefix(s)
        if _PARTLIKE.match(s) and s not in out:
            out[s] = tok
    for line in group_lines(toks):
        comps = [compact(t.text) for t in line]
        for i, c in enumerate(comps):
            seqs = [c]
            if i + 1 < len(comps):
                seqs.append(c + comps[i + 1])
            for s in seqs:
                s = fix_prefix(s)
                if _PARTLIKE.match(s) and (s not in out or line[i].conf > out[s].conf):
                    out[s] = line[i]
    return out


class McuRecognizer:
    def __init__(self, catalog: Catalog | None = None,
                 backend: OcrBackend | str = "auto",
                 accept: float = ACCEPT, margin: float = MARGIN,
                 card_accept: float = CARD_ACCEPT, hires_tile: int = 1400,
                 refine: str = "auto", refine_cfg: RefineConfig | None = None):
        """refine: "auto" runs the extra crop pass (vision/refine.py) only when the
        cheaper passes couldn't decide. Can also be "always" or "off".
        """
        self.catalog = catalog or Catalog.load()
        self.refine_mode = refine
        self.refine_cfg = refine_cfg or RefineConfig()
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.accept, self.margin, self.card_accept = accept, margin, card_accept
        self.hires_tile = hires_tile
        self.card_rec = CardRecognizer(self.catalog)
        self._cards = {c.key: c for c in self.catalog.mcu_cards}

    # ---- Matching part numbers ----
    def score_parts(self, cands: dict[str, OcrToken],
                    allowed: set[str] | None = None) -> list[Candidate]:
        res: list[Candidate] = []
        for m in self.catalog.mcus:
            if allowed is not None and m.compact not in allowed:
                continue
            need = m.compact
            nb = _bigrams(need)
            best, best_s, best_t = 0.0, None, None
            for s, tok in cands.items():
                # quick pre-check, so this stays fast with hundreds of MCUs
                if len(nb & _bigrams(s)) < 0.4 * len(nb):
                    continue
                if s.startswith(need):
                    r = 1.0
                else:
                    r = best_substring_ratio(need, s)
                    # a marking SHORTER than the part number was cut off, so lower its score
                    if len(s) < len(need):
                        r *= len(s) / len(need)
                if r > best or (r == best and best_t is not None and tok.conf > best_t.conf):
                    best, best_s, best_t = r, s, tok
            if best > 0:
                res.append(Candidate(m.part, m.part, best, m.pid,
                                     {"read": best_s, "vendor": m.vendor,
                                      "ocr_conf": round(best_t.conf, 3),
                                      "glyph_px": round(best_t.height, 1),
                                      "box": np.round(best_t.poly).astype(int).tolist()}))
        res.sort(key=lambda c: (-c.score, -c.detail.get("ocr_conf", 0)))
        return res[:3]

    def score_markings(self, cands: dict[str, OcrToken], cards: Iterable[McuCard],
                       allowed: set[str] | None = None) -> list[Candidate]:
        """Compare chip text with the marking each card shows in its shop photo
        (``McuCard.markings``).

        This covers chips whose marking isn't the part number, such as Kinetis
        'MK64FN1M0V12' for MK64FN1M0VDC12, 'M20A7V' for MK20DX128VFM5, or TI
        'TM4C1299NCZAD13'. Returns MCU part candidates.
        """
        best: dict[str, Candidate] = {}
        for c in cards:
            if not c.markings or not c.mcu or (allowed is not None and compact(c.mcu) not in allowed):
                continue
            for pat in c.markings:
                if len(pat) < 5:
                    continue
                for st, tok in cands.items():
                    if pat in st:
                        r = 1.0
                    else:
                        r = best_substring_ratio(pat, st)
                        if len(st) < len(pat):
                            r *= len(st) / len(pat)
                    if r < 0.8:
                        continue
                    cur = best.get(c.mcu)
                    if cur is None or r > cur.score:
                        m = self.catalog.mcu_by_part(c.mcu)
                        best[c.mcu] = Candidate(c.mcu, c.mcu, r, m.pid if m else None,
                                                {"read": st, "marking": pat, "via": "card marking",
                                                 "ocr_conf": round(tok.conf, 3),
                                                 "glyph_px": round(tok.height, 1),
                                                 "box": np.round(tok.poly).astype(int).tolist()})
        return sorted(best.values(), key=lambda c: -c.score)

    @staticmethod
    def merge_parts(a: list[Candidate], b: list[Candidate]) -> list[Candidate]:
        out: dict[str, Candidate] = {}
        for c in a + b:
            k = compact(c.key)
            if k not in out or c.score > out[k].score:
                out[k] = c
        return sorted(out.values(), key=lambda c: (-c.score, -c.detail.get("ocr_conf", 0)))[:3]

    def score_cards(self, tokens: Sequence[OcrToken]) -> list[Candidate]:
        """Match card names, for cards that have no layout group (from the CSV or products.db).

        Cards with a group are handled by ``card_ocr``.
        """
        return self.card_rec.loose_candidates(tokens)

    # ---- Deciding the card ----
    def decide_card(self, ev: CardEvidence, group_cards: list[McuCard],
                    mcu_dec: Candidate | None, parts: list[Candidate],
                    loose: list[Candidate], board_key: str | None,
                    conflict: str | None = None) -> Recognition:
        groups = ev.decided_groups()
        gscore = groups[0].score if groups else 0.0
        ginfo = None
        if groups:
            g = groups[0].group
            ginfo = {"key": g.key, "title": g.title,
                     "groups": [x.group.key for x in groups],
                     "cards": len(group_cards), "score": round(gscore, 3),
                     "why": groups[0].why,
                     **({"box": np.round(groups[0].token.poly).astype(int).tolist()}
                        if groups[0].token is not None else {})}

        def cand(c: McuCard, score: float, via: str) -> Candidate:
            return Candidate(c.key, c.name, score, c.pid,
                             {"mcu": c.mcu, "group": c.group, "via": via,
                              **({"empty": True, "compatible": c.mcus[:8]} if c.empty else {})})

        def rec(cands, dec, reason):
            r = Recognition("mcu_card", cands[:3], dec, reason)
            r.group = ginfo
            return r

        # 1. a PID label read on the card or a sticker
        if ev.pid_cards:
            pc = sorted(ev.pid_cards, key=lambda c: c not in group_cards)[0]
            c = cand(pc, 0.99, "pid label")
            return rec([c], c, f"PID label {pc.pid} read -> {pc.name}")

        mcu_part = compact(mcu_dec.key) if mcu_dec else None

        def carriers(cards):
            hit = [c for c in cards if mcu_part in {compact(p) for p in c.mcus}]
            full = [c for c in hit if not c.empty]
            return full or hit  # a card with a chip beats an empty one

        if not groups:
            if loose:  # cards without a group: match by name
                d, why = decide(loose, self.card_accept, self.margin, "MCU card")
                return rec(loose, d, why)
            if mcu_dec is None:
                why = ("no MCU card title read" if not ev.groups else
                       f"card title unclear: best layout {ev.groups[0].group.title} "
                       f"{ev.groups[0].score:.2f} < 0.75")
                return rec([], None, why)
            pool = self.card_rec.all_cards(board_key)
            hit = carriers(pool)
            cands = [cand(c, mcu_dec.score * 0.97, "chip marking only") for c in hit]
            if len(hit) == 1:
                return rec(cands, cands[0], f"{hit[0].name}: only card carrying "
                           f"{mcu_dec.key} (card title not read)")
            return rec(cands, None, (f"{len(hit)} cards carry {mcu_dec.key}; card title not "
                                     "read" if hit else f"no card in the catalog carries {mcu_dec.key}"))

        title = groups[0].group.title + (" (+%d look-alike layouts)" % (len(groups) - 1)
                                         if len(groups) > 1 else "")
        if conflict:
            cands = [cand(c, gscore * 0.5, "title") for c in group_cards]
            return rec(cands, None, conflict)
        if len(group_cards) == 1:
            c = cand(group_cards[0], gscore, "single-card layout")
            return rec([c], c, f"{title}: only card with this layout")
        if mcu_dec is not None:
            hit = carriers(group_cards)
            if len(hit) == 1:
                c = cand(hit[0], min(gscore, mcu_dec.score), "title + chip marking")
                rest = [cand(x, gscore * 0.5, "title") for x in group_cards if x is not hit[0]]
                return rec([c] + rest, c, f"{title} + chip {mcu_dec.key} -> {hit[0].name}")
            if hit:
                cands = [cand(x, min(gscore, mcu_dec.score), "title + chip marking") for x in hit]
                return rec(cands, None, f"{title}: {len(hit)} cards carry {mcu_dec.key} "
                                        f"({', '.join(x.name for x in hit[:3])})")
        # no usable chip text: rank the group's cards by how well their MCU was read
        score_of = {compact(p.key): p.score for p in parts}
        ranked = sorted(group_cards, key=lambda c: -max([score_of.get(compact(m), 0)
                                                          for m in c.mcus] or [0]))
        cands = [cand(c, gscore * (0.5 + 0.5 * max([score_of.get(compact(m), 0)
                                                     for m in c.mcus] or [0])), "title")
                 for c in ranked]
        return rec(cands, None, f"title '{groups[0].group.title}' is printed on "
                                f"{len(group_cards)} cards; the chip marking decides "
                                f"which one ({'not read' if not parts else 'unclear'})")

    def _unlisted(self, cands: dict[str, OcrToken], parts: list[Candidate]) -> list[str]:
        known = {c.detail.get("read") for c in parts if c.score >= self.accept}
        out = []
        for s in sorted(cands, key=lambda s: -cands[s].conf):
            if s in known:
                continue
            if any(s.startswith(p) for p in PREFIXES):
                out.append(s)
        return out[:5]

    # ---- Main entry point ----
    def recognise(self, pre: Preprocessed, tokens: list[OcrToken] | None = None,
                  hires: np.ndarray | None = None,
                  allowed_parts: Iterable[str] | None = None,
                  rotations_read: Iterable[int] = (),
                  board_key: str | None = None,
                  chip_ratio: float | None = None
                  ) -> tuple[Recognition, Recognition]:
        """Identify the MCU and the MCU card. Returns (mcu result, mcu_card result).

        tokens:         OCR words that were already read (e.g. by the board step).
        hires:          optional full-resolution image, used for a tiled pass
                        when the chip text is too small in the processed image.
        allowed_parts:  only consider MCUs that work with the detected board.
        rotations_read: page rotations the board step already read.
        board_key:      the identified dev board; cards that don't support it are
                        dropped.
        chip_ratio:     chip side / card side, measured on the card crop. Cards
                        whose chip can't be that size are dropped
                        (see ``vision.package``).
        """
        t0 = time.perf_counter()
        allowed = {compact(p) for p in allowed_parts} if allowed_parts else None
        toks = list(tokens or [])
        done = rotations_covered(self.backend, {t.rotation for t in toks} | set(rotations_read))
        missing = [r for r in (0, 90, 180, 270) if r not in done]
        if missing:
            toks += read_text(pre.image, self.backend, rotations=missing)

        scope = {"allowed": allowed}

        def evaluate(ts):
            cands = candidate_strings(ts)
            ev = self.card_rec.evidence(ts, board_key)
            gcards = self.card_rec.cards_of(ev.decided_groups(), board_key)
            gcards, dropped = package.filter_cards(gcards, chip_ratio)
            scope["package_dropped"] = dropped
            gparts = {compact(p) for c in gcards for p in c.mcus}
            # if we read a card title, only look at the MCUs of that group
            eff = (gparts & allowed if allowed is not None else gparts) if gparts else allowed
            scope.update(allowed=eff, ev=ev, gcards=gcards, restricted=bool(gparts))
            parts = self.score_parts(cands, eff)
            pool = gcards or self.card_rec.all_cards(board_key)
            parts = self.merge_parts(parts, self.score_markings(cands, pool, eff))
            cards = self.score_cards(ts)
            return cands, parts, cards

        cands, parts, cards = evaluate(toks)
        passes = ["full"]
        d, _ = decide(parts, self.accept, self.margin, "MCU part number")
        if d is None and hires is not None and max(hires.shape[:2]) > max(pre.image.shape[:2]) * 1.2:
            hi = read_text(hires, self.backend, rotations=(0, 90, 180, 270),
                           tile=self.hires_tile)
            # Convert boxes from the original photo to the processed photo, like all
            # other words (evidence boxes, letter size and the refine step need that).
            for t in hi:
                hom = np.hstack([t.poly, np.ones((len(t.poly), 1))]) @ pre.M.T
                t.poly = hom[:, :2] / hom[:, 2:3]
                t.variant = "hires-" + t.variant
            toks += hi
            cands, parts, cards = evaluate(toks)
            passes.append("hires-tiles")
            d, _ = decide(parts, self.accept, self.margin, "MCU part number")
        crops = []
        if self.refine_mode == "always" or (self.refine_mode == "auto" and d is None):
            # second pass: find text, cut it out, enlarge, clean up, read again
            seed_toks = [t for t in toks if not t.variant.startswith("refine")]
            def confident(new_toks):
                ps = self.score_parts(candidate_strings(new_toks), scope["allowed"])
                return decide(ps, self.accept, self.margin, "MCU part number")[0] is not None
            new, crops = refine(pre, self.backend, seed_toks, hires=hires, cfg=self.refine_cfg,
                                stop=confident if self.refine_mode == "auto" else None)
            if new:
                toks += new
                cands, parts, cards = evaluate(toks)
            passes.append(f"refine({len(crops)} crops)")
        t1 = time.perf_counter()

        mcu_dec, mcu_reason = decide(parts, self.accept, self.margin, "MCU part number")
        ev, gcards = scope["ev"], scope["gcards"]
        conflict = None
        if scope["restricted"] and mcu_dec is None:
            # The chip may clearly read as a part that no card in this group carries
            # (the title was misread, or the product was renamed). Trust the chip for
            # the MCU, and leave the card undecided.
            outside = self.score_parts(cands, allowed)
            od, oreason = decide(outside, self.accept, self.margin, "MCU part number")
            if od is not None:
                parts, mcu_dec, mcu_reason = outside, od, oreason
                conflict = (f"chip reads {od.key}, but no card with layout "
                            f"'{ev.decided_groups()[0].group.title}' carries it")
        elif scope["restricted"] and mcu_dec is not None:
            mcu_reason += f" (among the {len(gcards)} cards of layout '{ev.groups[0].group.title}')"
        card_rec = self.decide_card(ev, gcards, mcu_dec, parts, cards, board_key, conflict)
        if scope.get("package_dropped"):
            dropped = scope["package_dropped"]
            note = (f"chip body {chip_ratio:.2f} of the card side fits "
                    f"{'/'.join(map(str, package.pins_for(chip_ratio))) or '?'}-pin QFP: "
                    f"dropped {len(dropped)} card(s) with "
                    f"{'/'.join(sorted({str(package.card_pins(c)) for c in dropped}))} pins")
            card_rec.reason += f"; {note}"
            if card_rec.group is not None:
                card_rec.group["package"] = note

        # If we know the card and it has exactly one MCU soldered on, that's the MCU.
        card_dec = card_rec.decision
        if card_dec and card_dec.detail.get("mcu") and card_dec.detail.get("via") != "title + chip marking":
            part = card_dec.detail["mcu"]
            m = self.catalog.mcu_by_part(part)
            implied = card_dec.score * 0.95
            existing = next((c for c in parts if compact(c.key) == compact(part)), None)
            if existing:
                existing.score = max(existing.score, implied)
                existing.detail["also_from_card"] = card_dec.key
            elif allowed is None or compact(part) in allowed:
                parts.append(Candidate(part, part, implied, m.pid if m else None,
                                       {"from_card": card_dec.key}))
            parts.sort(key=lambda c: -c.score)
            parts = parts[:3]
            mcu_dec, mcu_reason = decide(parts, self.accept, self.margin, "MCU part number")
            if mcu_dec is not None and mcu_dec.detail.get("from_card"):
                mcu_reason = f"{mcu_dec.key} from the card: {card_rec.reason}"

        timings = {"ocr+match": round((t1 - t0) * 1000), "passes": passes}
        mcu_rec = Recognition("mcu", parts, mcu_dec, mcu_reason,
                              evidence=[{"read": c.detail.get("read"),
                                         "box": c.detail.get("box"),
                                         "glyph_px": c.detail.get("glyph_px")}
                                        for c in parts[:1] if c.detail.get("read")],
                              raw_reads=self._unlisted(cands, parts), timings_ms=timings,
                              crops=crops)
        card_rec.timings_ms = timings
        return mcu_rec, card_rec
