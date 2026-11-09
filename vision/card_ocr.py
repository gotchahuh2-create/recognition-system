"""Work out which MCU card is in the photo.

The data comes from ``data/mcu_cards.json`` (207 cards in 62 layout groups,
taken from mikroe.com). The important thing about this data: many cards
print the same title. "MCU CARD for STM32" is printed on 19 different
cards, and "MCU CARD 29 for STM32" on 10. Cards in the same group look
identical except for the chip soldered on them (and the PID sticker). So
we identify the card in two steps:

1. **Which group**, from the text printed on the card:
   * The title, split into parts: ``MCU CARD [N] for <FAMILY>`` or
     ``SiBRAIN for <FAMILY>`` (the product was renamed, so both work).
     The number and family must match exactly. A fuzzy match on the whole
     string can't tell "MCU CARD for STM32" from "MCU CARD 2 for STM32",
     or "SiBRAIN for STM32" from "SiBRAIN for STM32L4".
   * The port box ("16-BIT PORT" / "8-BIT PORT") as a sanity check.
   * The port pin labels next to mikroBUS 1. These separate groups with
     the same title (ATmega2560 cards: PD2 PD3 PD0 PD1, ATmega6450 cards:
     PE0 PE1 PE4 PE5).
   * Older 6th/7th-gen cards: the logo text ("EasyMx PRO v7"), by name.
   * The dev board: cards that don't support this board are dropped.

2. **Which card** in that group:
   * A PID label ``MIKROE-1234`` read anywhere means that card.
   * If the group has only one card, it's that card (and so that MCU).
   * Otherwise use the chip marking: the MCU step only considers the
     group's MCUs, and the card with the chosen MCU is the answer.
   * If none of that works, don't guess. List the group's cards and let
     the user pick. Never choose one of 19 identical-looking cards from
     the title alone.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .card_text import CardTextMatcher, ProfileScore, extract
from .catalog import Catalog, CardGroup, McuCard
from .ocr import OcrToken, group_lines
from .results import Candidate
from .textmatch import NameIndex, best_substring_ratio, compact, ocr_ratio, words

MIN_TOKEN_CONF = 0.3
GROUP_ACCEPT = 0.75  # same as the old card-title threshold
GROUP_MARGIN = 0.05
PORT_MISMATCH = 0.7  # multiply the group score by this if the port box disagrees
PIN_BONUS = 0.04  # bonus for each distinctive mikroBUS 1 pin label found (max 3)
# Settings for matching all the other text on the card (vision/card_text.py).
# These were tuned on synthetic cards.
PROFILE_STEP = 0.025  # same title on several groups: +/- this per point of evidence
PROFILE_CAP = 0.15
PROFILE_MIN = 6.0  # no title read: evidence needed to suggest a group
PROFILE_RATIO = 1.5  # ...and it must be this many times the runner-up's
DROPPED_DIGIT = 0.9  # for numbered groups when the title is read without its number

# characters OCR often confuses inside the fixed words of a title
_FOR = r"F[O0DQ]R"
_CARD_RE = re.compile(r"C[A4]RD([0-9OIL]{0,2}?)" + _FOR + r"([A-Z0-9]+)")
_SIB_RE = re.compile(r"S[I1L]BR[A4][I1L]N" + _FOR + r"([A-Z0-9]+)")
_PORT_RE = re.compile(r"(16|1G|I6|L6|8|B)[-]?B[I1L]T[-]?P[O0D]RT")
_PID_RE = re.compile(r"M[I1L]KR[O0]E[-]?([0-9OIlSB]{3,5})")
_DIGIT_FIX = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})


@dataclass
class TitleRead:
    kind: str  # "card" or "sibrain"
    number: int | None
    family: str  # family name as spelled in the catalog, e.g. "PIC24/dsPIC33"
    text: str  # what was read (the line without spaces)
    token: OcrToken | None = None
    score: float = 1.0  # how well the family matched


@dataclass
class GroupScore:
    group: CardGroup
    score: float
    why: list[str] = field(default_factory=list)
    token: OcrToken | None = None


@dataclass
class CardEvidence:
    """Everything the card step read. Used by McuRecognizer."""
    groups: list[GroupScore]                     # best first
    titles: list[TitleRead] = field(default_factory=list)
    port_width: str | None = None
    pid_cards: list[McuCard] = field(default_factory=list)
    board_filtered: bool = False
    # matches on all the card's text, best first
    profile: list[ProfileScore] = field(default_factory=list)

    def decided_groups(self) -> list[GroupScore]:
        """Return the top group plus every group tied with it.

        Ties happen when the same title is printed on several layouts. The card
        must be one of these groups.
        """
        if not self.groups or self.groups[0].score < GROUP_ACCEPT:
            return []
        top = self.groups[0].score
        return [g for g in self.groups if top - g.score < GROUP_MARGIN]


class CardRecognizer:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.cards = {c.key: c for c in catalog.mcu_cards}
        self.groups = catalog.card_groups
        fams = {g.family for g in self.groups.values() if g.family}
        # longest first, so "STM32L4" is tried before "STM32" and "PIC32" before "PIC"
        self.families = sorted(fams, key=lambda f: -len(compact(f)))
        legacy = [(g.key, g.title) for g in self.groups.values() if not g.family]
        self.legacy_index = NameIndex(legacy) if legacy else None
        # cards that have no group (from the CSV or products.db): match by name
        loose = [(c.key, c.name) for c in catalog.mcu_cards if c.group not in self.groups]
        self.loose_index = NameIndex(loose) if loose else None
        self.text_matcher = CardTextMatcher(self.groups)

    # ---- Reading the title ----
    def _family(self, rest: str) -> tuple[str, float] | None:
        best = None
        for f in self.families:
            fc = compact(f)
            head = rest[:len(fc)]
            if len(head) < max(3, len(fc) - 1):
                continue
            r = 1.0 if head == fc else ocr_ratio(fc, head)
            if r >= 0.8 and (best is None or r > best[1] + 1e-9):
                best = (f, r)
        return best

    def read_titles(self, lines: list[list[OcrToken]]) -> list[TitleRead]:
        out = []
        for i, line in enumerate(lines):
            texts = [compact(t.text) for t in line]
            # the title may be split over two lines ("MCU CARD 29" / "for STM32")
            variants = ["".join(texts)]
            if i + 1 < len(lines):
                variants.append(variants[0] + "".join(compact(t.text) for t in lines[i + 1]))
            if len(texts) > 1:
                # For vertical titles OCR sometimes flips one word but not the other, so
                # the words come out in reverse order (e.g. 'for STH32' + 'MCU CARD').
                variants.append("".join(reversed(texts)))
            for j, s in enumerate(variants):
                for kind, rx in (("card", _CARD_RE), ("sibrain", _SIB_RE)):
                    for m in rx.finditer(s):
                        if kind == "card":
                            num_s, rest = m.group(1), m.group(2)
                            num = int(num_s.translate(_DIGIT_FIX)) if num_s else None
                        else:
                            num, rest = None, m.group(1)
                        fam = self._family(rest)
                        if fam is None:
                            continue
                        tok = max(line, key=lambda t: len(t.text))
                        out.append(TitleRead(kind, num, fam[0], m.group(0), tok,
                                             fam[1] * (0.97 if j else 1.0)))
                if out and j == 0:
                    break  # found it on this line alone, so don't also try joining the next line
        # remove duplicates (the same title can show up in two line joins)
        seen, uniq = set(), []
        for t in sorted(out, key=lambda t: -t.score):
            k = (t.kind, t.number, t.family)
            if k not in seen:
                seen.add(k)
                uniq.append(t)
        return uniq

    @staticmethod
    def _port_width(joined: str) -> str | None:
        m = _PORT_RE.search(joined)
        if not m:
            return None
        return "8-BIT PORT" if m.group(1) in ("8", "B") else "16-BIT PORT"

    def _pid_cards(self, joined_lines: Iterable[str]) -> list[McuCard]:
        by_pid = {compact(c.pid): c for c in self.cards.values() if c.pid}
        found = []
        for s in joined_lines:
            for m in _PID_RE.finditer(s):
                c = by_pid.get("MIKROE" + m.group(1).translate(_DIGIT_FIX))
                if c and c not in found:
                    found.append(c)
        return found

    # ---- Scoring the groups ----
    def evidence(self, tokens: Sequence[OcrToken], board_key: str | None = None
                 ) -> CardEvidence:
        toks = [t for t in tokens if t.conf >= MIN_TOKEN_CONF or t.variant.startswith("refine")]
        lines = group_lines(toks)
        joined = ["".join(compact(t.text) for t in ln) for ln in lines]
        all_words = {compact(w) for t in toks for w in words(t.text)}
        titles = self.read_titles(lines)
        port = next((p for p in (self._port_width(s) for s in joined) if p), None)

        scores: dict[str, GroupScore] = {}

        def put(g: CardGroup, s: float, why: str, tok=None):
            cur = scores.get(g.key)
            if cur is None or s > cur.score:
                scores[g.key] = GroupScore(g, s, [why], tok)

        for t in titles:
            for g in self.groups.values():
                if not g.family:
                    continue
                for title in [g.title] + g.aliases:
                    k, n, f = _parse_title(title)
                    if k == t.kind and n == t.number and compact(f) == compact(t.family):
                        put(g, t.score, f"title '{title}' read as {t.text}", t.token)
                    elif (k == t.kind and t.number is None and n is not None
                          and compact(f) == compact(t.family)):
                        # Small titles often lose the digit ("MCU CARD 2 for Kinetis" is
                        # read as
                        # CARDFORKINETIS). Keep the numbered groups as options at a slightly
                        # lower
                        # score. Only the card's other labels can push them back up.
                        put(g, t.score * DROPPED_DIGIT,
                            f"title read as {t.text} (no number): '{title}' if a digit was lost",
                            t.token)
        if self.legacy_index is not None:
            ws = [w for t in toks for w in words(t.text)]
            for key, title, s in self.legacy_index.top_k(ws, 6):
                if s >= 0.8:
                    put(self.groups[key], s * 0.95, f"legacy card logo '{title}' ({s:.2f})")

        # port box: groups that don't match it lose points
        if port:
            for gs in scores.values():
                if gs.group.port_width and gs.group.port_width != port:
                    gs.score *= PORT_MISMATCH
                    gs.why.append(f"port box reads {port}, layout has {gs.group.port_width}")
                elif gs.group.port_width:
                    gs.why.append(f"port box {port}")

        # compare everything else printed on the card (mikroBUS pin labels, port
        # pins, other text) with each group's known text
        feats = extract(toks)
        profile = [p for p in self.text_matcher.score(feats) if p.evidence > 0]
        prof = {p.group: p for p in profile}
        # groups we only added because the title was read without its number
        dropped = {k for k, gs in scores.items() if gs.why and "(no number)" in gs.why[0]}
        exact = [k for k in scores if k not in dropped]
        if dropped and exact:
            best_exact = max((prof[k].net if k in prof else 0.0) for k in exact)
            tm = self.text_matcher

            def fits_exact(pair: str) -> bool:
                # A label that the title's own group has (or might have, if we only know
                # its pins) is not evidence against that group.
                pin = pair.split(":", 1)[1]
                return any(pair in tm.g_pairs.get(e, ()) or
                           (not tm.g_pairs.get(e) and pin in tm.g_pins.get(e, ()))
                           for e in exact)
            # if we have no pin data for the title's group, ask for more evidence
            need = 2 if all(tm.g_pins.get(e) for e in exact) else 4
            for k in dropped:
                p = prof.get(k)
                against = [x for x in (p.distinctive if p else []) if not fits_exact(x)]
                if (p and len(against) >= need and p.net >= PROFILE_MIN
                        and p.net >= 2 * max(best_exact, 0.1)):
                    scores[k].score = 1.0
                    scores[k].why.append("its own mikroBUS labels read: "
                                         + ", ".join(against[:6]) + f" (evidence {p.net:.1f} "
                                         f"vs {best_exact:.1f} for the title as read)")
                    for e in exact:
                        scores[e].score -= 0.1
                        scores[e].why.append(f"labels fit {self.groups[k].title} better")
        in_play = [k for k in exact if self.text_matcher.has_profile(k)]
        if len(exact) > 1 and in_play and any(k in prof for k in exact):
            # Same title on several groups: the groups whose labels were NOT seen lose
            # the points the best one gained.
            nets = {k: (prof[k].net if k in prof else 0.0) for k in exact}
            for k in exact:
                gs = scores[k]
                if not self.text_matcher.has_profile(k):
                    continue
                other = max(v for kk, v in nets.items() if kk != k)
                d = max(-PROFILE_CAP, min(PROFILE_CAP, PROFILE_STEP * (nets[k] - other)))
                if abs(d) < 1e-6:
                    continue
                gs.score = min(1.0, gs.score + d)
                if k in prof and prof[k].matched:
                    gs.why.append(f"card text {', '.join(prof[k].matched[:6])}"
                                  f"{' …' if len(prof[k].matched) > 6 else ''} "
                                  f"(evidence {prof[k].net:.1f})")
                else:
                    gs.why.append("its card labels not seen")
        elif len(exact) == 1:
            k = exact[0]
            if k in prof and prof[k].matched:
                scores[k].why.append(f"card text agrees: {', '.join(prof[k].matched[:6])}")
        if not scores and profile:
            # No title read: the labels alone can pick a group when there are many of
            # them and they clearly point to one group.
            top = profile[0]
            second = profile[1].net if len(profile) > 1 else 0.0
            if top.net >= PROFILE_MIN and top.net >= PROFILE_RATIO * max(second, 0.1):
                for p in profile[:3]:
                    if p.net <= 0:
                        break
                    g = self.groups[p.group]
                    put(g, 0.55 + 0.4 * (1 - math.exp(-p.net / 8.0)),
                        f"card text matches {g.title}: {', '.join(p.matched[:6])}"
                        f"{' …' if len(p.matched) > 6 else ''} (evidence {p.net:.1f}, no title read)")
                if port:
                    for gs in scores.values():
                        if gs.group.port_width and gs.group.port_width != port:
                            gs.score *= PORT_MISMATCH

        board_filtered = False
        if board_key and self.catalog.board_has_card_data(board_key):
            keep = {k for k, gs in scores.items()
                    if any(board_key in self.cards[c].boards or not self.cards[c].boards
                           for c in gs.group.cards)}
            if keep and keep != set(scores):
                for k in set(scores) - keep:
                    del scores[k]
                board_filtered = True

        ranked = sorted(scores.values(), key=lambda g: (-g.score, len(g.group.cards)))
        return CardEvidence(ranked, titles, port, self._pid_cards(joined), board_filtered,
                            profile[:5])

    def loose_candidates(self, tokens: Sequence[OcrToken]) -> list[Candidate]:
        """Match cards that have no layout group (seed CSV / products.db) by name."""
        if self.loose_index is None:
            return []
        ws = [w for t in tokens if t.conf >= MIN_TOKEN_CONF for w in words(t.text)]
        if not any(ocr_ratio("CARD", w) >= 0.75 or best_substring_ratio("CARD", compact(w)) >= 0.75
                   for w in ws):
            return []
        return [Candidate(k, self.cards[k].name, s, self.cards[k].pid,
                          {"mcu": self.cards[k].mcu, "via": "card name"})
                for k, _, s in self.loose_index.top_k(ws, 3) if s > 0]

    def cards_of(self, groups: Iterable[GroupScore], board_key: str | None = None
                 ) -> list[McuCard]:
        out = []
        check_board = bool(board_key) and self.catalog.board_has_card_data(board_key)
        for gs in groups:
            for k in gs.group.cards:
                c = self.cards[k]
                if check_board and c.boards and board_key not in c.boards:
                    continue
                out.append(c)
        return out


    def all_cards(self, board_key: str | None = None) -> list[McuCard]:
        """Every card that has a group (only those that support the board, if we know it)."""
        check = bool(board_key) and self.catalog.board_has_card_data(board_key)
        return [c for c in self.cards.values() if c.group in self.groups
                and not (check and c.boards and board_key not in c.boards)]


def _parse_title(title: str) -> tuple[str | None, int | None, str]:
    m = re.match(r"^(MCU CARD|SiBRAIN)\s+(?:(\d+)\s+)?for\s+(\S+)$", title.strip(), re.I)
    if not m:
        return None, None, title
    return ("card" if m.group(1).upper() == "MCU CARD" else "sibrain",
            int(m.group(2)) if m.group(2) else None, m.group(3))
