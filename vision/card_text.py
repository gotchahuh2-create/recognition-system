"""Match all the text read on a card against what each layout group should have printed on it.

For each group, ``data/mcu_cards.json`` has a ``text`` profile:

* ``pin_map``     which port pin each of the 12 signals on mikroBUS 1-5
                  goes to (11 groups, read from the 1600 px shop photos).
                  That's 60 labels, almost a unique fingerprint.
* ``port_pins``   every port pin label (37 groups also have the mikroBUS 1
                  fingerprint).
* ``strings``     other printed text ("CAN", "VLCD JMP", "HW REV. 1.01", ...).
* ``header_pins`` older 7th-gen cards: the 4 rows of pin headers.

What we look for in the OCR words:

``pair``   a signal label next to a port pin on the same line ("PC2 AN",
           or stuck together as "AN:PC2"). This is the strongest clue,
           because that's how the card prints them.
``pin``    a port pin label anywhere on the card.
``string`` any other printed text (fuzzy match that allows for OCR errors).

Each clue is weighted by how rare it is across groups:
``log(1 + G / groups_with_it)``. "SCK:PA5" is on most STM32 cards, so it
says little. "INT:PD14" is on just one, so it says a lot. A group's
evidence ``E`` is the total weight of the clues it has. Clues that the
group does NOT have are added up too (``against``). That way a card where
only 3 of 20 pins fit loses to one where 18 fit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .catalog import CardGroup
from .ocr import OcrToken, group_lines
from .textmatch import compact, ocr_ratio

SIGNALS = ("AN", "RST", "CS", "SCK", "MISO", "MOSI", "PWM", "INT", "RX", "TX", "SCL", "SDA")
_SIG_FIX = {"5CK": "SCK", "SCX": "SCK", "M1SO": "MISO", "MIS0": "MISO", "M0SI": "MOSI",
            "MOS1": "MOSI", "PVM": "PWM", "PWN": "PWM", "1NT": "INT", "RSI": "RST",
            "R5T": "RST", "C5": "CS", "5CL": "SCL", "5DA": "SDA", "SCI": "SCL", "TK": "TX"}
# what port pin labels look like on MikroE cards: PA5, PE13, RD15, PTB16, GP162, P1.2,
# ADC_A3
_PIN = re.compile(r"^(?:P[A-K]\d{1,2}|PT[A-E]\d{1,2}|R[A-K]\d{1,2}|GP\d{3}|P\d{1,2}\.?\d|"
                  r"ADCA\d)$")
# fix common OCR mistakes in a pin label: first P or R, then a port letter, then digits
_DIG = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "L": "1", "S": "5",
                      "B": "8", "Z": "2", "G": "6"})

W_PAIR, W_PIN, W_STRING = 1.0, 0.5, 0.8
DISTINCT_DF = 3
MIN_STRING_RATIO = 0.85


def norm_signal(s: str) -> str | None:
    c = compact(s)
    c = _SIG_FIX.get(c, c)
    return c if c in SIGNALS else None


def norm_pin(s: str) -> str | None:
    """Clean up a port pin label: 'PEI3' -> 'PE13', 'P1.2' stays. Returns None if it isn't a
    pin label.
    """
    raw = s.strip().upper().replace(",", ".")
    raw = re.sub(r"[^A-Z0-9.]", "", raw)
    if not raw or len(raw) > 6:
        return None
    if raw.startswith("ADC"):
        c = "ADCA" + raw[4:].translate(_DIG) if len(raw) > 4 else raw
        return c if _PIN.match(c) else None
    if raw.startswith("PT") and len(raw) >= 4:
        c = raw[:3] + raw[3:].translate(_DIG)
    elif raw.startswith("GP"):
        c = "GP" + raw[2:].translate(_DIG)
    elif raw[0] in "PR8" and len(raw) >= 3:
        head = "P" if raw[0] == "8" else raw[0]
        port = raw[1].replace("0", "D").replace("O", "D").replace("8", "B").replace("5", "S")
        c = head + port + raw[2:].translate(_DIG)
    else:
        return None
    return c if _PIN.match(c) else None


def split_glued(s: str) -> list[tuple[str | None, str | None]]:
    """Split a signal and pin that OCR glued together ('ANPC2' or 'PC2AN') into [(signal,
    pin)].
    """
    c = compact(s)
    out = []
    for sig in sorted(SIGNALS, key=len, reverse=True):
        if c.startswith(sig) and norm_pin(c[len(sig):]):
            out.append((sig, norm_pin(c[len(sig):])))
        if c.endswith(sig) and norm_pin(c[:-len(sig)]):
            out.append((sig, norm_pin(c[:-len(sig)])))
    return out


@dataclass
class TextFeatures:
    pairs: set[str] = field(default_factory=set)  # like "INT:PD14"
    pins: set[str] = field(default_factory=set)
    words: list[str] = field(default_factory=list)  # lines without spaces, plus single words
    signals: int = 0  # how many signal labels we saw


def extract(tokens: Sequence[OcrToken]) -> TextFeatures:
    f = TextFeatures()
    for line in group_lines(tokens):
        items = []
        for t in line:
            for w in t.text.split():
                glued = split_glued(w)
                if glued:
                    for sig, pin in glued:
                        f.pairs.add(f"{sig}:{pin}")
                        f.pins.add(pin)
                        f.signals += 1
                    items.append(("x", None))
                    continue
                sig, pin = norm_signal(w), norm_pin(w)
                if sig:
                    f.signals += 1
                    items.append(("s", sig))
                elif pin:
                    f.pins.add(pin)
                    items.append(("p", pin))
                else:
                    items.append(("x", None))
        # a signal goes with the pin right next to it on the line
        for i, (k, v) in enumerate(items):
            if k != "s":
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(items) and items[j][0] == "p":
                    f.pairs.add(f"{v}:{items[j][1]}")
        joined = "".join(compact(t.text) for t in line)
        if len(joined) >= 3:
            f.words.append(joined)
        f.words += [compact(t.text) for t in line if len(compact(t.text)) >= 3]
    return f


@dataclass
class ProfileScore:
    group: str
    evidence: float  # total weight of the clues this group has
    against: float  # weight of pins / pairs we read that this group doesn't have
    matched: list[str]
    # signal:pin pairs we matched that appear on at most DISTINCT_DF layouts
    distinctive: list[str] = field(default_factory=list)

    @property
    def net(self) -> float:
        return self.evidence - 0.5 * self.against


class CardTextMatcher:
    def __init__(self, groups: dict[str, CardGroup]):
        self.groups = groups
        self.g_pairs: dict[str, set[str]] = {}
        self.g_pins: dict[str, set[str]] = {}
        self.g_strings: dict[str, list[str]] = {}
        for k, g in groups.items():
            t = g.text or {}
            pairs = {f"{s}:{p}" for sig in (t.get("pin_map") or {}).values()
                     for s, p in sig.items() if p}
            pins = set(t.get("port_pins") or []) | set(t.get("header_pins") or []) | set(g.pins)
            pins = {p for p in (norm_pin(x) or x for x in pins)}
            self.g_pairs[k], self.g_pins[k] = pairs, pins
            self.g_strings[k] = [compact(s) for s in t.get("strings") or [] if len(compact(s)) >= 3]
        n = max(len(groups), 1)

        def idf(sets: Iterable[set]) -> dict[str, float]:
            df: dict[str, int] = {}
            for s in sets:
                for x in s:
                    df[x] = df.get(x, 0) + 1
            return {x: math.log(1 + n / d) for x, d in df.items()}

        self.df_pair: dict[str, int] = {}
        for ps in self.g_pairs.values():
            for x in ps:
                self.df_pair[x] = self.df_pair.get(x, 0) + 1
        self.idf_pair = idf(self.g_pairs.values())
        self.idf_pin = idf(self.g_pins.values())
        self.idf_str = idf(set(v) for v in self.g_strings.values())

    def has_profile(self, key: str) -> bool:
        return bool(self.g_pairs.get(key) or self.g_pins.get(key) or self.g_strings.get(key))

    def score(self, feats: TextFeatures, keys: Iterable[str] | None = None
              ) -> list[ProfileScore]:
        keys = list(keys) if keys is not None else list(self.groups)
        # only keep clues that appear in at least one profile (the rest is noise or board
        # text)
        known_pairs = {p for p in feats.pairs if p in self.idf_pair}
        known_pins = {p for p in feats.pins if p in self.idf_pin}
        out = []
        for k in keys:
            ev = ag = 0.0
            hit: list[str] = []
            gp, gn = self.g_pairs.get(k, set()), self.g_pins.get(k, set())
            for p in known_pairs:
                if p in gp:
                    ev += W_PAIR * self.idf_pair[p]
                    hit.append(p)
                elif gp:  # only a group with a pin map can contradict a pair
                    ag += W_PAIR * self.idf_pair[p]
            for p in known_pins:
                if p in gn:
                    ev += W_PIN * self.idf_pin[p]
                    if not any(h.endswith(":" + p) for h in hit):
                        hit.append(p)
                elif gp:  # we know all its pins, so an unknown pin counts against it
                    ag += W_PIN * self.idf_pin[p]
            for s in self.g_strings.get(k, []):
                best = max((1.0 if s in w else ocr_ratio(s, w) for w in feats.words), default=0)
                if best >= MIN_STRING_RATIO:
                    ev += W_STRING * self.idf_str.get(s, 1.0) * best
                    hit.append(f"'{s}'")
            dist = [p for p in hit if ":" in p and self.df_pair.get(p, 99) <= DISTINCT_DF]
            out.append(ProfileScore(k, ev, ag, hit, dist))
        out.sort(key=lambda s: -s.net)
        return out
