"""Guess the MCU's package size by comparing the chip to the card it sits on.

In low-resolution photos the text on the chip is unreadable (after a phone
photo goes through a messenger app, each letter can be about 2 px wide).
The chip itself is still easy to see as a dark square, though, and all
8th-gen MCU cards have the same outline. So the ratio "chip side / card
side" tells us the package size, and for square QFP chips the package size
tells us the pin count:

    TQFP/LQFP-100  14 mm  -> ratio ~0.22
    TQFP/LQFP-144  20 mm  -> ratio ~0.31
    LQFP-176       24 mm  -> ratio ~0.37

This can't tell apart cards that share one PCB layout (same board, same
chip size). It does help with different layouts that print the same title.
For example "SiBRAIN for PIC32" appears on both MCU CARD for PIC32 (100-pin
PIC32MX) and MCU CARD 3 for PIC32 (144-pin PIC32MZ). On one real photo this
check cut the list from 10 possible cards to 4.

The filter is gentle: it only removes a card when the size clearly doesn't
fit, and it never removes every card.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# Physical side length of an 8th-gen MCU card / SiBRAIN, in mm.
# This is an estimate based on three real UNI-DS v8 photos (LQFP-144 chips
# with a 20 mm body gave ratios of 0.298 to 0.314). Check it again if cards
# of a different size show up.
CARD_SIDE_MM = 65.0

# Body size (mm) of square QFP packages for each pin count (standard JEDEC sizes).
QFP_BODY_MM = {32: 7.0, 44: 10.0, 48: 7.0, 52: 10.0, 64: 10.0, 80: 12.0, 100: 14.0,
               128: 14.0, 144: 20.0, 176: 24.0, 208: 28.0}

# How far the measured ratio may be from the expected one, on a log scale.
# 0.15 allows about +/-16 %. A 100-pin and a 144-pin chip differ by 36 %.
TOL = 0.15


def expected_ratio(pins: int | None) -> float | None:
    mm = QFP_BODY_MM.get(int(pins)) if pins else None
    return mm / CARD_SIDE_MM if mm else None


def compatible(pins: int | None, ratio: float, tol: float = TOL) -> bool | None:
    """Does this pin count fit the measured ratio?

    Returns None if we don't know the chip size for that pin count.
    """
    e = expected_ratio(pins)
    if e is None or ratio <= 0:
        return None
    return abs(math.log(ratio / e)) <= tol


def pins_for(ratio: float, tol: float = TOL) -> list[int]:
    return sorted(p for p in QFP_BODY_MM if compatible(p, ratio, tol))


def card_pins(card) -> int | None:
    p = getattr(card, "pins", None) or (getattr(card, "detail", None) or {}).get("pins")
    try:
        return int(p) if p else None
    except (TypeError, ValueError):
        return None


def filter_cards(cards: Sequence, ratio: float | None, tol: float = TOL) -> tuple[list, list]:
    """Split cards into (kept, dropped) by the measured ratio.

    Cards with an unknown pin count are always kept.
    """
    cards = list(cards)
    if not ratio or not cards:
        return cards, []
    kept = [c for c in cards if compatible(card_pins(c), ratio, tol) is not False]
    if not kept:
        return cards, []  # the size fits no card at all, so don't trust it
    dropped = [c for c in cards if c not in kept]
    return kept, dropped


def chip_ratio(chip_quad, card_shape: Iterable[int], pad: float = 0.02) -> float:
    """Return chip side / card side.

    ``card_shape`` is the (h, w) of the straightened card crop.
    ``crop_card`` adds ``pad`` of extra margin on each side, which we remove here.
    """
    import numpy as np
    q = np.asarray(chip_quad, float)
    side = float(np.mean([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]))
    h, w = list(card_shape)[:2]
    card = (h + w) / 2.0 / (1 + 2 * pad)
    return side / card if card > 0 else 0.0
