"""Tests for identifying MCU cards using the card dataset (data/mcu_cards.json).

These work on words directly (no OCR engine needed): the OCR output is
given as text lines, so the tests check the matching rules themselves:

* the title only names a layout GROUP; a group with many cards never picks
  one from the title alone;
* the number and family in "MCU CARD [N] for <FAMILY>" must match exactly;
* pin labels tell apart groups that share a title (SiBRAIN for ATmega);
* the chip marking, limited to the group's MCUs, picks the card;
* a PID label decides directly; a chip from outside the group is a conflict.
"""

import numpy as np
import pytest

from vision.catalog import Catalog
from vision.mcu_ocr import McuRecognizer, fix_prefix
from vision.ocr import OcrToken
from vision.preprocess import preprocess


@pytest.fixture(scope="module")
def rec():
    return McuRecognizer(Catalog.load(), backend="tesseract", refine="off")


@pytest.fixture(scope="module")
def pre():
    return preprocess(np.zeros((400, 800, 3), np.uint8))


def toks(lines):
    out = []
    for i, line in enumerate(lines):
        x = 10
        for w in line.split():
            out.append(OcrToken(w, 0.9, np.array([[x, 40 * i], [x + 12 * len(w), 40 * i],
                                                  [x + 12 * len(w), 40 * i + 20],
                                                  [x, 40 * i + 20]], float)))
            x += 12 * len(w) + 15
    return out


def run(rec, pre, lines, board=None):
    return rec.recognise(pre, toks(lines), rotations_read=(0, 90, 180, 270), board_key=board)


def key(r):
    return r.decision.key if r.decision else None


def test_dataset_loaded():
    c = Catalog.load()
    assert len(c.mcu_cards) >= 207 and len(c.card_groups) == 62
    g = c.card_groups["mcu-card-for-stm32"]
    assert len(g.cards) == 19 and "SiBRAIN for STM32" in g.aliases
    # an old name's suffix merges into the existing part, no duplicate entry
    parts = [m.part for m in c.mcus]
    assert "STM32F407VG" in parts and "STM32F407VGT6" not in parts
    assert "uni-ds-v8" in c.card_by_key("mcu-card-for-stm32-stm32f429zi").boards


def test_title_alone_does_not_pick_one_of_many(rec, pre):
    m, card = run(rec, pre, ["MCU CARD 29 for STM32", "16-BIT PORT"])
    assert card.decision is None and card.group["key"] == "mcu-card-29-for-stm32"
    assert card.group["cards"] == 10 and len(card.candidates) == 3
    assert m.decision is None


def test_title_plus_chip_picks_card(rec, pre):
    m, card = run(rec, pre, ["MCU CARD 29 for STM32", "16-BIT PORT", "STM32F405RGT6"])
    assert key(m) == "STM32F405RG"
    assert key(card) == "mcu-card-29-for-stm32-stm32f405rg"


def test_number_must_match(rec, pre):
    # "MCU CARD for STM32" (19 cards) vs "MCU CARD 2 for STM32" (1 card)
    _, card = run(rec, pre, ["MCU CARD 2 for STM32"])
    assert key(card) == "mcu-card-2-for-stm32-stm32f042k6"
    _, card = run(rec, pre, ["MCU CARD for STM32"])
    assert card.decision is None and card.group["key"] == "mcu-card-for-stm32"


def test_family_must_match(rec, pre):
    m, card = run(rec, pre, ["SiBRAIN for STM32L4", "16-BIT PORT"])
    assert key(card) == "sibrain-for-stm32l4r5zi" and key(m) == "STM32L4R5ZI"


def test_glued_title_as_on_real_photo(rec, pre):
    _, card = run(rec, pre, ["MCU CARDFOR STM32"], board="uni-ds-v8")
    assert card.group["key"] == "mcu-card-for-stm32" and card.decision is None


def test_single_card_group_implies_mcu(rec, pre):
    m, card = run(rec, pre, ["MCU CARD for MSP432", "8-BIT PORT"])
    assert key(card) == "mcu-card-for-msp432-msp432p401r" and key(m) == "MSP432P401R"


@pytest.mark.parametrize("pins,group", [("PE0 PE1 PE4 PE5", "sib-atmega-b"),
                                        ("PD2 PD3 PD0 PD1", "sib-atmega-a")])
def test_pin_labels_separate_same_title(rec, pre, pins, group):
    _, card = run(rec, pre, ["SiBRAIN for ATmega", "8-BIT PORT", pins])
    assert card.group["key"] == group and card.group["groups"] == [group]


def test_pid_label(rec, pre):
    m, card = run(rec, pre, ["MCU CARD for STM32", "MIKROE-3474"])
    assert key(card) == "mcu-card-for-stm32-stm32f429zi" and key(m) == "STM32F429ZI"


def test_chip_outside_group_is_a_conflict(rec, pre):
    m, card = run(rec, pre, ["MCU CARD 29 for STM32", "PIC18F65K40 -I/PT"])
    assert key(m) == "PIC18F65K40"  # we trust the chip for the MCU
    assert card.decision is None and "no card" in card.reason


def test_chip_only_unique_card(rec, pre):
    _, card = run(rec, pre, ["STM32F446RET6"])
    assert key(card) == "mcu-card-29-for-stm32-stm32f446re"


def test_board_filters_cards(rec, pre):
    c = Catalog.load()
    _, card = run(rec, pre, ["MCU CARD for STM32"], board="uni-ds-v8")
    assert all("uni-ds-v8" in c.card_by_key(x.key).boards for x in card.candidates)


def test_fix_prefix_keeps_known_prefix():
    assert fix_prefix("MSP432P401R") == "MSP432P401R"
    assert fix_prefix("5TM32F407ZG") == "STM32F407ZG"
