"""Tests for low-resolution card photos: package check, snapping to the frame,
title / marking parsing, and the marking verifier that simulates the photo.
"""

import cv2
import numpy as np

from vision import package
from vision.card_locate import CardLocator, _snap_to_frame, yellow_mask
from vision.catalog import Catalog
from vision.faint.synth import (MarkingVerifier, Hypothesis, degrade, marking_lines,
                                render_cells)
from vision.mcu_ocr import _stacked, candidate_strings
from vision.ocr import OcrToken
from vision.textmatch import ocr_ratio


# ---- Package size ----
def test_package_ratio_separates_100_and_144_pins():
    # real photos: chip / card side 0.298-0.314 (LQFP-144, 20 mm)
    for r in (0.298, 0.314):
        assert package.compatible(144, r) and not package.compatible(100, r)
    assert package.compatible(100, 0.215) and not package.compatible(144, 0.215)
    assert package.compatible(None, 0.3) is None


def test_package_filter_drops_mx_cards_of_the_shared_pic32_title():
    cat = Catalog.load()
    cards = [c for c in cat.mcu_cards if c.group in ("mcu-card-for-pic32", "mcu-card-3-for-pic32")]
    kept, dropped = package.filter_cards(cards, 0.30)
    assert kept and all(package.card_pins(c) == 144 for c in kept)
    assert dropped and all(package.card_pins(c) == 100 for c in dropped)
    # a ratio that fits no card is ignored instead of emptying the group
    assert package.filter_cards(cards, 0.05)[0] == cards


# ---- Frame ----
def _card_scene(touching: bool) -> np.ndarray:
    img = np.full((600, 600, 3), (40, 40, 40), np.uint8)
    yel = (0, 200, 230)
    if touching:  # board's socket outline, joined to the card frame by a strip of labels
        cv2.rectangle(img, (130, 110), (470, 490), yel, 4)
        cv2.rectangle(img, (130, 110), (470, 150), yel, -1)
    cv2.rectangle(img, (160, 160), (440, 440), (20, 20, 20), -1)
    cv2.rectangle(img, (160, 160), (440, 440), yel, 6)
    for y in range(200, 420, 30):           # label boxes inside: yellow, with gaps
        cv2.rectangle(img, (380, y), (420, y + 18), yel, -1)
    return img


def test_snap_moves_an_overreaching_quad_onto_the_card_frame():
    img = _card_scene(True)
    m = yellow_mask(img)
    q = np.array([[130, 110], [470, 110], [470, 490], [130, 490]], float)
    snapped, inset = _snap_to_frame(m, q)
    assert inset > 0.02
    assert np.abs(snapped - [[157, 157], [443, 157], [443, 443], [157, 443]]).max() <= 6


def test_snap_leaves_a_clean_frame_alone():
    img = _card_scene(False)
    m = yellow_mask(img)
    q = np.array([[157, 157], [443, 157], [443, 443], [157, 443]], float)
    snapped, inset = _snap_to_frame(m, q)
    assert inset <= 0.02 and np.abs(snapped - q).max() <= 3


def test_locator_finds_the_card_inside_a_touching_outline():
    locs = CardLocator().frame_candidates(_card_scene(True))
    assert locs
    w, h = locs[0].size_px
    assert 270 <= w <= 300 and 270 <= h <= 300


# ---- Text ----
def _tok(text, x, y, h=20.0, rot=0, variant="chip-x"):
    w = 0.6 * h * len(text)
    poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    return OcrToken(text, 0.9, poly, rotation=rot, variant=variant)


def test_three_stacked_pic32_lines_join():
    lines = [[_tok("PIC32", 0, 0)], [_tok("MZ2048EFH", 0, 30)], [_tok("144-I/PH", 0, 60)]]
    joined = [s for s, _ in _stacked(lines)]
    assert "PIC32MZ2048EFH144IPH" in joined
    cands = candidate_strings([t for ln in lines for t in ln])
    assert any(s.startswith("PIC32MZ2048EFH144") for s in cands)


def test_m_h_confusable_in_titles():
    assert ocr_ratio("STM32", "STH32") >= 0.9


def test_reversed_title_tokens_are_read():
    from vision.card_ocr import CardRecognizer
    cr = CardRecognizer(Catalog.load())
    line = [_tok("for STH32", 0, 0, variant="card-gray"), _tok("MCU CARD", 200, 0, variant="card-gray")]
    reads = cr.read_titles([line])
    assert reads and reads[0].family == "STM32" and reads[0].number is None


# ---- Marking verifier ----
def test_marking_lines():
    assert marking_lines("PIC32MZ2048EFH144") == ["MZ2048EFH"]
    assert marking_lines("STM32F429ZI") == ["STM32F429ZI"]
    assert marking_lines("MK64FN1M0VDC12", ["MK64FN1M0V12"]) == ["MK64FN1M0V12"]


def _synthetic_chip(text: str, cap: float, f: float, side: int = 600) -> np.ndarray:
    img = np.zeros((side, side), np.float32)
    line = render_cells(text, cap, 0.85 * cap)
    y, x = side // 2 - line.shape[0] // 2, (side - line.shape[1]) // 2
    img[y:y + line.shape[0], x:x + line.shape[1]] = line
    return degrade(img, f) * 6.0 + np.random.default_rng(0).normal(0, 0.3, img.shape).astype(np.float32)


def test_verifier_decides_when_the_marking_is_resolved():
    hyps = [Hypothesis("a", "A", ["MZ2048EFH"]), Hypothesis("b", "B", ["MZ1024EFE"])]
    body = np.ones((600, 600), bool)
    obs = _synthetic_chip("MZ2048EFH", 30.0, 1.5)  # letters ~20 px tall in the photo
    r = MarkingVerifier().verify(obs, body, +1, 30.0, 1.5, hyps)
    assert r.decision == "a", r.reason


def test_verifier_abstains_when_the_marking_is_below_resolution():
    hyps = [Hypothesis("a", "A", ["MZ2048EFH"]), Hypothesis("b", "B", ["MZ1024EFE"])]
    body = np.ones((600, 600), bool)
    obs = _synthetic_chip("MZ2048EFH", 30.0, 12.0)  # letters ~2.5 px tall: the size in real photos
    r = MarkingVerifier().verify(obs, body, +1, 30.0, 12.0, hyps)
    assert r.decision is None
