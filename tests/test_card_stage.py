"""Tests for the MCU card step: find the card -> crop and zoom -> read all text -> match.

* dataset import: text profiles and chip marking patterns;
* matching all the card's text (on words, no OCR engine): pin labels name a
  layout even without a title, a title that lost its digit is only
  recovered from distinctive labels, and common labels never override a
  title that was read;
* chip markings: stacked lines, ST's '32F...' without the prefix, Kinetis
  short codes;
* finding the card on the two real UNI-DS v8 photos and on a drawn card;
* one full read of a drawn card (needs Tesseract).
"""

import importlib.util
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.card_locate import CardLocator, crop_card
from vision.catalog import Catalog
from vision.mcu_ocr import McuRecognizer, fix_prefix
from vision.ocr import OcrToken
from vision.preprocess import preprocess

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "tests" / "real"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cat():
    return Catalog.load()


@pytest.fixture(scope="module")
def rec(cat):
    return McuRecognizer(cat, backend="tesseract", refine="off")


@pytest.fixture(scope="module")
def pre():
    return preprocess(np.zeros((400, 800, 3), np.uint8))


def toks(lines, variant="gray"):
    out = []
    for i, line in enumerate(lines):
        x = 10
        for w in line.split():
            out.append(OcrToken(w, 0.9, np.array([[x, 40 * i], [x + 12 * len(w), 40 * i],
                                                  [x + 12 * len(w), 40 * i + 20],
                                                  [x, 40 * i + 20]], float), variant=variant))
            x += 12 * len(w) + 15
    return out


def run(rec, pre, tokens, board=None):
    return rec.recognise(pre, tokens, rotations_read=(0, 90, 180, 270), board_key=board)


def mb_lines(cat, group, mb="MIKROBUS 3"):
    """Return the group's mikroBUS column as the card prints it: 'PC2 AN' per row."""
    m = cat.card_groups[group].text["pin_map"][mb]
    return [f"{p} {s}" for s, p in m.items()]


# ---- Import ----
def test_import_profiles_and_markings(cat):
    imp = _load("import_mcu_cards", ROOT / "scripts" / "import_mcu_cards.py")
    mp = imp.marking_patterns
    assert mp("ARM / STM32F405 / RGT6 / GH22B VG / CHN GH 406", "clear", "STM32F405RG") == ["STM32F405RGT6"]
    assert mp("NXP logo / MK64 / FN1M0V12 / 1N83J / XAAGBD", "clear", "MK64FN1M0VDC12") == ["MK64FN1M0V12"]
    assert mp("NXP logo / M20A7V / 4N86B / ABDNR", "clear", "MK20DX128VFM5") == ["M20A7V"]
    assert mp("ARM / 32F205RBT6 / GH20W VQ", "WRONG_CHIP", "STM32F401RB") == []
    assert mp("none - empty 48-pin footprint", None, None) == []
    g = cat.card_groups["mcu-card-29-for-stm32"]
    assert len(g.text["pin_map"]) == 5 and g.text["frame"] == "yellow"
    assert cat.card_by_key("mcu-card-for-kinetis-mk64fn1m0vdc12").markings == ["MK64FN1M0V12"]


# ---- Text profile ----
def test_pin_labels_name_layout_without_title(rec, pre, cat):
    _, card = run(rec, pre, toks(mb_lines(cat, "mcu-card-29-for-stm32")
                                 + mb_lines(cat, "mcu-card-29-for-stm32", "MIKROBUS 4")))
    assert card.group and card.group["key"] == "mcu-card-29-for-stm32"
    assert "no title read" in " ".join(card.group["why"])
    assert card.decision is None  # 10 cards share this layout, so the chip has to decide


def test_few_common_labels_do_not_name_a_layout(rec, pre):
    _, card = run(rec, pre, toks(["PA5 SCK", "PA6 MISO", "PA7 MOSI", "PB6 SCL"]))
    assert card.group is None


def test_dropped_title_digit_recovered_from_labels(rec, pre, cat):
    lines = ["MCU CARD for Kinetis"] + mb_lines(cat, "mcu-card-2-for-kinetis") \
        + mb_lines(cat, "mcu-card-2-for-kinetis", "MIKROBUS 1")
    _, card = run(rec, pre, toks(lines))
    assert card.group["key"] == "mcu-card-2-for-kinetis"
    # the title on its own keeps the layout it names
    _, card = run(rec, pre, toks(["MCU CARD for Kinetis"]))
    assert card.group["key"] == "mcu-card-for-kinetis"


def test_common_labels_never_flip_a_read_title(rec, pre):
    _, card = run(rec, pre, toks(["MCU CARD for STM32", "16-BIT PORT",
                                  "PA5 SCK", "PA6 MISO", "PA7 MOSI", "PB8 SCL", "PB9 SDA"]))
    assert card.group["key"] == "mcu-card-for-stm32"


def test_glued_signal_pin_labels():
    from vision.card_text import extract
    f = extract(toks(["PC2AN PEI3 RST", "SCK:PA5"]))
    assert "AN:PC2" in f.pairs and "RST:PE13" in f.pairs and "SCK:PA5" in f.pairs


# ---- Chip marking ----
def test_stacked_chip_lines_and_shop_marking(rec, pre):
    # Kinetis prints 'MK64' / 'FN1M0V12', not the ordering part number MK64FN1M0VDC12
    tk = toks(["MCU CARD for Kinetis"]) + toks(["", "", "MK64", "FN1M0V12", "1N83J"], "chip-otsu")
    mcu, card = run(rec, pre, tk)
    assert card.decision.key == "mcu-card-for-kinetis-mk64fn1m0vdc12"
    assert mcu.decision.key == "MK64FN1M0VDC12"


def test_st_marking_without_stm_prefix(rec, pre):
    assert fix_prefix("32F215RET6") == "STM32F215RET6"
    tk = toks(["MCU CARD 29 for STM32"]) + toks(["", "", "32F215RET6", "GH20P VG"], "chip-enhanced")
    mcu, card = run(rec, pre, tk)
    assert card.decision.key == "mcu-card-29-for-stm32-stm32f215re"


# ---- Finding the card ----
@pytest.mark.parametrize("name,center", [("uni-ds-v8_stm32f429zi_1.jpg", (196, 682)),
                                         ("uni-ds-v8_stm32f429zi_2.jpg", (211, 600))])
def test_locator_finds_card_on_real_photos(name, center):
    img = cv2.imread(str(REAL / name))
    locs = CardLocator().locate(img)
    assert locs and locs[0].method.startswith("frame")
    c = locs[0].quad.mean(0)
    assert np.hypot(*(c - center)) < 15
    assert 200 < max(locs[0].size_px) < 320  # the card, not the whole board
    cc = crop_card(img, locs[0])
    assert max(cc.image.shape[:2]) >= 900 and cc.zoom == pytest.approx(4.0, rel=0.05)


def test_locator_ignores_plain_board():
    img = np.full((900, 1200, 3), 25, np.uint8)
    cv2.rectangle(img, (40, 40), (1160, 860), (20, 200, 240), 4)  # just the board outline
    cv2.line(img, (300, 200), (700, 200), (20, 200, 240), 3)
    assert CardLocator().locate(img) == []


@pytest.fixture(scope="module")
def synth():
    return _load("make_synthetic_cards", ROOT / "scripts" / "make_synthetic_cards.py")


def _scene(synth, cat, key, frac, rot):
    import json
    data = json.loads((ROOT / "data" / "mcu_cards.json").read_text())
    c = next(x for x in data["cards"] if x["key"] == key)
    g = data["groups"][c["group"]]
    card = synth.render_card(g, dict(c, _marking_lines=synth.marking_lines(c)),
                             rng=random.Random(1))
    scene = synth.render_scene(card, rot90=rot, card_frac=frac, seed=3)
    return synth.degrade(scene, seed=3, tilt=2, persp=0.02, blur=0.7, out_long=2600,
                         noise=3, jpeg=90)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs Tesseract")
def test_rendered_card_end_to_end(synth, cat):
    from vision.loader import photo_from_array
    from vision.pipeline import Pipeline
    img = _scene(synth, cat, "mcu-card-29-for-stm32-stm32f215re", 0.4, 1)
    res = Pipeline(cat, "tesseract", layouts=False).run(photo_from_array(img, "syn.jpg"))
    assert res.mcu_source == "card" and res.card_crop is not None
    assert res.mcu_card.decision and res.mcu_card.decision.key == "mcu-card-29-for-stm32-stm32f215re"
    assert res.mcu.decision and res.mcu.decision.key == "STM32F215RE"
    d = res.as_dict()["card"]
    assert d["used"] and d["located"]["zoom"] >= 1.0 and d["read"]["chip"] is not None
