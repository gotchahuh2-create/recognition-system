"""Tests on real photos.

Two phone photos of the same UNI-DS v8 with an MCU CARD for STM32
(STM32F429ZIT6). The layout in data/layouts.json was calibrated from photo
1 only; photo 2 (board cut off at two edges, a strip of glare) was kept
aside for testing. A third photo shows a UNI-DS v8 with a SiBRAIN for
PIC32MZ2048EFH144 (layout MCU CARD 3 for PIC32).

All three are 960x1280 JPEGs sent through a messenger, so the chip text is
only about 3.5 px tall and can't be read (OCR needs the chip to be about
250 px wide, and here it's 71-77 px; see eval/synth_eval_synthetic.md).
What we DO require: the card's layout group, the package check, and never
a wrong card or MCU.

Main pass condition: the board is identified correctly. We also check that
the stored layout is placed using the board's labels, and that the MCU is
read ONLY from the MCU card crop, never from the whole photo.
"""

from pathlib import Path

import numpy as np
import pytest

from vision.loader import load_photo
from vision.textmatch import NameIndex

REAL = Path(__file__).parent / "real"
PHOTOS = sorted(REAL.glob("uni-ds-v8_*.jpg"))
#: photo -> (layout group, MCU on the card)
TRUTH = {"uni-ds-v8_stm32f429zi": ("mcu-card-for-stm32", "STM32F429ZI"),
         "uni-ds-v8_pic32mz2048efh144": ("mcu-card-3-for-pic32", "PIC32MZ2048EFH144")}


def test_glued_name_matches():
    idx = NameIndex([("uni", "UNI-DS v8"), ("uni", "UNI DS v8"), ("e7", "EasyPIC v7"),
                     ("e8", "EasyPIC v8")])
    best = idx.top_k(["UNI-DSV8", "MCU", "CARDFOR", "STY"], 1)[0]
    assert best[0] == "uni" and best[2] >= 0.9
    # the version must appear exactly: a glued "v8" never counts for "v7"
    s = dict((k, v) for k, _, v in idx.top_k(["EASYPICV8"], 3))
    assert s["e8"] >= 0.9 and s.get("e7", 0) < 0.72


@pytest.fixture(scope="module")
def pipeline():
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("layout anchors were calibrated with RapidOCR")
    from vision.pipeline import Pipeline
    return Pipeline(backend="rapidocr")


@pytest.mark.parametrize("path", PHOTOS, ids=[p.name for p in PHOTOS])
def test_uni_ds_v8(pipeline, path):
    r = pipeline.run(load_photo(path))
    assert r.board.decision is not None and r.board.decision.key == "uni-ds-v8", r.board.reason
    assert r.placement is not None and r.placement.method == "anchors-homography"
    assert r.placement.inliers >= 6
    assert r.mcu_source == "card"  # read from the found card, not the layout area
    names = [x.name for x in r.rois]
    assert names == ["MCU card"] + [f"mikroBUS {i}" for i in range(1, 6)]
    h, w = r.rois[0].image.shape[:2]
    assert 0.8 < w / h < 1.25
    group, part = TRUTH[path.stem.rsplit("_", 1)[0]]
    assert r.mcu_card.group is not None and r.mcu_card.group["key"] == group, r.mcu_card.reason
    # the card is never one of the group's look-alikes, and the MCU never another chip
    assert r.mcu_card.decision is None or part.lower() in r.mcu_card.decision.key
    assert r.mcu.decision is None or r.mcu.decision.key.startswith(part)
    # the found card is the card itself, not the socket outline around it
    w, h = r.card_crop.location.size_px
    assert 0.85 < w / h < 1.18
    assert r.card_reading.chip is not None
