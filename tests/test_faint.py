"""Tests for vision.faint, the experimental faint chip marking reader."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_faint_chips as M  # noqa: E402

from vision.faint import FaintChipReader, extract_chip  # noqa: E402
from vision.faint import cluster as C  # noqa: E402
from vision.faint import features as F  # noqa: E402
from vision.faint.debug import StageSaver  # noqa: E402
from vision.faint.fusion import _vote, fuse  # noqa: E402
from vision.faint.preserve import gentle_clean, stroke_score, text_axis  # noqa: E402
from vision.faint.region import order_quad  # noqa: E402
from vision.ocr import OcrToken  # noqa: E402

HAS_TESS = shutil.which("tesseract") is not None
LINES = ["STM32F405", "RGT6", "GH22B VG"]


def chip(delta=14, angle=0.0, glare=True, seed=0):
    return M.render(LINES, delta, random.Random(seed), glare=glare, angle=angle)


def test_order_quad_diamond_has_four_distinct_corners():
    q = order_quad([[100, 0], [200, 100], [100, 200], [0, 100]])
    assert len({tuple(p) for p in q.tolist()}) == 4


@pytest.mark.parametrize("angle", [0.0, 45.0])
def test_region_finds_dark_central_body(angle):
    img = chip(angle=angle)
    cc = extract_chip(img)
    assert cc.method in ("central", "blob")
    side = np.linalg.norm(cc.quad[1] - cc.quad[0])
    # the drawn chip is 300 px wide in a 484 px photo
    assert 240 < side < 360
    assert abs(cc.quad.mean(0) - np.array(img.shape[1::-1]) / 2).max() < 40


def test_stroke_score_prefers_text_over_noise():
    g = 30.0
    body = np.ones((400, 400), bool)
    text = np.zeros((400, 400), np.uint8)
    cv2.putText(text, "STM32F405", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 4)
    rng = np.random.default_rng(0)
    noise = (cv2.GaussianBlur(rng.normal(0, 1, (400, 400)).astype(np.float32), (0, 0), 3) > 0.25)
    assert stroke_score(text > 0, body, g) > 2 * stroke_score(noise, body, g)
    assert text_axis(text > 0, body, g)[0] == "h"
    assert text_axis(cv2.rotate(text, cv2.ROTATE_90_CLOCKWISE) > 0, body, g)[0] == "v"


def test_gentle_clean_keeps_strokes_drops_blobs():
    g = 30.0
    body = np.ones((300, 300), bool)
    m = np.zeros((300, 300), bool)
    m[50:80, 40:46] = True                 # a stroke
    m[150:260, 150:260] = True             # glare patch
    m[20, 20] = True                       # speck
    out = gentle_clean(m, body, g)
    assert out[60, 42] and not out[200, 200] and not out[20, 20]


def test_character_vote_fixes_single_errors():
    assert _vote("STM32F405", [("STM32F405", 1), ("STH32F405", 1), ("STM32F4O5", 1),
                               ("STM32F405", 1)]) == "STM32F405"
    # a letter inserted by the majority is kept
    assert _vote("STM32F45", [("STM32F45", 0.5), ("STM32F405", 1), ("STM32F405", 1)]) == "STM32F405"


def test_fusion_clusters_lines_across_variants():
    def tok(text, y, var):
        return OcrToken(text, 0.8, np.array([[10, y], [200, y], [200, y + 30], [10, y + 30]], float),
                        0, var)
    per = {"a": [tok("STM32F405", 10, "a"), tok("RGT6", 60, "a")],
           "b": [tok("STH32F405", 12, "b"), tok("RGT6", 61, "b")],
           "c": [tok("STM32F4O5", 9, "c")]}
    fused, rel = fuse(per)
    texts = [f.text for f in fused]
    assert "STM32F405" in texts and "RGT6" in texts
    assert set(rel) == {"a", "b", "c"}


def test_cluster_orients_score_so_marking_is_positive():
    img = chip(delta=30, glare=True)
    cc = extract_chip(img)
    pf = F.compute(cc.image, cc.body_mask, cc.glyph_px)
    cr = C.cluster(pf, cc.body_mask)
    # the marking is LIGHTER than the chip, so high scores must be light pixels
    hi = (cr.score > 2.5) & cc.body_mask
    assert hi.sum() > 500
    assert np.median(pf.smooth["lab"][:, :, 0][hi]) > 1.0
    assert cr.polarity == 1


def test_debug_saver_writes_every_stage(tmp_path):
    img = chip(delta=30)
    sv = StageSaver(tmp_path, "t")
    cc = extract_chip(img, saver=sv)
    reader = FaintChipReader.__new__(FaintChipReader)  # preprocessing only, no OCR engine needed
    from vision.faint.reader import FaintConfig
    reader.cfg = FaintConfig()
    vs, scores, tm, cr, masks = reader.preprocess(cc, sv)
    names = {p.name for p in tmp_path.iterdir()}
    for stage in ("1a_located", "1b_rectified", "2b_glare", "3_lab_resid", "4a_rgb_kmeans",
                  "4a_hsv_kmeans", "4a_lab_kmeans", "4c_vote", "4d_fisher", "5_", "7_variant"):
        assert any(stage in n for n in names), stage
    assert len(vs) >= 4 and all(v.image.dtype == np.uint8 for v in vs)


@pytest.mark.skipif(not HAS_TESS, reason="tesseract not installed")
def test_reads_faint_marking_end_to_end():
    from vision.catalog import Catalog
    from vision.mcu_ocr import McuRecognizer, candidate_strings
    img = chip(delta=30, glare=True, seed=1)
    fr = FaintChipReader("tesseract").read(img)
    rec = McuRecognizer(Catalog.load(), "tesseract", refine="off")
    from vision.mcu_ocr import ACCEPT, MARGIN
    parts = rec.score_parts(candidate_strings(fr.tokens))
    assert "STM32F405RG" in [p.key for p in parts[:3]]
    # never a confident wrong answer: anything else on top must fail the margin rule
    if parts[0].key != "STM32F405RG":
        assert not (parts[0].score >= ACCEPT and parts[0].score - parts[1].score >= MARGIN)
    assert fr.rotations in ([0, 180], [180, 0])


def test_read_card_rejects_unknown_chip_mode():
    from vision.card_reader import read_card
    with pytest.raises(ValueError):
        read_card(None, "tesseract", chip_mode="nope")


def test_card_order_quad_keeps_four_corners_at_45_degrees():
    from vision.card_locate import order_quad as card_order_quad
    q = card_order_quad(cv2.boxPoints(((100, 100), (80, 80), 45)))
    assert len({tuple(np.round(p, 3)) for p in q.tolist()}) == 4
    # ordinary rectangles stay unchanged
    r = card_order_quad([[10, 10], [110, 12], [108, 80], [8, 78]])
    assert r.tolist() == [[10, 10], [110, 12], [108, 80], [8, 78]]
