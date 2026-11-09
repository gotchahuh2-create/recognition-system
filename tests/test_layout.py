"""Tests for reading only the MCU card area, using the board's stored layout.

Fake scene: a "UNI-DS v8" board with printed labels all over it, an MCU
card with an STM32F429ZIT6 chip, and a decoy chip elsewhere on the board
marked STM32F407ZGT6. Reading the whole photo sees both part numbers. With
the layout, the MCU step only looks at the card, so only the card's chip
counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import layout as L  # noqa: E402
from vision.catalog import Catalog  # noqa: E402
from vision.loader import photo_from_array  # noqa: E402
from vision.ocr import OcrToken, available_backends  # noqa: E402
from vision.pipeline import Pipeline  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
needs_ocr = pytest.mark.skipif(not available_backends(), reason="no OCR backend")

# where the board is in the drawn image, and the layout boxes (relative to the board)
BX0, BY0, BX1, BY1 = 150, 120, 2650, 1920
MCU_BOX = [0.36, 0.30, 0.64, 0.62]
SOCKETS = [("mikroBUS 1", [0.04, 0.08, 0.20, 0.30]), ("mikroBUS 2", [0.80, 0.08, 0.96, 0.30])]
LABELS = [("UNI-DS v8", 0.30, 0.05, 64), ("USB UART", 0.06, 0.78, 34), ("POWER SUPPLY", 0.70, 0.80, 34),
          ("CODEGRIP", 0.06, 0.88, 34), ("ETHERNET", 0.40, 0.88, 34), ("AUDIO OUT", 0.72, 0.90, 34),
          ("LED PORTA", 0.06, 0.38, 34), ("BUTTONS", 0.80, 0.40, 34), ("CAN BUS", 0.06, 0.55, 34),
          ("RS485", 0.82, 0.58, 34), ("mikroBUS 1", 0.05, 0.32, 30), ("mikroBUS 2", 0.81, 0.32, 30)]


def _f(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def render() -> np.ndarray:
    W, H = 2800, 2040
    im = Image.new("RGB", (W, H), (205, 205, 200))
    d = ImageDraw.Draw(im)
    d.rectangle((BX0, BY0, BX1, BY1), fill=(25, 30, 60))
    bw, bh = BX1 - BX0, BY1 - BY0
    xy = lambda u, v: (BX0 + u * bw, BY0 + v * bh)
    for t, u, v, s in LABELS:
        d.text(xy(u, v), t, font=_f(FONT, s), fill=(235, 235, 230))
    for _, (u0, v0, u1, v1) in SOCKETS:          # empty socket headers
        d.rectangle((*xy(u0 + .01, v0 + .02), *xy(u0 + .03, v1 - .02)), fill=(20, 20, 20))
        d.rectangle((*xy(u1 - .03, v0 + .02), *xy(u1 - .01, v1 - .02)), fill=(20, 20, 20))
    u0, v0, u1, v1 = MCU_BOX                     # MCU card + chip
    d.rectangle((*xy(u0, v0), *xy(u1, v1)), fill=(20, 60, 25))
    d.text(xy(u0 + .02, v0 + .02), "MCU CARD", font=_f(FONT, 30), fill=(230, 230, 225))
    cx, cy = xy(0.42, 0.38)
    d.rectangle((cx, cy, cx + 480, cy + 300), fill=(32, 32, 34))
    for i, line in enumerate(["STM32F429ZIT6", "GH21Q 93"]):
        d.text((cx + 30, cy + 60 + i * 80), line, font=_f(MONO, 46), fill=(175, 175, 170))
    dx, dy = xy(0.40, 0.66)                      # decoy chip outside the card
    d.rectangle((dx, dy, dx + 520, dy + 110), fill=(32, 32, 34))
    d.text((dx + 30, dy + 25), "STM32F407ZGT6", font=_f(MONO, 50), fill=(175, 175, 170))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def corners():
    return np.array([[BX0, BY0], [BX1, BY0], [BX1, BY1], [BX0, BY1]], float)


def transform(img, k90=1, persp=0.04, scale=1.0, seed=3):
    """Rotate by k*90 degrees clockwise, add some perspective, and scale.

    Returns the image and the 3x3 transform from the original to the new one.
    """
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-persp, persp, (4, 2)).astype(np.float32) * min(w, h)
    P = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, P, (w, h), borderValue=(205, 205, 200))
    R = np.eye(3)
    for _ in range(k90 % 4):
        hh, ww = out.shape[:2]
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        R = np.array([[0, -1, hh - 1], [1, 0, 0], [0, 0, 1]], float) @ R
    S = np.diag([scale, scale, 1.0])
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out, S @ R @ P


def catalog() -> Catalog:
    c = Catalog.load()
    return c


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    img = render()
    lay = L.calibrate(img, "uni-ds-v8", corners(), MCU_BOX, SOCKETS, "auto", source="syn")
    lays = L.Layouts()
    lays["uni-ds-v8"] = lay
    p = lays.save(tmp_path_factory.mktemp("lay") / "layouts.json")
    return img, lay, p


# ---- Without OCR ----
def test_box_quad_and_json_roundtrip(tmp_path):
    q = L.box_quad([0.1, 0.2, 0.3, 0.6], pad=0.5)
    assert np.allclose(q[0], [0.0, 0.0]) and np.allclose(q[2], [0.4, 0.8])
    lay = L.BoardLayout("b", 1.4, [0.1, 0.1, 0.2, 0.2], [L.Socket("mikroBUS 1", [0, 0, .1, .1])],
                        [L.Anchor("USB UART", 0.5, 0.5, 0.01)])
    lays = L.Layouts()
    lays["b"] = lay
    back = L.Layouts.load(lays.save(tmp_path / "x.json"))["b"]
    assert back.aspect == pytest.approx(1.4) and back.sockets[0].slot == "mikroBUS 1"
    assert back.anchors[0].key == "USBUART"


def test_locate_from_tokens_any_orientation():
    """Pure geometry check.

    Anchors seen at their true positions after a 90 degree turn, some
    perspective and noise: the MCU area must land where it should.
    """
    lay = L.BoardLayout("b", 1.5, MCU_BOX, [], [
        L.Anchor(t, u + 0.03, v + 0.01) for t, u, v, _ in LABELS if t != "mikroBUS 1"])
    H_true = cv2.getPerspectiveTransform(
        L.box_quad([0, 0, 1, 1]).astype(np.float32),
        np.float32([[1500, 200], [1550, 1700], [300, 1650], [250, 260]]))  # rotated 90 degrees
    rng = np.random.default_rng(0)
    toks = []
    for a in lay.anchors:
        c = L.apply_h(H_true, [[a.u, a.v]])[0] + rng.normal(0, 2, 2)
        poly = np.array([c + [-40, -10], c + [40, -10], c + [40, 10], c + [-40, 10]])
        toks.append(OcrToken(a.text, 0.95, poly))
    toks.append(OcrToken("GND", 0.9, np.array([[0, 0], [9, 0], [9, 9], [0, 9]], float)))

    class FakePre:
        image = np.zeros((1800, 1800, 3), np.uint8)
        M = np.eye(3)
    pl = L.locate(lay, toks, FakePre())
    assert pl is not None and pl.method.startswith("anchors")
    want = L.apply_h(H_true, L.box_quad(MCU_BOX))
    assert np.abs(pl.to_pre(L.box_quad(MCU_BOX)) - want).max() < 15


def test_locate_refuses_with_too_few_anchors():
    lay = L.BoardLayout("b", 1.5, MCU_BOX, [], [L.Anchor("USB UART", 0.1, 0.8)])

    class FakePre:
        image = np.zeros((1000, 1000, 3), np.uint8)
        M = np.eye(3)
    tok = OcrToken("USB UART", 0.9, np.array([[0, 0], [90, 0], [90, 20], [0, 20]], float))
    assert L.locate(lay, [tok], FakePre()) is None


# ---- With OCR ----
@needs_ocr
def test_calibration_keeps_board_text_only(calibrated):
    _, lay, _ = calibrated
    keys = {a.key for a in lay.anchors}
    assert {"USBUART", "ETHERNET", "CODEGRIP"} <= keys
    # text on the MCU card or chips inside the card area must not become board anchors
    assert not any("STM32F429" in k or k == "MCUCARD" for k in keys)
    assert len(lay.anchors) >= 8


@needs_ocr
@pytest.mark.parametrize("k90", [0, 1, 2, 3])
def test_two_stage_reads_only_the_card(calibrated, k90):
    img, lay, path = calibrated
    photo_img, A = transform(img, k90=k90, persp=0.035, scale=0.8, seed=k90)
    pipe = Pipeline(catalog(), "auto", layouts=path, card_stage=False)  # the older layout-area path
    res = pipe.run(photo_from_array(photo_img, f"syn_rot{k90 * 90}.png"))
    assert res.board.decision and res.board.decision.key == "uni-ds-v8"
    assert res.placement is not None, "layout not placed"
    assert res.mcu_source == "roi"
    assert res.mcu.decision is not None and res.mcu.decision.key == "STM32F429ZI", res.mcu.reason
    # the MCU card area covers the real card (mapped through the known transform)
    card_true = L.apply_h(A, L.box_quad(MCU_BOX) * [BX1 - BX0, BY1 - BY0] + [BX0, BY0])
    roi = next(r for r in res.rois if r.name == "MCU card")
    inside = cv2.pointPolygonTest(roi.quad_orig.astype(np.float32).reshape(-1, 1, 2),
                                  tuple(map(float, card_true.mean(0))), True)
    assert inside > 0
    assert {r.name for r in res.rois} == {"MCU card", "mikroBUS 1", "mikroBUS 2"}
    d = res.as_dict()
    assert d["layout"]["mcu_source"] == "roi" and len(d["layout"]["rois"]) == 3


@needs_ocr
def test_without_layout_the_decoy_competes(calibrated):
    """For comparison: reading the whole photo also sees the decoy chip."""
    img, _, _ = calibrated
    pipe = Pipeline(catalog(), "auto", layouts=False, card_stage=False)
    res = pipe.run(photo_from_array(img, "syn.png"))
    reads = {c.key for c in res.mcu.candidates}
    assert res.placement is None and res.mcu_source == "photo"
    assert "STM32F407ZG" in reads


@needs_ocr
def test_card_stage_ignores_the_decoy_without_a_layout(calibrated):
    """Without a layout, the card step still ignores the decoy.

    The card is found from its title (read during the board step) and only
    the card is read, so the decoy chip elsewhere on the board isn't.
    """
    img, _, _ = calibrated
    res = Pipeline(catalog(), "auto", layouts=False).run(photo_from_array(img, "syn.png"))
    assert res.mcu_source == "card" and res.card_crop is not None
    assert res.mcu.decision is not None and res.mcu.decision.key == "STM32F429ZI"
    assert "STM32F407ZG" not in {c.key for c in res.mcu.candidates}
