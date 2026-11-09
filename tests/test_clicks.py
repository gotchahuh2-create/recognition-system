"""Tests for the DINOv2 Click recognition package (vision.clicks).

The ONNX Runtime code runs on a tiny hand-built model with the same
interface as the real one (tests/onnx_toy.py), so no torch and no download
are needed. Accuracy with the toy model means nothing; these tests check
that the pieces fit together and that the decision rules work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_toy import write_toy_model  # noqa: E402
from vision.catalog import Catalog, Click, Socket  # noqa: E402
from vision.clicks import (ClickConfig, ClickRecognizer, FlatIndex, Gallery,  # noqa: E402
                           GalleryMismatch, OnnxEmbedder, assign_slots)
from vision.clicks.augment import Augmenter  # noqa: E402
from vision.clicks.build import build_gallery, scan_images  # noqa: E402
from vision.clicks.embedder import Embedder, l2n  # noqa: E402
from vision.clicks.preprocess import (PrepConfig, crop_box, pad_square, prepare,  # noqa: E402
                                      size_class, trim_margin)
from vision.clicks.rerank import (ClickOcr, match_name, normalize_click_text,  # noqa: E402
                                  token_set_ratio, _indel_ratio)
from vision.ocr import OcrBackend, effective_rotations, rotations_covered  # noqa: E402

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return write_toy_model(tmp_path_factory.mktemp("m") / "toy.onnx")


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    """4 fake Clicks (2 of them a numbered series), with 2 photos each."""
    d = tmp_path_factory.mktemp("syn")
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_synthetic_clicks as msc
    msc.main([str(d), "--n", "6", "--photos", "2"])
    (d / "data").mkdir()
    (d / "data" / "clicks.csv").write_text((d / "clicks.csv").read_text())
    return d


# ---- Preprocessing ----
def test_pad_square_keeps_aspect_and_prepare_shape():
    img = np.zeros((100, 300, 3), np.uint8)
    img[:, :] = (0, 0, 255)
    sq = pad_square(img)
    assert sq.shape[:2] == (300, 300)
    # the board fills the middle third, untouched (never stretched)
    assert (sq[100:200, :, 2] == 255).all() and (sq[:100, :, 2] != 255).all()
    x = prepare(img, PrepConfig(size=224))
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    assert PrepConfig(224, clahe=True).version != PrepConfig(224).version


def test_trim_margin_and_crop_box():
    img = np.full((200, 300, 3), 255, np.uint8)
    img[50:150, 80:220] = 30
    t = trim_margin(img)
    assert abs(t.shape[0] - 100) <= 6 and abs(t.shape[1] - 140) <= 6
    c = crop_box(img, (80, 50, 140, 100), expand=0.1)
    assert c.shape[:2] == (120, 168)
    c = crop_box(img, (0, 0, 50, 50), expand=0.1)  # cut off at the image border
    assert c.shape[:2] == (55, 55)


def test_size_class():
    assert size_class(286, 254) == "S"
    assert size_class(254, 429) == "M"  # works at any rotation
    assert size_class(572, 254) == "L"


def test_augmenter_outputs_valid_images():
    aug = Augmenter(seed=3)
    board = np.full((120, 200, 3), (40, 110, 30), np.uint8)
    for out in aug.many(board, 6):
        assert out.dtype == np.uint8 and out.ndim == 3 and min(out.shape[:2]) > 20


# ---- Embedder and gallery ----
def test_onnx_embedder(toy):
    e = OnnxEmbedder(toy)
    assert e.dim == 147 and e.prep.size == 224
    assert e.model_version.startswith("toy-grid7")
    v = e.embed([np.zeros((50, 80, 3), np.uint8), np.full((90, 40, 3), 200, np.uint8)])
    assert v.shape == (2, 147)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert e.embed([]).shape == (0, 147)


def test_flat_index_matches_bruteforce():
    rng = np.random.default_rng(0)
    g = l2n(rng.normal(size=(300, 16)))
    q = l2n(rng.normal(size=(2, 16)))
    sims, ids = FlatIndex(g, use_faiss=False).search(q, 10)
    ref = np.argsort(-(q @ g.T), axis=1)[:, :10]
    assert (ids == ref).all()
    assert np.all(np.diff(sims, axis=1) <= 1e-6)


def _gallery(vecs: dict[str, list], mv="m:cls:224:fp32", pv="pp1-padsq-224") -> Gallery:
    d = len(next(iter(vecs.values()))[0])
    g = Gallery.empty(d, mv, pv)
    for pid, vs in vecs.items():
        g.add(pid, l2n(np.array(vs, np.float32)), "catalog", name=pid)
    return g


def test_gallery_search_groups_by_pid_max_and_roundtrip(tmp_path):
    g = _gallery({"A": [[1, 0, 0], [0.6, 0.8, 0]], "B": [[0.9, 0.1, 0]], "C": [[0, 0, 1]]})
    hits = g.search(l2n(np.array([[1, 0.05, 0]], np.float32)), k=10, top=5)
    assert [h[0] for h in hits][:2] == ["A", "B"]
    assert len({h[0] for h in hits}) == len(hits)       # one row per PID
    g.save(tmp_path / "g")
    info = json.loads((tmp_path / "g" / "gallery.json").read_text())
    assert info["count"] == 4 and info["pids"] == 3
    g2 = Gallery.load(tmp_path / "g", "m:cls:224:fp32", "pp1-padsq-224")
    assert np.allclose(g2.vectors, g.vectors) and g2.meta[0]["pid"] == "A"
    with pytest.raises(GalleryMismatch):
        Gallery.load(tmp_path / "g", "other:cls:224:fp32", "pp1-padsq-224")
    with pytest.raises(GalleryMismatch):
        Gallery.load(tmp_path / "g", "m:cls:224:fp32", "pp1-padsq-336")
    assert set(g2.without_pids(["A"]).pids) == {"B", "C"}


def test_build_gallery_and_recognise_catalog_image(toy, syn, tmp_path):
    emb = OnnxEmbedder(toy)
    assert len(scan_images(syn / "catalog_images")) == 6
    g = build_gallery(emb, syn / "catalog_images", tmp_path / "gal", augment=3,
                      progress=None)
    assert len(g) == 6 * 4 and {m["source"] for m in g.meta} == {"catalog", "augmented"}
    cat = Catalog.load(syn / "data")
    rec = ClickRecognizer(emb, str(tmp_path / "gal"), None, cat,
                          ClickConfig(ocr=False, tta=False))
    img = trim_margin(cv2.imread(str(syn / "catalog_images" / "MIKROE-9004" / "front.jpg")))
    r = rec.recognise(img)          # a tight crop of a gallery image
    assert r.recognition.candidates[0].pid == "MIKROE-9004"
    assert r.recognition.candidates[0].score > 0.99  # it's in the gallery
    assert r.outcome in ("accept", "ask")
    assert r.manifest()["outcome"] == r.outcome
    # adding a Click needs no retraining: just add it and it's found
    from vision.clicks.build import add_images
    n = add_images(g, emb, "MIKROE-9999", [syn / "catalog_images" / "MIKROE-9000" / "front.jpg"],
                   augment=1, name="Copy Click")
    assert n == 2 and "MIKROE-9999" in g.pids


# ---- OCR re-rank helpers ----
def test_normalize_click_text():
    assert normalize_click_text("Thermo 2 click  MIKROE-1234") == "THERMO 2"
    assert normalize_click_text("THERMO2CLICK www.mikroe.com") == "THERMO 2"
    assert normalize_click_text("Heart Rate 4 Click") == "HEART RATE 4"


def test_token_set_ratio_fallback_matches_definition():
    assert token_set_ratio("THERMO 2 GND VCC", "THERMO 2") == 100
    assert token_set_ratio("ACCEL", "ACCEL") == 100
    assert token_set_ratio("", "ACCEL") == 0
    assert _indel_ratio("ABCD", "ABCE") == pytest.approx(75.0)


def test_numbered_series_rule():
    ocr = normalize_click_text("THERMO 3 click")
    assert match_name(ocr, "Thermo 3 Click").ratio >= 90
    assert match_name(ocr, "Thermo 2 Click").ratio == 0  # the number must match exactly
    assert match_name(ocr, "Thermo Click").ratio == 0  # OCR read a number, but the name has none
    nonum = normalize_click_text("THERMO click")
    assert match_name(nonum, "Thermo 2 Click").ratio == 0  # no number read: OCR doesn't vote
    assert match_name(nonum, "Thermo Click").ratio >= 90


# ---- Decision rule ----
class _Fixed(Embedder):
    """Always returns the same vector, so the test controls the search scores."""

    def __init__(self, v, mv="m:cls:224:fp32"):
        self.v, self.dim, self.model_version = l2n(np.array(v, np.float32)), len(v), mv

    def embed(self, crops):
        return np.repeat(self.v[None], len(crops), 0)


def _rec(q, vecs, clicks=(), **cfg):
    cat = Catalog([], [], [], list(clicks))
    return ClickRecognizer(_Fixed(q), _gallery(vecs), None, cat,
                           ClickConfig(ocr=False, tta=False, **cfg))


IMG = np.zeros((60, 100, 3), np.uint8)


def test_accept_ask_unknown():
    # cos(A)=1.0, cos(B)=0.6 -> clear accept
    r = _rec([1, 0], {"A": [[1, 0]], "B": [[0.6, 0.8]]}).recognise(IMG)
    assert r.outcome == "accept" and r.recognition.decision.pid == "A"
    # two almost identical references -> ask, with the top 3
    r = _rec([1, 0], {"A": [[1, 0]], "B": [[0.999, 0.04]]}).recognise(IMG)
    assert r.outcome == "ask" and len(r.recognition.candidates) == 2
    # nothing close -> unknown
    r = _rec([1, 0], {"A": [[0.3, 0.95]], "B": [[0.2, 0.98]]}).recognise(IMG)
    assert r.outcome == "unknown" and r.recognition.decision is None


def test_confusion_group_forces_ask():
    clicks = [Click("A", "Relay Click", group="relay"), Click("B", "Relay Pro Click", group="relay")]
    r = _rec([1, 0], {"A": [[1, 0]], "B": [[0.8, 0.6]]}, clicks).recognise(IMG)
    assert r.outcome == "ask" and "near-identical" in r.recognition.reason


def test_ocr_promotes_and_confirms():
    clicks = [Click("A", "Thermo 2 Click"), Click("B", "Thermo 3 Click")]
    rec = _rec([1, 0], {"A": [[1, 0]], "B": [[0.995, 0.1]]}, clicks)
    rec.cfg.ocr = True
    ocr = ClickOcr("THERMO 3 click", normalize_click_text("THERMO 3 click"), 0.9, 0)
    r = rec.recognise(IMG, ocr_res=ocr)
    assert r.recognition.candidates[0].pid == "B"  # moved up to first place
    assert r.outcome == "accept" and "OCR confirmed" in r.recognition.reason
    # without OCR the same scores are a numbered pair -> ask
    rec.cfg.ocr = False
    assert rec.recognise(IMG).outcome == "ask"


def test_size_prior_penalises_wrong_size():
    clicks = [Click("A", "Alpha Click", size="L"), Click("B", "Beta Click", size="S")]
    rec = _rec([1, 0], {"A": [[1, 0]], "B": [[0.99, 0.14]]}, clicks)
    r = rec.recognise(np.zeros((300, 300, 3), np.uint8), box=(10, 10, 110, 100))  # S-sized shape
    assert r.recognition.candidates[0].pid == "B"


def test_redecide_and_thresholds_file(tmp_path):
    rec = _rec([1, 0], {"A": [[1, 0]], "B": [[0.6, 0.8]]})
    r = rec.recognise(IMG)
    assert rec.redecide(r, ClickConfig(t_accept=0.99, t_unknown=0.5, margin=0.05))[0] == "accept"
    assert rec.redecide(r, ClickConfig(t_accept=1.5, t_unknown=0.5))[0] == "ask"
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps({"t_accept": 0.8, "t_unknown": 0.3, "margin": 0.02, "junk": 1}))
    c = ClickConfig.from_file(p, tta=False)
    assert (c.t_accept, c.t_unknown, c.margin, c.tta) == (0.8, 0.3, 0.02, False)


def test_assign_slots():
    sockets = [Socket("b", "mikroBUS 1", 0.25, 0.5), Socket("b", "mikroBUS 2", 0.75, 0.5)]
    assert assign_slots([(0.7, 0.4), (0.2, 0.6)], sockets) == ["mikroBUS 2", "mikroBUS 1"]
    assert assign_slots([(0.7, 0.4)], []) == [None]


# ---- How OCR engines handle rotation ----
class _Rot(OcrBackend):
    handles_rotation = True


class _NoRot(OcrBackend):
    handles_rotation = False


def test_effective_rotations():
    assert effective_rotations(_Rot(), (0, 180)) == [0]
    assert effective_rotations(_Rot(), (0, 90, 180, 270)) == [0, 90]
    assert effective_rotations(_NoRot(), (0, 90, 180, 270)) == [0, 90, 180, 270]
    assert rotations_covered(_Rot(), [0]) == {0, 180}
    assert rotations_covered(_NoRot(), [0]) == {0}


def test_rapidocr_backend_limits_detector_resize():
    pytest.importorskip("rapidocr_onnxruntime")
    from vision.ocr import RapidOcrBackend
    be = RapidOcrBackend()
    # a small crop must not be blown up to "short side 736 px"
    crop = np.full((60, 260, 3), 255, np.uint8)
    cv2.putText(crop, "STM32F407", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    import time
    t0 = time.perf_counter()
    toks = be.read(crop)
    assert time.perf_counter() - t0 < 2.0  # this used to take ~1.9 s per crop on a fast PC
    assert any("STM32" in t.text for t in toks)


# ---- Pipeline and app ----
def test_pipeline_click_step(toy, syn, tmp_path):
    from vision.loader import load_photo
    from vision.pipeline import Pipeline
    emb = OnnxEmbedder(toy)
    build_gallery(emb, syn / "catalog_images", tmp_path / "gal", augment=2, progress=None)
    cat = Catalog.load(syn / "data")
    rec = ClickRecognizer(emb, str(tmp_path / "gal"), None, cat, ClickConfig(ocr=False))

    class _NoOcr(OcrBackend):
        name = "none"

        def read(self, img):
            return []

    pipe = Pipeline(cat, _NoOcr(), mcu=False, clicks=rec, click_mode="photo")
    photo = load_photo(next((syn / "photos").glob("MIKROE-9001_*.jpg")))
    res = pipe.run(photo)
    m = res.manifest()
    assert len(m["clicks"]) == 1 and m["clicks"][0]["outcome"] in ("accept", "ask", "unknown")
    assert "clicks" in res.as_dict() and res.timings_ms["clicks"] >= 0
    res2 = pipe.run(photo, click_boxes=[[10, 10, 100, 80], [50, 40, 60, 60]])
    assert len(res2.clicks) == 2 and res2.clicks[0].box == [10.0, 10.0, 100.0, 80.0]
    pipe.click_mode = "boxes"
    assert pipe.run(photo).clicks == []
