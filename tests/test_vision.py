"""Unit and smoke tests. Run with: python -m pytest -q

The end-to-end tests draw fake boards and need an OCR engine (Tesseract is
enough). They are skipped if none is installed.
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vision import loader, preprocess as pp, textmatch as tm  # noqa: E402
from vision.catalog import Catalog  # noqa: E402
from vision.ocr import OcrToken, _unrotate_poly, available_backends, group_lines  # noqa: E402

HAS_OCR = bool(available_backends())


# ---- Loader ----
def test_parse_filename_protocol():
    m = loader.parse_filename("MIKROE-1234_phoneA_lamp_20deg_01")
    assert m["pid"] == "MIKROE-1234" and m["device"] == "phoneA"
    assert m["lighting"] == "lamp" and m["angle_deg"] == 20 and m["index"] == 1


def test_parse_filename_tolerant():
    m = loader.parse_filename("desk_photo")
    assert m["pid"] is None and m["device"] == "desk"


def test_exif_orientation_applied(tmp_path):
    from PIL import Image
    im = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = im.getexif()
    exif[0x0112] = 6  # EXIF orientation 6 = "rotate 90 degrees clockwise to display"
    p = tmp_path / "MIKROE-1_phone_lamp_01.jpg"
    im.save(p, exif=exif)
    ph = loader.load_photo(p)
    assert (ph.width, ph.height) == (20, 40)
    assert ph.pid == "MIKROE-1"


def test_unicode_path_and_split_hash(tmp_path):
    p = tmp_path / "плата_č.png"
    cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tofile(str(p))
    photos = list(loader.iter_photos(tmp_path))
    assert len(photos) == 1
    assert loader.split_hash(photos) == loader.split_hash(list(reversed(photos)))


# ---- Preprocessing ----
def _lines_image(angle):
    img = np.full((800, 1200, 3), 200, np.uint8)
    for y in range(100, 700, 60):
        cv2.line(img, (100, y), (1100, y), (20, 20, 20), 4)
    img, _ = pp.rotate_bound(img, -angle, border=cv2.BORDER_CONSTANT)
    return img


@pytest.mark.parametrize("angle", [-12, -4, 0, 6, 20])
def test_estimate_skew(angle):
    img = _lines_image(angle)
    est = pp.estimate_skew(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert abs(est - angle) < 1.0


def test_preprocess_mapping_roundtrip():
    img = _lines_image(8)
    pre = pp.preprocess(img)
    pts = np.array([[100.0, 200.0], [600.0, 400.0]])
    hom = np.hstack([pts, np.ones((2, 1))]) @ pre.M.T
    back = pre.to_original(hom[:, :2] / hom[:, 2:])
    assert np.allclose(back, pts, atol=1e-6)


def test_four_point_warp_size():
    img = np.zeros((500, 500, 3), np.uint8)
    out, H = pp.four_point_warp(img, [[400, 50], [50, 50], [50, 350], [400, 350]])
    assert out.shape[:2] == (300, 350)


def test_quality_flags_blur_and_glare():
    img = np.full((1200, 1600, 3), 60, np.uint8)
    q = pp.assess_quality(cv2.GaussianBlur(img, (0, 0), 8))
    assert any("blurry" in w for w in q.warnings)
    img[200:700, 200:900] = 255
    assert pp.assess_quality(img).glare_frac > 0.1


def test_tiles_cover_image():
    img = np.zeros((2500, 3100), np.uint8)
    cov = np.zeros_like(img)
    for x, y, t in pp.iter_tiles(img, 1200):
        cov[y:y + t.shape[0], x:x + t.shape[1]] = 1
    assert cov.all()


# ---- Geometry ----
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unrotate_poly(k):
    h, w = 30, 50
    img = np.zeros((h, w), np.uint8)
    img[5, 40] = 255
    r = pp.rotate90(img, k)
    yy, xx = np.argwhere(r)[0]
    back = _unrotate_poly(np.array([[xx, yy]], float), k, w, h)
    assert np.allclose(back, [[40, 5]])


def test_group_lines_sparse_tokens():
    def tok(t, x, y):
        return OcrToken(t, 0.9, np.array([[x, y], [x + 80, y], [x + 80, y + 30], [x, y + 30]], float))
    toks = [tok("for", 200, 100), tok("Fusion", 100, 102), tok("ARM", 300, 98),
            tok("GND", 100, 400)]
    lines = [[t.text for t in ln] for ln in group_lines(toks)]
    assert ["Fusion", "for", "ARM"] in lines and ["GND"] in lines


# ---- Text matching ----
def test_ocr_ratio_confusables():
    assert tm.ocr_ratio("STM32F407ZG", "5TM32F4O7ZG") > tm.ratio("STM32F407ZG", "5TM32F4O7ZG")


def test_name_index_versions_strict():
    idx = tm.NameIndex([("v7", "EasyPIC v7"), ("v8", "EasyPIC v8")])
    top = idx.top_k(tm.words("EASYPIC v8"), 2)
    assert top[0][0] == "v8" and top[0][2] - top[1][2] > 0.1


def test_name_index_missing_family_word_penalised():
    cat = Catalog.from_csv()
    idx = tm.NameIndex([(b.key, b.name) for b in cat.boards])
    top = dict((k, s) for k, _, s in idx.top_k(tm.words("MCU CARD 10 FOR PIC"), 5))
    assert top.get("fusion-pic-v8", 0) < 0.6


def test_mcu_suffix_and_prefix_repair():
    from vision.mcu_ocr import McuRecognizer, candidate_strings
    def tok(t, x):
        return OcrToken(t, 0.9, np.array([[x, 0], [x + 100, 0], [x + 100, 20], [x, 20]], float))
    cands = candidate_strings([tok("5TM32F407ZGT6", 0)])
    assert "STM32F407ZGT6" in cands
    rec = McuRecognizer.__new__(McuRecognizer)
    rec.catalog = Catalog.from_csv()
    top = rec.score_parts(cands)
    assert top[0].key == "STM32F407ZG" and top[0].score == 1.0
    assert top[0].score - top[1].score >= 0.05          # vs STM32F407VG


# ---- End-to-end ----
@pytest.mark.skipif(not HAS_OCR, reason="no OCR backend installed")
def test_end_to_end_synthetic(tmp_path):
    import make_synthetic
    from vision.pipeline import Pipeline
    make_synthetic.main([str(tmp_path)])
    labels = loader.read_labels_csv(tmp_path / "labels.csv")
    pipe = Pipeline()
    wrong = []
    for ph in loader.iter_photos(tmp_path):
        lab = labels[ph.name]
        res = pipe.run(ph)
        rec = res.board if lab["kind"] == "board" else res.mcu
        if lab["label"] == "none":
            ok = rec.decision is None
        else:
            ok = rec.decision is not None and lab["label"] in (rec.decision.key, rec.decision.name)
        if not ok:
            wrong.append((ph.name, rec.reason))
    assert not wrong, wrong


# ---- Enhance / refine / eval / app ----
from vision import enhance as en  # noqa: E402


def _text_img(text="STM32F429ZIT6", scale=1.2, light=True, size=(160, 700)):
    bg, fg = (30, 200) if light else (220, 20)
    img = np.full(size, bg, np.uint8)
    cv2.putText(img, text, (20, size[0] // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 3)
    return img


def test_noise_sigma_estimate():
    rng = np.random.default_rng(0)
    flat = np.full((300, 300), 120, np.float64)
    noisy = np.clip(flat + rng.normal(0, 10, flat.shape), 0, 255).astype(np.uint8)
    assert en.noise_sigma(flat.astype(np.uint8)) < 1
    assert 7 < en.noise_sigma(noisy) < 13


def test_binarizers_make_text_dark():
    img = _text_img(light=True)
    for b in (en.otsu(img), en.sauvola(255 - img)):
        assert (b == 255).mean() > 0.5  # mostly white background
    assert en.text_is_light(img) and not en.text_is_light(255 - img)


def test_upscale_falls_back_without_sr_model():
    img = _text_img(size=(40, 120))
    out = en.upscale(img, 3.0, "edsr")  # there's no model file in the test environment
    assert out.shape == (120, 360)


def test_enhance_crop_variants_shape():
    crop = cv2.resize(_text_img(), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    vs = en.enhance_crop(crop, glyph_px=8)
    assert [v for v, _ in vs] == list(en.CROP_VARIANTS)
    assert all(im.shape[0] > crop.shape[0] * 3 for _, im in vs)  # enlarged about 5x


def test_gamma_lift_brightens_dark():
    dark = np.full((50, 50, 3), 25, np.uint8)
    out, g = en.gamma_lift(dark)
    assert g > 1 and out.mean() > 60


def test_detect_text_regions_finds_marking_inside_chip_frame():
    from vision.refine import detect_text_regions
    img = np.full((900, 1200), 50, np.uint8)
    cv2.rectangle(img, (300, 200), (900, 700), 170, 25)  # light frame around the chip
    cv2.putText(img, "STM32F429ZIT6", (360, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 190, 3)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    regs = detect_text_regions(img)
    assert any(r.bbox[0] < 400 and r.bbox[2] > 600 and 260 < (r.bbox[1] + r.bbox[3]) / 2 < 340
               for r in regs), [r.bbox for r in regs]


def test_cer_wer():
    sys.path.insert(0, str(ROOT / "eval"))
    import ocr_eval
    assert ocr_eval.cer("STM32F407ZG", "STM32F407ZGT6", prefix=True) == 0
    assert abs(ocr_eval.cer("ABCD", "ABXD") - 0.25) < 1e-9
    assert ocr_eval.wer("Fusion for ARM v8", "FUSION FOR ARM v7") == 0.25


@pytest.mark.skipif(not HAS_OCR, reason="no OCR backend installed")
def test_refine_recovers_hard_scenes(tmp_path):
    """The crop re-read step must solve scenes that reading the whole image can't."""
    import make_synthetic
    from vision.pipeline import Pipeline
    make_synthetic.main([str(tmp_path), "--only-hard"])
    labels = loader.read_labels_csv(tmp_path / "labels.csv")
    must = {"hard_lowlight.jpg", "hard_far_small_chip.jpg", "hard_defocus_pic.jpg"}
    pipe = Pipeline()
    for ph in loader.iter_photos(tmp_path):
        if ph.name not in must:
            continue
        res = pipe.run(ph)
        assert res.mcu.decision is not None, (ph.name, res.mcu.reason)
        assert res.mcu.decision.key == labels[ph.name]["label"]


@pytest.mark.skipif(not HAS_OCR, reason="no OCR backend installed")
def test_app_recognize_and_label(tmp_path):
    """The app server: page, Photos tab recognise / re-run / label, info."""
    import asyncio
    import json
    import threading
    import urllib.error
    import urllib.request
    from vision.app.bench import State
    from vision.app.server import AppServer
    img = cv2.imencode(".jpg", cv2.cvtColor(_text_img(), cv2.COLOR_GRAY2BGR))[1].tobytes()
    srv = AppServer(State(None, tmp_path / "ph"), tmp_path / "cap", 0, 0, ["127.0.0.1"],
                    recognize=False)
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    async def start():
        await srv.start("127.0.0.1", "127.0.0.1")
        ready.set()
    threading.Thread(target=lambda: (loop.run_until_complete(start()), loop.run_forever()),
                     daemon=True).start()
    assert ready.wait(10)
    base = f"http://127.0.0.1:{srv._servers[1].sockets[0].getsockname()[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Scan board" in page and "Drop photos" in page
        info = json.loads(urllib.request.urlopen(base + "/api/info").read())
        assert info["scan"]["steps"] == 2 and info["bench"]["catalog"]["board"]
        req = urllib.request.Request(base + "/api/recognize?refine=off", data=img,
                                     headers={"X-Filename": "chip%20one.jpg"})
        d = json.loads(urllib.request.urlopen(req).read())
        assert d["name"] == "chip one.jpg" and "manifest" in d["result"]
        assert d["original"]["src"].startswith("data:image/jpeg")
        req = urllib.request.Request(base + "/api/rerun", data=json.dumps(
            {"id": d["id"], "options": {"refine": "off", "chip_mode": "classic"}}).encode())
        assert json.loads(urllib.request.urlopen(req).read())["id"] == d["id"]
        req = urllib.request.Request(base + "/api/label", data=json.dumps(
            {"id": d["id"], "kind": "mcu", "label": "STM32F429ZI"}).encode())
        out = json.loads(urllib.request.urlopen(req).read())
        assert out["rows"] == 1 and (tmp_path / "ph" / "chip_one.jpg").is_file()
        assert "STM32F429ZI" in (tmp_path / "ph" / "labels.csv").read_text()
        req = urllib.request.Request(base + "/api/rerun", data=b'{"id": "gone"}')
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 410
    finally:
        asyncio.run_coroutine_threadsafe(srv.stop(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)