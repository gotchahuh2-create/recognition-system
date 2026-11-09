"""Tests for the phone camera (vision.app): quality checks, auto-capture,
full-photo check, QR code, WebSocket transport and the two-step scan.
"""

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import random
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.app import qr
from vision.app.quality import (CaptureGate, FrameAnalyzer, FrameMeta, analyze_photo,
                                detect_board, quad_geometry, reblur)

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "real"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scene():
    """A fake UNI-DS-like board with a drawn MCU card (3000 px), on a desk."""
    synth = _load("make_synthetic_cards", ROOT / "scripts" / "make_synthetic_cards.py")
    data = json.loads((ROOT / "data" / "mcu_cards.json").read_text())
    c = next(x for x in data["cards"] if x["key"] == "mcu-card-29-for-stm32-stm32f215re")
    g = data["groups"][c["group"]]
    card = synth.render_card(g, dict(c, _marking_lines=synth.marking_lines(c)), rng=random.Random(1))
    img = synth.degrade(synth.render_scene(card, rot90=0, card_frac=0.4, seed=3), seed=3, tilt=2,
                        persp=0.02, blur=0.5, out_long=3000, noise=3, jpeg=92)
    return cv2.copyMakeBorder(img, 250, 250, 330, 330, cv2.BORDER_CONSTANT, value=(185, 190, 190))


def _stream(img, n=6, mode="board", gyro=None, photo_scale=1.0, frames=None):
    """Feed ``n`` preview frames (plus the full-res crops the analyser asks for)."""
    H, W = img.shape[:2]
    a, gate = FrameAnalyzer(), CaptureGate()
    meta = dict(mode=mode, video_w=W, video_h=H, photo_w=int(W * photo_scale),
                photo_h=int(H * photo_scale), gyro=gyro)
    crop, fired, reps = None, 0, []
    for i in range(n):
        src = frames(i) if frames else img
        fr = cv2.resize(src, (960, int(960 * H / W)), interpolation=cv2.INTER_AREA)
        r = a.analyze(fr, crop, FrameMeta.from_dict(meta), now=100 + 0.2 * i)
        f, _ = gate.update(r, now=100 + 0.2 * i)
        fired += f
        reps.append(r)
        if r.crop_request:
            x, y, w, h = r.crop_request
            crop = src[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)]
            s = min(1.0, 1024 / max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            meta.update(crop_rect=r.crop_request, crop_scale=s)
    return fired, reps


def test_reblur_orders_blur_levels():
    rng = np.random.default_rng(0)
    img = np.zeros((200, 200), np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, 180, 2)
        cv2.putText(img, "STM32", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
    vals = [max(reblur(cv2.GaussianBlur(img, (0, 0), s) if s else img)) for s in (0, 1, 2, 3)]
    assert vals == sorted(vals) and vals[0] < 0.35 < vals[2]
    k = np.zeros((9, 9), np.float32)
    k[4, :] = 1 / 9
    h, v = reblur(cv2.filter2D(img, -1, k))
    assert h - v > 0.1  # horizontal motion: only horizontal blur


def test_board_found_on_real_photos_and_cut_off_detected():
    for f in sorted(REAL.glob("*.jpg")):
        b = detect_board(cv2.imread(str(f)))
        assert b is not None and b.score > 0.8, f.name
        assert quad_geometry(b.quad)["area"] > 0.5 * 960 * 1280
    # photo 2 is the one not used for calibration, with the board cut off at two edges
    _, reps = _stream(cv2.imread(str(REAL / "uni-ds-v8_stm32f429zi_2.jpg")), n=1)
    assert reps[0].metric("framing").level == "bad"
    assert "back" in reps[0].instruction


def test_real_photo_chip_size_matches_measurement():
    # the chip is only 71-77 px wide on these 960x1280 photos
    for f in sorted(REAL.glob("*.jpg")):
        chk = analyze_photo(cv2.imread(str(f)))
        assert chk.chip_px is not None and 65 <= chk.chip_px <= 85, (f.name, chk.chip_px)
        assert chk.needs_closeup  # the board photo is fine; the MCU needs a close-up


def test_good_stream_captures_and_bad_streams_do_not(scene):
    fired, reps = _stream(scene)
    assert fired == 1
    assert reps[-1].chip_px_photo and reps[-1].chip_px_photo > 300
    assert reps[0].metric("focus").level == "bad"  # never before the focus has been measured
    blur = cv2.GaussianBlur(scene, (0, 0), 2.2)
    assert _stream(blur)[0] == 0
    assert _stream(scene, gyro=30)[0] == 0
    assert _stream((scene * 0.25).astype(np.uint8))[0] == 0
    assert _stream(scene, frames=lambda i: np.roll(scene, 60 * i, axis=1))[0] == 0
    H, W = scene.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
                                    np.float32([[W * .12, 0], [W * .88, 0], [W, H], [0, H]]))
    persp = cv2.warpPerspective(scene, M, (W, H), borderValue=(185, 190, 190))
    fired, reps = _stream(persp)
    assert fired == 0 and "lower its top edge" in reps[-1].instruction


def test_small_chip_is_a_hint_in_board_mode_and_blocking_in_mcu_mode(scene):
    fired, reps = _stream(scene, photo_scale=0.5)  # photo is half the video size
    assert fired == 1 and reps[-1].chip_px_photo < 250
    assert any("close-up" in h for h in reps[-1].hints)
    fired, reps = _stream(scene, mode="mcu", photo_scale=0.5)
    assert fired == 0 and "closer" in reps[-1].instruction.lower()


def test_photo_check_retakes_blur_and_glare(scene):
    assert analyze_photo(scene).ok
    assert not analyze_photo(cv2.GaussianBlur(scene, (0, 0), 2)).ok
    from vision.card_locate import CardLocator
    c = max(CardLocator().frame_candidates(scene), key=lambda l: l.score).quad.mean(0)
    g = scene.copy()
    cv2.circle(g, tuple(int(v) for v in c), 30, (255, 255, 255), -1)
    chk = analyze_photo(cv2.GaussianBlur(g, (0, 0), 1.0))
    assert not chk.ok and "glare" in chk.reasons[0]


def test_chip_found_when_a_sharp_marking_notches_the_body(scene):
    """find_chip must still find the chip when sharp text cuts into it.

    On a real close-up, the readable marking cut the dark/smooth mask down
    to a fill of 0.7499, so the chip wasn't found and nothing was read.
    """
    from vision.card_locate import CardLocator, crop_card
    from vision.card_reader import find_chip
    loc = max(CardLocator().frame_candidates(scene), key=lambda l: l.score)
    cc = crop_card(scene, loc, 1800, 4.0)
    chips = find_chip(cc.image)
    assert chips
    side = np.linalg.norm(chips[0].quad[1] - chips[0].quad[0])
    assert 0.25 < side / cc.image.shape[0] < 0.4


@pytest.mark.parametrize("text", ["https://192.168.1.23:8443/?t=Zk3x9QpL2mN8vR4s",
                                  "https://10.0.0.5:8443/?t=" + "abc123XYZ_-" * 20])
def test_qr_code_decodes(text):
    m = np.array(qr.encode(text), np.uint8)
    img = np.pad((1 - m) * 255, 4, constant_values=255).astype(np.uint8)
    img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    val, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert val == text
    assert qr.svg(text).startswith("<svg")


async def _ws_client(port, path):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    key = base64.b64encode(os.urandom(16)).decode()
    w.write((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
             f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = await r.readuntil(b"\r\n\r\n")
    assert b"101" in head.split(b"\r\n")[0]
    return r, w


def _frame(op, payload):
    mask = os.urandom(4)
    n = len(payload)
    hd = bytes([0x80 | op]) + (bytes([0x80 | n]) if n < 126 else
                               bytes([0x80 | 126]) + struct.pack("!H", n) if n < 65536 else
                               bytes([0x80 | 127]) + struct.pack("!Q", n))
    return hd + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


async def _read(r):
    b1, b2 = await r.readexactly(2)
    n = b2 & 0x7F
    if n == 126:
        n = struct.unpack("!H", await r.readexactly(2))[0]
    elif n == 127:
        n = struct.unpack("!Q", await r.readexactly(8))[0]
    data = await r.readexactly(n)
    return json.loads(data) if b1 & 0x0F == 1 else data


def test_phone_session_over_websocket(scene, tmp_path):
    """Full phone session over WebSocket.

    Checks the token, the preview analysis round trip, sending the original
    photo in chunks with a checksum, the photo check, and that the photo is
    saved exactly as sent.
    """
    from vision.app.server import AppServer

    async def main():
        srv = AppServer(None, tmp_path, 0, 0, ["127.0.0.1"])
        await srv.start("127.0.0.1", "127.0.0.1")
        pport = srv._servers[0].sockets[0].getsockname()[1]
        with pytest.raises(AssertionError):  # wrong token: connection is refused
            await _ws_client(pport, "/ws?t=nope")
        r, w = await _ws_client(pport, f"/ws?t={srv.session.token}")
        cfg = await _read(r)
        assert cfg["type"] == "config" and cfg["mode"] == "board"
        H, W = scene.shape[:2]
        fr = cv2.resize(scene, (960, int(960 * H / W)))
        jpg = cv2.imencode(".jpg", fr)[1].tobytes()
        head = json.dumps({"video_w": W, "video_h": H, "photo_w": W, "photo_h": H}).encode()
        w.write(_frame(2, b"\x01" + struct.pack("!I", len(head)) + head
                       + struct.pack("!I", len(jpg)) + jpg))
        msg = await asyncio.wait_for(_read(r), 20)
        assert msg["type"] == "analysis" and msg["report"]["board_quad"]
        data = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        w.write(_frame(1, json.dumps({"type": "photo_begin", "id": "a", "size": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest(),
                                      "mime": "image/jpeg"}).encode()))
        for off in range(0, len(data), 200_000):
            w.write(_frame(2, b"\x02" + struct.pack("!I", off) + data[off:off + 200_000]))
        w.write(_frame(1, json.dumps({"type": "photo_end", "id": "a"}).encode()))
        seen = []
        while "result" not in seen:
            m = await asyncio.wait_for(_read(r), 30)
            seen.append(m["type"])
        assert seen[:2] == ["photo_received", "accepted"]
        saved = sorted(tmp_path.glob("*_board_01.jpg"))
        assert saved and saved[0].read_bytes() == data
        side = json.loads(saved[0].with_suffix(".json").read_text())
        assert side["photo_check"]["ok"] and side["photo_check"]["chip_px"] > 300
        # Two steps: the board photo always leads to the MCU close-up, which waits
        # until the phone says it's ready.
        cfg = await asyncio.wait_for(_read(r), 10)
        assert cfg["type"] == "config" and cfg["mode"] == "mcu" and cfg["step"] == 2
        assert cfg["hold"] > 0 and not cfg["done"]
        assert srv.session.hold_until > 0
        w.write(_frame(1, json.dumps({"type": "ready"}).encode()))
        for _ in range(50):
            if srv.session.hold_until == 0:
                break
            await asyncio.sleep(0.02)
        assert srv.session.hold_until == 0
        # step 2 (sent like the camera app's upload): finishes the scan
        w.write(_frame(1, json.dumps({"type": "photo_begin", "id": "b", "size": len(data),
                                      "mime": "image/jpeg"}).encode()))
        w.write(_frame(2, b"\x02" + struct.pack("!I", 0) + data))
        w.write(_frame(1, json.dumps({"type": "photo_end", "id": "b"}).encode()))
        last = None
        for _ in range(20):
            last = await asyncio.wait_for(_read(r), 30)
            if last["type"] in ("result", "retake"):
                break
        if last["type"] == "result":
            assert last["next"] is None and srv.session.done
        assert sorted(tmp_path.glob("*_mcu_02.jpg"))
        w.close()
        await srv.stop()

    asyncio.run(main())


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
def test_self_signed_certificate(tmp_path):
    from vision.app.net import cert_fingerprint, ensure_cert, ssl_context
    crt, key = ensure_cert(["192.168.1.50"], tmp_path)
    ssl_context(crt, key)
    assert len(cert_fingerprint(crt).split(":")) == 8
    assert ensure_cert(["192.168.1.50"], tmp_path)[0].stat().st_mtime == crt.stat().st_mtime


def test_scan_processes_only_after_both_photos(scene, tmp_path, monkeypatch):
    """Nothing is recognised until both photos are in.

    Board photo: saved, nothing recognised, step 2 is next. Close-up: the
    fast board pass runs first, then the MCU pass with the board as a hint.
    """
    from types import SimpleNamespace as NS
    import vision.app.server as server
    from vision.app.server import AppServer

    calls = []

    class Pipe:
        def __init__(self, kind):
            self.kind = kind

        def run(self, photo, quad=None, board_hint=None):
            calls.append((self.kind, board_hint))
            dec = NS(name="UNI-DS v8")
            rec = lambda d: NS(decision=d, candidates=[])          # noqa: E731
            return NS(board=rec(dec), mcu=rec(dec if self.kind == "mcu" else None),
                      mcu_card=rec(None),
                      manifest=lambda: {"board": {"name": "UNI-DS v8"}, "mcu": None,
                                        "mcu_card": None, "mcu_card_layout": None,
                                        "clicks": [], "needs_user": []})

    class St:
        def board_pipeline(self):
            return Pipe("board")

        def scan_pipeline(self):
            return Pipe("mcu")

    ok = NS(ok=True, reasons=[], as_dict=lambda: {"ok": True, "reasons": []},
            board_quad=None, board_score=0.0)
    monkeypatch.setattr(server, "analyze_photo", lambda img, mode="board": ok)
    monkeypatch.setattr(server, "build_response", lambda *a, **k: {"result": {}})
    data = cv2.imencode(".jpg", scene)[1].tobytes()

    async def main():
        srv = AppServer(St(), tmp_path, 0, 0, ["127.0.0.1"])
        await srv._handle_photo(data, {"mime": "image/jpeg"})
        assert calls == [] and srv.session.mode == "mcu" and srv.session.board_shot
        await srv._handle_photo(data, {"mime": "image/jpeg"})
        assert [c[0] for c in calls] == ["board", "mcu"]
        assert calls[1][1] is not None  # the board hint was passed to the MCU pass
        assert srv.session.done

    asyncio.run(main())
