"""The vision app server. A single asyncio process: no cloud, no extra packages.

It listens in two places:

* PC side     http://127.0.0.1:8765/  (this computer only). Serves the app
              page (``web/index.html``) with the two-step phone Scan and the
              Photos tab (drop, re-run and label photos). Routes:
                GET  /, /api/info, /ws (scan state + live preview)
                POST /api/recognize, /api/rerun, /api/label (Photos tab)
* Phone side  https://<LAN-IP>:8443/?t=<token>  (all network interfaces,
              https because the camera API needs a secure page). Serves the
              guided camera page (``web/phone.html``) and its WebSocket.
              Every request must include the session token from the QR code.

A scan always has two steps:

    1  board     the whole board in view -> board (+ layout, Clicks)
    2  MCU       the phone moved close to the MCU card -> MCU / card

Nothing is recognised until both photos are in. Then the board photo gets
the fast board-only pass (``State.board_pipeline``: just the board name)
and the close-up gets the full MCU reading, with the board as a hint. If
the close-up gives no MCU, the board photo gets the full pass as a backup.
Between the steps the phone shows what to do next, and automatic capture
waits until the user is ready (or a few seconds have passed).

Messages from the phone to the server (WebSocket):
    text   {"type":"hello", "video_w","video_h","photo_w","photo_h","image_capture",...}
    binary 0x01 preview: u32 header length, header JSON (``quality.FrameMeta``),
           u32 frame length, frame JPEG (~960 px), crop JPEG (full res, optional)
    text   {"type":"photo_begin","id","size","sha256","mime","width","height","source"}
    binary 0x02 photo chunk: u32 offset, bytes  (the ORIGINAL file, unchanged)
    text   {"type":"photo_end","id"}
    text   {"type":"ready"} / {"type":"shutter"} / {"type":"restart"} /
           {"type":"mode","mode":"board|mcu"}

Messages from the server to the phone:
    {"type":"config", mode, step, steps, message, done, summary, hold}
    {"type":"analysis", report, progress}       after every preview frame
    {"type":"capture", focus_point}             take the full-resolution photo now
    {"type":"photo_received"} / {"type":"retake", reasons} / {"type":"accepted"}
    {"type":"result", summary, next}            step finished (next is "mcu" or None)

Photos are saved exactly as received (``photos/captures/``), with a JSON
file next to each one holding the quality numbers of the last preview and
of the full photo. That way the thresholds in ``quality.py`` can be tuned
on real photos later.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import struct
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import __version__
from . import qr
from .net import (Request, WebSocket, WebSocketClosed, read_request, response)
from .bench import MAX_UPLOAD, build_response, parse_options
from .quality import CaptureGate, FrameAnalyzer, FrameMeta, analyze_photo

WEB = Path(__file__).resolve().parent / "web"
MAX_PHOTO = 80 * 1024 * 1024
STEPS = 2
HOLD_S = 4.0  # between steps: no auto-capture until "ready" or this many seconds
MAX_RETAKES = 2  # automatic retakes before we use the best photo anyway
CAPTURE_TIMEOUT_S = 20.0
PC_PREVIEW_FPS = 6.0

MESSAGES = {
    "board": "Fit the whole board inside the frame",
    "mcu": "Move closer until the MCU card fills the square",
}
TITLES = {"board": "Scan the board", "mcu": "Move closer to the MCU"}


@dataclass
class Upload:
    id: str
    size: int
    sha256: str | None
    mime: str
    source: str
    meta: dict
    buf: bytearray = field(default_factory=bytearray)
    received: int = 0


@dataclass
class Session:
    token: str
    mode: str = "board"
    step: int = 1
    attempts: int = 0
    phone: WebSocket | None = None
    phone_info: dict = field(default_factory=dict)
    analyzer: FrameAnalyzer = field(default_factory=FrameAnalyzer)
    gate: CaptureGate = field(default_factory=CaptureGate)
    analyzing: bool = False
    upload: Upload | None = None
    capture_sent_at: float | None = None
    last_report: dict | None = None
    last_meta: dict | None = None
    busy: bool = False  # a photo check or recognition is running
    done: bool = False  # step finished: no more auto-capture until the next step
    hold_until: float = 0.0  # between steps: auto-capture is paused until this time
    board_result: object | None = None  # pipeline.PhotoResult for the board photo
    mcu_result: object | None = None  # ...and for the MCU close-up
    board_shot: tuple | None = None  # (photo, name, path, check), waiting for the close-up
    mcu_shot: tuple | None = None
    best_rejected: tuple | None = None  # (score, photo, name, check, path) if retakes run out
    captures: list = field(default_factory=list)


class AppServer:
    """Set up the app server.

    ``state``: a ``bench.State`` (catalog, pipelines, Photos tab), or None to
    only save photos (no recognition, no Photos tab).

    ``recognize=False`` keeps the Photos tab but makes the scan only save photos.
    """

    def __init__(self, state=None, captures_dir: Path = Path("photos/captures"),
                 phone_port: int = 8443, ui_port: int = 8765, ips: list[str] | None = None,
                 ssl_ctx=None, rectify: bool = False, cert_fp: str | None = None,
                 recognize: bool = True):
        self.state = state
        self.recognize = recognize and state is not None
        self.captures_dir = Path(captures_dir)
        self.phone_port, self.ui_port = phone_port, ui_port
        self.ips = ips or []
        self.ssl_ctx = ssl_ctx
        self.rectify = rectify
        self.cert_fp = cert_fp
        self.session = Session(secrets.token_urlsafe(12))
        self.pcs: set[WebSocket] = set()
        self.analysis_pool = ThreadPoolExecutor(2, thread_name_prefix="cam-analysis")
        self.work_pool = ThreadPoolExecutor(1, thread_name_prefix="cam-recognize")
        self._pc_last_preview = 0.0
        self._servers = []

    # ---- URLs ----
    @property
    def scheme(self) -> str:
        return "https" if self.ssl_ctx else "http"

    def phone_urls(self) -> list[str]:
        return [f"{self.scheme}://{ip}:{self.phone_port}/?t={self.session.token}" for ip in self.ips]

    def info(self) -> dict:
        urls = self.phone_urls()
        return {"version": __version__, "phone_urls": urls,
                "qr": [qr.svg(u, scale=6) for u in urls[:3]],
                "secure": bool(self.ssl_ctx), "cert_fingerprint": self.cert_fp,
                "recognition": self.recognize, "phone_port": self.phone_port,
                "captures_dir": str(self.captures_dir.resolve()),
                "bench": self.state.info() if self.state is not None else None,
                "scan": self.scan_state()}

    def scan_state(self) -> dict:
        s = self.session
        return {"mode": s.mode, "step": s.step, "steps": STEPS, "phone": s.phone is not None,
                "phone_info": s.phone_info, "busy": s.busy, "attempts": s.attempts, "done": s.done,
                "title": "Done" if s.done else TITLES[s.mode],
                "message": MESSAGES[s.mode], "captures": s.captures[-12:],
                "manifest": self.combined_manifest(),
                "has_board": s.board_shot is not None, "has_mcu": s.mcu_shot is not None}

    # ---- Start / stop ----
    async def start(self, phone_host: str = "0.0.0.0", ui_host: str = "127.0.0.1") -> None:
        self._servers.append(await asyncio.start_server(
            self._phone_conn, phone_host, self.phone_port, ssl=self.ssl_ctx,
            limit=1 << 20))
        self._servers.append(await asyncio.start_server(self._pc_conn, ui_host, self.ui_port))
        if self.recognize:  # load the OCR engine in the meantime
            asyncio.get_running_loop().run_in_executor(self.work_pool, self.pipeline)

    async def stop(self) -> None:
        for s in self._servers:
            s.close()
        for s in self._servers:
            try:
                await asyncio.wait_for(s.wait_closed(), 2)
            except asyncio.TimeoutError:
                pass

    def pipeline(self):
        if not self.recognize:
            return None
        self.state.board_pipeline()
        return self.state.scan_pipeline()

    # ---- Sending to all PC pages ----
    async def to_pcs(self, obj=None, data: bytes | None = None) -> None:
        for ws in list(self.pcs):
            try:
                if obj is not None:
                    await ws.send_json(obj)
                if data is not None:
                    await ws.send_bytes(data)
            except WebSocketClosed:
                self.pcs.discard(ws)

    async def to_phone(self, obj) -> None:
        ws = self.session.phone
        if ws is not None:
            try:
                await ws.send_json(obj)
            except WebSocketClosed:
                pass

    async def push_state(self) -> None:
        await self.to_pcs({"type": "state", "state": self.scan_state()})

    async def send_config(self, message: str | None = None) -> None:
        s = self.session
        hold = max(0.0, s.hold_until - time.monotonic())
        await self.to_phone({"type": "config", "mode": s.mode, "step": s.step, "steps": STEPS,
                             "title": TITLES[s.mode], "message": message or MESSAGES[s.mode],
                             "done": s.done, "hold": round(hold, 1),
                             "summary": _summary(self.combined_manifest())})
        await self.push_state()

    # ==== Phone side ====
    async def _phone_conn(self, reader, writer) -> None:
        try:
            req = await read_request(reader)
            if req is None:
                return
            tok_ok = secrets.compare_digest(req.query.get("t", ""), self.session.token)
            if req.path in ("/", "/phone", "/index.html"):
                if not tok_ok:
                    writer.write(response(403, _page("This link is from an earlier session. "
                                                     "Scan the QR code on the PC again."),
                                          "text/html; charset=utf-8"))
                else:
                    writer.write(response(200, (WEB / "phone.html").read_bytes(),
                                          "text/html; charset=utf-8",
                                          {"Permissions-Policy": "camera=(self)"}))
            elif req.path == "/ws" and WebSocket.is_upgrade(req):
                if not tok_ok:
                    writer.write(response(403, b"bad token"))
                else:
                    ws = WebSocket(reader, writer)
                    await ws.accept(req)
                    await self._phone_ws(ws)
                    return
            elif req.path == "/api/upload" and req.method == "POST":
                if not tok_ok:
                    writer.write(response(403, b"bad token"))
                else:
                    n = int(req.headers.get("content-length") or 0)
                    if n <= 0 or n > MAX_PHOTO:
                        writer.write(response(413, b"bad size"))
                    else:
                        data = await reader.readexactly(n)
                        name = req.headers.get("x-filename", "upload.jpg")
                        asyncio.create_task(self._photo_arrived(
                            data, {"source": "file-input", "name": name}))
                        writer.write(response(200, b'{"ok":true}', "application/json"))
            elif req.path == "/favicon.ico":
                writer.write(response(404))
            else:
                writer.write(response(404, b"not found"))
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception:
            traceback.print_exc()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _phone_ws(self, ws: WebSocket) -> None:
        s = self.session
        old = s.phone
        s.phone = ws
        if old is not None and old is not ws:
            await old.close(4000)  # only one phone at a time: the newest one wins
        s.analyzer, s.gate = FrameAnalyzer(), CaptureGate()
        await self.send_config()
        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    await self._phone_text(json.loads(msg))
                elif msg[:1] == b"\x01":
                    if not s.analyzing:
                        s.analyzing = True
                        asyncio.create_task(self._preview(msg))
                    else:
                        await ws.send_json({"type": "analysis", "dropped": True})
                elif msg[:1] == b"\x02":
                    self._chunk(msg)
        except WebSocketClosed:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            if s.phone is ws:
                s.phone = None
                await self.push_state()

    async def _phone_text(self, m: dict) -> None:
        s = self.session
        t = m.get("type")
        if t == "hello":
            s.phone_info = {k: m.get(k) for k in ("video_w", "video_h", "photo_w", "photo_h",
                                                  "image_capture", "ua", "torch", "focus")}
            await self.push_state()
        elif t == "mode" and m.get("mode") in ("board", "mcu"):
            await self.set_mode(m["mode"])
        elif t == "ready":
            s.hold_until = 0.0
            s.gate.reset(cooldown=False)
        elif t == "shutter":
            await self.fire_capture(manual=True)
        elif t == "restart":
            await self.restart()
        elif t == "photo_begin":
            size = int(m.get("size") or 0)
            if not 0 < size <= MAX_PHOTO:
                await self.to_phone({"type": "error", "message": "photo size not accepted"})
                return
            s.upload = Upload(str(m.get("id")), size, m.get("sha256"), m.get("mime", "image/jpeg"),
                              m.get("source", "?"), {k: m.get(k) for k in
                                                     ("width", "height", "settings", "source")})
            s.upload.buf = bytearray(size)
            await self.to_pcs({"type": "transfer", "size": size, "received": 0})
        elif t == "photo_end":
            up = s.upload
            s.upload = None
            if up is None or up.id != str(m.get("id")):
                return
            data = bytes(up.buf)
            if up.received != up.size:
                await self.to_phone({"type": "retake", "reasons": ["transfer incomplete"]})
                s.gate.armed = True
                return
            if up.sha256 and hashlib.sha256(data).hexdigest() != up.sha256:
                await self.to_phone({"type": "retake", "reasons": ["transfer corrupted"]})
                s.gate.armed = True
                return
            await self.to_phone({"type": "photo_received", "bytes": len(data)})
            await self._photo_arrived(data, {**up.meta, "mime": up.mime, "sha256": up.sha256})
        elif t == "capture_failed":
            s.gate.armed = True
            s.capture_sent_at = None
            await self.to_pcs({"type": "log", "text": f"phone could not capture: {m.get('message')}"})

    def _chunk(self, msg: bytes) -> None:
        up = self.session.upload
        if up is None or len(msg) < 5:
            return
        off = struct.unpack("!I", msg[1:5])[0]
        part = msg[5:]
        if off + len(part) > up.size:
            return
        up.buf[off:off + len(part)] = part
        up.received += len(part)

    # ---- Live preview ----
    async def _preview(self, msg: bytes) -> None:
        s = self.session
        try:
            hl = struct.unpack("!I", msg[1:5])[0]
            meta_d = json.loads(msg[5:5 + hl])
            p = 5 + hl
            fl = struct.unpack("!I", msg[p:p + 4])[0]
            fjpg = msg[p + 4:p + 4 + fl]
            cjpg = msg[p + 4 + fl:]
            meta = FrameMeta.from_dict({**meta_d, "mode": s.mode})
            loop = asyncio.get_running_loop()
            rep, frame = await loop.run_in_executor(self.analysis_pool, self._analyze,
                                                    fjpg, cjpg, meta)
            if rep is None:
                return
            fire, prog = s.gate.update(rep)
            if s.busy or s.capture_sent_at is not None or s.done or \
                    time.monotonic() < s.hold_until:
                fire, prog = False, 0.0
            d = rep.as_dict()
            s.last_report, s.last_meta = d, meta_d
            await self.to_phone({"type": "analysis", "report": d, "progress": round(prog, 2),
                                 "seq": meta_d.get("seq")})
            now = time.monotonic()
            if self.pcs and now - self._pc_last_preview >= 1 / PC_PREVIEW_FPS:
                self._pc_last_preview = now
                head = json.dumps({"report": d, "progress": prog, "mode": s.mode}).encode()
                await self.to_pcs(data=b"\x01" + struct.pack("!I", len(head)) + head + fjpg)
            if fire:
                await self.fire_capture(focus=rep.focus_point)
            elif s.capture_sent_at is not None and now - s.capture_sent_at > CAPTURE_TIMEOUT_S:
                s.capture_sent_at = None
                s.gate.armed = True
        except Exception:
            traceback.print_exc()
        finally:
            s.analyzing = False

    def _analyze(self, fjpg: bytes, cjpg: bytes, meta: FrameMeta):
        frame = cv2.imdecode(np.frombuffer(fjpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None, None
        crop = cv2.imdecode(np.frombuffer(cjpg, np.uint8), cv2.IMREAD_COLOR) if cjpg else None
        return self.session.analyzer.analyze(frame, crop, meta), frame

    async def fire_capture(self, focus=None, manual: bool = False) -> None:
        s = self.session
        if s.phone is None or s.busy or s.capture_sent_at is not None or (s.done and not manual):
            return
        s.gate.armed = False
        s.capture_sent_at = time.monotonic()
        await self.to_phone({"type": "capture", "focus_point": focus, "manual": manual})
        await self.to_pcs({"type": "log", "text": "capturing full-resolution photo"
                                                  + (" (manual)" if manual else "")})

    # ---- Full photo ----
    async def _photo_arrived(self, data: bytes, meta: dict) -> None:
        s = self.session
        s.capture_sent_at = None
        if s.busy:
            await self.to_phone({"type": "error", "message": "still working on the previous photo"})
            return
        s.busy = True
        await self.push_state()
        try:
            await self._handle_photo(data, meta)
        except Exception as e:
            traceback.print_exc()
            await self.to_pcs({"type": "log", "text": f"error: {type(e).__name__}: {e}"})
            await self.to_phone({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            s.busy = False
            s.gate.reset()
            s.gate.armed = not s.done
            await self.push_state()

    async def _handle_photo(self, data: bytes, meta: dict) -> None:
        from ..loader import PhotoLoadError, load_bytes
        s = self.session
        loop = asyncio.get_running_loop()
        mode = s.mode
        ext = {"image/png": ".png", "image/heic": ".heic", "image/heif": ".heic",
               "image/webp": ".webp"}.get(meta.get("mime") or "", "")
        if not ext:
            ext = Path(meta.get("name") or "x.jpg").suffix.lower() or ".jpg"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{mode}_{len(s.captures) + 1:02d}{ext}"
        try:
            photo = await loop.run_in_executor(self.work_pool, load_bytes, data, name)
        except PhotoLoadError as e:
            await self.to_phone({"type": "retake", "reasons": [f"photo could not be decoded: {e}"]})
            return
        check = await loop.run_in_executor(self.work_pool, analyze_photo, photo.image, mode)
        path = self._save(data, name, meta, check)
        thumb = _thumb(photo.image)
        cap = {"name": name, "path": str(path), "mode": mode, "w": photo.width, "h": photo.height,
               "bytes": len(data), "check": check.as_dict(), "source": meta.get("source"),
               "thumb": thumb}
        s.captures.append({k: v for k, v in cap.items() if k != "thumb"})
        await self.to_pcs({"type": "photo", "capture": cap})
        if not check.ok:
            score = -len(check.reasons)
            if s.best_rejected is None or score > s.best_rejected[0]:
                s.best_rejected = (score, photo, name, check, path)
            if s.attempts < MAX_RETAKES:
                s.attempts += 1
                await self.to_phone({"type": "retake", "reasons": check.reasons,
                                     "attempt": s.attempts})
                return
            # out of retakes: use the best photo we have, and flag it
            _, photo, name, check, path = s.best_rejected
            await self.to_pcs({"type": "log", "text": "retakes used up — recognising the best "
                                                      "photo anyway: " + "; ".join(check.reasons)})
        s.attempts, s.best_rejected = 0, None
        await self.to_phone({"type": "accepted", "check": check.as_dict()})
        if mode == "board":
            # A new board photo starts a new scan: the close-up comes next, and
            # nothing is recognised until both photos are in.
            s.board_shot, s.mcu_shot = (photo, name, path, check), None
            s.board_result = s.mcu_result = None
            await self.to_pcs({"type": "result", "mode": None, "payload": None, "manifest": None,
                               "keep_captures": True})
            await self._advance(check, None)
            return
        s.mcu_shot = (photo, name, path, check)
        if self.recognize:
            await self._process()
        await self._advance(check, None)

    async def _process(self) -> None:
        """Recognise both photos once they're in.

        The board photo gets the fast board-only pass (just the board name),
        and the close-up gets the full MCU reading, using the board from the
        first photo as a hint.
        """
        s = self.session
        loop = asyncio.get_running_loop()
        if s.board_shot is not None and s.board_result is None:
            photo, name, path, check = s.board_shot
            await self.to_phone({"type": "status", "message": "Recognising the board…"})
            await self.to_pcs({"type": "log", "text": f"recognising the board ({name})…"})
            quad = (np.array(check.board_quad, float)
                    if self.rectify and check.board_quad and check.board_score >= 0.85 else None)
            res, payload = await loop.run_in_executor(self.work_pool, self._recognize,
                                                      photo, name, quad, None, "board")
            s.board_result = res
            self._save_result(path, res)
            await self.to_pcs({"type": "result", "mode": "board", "payload": payload,
                               "manifest": self.combined_manifest()})
        if s.mcu_shot is not None:
            photo, name, path, check = s.mcu_shot
            await self.to_phone({"type": "status", "message": "Reading the MCU…"})
            await self.to_pcs({"type": "log", "text": f"reading the MCU ({name})…"})
            hint = s.board_result.board if s.board_result is not None else None
            res, payload = await loop.run_in_executor(self.work_pool, self._recognize,
                                                      photo, name, None, hint, "mcu")
            s.mcu_result = res
            self._save_result(path, res)
            await self.to_pcs({"type": "result", "mode": "mcu", "payload": payload,
                               "manifest": self.combined_manifest()})
        # If the close-up gave neither an MCU nor a card, or the fast pass missed the
        # board, run the full pass on the board photo (4K photos often show the chip
        # text too).
        mr, br = s.mcu_result, s.board_result
        mcu_missing = mr is None or (mr.mcu.decision is None and mr.mcu_card.decision is None)
        board_missing = br is None or br.board.decision is None
        if s.board_shot is not None and (mcu_missing or board_missing):
            photo, name, path, check = s.board_shot
            await self.to_phone({"type": "status", "message": "Taking a closer look at the board photo…"})
            await self.to_pcs({"type": "log", "text": ("close-up gave no MCU" if mcu_missing else
                                                       "board not read") + " — full pass on the board photo"})
            res, payload = await loop.run_in_executor(self.work_pool, self._recognize,
                                                      photo, name, None, None, "mcu")
            if res.mcu.decision is not None or res.mcu_card.decision is not None or \
                    s.board_result is None or s.board_result.board.decision is None:
                s.board_result = res
                self._save_result(path, res)
                await self.to_pcs({"type": "result", "mode": "board", "payload": payload,
                                   "manifest": self.combined_manifest()})

    def _recognize(self, photo, name, quad, hint, kind: str = "mcu"):
        pipe = self.state.board_pipeline() if kind == "board" else self.pipeline()
        res = pipe.run(photo, quad=quad, board_hint=hint)
        opt = {"quad": quad}
        payload = build_response(self.state, photo.sha1[:16], name, photo, res, opt)
        return res, payload

    async def _advance(self, check, res) -> None:
        """Move on after a step.

        The board photo always leads to the MCU close-up (step 2), and the
        close-up ends the scan.
        """
        s = self.session
        summary = _summary(self.combined_manifest())
        if s.mode == "board":
            s.mode, s.step = "mcu", 2
            s.analyzer, s.gate = FrameAnalyzer(), CaptureGate()
            s.hold_until = time.monotonic() + HOLD_S
            await self.to_phone({"type": "result", "summary": summary, "next": "mcu"})
            await self.send_config()
            await self.to_pcs({"type": "log", "text": "step 1 done — move the phone close to the MCU"})
            return
        s.done = True
        await self.to_phone({"type": "result", "summary": summary, "next": None})
        await self.push_state()
        await self.to_pcs({"type": "log", "text": "scan finished"})

    def combined_manifest(self) -> dict | None:
        """Combine the results of both photos.

        Board (and Clicks) come from the board photo. MCU and card come from
        the close-up if it found them, otherwise from the board photo.
        """
        s = self.session
        r1, r2 = s.board_result, s.mcu_result
        if r1 is None and r2 is None:
            return None
        m1 = r1.manifest() if r1 is not None else None
        m2 = r2.manifest() if r2 is not None else None
        if m1 is None:
            return m2
        if m2 is None:
            return m1
        out = dict(m1)
        for k in ("mcu", "mcu_card", "mcu_card_layout"):
            if m2.get(k) or not m1.get(k):
                out[k] = m2.get(k)
        out["needs_user"] = ([n for n in m1["needs_user"] if not n["field"].startswith("mcu")] +
                             [n for n in m2["needs_user"] if n["field"].startswith("mcu")
                              and not out.get(n["field"])])
        if out.get("board") is None and m2.get("board"):
            out["board"] = m2["board"]
        return out

    def _save(self, data: bytes, name: str, meta: dict, check) -> Path:
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        path = self.captures_dir / name
        path.write_bytes(data)  # the original file, unchanged
        side = {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "mode": self.session.mode, "transfer": meta, "photo_check": check.as_dict(),
                "last_preview": {"meta": self.session.last_meta,
                                 "report": self.session.last_report},
                "phone": self.session.phone_info}
        path.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str), "utf-8")
        return path

    def _save_result(self, path: Path, res) -> None:
        side = path.with_suffix(".json")
        try:
            d = json.loads(side.read_text("utf-8"))
            d["recognition"] = res.manifest()
            side.write_text(json.dumps(d, indent=1, default=str), "utf-8")
        except (OSError, ValueError):
            pass

    async def set_mode(self, mode: str) -> None:
        s = self.session
        s.mode, s.step = mode, (1 if mode == "board" else 2)
        s.attempts, s.best_rejected, s.done, s.hold_until = 0, None, False, 0.0
        s.analyzer, s.gate = FrameAnalyzer(), CaptureGate()
        await self.send_config()

    async def restart(self) -> None:
        s = self.session
        s.board_result = s.mcu_result = None
        s.board_shot = s.mcu_shot = None
        s.captures = []
        await self.set_mode("board")
        await self.to_pcs({"type": "result", "mode": None, "payload": None, "manifest": None})

    # ==== PC side ====
    async def _pc_conn(self, reader, writer) -> None:
        try:
            req = await read_request(reader)
            if req is None:
                return
            host = req.headers.get("host", "")
            if not (host.startswith("127.0.0.1") or host.startswith("localhost")):
                writer.write(response(403, b"the app page is served on 127.0.0.1 only"))
            elif req.path in ("/", "/index.html") and req.method == "GET":
                writer.write(response(200, (WEB / "index.html").read_bytes(),
                                      "text/html; charset=utf-8"))
            elif req.path == "/api/info":
                writer.write(response(200, json.dumps(self.info()).encode(), "application/json"))
            elif req.path == "/ws" and WebSocket.is_upgrade(req):
                ws = WebSocket(reader, writer)
                await ws.accept(req)
                self.pcs.add(ws)
                await ws.send_json({"type": "state", "state": self.scan_state()})
                try:
                    while True:
                        m = await ws.recv()
                        if isinstance(m, str):
                            await self._pc_text(json.loads(m))
                except WebSocketClosed:
                    pass
                finally:
                    self.pcs.discard(ws)
                return
            elif req.method == "POST" and req.path in ("/api/recognize", "/api/rerun", "/api/label"):
                n = int(req.headers.get("content-length") or 0)
                if n > MAX_UPLOAD:
                    writer.write(_json({"error": f"File too large ({n / 1e6:.0f} MB)."}, 413))
                else:
                    body = await reader.readexactly(n) if n else b""
                    writer.write(await self._bench(req, body))
            elif req.path == "/favicon.ico":
                writer.write(response(404))
            else:
                writer.write(response(404, b"not found"))
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception:
            traceback.print_exc()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _bench(self, req, body: bytes) -> bytes:
        """Photos tab routes. Recognition uses the same single worker thread."""
        from ..loader import PhotoLoadError
        st = self.state
        if st is None:
            return _json({"error": "Recognition is off (started with --no-recognize)."}, 503)
        loop = asyncio.get_running_loop()
        try:
            if req.path == "/api/label":
                out, code = st.save_label(json.loads(body or b"{}"))
                return _json(out, code)
            if req.path == "/api/recognize":
                name = urllib.parse.unquote(req.headers.get("x-filename") or "photo.jpg")
                opt = parse_options(req.query, st.defaults)
                data = body
            else:
                r = json.loads(body or b"{}")
                item = st.photos.get(r.get("id", ""))
                if item is None:
                    return _json({"error": "This photo is no longer in memory. Add it again."}, 410)
                data, name = item
                opt = parse_options(r.get("options", {}), st.defaults)
            out = await loop.run_in_executor(self.work_pool, st.recognize, data, name, opt)
            return _json(out)
        except PhotoLoadError as e:
            return _json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            return _json({"error": f"{type(e).__name__}: {e}"}, 500)

    async def _pc_text(self, m: dict) -> None:
        t = m.get("type")
        if t == "mode" and m.get("mode") in ("board", "mcu"):
            await self.set_mode(m["mode"])
        elif t == "shutter":
            await self.fire_capture(manual=True)
        elif t == "restart":
            await self.restart()
        elif t == "new_token":
            old = self.session.phone
            self.session = Session(secrets.token_urlsafe(12))
            if old is not None:
                await old.close(4001)
            await self.to_pcs({"type": "info", "info": self.info()})
        elif t == "ready":
            self.session.hold_until = 0.0


# ---- Helpers ----
def _thumb(img: np.ndarray, side: int = 360) -> str:
    s = side / max(img.shape[:2])
    t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _summary(man: dict | None) -> dict:
    if not man:
        return {}

    def name(k):
        v = man.get(k)
        return v["name"] if v else None
    lay = man.get("mcu_card_layout")
    return {"board": name("board"), "mcu": name("mcu"), "mcu_card": name("mcu_card"),
            "layout": lay["title"] if lay else None,
            "clicks": [c.get("name") or c.get("pid") for c in man.get("clicks") or []
                       if c.get("outcome") == "accept"],
            "open": [n["field"] for n in man.get("needs_user", [])]}


def _json(obj, code: int = 200) -> bytes:
    return response(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")


# old name, kept so existing imports still work
CameraServer = AppServer


def _page(text: str) -> bytes:
    return (f"<!doctype html><meta name=viewport content='width=device-width'><body style='"
            f"font:17px/1.4 -apple-system,system-ui,sans-serif;background:#141414;color:#e6e6e6;"
            f"padding:32px 24px'>{text}").encode()
