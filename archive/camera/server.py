"""Smart camera acquisition server: PC <-> phone over the local Wi-Fi.

Two listeners in one asyncio process, no cloud, no extra packages:

* phone side  https://<LAN-IP>:8443/?t=<token>   (all interfaces, TLS — the
              camera API needs a secure page). Serves the guided camera page
              and its WebSocket; every request must carry the session token
              that is in the QR code.
* PC side     http://127.0.0.1:8766/              (loopback only). Shows the
              QR code, the live preview with the quality overlay, the
              captured photos and the recognition result.

Phone -> server WebSocket messages
    text   {"type":"hello", "video_w","video_h","photo_w","photo_h","image_capture",…}
    binary 0x01 preview: u32 header length, header JSON (``quality.FrameMeta``),
           u32 frame length, frame JPEG (~960 px), crop JPEG (native px, optional)
    text   {"type":"photo_begin","id","size","sha256","mime","width","height","source"}
    binary 0x02 photo chunk: u32 offset, bytes  (the ORIGINAL file, unchanged)
    text   {"type":"photo_end","id"}
    text   {"type":"mode","mode":"board|mcu"} / {"type":"shutter"} / {"type":"restart"}

Server -> phone
    {"type":"config", mode, step, message}      what to photograph now
    {"type":"analysis", report, progress}       after every preview frame
    {"type":"capture", focus_point}             take the full-resolution photo now
    {"type":"photo_received"} / {"type":"retake", reasons} / {"type":"result", summary}

Captures are saved byte-for-byte (``photos/captures/``) with a JSON sidecar
holding the quality metrics of the last preview and of the full photo, so the
thresholds in ``quality.py`` can be tuned on real captures later.
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import __version__
from . import qr
from .net import (Request, WebSocket, WebSocketClosed, read_request, response)
from .quality import CaptureGate, FrameAnalyzer, FrameMeta, analyze_photo

WEB = Path(__file__).resolve().parent / "web"
MAX_PHOTO = 80 * 1024 * 1024
MAX_RETAKES = 2                     # automatic retakes before the best photo is used anyway
CAPTURE_TIMEOUT_S = 20.0
PC_PREVIEW_FPS = 6.0

MESSAGES = {
    "board": "Step 1 — photograph the whole board inside the frame",
    "mcu": "Step 2 — close-up: fill the square with the MCU card",
}


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
    busy: bool = False                      # photo check / recognition running
    done: bool = False                      # finished: no more auto-capture until a new step
    board_result: object | None = None      # pipeline.PhotoResult of the board shot
    mcu_result: object | None = None        # ... of the MCU close-up
    best_rejected: tuple | None = None      # (score, photo, name, check, path): used when retakes run out
    captures: list = field(default_factory=list)


class CameraServer:
    def __init__(self, pipeline_factory=None, captures_dir: Path = Path("photos/captures"),
                 phone_port: int = 8443, ui_port: int = 8766, ips: list[str] | None = None,
                 ssl_ctx=None, rectify: bool = False, cert_fp: str | None = None):
        self.pipeline_factory = pipeline_factory
        self._pipe = None
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

    # ------------------------------------------------------------------ urls
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
                "recognition": self.pipeline_factory is not None,
                "captures_dir": str(self.captures_dir.resolve()),
                "state": self.state()}

    def state(self) -> dict:
        s = self.session
        return {"mode": s.mode, "step": s.step, "phone": s.phone is not None,
                "phone_info": s.phone_info, "busy": s.busy, "attempts": s.attempts, "done": s.done,
                "message": MESSAGES[s.mode], "captures": s.captures[-12:],
                "manifest": self.combined_manifest()}

    # ------------------------------------------------------------- lifecycle
    async def start(self, phone_host: str = "0.0.0.0", ui_host: str = "127.0.0.1") -> None:
        self._servers.append(await asyncio.start_server(
            self._phone_conn, phone_host, self.phone_port, ssl=self.ssl_ctx,
            limit=1 << 20))
        self._servers.append(await asyncio.start_server(self._pc_conn, ui_host, self.ui_port))
        if self.pipeline_factory is not None:      # warm up OCR + catalogue meanwhile
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
        if self._pipe is None and self.pipeline_factory is not None:
            self._pipe = self.pipeline_factory()
        return self._pipe

    # ------------------------------------------------------------- broadcast
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
        await self.to_pcs({"type": "state", "state": self.state()})

    async def send_config(self, message: str | None = None) -> None:
        s = self.session
        await self.to_phone({"type": "config", "mode": s.mode, "step": s.step,
                             "message": message or MESSAGES[s.mode]})
        await self.push_state()

    # ================================================================ phone
    async def _phone_conn(self, reader, writer) -> None:
        try:
            req = await read_request(reader)
            if req is None:
                return
            tok_ok = secrets.compare_digest(req.query.get("t", ""), self.session.token)
            if req.path in ("/", "/phone", "/index.html"):
                if not tok_ok:
                    writer.write(response(403, _page("Scan the current QR code on the PC — this "
                                                     "link is from an old session."),
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
            await old.close(4000)               # one phone at a time: the newest wins
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

    # ---------------------------------------------------------------- preview
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
            if s.busy or s.capture_sent_at is not None:
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

    # ------------------------------------------------------------------ photo
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
            # out of retakes: use the best photo we have, flag it
            _, photo, name, check, path = s.best_rejected
            await self.to_pcs({"type": "log", "text": "retakes used up — recognising the best "
                                                      "photo anyway: " + "; ".join(check.reasons)})
        s.attempts, s.best_rejected = 0, None
        await self.to_phone({"type": "accepted", "check": check.as_dict()})
        if self.pipeline_factory is None:
            await self._advance(check, None)
            return
        await self.to_phone({"type": "status", "message": "Recognising…"})
        await self.to_pcs({"type": "log", "text": f"recognising {name}…"})
        quad = (np.array(check.board_quad, float)
                if self.rectify and mode == "board" and check.board_quad and check.board_score >= 0.85
                else None)
        hint = s.board_result.board if (mode == "mcu" and s.board_result is not None) else None
        res, payload = await loop.run_in_executor(self.work_pool, self._recognize,
                                                  photo, name, quad, hint)
        if mode == "board":
            s.board_result = res
        else:
            s.mcu_result = res
        self._save_result(path, res)
        await self.to_pcs({"type": "result", "mode": mode, "payload": payload,
                           "manifest": self.combined_manifest()})
        await self._advance(check, res)

    def _recognize(self, photo, name, quad, hint):
        from ..app import build_response
        pipe = self.pipeline()
        res = pipe.run(photo, quad=quad, board_hint=hint)
        opt = {"quad": quad}
        payload = build_response(None, photo.sha1[:16], name, photo, res, opt)
        return res, payload

    async def _advance(self, check, res) -> None:
        """After a recognised photo: done, or go on to the MCU close-up."""
        s = self.session
        summary = _summary(self.combined_manifest())
        if s.mode == "board":
            # close-up only while something about the MCU is still open: the
            # MCU, or the card when its layout group was read but not the card
            mcu_open = res is None or res.mcu.decision is None or \
                (res.mcu_card.decision is None and bool(res.mcu_card.group))
            if mcu_open and (check.needs_closeup or res is not None):
                s.mode, s.step = "mcu", 2
                why = (f"the MCU is only {check.chip_px:.0f} px in this photo"
                       if check.chip_px else "the MCU marking needs a closer look")
                await self.to_phone({"type": "result", "summary": summary, "next": "mcu"})
                await self.send_config(f"Board done. Now a close-up of the MCU — {why}")
                return
        s.done = True
        await self.to_phone({"type": "result", "summary": summary, "next": None})
        await self.to_pcs({"type": "log", "text": "done — Capture now retakes, New setup starts over"})

    def combined_manifest(self) -> dict | None:
        """Board (+ Clicks) from the board shot, MCU / card from the close-up
        when it decided them, else from the board shot."""
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
        path.write_bytes(data)                      # the original, byte for byte
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
        s.attempts, s.best_rejected, s.done = 0, None, False
        s.analyzer, s.gate = FrameAnalyzer(), CaptureGate()
        await self.send_config()

    async def restart(self) -> None:
        s = self.session
        s.board_result = s.mcu_result = None
        s.captures = []
        await self.set_mode("board")
        await self.to_pcs({"type": "result", "mode": None, "payload": None, "manifest": None})

    # =================================================================== PC
    async def _pc_conn(self, reader, writer) -> None:
        try:
            req = await read_request(reader)
            if req is None:
                return
            host = req.headers.get("host", "")
            if not (host.startswith("127.0.0.1") or host.startswith("localhost")):
                writer.write(response(403, b"PC page is served on 127.0.0.1 only"))
            elif req.path in ("/", "/index.html"):
                writer.write(response(200, (WEB / "pc.html").read_bytes(),
                                      "text/html; charset=utf-8"))
            elif req.path == "/api/info":
                writer.write(response(200, json.dumps(self.info()).encode(), "application/json"))
            elif req.path == "/ws" and WebSocket.is_upgrade(req):
                ws = WebSocket(reader, writer)
                await ws.accept(req)
                self.pcs.add(ws)
                await ws.send_json({"type": "state", "state": self.state()})
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


# --------------------------------------------------------------------------- #
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
            "open": [n["field"] for n in man.get("needs_user", [])]}


def _page(text: str) -> bytes:
    return (f"<!doctype html><meta name=viewport content='width=device-width'><body style='"
            f"font:16px system-ui;background:#1b1b1b;color:#ddd;padding:24px'>{text}").encode()
