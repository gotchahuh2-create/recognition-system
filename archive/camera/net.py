"""Local network helpers: LAN address, self-signed TLS certificate, WebSocket.

Why TLS at all on a local network: mobile browsers only open the camera
(``getUserMedia``) on a *secure* page — https:// or localhost. The phone
reaches the PC by its LAN address, so the page must be https. A self-signed
certificate is generated once with the ``openssl`` command (present on Linux
and macOS, Git for Windows) and stored in the user cache; the phone shows a
one-time warning ("Advanced -> Proceed"). Nothing leaves the local network.

The WebSocket implementation (RFC 6455, server side) is the minimum the
camera needs: text + binary messages, fragmentation, ping/pong, close.
Kept in the standard library so the camera adds no dependency.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import shutil
import socket
import ssl
import struct
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------- #
# LAN address
# --------------------------------------------------------------------------- #
def lan_ips() -> list[str]:
    """Private IPv4 addresses of this machine, the default-route one first."""
    out: list[str] = []
    for probe in ("192.0.2.1", "10.255.255.255", "172.16.255.255"):
        try:     # UDP connect sends nothing; it only picks the outgoing interface
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                ip = s.getsockname()[0]
            if ip not in out:
                out.append(ip)
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in out:
                out.append(ip)
    except OSError:
        pass
    good = []
    for ip in out:
        a = ipaddress.ip_address(ip)
        if a.is_private and not a.is_loopback and not a.is_link_local:
            good.append(ip)
    return good


# --------------------------------------------------------------------------- #
# TLS
# --------------------------------------------------------------------------- #
def cert_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "mikroe-vision" / "camera-tls"


def ensure_cert(ips: list[str], folder: Path | None = None) -> tuple[Path, Path]:
    """(cert.pem, key.pem) valid for ``ips`` + localhost; regenerated when the
    address list changed. Raises RuntimeError if openssl is missing."""
    folder = folder or cert_dir()
    folder.mkdir(parents=True, exist_ok=True)
    crt, key, info = folder / "cert.pem", folder / "key.pem", folder / "cert.json"
    want = sorted(set(ips) | {"127.0.0.1"})
    try:
        have = json.loads(info.read_text())["ips"]
    except (OSError, ValueError, KeyError):
        have = None
    if crt.is_file() and key.is_file() and have is not None and set(want) <= set(have):
        return crt, key
    exe = shutil.which("openssl")
    if exe is None:
        raise RuntimeError("openssl not found: install it, or pass --cert/--key, or run with "
                           "--http (camera then only via the phone's own camera app)")
    san = ",".join([f"IP:{ip}" for ip in want] + ["DNS:localhost"])
    cmd = [exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "825",
           "-keyout", str(key), "-out", str(crt), "-subj", "/CN=MikroE vision camera (local)",
           "-addext", f"subjectAltName={san}", "-addext", "extendedKeyUsage=serverAuth",
           "-addext", "basicConstraints=critical,CA:FALSE"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"openssl failed: {r.stderr.strip()[:400]}")
    try:
        key.chmod(0o600)
    except OSError:
        pass
    info.write_text(json.dumps({"ips": want}))
    return crt, key


def ssl_context(crt: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(crt), str(key))
    return ctx


def cert_fingerprint(crt: Path) -> str:
    der = ssl.PEM_cert_to_DER_cert(crt.read_text())
    h = hashlib.sha256(der).hexdigest().upper()
    return ":".join(h[i:i + 2] for i in range(0, 16, 2))   # first 8 bytes, for display


# --------------------------------------------------------------------------- #
# HTTP request (minimal)
# --------------------------------------------------------------------------- #
class Request:
    def __init__(self, method: str, target: str, headers: dict[str, str]):
        from urllib.parse import parse_qs, urlsplit
        self.method, self.target, self.headers = method, target, headers
        u = urlsplit(target)
        self.path = u.path
        self.query = {k: v[0] for k, v in parse_qs(u.query).items()}


async def read_request(reader: asyncio.StreamReader, limit: int = 16384) -> Request | None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        return None
    if len(head) > limit:
        return None
    lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, _ = lines[0].split(" ", 2)
    except ValueError:
        return None
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return Request(method.upper(), target, headers)


def response(code: int, body: bytes = b"", ctype: str = "text/plain; charset=utf-8",
             extra: dict | None = None) -> bytes:
    reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
              405: "Method Not Allowed", 413: "Payload Too Large",
              500: "Internal Server Error"}.get(code, "OK")
    hdr = {"Content-Type": ctype, "Content-Length": str(len(body)), "Cache-Control": "no-store",
           "Connection": "close", "X-Content-Type-Options": "nosniff", **(extra or {})}
    return (f"HTTP/1.1 {code} {reason}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdr.items())
            + "\r\n").encode("latin-1") + body


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketClosed(Exception):
    pass


class WebSocket:
    """Server side of one WebSocket connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 max_message: int = 96 * 1024 * 1024):
        self.reader, self.writer, self.max_message = reader, writer, max_message
        self.closed = False
        self._wlock = asyncio.Lock()

    @staticmethod
    def is_upgrade(req: Request) -> bool:
        return (req.headers.get("upgrade", "").lower() == "websocket"
                and "sec-websocket-key" in req.headers)

    async def accept(self, req: Request) -> None:
        acc = base64.b64encode(hashlib.sha1((req.headers["sec-websocket-key"] + _GUID)
                                            .encode()).digest()).decode()
        self.writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Accept: {acc}\r\n\r\n").encode())
        await self.writer.drain()

    async def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self.closed:
            raise WebSocketClosed()
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, n)
        elif n < 65536:
            head = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        async with self._wlock:
            try:
                self.writer.write(head + payload)
                await self.writer.drain()
            except (ConnectionError, RuntimeError) as e:
                self.closed = True
                raise WebSocketClosed() from e

    async def send_json(self, obj) -> None:
        await self._send_frame(0x1, json.dumps(obj, ensure_ascii=False).encode())

    async def send_bytes(self, data: bytes) -> None:
        await self._send_frame(0x2, data)

    async def recv(self):
        """Next message: str (text) or bytes (binary). Raises WebSocketClosed."""
        buf, kind = bytearray(), None
        while True:
            try:
                b1, b2 = await self.reader.readexactly(2)
                n = b2 & 0x7F
                if n == 126:
                    n = struct.unpack("!H", await self.reader.readexactly(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", await self.reader.readexactly(8))[0]
                mask = await self.reader.readexactly(4) if b2 & 0x80 else None
                if len(buf) + n > self.max_message:
                    await self.close(1009)
                    raise WebSocketClosed()
                data = await self.reader.readexactly(n) if n else b""
            except (asyncio.IncompleteReadError, ConnectionError) as e:
                self.closed = True
                raise WebSocketClosed() from e
            if mask:
                data = _unmask(data, mask)
            op, fin = b1 & 0x0F, b1 & 0x80
            if op == 0x8:
                await self.close()
                raise WebSocketClosed()
            if op == 0x9:
                await self._send_frame(0xA, data)
                continue
            if op == 0xA:
                continue
            if op in (0x1, 0x2):
                kind, buf = op, bytearray(data)
            elif op == 0x0:
                buf += data
            if fin and kind is not None:
                return buf.decode("utf-8") if kind == 0x1 else bytes(buf)

    async def close(self, code: int = 1000) -> None:
        if self.closed:
            return
        try:
            await self._send_frame(0x8, struct.pack("!H", code))
        except WebSocketClosed:
            pass
        self.closed = True
        try:
            self.writer.close()
        except Exception:
            pass


def _unmask(data: bytes, mask: bytes) -> bytes:
    if not data:
        return data
    n = len(data)
    m = (mask * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(m, "little")).to_bytes(n, "little")
