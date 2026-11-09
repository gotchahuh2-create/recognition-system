"""Smart camera acquisition: use a phone as a guided camera for the PC.

    python -m vision.camera                  # opens the PC page, scan its QR code
    python -m vision.camera --no-recognize   # capture only (photos/captures/)
    python -m vision.camera --backend rapidocr --chip-mode both

The PC hosts everything; the phone and the PC must be on the same Wi-Fi.
The phone page is https with a self-signed certificate (the camera API needs
a secure page): accept the one-time browser warning on the phone.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import webbrowser
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vision.camera", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8443, help="phone-facing port (LAN, https)")
    ap.add_argument("--ui-port", type=int, default=8766, help="PC page port (127.0.0.1)")
    ap.add_argument("--host-ip", action="append", default=None,
                    help="LAN address to put in the QR code (repeatable; default: detected)")
    ap.add_argument("--http", action="store_true",
                    help="no TLS. Live camera then works only where the browser allows it on "
                         "http (e.g. Chrome flag 'unsafely-treat-insecure-origin-as-secure'); "
                         "the phone page falls back to the phone's own camera app")
    ap.add_argument("--cert", type=Path, help="own TLS certificate (PEM)")
    ap.add_argument("--key", type=Path, help="own TLS key (PEM)")
    ap.add_argument("--captures-dir", type=Path, default=Path("photos/captures"))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-recognize", action="store_true", help="capture only, no OCR")
    ap.add_argument("--rectify", action="store_true",
                    help="perspective-correct the board shot to the detected board outline "
                         "before recognition (off: the pipeline deskews / fits the layout itself)")
    g = ap.add_argument_group("recognition (same as python -m vision)")
    g.add_argument("--backend", default="auto")
    g.add_argument("--catalog", default=None)
    g.add_argument("--upscaler", default="bicubic")
    g.add_argument("--chip-mode", default="classic", choices=["classic", "faint", "both"])
    a = ap.parse_args(argv)

    from .net import cert_fingerprint, ensure_cert, lan_ips, ssl_context
    from .server import CameraServer

    ips = a.host_ip or lan_ips()
    if not ips:
        print("warning: no LAN address found — is this PC on Wi-Fi / Ethernet? "
              "Pass --host-ip 192.168.x.y", file=sys.stderr)
        ips = ["127.0.0.1"]
    ctx = fp = None
    if not a.http:
        try:
            crt, key = (a.cert, a.key) if a.cert and a.key else ensure_cert(ips)
            ctx, fp = ssl_context(crt, key), cert_fingerprint(crt)
        except RuntimeError as e:
            print(f"TLS: {e}\nfalling back to --http", file=sys.stderr)

    factory = None
    if not a.no_recognize:
        def factory():
            from ..pipeline import Pipeline
            from ..refine import RefineConfig
            return Pipeline(a.catalog, a.backend, refine_cfg=RefineConfig(upscaler=a.upscaler),
                            upscaler=a.upscaler, chip_mode=a.chip_mode)

    srv = CameraServer(factory, a.captures_dir, a.port, a.ui_port, ips, ctx, a.rectify, fp)

    async def run():
        await srv.start()
        ui = f"http://127.0.0.1:{a.ui_port}/"
        print(f"camera: PC page {ui}")
        for u in srv.phone_urls():
            print(f"        phone   {u}")
        if fp:
            print(f"        certificate fingerprint {fp}… (self-signed, accept it on the phone once)")
        print("        Ctrl+C to stop")
        if not a.no_browser:
            threading.Timer(0.5, lambda: webbrowser.open(ui)).start()
        try:
            await asyncio.Event().wait()
        finally:
            await srv.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"cannot listen: {e} (port in use? try --port / --ui-port)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
