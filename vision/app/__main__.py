"""The vision app: scan a board with your phone, or test photos.

    python -m vision.app                      # opens http://127.0.0.1:8765/
    python -m vision.app --no-browser --gallery gallery/
    python -m vision.app --backend rapidocr --chip-mode both
    python -m vision.app --no-recognize       # only save photos (photos/captures/)

Scan: the page shows a QR code. Scan it with your phone (on the same Wi-Fi).
The phone walks you through two steps, first the whole board and then a
close-up of the MCU, and takes each photo automatically once it's sharp,
straight and free of glare.

Photos: drop or paste photos to recognise them, re-run them with other
settings, or save them with a label to --photos-dir.

The phone page uses https with a self-signed certificate (browsers only
allow camera access on secure pages), so accept the one-time warning on
the phone.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import webbrowser
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vision.app", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765, help="app page port (127.0.0.1)")
    ap.add_argument("--phone-port", type=int, default=8443, help="phone page port (LAN, https)")
    ap.add_argument("--host-ip", action="append", default=None,
                    help="LAN address to put in the QR code (repeatable; default: detected)")
    ap.add_argument("--http", action="store_true",
                    help="no TLS for the phone page. The live camera then works only where the "
                         "browser allows it on http; the phone falls back to its camera app")
    ap.add_argument("--cert", type=Path, help="own TLS certificate (PEM)")
    ap.add_argument("--key", type=Path, help="own TLS key (PEM)")
    ap.add_argument("--captures-dir", type=Path, default=Path("photos/captures"))
    ap.add_argument("--photos-dir", type=Path, default=Path("photos"),
                    help="where 'Save to dataset' copies photos and labels.csv")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-recognize", action="store_true", help="scan captures only, no OCR")
    ap.add_argument("--rectify", action="store_true",
                    help="perspective-correct the board shot to the detected board outline "
                         "before recognition (off: the pipeline deskews / fits the layout itself)")
    g = ap.add_argument_group("recognition (same as python -m vision)")
    g.add_argument("--backend", default="auto")
    g.add_argument("--catalog", default=None, help="data folder with CSVs or products.db")
    g.add_argument("--upscaler", default="bicubic")
    g.add_argument("--chip-mode", default="classic", choices=["classic", "faint", "both"])
    g.add_argument("--gallery", default=None,
                   help="Click gallery folder (python -m vision.clicks build); enables Clicks")
    g.add_argument("--click-model", default="models/dinov2s_reg_224.onnx")
    a = ap.parse_args(argv)

    from ..ocr import available_backends
    from .bench import State
    from .net import cert_fingerprint, ensure_cert, lan_ips, ssl_context
    from .server import AppServer

    if not available_backends():
        print("warning: no OCR backend installed (paddleocr / rapidocr_onnxruntime / tesseract)",
              file=sys.stderr)
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

    state = State(a.catalog, a.photos_dir, a.gallery, a.click_model,
                  defaults={"backend": a.backend, "upscaler": a.upscaler, "chip_mode": a.chip_mode})
    srv = AppServer(state, a.captures_dir, a.phone_port, a.port, ips, ctx, a.rectify, fp,
                    recognize=not a.no_recognize)

    async def run():
        await srv.start()
        ui = f"http://127.0.0.1:{a.port}/"
        print(f"vision app  {ui}")
        for u in srv.phone_urls():
            print(f"     phone  {u}")
        if fp:
            print(f"     certificate {fp}… (self-signed, accept it on the phone once)")
        print("     Ctrl+C to stop")
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
        print(f"cannot listen: {e} (port in use? try --port / --phone-port)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
