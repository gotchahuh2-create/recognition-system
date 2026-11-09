"""Drag-and-drop test window for real photos.

    python -m vision.app                     # opens http://127.0.0.1:8765
    python -m vision.app --port 9000 --photos-dir photos --no-browser

A local web page (no extra dependencies: stdlib http.server + OpenCV) where
you drop photos (or paste / pick them) and see, per photo:

* board / MCU / MCU-card decisions with confidence, reason and top-3 options,
* OCR evidence boxes and the refined crops drawn on the photo (original or
  preprocessed view), the enhanced crop images and what was read from them,
* quality warnings, preprocessing steps, timings, raw JSON,
* re-run with other settings (backend, refinement, upscaler, auto-perspective)
  or with a 4-click perspective correction,
* "Save to dataset": copies the photo into ``--photos-dir`` and appends a row
  to its ``labels.csv`` so ``python eval/ocr_eval.py photos/`` can score it.
* Clicks (with ``--gallery``): draw a box per Click (or treat the whole photo
  as one Click) -> DINOv2 retrieval -> accept / ask (top-3) / unknown.

Listens on 127.0.0.1 only. Photos stay in memory (last 40) until saved.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import threading
import traceback
import urllib.parse
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .catalog import Catalog
from .enhance import available_upscalers
from .loader import PhotoLoadError, load_bytes
from .ocr import available_backends
from .pipeline import Pipeline
from .preprocess import PreprocessConfig, resize_long_side
from .refine import RefineConfig

STATIC = Path(__file__).resolve().parent / "webui"
MAX_UPLOAD = 60 * 1024 * 1024
PREVIEW_LONG = 1600


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, catalog_src, photos_dir: Path, gallery: str | None = None,
                 click_model: str | None = None):
        self.catalog = Catalog.load(catalog_src)
        self.catalog_src = catalog_src
        self.photos_dir = photos_dir
        self.photos: OrderedDict[str, tuple[bytes, str]] = OrderedDict()  # id -> (bytes, name)
        self.pipes: dict[tuple, Pipeline] = {}
        self.lock = threading.Lock()          # OCR engines are not all thread-safe
        self.gallery_dir, self.click_model = gallery, click_model
        self.click_error: str | None = None
        self._click_parts = None              # (embedder, gallery), loaded once
        self._click_recs: dict[str, object] = {}

    def gallery_names(self) -> list[str]:
        if not self.gallery_dir:
            return []
        try:
            info = json.loads((Path(self.gallery_dir) / "gallery.json").read_text("utf-8"))
            return list(info.get("names", {}))
        except (OSError, ValueError):
            return []

    def click_recognizer(self, backend: str):
        """One ClickRecognizer per OCR backend, sharing embedder + gallery."""
        if not self.gallery_dir:
            return None
        if backend in self._click_recs:
            return self._click_recs[backend]
        from .clicks import ClickRecognizer, Gallery, get_embedder
        from .clicks.recognizer import ClickConfig
        try:
            if self._click_parts is None:
                emb = get_embedder(self.click_model)
                self._click_parts = (emb, Gallery.load(self.gallery_dir, emb.model_version,
                                                       emb.preprocess_version))
            emb, gal = self._click_parts
            cfg = ClickConfig.from_file(Path(self.gallery_dir) / "thresholds.json")
            rec = ClickRecognizer(emb, gal, backend, self.catalog, cfg)
        except Exception as e:      # missing model / gallery: keep OCR working
            self.click_error = f"{type(e).__name__}: {e}"
            print(f"[clicks] disabled: {self.click_error}")
            rec = None
        self._click_recs[backend] = rec
        return rec

    def remember(self, data: bytes, name: str, pid: str) -> None:
        self.photos[pid] = (data, name)
        self.photos.move_to_end(pid)
        while len(self.photos) > 40:
            self.photos.popitem(last=False)

    def pipeline(self, opt: dict) -> Pipeline:
        key = (opt["backend"], opt["refine"], opt["upscaler"],
               opt["auto_perspective"], opt["long_side"], opt["clicks"])
        if key not in self.pipes:
            self.pipes[key] = Pipeline(
                self.catalog, opt["backend"],
                PreprocessConfig(long_side=opt["long_side"],
                                 auto_perspective=opt["auto_perspective"]),
                refine=opt["refine"],
                refine_cfg=RefineConfig(upscaler=opt["upscaler"]),
                clicks=self.click_recognizer(opt["backend"]) if opt["clicks"] != "off" else None,
                click_mode=opt["clicks"])
        return self.pipes[key]


def parse_options(q: dict) -> dict:
    def get(k, default):
        v = q.get(k)
        if isinstance(v, list):
            v = v[0] if v else None
        return default if v in (None, "") else v
    quad = get("quad", None)
    pts = None
    if quad:
        vals = [float(x) for x in str(quad).split(",")]
        if len(vals) == 8:
            pts = np.array(vals, float).reshape(4, 2)
    boxes = []
    for part in str(get("click_boxes", "") or "").split(";"):
        vals = [float(x) for x in part.split(",") if x.strip()]
        if len(vals) == 4 and vals[2] > 1 and vals[3] > 1:
            boxes.append(vals)
    return {"backend": get("backend", "auto"), "refine": get("refine", "auto"),
            "upscaler": get("upscaler", "bicubic"),
            "auto_perspective": str(get("auto_perspective", "0")).lower() in ("1", "true", "on"),
            "long_side": int(get("long_side", 2000)), "quad": pts,
            "clicks": get("clicks", "boxes") if get("clicks", "boxes") in ("off", "boxes", "photo")
            else "boxes", "click_boxes": boxes}


# --------------------------------------------------------------------------- #
# encoding helpers
# --------------------------------------------------------------------------- #
def b64img(img: np.ndarray, ext: str = ".jpg", quality: int = 85) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _poly(p) -> list[list[float]]:
    return np.round(np.asarray(p, float).reshape(-1, 2), 1).tolist()


def _to_pre(pre, pts: np.ndarray) -> np.ndarray:
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ pre.M.T
    return hom[:, :2] / hom[:, 2:3]


def build_response(state: State, pid: str, name: str, photo, res, opt: dict) -> dict:
    pre = res.pre
    d = res.as_dict()
    # overlays in both frames: processed (as computed) and original photo
    overlays = []

    def add(kind, label, poly):
        if poly is None:
            return
        p = np.asarray(poly, float).reshape(-1, 2)
        overlays.append({"kind": kind, "label": label, "pre": _poly(p),
                         "orig": _poly(pre.to_original(p))})

    for ev in res.board.evidence:
        add("board", ev.get("text", ""), ev.get("box"))
    for i, c in enumerate(res.clicks):
        if c.box is not None:
            x, y, w, h = c.box
            p = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
            top = c.recognition.candidates[0].name if c.recognition.candidates else "?"
            label = f"{i + 1}: {top}" + ("" if c.outcome == "accept" else f" ({c.outcome})")
            overlays.append({"kind": "click", "label": label,
                             "pre": _poly(_to_pre(pre, p)), "orig": _poly(p)})
    for r in res.rois:              # stage-1b layout areas (MCU card, sockets)
        add("roi", r.name, r.quad_pre)
    if res.card_crop is not None:   # 0.6: the card as located on the photo
        q = res.card_crop.location.quad
        hom = np.hstack([q, np.ones((4, 1))]) @ pre.M.T
        add("card", f"MCU card ({res.card_crop.location.method})", hom[:, :2] / hom[:, 2:3])
    for c in res.mcu.candidates[:1]:
        if c.detail.get("box") is not None:
            add("mcu", c.detail.get("read", ""), c.detail["box"])
    crops = []
    for i, c in enumerate(res.mcu.crops):
        best = max(c.reads, key=lambda r: r[3], default=None)
        add("crop", f"#{i + 1}", c.region.pts)
        img = c.best_variant
        if img is not None and img.shape[1] > 700:
            img = cv2.resize(img, None, fx=700 / img.shape[1], fy=700 / img.shape[1],
                             interpolation=cv2.INTER_AREA)
        src = c.crop
        if src is not None and max(src.shape[:2]) > 500:
            src = resize_long_side(src, 500, allow_upscale=False)[0]
        crops.append({"index": i + 1, "source": c.region.source,
                      "glyph_px": round(c.region.glyph_px, 1),
                      "reads": [{"variant": v, "rotation": r, "text": t, "conf": round(cf, 2)}
                                for v, r, t, cf in c.reads],
                      "best": best[2] if best else None,
                      "enhanced": b64img(img, ".png") if img is not None else None,
                      "source_img": b64img(src) if src is not None else None})
    # stage-1b crops (MCU card + every mikroBUS socket): what stage 2 will see
    rois = [{"name": r.name, "visible": round(r.visible, 2),
             "size": [int(r.image.shape[1]), int(r.image.shape[0])],
             "img": b64img(resize_long_side(r.image, 360, allow_upscale=False)[0])}
            for r in res.rois]
    card = None
    if res.card_crop is not None:
        cc, rd = res.card_crop, res.card_reading
        cimg = cc.image.copy()
        if rd is not None and rd.chip is not None:
            cv2.polylines(cimg, [rd.chip.quad.astype(np.int32).reshape(-1, 1, 2)], True,
                          (0, 170, 255), max(2, cimg.shape[0] // 300))
        card = {**cc.as_dict(),
                "img": b64img(resize_long_side(cimg, 520, allow_upscale=False)[0]),
                "chip_img": (b64img(resize_long_side(rd.chip.image, 360, allow_upscale=False)[0])
                             if rd is not None and rd.chip is not None and rd.chip.image is not None
                             else None),
                "text": rd.text_lines(60) if rd is not None else [],
                "passes": rd.passes if rd is not None else {},
                "used": res.mcu_source == "card"}
    d.pop("pre", None)
    for k in ("board", "mcu", "mcu_card"):
        d[k].pop("crops", None)
    orig_prev, s_o = resize_long_side(photo.image, PREVIEW_LONG, allow_upscale=False)
    pre_prev, s_p = resize_long_side(pre.image, PREVIEW_LONG, allow_upscale=False)
    return {
        "id": pid, "name": name, "result": d, "options": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in opt.items()},
        "original": {"src": b64img(orig_prev), "w": photo.width, "h": photo.height},
        "processed": {"src": b64img(pre_prev), "w": pre.image.shape[1], "h": pre.image.shape[0]},
        "overlays": overlays, "crops": crops, "rois": rois, "card": card,
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"vision-ui/{__version__}"

        def log_message(self, fmt, *args):   # quieter console
            if "/api/" in (args[0] if args else ""):
                super().log_message(fmt, *args)

        # ---------------------------------------------------------- helpers
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError(f"file too large ({n / 1e6:.0f} MB)")
            return self.rfile.read(n)

        # ------------------------------------------------------------ routes
        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if url.path == "/api/info":
                c = state.catalog
                return self._json({
                    "version": __version__, "backends": available_backends(),
                    "clicks": bool(state.gallery_dir), "click_error": state.click_error,
                    "upscalers": available_upscalers(),
                    "photos_dir": str(state.photos_dir.resolve()),
                    "catalog": {"board": [b.name for b in c.boards],
                                "mcu": [m.part for m in c.mcus],
                                "mcu_card": [m.name for m in c.mcu_cards],
                                "click": sorted({k.pid for k in c.clicks} |
                                                set(state.gallery_names()))}})
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            try:
                if url.path == "/api/recognize":
                    data = self._body()
                    name = urllib.parse.unquote(self.headers.get("X-Filename") or "photo.jpg")
                    return self._recognize(data, name, parse_options(q))
                if url.path == "/api/rerun":
                    req = json.loads(self._body() or b"{}")
                    item = state.photos.get(req.get("id", ""))
                    if item is None:
                        return self._json({"error": "photo no longer in memory; drop it again"}, 410)
                    return self._recognize(item[0], item[1], parse_options(req.get("options", {})))
                if url.path == "/api/label":
                    return self._label(json.loads(self._body() or b"{}"))
            except Exception as e:     # report to the page instead of a dead socket
                traceback.print_exc()
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            self._send(404, b"not found", "text/plain")

        def _recognize(self, data: bytes, name: str, opt: dict):
            try:
                photo = load_bytes(data, name)
            except PhotoLoadError as e:
                return self._json({"error": str(e)}, 400)
            pid = photo.sha1[:16]
            state.remember(data, name, pid)
            with state.lock:
                pipe = state.pipeline(opt)
                res = pipe.run(photo, quad=opt["quad"], click_boxes=opt["click_boxes"])
            return self._json(build_response(state, pid, name, photo, res, opt))

        def _label(self, req: dict):
            item = state.photos.get(req.get("id", ""))
            kind = (req.get("kind") or "").strip()
            label = (req.get("label") or "").strip()
            if item is None:
                return self._json({"error": "photo no longer in memory; drop it again"}, 410)
            if kind not in ("board", "mcu", "mcu_card", "click") or not label:
                return self._json({"error": "need kind (board/mcu/mcu_card/click) and label"},
                                  400)
            data, name = item
            folder = state.photos_dir
            folder.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"[^\w.-]+", "_", Path(name).stem) or "photo"
            ext = Path(name).suffix.lower() or ".jpg"
            target = folder / f"{stem}{ext}"
            labels = folder / "labels.csv"
            existing = {}
            if labels.is_file():
                with open(labels, newline="", encoding="utf-8") as f:
                    existing = {r["filename"]: r for r in csv.DictReader(f)}
            i = 1
            while target.exists() and target.read_bytes() != data:
                target = folder / f"{stem}_{i}{ext}"
                i += 1
            target.write_bytes(data)
            dist = str(req.get("distance_cm") or "").strip()
            box = req.get("box") or ""
            if isinstance(box, (list, tuple)):
                box = ",".join(str(int(round(float(v)))) for v in box[:4])
            existing[target.name] = {"filename": target.name, "kind": kind, "label": label,
                                     "distance_cm": dist, "box": box,
                                     "session": (req.get("session") or "").strip()}
            cols = ["filename", "kind", "label", "distance_cm", "box", "session"]
            with open(labels, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, cols, extrasaction="ignore")
                w.writeheader()
                for r in existing.values():
                    w.writerow({k: r.get(k, "") for k in cols})
            return self._json({"saved": str(target), "labels": str(labels),
                               "rows": len(existing)})

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vision.app", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--catalog", default=None, help="data folder with CSVs or products.db")
    ap.add_argument("--photos-dir", type=Path, default=Path("photos"),
                    help="where 'Save to dataset' copies photos and labels.csv")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--gallery", default=None,
                    help="Click gallery folder (python -m vision.clicks build); enables Clicks")
    ap.add_argument("--click-model", default="models/dinov2s_reg_224.onnx")
    a = ap.parse_args(argv)
    if not available_backends():
        print("warning: no OCR backend installed (paddleocr / rapidocr_onnxruntime / tesseract)")
    state = State(a.catalog, a.photos_dir, a.gallery, a.click_model)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(state))
    url = f"http://{a.host}:{a.port}/"
    print(f"vision test window: {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
