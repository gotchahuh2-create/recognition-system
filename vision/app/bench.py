"""The Photos tab: recognise dropped photos, re-run them with other settings, and save them
with a label.

The HTTP routes are in ``server.py`` (``/api/recognize``, ``/api/rerun``,
``/api/label``). This module keeps the state and builds the responses,
and the scan flow uses it too.
"""

from __future__ import annotations

import base64
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from ..catalog import Catalog
from ..enhance import available_upscalers
from ..loader import PhotoLoadError, load_bytes  # noqa: F401
from ..ocr import available_backends
from ..pipeline import Pipeline
from ..preprocess import PreprocessConfig, resize_long_side
from ..refine import RefineConfig

MAX_UPLOAD = 60 * 1024 * 1024
PREVIEW_LONG = 1600


# ---- State ----
class State:
    def __init__(self, catalog_src, photos_dir: Path, gallery: str | None = None,
                 click_model: str | None = None, defaults: dict | None = None):
        self.catalog = Catalog.load(catalog_src)
        # settings for the scan (and the starting values of the Photos tab menus)
        self.defaults = {"backend": "auto", "refine": "auto", "upscaler": "bicubic",
                         "auto_perspective": False, "long_side": 2000, "clicks": "boxes",
                         "chip_mode": "classic", **(defaults or {})}
        self.catalog_src = catalog_src
        self.photos_dir = photos_dir
        self.photos: OrderedDict[str, tuple[bytes, str]] = OrderedDict()  # id -> (bytes, name)
        self.pipes: dict[tuple, Pipeline] = {}
        # recognition runs on a single worker thread (see server.py) because OCR engines
        # aren't thread-safe
        self.gallery_dir, self.click_model = gallery, click_model
        self.click_error: str | None = None
        self._click_parts = None  # (embedder, gallery), loaded only once
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
        """Return a ClickRecognizer for this OCR engine. They all share one embedder and
        gallery.
        """
        if not self.gallery_dir:
            return None
        if backend in self._click_recs:
            return self._click_recs[backend]
        from ..clicks import ClickRecognizer, Gallery, get_embedder
        from ..clicks.recognizer import ClickConfig
        try:
            if self._click_parts is None:
                emb = get_embedder(self.click_model)
                self._click_parts = (emb, Gallery.load(self.gallery_dir, emb.model_version,
                                                       emb.preprocess_version))
            emb, gal = self._click_parts
            cfg = ClickConfig.from_file(Path(self.gallery_dir) / "thresholds.json")
            rec = ClickRecognizer(emb, gal, backend, self.catalog, cfg)
        except Exception as e:  # model or gallery missing: keep OCR working anyway
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
               opt["auto_perspective"], opt["long_side"], opt["clicks"], opt["chip_mode"])
        if key not in self.pipes:
            self.pipes[key] = Pipeline(
                self.catalog, opt["backend"],
                PreprocessConfig(long_side=opt["long_side"],
                                 auto_perspective=opt["auto_perspective"]),
                refine=opt["refine"],
                refine_cfg=RefineConfig(upscaler=opt["upscaler"]), upscaler=opt["upscaler"],
                chip_mode=opt["chip_mode"],
                clicks=self.click_recognizer(opt["backend"]) if opt["clicks"] != "off" else None,
                click_mode=opt["clicks"])
        return self.pipes[key]

    def scan_pipeline(self) -> Pipeline:
        """The pipeline used by the two-step scan (with the command-line settings)."""
        return self.pipeline({**self.defaults, "quad": None, "click_boxes": []})

    def board_pipeline(self) -> Pipeline:
        """A fast board-only pass for the scan's first photo.

        The board name is printed large, so plain OCR at 2000 px is enough
        (1600 px missed it on a 4K iPhone photo). It skips the MCU and card
        steps, the crop re-reads and the layout fit, because the MCU comes from
        the close-up anyway. On a real 4K board photo with Tesseract this takes
        5 s instead of 20 s.
        """
        if "board-fast" not in self.pipes:
            self.pipes["board-fast"] = Pipeline(
                self.catalog, self.defaults["backend"],
                PreprocessConfig(long_side=2000, auto_perspective=False),
                mcu=False, refine="off", layouts=False, card_stage=False)
        return self.pipes["board-fast"]

    def info(self) -> dict:
        c = self.catalog
        return {"backends": available_backends(), "upscalers": available_upscalers(),
                "clicks": bool(self.gallery_dir), "click_error": self.click_error,
                "photos_dir": str(self.photos_dir.resolve()),
                "defaults": {k: v for k, v in self.defaults.items()},
                "catalog": {"board": [b.name for b in c.boards],
                            "mcu": [m.part for m in c.mcus],
                            "mcu_card": [m.name for m in c.mcu_cards],
                            "click": sorted({k.pid for k in c.clicks} | set(self.gallery_names()))}}

    def recognize(self, data: bytes, name: str, opt: dict) -> dict:
        """Recognise one photo. This blocks, so only call it from the recognition thread."""
        photo = load_bytes(data, name)  # a PhotoLoadError becomes HTTP 400
        pid = photo.sha1[:16]
        self.remember(data, name, pid)
        res = self.pipeline(opt).run(photo, quad=opt["quad"], click_boxes=opt["click_boxes"])
        return build_response(self, pid, name, photo, res, opt)

    def save_label(self, req: dict) -> tuple[dict, int]:
        """Copy a stored photo into ``photos_dir`` and add or update its row in ``labels.csv``.

        ``eval/ocr_eval.py`` scores photos from that file.
        """
        item = self.photos.get(req.get("id", ""))
        kind = (req.get("kind") or "").strip()
        label = (req.get("label") or "").strip()
        if item is None:
            return {"error": "This photo is no longer in memory. Add it again."}, 410
        if kind not in ("board", "mcu", "mcu_card", "click") or not label:
            return {"error": "Choose a kind (board, mcu, mcu_card, click) and enter a label."}, 400
        data, name = item
        folder = self.photos_dir
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
        return {"saved": str(target), "labels": str(labels), "rows": len(existing)}, 200


def parse_options(q: dict, defaults: dict | None = None) -> dict:
    dfl = defaults or {}

    def get(k, default):
        default = dfl.get(k, default)
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
            "chip_mode": get("chip_mode", "classic") if get("chip_mode", "classic")
            in ("classic", "faint", "both") else "classic",
            "long_side": int(get("long_side", 2000)), "quad": pts,
            "clicks": get("clicks", "boxes") if get("clicks", "boxes") in ("off", "boxes", "photo")
            else "boxes", "click_boxes": boxes}


# ---- Image encoding helpers ----
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
    # overlays for both the processed photo (as computed) and the original photo
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
    for r in res.rois:  # layout areas (MCU card, sockets)
        add("roi", r.name, r.quad_pre)
    if res.card_crop is not None:  # where the card was found on the photo
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
    # crops of the MCU card and every mikroBUS socket: what the Click step will see
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


