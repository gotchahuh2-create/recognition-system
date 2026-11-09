"""Second-chance reading for small text: find it, cut it out, clean it up, read it again.

OCR engines (Tesseract especially) fail on tiny or blurry markings unless
the text is first cut out, enlarged (4-6x bicubic, or super-resolution),
cleaned up and turned black and white. Finding the text first and then
reading just that crop works much better than reading the whole image.

Step 1: find areas that might hold text (in processed-image coordinates):
  * **seeded**: words the whole-photo pass already found but couldn't
    match confidently (pieces of a part number, low-confidence reads).
    We widen the box along the line, because missing characters are
    usually at the ends.
  * **detected**: a simple shape-based text-line detector that finds
    markings the first pass missed completely (blur, low light).

Step 2: for each area, cut a straightened crop out of the *original* photo
(which has the most pixels), make several cleaned-up versions with
``enhance.enhance_crop``, and read them as a single line or small block
(``OcrBackend.read_crop``; for Tesseract that means single-line mode with
only part-number characters allowed and dictionaries turned off).

The words we return are in processed-image coordinates, so the usual
matchers (mcu_ocr.candidate_strings / score_parts) can use them as-is. The
catalog match then acts as the spell-check.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import re
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from . import enhance
from .ocr import OcrBackend, OcrToken, TesseractBackend, group_lines
from .preprocess import Preprocessed, four_point_warp, rotate90

_ALNUM = re.compile(r"[A-Z0-9]")


@dataclass
class Region:
    pts: np.ndarray  # 4x2 corners in processed-image coordinates (any order)
    glyph_px: float  # estimated letter height in processed px
    vertical: bool  # text runs top to bottom
    block: bool                # probably several lines
    source: str  # "seed" or "detect"
    score: float = 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0, y0 = self.pts.min(0)
        x1, y1 = self.pts.max(0)
        return float(x0), float(y0), float(x1), float(y1)


@dataclass
class CropRead:
    """What one area produced. Kept for the UI and debug output."""
    region: Region
    # (variant, rotation, text, confidence)
    reads: list[tuple[str, int, str, float]] = field(default_factory=list)
    crop: np.ndarray | None = None  # the straightened crop (BGR)
    best_variant: np.ndarray | None = None  # the cleaned-up image that gave the best read

    def as_dict(self) -> dict:
        return {"box": np.round(self.region.pts).astype(int).tolist(),
                "source": self.region.source, "glyph_px": round(self.region.glyph_px, 1),
                "block": self.region.block,
                "reads": [{"variant": v, "rotation": r, "text": t, "conf": round(c, 3)}
                          for v, r, t, c in self.reads]}


# ---- Step 1: finding text areas ----
def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _contains(a, b, tol: float = 0.9) -> bool:
    """True if box b is (mostly) inside box a."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return area_b > 0 and inter / area_b >= tol


def _rect_pts(cx, cy, w, h, ang) -> np.ndarray:
    return cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float64)


def seed_regions(tokens: Sequence[OcrToken], interesting) -> list[Region]:
    """Make line boxes around words that look like part of a chip marking."""
    out: list[Region] = []
    for line in group_lines(tokens):
        if not any(interesting(t) for t in line):
            continue
        pts = np.vstack([t.poly for t in line]).astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        hgt = float(np.median([t.height for t in line]))
        vertical = line[0].rotation in (90, 270)
        # widen along the text (to catch lost end characters) and a little across it
        if (w >= h) != vertical:
            w, h = w + 3.0 * hgt, h + 0.8 * hgt
        else:
            w, h = w + 0.8 * hgt, h + 3.0 * hgt
        out.append(Region(_rect_pts(cx, cy, w, h, ang), hgt, vertical, False, "seed", 2.0))
    return out


def detect_text_regions(gray: np.ndarray, max_regions: int = 16,
                        min_h: int = 7, max_h: int = 160) -> list[Region]:
    """Find text lines using simple image operations (no trained model).

    Text is lots of short, high-contrast strokes close together. We
    highlight strokes (light or dark), smear them along the reading
    direction so the letters of a line merge into one blob, and keep the
    blobs whose shape, fill and stroke pattern look like text. This runs for
    both horizontal and vertical text.
    """
    h0, w0 = gray.shape[:2]
    s = min(1.0, 1600 / max(h0, w0))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    # noise would look like "strokes", so smooth more when the photo is noisy
    sig = enhance.noise_sigma(g)
    if sig > 4:
        g = cv2.GaussianBlur(cv2.medianBlur(g, 5), (5, 5), 0)
    else:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    out: list[Region] = []
    for k in (31, 61):  # largest letter size the filter keeps
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        hat = np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se),
                         cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se))
        hat = cv2.GaussianBlur(hat, (3, 3), 0)
        otsu_t, _ = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Text covers only a few % of the photo. On noisy photos Otsu would split the
        # noise instead, so we also require pixels to stand well out from the
        # median (median + 4 * MAD).
        med = float(np.median(hat))
        mad = float(np.median(np.abs(hat.astype(np.float32) - med))) * 1.4826
        thr = max(otsu_t, med + 4 * mad, 10)
        bw = np.where(hat > thr, 255, 0).astype(np.uint8)
        for vertical in (False, True):
            kw, kh = max(5, k // 3), 3
            ks = (kh, kw) if vertical else (kw, kh)
            m = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, ks))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            # RETR_LIST, not EXTERNAL: otherwise a chip's light outline would hide the
            # marking inside it
            cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                (cx, cy), (w, h), ang = cv2.minAreaRect(c)
                long, short = max(w, h), min(w, h)
                if short < min_h * s or short > max_h * s or long > 0.85 * max(g.shape):
                    continue
                aspect = long / max(short, 1)
                if aspect < 1.8 or aspect > 30:
                    continue
                x, y, bwid, bhgt = cv2.boundingRect(c)
                if (bhgt > bwid) != vertical:  # the blob must run the same way we smeared it
                    continue
                roi = bw[y:y + bhgt, x:x + bwid]
                fill = float((roi > 0).mean()) if roi.size else 0.0
                if fill < 0.15:
                    continue
                # Stroke pattern: a line drawn along the text crosses many letter strokes,
                # while a trace or pad edge is crossed only once or twice. (We threshold the
                # blob locally, because on blurry text the global mask fills the gaps
                # between letters.)
                rg = g[y:y + bhgt, x:x + bwid]
                if rg.size == 0:
                    continue
                _, lb = cv2.threshold(rg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                r2 = (lb.T if vertical else lb) > 0
                runs = np.count_nonzero(np.diff(r2.astype(np.int8), axis=1) == 1, axis=1)
                trans = int(np.percentile(runs, 75)) if runs.size else 0
                if trans < 3:
                    continue
                block = aspect < 3.2
                glyph = short / (2.6 if block else 1.25)
                score = min(fill * 2, 1.0) * min(aspect / 4, 1.5) * (1 + min(trans, 12) / 12)
                pad = 0.35 * glyph
                pts = _rect_pts(cx / s, cy / s, (w + 2 * pad) / s, (h + 2 * pad) / s, ang)
                out.append(Region(pts, glyph / s, vertical, block, "detect", score))
    out.sort(key=lambda r: -r.score)
    return _nms(out, max_regions)


def _nms(regions: list[Region], keep: int, iou: float = 0.45) -> list[Region]:
    out: list[Region] = []
    for r in sorted(regions, key=lambda r: -r.score):
        if any(_iou(r.bbox, o.bbox) > iou or _contains(o.bbox, r.bbox) for o in out):
            continue
        out.append(r)
        if len(out) >= keep:
            break
    return out


# ---- Main entry point ----
@dataclass
class RefineConfig:
    max_regions: int = 10
    upscaler: str = "bicubic"  # bicubic, lanczos, espcn, fsrcnn or edsr
    variants: tuple[str, ...] = enhance.CROP_VARIANTS
    target_glyph: float = enhance.TARGET_GLYPH_PX
    detect: bool = True  # also look for text areas the first pass missed
    # With dictionaries off, Tesseract often reports confidence 0 for strings
    # that are correct but unusual. So we don't filter crop reads by confidence;
    # the strict catalog match (mcu_ocr.ACCEPT / MARGIN) does the filtering.
    min_conf: float = 0.0


def refine(pre: Preprocessed, backend: OcrBackend, tokens: Sequence[OcrToken] = (),
           hires: np.ndarray | None = None, cfg: RefineConfig | None = None,
           interesting=None, workers: int | None = None, stop=None
           ) -> tuple[list[OcrToken], list[CropRead]]:
    """Run steps 1 and 2. Returns (new words in processed coordinates, a log per crop)."""
    cfg = cfg or RefineConfig()
    interesting = interesting or (lambda t: len(_ALNUM.findall(t.text.upper())) >= 3
                                  and any(ch.isdigit() for ch in t.text))
    regions = seed_regions(tokens, interesting)
    if cfg.detect:
        regions += detect_text_regions(pre.gray, cfg.max_regions)
    regions = _nms(regions, cfg.max_regions)
    if not regions:
        return [], []

    rec = backend
    src = hires if hires is not None else pre.image
    use_orig = hires is not None

    def job(ri_region):
        ri, r = ri_region
        pts_src = pre.to_original(r.pts) if use_orig else r.pts
        crop, _ = four_point_warp(src, pts_src)
        if crop.size == 0 or min(crop.shape[:2]) < 4:
            return ri, CropRead(r), []
        # the letter size in the *original* photo decides how much to enlarge
        ratio = (np.linalg.norm(pts_src[0] - pts_src[1]) /
                 max(np.linalg.norm(r.pts[0] - r.pts[1]), 1e-6))
        glyph_src = r.glyph_px * ratio
        crop_rot = crop
        rots = (0, 180)
        if crop.shape[0] > crop.shape[1] * 1.2:  # tall crop, so the text runs vertically
            rots = (90, 270)
        if getattr(rec, "handles_rotation", False):  # the engine flips 180 degree lines by itself
            rots = rots[:1]
        variants = enhance.enhance_crop(crop, glyph_src, cfg.upscaler, cfg.variants,
                                        cfg.target_glyph)
        log = CropRead(r, crop=crop)
        toks: list[OcrToken] = []
        best_conf = -1.0

        def read_one(vname, vimg, rot):
            img = rotate90(vimg, rot // 90)
            try:
                reads = rec.read_crop(img, block=r.block)
            except Exception as e:  # one bad crop shouldn't break the whole photo
                log.reads.append((vname, rot, f"<error {type(e).__name__}>", 0.0))
                return img, []
            lines: dict[tuple, list[OcrToken]] = {}
            for t in reads:
                lines.setdefault(tuple(t.line_key), []).append(t)
            out = []
            for li, (_, ts) in enumerate(sorted(lines.items(),
                                                key=lambda kv: kv[1][0].center[1])):
                ts.sort(key=lambda t: t.center[0])
                text = " ".join(t.text for t in ts).strip()
                conf = float(np.mean([t.conf for t in ts]))
                if text and conf >= cfg.min_conf and len(_ALNUM.findall(text)) >= 3:
                    out.append((li, text, conf))
            return img, out

        def plaus(reads) -> float:
            return sum(len(_ALNUM.findall(t)) * (0.2 + c) for _, t, c in reads)

        # Try both orientations with the first variant, then only keep the better one
        # (unless they're close). This halves the number of Tesseract calls.
        probe = {rot: read_one(variants[0][0], variants[0][1], rot) for rot in rots}
        ranked = sorted(rots, key=lambda rt: -plaus(probe[rt][1]))
        keep = [ranked[0]]
        if len(ranked) > 1 and plaus(probe[ranked[1]][1]) >= 0.8 * plaus(probe[ranked[0]][1]):
            keep.append(ranked[1])
        for vi, (vname, vimg) in enumerate(variants):
            for rot in keep:
                img, reads = probe[rot] if vi == 0 else read_one(vname, vimg, rot)
                for li, text, conf in reads:
                    log.reads.append((vname, rot, text, conf))
                    toks.append(OcrToken(text, conf, r.pts.copy(), rotation=rot,
                                         variant=f"refine:{ri}:{vname}:{rot}:{li}",
                                         line_key=("refine", ri, vname, rot, li)))
                    if conf > best_conf:
                        best_conf, log.best_variant = conf, img
        return ri, log, toks

    jobs = list(enumerate(regions))
    n = workers if workers is not None else (min(8, os.cpu_count() or 2)
                                             if isinstance(rec, TesseractBackend) else 1)
    results = []
    # Areas are sorted best first. ``stop`` lets the caller quit early once a
    # crop already gave a confident catalog match.
    if n > 1 and len(jobs) > 1:
        with cf.ThreadPoolExecutor(n) as ex:
            futs = [ex.submit(job, j) for j in jobs]
            for f in cf.as_completed(futs):
                results.append(f.result())
                if stop is not None and stop([t for _, _, ts in results for t in ts]):
                    for g in futs:
                        g.cancel()
                    break
    else:
        for j in jobs:
            results.append(job(j))
            if stop is not None and stop([t for _, _, ts in results for t in ts]):
                break
    results.sort(key=lambda x: x[0])
    tokens_out = [t for _, _, ts in results for t in ts]
    return tokens_out, [log for _, log, _ in results]
