"""All OCR engines behind one common interface.

Supported engines (all Apache-2.0 licensed, so fine to ship in a
closed-source MIKROE tool):

* ``paddle``    PaddleOCR. Works with both the 2.x (``.ocr``) and 3.x
                (``.predict``) APIs. ``pip install paddleocr paddlepaddle``
* ``rapidocr``  The same PP-OCR models running on ONNX Runtime. Lightweight
                and CPU-only. ``pip install rapidocr_onnxruntime`` (or
                ``rapidocr``)
* ``tesseract`` The classic engine, called through the ``tesseract``
                command (no Python binding needed). Worse at reading
                silkscreen, but always available, so it's used when the
                others aren't installed.

``get_backend("auto")`` uses the first one that is installed, in the order
paddle, rapidocr, tesseract.

Every engine returns ``OcrToken`` objects whose outline is in the
coordinates of the image passed to ``read_text`` (rotations, variants and
tiles are all mapped back).
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import io
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from .preprocess import iter_tiles, ocr_variants, rotate90


@dataclass
class OcrToken:
    text: str
    conf: float                    # 0..1
    poly: np.ndarray  # 4x2 float, in the coordinates of the image given to read_text
    rotation: int = 0  # how far the image was rotated for this read: 0, 90, 180 or 270
    variant: str = "gray"
    line_key: tuple = field(default=())  # words with the same key are on the same line

    @property
    def center(self) -> tuple[float, float]:
        c = self.poly.mean(0)
        return float(c[0]), float(c[1])

    @property
    def height(self) -> float:
        # the short side of the box is about the letter height, whatever the rotation
        e = np.linalg.norm(np.diff(np.vstack([self.poly, self.poly[:1]]), axis=0), axis=1)
        return float(min(e[0], e[1]))

    def as_dict(self) -> dict:
        return {"text": self.text, "conf": round(self.conf, 3),
                "box": np.round(self.poly).astype(int).tolist(),
                "rotation": self.rotation, "variant": self.variant}


TESS_NOISE_MAX = 6.0


class OcrBackend:
    name = "base"
    # image versions worth running for this engine (see preprocess.ocr_variants)
    default_variants: tuple[str, ...] = ("gray",)
    # True when the engine can flip upside-down lines by itself (0 vs 180 degrees).
    # 90 and 270 degrees still need their own pass.
    handles_rotation = False

    def read(self, img: np.ndarray) -> list[OcrToken]:  # pragma: no cover
        raise NotImplementedError

    def read_crop(self, img: np.ndarray, block: bool = False) -> list[OcrToken]:
        """Read a single text line (or small block) that was already cut out and cleaned up.

        Neural engines run their own text detector on the crop, which is fine.
        """
        return self.read(img)


# ---- Tesseract (command line) ----
class TesseractBackend(OcrBackend):
    name = "tesseract"
    default_variants = ("gray", "inverted")

    # The characters that appear on chip markings. Limiting Tesseract to these
    # (and turning off its English dictionaries) helps a lot, because part
    # numbers aren't English words.
    CROP_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/."

    def __init__(self, lang: str = "eng", psm: int = 11, cmd: str | None = None,
                 timeout: float = 30.0):
        self.cmd = cmd or shutil.which("tesseract")
        if not self.cmd:
            raise RuntimeError("tesseract binary not found")
        self.lang, self.psm, self.timeout = lang, psm, timeout

    def read_crop(self, img: np.ndarray, block: bool = False) -> list[OcrToken]:
        # psm 6 ("uniform block") also reads single lines, while psm 7 returns
        # garbage when a "line" crop actually contains two marking lines.
        return self.read(img, psm=6,
                         extra=["-c", f"tessedit_char_whitelist={self.CROP_WHITELIST}",
                                "-c", "load_system_dawg=0", "-c", "load_freq_dawg=0"])

    def read(self, img: np.ndarray, psm: int | None = None,
             extra: list[str] | None = None) -> list[OcrToken]:
        ok, png = cv2.imencode(".png", img)
        if not ok:
            return []
        env = dict(os.environ, OMP_THREAD_LIMIT="1")  # we run several in parallel ourselves
        try:
            p = subprocess.run(
                [self.cmd, "stdin", "stdout", "-l", self.lang, "--psm", str(psm or self.psm),
                 *(extra or []), "tsv"],
                input=png.tobytes(), capture_output=True, timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired:
            # Some inputs (very noisy images) make Tesseract hang. Treat that as "no
            # text" instead of failing the whole photo.
            print(f"[ocr] tesseract timed out after {self.timeout:.0f} s on "
                  f"{img.shape[1]}x{img.shape[0]} image; skipped")
            return []
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode(errors="replace")[:500])
        toks: list[OcrToken] = []
        rows = csv.DictReader(io.StringIO(p.stdout.decode("utf-8", "replace")),
                              delimiter="\t", quoting=csv.QUOTE_NONE)
        for r in rows:
            try:
                if r["level"] != "5":
                    continue
                text = (r["text"] or "").strip()
                conf = float(r["conf"])
            except (KeyError, ValueError, TypeError):
                continue
            if not text or conf < 0:
                continue
            x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
            poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
            toks.append(OcrToken(text, conf / 100.0, poly,
                                 line_key=(r["block_num"], r["par_num"], r["line_num"])))
        return toks


# ---- PaddleOCR ----
class PaddleBackend(OcrBackend):
    name = "paddle"
    handles_rotation = True  # its angle classifier flips upside-down lines (0 vs 180)

    def __init__(self, lang: str = "en", **kw):
        from paddleocr import PaddleOCR  # type: ignore
        try:  # PaddleOCR 3.x
            self.engine = PaddleOCR(lang=lang, use_doc_orientation_classify=False,
                                    use_doc_unwarping=False,
                                    use_textline_orientation=True, **kw)
            self.v3 = True
        except TypeError:  # PaddleOCR 2.x
            self.engine = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False, **kw)
            self.v3 = False

    def read(self, img: np.ndarray) -> list[OcrToken]:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        toks: list[OcrToken] = []
        if self.v3:
            for res in self.engine.predict(img) or []:
                d = res.json.get("res", res) if hasattr(res, "json") else res
                for i, (t, s, poly) in enumerate(zip(d["rec_texts"], d["rec_scores"],
                                                     d["rec_polys"])):
                    toks.append(OcrToken(t, float(s), np.asarray(poly, float).reshape(-1, 2)[:4],
                                         line_key=("p", i)))
        else:
            out = self.engine.ocr(img, cls=True) or []
            for i, (poly, (t, s)) in enumerate((out[0] or []) if out else []):
                toks.append(OcrToken(t, float(s), np.asarray(poly, float), line_key=("p", i)))
        return toks


# ---- RapidOCR (PP-OCR models on ONNX Runtime) ----
class RapidOcrBackend(OcrBackend):
    name = "rapidocr"
    # The PP-OCR angle classifier flips upside-down lines by itself, so a 180
    # degree pass just repeats the 0 degree pass (and 270 repeats 90).
    # See ``effective_rotations``.
    handles_rotation = True

    # By default PP-OCR's detector scales images up so the short side is at least
    # 736 px. That turned a 120x500 px crop into 736x3070 and took about 1.9 s
    # (a page or tile took about 1.3 s). "Long side at most 2016 px" leaves pages
    # (at most 2000 px after preprocessing) and high-res tiles unchanged, and
    # stops small crops from being blown up.
    DET_LIMIT = {"det_limit_type": "max", "det_limit_side_len": 2016}

    def __init__(self, **kw):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self.new_api = False
        except ImportError:
            from rapidocr import RapidOCR  # type: ignore
            self.new_api = True
        if self.new_api:  # rapidocr 2+: settings go in a flat "Section.key" dict
            params = {"Det.limit_type": "max", "Det.limit_side_len": 2016}
            params.update(kw.pop("params", {}) or {})
            try:
                self.engine = RapidOCR(params=params, **kw)
            except TypeError:  # older 2.x versions without ``params`` (not tested)
                self.engine = RapidOCR(**kw)
        else:
            self.engine = RapidOCR(**{**self.DET_LIMIT, **kw})

    @staticmethod
    def _bgr(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    def read(self, img: np.ndarray) -> list[OcrToken]:
        img = self._bgr(img)
        toks: list[OcrToken] = []
        if self.new_api:
            out = self.engine(img)
            if out is None or getattr(out, "txts", None) is None:
                return []
            for i, (t, s, b) in enumerate(zip(out.txts, out.scores, out.boxes)):
                toks.append(OcrToken(t, float(s), np.asarray(b, float), line_key=("r", i)))
        else:
            res, _ = self.engine(img)
            # each item is [box, text, score] (plus word boxes when return_word_box is on)
            for i, item in enumerate(res or []):
                b, t, s = item[0], item[1], item[2]
                toks.append(OcrToken(t, float(s), np.asarray(b, float), line_key=("r", i)))
        return toks


# ---- Choosing an engine ----
_CACHE: dict[str, OcrBackend] = {}


def available_backends() -> list[str]:
    out = []
    for name, mod in (("paddle", "paddleocr"), ("rapidocr", "rapidocr_onnxruntime"),
                      ("rapidocr", "rapidocr")):
        try:
            __import__(mod)
            if name not in out:
                out.append(name)
        except Exception:
            pass
    if shutil.which("tesseract"):
        out.append("tesseract")
    return out


def get_backend(name: str = "auto", **kw) -> OcrBackend:
    """Create an OCR engine, or reuse the cached one. Loading models takes seconds."""
    if name == "auto":
        av = available_backends()
        if not av:
            raise RuntimeError("no OCR backend: install paddleocr, rapidocr_onnxruntime "
                               "or tesseract")
        name = av[0]
    key = name + repr(sorted(kw.items()))
    if key not in _CACHE:
        cls = {"paddle": PaddleBackend, "rapidocr": RapidOcrBackend,
               "tesseract": TesseractBackend}[name]
        _CACHE[key] = cls(**kw)
    return _CACHE[key]


# ---- Reading in several passes ----
def effective_rotations(be: OcrBackend, rotations: Sequence[int]) -> list[int]:
    """Return the rotations worth running on this engine.

    Engines with an angle classifier already read 180-degree text during
    the 0-degree pass, so 0/180 (and 90/270) count as one pass each. We keep
    the first of each pair that was asked for.
    """
    out: list[int] = []
    seen: set[int] = set()
    for r in rotations:
        cls = (r // 90) % 2 if be.handles_rotation else (r // 90) % 4
        if cls not in seen:
            seen.add(cls)
            out.append(r)
    return out


def rotations_covered(be: OcrBackend, rotations: Sequence[int]) -> set[int]:
    """Return every rotation that a set of passes has effectively covered."""
    out = {r % 360 for r in rotations}
    if be.handles_rotation:
        out |= {(r + 180) % 360 for r in out}
    return out


def _unrotate_poly(poly: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Map points from an image rotated k*90 degrees clockwise back to the original.

    w, h are the size of the ORIGINAL image.
    """
    x, y = poly[:, 0], poly[:, 1]
    k %= 4
    if k == 0:
        return poly
    if k == 1:  # rotated clockwise: (x', y') = (h-1-y, x)
        return np.stack([y, h - 1 - x], 1)
    if k == 2:
        return np.stack([w - 1 - x, h - 1 - y], 1)
    return np.stack([w - 1 - y, x], 1)  # k == 3 (counter-clockwise)


def read_text(img: np.ndarray, backend: OcrBackend | str = "auto",
              rotations: Sequence[int] = (0,), variants: Sequence[str] | None = None,
              tile: int | None = None, min_conf: float = 0.3,
              workers: int | None = None) -> list[OcrToken]:
    """Run OCR for every rotation and variant (and tile) and return all words in ``img``
    coordinates.

    rotations: degrees clockwise, any of 0/90/180/270. Boards and chips can
               lie on a desk any way round; deskewing only fixes small tilts.
    tile:      tile size in px, for tiny text like chip markings. None reads
               the whole image at once.
    """
    be = get_backend(backend) if isinstance(backend, str) else backend
    rotations = effective_rotations(be, rotations)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    variants = tuple(variants or be.default_variants)
    if isinstance(be, TesseractBackend):
        # Tesseract's sparse-text mode turns camera noise into thousands of fake
        # "characters" (55 s on one dark phone photo). A 3x3 median filter removes
        # the speckles but keeps letter strokes. Neural engines don't need this.
        from .enhance import noise_sigma
        if noise_sigma(gray[::2, ::2]) > TESS_NOISE_MAX:
            gray = cv2.medianBlur(gray, 3)

    jobs = []
    for rot in rotations:
        k = (rot // 90) % 4
        g = rotate90(gray, k)
        for vname, vimg in ocr_variants(g, variants):
            if tile:
                for x0, y0, t in iter_tiles(vimg, tile):
                    jobs.append((k, vname, x0, y0, t))
            else:
                jobs.append((k, vname, 0, 0, vimg))

    def run(job):
        k, vname, x0, y0, t = job
        toks = be.read(t)
        for tk in toks:
            tk.poly = tk.poly + np.array([x0, y0], float)
            tk.poly = _unrotate_poly(tk.poly, k, w, h)
            tk.rotation, tk.variant = k * 90, vname
            tk.line_key = (k, vname, x0, y0) + tuple(tk.line_key)
        return toks

    # Tesseract runs as separate processes, so it parallelises well. The neural engines
    # aren't thread-safe.
    n = workers if workers is not None else (min(8, os.cpu_count() or 2)
                                             if isinstance(be, TesseractBackend) else 1)
    out: list[OcrToken] = []
    if n > 1 and len(jobs) > 1:
        with cf.ThreadPoolExecutor(n) as ex:
            for toks in ex.map(run, jobs):
                out.extend(toks)
    else:
        for j in jobs:
            out.extend(run(j))
    return [t for t in out if t.conf >= min_conf and t.text.strip()]


def _reading_frame(t: OcrToken) -> tuple[float, float]:
    """Return the word's centre in its reading direction: u along the line, v across lines."""
    x, y = t.center
    return {0: (x, y), 90: (-y, x), 180: (-x, -y), 270: (y, -x)}[t.rotation]


def group_lines(tokens: Sequence[OcrToken], v_tol: float = 0.6,
                max_gap: float = 3.0) -> list[list[OcrToken]]:
    """Group words into lines based on where they are.

    The engines' own line IDs aren't enough: Tesseract's sparse mode
    (psm 11) puts every word in its own block. Words join a line when they
    were read at the same rotation and variant, sit at the same height
    (within v_tol letter heights) and are close together along the line
    (gap under max_gap letter heights). Each line is sorted in reading order.
    """
    groups: dict[tuple, list[OcrToken]] = {}
    for t in tokens:
        groups.setdefault((t.rotation, t.variant), []).append(t)
    out: list[list[OcrToken]] = []
    for toks in groups.values():
        items = sorted(((*_reading_frame(t), t) for t in toks), key=lambda a: (a[1], a[0]))
        lines: list[dict] = []
        for u, v, t in items:
            hgt = max(t.height, 1.0)
            half_w = max(np.ptp(t.poly[:, 0]), np.ptp(t.poly[:, 1])) / 2
            placed = False
            for ln in lines:
                if abs(v - ln["v"]) < v_tol * max(hgt, ln["h"]):
                    gap = max(ln["u0"] - (u + half_w), (u - half_w) - ln["u1"], 0)
                    if gap < max_gap * max(hgt, ln["h"]):
                        ln["toks"].append((u, t))
                        ln["u0"], ln["u1"] = min(ln["u0"], u - half_w), max(ln["u1"], u + half_w)
                        n = len(ln["toks"])
                        ln["v"] += (v - ln["v"]) / n
                        ln["h"] += (hgt - ln["h"]) / n
                        placed = True
                        break
            if not placed:
                lines.append({"v": v, "h": hgt, "u0": u - half_w, "u1": u + half_w,
                              "toks": [(u, t)]})
        for ln in sorted(lines, key=lambda l: l["v"]):
            out.append([t for _, t in sorted(ln["toks"], key=lambda a: a[0])])
    return out
