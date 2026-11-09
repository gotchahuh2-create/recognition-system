"""Faint chip marking reader: runs steps 1 to 8 from start to finish.

    region -> features -> cluster -> threshold -> (preserve) -> variants
    -> OCR on every variant -> fusion

Keeping OCR time down: on a 2-core PC, Tesseract takes about 0.5-1.5 s to
read one ~1000 px image, and the chip can be rotated any multiple of 90
degrees. With ``orient="geometry"`` (the default) we work out which way the
text lines run from the masks (``preserve.text_axis``) and read each
variant only at the two matching rotations (0/180 or 90/270; OCR and fusion
sort out which of the two is upright). If the direction isn't clear, all
four rotations are read. ``"ocr"`` reads the gray variants at all four
rotations first, then the rest at the best ones. ``"all"`` reads everything
at all four.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..ocr import OcrBackend, OcrToken, get_backend, read_text
from . import cluster as C
from . import features as F
from . import fusion as U
from . import region as R
from . import threshold as T
from . import variants as V
from .debug import StageSaver

ROTATIONS = (0, 90, 180, 270)
_PARTISH = re.compile(r"[A-Z0-9]{4,}")


@dataclass
class FaintConfig:
    k: int = 4  # k-means groups per colour space
    n_best: int = 3  # how many thresholded masks go to OCR (best stroke scores)
    soft: bool = True  # include the gray variants
    target_glyph: float = 40.0  # letter height of the OCR images
    orient: str = "geometry"  # "geometry": text direction from the masks -> 2 rotations;
                                     # "ocr": gray variants at 4 rotations, the rest at the
                                     # best;
                                     # "all": every variant at 4 rotations
    denoise: bool = True
    bg_glyphs: float = 1.6  # background window, in letter heights
    bg_mode: str = "median"  # "median" (follows glare) or "gauss"
    min_conf: float = 0.0  # OCR confidence on chip reads isn't reliable, so don't filter by it
    ocr_mode: str = "crop"           # "crop": backend.read_crop (Tesseract: block mode, part-number
                                     # characters only, no dictionaries); "page": read_text
                                     # (sparse text)


@dataclass
class FaintReading:
    tokens: list[OcrToken]  # raw words of all variants + combined lines, in chip px
    fused: list[U.FusedLine]
    reliability: dict[str, float]
    chip: R.ChipCrop
    per_variant: dict[str, list[OcrToken]] = field(default_factory=dict)
    stroke_scores: dict[str, float] = field(default_factory=dict)
    rotations: list[int] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"chip": self.chip.as_dict(), "rotations": self.rotations,
                "fused": [f.as_dict() for f in self.fused[:8]],
                "variants": {v: " | ".join(l.text for l in U.lines_of(v, t))[:120]
                             for v, t in self.per_variant.items()},
                "reliability": self.reliability, "timings_ms": self.timings_ms,
                "top_masks": dict(sorted(self.stroke_scores.items(), key=lambda kv: -kv[1])[:6])}


def _rot_score(toks: list[OcrToken]) -> dict[int, float]:
    s = {r: 0.0 for r in ROTATIONS}
    for t in toks:
        m = _PARTISH.findall(t.text.upper())
        if m:
            s[t.rotation] += (0.3 + t.conf) * sum(len(x) for x in m)
    return s


def read_crop_rotations(img: np.ndarray, be: OcrBackend, rots) -> list[OcrToken]:
    """Run ``backend.read_crop`` at each rotation and map the words back to ``img``.

    Same bookkeeping as ``ocr.read_text``. Rotations run in parallel threads,
    since Tesseract runs as a separate process.
    """
    import concurrent.futures as cf
    import os
    from ..ocr import TesseractBackend, _unrotate_poly
    from ..preprocess import rotate90
    h, w = img.shape[:2]

    def one(rot):
        k = (rot // 90) % 4
        toks = be.read_crop(rotate90(img, k))
        for i, t in enumerate(toks):
            t.poly = _unrotate_poly(t.poly, k, w, h)
            t.rotation, t.variant = k * 90, "gray"
            t.line_key = (k,) + tuple(t.line_key)
        return toks
    n = min(len(rots), os.cpu_count() or 2) if isinstance(be, TesseractBackend) else 1
    out: list[OcrToken] = []
    if n > 1:
        with cf.ThreadPoolExecutor(n) as ex:
            for ts in ex.map(one, rots):
                out.extend(ts)
    else:
        for r in rots:
            out.extend(one(r))
    return [t for t in out if t.text.strip()]


def masks_for_axis(cr, chip, top_masks):
    """Return (axis, h, v) for the masks most likely to contain the text."""
    from .preserve import gentle_clean, text_axis
    g = chip.glyph_px
    cands = [gentle_clean(cr.votes >= 2, chip.body_mask, g), cr.score > 2.5] + list(top_masks)
    return [text_axis(m, chip.body_mask, g) for m in cands]


class FaintChipReader:
    def __init__(self, backend: OcrBackend | str = "auto", cfg: FaintConfig | None = None):
        self.be = get_backend(backend) if isinstance(backend, str) else backend
        self.cfg = cfg or FaintConfig()

    def preprocess(self, chip: R.ChipCrop, saver: StageSaver | None = None):
        cfg, sv = self.cfg, saver or StageSaver()
        t = {}
        t0 = time.perf_counter()
        pf = F.compute(chip.image, chip.body_mask, chip.glyph_px, cfg.denoise, cfg.bg_glyphs,
                       cfg.bg_mode, sv)
        t["features"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cr = C.cluster(pf, chip.body_mask, cfg.k, sv)
        t["cluster"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        masks, scores = T.threshold_all(cr.score, pf.smooth["lab"][:, :, 0], cr.polarity,
                                        chip.body_mask, pf.glyph_px, sv)
        t["threshold"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        vs = V.build(pf, cr, masks, scores, chip.body_mask, cfg.n_best, cfg.target_glyph,
                     cfg.soft, sv)
        t["variants"] = time.perf_counter() - t0
        return vs, scores, {k: round(v * 1000) for k, v in t.items()}, cr, masks

    def _ocr(self, v: V.Variant, rots) -> list[OcrToken]:
        if self.cfg.ocr_mode == "crop":
            toks = read_crop_rotations(v.image, self.be, rots)
        else:
            toks = read_text(v.image, self.be, rotations=rots, variants=("gray",),
                             min_conf=self.cfg.min_conf)
        for tk in toks:
            tk.poly = v.to_chip(tk.poly)
            tk.variant = f"chip-faint-{v.name}"
        return toks

    def read_chip(self, chip: R.ChipCrop, saver: StageSaver | None = None) -> FaintReading:
        sv = saver or StageSaver()
        if sv.enabled and not any("1b_rectified" in n for n, _ in sv.saved):
            # the chip was found elsewhere (by the card step), but still save the step 1
            # image
            sv.save(f"1b_rectified_chip_{chip.method}", chip.image)
            bm = chip.image.copy()
            bm[~chip.body_mask] //= 4
            sv.save("1c_body_mask", bm)
        t_all = time.perf_counter()
        vs, scores, tm, cr, masks = self.preprocess(chip, sv)
        t0 = time.perf_counter()
        per: dict[str, list[OcrToken]] = {}
        rots = list(ROTATIONS)
        axis = None
        if self.cfg.orient == "geometry":
            top = [masks[n] for n in sorted(masks, key=lambda n: -scores[n])[:2]]
            best = max(masks_for_axis(cr, chip, top), key=lambda am: am[1] + am[2])
            axis = best[0]
            if axis == "h":
                rots = [0, 180]
            elif axis == "v":
                rots = [90, 270]
        probe_v = [v for v in vs if v.name in ("fis-soft", "L-flat")] if self.cfg.orient == "ocr" else []
        for v in probe_v:
            per[v.name] = self._ocr(v, ROTATIONS)
        if probe_v:
            rs = _rot_score([t for v in probe_v for t in per[v.name]])
            best = max(rs.values())
            if best > 0:
                rots = [r for r, s in sorted(rs.items(), key=lambda kv: -kv[1]) if s >= 0.5 * best][:2]
        for v in vs:
            if v.name not in per:
                per[v.name] = self._ocr(v, rots)
        tm["ocr"] = round((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        fused, rel = U.fuse(per)
        tm["fusion"] = round((time.perf_counter() - t0) * 1000)
        tm["total"] = round((time.perf_counter() - t_all) * 1000)
        toks = [t for ts in per.values() for t in ts] + U.fused_tokens(fused)
        if sv.enabled:
            self._draw(chip, fused, sv)
            sv.sheet()
        return FaintReading(toks, fused, rel, chip, per, scores, rots, tm,
                            {"polarity": cr.polarity, "axis": axis, **{k: v for k, v in cr.info.items() if k != "fisher"}})

    def read(self, img: np.ndarray, quad: np.ndarray | None = None,
             saver: StageSaver | None = None) -> FaintReading:
        """Find and read a chip in any image (a close-up or a card crop)."""
        sv = saver or StageSaver()
        chip = R.extract_chip(img, quad, saver=sv)
        return self.read_chip(chip, sv)

    @staticmethod
    def _draw(chip: R.ChipCrop, fused: list[U.FusedLine], sv: StageSaver) -> None:
        import cv2
        vis = chip.image.copy()
        for f in fused[:6]:
            p = np.round(f.poly).astype(np.int32)
            cv2.polylines(vis, [p], True, (0, 255, 0), 2)
            cv2.putText(vis, f"{f.text} ({f.support:.2f})", (int(p[0, 0]), max(14, int(p[0, 1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        sv.save("8_fused_result", vis)


def read_faint(img: np.ndarray, backend: OcrBackend | str = "auto", quad=None,
               debug_dir=None, prefix: str = "faint", cfg: FaintConfig | None = None) -> FaintReading:
    return FaintChipReader(backend, cfg).read(img, quad, StageSaver(debug_dir, prefix))
