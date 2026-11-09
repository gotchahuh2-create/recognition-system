"""Recognise a Click board: crop -> DINOv2 descriptor -> gallery search -> re-rank -> accept /
ask / unknown.

Re-ranking steps, in the order they were added (and measured):
  1. Extra views: the query, a mirrored copy and 2 small crops, averaged.
  2. OCR bonus (``rerank``): a name match of 90+ moves a Click to first
     place, 75-89 adds +0.10.
  3. Board size: candidates whose S/M/L size doesn't match the crop's shape
     lose ``size_penalty`` (only when a tight box was given).
  4. Local feature check for hard cases (off by default): AKAZE feature
     matches + RANSAC between the query and the top candidates' images.
     (OpenCV's AKAZE is used instead of LightGlue+ALIKED/DISK: no torch
     needed, BSD/Apache licensed.)

When to give up. The thresholds are placeholders until ``eval/click_eval.py``
tunes them on the validation set and writes ``thresholds.json`` next to the
gallery:

    s1 >= T_accept and (margin >= M, or OCR confirmed #1)   -> accept
         (s = cosine similarity; margin = gap between #1 and #2 after re-ranking)
    s1 >= T_unknown, small margin or look-alike boards      -> ask (top 3)
    s1 <  T_unknown                                         -> unknown
         (an OCR match against the whole catalog may still offer a candidate -> ask)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..catalog import Catalog
from ..ocr import OcrBackend, get_backend
from ..results import Candidate, Recognition
from .embedder import Embedder, get_embedder, l2n
from .gallery import Gallery
from .preprocess import crop_box, size_class
from .rerank import ClickOcr, match_name, ocr_click, series_base

OUTCOMES = ("accept", "ask", "unknown")


@dataclass
class ClickConfig:
    k: int = 50  # how many vectors to search
    top: int = 5  # how many PIDs to keep for re-ranking
    t_accept: float = 0.70  # placeholder values; tune them with eval/click_eval.py
    t_unknown: float = 0.45
    margin: float = 0.05
    tta: bool = True
    ocr: bool = True
    ocr_promote: float = 90.0
    ocr_bonus_min: float = 75.0
    ocr_bonus: float = 0.10
    size_prior: bool = True
    size_penalty: float = 0.05
    verify: bool = False
    verify_min_inliers: int = 15
    box_expand: float = 0.08        # grow user/detector boxes by 5-10 %

    @classmethod
    def from_file(cls, path: str | Path, **over) -> "ClickConfig":
        d = {}
        p = Path(path)
        if p.is_file():
            d = {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                 if k in cls.__dataclass_fields__}
        d.update({k: v for k, v in over.items() if v is not None})
        return cls(**d)


@dataclass
class ClickResult:
    outcome: str  # accept, ask or unknown
    recognition: Recognition
    box: list[float] | None              # x, y, w, h in the original photo
    ocr: ClickOcr | None = None
    slot: str | None = None
    timings_ms: dict = field(default_factory=dict)
    raw: list = field(default_factory=list, repr=False)  # ranked candidates, before the decision

    def manifest(self) -> dict:
        d = self.recognition.decision
        top = self.recognition.candidates
        return {"pid": d.pid if d else None, "name": d.name if d else None,
                "slot": self.slot, "confidence": round(d.score, 3) if d else None,
                "outcome": self.outcome, "box": self.box,
                "options": [c.as_dict() for c in top] if self.outcome != "accept" else []}

    def as_dict(self) -> dict:
        r = self.recognition.as_dict()
        r.update({"outcome": self.outcome, "box": self.box, "slot": self.slot,
                  "ocr": asdict(self.ocr) if self.ocr else None,
                  "timings_ms": self.timings_ms})
        return r


def tta_views(crop: np.ndarray) -> list[np.ndarray]:
    """Return the query, a mirrored copy, and two ~90 % crops (top-left and bottom-right)."""
    h, w = crop.shape[:2]
    ch, cw = max(int(h * 0.9), 1), max(int(w * 0.9), 1)
    return [crop, crop[:, ::-1].copy(), crop[:ch, :cw], crop[h - ch:, w - cw:]]


class _Verifier:
    """Count matching AKAZE features (after RANSAC) between the query and reference images."""

    def __init__(self):
        self.det = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._cache: dict[str, tuple] = {}

    def _feat(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        f = 640 / max(g.shape[:2])
        if f < 1:
            g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return self.det.detectAndCompute(g, None)

    def ref(self, path: str):
        if path not in self._cache:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) \
                if path and Path(path).is_file() else None
            self._cache[path] = self._feat(img) if img is not None else (None, None)
        return self._cache[path]

    def inliers(self, q, ref) -> int:
        (kq, dq), (kr, dr) = q, ref
        if dq is None or dr is None or len(kq) < 8 or len(kr) < 8:
            return 0
        good = [m for m, n in (p for p in self.bf.knnMatch(dq, dr, k=2) if len(p) == 2)
                if m.distance < 0.8 * n.distance]
        if len(good) < 8:
            return 0
        a = np.float32([kq[m.queryIdx].pt for m in good])
        b = np.float32([kr[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0


class ClickRecognizer:
    def __init__(self, embedder: Embedder | str, gallery: Gallery | str,
                 backend: OcrBackend | str | None = "auto", catalog: Catalog | None = None,
                 cfg: ClickConfig | None = None):
        self.embedder = get_embedder(embedder) if isinstance(embedder, (str, Path)) else embedder
        if isinstance(gallery, (str, Path)):
            gdir = Path(gallery)
            gallery = Gallery.load(gdir)
            if cfg is None:
                cfg = ClickConfig.from_file(gdir / "thresholds.json")
        gallery.check(self.embedder.model_version, self.embedder.preprocess_version)
        self.gallery = gallery
        self.cfg = cfg or ClickConfig()
        self.catalog = catalog or Catalog.load()
        self.names = {**gallery.names, **self.catalog.click_names()}
        self.sizes = {c.pid: c.size for c in self.catalog.clicks if c.size}
        self.groups = {c.pid: c.group for c in self.catalog.clicks if c.group}
        self.backend = None
        if backend is not None and self.cfg.ocr:
            try:
                self.backend = get_backend(backend) if isinstance(backend, str) else backend
            except Exception as e:  # OCR only helps re-rank, it isn't required
                print(f"[clicks] OCR re-rank disabled: {e}")
        self._verifier = _Verifier() if self.cfg.verify else None

    def name(self, pid: str) -> str:
        return self.names.get(pid, pid)

    # ---- Descriptors ----
    def describe(self, crop: np.ndarray, tta: bool | None = None) -> np.ndarray:
        """Compute the query descriptor. With extra views, it's the average of 4 views,
        re-normalised.
        """
        views = tta_views(crop) if (self.cfg.tta if tta is None else tta) else [crop]
        return l2n(self.embedder.embed(views).mean(0))

    # ---- Main entry point ----
    def recognise(self, image: np.ndarray, box: Sequence[float] | None = None,
                  ocr: bool | None = None, q: np.ndarray | None = None,
                  ocr_res: ClickOcr | None = None, gallery: Gallery | None = None,
                  cfg: ClickConfig | None = None) -> ClickResult:
        """Recognise one Click.

        ``box`` is (x, y, w, h) in ``image`` pixels, or None when the whole photo
        shows one Click.

        ``q`` / ``ocr_res`` reuse a descriptor / OCR result computed earlier, and
        ``gallery`` / ``cfg`` override the defaults (used in evaluation: one
        descriptor tested against many galleries and thresholds).
        """
        cfg = cfg or self.cfg
        gallery = gallery or self.gallery
        t0 = time.perf_counter()
        crop = crop_box(image, box, cfg.box_expand) if box is not None else image
        if q is None:
            q = self.describe(crop, cfg.tta)
        t1 = time.perf_counter()
        hits = gallery.search(q[None], cfg.k, cfg.top)
        t2 = time.perf_counter()

        cands = []
        for pid, cos, row in hits:
            cands.append({"pid": pid, "cos": cos, "row": row, "rank": cos,
                          "ocr_ratio": None, "bonus": 0.0, "penalty": 0.0,
                          "number_ok": None})

        # step 2: OCR re-rank
        use_ocr = (cfg.ocr if ocr is None else ocr) and (self.backend is not None
                                                         or ocr_res is not None)
        if not use_ocr:
            ocr_res = None
        elif cands:
            if ocr_res is None:
                ocr_res = ocr_click(crop, self.backend)
            if ocr_res.norm:
                for c in cands:
                    m = match_name(ocr_res.norm, self.name(c["pid"]))
                    c["ocr_ratio"], c["number_ok"] = round(m.ratio, 1), m.number_ok
                    if m.ratio >= cfg.ocr_promote:
                        c["bonus"] = 1.0
                    elif m.ratio >= cfg.ocr_bonus_min:
                        c["bonus"] = cfg.ocr_bonus
        t3 = time.perf_counter()

        # step 3: board size (only meaningful for a tight box)
        crop_size = size_class(box[2], box[3]) if box is not None else None
        if cfg.size_prior and crop_size:
            for c in cands:
                s = self.sizes.get(c["pid"])
                if s and s != crop_size:
                    c["penalty"] = cfg.size_penalty

        for c in cands:
            c["rank"] = c["cos"] + c["bonus"] - c["penalty"]
        cands.sort(key=lambda c: -c["rank"])

        # step 4: local feature check for hard cases
        inl_note = None
        if cfg.verify and self._verifier is None:
            self._verifier = _Verifier()
        if cfg.verify and len(cands) > 1 and \
                cands[0]["cos"] - cands[1]["cos"] < cfg.margin:
            qf = self._verifier._feat(crop)
            for c in cands[:3]:
                path = gallery.meta[c["row"]]["path"]
                c["inliers"] = self._verifier.inliers(qf, self._verifier.ref(path))
            top3 = sorted(cands[:3], key=lambda c: -c.get("inliers", 0))
            if top3[0].get("inliers", 0) >= cfg.verify_min_inliers and \
                    top3[0].get("inliers", 0) >= 1.5 * max(top3[1].get("inliers", 0), 1) and \
                    top3[0] is not cands[0] and cands[0]["bonus"] < 1.0:
                cands = top3 + cands[3:]
                inl_note = f"local features: {top3[0]['inliers']} inliers"
        t4 = time.perf_counter()

        ranked = [dict(c) for c in cands]
        outcome, reason, cands = self._decide(cands, ocr_res, cfg)
        if inl_note:
            reason += f" ({inl_note})"
        top = [Candidate(c["pid"], self.name(c["pid"]), float(c["cos"]), c["pid"],
                         {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in c.items() if k not in ("pid", "row") and v is not None})
               for c in cands[:3]]
        decision = top[0] if outcome == "accept" and top else None
        timings = {"embed": round((t1 - t0) * 1000), "search": round((t2 - t1) * 1000, 1),
                   "ocr": round((t3 - t2) * 1000), "verify": round((t4 - t3) * 1000),
                   "total": round((time.perf_counter() - t0) * 1000)}
        rec = Recognition("click", top, decision, reason,
                          evidence=[{"read": ocr_res.text, "conf": round(ocr_res.conf, 3)}]
                          if ocr_res and ocr_res.text else [],
                          timings_ms=timings)
        return ClickResult(outcome, rec, [float(v) for v in box] if box is not None else None,
                           ocr_res, timings_ms=timings, raw=ranked)

    def redecide(self, res: ClickResult, cfg: ClickConfig) -> tuple[str, str | None]:
        """Re-decide an earlier result with different thresholds. Returns (outcome, top PID).

        The ranking doesn't depend on the thresholds, so this is cheap. Used
        when tuning thresholds.
        """
        outcome, _, cands = self._decide([dict(c) for c in res.raw], res.ocr, cfg)
        return outcome, (cands[0]["pid"] if cands else None)

    # ---- When to give up ----
    def _decide(self, cands: list[dict], ocr_res: ClickOcr | None,
                cfg: ClickConfig | None = None):
        cfg = cfg or self.cfg
        if not cands:
            return "unknown", "gallery is empty", cands
        c1 = cands[0]
        s1 = c1["cos"]
        # Use the margin after re-ranking: OCR or size evidence for the runner-up
        # must shrink the margin, not just the raw cosine gap.
        gap = c1["rank"] - cands[1]["rank"] if len(cands) > 1 else s1
        strong = [c for c in cands if c["bonus"] >= 1.0]
        ocr_ok = c1["bonus"] >= 1.0 and len(strong) == 1

        if s1 < cfg.t_unknown and not ocr_ok:
            fb = self._catalogue_fallback(ocr_res, {c["pid"] for c in cands}, cfg)
            if fb:
                return "ask", (f"retrieval {s1:.2f} < {cfg.t_unknown:.2f}; OCR read "
                               f"'{ocr_res.text}' ~ {self.name(fb['pid'])}"), [fb] + cands
            return "unknown", (f"unknown Click: best {self.name(c1['pid'])} {s1:.2f} < "
                               f"{cfg.t_unknown:.2f} - take a closer photo or pick it"), cands

        same_group = len(cands) > 1 and (
            (self.groups.get(c1["pid"]) and
             self.groups.get(c1["pid"]) == self.groups.get(cands[1]["pid"])) or
            series_base(self.name(c1["pid"])) == series_base(self.name(cands[1]["pid"])))
        if s1 >= cfg.t_accept and ocr_ok:
            return "accept", (f"{self.name(c1['pid'])} {s1:.2f}, OCR confirmed "
                              f"({c1['ocr_ratio']:.0f})"), cands
        if s1 >= cfg.t_accept and gap >= cfg.margin and not same_group:
            return "accept", (f"{self.name(c1['pid'])} {s1:.2f} "
                              f"(margin {gap:.2f})"), cands
        if same_group and s1 >= cfg.t_accept:
            why = f"near-identical: {self.name(c1['pid'])} vs {self.name(cands[1]['pid'])}"
        elif s1 < cfg.t_accept:
            why = f"best {self.name(c1['pid'])} {s1:.2f} < {cfg.t_accept:.2f}"
        else:
            why = (f"ambiguous: {self.name(c1['pid'])} {c1['rank']:.2f} vs "
                   f"{self.name(cands[1]['pid'])} {cands[1]['rank']:.2f}")
        return "ask", why + " - pick from the top 3", cands

    def _catalogue_fallback(self, ocr_res: ClickOcr | None, exclude: set,
                            cfg: ClickConfig) -> dict | None:
        """All search scores are too low: match OCR text against every known name.

        The result is never accepted automatically.
        """
        if ocr_res is None or not ocr_res.norm:
            return None
        best, best_r = None, 0.0
        for pid, name in self.names.items():
            if pid in exclude:
                continue
            r = match_name(ocr_res.norm, name).ratio
            if r > best_r:
                best, best_r = pid, r
        if best is None or best_r < cfg.ocr_promote:
            return None
        return {"pid": best, "cos": 0.0, "row": -1, "rank": 0.0, "ocr_ratio": best_r,
                "bonus": 1.0, "penalty": 0.0, "number_ok": None, "from": "ocr-catalogue"}


# ---- Slot assignment (simple rule: nearest mikroBUS socket on the board) ----
def assign_slots(centres_uv: Sequence[tuple[float, float]], sockets) -> list[str | None]:
    """Assign each Click to the nearest free socket, one Click per socket.

    Click centres are board-relative (u, v from 0 to 1). Returns None for
    each Click when the board has no socket map.
    """
    if not sockets:
        return [None] * len(centres_uv)
    pairs = sorted((float(np.hypot(u - s.u, v - s.v)), i, j)
                   for i, (u, v) in enumerate(centres_uv) for j, s in enumerate(sockets))
    out: list[str | None] = [None] * len(centres_uv)
    used = set()
    for _, i, j in pairs:
        if out[i] is None and j not in used:
            out[i], _ = sockets[j].slot, used.add(j)
    return out
