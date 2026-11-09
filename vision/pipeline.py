"""The full recognition pipeline: photo -> clean-up -> read the board -> read the MCU.

The result uses the setup manifest field names:
{board, mcu, clicks: [{pid, name, slot, confidence}]}.

How it works, in order:

1. Board. We read the text on the photo and match it to a known board.

2. MCU card. Once we know the board, we look for the MCU card on the photo
   (by its yellow frame, by where the board's layout says it should be, or
   by its printed title). We cut it out of the original photo, straighten
   it, zoom in and read all of its text: the title, the port labels, the
   mikroBUS pin labels, the chip marking (from four angles) and any PID
   sticker. All of that is matched against the card dataset.

3. Layout fallback. If the card can't be found or read, we use the board's
   stored layout (``data/layouts.json``, see ``vision.layout``) to find
   where the MCU card should be, and read just that area at full
   resolution. If the board has no layout, we read the whole photo.

4. Clicks (optional). If a ``ClickRecognizer`` is given, each Click board
   is recognised with DINOv2 image search (``vision.clicks``). That is one
   crop per box the user drew, or the whole photo as one Click
   (``click_mode="photo"``). The layout also gives each mikroBUS socket its
   slot name, but only when the photo was straightened to the board outline
   (the user clicked its 4 corners).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .board_ocr import BoardRecognizer
from .card_locate import CardCrop, CardLocator, crop_card
from .card_reader import CardReading, read_card
from .catalog import DATA_DIR, Catalog
from . import layout as L
from . import package
from .loader import Photo
from .mcu_ocr import McuRecognizer
from .ocr import get_backend, read_text, rotations_covered
from .preprocess import Preprocessed, PreprocessConfig, assess_quality, preprocess
from pathlib import Path
from .refine import RefineConfig
from .results import Recognition


@dataclass
class PhotoResult:
    photo: str
    quality: dict
    preprocess_steps: list[str]
    board: Recognition
    mcu: Recognition
    mcu_card: Recognition
    timings_ms: dict = field(default_factory=dict)
    pre: Preprocessed | None = field(default=None, repr=False)  # kept for drawing debug images
    clicks: list = field(default_factory=list)  # list of clicks.ClickResult
    placement: "L.Placement | None" = None  # where the board layout sits on the photo
    rois: list = field(default_factory=list, repr=False)  # list of layout.RoiCrop
    mcu_source: str = "photo"  # where the MCU was read from: "card", "roi" or "photo"
    # the card we found, cut out and zoomed
    card_crop: CardCrop | None = field(default=None, repr=False)
    card_reading: CardReading | None = field(default=None, repr=False)
    card_locations: list = field(default_factory=list, repr=False)

    def manifest(self) -> dict:
        """Build the setup manifest (board and MCU only).

        Anything we were not sure about is left as null and added to
        ``needs_user`` with the top 3 options for the user to pick from.
        """
        def pick(r: Recognition):
            if r.decision is None:
                return None
            return {"key": r.decision.key, "name": r.decision.name,
                    "pid": r.decision.pid, "confidence": round(r.decision.score, 3)}
        needs = [{"field": r.kind, "reason": r.reason,
                  "options": [c.as_dict() for c in r.candidates]}
                 for r in (self.board, self.mcu) if r.abstained]
        # Only ask the user about the card if we at least read its layout. Then the
        # options are the few cards that share that layout, not the whole catalog.
        if self.mcu_card.abstained and self.mcu_card.candidates:
            needs.append({"field": "mcu_card", "reason": self.mcu_card.reason,
                          "options": [c.as_dict() for c in self.mcu_card.candidates]})
        clicks = [c.manifest() for c in self.clicks]
        needs += [{"field": f"clicks[{i}]", "reason": c.recognition.reason,
                   "outcome": c.outcome,
                   "options": [o.as_dict() for o in c.recognition.candidates]}
                  for i, c in enumerate(self.clicks) if c.outcome != "accept"]
        return {"board": pick(self.board), "mcu": pick(self.mcu),
                "mcu_card": pick(self.mcu_card),
                "mcu_card_layout": ({k: self.mcu_card.group[k] for k in ("key", "title", "cards")}
                                    if self.mcu_card.group else None),
                "clicks": clicks, "needs_user": needs}

    def as_dict(self) -> dict:
        return {"photo": self.photo, "manifest": self.manifest(),
                "quality": self.quality, "preprocess": self.preprocess_steps,
                "layout": ({**self.placement.as_dict(), "mcu_source": self.mcu_source,
                            "rois": [r.as_dict() for r in self.rois]}
                           if self.placement else None),
                "card": self.card_dict(),
                "board": self.board.as_dict(), "mcu": self.mcu.as_dict(),
                "mcu_card": self.mcu_card.as_dict(),
                "clicks": [c.as_dict() for c in self.clicks], "timings_ms": self.timings_ms}


    def card_dict(self) -> dict | None:
        """Describe where the card was found, how much it was zoomed and what we read on it."""
        if self.card_crop is None and not self.card_locations:
            return None
        return {"located": self.card_crop.as_dict() if self.card_crop else None,
                "other_locations": [l.as_dict() for l in self.card_locations[1:3]],
                "read": self.card_reading.as_dict() if self.card_reading else None,
                "used": self.mcu_source == "card"}


class Pipeline:
    def __init__(self, catalog: Catalog | str | None = None, backend: str = "auto",
                 cfg: PreprocessConfig | None = None, mcu: bool = True,
                 refine: str = "auto", refine_cfg: RefineConfig | None = None,
                 clicks=None, click_mode: str = "boxes",
                 layouts: "L.Layouts | str | Path | bool | None" = None,
                 roi_long_side: int = 1600, roi_fallback: bool = False,
                 card_stage: bool = True, card_zoom_target: int = 1800,
                 card_max_zoom: float = 4.0, upscaler: str = "bicubic",
                 chip_mode: str = "classic", faint_debug: str | Path | None = None,
                 verify_marking: bool = False):
        """Settings:

        clicks: a ``vision.clicks.ClickRecognizer``, or None to skip Clicks.
        click_mode: "boxes" (only boxes the user drew), "photo" (if there are no
            boxes, treat the whole photo as one Click) or "off".
        layouts: board layouts used to find the MCU card. None loads
            ``layouts.json`` from the catalog folder, False turns this off and the
            whole photo is read instead.
        roi_long_side: the MCU card crop is resized to this many pixels before OCR.
        roi_fallback: if the card crop gives no answer, also read the whole photo.
            Slower, and usually pointless because the crop is where the MCU is.
        card_stage: find the MCU card, zoom in and read all its text. False skips
            this step.
        card_zoom_target / card_max_zoom: the card crop is enlarged until its long
            side is card_zoom_target px, but by no more than card_max_zoom times.
        chip_mode: how the chip text is cleaned up before OCR. "classic" is the
            standard method, "faint" is the experimental ``vision.faint`` method for
            hard-to-see markings, "both" tries both.
        faint_debug: folder where the faint method saves an image for each step.
        verify_marking: if we know the layout group but couldn't read the chip,
            simulate how each possible chip marking would look at this photo's
            resolution and compare (``vision.faint.synth``). Only decides when one
            option is clearly better.
        """
        self.catalog = catalog if isinstance(catalog, Catalog) else Catalog.load(catalog)
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.cfg = cfg or PreprocessConfig()
        self.boards = BoardRecognizer(self.catalog, self.backend)
        self.mcus = (McuRecognizer(self.catalog, self.backend, refine=refine,
                                   refine_cfg=refine_cfg) if mcu else None)
        self.clicks = clicks
        self.click_mode = click_mode
        if layouts is False:
            self.layouts = L.Layouts()
        elif isinstance(layouts, L.Layouts):
            self.layouts = layouts
        else:
            if layouts is None:
                src = Path(catalog) if isinstance(catalog, (str, Path)) else None
                layouts = (src / "layouts.json" if src is not None and src.is_dir()
                           else DATA_DIR / "layouts.json")
            self.layouts = L.Layouts.load(layouts)
        self.roi_long_side = roi_long_side
        self.roi_fallback = roi_fallback
        self.card_stage = card_stage and mcu
        self.card_zoom_target, self.card_max_zoom = card_zoom_target, card_max_zoom
        self.upscaler = upscaler
        self.chip_mode, self.faint_debug = chip_mode, faint_debug
        self.verify_marking = verify_marking and mcu
        self.locator = CardLocator()

    # ---- Finding the board by its layout ----
    # If the board name couldn't be read, we can still recognise the board when
    # at least this many of its labels are found in consistent positions.
    LAYOUT_ID_MIN_INLIERS = 8
    LAYOUT_ID_MIN_SHARE = 0.5
    # If fewer labels than this matched, read the page at the other rotations too and try
    # again.
    LAYOUT_STRONG_INLIERS = 6
    LAYOUT_PASS_LONG_SIDE = 1400

    def board_from_layouts(self, pre: Preprocessed, board: Recognition, tokens
                           ) -> Recognition:
        """Identify the board from its layout when its name couldn't be read.

        This happens, for example, when the board's title is upside down and
        OCR didn't flip it. If many of the board's known labels are found on the
        photo in positions that agree with each other (checked with RANSAC),
        that is strong evidence on its own.
        """
        if board.decision is not None or not self.layouts:
            return board
        fits = []
        for key, lay in self.layouts.items():
            pl = L.locate(lay, tokens, pre)
            if pl is not None and pl.method.startswith("anchors"):
                fits.append((pl.inliers, pl.inliers / max(pl.matched, 1), key, pl))
        fits.sort(key=lambda f: -f[0])
        if not fits:
            return board
        n, share, key, pl = fits[0]
        runner = fits[1][0] if len(fits) > 1 else 0
        if n < self.LAYOUT_ID_MIN_INLIERS or share < self.LAYOUT_ID_MIN_SHARE or runner * 2 > n:
            return board
        b = next((x for x in self.catalog.boards if x.key == key), None)
        if b is None:
            return board
        from .results import Candidate
        cand = Candidate(b.key, b.name, min(0.99, 0.7 + 0.02 * n), b.pid,
                         {"from_layout": pl.as_dict()})
        cands = [cand] + [c for c in board.candidates if c.key != b.key][:2]
        return Recognition("board", cands, cand,
                           f"{b.name} identified by layout: {n}/{pl.matched} board labels "
                           f"consistent ({board.reason})", board.evidence, board.raw_reads,
                           board.timings_ms, rotations_read=board.rotations_read)

    def place_layout(self, photo: Photo, pre: Preprocessed, board: Recognition,
                     tokens, quad=None):
        """Place the identified board's layout on the photo. Returns None if it doesn't fit."""
        lay = self.layouts.get(board.decision.key) if board.decision else None
        if lay is None:
            return None, None
        quad_pre = None
        if quad is not None:  # the photo was already cropped to the board (user clicked 4 corners)
            h, w = pre.image.shape[:2]
            quad_pre = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
        return lay, L.locate(lay, tokens, pre, quad_pre=quad_pre,
                             photo=photo.image if quad is None else None)

    def mcu_from_roi(self, photo: Photo, pre: Preprocessed, roi: "L.RoiCrop",
                     board_key: str | None = None):
        """Run the MCU recogniser on the MCU card area only.

        Boxes in the result are converted back to positions on the processed photo.
        """
        rpre = preprocess(roi.image, PreprocessConfig(
            long_side=self.roi_long_side, max_upscale=3.0, deskew=True,
            reduce_glare=self.cfg.reduce_glare, clahe_clip=self.cfg.clahe_clip))
        mcu, card = self.mcus.recognise(rpre, None, hires=roi.image, board_key=board_key)
        # chain of transforms: card crop (processed) -> card crop -> original photo ->
        # processed photo
        self._map_boxes(pre.M @ np.linalg.inv(roi.H) @ np.linalg.inv(rpre.M), mcu, card)
        mcu.timings_ms["source"] = card.timings_ms["source"] = "mcu-card roi"
        side = int(max(roi.image.shape[:2]))
        if mcu.decision is None and side < self.MCU_ROI_MIN_PX:
            # If the card is so small that the chip text physically can't be read,
            # tell the user that, rather than just "no part number found".
            # (In test photos a 256 px card area means letters about 3 px tall.)
            mcu.reason += (f"; the MCU card is only {side} px in this photo, the chip "
                           f"marking needs ~{self.MCU_ROI_MIN_PX} px: send the original "
                           f"photo as a file (messengers shrink photos to ~1280 px) or "
                           f"take a close-up of the card")
        return mcu, card

    @staticmethod
    def _map_boxes(T: np.ndarray, mcu: Recognition, card: Recognition) -> None:
        def m(box):
            return np.round(L.apply_h(T, np.asarray(box, float))).astype(int).tolist()
        if card.group and card.group.get("box") is not None:
            card.group["box"] = m(card.group["box"])
        for c in mcu.candidates + card.candidates:
            if c.detail.get("box") is not None:
                c.detail["box"] = m(c.detail["box"])
        for ev in mcu.evidence:
            if ev.get("box") is not None:
                ev["box"] = m(ev["box"])
        for cr in mcu.crops:
            cr.region.pts = L.apply_h(T, cr.region.pts)

    def mcu_from_card(self, pre: Preprocessed, crop: CardCrop, board_key: str | None,
                      name: str = "card"):
        """Read everything on the found and zoomed card, then identify the card and its MCU.

        Boxes in the result are converted back to positions on the processed photo.
        """
        reading = read_card(crop, self.backend, upscaler=self.upscaler, chip_mode=self.chip_mode,
                            debug_dir=self.faint_debug, debug_prefix=name)
        img = crop.image
        cpre = Preprocessed(img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), np.eye(3), 1.0, 0.0,
                            assess_quality(img), ["card crop"])
        ratio = (package.chip_ratio(reading.chip.quad, img.shape)
                 if reading.chip is not None and reading.chip.score >= 0.6 else None)
        mcu, card = self.mcus.recognise(cpre, list(reading.tokens), board_key=board_key,
                                        rotations_read=(0, 90, 180, 270), chip_ratio=ratio)
        self._map_boxes(pre.M @ np.linalg.inv(crop.H), mcu, card)
        if self.verify_marking and card.decision is None and card.group and reading.chip is not None:
            self.verify_card_marking(crop, reading, card, mcu, ratio, board_key)
        mcu.timings_ms["source"] = card.timings_ms["source"] = "mcu card (located + zoomed)"
        mcu.timings_ms["card_read"] = reading.timings_ms
        side = int(max(crop.native.shape[:2]))
        if mcu.decision is None and side < self.MCU_ROI_MIN_PX:
            mcu.reason += (f"; the MCU card is only {side} px in this photo, the chip "
                           f"marking needs ~{self.MCU_ROI_MIN_PX} px: send the original "
                           f"photo as a file (messengers shrink photos to ~1280 px) or "
                           f"take a close-up of the card")
        return mcu, card, reading

    def verify_card_marking(self, crop: CardCrop, reading: CardReading, card, mcu,
                            ratio: float | None, board_key: str | None) -> None:
        """Second opinion for when we know the layout group but couldn't read the chip.

        For each card in the group we simulate how its chip marking would look
        at this photo's resolution, and compare that with the faint-text map
        of the real chip (``vision.faint.synth``). We only pick a card (and so
        the MCU) when one option is clearly the best match.
        """
        import time as _t
        from .faint import cluster as FC, features as FF
        from .faint.region import PAD, rectify
        from .faint.synth import MarkingVerifier, hypotheses_for
        from .results import Candidate
        t0 = _t.perf_counter()
        groups = set(card.group.get("groups") or [card.group["key"]])
        from .card_ocr import GroupScore
        cards = self.mcus.card_rec.cards_of(
            [GroupScore(self.catalog.card_groups[g], 1.0) for g in sorted(groups)
             if g in self.catalog.card_groups], board_key)
        cards, _ = package.filter_cards(cards, ratio)
        hyps = hypotheses_for(cards)
        if len(hyps) < 2:
            return
        qn = reading.chip.quad / crop.zoom
        chip = rectify(crop.native, qn, method="card", upscaler=self.upscaler)
        pf = FF.compute(chip.image, chip.body_mask, chip.glyph_px)
        cr = FC.cluster(pf, chip.body_mask)
        native = float(np.mean([np.linalg.norm(qn[(i + 1) % 4] - qn[i]) for i in range(4)]))
        scale = chip.image.shape[0] / (native * (1 + 2 * PAD))
        res = MarkingVerifier().verify(pf.resid["lab"][:, :, 0], chip.body_mask, cr.polarity,
                                       chip.glyph_px, scale, hyps)
        card.group["marking_verifier"] = {**res.as_dict(),
                                          "ms": round((_t.perf_counter() - t0) * 1000)}
        if res.decision is None:
            card.reason += f"; {res.reason}"
            return
        c = next(x for x in cards if x.key == res.decision)
        dec = Candidate(c.key, c.name, 0.9, c.pid, {"mcu": c.mcu, "group": c.group,
                                                     "via": "title + synthetic marking match",
                                                     "margin": round(res.margin, 3)})
        card.decision, card.candidates = dec, [dec] + [x for x in card.candidates if x.key != c.key][:2]
        card.reason = f"{card.group['title']}: {res.reason}"
        if c.mcu:
            m = self.catalog.mcu_by_part(c.mcu)
            md = Candidate(c.mcu, c.mcu, 0.9, m.pid if m else None, {"from_card": c.key})
            mcu.decision, mcu.candidates = md, [md] + mcu.candidates[:2]
            mcu.reason = f"{c.mcu} from the card: {res.reason}"

    # If we are at least this sure about where the card is, but couldn't read it,
    # don't bother reading the layout area or whole photo: they show the same
    # pixels, just smaller.
    CARD_SURE = 0.8

    # Below this card size (px), the chip text is too small to read.
    # From eval/synth_eval_synthetic.md: OCR can read an LQFP-144 marking when
    # the chip is about 250 px wide, which means the card is about 800 px.
    # Real 960x1280 phone photos give cards of only 230-260 px (chip 71-77 px).
    MCU_ROI_MIN_PX = 800

    def run_clicks(self, photo: Photo, pre: Preprocessed, board: Recognition,
                   boxes=None, quad=None) -> list:
        if self.clicks is None or self.click_mode == "off":
            return []
        boxes = list(boxes or [])
        if not boxes and self.click_mode == "photo":
            boxes = [None]
        out = [self.clicks.recognise(photo.image, b) for b in boxes]
        # Slot names only work when the processed photo is exactly the board (user clicked 4
        # corners).
        sockets = self.catalog.sockets_for(board.decision.key) if board.decision else []
        if sockets and quad is not None and any(b is not None for b in boxes):
            from .clicks.recognizer import assign_slots
            h, w = pre.image.shape[:2]
            uv = []
            for b in boxes:
                if b is None:
                    uv.append((-9.0, -9.0))
                    continue
                c = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2, 1.0]]) @ pre.M.T
                uv.append((c[0, 0] / c[0, 2] / w, c[0, 1] / c[0, 2] / h))
            for r, slot, b in zip(out, assign_slots(uv, sockets), boxes):
                r.slot = slot if b is not None else None
        return out

    def run(self, photo: Photo, quad=None, click_boxes=None,
            board_hint: Recognition | None = None) -> PhotoResult:
        """Recognise everything on one photo.

        board_hint: the board found on an earlier photo of the same setup (for
        example in a camera session, a full board shot followed by an MCU
        close-up). It's only used if we can't read the board on this photo. In
        that case we don't place the layout, because a close-up shows too little
        of the board for it to fit.
        """
        t0 = time.perf_counter()
        pre = preprocess(photo.image, self.cfg, quad=quad)
        t1 = time.perf_counter()
        board, tokens = self.boards.recognise(pre)
        board = self.board_from_layouts(pre, board, tokens)
        hinted = False
        if board.decision is None and board_hint is not None and board_hint.decision is not None:
            hinted = True
            board = Recognition("board", list(board_hint.candidates), board_hint.decision,
                                f"{board_hint.decision.name} from the earlier photo of this "
                                f"setup (this photo: {board.reason})", board.evidence,
                                board.raw_reads, board.timings_ms,
                                rotations_read=board.rotations_read)
        t2 = time.perf_counter()
        lay, placement = (None, None) if hinted else self.place_layout(photo, pre, board,
                                                                       tokens, quad)
        if lay is not None and (placement is None or
                                placement.inliers < self.LAYOUT_STRONG_INLIERS):
            # The first board pass stops as soon as it reads the board name. If that
            # didn't give enough labels to place the layout reliably, read the
            # rotations it skipped (only when the board is known and has a layout).
            done = rotations_covered(self.backend, set(board.rotations_read or [0]))
            todo = [r for r in (0, 90, 180, 270) if r not in done]
            if todo:
                # Board labels are big, so a 1400 px image is enough (the processed photo
                # is often an upscaled phone photo). On one test photo this cut the time
                # from 9.9 s to 2.7 s and still matched 9 labels instead of 12.
                s = min(1.0, self.LAYOUT_PASS_LONG_SIDE / max(pre.image.shape[:2]))
                img = (cv2.resize(pre.image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                       if s < 1.0 else pre.image)
                extra = read_text(img, self.backend, rotations=todo)
                for t in extra:
                    t.poly = t.poly / s
                tokens = list(tokens) + extra
                board.rotations_read = sorted(rotations_covered(self.backend, done | set(todo)))
                lay2, pl2 = self.place_layout(photo, pre, board, tokens, quad)
                if pl2 is not None and (placement is None or pl2.inliers > placement.inliers):
                    placement = pl2
        rois = []
        if placement is not None:
            if lay.mcu:
                rois.append(L.crop_roi(photo.image, pre, placement, "MCU card", lay.mcu,
                                       L.MCU_PAD))
            rois += [L.crop_roi(photo.image, pre, placement, s.slot, s.box, L.SOCKET_PAD)
                     for s in lay.sockets]
        t_lay = time.perf_counter()
        mcu_roi = next((r for r in rois if r.name == "MCU card" and r.visible > 0.3), None)
        board_key = board.decision.key if board.decision else None
        source = "photo"
        locs, ccrop, reading = [], None, None
        if self.card_stage:
            locs = self.locator.locate(photo.image, tokens, pre.to_original,
                                       layout_quad=mcu_roi.quad_orig if mcu_roi is not None else None)
            if locs:
                ccrop = crop_card(photo.image, locs[0], self.card_zoom_target,
                                  self.card_max_zoom, upscaler=self.upscaler)
        t_loc = time.perf_counter()
        mcu = card = None
        if ccrop is not None:
            mcu, card, reading = self.mcu_from_card(
                pre, ccrop, board_key, photo.path.stem if photo.path else photo.sha1[:10])
            source = "card"
            if (mcu.decision is None and card.decision is None and not card.group
                    and ccrop.location.score < self.CARD_SURE):
                # Nothing useful came from the card crop (it might even be in the wrong
                # place), so fall back to the layout area / whole photo below. If we were
                # confident about the card's location we skip that, since it would only
                # re-read the same pixels at a smaller size.
                mcu = card = None
        if mcu is not None:
            pass
        elif self.mcus is None:
            mcu, card = (Recognition("mcu", [], None, "skipped"),
                         Recognition("mcu_card", [], None, "skipped"))
        elif mcu_roi is not None:
            mcu, card = self.mcu_from_roi(photo, pre, mcu_roi, board_key)
            source = "roi" if ccrop is None else "roi (card crop gave nothing)"
            if mcu.decision is None and card.decision is None and self.roi_fallback:
                mcu, card = self.mcus.recognise(pre, tokens, hires=photo.image,
                                                 rotations_read=board.rotations_read,
                                                 board_key=board_key)
                source = "photo (roi fallback)"
        else:
            mcu, card = self.mcus.recognise(pre, tokens, hires=photo.image,
                                             rotations_read=board.rotations_read,
                                             board_key=board_key)
            if ccrop is not None:
                source = "photo (card crop gave nothing)"
        t3 = time.perf_counter()
        clicks = self.run_clicks(photo, pre, board, click_boxes, quad)
        t4 = time.perf_counter()
        return PhotoResult(
            photo=photo.name, quality=pre.quality.as_dict(), preprocess_steps=pre.steps,
            board=board, mcu=mcu, mcu_card=card,
            timings_ms={"preprocess": round((t1 - t0) * 1000),
                        "board": round((t2 - t1) * 1000),
                        "layout": round((t_lay - t2) * 1000),
                        "card_locate": round((t_loc - t_lay) * 1000),
                        "mcu": round((t3 - t_loc) * 1000),
                        "clicks": round((t4 - t3) * 1000),
                        "total": round((t4 - t0) * 1000),
                        "backend": self.backend.name},
            pre=pre, clicks=clicks, placement=placement, rois=rois, mcu_source=source,
            card_crop=ccrop, card_reading=reading, card_locations=locs)
