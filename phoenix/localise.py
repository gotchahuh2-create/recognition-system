"""Experimental card localisation order: geometry first (hypothesis H-002).

Protected default (``vision.card_locate.CardLocator``): find closed yellow frames
on the whole photo; the board layout's MCU-card area only adds agreement.

Geometry-first (this module): when the board layout was placed on the photo,
search for the frame only inside the layout's MCU-card area (grown by
``GROW``), so frames elsewhere on the board can never win; if no frame is found
there, the layout area itself is the location. Without a placed layout it
behaves exactly like the default.

This is a registered alternative (``config/registry.yaml``: card_localise),
not a change to ``vision/``.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision.card_locate import CardLocation, CardLocator, order_quad

GROW = 0.15       # search window = layout MCU-card area grown by 15 % per side
MIN_FRAME_IN_ROI = 0.55  # frame score needed inside the window


class GeometryFirstLocator(CardLocator):
    name = "geometry_first"

    def locate(self, photo, tokens=(), to_orig=None, layout_quad=None):
        if layout_quad is None:
            return super().locate(photo, tokens, to_orig, layout_quad)
        q = order_quad(layout_quad)
        c = q.mean(0)
        win = c + (q - c) * (1 + 2 * GROW)
        h, w = photo.shape[:2]
        x0, y0 = np.floor(win.min(0)).astype(int)
        x1, y1 = np.ceil(win.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        out: list[CardLocation] = []
        if x1 - x0 > 32 and y1 - y0 > 32:
            sub = photo[y0:y1, x0:x1]
            for f in self.frame_candidates(sub):
                f.quad = f.quad + np.array([x0, y0], float)
                if f.score >= MIN_FRAME_IN_ROI:
                    f.method = "layout+frame"
                    f.score = min(1.0, f.score + 0.2)
                    f.evidence.append("frame found inside the layout's MCU-card window")
                    out.append(f)
        if not out:
            out.append(CardLocation(q, "layout", 0.8, ["board layout MCU-card area "
                                                         "(geometry-first, no frame inside)"]))
        out.sort(key=lambda l: -l.score)
        return out


def frame_area_ratio(loc: CardLocation, layout_quad) -> float:
    """Area of the chosen card quad relative to the layout area (diagnostics)."""
    a = cv2.contourArea(loc.quad.astype(np.float32))
    b = cv2.contourArea(order_quad(layout_quad).astype(np.float32))
    return float(a / b) if b else 0.0
