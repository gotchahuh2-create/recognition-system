"""Saves debug images for the faint-text reader.

Every step calls ``saver.save(name, img)``. If no folder is set, the saver
does nothing, so it costs nothing in normal use.

Images are saved as ``<prefix>_<NN>_<name>.png`` in the order they're made,
so a folder listing follows the steps. Float maps are stretched (1st to 99th
percentile) to 0..255, and masks become black/white. ``sheet()`` saves one
overview image with every step, labelled, for a quick look.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


def to_u8(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Turn any map into a uint8 image that can be shown (gray or BGR)."""
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.dtype == np.uint8:
        return img
    a = img.astype(np.float32)
    if a.ndim == 3 and a.shape[2] == 3:
        return np.dstack([to_u8(a[:, :, i], lo, hi) for i in range(3)])
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape[:2], np.uint8)
    p0, p1 = np.percentile(finite, [lo, hi])
    if p1 - p0 < 1e-9:
        return np.zeros(a.shape[:2], np.uint8)
    return np.clip((a - p0) * 255.0 / (p1 - p0), 0, 255).astype(np.uint8)


class StageSaver:
    def __init__(self, folder: str | Path | None = None, prefix: str = "faint",
                 max_side: int = 1400):
        self.folder = Path(folder) if folder else None
        self.prefix = re.sub(r"[^\w.-]+", "_", prefix)
        self.max_side = max_side
        self.saved: list[tuple[str, Path]] = []
        if self.folder:
            self.folder.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.folder is not None

    def child(self, suffix: str) -> "StageSaver":
        s = StageSaver(self.folder, f"{self.prefix}_{suffix}", self.max_side)
        s.saved = self.saved  # share one overview sheet
        return s

    def save(self, name: str, img: np.ndarray) -> None:
        if not self.folder or img is None or img.size == 0:
            return
        u = to_u8(img)
        h, w = u.shape[:2]
        f = self.max_side / max(h, w)
        if f < 1:
            u = cv2.resize(u, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        p = self.folder / f"{self.prefix}_{len(self.saved):02d}_{re.sub(r'[^\w.-]+', '_', name)}.png"
        cv2.imwrite(str(p), u)
        self.saved.append((name, p))

    def sheet(self, name: str = "sheet", tile: int = 320, cols: int = 6) -> Path | None:
        if not self.folder or not self.saved:
            return None
        tiles = []
        for label, p in self.saved:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            f = tile / max(h, w)
            im = cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
            canvas = np.full((tile + 22, tile, 3), 40, np.uint8)
            canvas[22:22 + im.shape[0], :im.shape[1]] = im
            cv2.putText(canvas, label[:38], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(canvas)
        rows = []
        for i in range(0, len(tiles), cols):
            r = tiles[i:i + cols]
            r += [np.full_like(tiles[0], 40)] * (cols - len(r))
            rows.append(np.hstack(r))
        out = self.folder / f"{self.prefix}_{name}.jpg"
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return out
