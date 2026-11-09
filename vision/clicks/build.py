"""Build or extend the reference gallery of Click images.

Accepted image folder layouts (you can mix them):

    catalog_images/MIKROE-1234/front.jpg        folder named after the PID
    catalog_images/MIKROE-1234_front.jpg        PID anywhere in the file name

For each PID the gallery gets every catalog photo (with the white margin
trimmed off), ``augment`` realistic variations of each, and, once there are
some, real photos from the *training* set (``real_dir``; never from the
validation or test sets). Adding a new Click just means computing its
image descriptors and adding them. No retraining needed.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..loader import IMAGE_EXTS, PhotoLoadError, load_photo
from .augment import Augmenter
from .embedder import Embedder
from .gallery import Gallery
from .preprocess import trim_margin

_PID_RE = re.compile(r"(MIKROE-\d{1,6})", re.IGNORECASE)


def pid_from_path(p: Path, root: Path) -> str | None:
    m = _PID_RE.search(p.stem)
    if m:
        return m.group(1).upper()
    rel = p.relative_to(root)
    if len(rel.parts) > 1:  # folder layout: the first folder is the PID
        m = _PID_RE.search(rel.parts[0])
        return m.group(1).upper() if m else rel.parts[0]
    return None


def scan_images(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    out: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            pid = pid_from_path(p, root)
            if pid:
                out.setdefault(pid, []).append(p)
    return out


def _read(p: Path) -> np.ndarray | None:
    try:
        return load_photo(p).image
    except PhotoLoadError as e:
        print(f"[gallery] skipped {p}: {e}")
        return None


def add_images(gallery: Gallery, embedder: Embedder, pid: str, paths: Iterable[Path],
               source: str = "catalog", augment: int = 12, aug: Augmenter | None = None,
               name: str | None = None) -> int:
    """Compute descriptors for one Click's images (plus variations of catalog images) and add
    them.

    Returns how many vectors were added.
    """
    gallery.check(embedder.model_version, embedder.preprocess_version)
    aug = aug or Augmenter()
    crops, ids, paths_out, sources = [], [], [], []
    for p in paths:
        img = _read(Path(p))
        if img is None:
            continue
        base = trim_margin(img) if source == "catalog" else img
        crops.append(base)
        ids.append(f"{pid}:{Path(p).name}")
        paths_out.append(str(p))
        sources.append(source)
        if source == "catalog":
            for j, a in enumerate(aug.many(base, augment)):
                crops.append(a)
                ids.append(f"{pid}:{Path(p).name}:aug{j}")
                paths_out.append(str(p))
                sources.append("augmented")
    if not crops:
        return 0
    vecs = embedder.embed(crops)
    for src in ("catalog", "augmented", "real"):
        sel = [i for i, s in enumerate(sources) if s == src]
        if sel:
            gallery.add(pid, vecs[sel], src, [ids[i] for i in sel],
                        [paths_out[i] for i in sel], name=name)
    return len(crops)


def build_gallery(embedder: Embedder, catalog_dir: str | Path | None,
                  out_dir: str | Path | None = None, augment: int = 12,
                  real_dir: str | Path | None = None, names: dict[str, str] | None = None,
                  backgrounds: Iterable[str | Path] = (), seed: int = 0,
                  base: Gallery | None = None,
                  progress: Callable[[str], None] | None = print) -> Gallery:
    names = names or {}
    g = base or Gallery.empty(embedder.dim, embedder.model_version, embedder.preprocess_version)
    aug = Augmenter(seed=seed, backgrounds=list(backgrounds))
    t0 = time.perf_counter()
    jobs = []
    if catalog_dir:
        jobs += [(pid, ps, "catalog") for pid, ps in scan_images(catalog_dir).items()]
    if real_dir:
        jobs += [(pid, ps, "real") for pid, ps in scan_images(real_dir).items()]
    for n, (pid, ps, src) in enumerate(jobs, 1):
        k = add_images(g, embedder, pid, ps, src, augment, aug, names.get(pid))
        if progress and (n % 25 == 0 or n == len(jobs)):
            progress(f"[gallery] {n}/{len(jobs)} image sets, {len(g)} vectors, "
                     f"{time.perf_counter() - t0:.0f} s (last {pid}: +{k})")
    for pid in g.pids:
        if pid in names:
            g.names[pid] = names[pid]
    if out_dir:
        g.save(out_dir)
    return g
