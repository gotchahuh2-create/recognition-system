"""The reference gallery: image descriptors tagged with PIDs, plus an exact search index.

    gallery/embeddings.npy   float32 [N, d], each row normalised to length 1
    gallery/meta.parquet     per row: pid, image_id, source (catalog/augmented/real),
                             model_version, preprocess_version, path
                             (meta.csv when pyarrow isn't installed)
    gallery/index.faiss      IndexFlatIP over the .npy (optional; rebuilt if out of date)
    gallery/gallery.json     versions, dim, count, PID -> name

Each Click has several vectors, and it scores by its best-matching one
(we never average them). Every row records the model and preprocessing
version, and loading with a different embedder fails with a clear error.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

META_COLS = ["pid", "image_id", "source", "model_version", "preprocess_version", "path"]

try:  # optional: plain NumPy search is fast enough for ~50k x 384
    import faiss  # type: ignore
except Exception:  # pragma: no cover - depends on install
    faiss = None


class GalleryMismatch(RuntimeError):
    """The gallery was built with a different model or preprocessing than the embedder."""


# ---- Exact search index ----
class FlatIndex:
    """Uses FAISS IndexFlatIP if installed, otherwise NumPy ``G @ q``.

    Both give the same result: since every vector has length 1, the inner
    product is the cosine similarity.
    """

    def __init__(self, vectors: np.ndarray, use_faiss: bool | None = None):
        self.vectors = np.ascontiguousarray(vectors, np.float32)
        self.use_faiss = (faiss is not None) if use_faiss is None else (use_faiss and faiss is not None)
        self.index = None
        if self.use_faiss and len(self.vectors):
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    @property
    def backend(self) -> str:
        return "faiss" if self.index is not None else "numpy"

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(q, np.float32))
        k = min(k, len(self.vectors))
        if k <= 0:
            return np.zeros((len(q), 0), np.float32), np.zeros((len(q), 0), np.int64)
        if self.index is not None:
            return self.index.search(q, k)
        sims = q @ self.vectors.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(sims, idx, 1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(part, order, 1), np.take_along_axis(idx, order, 1)


# ---- Gallery ----
@dataclass
class Gallery:
    vectors: np.ndarray
    meta: list[dict]
    model_version: str
    preprocess_version: str
    names: dict[str, str] = field(default_factory=dict)
    created: str = ""
    _index: FlatIndex | None = field(default=None, repr=False)

    # ---- Basics ----
    @classmethod
    def empty(cls, dim: int, model_version: str, preprocess_version: str) -> "Gallery":
        return cls(np.zeros((0, dim), np.float32), [], model_version, preprocess_version)

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def pids(self) -> list[str]:
        return sorted({m["pid"] for m in self.meta})

    @property
    def pid_array(self) -> np.ndarray:
        return np.array([m["pid"] for m in self.meta], dtype=object)

    @property
    def index(self) -> FlatIndex:
        if self._index is None or len(self._index.vectors) != len(self.vectors):
            self._index = FlatIndex(self.vectors)
        return self._index

    def name(self, pid: str) -> str:
        return self.names.get(pid, pid)

    def check(self, model_version: str, preprocess_version: str) -> None:
        if model_version != self.model_version or preprocess_version != self.preprocess_version:
            raise GalleryMismatch(
                f"gallery was built with model {self.model_version!r} / preprocessing "
                f"{self.preprocess_version!r} but the embedder is {model_version!r} / "
                f"{preprocess_version!r}; rebuild it: python -m vision.clicks build ...")

    def add(self, pid: str, vectors: np.ndarray, source: str,
            image_ids: Sequence[str] | None = None, paths: Sequence[str] | None = None,
            name: str | None = None) -> None:
        vectors = np.asarray(vectors, np.float32).reshape(-1, self.dim)
        n = len(vectors)
        image_ids = list(image_ids) if image_ids is not None else \
            [f"{pid}:{source}:{len(self.meta) + i}" for i in range(n)]
        paths = list(paths) if paths is not None else [""] * n
        self.vectors = np.vstack([self.vectors, vectors])
        for i in range(n):
            self.meta.append({"pid": pid, "image_id": image_ids[i], "source": source,
                              "model_version": self.model_version,
                              "preprocess_version": self.preprocess_version,
                              "path": paths[i]})
        if name:
            self.names[pid] = name
        self._index = None

    def subset(self, keep: Iterable[bool]) -> "Gallery":
        keep = np.asarray(list(keep), bool)
        return Gallery(self.vectors[keep], [m for m, k in zip(self.meta, keep) if k],
                       self.model_version, self.preprocess_version, dict(self.names),
                       self.created)

    def without_pids(self, pids: Iterable[str]) -> "Gallery":
        drop = set(pids)
        return self.subset(m["pid"] not in drop for m in self.meta)

    def only_sources(self, sources: Iterable[str]) -> "Gallery":
        s = set(sources)
        return self.subset(m["source"] in s for m in self.meta)

    # ---- Search ----
    def search(self, q: np.ndarray, k: int = 50, top: int = 5) -> list[tuple[str, float, int]]:
        """Return the best ``top`` PIDs as (pid, best cosine, row of the best vector).

        Searches the top k vectors, groups them by PID, and scores each Click
        by its best vector.
        """
        sims, ids = self.index.search(q, k)
        best: dict[str, tuple[float, int]] = {}
        for s, i in zip(sims[0], ids[0]):
            if i < 0:
                continue
            pid = self.meta[int(i)]["pid"]
            if pid not in best or s > best[pid][0]:
                best[pid] = (float(s), int(i))
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top]
        return [(pid, s, i) for pid, (s, i) in ranked]

    # ---- Saving / loading ----
    def save(self, folder: str | Path) -> Path:
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", np.ascontiguousarray(self.vectors, np.float32))
        meta_file = _write_meta(d, self.meta)
        self.created = self.created or time.strftime("%Y-%m-%dT%H:%M:%S")
        info = {"format": 1, "model_version": self.model_version,
                "preprocess_version": self.preprocess_version, "dim": self.dim,
                "count": len(self), "pids": len(self.pids), "created": self.created,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "meta_file": meta_file.name, "names": self.names}
        (d / "gallery.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
        if faiss is not None and len(self):
            faiss.write_index(self.index.index or FlatIndex(self.vectors, True).index,
                              str(d / "index.faiss"))
        return d

    @classmethod
    def load(cls, folder: str | Path, model_version: str | None = None,
             preprocess_version: str | None = None) -> "Gallery":
        d = Path(folder)
        info_p = d / "gallery.json"
        if not info_p.is_file():
            raise FileNotFoundError(f"no gallery at {d} (gallery.json missing)")
        info = json.loads(info_p.read_text(encoding="utf-8"))
        vecs = np.load(d / "embeddings.npy").astype(np.float32)
        meta = _read_meta(d, info.get("meta_file"))
        if len(meta) != len(vecs):
            raise GalleryMismatch(f"{d}: {len(vecs)} embeddings but {len(meta)} meta rows")
        bad = {(m["model_version"], m["preprocess_version"]) for m in meta} - \
            {(info["model_version"], info["preprocess_version"])}
        if bad:
            raise GalleryMismatch(f"{d}: rows from other model/preprocessing versions: {bad}")
        g = cls(vecs, meta, info["model_version"], info["preprocess_version"],
                info.get("names", {}), info.get("created", ""))
        if model_version is not None or preprocess_version is not None:
            g.check(model_version or g.model_version, preprocess_version or g.preprocess_version)
        idx_p = d / "index.faiss"
        if faiss is not None and idx_p.is_file():
            idx = faiss.read_index(str(idx_p))
            if idx.ntotal == len(vecs) and idx.d == vecs.shape[1]:
                fi = FlatIndex(np.zeros((0, vecs.shape[1]), np.float32), use_faiss=False)
                fi.vectors, fi.index, fi.use_faiss = g.vectors, idx, True
                g._index = fi
        return g


def _write_meta(d: Path, meta: list[dict]) -> Path:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        table = pa.table({c: [str(m.get(c, "")) for m in meta] for c in META_COLS})
        pq.write_table(table, d / "meta.parquet")
        (d / "meta.csv").unlink(missing_ok=True)
        return d / "meta.parquet"
    except ImportError:
        with open(d / "meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, META_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(meta)
        (d / "meta.parquet").unlink(missing_ok=True)
        return d / "meta.csv"


def _read_meta(d: Path, name: str | None) -> list[dict]:
    p = d / (name or "meta.parquet")
    if not p.is_file():
        p = d / "meta.csv" if (d / "meta.csv").is_file() else d / "meta.parquet"
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq  # type: ignore
        t = pq.read_table(p).to_pydict()
        n = len(next(iter(t.values()))) if t else 0
        return [{c: (t.get(c, [""] * n)[i] or "") for c in META_COLS} for i in range(n)]
    with open(p, newline="", encoding="utf-8") as f:
        return [{c: r.get(c, "") or "" for c in META_COLS} for r in csv.DictReader(f)]
