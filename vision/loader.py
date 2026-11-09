"""Load photos (and frames from videos) for the pipeline.

Turns phone or webcam photos into BGR numpy arrays, along with the extra
info the rest of the pipeline needs. It also:

* applies the EXIF rotation (phones save portrait photos sideways plus a
  flag saying how to rotate them),
* handles file paths with non-ASCII characters (cv2.imread can't on Windows),
* reads labels from the file name, for example:
      MIKROE-1234_phoneA_lamp_20deg_01.jpg
  -> pid=MIKROE-1234, device=phoneA, lighting=lamp, angle_deg=20, index=1
* hashes file contents, so we can check that a test set hasn't changed
  (results.csv stores this hash),
* grabs frames from short videos circling an object (1 frame per second),
  and tags each frame with its video, so frames from one video never end
  up in both the training and the test set.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Decoding a 200 MP image by mistake would use about 600 MB of memory, so refuse anything
# bigger.
MAX_PIXELS = 120_000_000

_PID_RE = re.compile(r"(MIKROE-\d{1,6})", re.IGNORECASE)
_ANGLE_RE = re.compile(r"^(\d{1,2})deg$", re.IGNORECASE)
_INDEX_RE = re.compile(r"^\d{1,4}$")


class PhotoLoadError(RuntimeError):
    pass


@dataclass
class Photo:
    """One decoded image, plus everything we know about where it came from."""

    image: np.ndarray  # BGR uint8, already rotated using EXIF
    path: Path | None
    sha1: str  # hash of the file bytes (or of the frame pixels)
    width: int
    height: int
    pid: str | None = None  # label taken from the file name, if any
    device: str | None = None
    lighting: str | None = None
    angle_deg: int | None = None
    index: int | None = None
    group_id: str | None = None  # the same for every frame of one video
    frame_time_s: float | None = None
    extra_tags: list[str] = field(default_factory=list)
    label: dict | None = None  # row from labels.csv, if there is one

    @property
    def name(self) -> str:
        if self.path is None:
            return self.sha1[:12]
        if self.frame_time_s is not None:
            return f"{self.path.name}@{self.frame_time_s:.1f}s"
        return self.path.name


# ---- File name labels ----
def parse_filename(stem: str) -> dict:
    """Read labels from a file name like ``MIKROE-1234_phoneA_lamp_20deg_01``.

    The parts are normally in that order, but the parser is forgiving: it
    finds the PID anywhere, the angle by its ``NNdeg`` form and the index as
    the last plain number. Of the remaining parts, the first is the device,
    the second is the lighting, and anything else goes into ``extra_tags``.
    """
    out: dict = {"pid": None, "device": None, "lighting": None,
                 "angle_deg": None, "index": None, "extra_tags": []}
    m = _PID_RE.search(stem)
    rest = stem
    if m:
        out["pid"] = m.group(1).upper()
        rest = (stem[: m.start()] + stem[m.end():])
    parts = [p for p in re.split(r"[_\s]+", rest) if p and p != "-"]
    free: list[str] = []
    for p in parts:
        a = _ANGLE_RE.match(p)
        if a:
            out["angle_deg"] = int(a.group(1))
        elif _INDEX_RE.match(p):
            out["index"] = int(p)  # the last number wins
        else:
            free.append(p)
    if free:
        out["device"] = free[0]
    if len(free) > 1:
        out["lighting"] = free[1]
    out["extra_tags"] = free[2:]
    return out


# ---- Decoding ----
def _decode_bytes(data: bytes, suffix: str) -> np.ndarray:
    """Decode an image with Pillow (which understands EXIF), or with OpenCV if that fails."""
    if suffix in {".heic", ".heif"}:
        try:  # optional: adds HEIC support
            import pillow_heif  # type: ignore
            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise PhotoLoadError("HEIC photo: `pip install pillow-heif` or export as JPEG") from e
    try:
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)  # apply the phone's rotation flag
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            arr = np.asarray(im)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except PhotoLoadError:
        raise
    except Exception:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise PhotoLoadError("could not decode image")
        return arr


def load_photo(path: str | Path, labels: dict[str, dict] | None = None) -> Photo:
    path = Path(path)
    if not path.is_file():
        raise PhotoLoadError(f"not a file: {path}")
    data = path.read_bytes()
    try:
        img = _decode_bytes(data, path.suffix.lower())
    except PhotoLoadError as e:
        raise PhotoLoadError(f"{path}: {e}") from e
    h, w = img.shape[:2]
    meta = parse_filename(path.stem)
    return Photo(
        image=img, path=path, sha1=hashlib.sha1(data).hexdigest(),
        width=w, height=h,
        pid=meta["pid"], device=meta["device"], lighting=meta["lighting"],
        angle_deg=meta["angle_deg"], index=meta["index"], extra_tags=meta["extra_tags"],
        label=(labels or {}).get(path.name),
    )


def photo_from_array(img: np.ndarray, name: str | None = None) -> Photo:
    """Wrap an image that is already in memory (for example from a webcam or an upload)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        raise PhotoLoadError("expected uint8 image")
    h, w = img.shape[:2]
    return Photo(image=img, path=Path(name) if name else None,
                 sha1=hashlib.sha1(img.tobytes()).hexdigest(), width=w, height=h)


def load_bytes(data: bytes, filename: str = "upload.jpg") -> Photo:
    """Decode an uploaded file (for example the bytes from FastAPI's UploadFile.read())."""
    suffix = Path(filename).suffix.lower()
    img = _decode_bytes(data, suffix)
    h, w = img.shape[:2]
    meta = parse_filename(Path(filename).stem)
    return Photo(image=img, path=Path(filename), sha1=hashlib.sha1(data).hexdigest(),
                 width=w, height=h, pid=meta["pid"], device=meta["device"],
                 lighting=meta["lighting"], angle_deg=meta["angle_deg"], index=meta["index"])


# ---- Videos ----
def iter_video_frames(path: str | Path, every_s: float = 1.0,
                      max_frames: int | None = None) -> Iterator[Photo]:
    """Yield one frame every ``every_s`` seconds.

    All frames share the same ``group_id`` (the video's file name), so a
    script that splits the data can keep them together.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PhotoLoadError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every_s)))
    meta = parse_filename(path.stem)
    i = n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = frame.shape[:2]
                yield Photo(image=frame, path=path,
                            sha1=hashlib.sha1(frame[::8, ::8].tobytes()).hexdigest(),
                            width=w, height=h, pid=meta["pid"], device=meta["device"],
                            lighting=meta["lighting"], angle_deg=meta["angle_deg"],
                            group_id=path.stem, frame_time_s=i / fps,
                            extra_tags=meta["extra_tags"])
                n += 1
                if max_frames and n >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


# ---- Folders ----
def read_labels_csv(path: str | Path) -> dict[str, dict]:
    """Read an optional ``labels.csv``. It needs at least a ``filename`` column.

    It's used for board and MCU photos, whose label isn't a Click PID:
        filename,kind,label
        fusion_arm_desk_01.jpg,board,Fusion for ARM v8
        card10_close_01.jpg,mcu,PIC18F65K40
    """
    with open(path, newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def iter_photos(source: str | Path | Iterable[str | Path], recursive: bool = True,
                include_videos: bool = False, video_every_s: float = 1.0,
                labels_csv: str | Path | None = None,
                on_error: str = "warn") -> Iterator[Photo]:
    """Yield photos from a file, a folder, or a list of paths.

    on_error: "warn" (print a message and skip), "skip" (skip silently) or "raise".
    If there is a ``labels.csv`` next to the photos, it's loaded automatically.
    """
    if isinstance(source, (str, Path)):
        src = Path(source)
        if src.is_dir():
            pattern = "**/*" if recursive else "*"
            paths = sorted(p for p in src.glob(pattern) if p.is_file())
            if labels_csv is None and (src / "labels.csv").is_file():
                labels_csv = src / "labels.csv"
        else:
            paths = [src]
    else:
        paths = [Path(p) for p in source]

    labels = read_labels_csv(labels_csv) if labels_csv else None
    for p in paths:
        ext = p.suffix.lower()
        try:
            if ext in IMAGE_EXTS:
                yield load_photo(p, labels)
            elif include_videos and ext in VIDEO_EXTS:
                yield from iter_video_frames(p, every_s=video_every_s)
        except PhotoLoadError as e:
            if on_error == "raise":
                raise
            if on_error == "warn":
                print(f"[loader] skipped: {e}")


def split_hash(photos: Iterable[Photo]) -> str:
    """Hash a set of photos, ignoring their order (stored in results.csv)."""
    h = hashlib.sha1()
    for s in sorted(p.sha1 for p in photos):
        h.update(s.encode())
    return h.hexdigest()[:16]
