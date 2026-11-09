"""Small shared helpers: repo paths, hashing, git identity, JSON IO."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
BENCH_DIR = ROOT / "bench"
DOCS_DIR = ROOT / "docs"
RUNS_DIR = ROOT / "runs"  # git-ignored run output (traces, scratch data)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj, n: int = 10) -> str:
    """Hash of a JSON-serialisable object, independent of key order."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def git_commit(root: Path = ROOT) -> str:
    """Short commit id, with ``+dirty`` when the work tree has changes; ``nogit`` without git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if not sha:
            return "nogit"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    return path


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
