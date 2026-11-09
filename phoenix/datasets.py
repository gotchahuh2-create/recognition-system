"""Benchmark datasets: manifests in ``bench/datasets/*.json``, images stay out of git.

A manifest lists every photo with its SHA-256 and its ground truth::

    {"name": "real3", "kind": "real", "root": "tests/real",
     "items": [{"file": "uni-ds-v8_stm32f429zi_1.jpg", "sha256": "...",
                "truth": {"board": "uni-ds-v8", "mcu": "STM32F429ZI",
                          "mcu_card": "mcu-card-for-stm32-stm32f429zi",
                          "mcu_card_layout": "mcu-card-for-stm32"},
                "verified": true}]}

Truth fields: ``board`` (board key, or ``"none"`` = the right answer is to
abstain), ``mcu`` (part number), ``mcu_card`` (card key), ``mcu_card_layout``
(layout group key), ``sockets`` ({slot: empty|occupied}). A missing field is
not scored. ``verified: false`` items carry only ``provisional`` labels
(today's predictions, to be confirmed by a person); they are measured for
agreement and regression, never for accuracy.

Synthetic sets are regenerated deterministically from their ``generator``
command into ``runs/data/<name>`` when missing.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .util import BENCH_DIR, ROOT, RUNS_DIR, read_json, rel, sha256_file, write_json

DATASET_DIR = BENCH_DIR / "datasets"
FIELDS = ("board", "mcu", "mcu_card", "mcu_card_layout")

# Socket occupancy read off the photos by eye (Claude, 2026-11-09; contact sheet in
# docs/experiments/E-006-*/). To be confirmed by the owner. Close-ups show no sockets.
_E, _O = "empty", "occupied"
SOCKET_TRUTH = {
    "uni-ds-v8_pic32mz2048efh144_1.jpg": _O, "uni-ds-v8_stm32f429zi_1.jpg": _E,
    "uni-ds-v8_stm32f429zi_2.jpg": _E, "20261105_144642_board_01.jpg": _E,
    "20261105_144827_board_01.jpg": _E, "20261105_151336_board_01.jpg": _O,
    "20261105_153735_board_01.jpg": _O,
}


def _sockets(file: str) -> dict:
    v = SOCKET_TRUTH.get(file)
    return {f"mikroBUS {i}": v for i in range(1, 6)} if v else {}


@dataclass
class Item:
    file: str
    path: Path
    sha256: str
    truth: dict
    verified: bool
    provisional: dict
    meta: dict


class Dataset:
    def __init__(self, manifest: dict, path: Path | None = None):
        self.m = manifest
        self.path = path
        self.name = manifest["name"]

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Dataset":
        p = Path(name_or_path)
        if not p.suffix:
            p = DATASET_DIR / f"{name_or_path}.json"
        return cls(read_json(p), p)

    @property
    def root(self) -> Path:
        if self.m.get("generator"):
            return RUNS_DIR / "data" / self.name
        return ROOT / self.m["root"]

    def ensure(self) -> Path:
        """Make sure the images exist (regenerate synthetic sets)."""
        root = self.root
        gen = self.m.get("generator")
        if gen and not any(root.glob("*.jpg")):
            root.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable] + [a.replace("{out}", str(root)) for a in gen["cmd"]]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        return root

    def items(self, check_hash: bool = False) -> list[Item]:
        root = self.ensure()
        out = []
        for it in self.m["items"]:
            p = root / it["file"]
            if not p.exists():
                raise FileNotFoundError(f"{self.name}: missing {p}")
            if check_hash and it.get("sha256") and sha256_file(p) != it["sha256"]:
                raise ValueError(f"{self.name}: {it['file']} changed (sha256 mismatch)")
            out.append(Item(it["file"], p, it.get("sha256", ""), it.get("truth", {}),
                            bool(it.get("verified", False)), it.get("provisional", {}),
                            {k: v for k, v in it.items()
                             if k not in ("file", "sha256", "truth", "verified", "provisional")}))
        return out


# ---------------------------------------------------------------- builders
def _manifest(name, kind, root, items, note, generator=None):
    m = {"name": name, "kind": kind, "version": 1, "note": note, "items": items}
    if generator:
        m["generator"] = generator
    else:
        m["root"] = root
    return m


def build_real3() -> dict:
    root = ROOT / "tests" / "real"
    truth = {  # from the file names + README 0.8 (layout groups checked against the photos)
        "uni-ds-v8_pic32mz2048efh144_1.jpg": ("PIC32MZ2048EFH144", "mcu-card-3-for-pic32-pic32mz2048efh144",
                                              "mcu-card-3-for-pic32"),
        "uni-ds-v8_stm32f429zi_1.jpg": ("STM32F429ZI", "mcu-card-for-stm32-stm32f429zi", "mcu-card-for-stm32"),
        "uni-ds-v8_stm32f429zi_2.jpg": ("STM32F429ZI", "mcu-card-for-stm32-stm32f429zi", "mcu-card-for-stm32"),
    }
    items = []
    for f, (mcu, card, group) in truth.items():
        items.append({"file": f, "sha256": sha256_file(root / f),
                      "truth": {"board": "uni-ds-v8", "mcu": mcu, "mcu_card": card,
                                "mcu_card_layout": group,
                                **({"sockets": _sockets(f)} if _sockets(f) else {})},
                      "verified": True,
                      "note": "960x1280 messenger copy: the chip marking is ~3 px, so abstaining "
                              "on mcu / mcu_card is the expected answer"})
    return _manifest("real3", "real", "tests/real", items,
                     "The three regression photos (README 0.4.1 / 0.8).")


def build_captures() -> dict:
    """Real phone captures. Truth comes from photos/captures/labels.csv once a person fills it."""
    root = ROOT / "photos" / "captures"
    labels = {}
    lab = root / "labels.csv"
    if lab.exists():
        with open(lab, newline="", encoding="utf-8") as f:
            labels = {r["filename"]: r for r in csv.DictReader(f)}
    items = []
    for jpg in sorted(root.glob("*.jpg")):
        side = jpg.with_suffix(".json")
        prov, mode = {}, None
        if side.exists():
            sc = json.loads(side.read_text(encoding="utf-8"))
            mode = sc.get("mode")
            rec = sc.get("recognition") or {}
            for k in FIELDS:
                v = rec.get(k)
                if isinstance(v, dict):
                    prov[k] = v.get("key")
        row = labels.get(jpg.name)
        truth = {k: row[k] for k in FIELDS if row and row.get(k)} if row else {}
        if _sockets(jpg.name):
            truth["sockets"] = _sockets(jpg.name)
        items.append({"file": jpg.name, "sha256": sha256_file(jpg), "truth": truth,
                      "verified": bool(row and row.get("verified", "").lower() in ("1", "yes", "true")),
                      "provisional": prov, "mode": mode,
                      **({"truth_source": {"sockets": "visual check by Claude, 2026-11-09"}}
                         if _sockets(jpg.name) else {}),
                      "session": jpg.name[:8]})
    return _manifest("captures", "real", "photos/captures", items,
                     "Phone captures of 5 Nov 2026 (2160x3840). Unverified until "
                     "photos/captures/labels.csv is filled (see bench/datasets/README.md).")


def build_synthetic(name: str, cmd: list[str], note: str) -> dict:
    out = RUNS_DIR / "data" / name
    if not any(out.glob("*.jpg")):
        subprocess.run([sys.executable] + [a.replace("{out}", str(out)) for a in cmd],
                       cwd=ROOT, check=True, capture_output=True)
    with open(out / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = []
    for r in rows:
        t = {}
        if r["kind"] == "board":
            t["board"] = r["label"]
        elif r["kind"] == "mcu":
            t["mcu"] = r["label"]
        elif r["kind"] == "mcu_card":
            t.update({"mcu_card": r["label"], "mcu": r.get("mcu") or None,
                      "mcu_card_layout": r.get("group") or None})
            t = {k: v for k, v in t.items() if v}
        items.append({"file": r["filename"], "sha256": sha256_file(out / r["filename"]),
                      "truth": t, "verified": True,
                      **({"preset": r["preset"]} if r.get("preset") else {})})
    return _manifest(name, "synthetic", None, items, note,
                     generator={"cmd": cmd, "deterministic": True})


def build_all() -> list[Path]:
    paths = [write_json(DATASET_DIR / "real3.json", build_real3()),
             write_json(DATASET_DIR / "captures.json", build_captures()),
             write_json(DATASET_DIR / "syn_ocr.json", build_synthetic(
                 "syn_ocr", ["scripts/make_synthetic.py", "{out}", "--hard"],
                 "16 synthetic scenes: 9 smoke (boards, chips, unknown board) + 7 hard chips "
                 "(defocus, motion, low light, small).")),
             write_json(DATASET_DIR / "syn_cards.json", build_synthetic(
                 "syn_cards", ["scripts/make_synthetic_cards.py", "{out}", "--n", "12",
                               "--presets", "close,mid,far"],
                 "36 synthetic MCU-card scenes from the card dataset: 12 cards x close/mid/far."))]
    return paths


def write_label_template() -> Path:
    """photos/captures/labels.TEMPLATE.csv, pre-filled with today's predictions to correct."""
    m = build_captures()
    out = ROOT / "photos" / "captures" / "labels.TEMPLATE.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", *FIELDS, "verified", "comment"])
        for it in m["items"]:
            p = it["provisional"]
            w.writerow([it["file"], *[p.get(k) or "" for k in FIELDS], "no",
                        f"{it.get('mode') or ''} shot; correct the values, then set verified=yes"])
    return out


if __name__ == "__main__":
    for p in build_all():
        print(rel(p))
