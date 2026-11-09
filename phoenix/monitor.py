"""System Monitoring: one trace record per recognition run.

A trace is a flat, JSON-serialisable summary of a ``PhotoResult``: what was
decided (or not) and why, how long each stage took, where the card was found,
the photo quality, and the configuration that produced it. Traces are the raw
material for the Diagnostics Engine and for regression comparison.

Sources:
* ``trace_of(result, ...)``   a run made by PHOENIX (bench, experiments);
* ``ingest_captures()``       the app's existing capture sidecars
                              (``photos/captures/*.json``), so every real phone
                              scan becomes telemetry without touching the app.

Traces are appended to ``runs/traces/<run_id>.jsonl`` (git-ignored).
"""

from __future__ import annotations

import json
from pathlib import Path

from .datasets import FIELDS
from .util import ROOT, RUNS_DIR

TRACE_DIR = RUNS_DIR / "traces"


def _dec(r):
    if r is None or r.decision is None:
        return None
    return {"key": r.decision.key, "score": round(float(r.decision.score), 3)}


def trace_of(result, *, run_id: str = "", item: str = "", dataset: str = "",
             config: dict | None = None, wall_ms: float | None = None) -> dict:
    m = result.manifest()
    recs = {"board": result.board, "mcu": result.mcu, "mcu_card": result.mcu_card}
    card = result.card_dict() or {}
    loc = card.get("located") or {}
    read = card.get("read") or {}
    t = {
        "run_id": run_id, "dataset": dataset, "item": item, "photo": result.photo,
        "config": config or {},
        "decisions": {k: _dec(v) for k, v in recs.items()},
        "decisions_flat": {"board": (m["board"] or {}).get("key"),
                           "mcu": (m["mcu"] or {}).get("key"),
                           "mcu_card": (m["mcu_card"] or {}).get("key"),
                           "mcu_card_layout": (m["mcu_card_layout"] or {}).get("key")},
        "reasons": {k: (v.reason if v is not None else None) for k, v in recs.items()},
        "top3": {k: [c.key for c in (v.candidates if v is not None else [])][:3]
                 for k, v in recs.items()},
        "raw_reads": {k: (v.raw_reads[:8] if v is not None else []) for k, v in recs.items()},
        "group_known": bool(result.mcu_card is not None and result.mcu_card.group),
        "timings_ms": dict(result.timings_ms),
        "wall_ms": round(wall_ms) if wall_ms is not None else None,
        "quality": result.quality,
        "mcu_source": result.mcu_source,
        "layout": ({"method": result.placement.method, "inliers": result.placement.inliers,
                    "matched": result.placement.matched}
                   if result.placement is not None else None),
        "card": ({"method": loc.get("method"), "score": loc.get("score"),
                  "size": loc.get("size"), "native_size": loc.get("native_size"),
                  "zoom": loc.get("zoom"),
                  "chip_found": bool(read.get("chip")) if read else None}
                 if loc else None),
        "needs_user": [n["field"] for n in m["needs_user"]],
        "sockets": [s.as_dict() for s in getattr(result, "sockets", []) or []],
    }
    return t


def append(run_id: str, traces: list[dict]) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    p = TRACE_DIR / f"{run_id}.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    return p


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ingest_captures(folder: Path = ROOT / "photos" / "captures") -> list[dict]:
    """Turn the app's capture sidecars into (partial) traces: field telemetry."""
    out = []
    for side in sorted(Path(folder).glob("*.json")):
        try:
            sc = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rec = sc.get("recognition") or {}
        pc = sc.get("photo_check") or {}
        dec = {k: (rec.get(k) or {}).get("key") if isinstance(rec.get(k), dict) else None
               for k in FIELDS}
        out.append({
            "run_id": "capture", "dataset": "captures", "item": sc.get("file", side.stem),
            "decisions_flat": dec,
            "reasons": {n["field"]: n.get("reason") for n in rec.get("needs_user", [])},
            "needs_user": [n["field"] for n in rec.get("needs_user", [])],
            "mode": sc.get("mode"),
            "photo_check": {k: pc.get(k) for k in ("ok", "reasons", "width", "height",
                                                   "board_score", "chip_px")},
            "phone": sc.get("phone"),
        })
    return out
