"""Diagnostics Engine: turn traces and bench reports into named, counted weaknesses.

Each rule looks at every scored item of a bench report (and its trace fields)
and returns the items it matches. A weakness has a stable id, a severity, a
count, example items and the hypothesis that addresses it (if any). The
register ``docs/weaknesses.md`` is regenerated from the latest baseline report.

Severity:
  critical  a wrong answer (the system's core promise is "abstain, never guess")
  major     a required capability missing, or a field abstains where the
            pixels should be enough
  minor     cost (time) or data-quality issues
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .util import DOCS_DIR, read_json

PORT_GLUE = re.compile(r"(16|8|32)BITPORT(FOR|F0R)", re.I)


@dataclass
class Weakness:
    id: str
    title: str
    severity: str
    items: list[str] = field(default_factory=list)
    detail: str = ""
    hypothesis: str | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "severity": self.severity,
                "count": len(self.items), "examples": self.items[:5], "detail": self.detail,
                "hypothesis": self.hypothesis}


def _reason(it, f):
    return (it.get("reasons") or {}).get(f) or ""


RULES = []


def rule(fn):
    RULES.append(fn)
    return fn


@rule
def wrong_answers(items):
    hits = [f"{k}:{f}" for k, it in items.items()
            for f, s in it["scores"].items() if s.get("outcome") == "wrong"]
    return Weakness("W-WRONG", "Wrong answers (decided, but not the truth)", "critical", hits,
                    "Any entry here breaks the abstain-first promise.", "H-007, H-005")


@rule
def layout_group_wrong(items):
    hits = [k for k, it in items.items()
            if it["scores"].get("mcu_card_layout", {}).get("outcome") == "wrong"]
    return Weakness("W-LAYOUT-GROUP", "MCU-card layout group reported but wrong", "critical", hits,
                    "manifest.mcu_card_layout is shown as a fact with no accept/margin gate; a "
                    "misread title family ('for PIC' vs 'for PIC32') offers the user the wrong "
                    "cards.", "H-005")


@rule
def layout_not_placed(items):
    # the board was identified and has a layout, but the layout could not be placed
    from vision.layout import Layouts
    from vision.catalog import DATA_DIR
    have = set(Layouts.load(DATA_DIR / "layouts.json").keys())
    hits = [k for k, it in items.items()
            if (it.get("decisions") or {}).get("board") in have and not it.get("layout")]
    return Weakness("W-NO-LAYOUT", "Board has a layout but it was not placed (no geometry-first, "
                    "no socket areas)", "major", hits,
                    "Anchors calibrated with another OCR engine / resolution do not match.", "H-008")


@rule
def card_too_small(items):
    hits = [k for k, it in items.items() if "the MCU card is only" in _reason(it, "mcu")]
    return Weakness("W-SMALL-CARD", "Card too small for the chip marking (expected abstain)",
                    "minor", hits, "Correct behaviour; counted so guided capture can be judged.")


@rule
def glued_port_title(items):
    hits = []
    for k, it in items.items():
        reads = " ".join(str(x) for x in (it.get("raw_reads") or {}).get("mcu", []))
        dec = it.get("decisions") or {}
        if PORT_GLUE.search(reads.replace(" ", "").replace("-", "")) and not dec.get("mcu_card_layout"):
            hits.append(k)
    return Weakness("W-GLUED-TITLE", "Port box and title family read as one word, no layout group",
                    "major", hits, "e.g. '16BITPORTFORSTM32' on capture 153743 (P11).", "H-001")


@rule
def chip_not_found(items):
    hits = [k for k, it in items.items()
            if (it.get("card") or {}).get("method") and (it.get("card") or {}).get("chip_found") is False]
    return Weakness("W-NO-CHIP", "Card located but no chip body found on it", "major", hits)


@rule
def no_card(items):
    hits = [k for k, it in items.items()
            if not it.get("card") and "mcu" in it["scores"]
            and it["scores"]["mcu"].get("outcome") in ("abstain", "unverified")
            and not (it.get("decisions") or {}).get("mcu")]
    return Weakness("W-NO-CARD", "No MCU card located and no MCU decided", "major", hits)


@rule
def abstain_with_truth_in_top3(items):
    hits = [f"{k}:{f}" for k, it in items.items() for f, s in it["scores"].items()
            if s.get("outcome") == "abstain" and s.get("top3")]
    return Weakness("W-NEAR-MISS", "Abstained although the truth was the top-1..3 candidate",
                    "minor", hits, "Candidates for a better tie-breaker, never for a lower threshold "
                    "without evidence (ADR-0037).")


@rule
def slow(items):
    hits = [f"{k} ({round(it['wall_ms'] / 1000)} s)" for k, it in items.items()
            if it.get("wall_ms", 0) > 20000]
    return Weakness("W-SLOW", "Runs over 20 s on the dev CPU", "minor", hits,
                    "See docs/reports/efficiency.md.", "H-006")


def diagnose(report: dict) -> list[Weakness]:
    items = report["items"]
    # traces carry raw reads; bench items carry decisions/scores/reasons/card
    out = [r(items) for r in RULES]
    order = {"critical": 0, "major": 1, "minor": 2}
    return sorted([w for w in out if w.items], key=lambda w: (order[w.severity], -len(w.items)))


def attach_raw_reads(report: dict, traces: list[dict]) -> dict:
    by = {f"{t['dataset']}/{t['item']}": t for t in traces}
    for k, it in report["items"].items():
        t = by.get(k)
        if t:
            it["raw_reads"] = t.get("raw_reads", {})
            it["layout"] = t.get("layout")
    return report


def to_markdown(ws: list[Weakness], report_id: str) -> str:
    L = ["# Weakness register", "",
         f"Generated by `python -m phoenix diagnose` from bench report `{report_id}`. "
         "Do not edit by hand; edit the rules in `phoenix/diagnose.py`.", "",
         "| id | severity | count | weakness | hypothesis | examples |", "|---|---|---|---|---|---|"]
    for w in ws:
        ex = "<br>".join(w.items[:3])
        L.append(f"| {w.id} | {w.severity} | {len(w.items)} | {w.title} | {w.hypothesis or '–'} | {ex} |")
    L.append("")
    for w in ws:
        if w.detail:
            L.append(f"- **{w.id}**: {w.detail}")
    return "\n".join(L) + "\n"


def write_register(report_path: Path, traces_path: Path | None = None) -> Path:
    from . import monitor
    rep = read_json(report_path)
    if traces_path and Path(traces_path).exists():
        attach_raw_reads(rep, monitor.load(traces_path))
    ws = diagnose(rep)
    out = DOCS_DIR / "weaknesses.md"
    out.write_text(to_markdown(ws, rep["run_id"]), encoding="utf-8")
    return out
