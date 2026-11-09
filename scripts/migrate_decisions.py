"""One-off: turn the DECISIONS.md table into one ADR file per row (docs/adr/).

The table stays the historical record (frozen); new decisions are written as ADR
files directly (docs/adr/TEMPLATE.md). Re-running overwrites only the migrated
ADRs 0001..N and docs/adr/README.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(text: str, n: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:n]) or "decision"


def rows(md: str):
    for line in md.splitlines():
        if not line.startswith("| 20"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) >= 4:
            yield cells[0], cells[1], cells[2], " | ".join(cells[3:])


def main() -> int:
    src = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    out = ROOT / "docs" / "adr"
    out.mkdir(parents=True, exist_ok=True)
    index = ["# Architecture decision records", "",
             "ADR-0001..%04d were migrated from `DECISIONS.md` (frozen, 2026-11-09). "
             "New decisions: copy `TEMPLATE.md` to the next number.", "",
             "| ADR | Date | Status | Decision |", "|---|---|---|---|"]
    n = 0
    for n, (date, decision, reason, evidence) in enumerate(rows(src), 1):
        status = "Rejected" if re.match(r"(?i)considered and rejected", decision) else "Accepted"
        name = f"{n:04d}-{slug(decision)}.md"
        (out / name).write_text(
            f"# ADR-{n:04d}: {decision.split(';')[0][:90]}\n\n"
            f"- Date: {date}\n- Status: {status}\n- Source: DECISIONS.md row {n} (migrated)\n\n"
            f"## Decision\n\n{decision}\n\n## Context and reason\n\n{reason}\n\n"
            f"## Evidence\n\n{evidence or '—'}\n", encoding="utf-8")
        index.append(f"| [{n:04d}]({name}) | {date} | {status} | {decision[:110]} |")
    index += ["", "## PHOENIX decisions (0100+)", "", "| ADR | Title |", "|---|---|"]
    for f in sorted(out.glob("01[0-9][0-9]-*.md")):
        title = f.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
        index.append(f"| [{f.name[:4]}]({f.name}) | {title} |")
    index += ["", "New decisions: copy `TEMPLATE.md` to the next free number from 0100 up."]
    (out / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"{n} ADRs written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
