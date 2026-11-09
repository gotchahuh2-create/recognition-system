"""Result types shared by all the recognisers.

The field names match the setup manifest the rest of the tool expects:
{board, mcu, clicks: [{pid, name, slot, confidence}]}.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    key: str  # board key, MCU part number or card key
    name: str
    score: float  # match score from 0 to 1
    pid: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"key": self.key, "name": self.name, "pid": self.pid,
             "score": round(self.score, 3)}
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class Recognition:
    """Outcome of one recogniser on one photo."""

    kind: str  # "board", "mcu" or "mcu_card"
    candidates: list[Candidate]                # best first, max 3
    decision: Candidate | None  # None means we are not sure and the user must choose
    reason: str  # why we decided, or why we did not
    evidence: list[dict] = field(default_factory=list)  # OCR words that led to the result
    # for example part numbers that are not in the catalog
    raw_reads: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    # refine.CropRead objects, shown in the UI for debugging
    crops: list = field(default_factory=list, repr=False)
    # Page rotations we already ran OCR on (even the ones that found nothing),
    # so later steps don't read them again.
    rotations_read: list = field(default_factory=list, repr=False)
    # Only for mcu_card: which layout group the card belongs to (key, title,
    # the cards in it, why). We can know the group even when we can't tell
    # which exact card it is.
    group: dict | None = None

    @property
    def abstained(self) -> bool:
        return self.decision is None

    @property
    def confidence(self) -> float:
        return self.decision.score if self.decision else 0.0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "decision": self.decision.as_dict() if self.decision else None,
            "abstained": self.abstained,
            "reason": self.reason,
            "candidates": [c.as_dict() for c in self.candidates],
            "raw_reads": self.raw_reads,
            "evidence": self.evidence,
            "timings_ms": self.timings_ms,
            **({"crops": [c.as_dict() for c in self.crops]} if self.crops else {}),
            **({"group": self.group} if self.group else {}),
        }


def decide(cands: list[Candidate], accept: float, margin: float,
           what: str) -> tuple[Candidate | None, str]:
    """Pick the best candidate, or abstain if the top two are too close to call."""
    if not cands or cands[0].score <= 0:
        return None, f"no {what} text found"
    top = cands[0]
    second = cands[1].score if len(cands) > 1 else 0.0
    if top.score < accept:
        return None, f"best {what} match {top.score:.2f} < {accept:.2f}"
    if top.score - second < margin:
        return None, (f"ambiguous: {top.name} {top.score:.2f} vs "
                      f"{cands[1].name} {second:.2f}")
    return top, f"{top.name} matched {top.score:.2f} (margin {top.score - second:.2f})"
