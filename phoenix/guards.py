"""Consistency guards: post-checks that can only turn an answer into an abstention.

A guard never creates or changes an answer; it withdraws one that contradicts
another decided field and says why (the field then goes to ``needs_user``).
That keeps the abstain-first promise even when two stages disagree.

``card_carries_mcu`` (hypothesis H-007): the reference benchmark found one
wrong card (syn_cards ``card_mid_mcu-card-3-for-stm32-stm32f303rc``): the MCU
was read correctly as STM32F303RC, but the package-size filter dropped every
card of the read layout and left a single *empty* card of a look-alike layout
(EasyMx PRO v7 Stellaris LM4F), which ``McuRecognizer.decide_card`` then
accepted as "only card with this layout" without checking that it can carry
the decided MCU. The guard withdraws a card decision whose card cannot carry
the decided MCU.

Registry stage ``consistency_guard``: ``off`` (v0.10) | ``on`` (experimental).
"""

from __future__ import annotations

from vision.textmatch import compact


def card_carries_mcu(result, catalog) -> str | None:
    card, mcu = result.mcu_card, result.mcu
    if card is None or mcu is None or card.decision is None or mcu.decision is None:
        return None
    c = next((x for x in catalog.mcu_cards if x.key == card.decision.key), None)
    if c is None:
        return None
    parts = {compact(p) for p in (c.mcus or [])} | ({compact(c.mcu)} if c.mcu else set())
    if compact(mcu.decision.key) in parts:
        return None
    note = (f"guard: card {c.key} cannot carry the decided MCU {mcu.decision.key} "
            f"({_what(c)})"
            " - card left to the user")
    card.decision = None
    card.reason = f"{card.reason}; {note}"
    return note


def _what(c) -> str:
    if not c.empty:
        return f"it carries {c.mcu}"
    return ("empty card for " + ", ".join(c.mcus[:3])) if c.mcus else \
        "empty card, no compatible MCU listed"


GUARDS = (card_carries_mcu,)


def apply_all(result, catalog) -> list[str]:
    return [n for g in GUARDS if (n := g(result, catalog))]


def replay(report: dict, catalog) -> dict:
    """Apply the guards to a stored bench report's decisions (guards only read decisions).

    Returns a copy whose items carry the guarded decisions and a ``guards`` note; scores are
    recomputed by ``bench.rescore``. Exactly equivalent to a re-run with the guard on, because a
    guard depends on nothing but the final decisions.
    """
    import copy
    from types import SimpleNamespace as NS
    from vision.results import Candidate, Recognition
    rep = copy.deepcopy(report)
    for key, it in rep["items"].items():
        d = it["decisions"]
        mk = lambda kind, k: Recognition(kind, [], Candidate(k, k, 1.0) if k else None, "")
        res = NS(mcu=mk("mcu", d.get("mcu")), mcu_card=mk("mcu_card", d.get("mcu_card")))
        notes = [n for g in GUARDS if (n := g(res, catalog))]
        if notes:
            d["mcu_card"] = None
            it["guards"] = notes
    return rep
