# ADR-0074: Board shot first

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 74 (migrated)

## Decision

Board shot first; MCU close-up only while the MCU or the card is still open; close-up recognised with `Pipeline.run(board_hint=)` (used only when its own board read abstains, no layout placement)

## Context and reason

Whole-board photos rarely give a 250 px chip; a close-up loses the board name

## Evidence

e2e: board shot decided MCU + card -> no close-up asked
