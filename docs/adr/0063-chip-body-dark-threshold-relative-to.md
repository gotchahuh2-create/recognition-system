# ADR-0063: Chip body "dark" threshold relative to the card (max(90, median+15))

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 63 (migrated)

## Decision

Chip body "dark" threshold relative to the card (max(90, median+15))

## Context and reason

p2 is over-exposed: chip body at gray 102, no chip found with the fixed 90

## Evidence

`card_reader.find_chip`
