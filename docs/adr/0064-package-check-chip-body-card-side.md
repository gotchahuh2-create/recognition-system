# ADR-0064: Package check: chip body / card side -> QFP body -> pin count

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 64 (migrated)

## Decision

Package check: chip body / card side -> QFP body -> pin count; cards with an impossible pin count leave the group (soft: never empties it)

## Context and reason

"SiBRAIN for PIC32" is the alias of two layouts (100-pin MX, 144-pin MZ): p1 10 cards -> 4; ratio 0.298-0.314 on all three photos vs 0.22 expected for 100-pin. CARD_SIDE_MM 65 is [ASSUMPTION] from these photos

## Evidence

`vision/package.py`
