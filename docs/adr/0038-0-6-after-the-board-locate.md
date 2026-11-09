# ADR-0038: 0.6: after the board, locate the MCU card itself, rectify it out of the ORIGINAL photo and

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 38 (migrated)

## Decision

0.6: after the board, locate the MCU card itself, rectify it out of the ORIGINAL photo and zoom it (long side → 1800 px, ×4 max), then read it on its own

## Context and reason

The 0.5 card step only had a crop for boards with a calibrated layout (UNI-DS v8) and read it once; every other board got the whole-photo pass, where card labels are a few px among hundreds of board labels

## Evidence

vision/card_locate.py, tests/test_card_stage.py
