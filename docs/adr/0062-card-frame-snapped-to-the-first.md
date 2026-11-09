# ADR-0062: Card frame snapped to the first CONTINUOUS yellow line inside the frame candidate (inner o

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 62 (migrated)

## Decision

Card frame snapped to the first CONTINUOUS yellow line inside the frame candidate (inner one when every side shows outline + gap + frame)

## Context and reason

p2: card frame touches UNI-DS v8's socket outline / label strip, one component, crop 10-30 px too big, title not read; snapped 270x286 -> 257x253, title read

## Evidence

`card_locate._snap_to_frame`, test_lowres_cards
