# ADR-0014: Engines with an angle classifier (RapidOCR, Paddle) read one rotation per 0/180 and 90/270

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 14 (migrated)

## Decision

Engines with an angle classifier (RapidOCR, Paddle) read one rotation per 0/180 and 90/270 pair

## Context and reason

`handles_rotation` existed but was unused; 180° passes repeated the 0° pass

## Evidence

`ocr.effective_rotations`, test_clicks.py
