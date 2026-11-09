# ADR-0048: No ROI / whole-photo fallback when the card was located with score ≥ 0.8 and its crop gave

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 48 (migrated)

## Decision

No ROI / whole-photo fallback when the card was located with score ≥ 0.8 and its crop gave nothing

## Context and reason

It would read the same pixels again, smaller; real photos: 34 → 31 s and 38 → 26 s (Tesseract, 2-core), same answers

## Evidence

Pipeline.CARD_SURE
