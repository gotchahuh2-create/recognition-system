# ADR-0026: Board identified by layout when the name OCR abstains (≥ 8 consistent anchors, ≥ 50 % of m

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 26 (migrated)

## Decision

Board identified by layout when the name OCR abstains (≥ 8 consistent anchors, ≥ 50 % of matched, 2× the runner-up)

## Context and reason

RapidOCR's angle classifier left an upside-down title unflipped ("LN 0Sv8") while flipping every other label

## Evidence

test_layout k90=2
