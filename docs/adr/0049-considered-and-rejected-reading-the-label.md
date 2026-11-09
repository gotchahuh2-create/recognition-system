# ADR-0049: Considered and rejected: reading the label boxes one by one (connected components → per-bo

- Date: 2026-11-04
- Status: Rejected
- Source: DECISIONS.md row 49 (migrated)

## Decision

Considered and rejected: reading the label boxes one by one (connected components → per-box OCR)

## Context and reason

At the real photos' 5 px label height no binarisation made the labels readable to Tesseract; per-box reads cost ~100 OCR calls per card. The blanked label image is one call per rotation and works when labels are ≥ ~15 px

## Evidence

prototype on tests/real
