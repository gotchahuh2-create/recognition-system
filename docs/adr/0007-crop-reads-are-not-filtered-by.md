# ADR-0007: Crop reads are not filtered by Tesseract confidence

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 7 (migrated)

## Decision

Crop reads are not filtered by Tesseract confidence

## Context and reason

With dictionaries off, Tesseract reports conf 0 for correct part numbers; the strict catalog match (0.90 / margin 0.05) is the gate

## Evidence

refine crops of hard_motion15 / hard_defocus7
