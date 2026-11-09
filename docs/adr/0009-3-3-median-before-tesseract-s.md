# ADR-0009: 3×3 median before Tesseract's whole-image pass on noisy photos (σ > 6)

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 9 (migrated)

## Decision

3×3 median before Tesseract's whole-image pass on noisy photos (σ > 6)

## Context and reason

v0.1 took 55 s per dark phone photo (noise read as characters); now 2–14 s, same answers

## Evidence

hard_lowlight*, timings
