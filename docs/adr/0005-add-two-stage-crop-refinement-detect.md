# ADR-0005: Add two-stage crop refinement (detect → crop from full-res photo → enlarge to ~44 px glyph

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 5 (migrated)

## Decision

Add two-stage crop refinement (detect → crop from full-res photo → enlarge to ~44 px glyphs → denoise/deblur/binarize → Tesseract block mode, whitelist, no dictionaries); runs only when cheaper passes abstain

## Context and reason

Research brief: enlarge + clean + binarize small text, detection→recognition; synthetic MCU set 6/10 → 8/10, 0 wrong

## Evidence

eval/ocr_eval.py --configs baseline,full; vision/refine.py
