# ADR-0061: The three real UNI-DS v8 photos (960x1280, no EXIF, progressive JPEG q~70 = messenger copi

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 61 (migrated)

## Decision

The three real UNI-DS v8 photos (960x1280, no EXIF, progressive JPEG q~70 = messenger copies) are NOT asked to yield the exact card: chip marking ~3.5 px high, ~2 px per glyph; layout group + package are required instead, exact card only from a larger photo

## Context and reason

Faint-pipeline maps show the PIC32 logo and three blob rows on p1, only the ST / ARM logos on p2/p3; the resolution-model verifier ranks the TRUE card last on p1 and a wrong card first on p2/p3 (margins 0.001-0.017)

## Evidence

tests/real, eval/synth_eval_synthetic.md
