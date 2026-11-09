# ADR-0054: "Text-likeness" = area of glyph-sized components chained into rows (similar height, small 

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 54 (migrated)

## Decision

"Text-likeness" = area of glyph-sized components chained into rows (similar height, small gap) or word-shaped; used to pick clusters, seeds, polarity, thresholds and text axis

## Context and reason

Plain size filters rated random noise blobs above real (merged) words and flipped the discriminant

## Evidence

test_stroke_score_prefers_text_over_noise
