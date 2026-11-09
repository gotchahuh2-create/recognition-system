# ADR-0052: Lighting flattened with a 1.6-glyph median per channel (not a Gaussian)

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 52 (migrated)

## Decision

Lighting flattened with a 1.6-glyph median per channel (not a Gaussian)

## Context and reason

A median follows broad glare and gradients but ignores strokes; a Gaussian/normalised-convolution background left the glare in the residual and k-means picked the glare ellipse as "text". 0.8 glyph was too small for bold dense marking (background followed the letters, residual inverted)

## Evidence

stage images 2c/4d, easy-chip check
