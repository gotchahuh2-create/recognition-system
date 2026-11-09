# ADR-0053: Robust z-scores get a floor (1 level raw, 0.05 smoothed)

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 53 (migrated)

## Decision

Robust z-scores get a floor (1 level raw, 0.05 smoothed)

## Context and reason

JPEG chroma is constant over 8×8 blocks, MAD ~0, the a/b/S channels exploded to huge z and dominated clustering

## Evidence

stage images 3_hsv_smooth_S
