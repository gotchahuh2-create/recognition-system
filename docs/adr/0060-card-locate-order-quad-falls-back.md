# ADR-0060: `card_locate.order_quad` falls back to angle ordering when the sum/diff rule returns a cor

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 60 (migrated)

## Decision

`card_locate.order_quad` falls back to angle ordering when the sum/diff rule returns a corner twice

## Context and reason

A QFP at exactly 45° tied the rule: chip close-up warped from a degenerate quad (seen on tests/real photo 1); other quads unchanged

## Evidence

test_card_order_quad_keeps_four_corners_at_45_degrees
