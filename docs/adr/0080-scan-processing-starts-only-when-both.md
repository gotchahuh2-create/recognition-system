# ADR-0080: Scan processing starts only when both photos are in

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 80 (migrated)

## Decision

Scan processing starts only when both photos are in; board photo: fast board-only pass (2000 px, no MCU / card / refine / layout); close-up: full MCU pass with board hint; full pass on the board photo only as a fallback

## Context and reason

Asked: board identification is easy, do not spend time on it; take both photos first. 1600 px missed the name on a real 4K iPhone still, 2000 px read it (5 s vs 20 s full)

## Evidence

test_scan_processes_only_after_both_photos, e2e with fake camera
