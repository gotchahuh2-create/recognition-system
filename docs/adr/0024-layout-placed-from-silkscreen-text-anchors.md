# ADR-0024: Layout placed from silkscreen text anchors already read by stage 1 (RANSAC homography ≥ 6 

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 24 (migrated)

## Decision

Layout placed from silkscreen text anchors already read by stage 1 (RANSAC homography ≥ 6 / similarity ≥ 3), outline quad only as fallback

## Context and reason

No extra OCR, works with the board at any 90° turn, partly occluded, on cluttered desks where outline detection fails

## Evidence

test_layout: 0/90/180/270° + perspective
