# ADR-0029: Weak layout fit (< 6 inliers) → read the skipped rotation once at ≤ 1400 px and refit

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 29 (migrated)

## Decision

Weak layout fit (< 6 inliers) → read the skipped rotation once at ≤ 1400 px and refit

## Context and reason

A similarity fit on 3 labels has no redundancy; full-res extra pass cost 9.9 s, 1400 px 2.7 s for 9+ labels

## Evidence

uni2 held-out photo
