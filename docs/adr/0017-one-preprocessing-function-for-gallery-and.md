# ADR-0017: One preprocessing function for gallery and query (pad to square, no stretch), versioned wi

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 17 (migrated)

## Decision

One preprocessing function for gallery and query (pad to square, no stretch), versioned with the model; galleries refuse to load against another version

## Context and reason

Report §2: mismatches must fail loudly instead of silently lowering accuracy

## Evidence

clicks/preprocess.py, GalleryMismatch
