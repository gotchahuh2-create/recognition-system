# ADR-0013: RapidOCR detector resize `limit_type=max, 2016` instead of PP-OCR's `min, 736`

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 13 (migrated)

## Decision

RapidOCR detector resize `limit_type=max, 2016` instead of PP-OCR's `min, 736`

## Context and reason

Default blew a 120×500 crop up to 736×3070 (1.9 s per read, 72 s worst photo); max-2016 leaves pages and tiles unchanged

## Evidence

ocr_eval rapidocr: same 16/16 correct, median 5.3 → 2.6 s, worst 72 → 15 s
