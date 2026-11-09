# ADR-0010: Hi-res tile tokens mapped into the processed frame

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 10 (migrated)

## Decision

Hi-res tile tokens mapped into the processed frame

## Context and reason

They were left in original-photo coordinates, so evidence boxes / glyph_px were wrong for hi-res reads

## Evidence

vision/mcu_ocr.py
