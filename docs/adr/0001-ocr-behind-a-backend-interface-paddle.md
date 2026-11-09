# ADR-0001: OCR behind a backend interface: paddle > rapidocr > tesseract

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 1 (migrated)

## Decision

OCR behind a backend interface: paddle > rapidocr > tesseract

## Context and reason

Plan picks PaddleOCR; Tesseract keeps the pipeline runnable where Paddle can't be installed; all Apache-2.0

## Evidence

vision/ocr.py
