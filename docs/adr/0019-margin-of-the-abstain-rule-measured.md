# ADR-0019: Margin of the abstain rule measured on re-ranked scores

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 19 (migrated)

## Decision

Margin of the abstain rule measured on re-ranked scores; strong OCR (≥90, number exact, unique) can replace the margin but never T_accept

## Context and reason

OCR evidence for the runner-up must reduce confidence; near-identical Clicks have tiny cosine gaps that only OCR separates

## Evidence

clicks/recognizer.py
