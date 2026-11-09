# ADR-0032: Card title identifies a layout group

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 32 (migrated)

## Decision

Card title identifies a layout group; the card is decided by chip marking, PID label, or a single-card group — never from a shared title

## Context and reason

Dataset: one title on up to 21 cards; title OCR alone cannot separate them

## Evidence

mikroe_mcu_cards_ocr/index.json, tests/test_cards.py
