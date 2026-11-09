# ADR-0044: Full-text match: signal:pin pairs (1.0), pins (0.5), other strings (0.8), each × IDF over 

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 44 (migrated)

## Decision

Full-text match: signal:pin pairs (1.0), pins (0.5), other strings (0.8), each × IDF over layouts; pins / pairs a fully-mapped layout lacks count against it

## Context and reason

Pairs like "INT:PD14" are how the card prints its labels and are near-unique; "SCK:PA5" is on most STM32 cards — IDF keeps it cheap

## Evidence

tests/test_card_stage.py
