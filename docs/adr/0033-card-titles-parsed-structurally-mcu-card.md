# ADR-0033: Card titles parsed structurally (`MCU CARD [N] for FAMILY`, `SiBRAIN for FAMILY`), number 

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 33 (migrated)

## Decision

Card titles parsed structurally (`MCU CARD [N] for FAMILY`, `SiBRAIN for FAMILY`), number and family exact

## Context and reason

Whole-string fuzzy match scores "MCU CARD for STM32" vs "MCU CARD 2 for STM32" at 0.94 and "SiBRAIN for STM32" inside "…STM32L4" at 1.0

## Evidence

test_number_must_match, test_family_must_match
