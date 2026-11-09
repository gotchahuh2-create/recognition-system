# ADR-0031: MCU abstains with a "photo too small" reason when the card crop < 1000 px

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 31 (migrated)

## Decision

MCU abstains with a "photo too small" reason when the card crop < 1000 px

## Context and reason

Real photos: card 256–290 px, marking ~3 px — not readable by any OCR; better to ask for a close-up than guess from "MCU CARD for STM32"

## Evidence

tests/real
