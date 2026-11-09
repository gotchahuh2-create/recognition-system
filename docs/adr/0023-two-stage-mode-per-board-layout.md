# ADR-0023: Two-stage mode: per-board layout (`data/layouts.json`, board-relative 0–1, board upright)

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 23 (migrated)

## Decision

Two-stage mode: per-board layout (`data/layouts.json`, board-relative 0–1, board upright); MCU read only from the rectified MCU-card ROI

## Context and reason

Whole-photo MCU pass reads every chip and label on the board (decoy chips, noise text) and tiles/refines the whole photo; the card position on a given board is fixed

## Evidence

tests/test_layout.py: decoy STM32F407ZGT6 outside the card competes without layout, ignored with it
