# ADR-0076: `card_reader.find_chip` accepts a square chip body whose convex hull is a clean rectangle 

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 76 (migrated)

## Decision

`card_reader.find_chip` accepts a square chip body whose convex hull is a clean rectangle (hull/rect >= 0.9, area/hull >= 0.6) although its mask fill is < 0.75

## Context and reason

On a sharp close-up the legible marking notched the dark+smooth mask: fill 0.7499 -> no chip, MCU abstained on a 319 px chip

## Evidence

close-up capture: abstain -> STM32F215RE + card; synthetic card set (36) and the 3 real photos: identical outcomes before/after
