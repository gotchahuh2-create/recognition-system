# ADR-0039: Card located primarily by its closed yellow frame (component fill + morphological opening 

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 39 (migrated)

## Decision

Card located primarily by its closed yellow frame (component fill + morphological opening + rectangle fit); layout ROI and a stage-1 title read only add agreement or a fallback

## Context and reason

146 of 207 cards in the dataset are "black soldermask, yellow/gold border frame"; the board's yellow lines are open (rectangularity 0.02 vs 0.91–0.98 for the card on both real photos); needs no calibration per board

## Evidence

real UNI-DS v8 photos: card found on both in ~25 ms
