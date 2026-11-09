# ADR-0006: Global denoise / gamma lift / sharpen implemented but **off by default** (UI toggle "globa

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 6 (migrated)

## Decision

Global denoise / gamma lift / sharpen implemented but **off by default** (UI toggle "global enhance", CLI --global-enhance)

## Context and reason

On the synthetic set they *lowered* MCU accuracy of the whole-image pass 6/10 → 4/10 (NLM + gamma soften thin laser strokes); refinement already does these per crop

## Evidence

ablation run 3 Nov
