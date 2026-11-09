# ADR-0050: Faint chip markings: new modular pipeline `vision/faint` (region → RGB/HSV/LAB features → 

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 50 (migrated)

## Decision

Faint chip markings: new modular pipeline `vision/faint` (region → RGB/HSV/LAB features → multi-stage clustering → many thresholds → gentle cleanup → variants → OCR → fusion), opt-in via `--chip-mode faint\|both`; default stays `classic`

## Context and reason

Experimental; it is slower (~3 s vs ~1–2 s per chip) and only measured on synthetic chips so far

## Evidence

eval/faint_eval.md
