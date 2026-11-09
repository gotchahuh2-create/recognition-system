# ADR-0100: PHOENIX is a layer around `vision/`, which stays protected

- Date: 2026-11-09
- Status: Accepted (approved by the project owner, 2026-11-09)

## Context

The audit (docs/architecture/audit-2026-11-09.md) found a mature, evidence-driven recognition core with
zero wrong answers on every recorded benchmark, but no version control, no labelled real photos, scattered
tuning constants and no socket stage.

## Decision

- `vision/` is Layer 0 and is not edited by PHOENIX. Its six preserved components (Standard Board OCR,
  MCU Reader, pixel-by-pixel clustering, Resolution Booster, MCU and Socket Identifiers, Guided Scanning)
  are registered as `protected` in `config/registry.yaml`.
- Layer 1 (`phoenix/stages.py`) builds the pipeline from a configuration: registered implementations per
  stage + parameter overrides. New algorithms live in `phoenix/` (or a reviewed change to `vision/`) and
  enter as `experimental`.
- Layer 2 (`phoenix/`) adds monitoring, diagnostics, experiments, evaluation, adaptation, memory,
  validation/recovery.

## Consequences

Every behaviour change is a configuration change with evidence, or a reviewed code change with an ADR.
Recognition works exactly as before when `config/active.yaml` points at `v1`.

## Rollback

Delete nothing: `python -m vision ...` still runs the core directly, without PHOENIX.
