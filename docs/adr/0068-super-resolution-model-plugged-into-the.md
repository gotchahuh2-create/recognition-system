# ADR-0068: Super-resolution model plugged into the faint chip stage (`region.rectify(..., upscaler=)`

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 68 (migrated)

## Decision

Super-resolution model plugged into the faint chip stage (`region.rectify(..., upscaler=)`: warp at native size, SR enlarges, then colour features)

## Context and reason

Asked for; not measurable in the build environment (model download blocked) — falls back to bicubic. SR cannot restore glyphs that are 2 px wide; judge it with `eval/synth_eval.py` / real photos at 150-250 px chips

## Evidence

`python scripts/get_sr_models.py`, `--upscaler espcn --chip-mode faint`
