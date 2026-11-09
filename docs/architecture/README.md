# PHOENIX architecture

Status: implemented as of 2026-11-09 (phases 0-5 scaffolding). Audit and proposal:
[audit-2026-11-09.md](audit-2026-11-09.md). Decisions: ADR-0100..0106.

## Layers

```
Layer 2  phoenix/  engineering loop
         monitor -> diagnose -> experiments -> bench (evaluation) -> adapt (proposal)
                 -> validate (gates) -> owner approval -> config version + ADR -> memory (docs/)
                 ^ traces, reports                            | promoted config (config/active.yaml)
Layer 1  phoenix/stages.py   PhoenixPipeline(RunConfig): registry choices + param overrides
         preprocess -> board -> geometry -> localise -> recognise -> sockets -> manifest
                 | calls, never edits
Layer 0  vision/   protected core (v0.10): the six preserved components
```

## Stage sequence and where each stage lives

| # | Stage | Implementation(s) | Code | Registry key |
|---|---|---|---|---|
| 1 | preprocess | quality, deskew, CLAHE, warp | `vision/preprocess.py` | – |
| 2 | board | name OCR + IDF match; layout-anchor ID fallback | `vision/board_ocr.py`, `Pipeline.board_from_layouts` | `board_read`, `ocr_backend` |
| 3 | geometry | layout placement from silkscreen anchors (RANSAC) | `vision/layout.py` | `layout` |
| 4 | localise | MCU card: frame-first (protected) / geometry-first (experimental); socket ROIs from layout | `vision/card_locate.py`, `phoenix/localise.py` | `card_localise` |
| 5 | recognise | card read (page, labels, chip) + title/pin/chip/PID match; faint chip mode; verifier; refine | `vision/card_reader.py`, `card_ocr.py`, `mcu_ocr.py`, `faint/`, `refine.py`, `enhance.py` | `chip_read`, `upscaler`, `marking_verifier`, `refine` |
| 6 | sockets | occupancy empty / occupied / unknown (experimental); Clicks only for user boxes today | `phoenix/sockets.py`, `vision/clicks/` | `socket_occupancy` |
| 7 | manifest | `{board, mcu, mcu_card, mcu_card_layout, clicks, needs_user}` | `vision/pipeline.py:PhotoResult` | – |

Guided Scanning (`vision/app/`) sits in front of stage 1: it decides when the phone takes the photo.

## Module interfaces

| Interface | Producer -> consumer | Defined in |
|---|---|---|
| `Recognition` / `Candidate` / `decide()` | every recogniser -> pipeline, manifest | `vision/results.py` (unchanged) |
| `RunConfig {name, stages, params}` + `hash` | config files / experiments -> `PhoenixPipeline` | `phoenix/stages.py` |
| `config/registry.yaml` | people -> `stages.resolve` (validates choices) | – |
| `config/params.yaml` + `params.apply` / `apply_instance` | `params extract` -> runs | `phoenix/params.py` |
| Trace (one JSON object per run) | `monitor.trace_of` -> `runs/traces/*.jsonl` -> diagnose | `phoenix/monitor.py` |
| Dataset manifest | `datasets.build_*` -> bench | `phoenix/datasets.py` |
| Bench report (json + md) | `bench.run` -> compare, diagnose, experiments, STATE | `phoenix/bench.py` |
| Gate result `compare(cand, base)` | bench -> experiments, validate | `phoenix/bench.py` |
| Experiment spec / results | people -> `experiments.run` -> adapt | `phoenix/experiments.py` |
| Proposal | adapt -> owner -> validate.promote | `phoenix/adapt.py`, `phoenix/validate.py` |
| `SocketResult {slot, occupancy, score, reason, features}` | sockets stage -> trace / manifest | `phoenix/sockets.py` |

## Guardrails

See `CLAUDE.md` and ADR-0105. In short: `vision/` protected; experiments may only pick registered
implementations and override unbound parameters; zero new wrong answers; owner approval for every
promotion; rollback by config version or git tag.
