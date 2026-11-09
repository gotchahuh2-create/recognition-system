# Working on this repository (PHOENIX)

Read `docs/STATE.md` first: it is generated and says what works, what is measured and what is open.

## What this is

MikroE hardware recognition: photo -> board -> MCU card / MCU -> mikroBUS sockets (-> Clicks), feeding a
NECTO Studio setup manifest. `vision/` is the recognition core (v0.10). `phoenix/` is the engineering layer
that measures, diagnoses, experiments and proposes changes around it.

## Rules

1. **`vision/` is protected.** Do not edit it as part of an experiment. A change to `vision/` is a normal,
   reviewed code change with an ADR (`docs/adr/TEMPLATE.md`) and a full bench + test run.
2. **Never remove or replace a preserved component** (Standard Board OCR, MCU Reader, pixel-by-pixel
   clustering `vision/faint`, Resolution Booster, MCU and Socket Identifiers, Guided Scanning) without the
   owner's explicit approval. Their status is `protected` in `config/registry.yaml`.
3. **Order of recognition:** board first, then MCU card and sockets localised from the board's layout
   geometry, then region-specific recognition. Keep uncertainty: every field abstains rather than guesses
   (`vision/results.decide`); sockets are `empty` / `occupied` / `unknown`.
4. **No autonomous code modification.** Improvements go hypothesis -> experiment -> proposal -> human
   approval -> new config version (`docs/adr/0105-*`).
5. **Zero new wrong answers** is the first gate for any change (`phoenix.bench.compare`).
6. Thresholds: change them through `config/params.yaml` overrides in an experiment, not in code.
   `bound` parameters need a code change + ADR.
7. `vision/camera/` and `vision/app.py` are legacy (ADR-0101): do not extend them.

## Commands

```bash
python -m pytest -q                                   # unit + smoke tests (vision + phoenix)
python -m phoenix bench --datasets syn_ocr,real3       # benchmark the active config
python -m phoenix gate <base_run_id> <cand_run_id>     # regression gates between two reports
python -m phoenix diagnose                             # docs/weaknesses.md from the reference report
python -m phoenix experiment E-001                     # run an experiment spec -> results + proposals
python -m phoenix promote P-001 --approved-by "Name"   # only with the owner's approval
python -m phoenix rollback                             # back to the previous config version
python -m phoenix state                                # regenerate docs/STATE.md
python -m phoenix params check                         # registry vs code drift
```

## Where things are

| What | Where |
|---|---|
| Architecture, interfaces | `docs/architecture/README.md` |
| Decisions | `docs/adr/` (0001-0081 migrated from `DECISIONS.md`, PHOENIX from 0100) |
| Hypotheses / experiments / proposals | `docs/hypotheses/`, `docs/experiments/`, `docs/proposals/` |
| Weaknesses | `docs/weaknesses.md` (generated) |
| Efficiency study | `docs/reports/efficiency.md` |
| Config | `config/params.yaml`, `config/registry.yaml`, `config/versions/`, `config/active.yaml` |
| Benchmarks | `bench/datasets/` (manifests), `bench/reports/`, `bench/baselines.yaml` |
| Run traces (git-ignored) | `runs/traces/` |

## Environment notes

Two-core CPU timings are the reference. Tesseract 5 is enough for the tests; RapidOCR enables the
real-photo test (`tests/test_real_photos.py`). Keep only `opencv-python-headless` (README 0.3 note).
