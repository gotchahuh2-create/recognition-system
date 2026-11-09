# Algorithm efficiency report (2026-11-09)

Shareable version with charts: Claude doc "PHOENIX — Algorithm Efficiency Report"
(https://claude.ai/artifact/3jjLJ9tXGamsd19EkyGuVV). Reproduce: `python -m phoenix bench`,
`python -m phoenix experiment E-003`, `python scripts/phoenix_ocr_budget.py`.

**Time goes to text, not to images.** OCR calls plus fuzzy matching of what was read take 97-99 % of
every photo. A phone capture averages 38 s (p95 88 s) on a 2-core CPU with Tesseract. Preprocessing,
yellow-frame card search, layout fit and socket analysis take well under a second together.

## Latency today (reference run `20261109-102028-1fbbb0d-dda4621290`, v1, Tesseract 5.3.4, 2 cores)

| Set | n | mean | p50 | p95 | max | board step | MCU step | rest |
|---|---|---|---|---|---|---|---|---|
| Phone captures 2160x3840 | 10 | 38.5 s | 25.9 s | 88.5 s | 117.1 s | 6.7 s | 31.1 s | 0.6 s |
| Messenger photos 960x1280 | 3 | 32.3 s | 30.9 s | 36.8 s | 37.4 s | 16.0 s | 15.9 s | 0.4 s |
| Synthetic MCU cards | 36 | 14.4 s | 13.0 s | 17.9 s | 30.1 s | 4.5 s | 9.4 s | 0.4 s |
| Synthetic boards and chips | 16 | 4.9 s | 2.6 s | 17.2 s | 19.6 s | 1.9 s | 2.8 s | 0.1 s |

## Hotspots (cProfile run `20261109-095017-b6c586f-dda4621290`, same decisions)

`vision.textmatch.ocr_distance` (pure-Python weighted Levenshtein) ran 23,035,318 times for 65 photos:
610 s own time; fuzzy matching as a whole 757 s of 1,760 s profiled (43 %, an upper bound: profiling
inflates Python calls; OCR worker threads are invisible to cProfile). 82.5 % of the calls repeated a pair.

| Function | Calls | Cumulative s |
|---|---|---|
| `mcu_ocr.McuRecognizer.recognise` | 67 | 902 |
| `textmatch.ocr_ratio` -> `ocr_distance` | 23.0 M | 797 |
| `textmatch.NameIndex.score_all` | 26,420 | 743 |
| `card_ocr.CardRecognizer.evidence` | 106 | 657 |
| `ocr.read_text` / `TesseractBackend.read` | 351 / 3,849 | 573 / 626 |
| `board_ocr.BoardRecognizer.recognise` | 65 | 547 |
| `textmatch.best_substring_ratio` | 1.41 M | 538 |
| `card_text.CardTextMatcher.score` | 106 | 204 |
| `refine.refine` | 34 | 127 |

## Algorithm by algorithm

| Algorithm | Time / photo | Cost driver | Verdict |
|---|---|---|---|
| Preprocess | 0.1-0.4 s | pixels | efficient |
| Board name OCR + match | 1.9-16.0 s | 2 rotation pairs x 2 variants of OCR; every line segment x every alias x glued-substring search | expensive on real boards |
| Layout placement | 0.02-0.33 s | tokens x anchors, RANSAC | cheap; with v1 anchors rarely succeeds under Tesseract |
| Card location (yellow frame) | 0.03-0.08 s | components at <= 1600 px | efficient |
| Card read | in MCU step | 3 passes x 4 turns x 2 variants of OCR on 1800 px | OCR-bound |
| Card / layout matching | in MCU step | 62 layout titles + pin profiles x words read | matching-bound |
| MCU part matching + hi-res tiles | 2.8-31.1 s (MCU step) | 224 MCUs x candidates; 1400 px tiles x 4 turns when undecided | largest cost on captures |
| Refinement | when undecided | crops x variants, early stop | bounded |
| Faint clustering (opt-in) | +~3.3 s / chip (README) | per-pixel features, k-means, thresholds, OCR per variant | not default |
| Marking verifier (opt-in) | +2-6 s (README) | candidates x render x NCC | not default |
| Socket occupancy (new) | < 0.01 s / socket | colour + texture on 400 px crops | negligible |
| Geometry-first card search (new) | = frame search | frame search in one window | no measurable change |

## OCR engine cost (`scripts/phoenix_ocr_budget.py`, `runs/ocr_budget_v1.json`)

Fixed cost per Tesseract call (process start + model load): 0.082 s.

| Photo | Total | OCR calls | Sum of call time | Fixed start-up |
|---|---|---|---|---|
| Synthetic chip close-up | 2.5 s | 8 | 2.8 s | 0.7 s |
| Synthetic card, close | 14.7 s | 32 | 17.0 s | 2.6 s |
| Synthetic card, far (undecided) | 17.4 s | 106 | 20.7 s | 8.7 s |
| Messenger photo (undecided) | 37.4 s | 165 | 45.6 s | 13.5 s |
| Capture, board | 25.6 s | 91 | 32.0 s | 7.5 s |
| Capture, close-up | 25.9 s | 91 | 32.6 s | 7.5 s |

## Measured improvements (identical decisions)

| Experiment | Change | Total time | Decisions changed | Proposal |
|---|---|---|---|---|
| E-003 | memoised `ocr_distance` / `normalize` / `compact` | 1078 s -> 825 s (-23 %) on 65 photos | 0 | P-002 |
| E-005 | layouts v2 (Tesseract 4K anchors) | 481 s -> 372 s (-23 %) on 13 real photos | 0 | P-003 |

## Recommendations

| # | Change | Gain | Status |
|---|---|---|---|
| R1 | `text_match: exact_cached` | -23 %, exact | P-002, needs approval |
| R2 | `layout: v2` | -23 % on real photos, placement 1/13 -> 11/13 | P-003, needs approval |
| R3 | RapidOCR on the owner's PC, benchmark there | README 0.3: 2.6 s median on synthetic | run on PC |
| R4 | skip hi-res tiles + refinement when the card is < 800 px native | 106-165 -> ~30 OCR calls on undecided small photos (estimate) | code change + ADR |
| R5 | persistent Tesseract (C API backend) | ~0.08 s x calls = 7-13 s per real photo | new backend |
| R6 | exact compact-name lookup before fuzzy board scoring | board step 2-16 s, most photos decided on first hit (estimate) | code change + ADR |

## Caveats

Tesseract only (RapidOCR not installable in the build environment; the app's captures were recognised with
another engine and gave different answers). Two-core cloud CPU. Captures have no ground truth for
board/card/MCU yet.
