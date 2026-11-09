# PHOENIX state

Generated 2026-11-09 11:45 by `python -m phoenix state` at git `6ba4790+dirty`. Read this first; details are linked.

## Configuration

- Active config: **v1** (last known good: v1); history in `config/active.yaml`.
- Parameters: 105 in `config/params.yaml` (14 bound, need a code change to tune).
- Stage choices and their status: `config/registry.yaml`.

## Reference benchmark

Report `20261109-102028-1fbbb0d-dda4621290` (backend `tesseract`).

| dataset | field | correct | wrong | abstain | n |
|---|---|---|---|---|---|
| syn_ocr | board | 6 | 0 | 0 | 6 |
| syn_ocr | mcu | 8 | 0 | 2 | 10 |
| syn_cards | mcu | 25 | 0 | 11 | 36 |
| syn_cards | mcu_card | 24 | 1 | 11 | 36 |
| syn_cards | mcu_card_layout | 36 | 0 | 0 | 36 |
| real3 | board | 3 | 0 | 0 | 3 |
| real3 | mcu | 0 | 0 | 3 | 3 |
| real3 | mcu_card | 0 | 0 | 3 | 3 |
| real3 | mcu_card_layout | 0 | 1 | 2 | 3 |
| captures | board | 0 | 0 | 0 | 0 (+10 unverified, 10 agree) |
| captures | mcu | 0 | 0 | 0 | 0 (+5 unverified, 0 agree) |
| captures | mcu_card | 0 | 0 | 0 | 0 (+4 unverified, 0 agree) |
| captures | mcu_card_layout | 0 | 0 | 0 | 0 (+5 unverified, 0 agree) |

## Hypotheses

| id | status | title | weakness | experiment |
|---|---|---|---|---|
| H-001 | open | A title read glued to the port box still names the layout family | W-GLUED-TITLE | none yet (needs a parser variant in code) |
| H-002 | supported | Geometry-first card localisation is at least as good as frame-first when a layout is placed | brief requirement (board -> geometry -> region) | E-001, E-005 |
| H-003 | blocked | ESPCN/EDSR super-resolution reads more chip markings than bicubic at 150-250 px chips | W-SMALL-CARD | none |
| H-004 | supported (in-sample) | Colour shares in the layout's socket area separate empty from occupied mikroBUS sockets | P3 (no socket stage) | E-002, E-005, E-006 |
| H-005 | open | The reported layout group needs its own confidence gate | W-LAYOUT-GROUP | none yet (needs code in vision/card_ocr.py -> ADR) |
| H-006 | supported | The same answers in less time | W-SLOW | E-003 (+ docs/reports/efficiency.md) |
| H-007 | supported | A card decision must be able to carry the decided MCU | W-WRONG | E-004 |
| H-008 | supported | Layout anchors read by the engine and resolution in use make geometry-first localisation possible | W-NO-LAYOUT | E-005 |

## Experiments

- **E-001** Geometry-first card localisation (H-002): geometry_first: passes
- **E-002** Socket occupancy heuristic v0 (H-004): heuristic_v0: passes
- **E-003** Memoised fuzzy text matching (exact acceleration) (H-006): exact_cached: passes+improves
- **E-004** Withdraw a card decision that cannot carry the decided MCU (H-007): guard_on: passes+improves
- **E-005** Re-calibrated UNI-DS v8 layout anchors (engine- and resolution-independent placement) (H-008): layout_v2: passes+improves; layout_v2_geometry_sockets: passes+improves
- **E-006** Socket occupancy heuristic v1 (colour + centre texture) vs v0 (H-004): heuristic_v1: passes+improves

## Proposals

- **P-001** pending: {'stages': {'consistency_guard': 'on'}, 'params': {}} (from E-004)
- **P-002** pending: {'stages': {'text_match': 'exact_cached'}, 'params': {}} (from E-003)
- **P-003** pending: {'stages': {'layout': 'v2'}, 'params': {}} (from E-005)
- **P-004** superseded: {'stages': {'layout': 'v2', 'card_localise': 'geometry_first', 'socket_occupancy': 'heuristic_v0'}, 'params': {}} (from E-005)
- **P-005** pending: {'stages': {'layout': 'v2', 'card_localise': 'geometry_first', 'socket_occupancy': 'heuristic_v1'}, 'params': {}} (from E-006)

## Where to look

- Weaknesses: `docs/weaknesses.md` · Decisions: `docs/adr/README.md` · Architecture: `docs/architecture/README.md` · Efficiency: `docs/reports/efficiency.md`
- Rules for working here: `CLAUDE.md`
