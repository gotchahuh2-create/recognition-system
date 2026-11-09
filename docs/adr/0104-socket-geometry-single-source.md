# ADR-0104: Socket geometry comes from `data/layouts.json`; `data/sockets.csv` is derived

- Date: 2026-11-09
- Status: Accepted

## Context

Two sources described mikroBUS sockets: `layouts.json` (boxes, used to crop ROIs) and `sockets.csv`
(centres, used by Click slot assignment, empty).

## Decision

`layouts.json` is the source of truth. `sockets.csv` is regenerated from it by
`phoenix.sockets.derive_sockets_csv` and must not be edited by hand. New boards get sockets by
calibrating their layout (`python -m vision.layout calibrate ...`), or from the product database once
it is provided.
