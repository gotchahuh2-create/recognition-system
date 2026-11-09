---
id: H-004
title: Colour shares in the layout's socket area separate empty from occupied mikroBUS sockets
status: supported (in-sample)
weakness: P3 (no socket stage)
experiment: E-002, E-005, E-006
---

## Claim

An empty socket area is dark solder mask + yellow silkscreen; a Click covers it with other colours.
`heuristic_v0` (phoenix/sockets.py): foreign share >= 0.30 -> occupied; <= 0.08 with silkscreen -> empty;
else unknown. Expect: every occupied socket on the real captures `occupied`, no socket called `empty`
when a Click is present.

## Test

E-002 on captures (board shots, 5 sockets each, all occupied by Clicks on 5 Nov). Missing: photos with
EMPTY sockets and dark-PCB Clicks — needed before promotion.

## Result

E-002 (v1 layouts): no socket areas (layout not placed). E-005 v0 with layouts v2: 15/35 sockets correct, 0 wrong, 20 unknown (all empty sockets). E-006 v1: 33/35 correct, 0 wrong, 2 unknown; held-out photo 3/5 correct, 2 unknown. Thresholds fitted on 30 sockets of one board with green Clicks: collect empty/dark-Click photos of other boards before trusting 'empty'. Proposal P-005.
