---
id: H-002
title: Geometry-first card localisation is at least as good as frame-first when a layout is placed
status: supported
weakness: brief requirement (board -> geometry -> region)
experiment: E-001, E-005
---

## Claim

Searching for the card frame only inside the layout's MCU-card window (grown 15 %) gives the same or
better card / MCU decisions as the whole-photo frame search, and is not slower, on boards with a layout.

## Test

E-001: `card_localise: geometry_first` vs baseline on real3 + captures (UNI-DS v8, the only board with a
layout) and syn_cards (no layout: must be identical).

## Result

E-001 (v1 layouts) no effect: the layout was placed on 1/13 real photos. With layouts v2 (E-005, variant 2): geometry-first on 11/13 placements, 0 new wrong, the wrong PIC32 layout group on real3 became an abstention, 5 layout decisions on unlabelled captures changed (judge after labelling). Proposal P-005 (with layout v2 + sockets v1).
