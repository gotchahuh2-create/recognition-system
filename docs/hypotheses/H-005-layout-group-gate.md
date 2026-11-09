---
id: H-005
title: The reported layout group needs its own confidence gate
status: open
weakness: W-LAYOUT-GROUP
experiment: none yet (needs code in vision/card_ocr.py -> ADR)
---

## Claim

`manifest.mcu_card_layout` is shown as a fact whenever a title parses. With Tesseract the PIC32 real
photo reads `MCU CARD 3 for PIC` (family PIC, layout with 27 cards) instead of `for PIC32` (4 cards):
a wrong layout and a wrong option list for the user. Requiring agreement from a second signal (pin
labels, package size, port box) or a family-prefix check (`PIC` is a prefix of `PIC32` and the read is at
the token's end) before reporting the group removes this wrong answer.

## Test

Needs a code change in the protected MCU Reader (title -> group), so: ADR + reviewed change, then
bench gate on all datasets.
