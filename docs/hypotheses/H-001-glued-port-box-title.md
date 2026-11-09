---
id: H-001
title: A title read glued to the port box still names the layout family
status: open
weakness: W-GLUED-TITLE
experiment: none yet (needs a parser variant in code)
---

## Claim

On real close-ups the port box and the title family are sometimes read as one token
(`16BITPORTFORSTM32`, capture `20261105_153743_mcu_02`, chip 463 px). Splitting such a token into
port width + `for FAMILY` and admitting the numbered layouts of that family (as ADR-0046 already does
for `CARDFORKINETIS`) recovers the layout group, and with the chip read, the card.

## Why we believe it

Trace of the capture: the only card text read was `16BITPORTFORSTM32`; mcu_card_layout = null although
the port width and family were both in that token.

## Test

Needs code: a registered `title_parse` implementation in `phoenix/` that pre-splits glued tokens before
`vision.card_ocr` sees them. Datasets: captures, syn_cards (no regression). Gate: zero new wrong.

## Result

Not run yet.
