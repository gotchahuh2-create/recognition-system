---
id: H-007
title: A card decision must be able to carry the decided MCU
status: supported
weakness: W-WRONG
experiment: E-004
---

## Claim

The reference benchmark's only wrong answer (syn_cards `card_mid_mcu-card-3-for-stm32-stm32f303rc`)
comes from `McuRecognizer.decide_card`: the package filter dropped every card of the read layout
(`MCU CARD 3 for STM32`, measured chip/card ratio 0.30 -> "144-pin") and left one empty card of a
look-alike layout (EasyMx PRO v7 Stellaris LM4F), which was accepted as "only card with this layout"
although the MCU was read as STM32F303RC. A guard that withdraws such a card removes the wrong answer
and loses no correct one.

## Root causes (for a later code change in the protected MCU Reader, ADR needed)

1. `decide_card` step "single-card layout" runs before the MCU/carrier check.
2. `package.filter_cards` is "soft" for the pooled list, not per read group: it can empty the top group
   while look-alike groups keep cards. (The synthetic renderer also draws the chip larger than a real
   LQFP-64, which is why the ratio was 0.30; real photos would rarely trigger it, but the logic flaw stands.)

## Test

E-004: `consistency_guard: on` vs reference. Expected: wrong 1 -> 0 on syn_cards, nothing else changes.

## Result

E-004 (exact replay): the one wrong card on syn_cards is withdrawn, nothing else changes. Proposal P-001. The root causes in McuRecognizer.decide_card / package.filter_cards still need a reviewed code change (ADR).
