# ADR-0046: A title read without a number ("CARDFORKINETIS") also admits the numbered layouts of that 

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 46 (migrated)

## Decision

A title read without a number ("CARDFORKINETIS") also admits the numbered layouts of that family at 0.9; one of them wins only with ≥ 2 distinctive pairs that CONTRADICT the title's layout (≥ 4 if that layout has no pin data)

## Context and reason

Small titles lose the digit (synthetic far scene read MCU CARD 2 for Kinetis as "for Kinetis"); but only 11 layouts have full pin maps, so "rare" pairs like SCL:PB8 are rare partly because data is missing — MCU CARD for STM32 has PB8 in its fingerprint and must not lose to MCU CARD 22

## Evidence

test_dropped_title_digit_recovered_from_labels, test_common_labels_never_flip_a_read_title
