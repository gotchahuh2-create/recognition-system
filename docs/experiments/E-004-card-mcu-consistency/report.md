# E-004: Withdraw a card decision that cannot carry the decided MCU

Hypothesis: H-007 · baseline report `20261109-102028-1fbbb0d-dda4621290`

## Variant `guard_on` — report `20261109-104030-489a8ed_dirty-cea865c933-replay`

- gates: no_new_wrong=pass, no_lost_correct=pass, latency_within_budget=pass, identical_decisions=FAIL
- new correct: 0 · lost correct: 0 · new wrong: 0 · wrong fixed: 1 · decisions changed: 1
- verdict: passes gates and improves

Wrong answers fixed:

- `syn_cards/card_mid_mcu-card-3-for-stm32-stm32f303rc.jpg:mcu_card`

| dataset | base p95 ms | variant p95 ms | ratio |
|---|---|---|---|
| syn_ocr | 17229 | 17229 | 1.0 |
| real3 | 36754 | 36754 | 1.0 |
| captures | 88450 | 88450 | 1.0 |
| syn_cards | 17863 | 17863 | 1.0 |
