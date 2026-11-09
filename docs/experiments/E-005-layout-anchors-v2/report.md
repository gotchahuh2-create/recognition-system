# E-005: Re-calibrated UNI-DS v8 layout anchors (engine- and resolution-independent placement)

Hypothesis: H-008 · baseline report `20261109-102028-1fbbb0d-dda4621290`

## Variant `layout_v2` — report `20261109-111228-6ba4790-4f71053050`

- gates: no_new_wrong=pass, no_lost_correct=pass, latency_within_budget=pass · decisions identical to baseline: yes
- new correct: 0 · lost correct: 0 · new wrong: 0 · wrong fixed: 0 · decisions changed: 0
- total processing time: 481.4 s -> 372.4 s (x0.774)
- verdict: passes gates and improves (same answers, faster)

| dataset | base p95 ms | variant p95 ms | ratio |
|---|---|---|---|
| real3 | 36754 | 30943 | 0.842 |
| captures | 88450 | 43726 | 0.494 |

## Variant `layout_v2_geometry_sockets` — report `20261109-111841-6ba4790-46b0dba7c0`

- gates: no_new_wrong=pass, no_lost_correct=pass, latency_within_budget=pass · decisions identical to baseline: no
- new correct: 15 · lost correct: 0 · new wrong: 0 · wrong fixed: 1 · decisions changed: 6
- total processing time: 481.4 s -> 387.4 s (x0.805)
- verdict: passes gates and improves

New correct:

- `captures/20261105_151336_board_01.jpg:socket:mikroBUS 1`
- `captures/20261105_151336_board_01.jpg:socket:mikroBUS 2`
- `captures/20261105_151336_board_01.jpg:socket:mikroBUS 3`
- `captures/20261105_151336_board_01.jpg:socket:mikroBUS 4`
- `captures/20261105_151336_board_01.jpg:socket:mikroBUS 5`
- `captures/20261105_153735_board_01.jpg:socket:mikroBUS 1`
- `captures/20261105_153735_board_01.jpg:socket:mikroBUS 2`
- `captures/20261105_153735_board_01.jpg:socket:mikroBUS 3`
- `captures/20261105_153735_board_01.jpg:socket:mikroBUS 4`
- `captures/20261105_153735_board_01.jpg:socket:mikroBUS 5`
- `real3/uni-ds-v8_pic32mz2048efh144_1.jpg:socket:mikroBUS 1`
- `real3/uni-ds-v8_pic32mz2048efh144_1.jpg:socket:mikroBUS 2`
- `real3/uni-ds-v8_pic32mz2048efh144_1.jpg:socket:mikroBUS 3`
- `real3/uni-ds-v8_pic32mz2048efh144_1.jpg:socket:mikroBUS 4`
- `real3/uni-ds-v8_pic32mz2048efh144_1.jpg:socket:mikroBUS 5`

Wrong answers fixed:

- `real3/uni-ds-v8_pic32mz2048efh144_1.jpg:mcu_card_layout`

| dataset | base p95 ms | variant p95 ms | ratio |
|---|---|---|---|
| real3 | 36754 | 29632 | 0.806 |
| captures | 88450 | 45669 | 0.516 |
