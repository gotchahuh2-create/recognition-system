# E-006: Socket occupancy heuristic v1 (colour + centre texture) vs v0

Hypothesis: H-004 · baseline report `20261109-111841-6ba4790-46b0dba7c0`

## Variant `heuristic_v1` — report `20261109-113712-6ba4790_dirty-efa4f7e683`

- gates: no_new_wrong=pass, no_lost_correct=pass, latency_within_budget=pass · decisions identical to baseline: yes
- new correct: 18 · lost correct: 0 · new wrong: 0 · wrong fixed: 0 · decisions changed: 0
- total processing time: 387.4 s -> 384.7 s (x0.993)
- verdict: passes gates and improves

New correct:

- `captures/20261105_144642_board_01.jpg:socket:mikroBUS 1`
- `captures/20261105_144642_board_01.jpg:socket:mikroBUS 2`
- `captures/20261105_144642_board_01.jpg:socket:mikroBUS 3`
- `captures/20261105_144642_board_01.jpg:socket:mikroBUS 4`
- `captures/20261105_144642_board_01.jpg:socket:mikroBUS 5`
- `captures/20261105_144827_board_01.jpg:socket:mikroBUS 1`
- `captures/20261105_144827_board_01.jpg:socket:mikroBUS 2`
- `captures/20261105_144827_board_01.jpg:socket:mikroBUS 3`
- `captures/20261105_144827_board_01.jpg:socket:mikroBUS 4`
- `captures/20261105_144827_board_01.jpg:socket:mikroBUS 5`
- `real3/uni-ds-v8_stm32f429zi_1.jpg:socket:mikroBUS 1`
- `real3/uni-ds-v8_stm32f429zi_1.jpg:socket:mikroBUS 2`
- `real3/uni-ds-v8_stm32f429zi_1.jpg:socket:mikroBUS 3`
- `real3/uni-ds-v8_stm32f429zi_1.jpg:socket:mikroBUS 4`
- `real3/uni-ds-v8_stm32f429zi_1.jpg:socket:mikroBUS 5`
- `real3/uni-ds-v8_stm32f429zi_2.jpg:socket:mikroBUS 1`
- `real3/uni-ds-v8_stm32f429zi_2.jpg:socket:mikroBUS 2`
- `real3/uni-ds-v8_stm32f429zi_2.jpg:socket:mikroBUS 3`

| dataset | base p95 ms | variant p95 ms | ratio |
|---|---|---|---|
| captures | 45669 | 45356 | 0.993 |
| real3 | 29632 | 30391 | 1.026 |
