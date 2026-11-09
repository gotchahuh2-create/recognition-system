# E-003: Memoised fuzzy text matching (exact acceleration)

Hypothesis: H-006 · baseline report `20261109-102028-1fbbb0d-dda4621290`

## Variant `exact_cached` — report `20261109-104041-8978102-524b7724b8`

- gates: no_new_wrong=pass, no_lost_correct=pass, latency_within_budget=pass · decisions identical to baseline: yes
- new correct: 0 · lost correct: 0 · new wrong: 0 · wrong fixed: 0 · decisions changed: 0
- total processing time: 1077.6 s -> 825.2 s (x0.766)
- verdict: passes gates and improves (same answers, faster)

| dataset | base p95 ms | variant p95 ms | ratio |
|---|---|---|---|
| syn_cards | 17863 | 13085 | 0.733 |
| real3 | 36754 | 29147 | 0.793 |
| captures | 88450 | 56330 | 0.637 |
| syn_ocr | 17229 | 15730 | 0.913 |
