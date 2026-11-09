# ADR-0103: Benchmarks are manifests; abstaining is never wrong; unverified data only measures agreement

- Date: 2026-11-09
- Status: Accepted

## Decision

- Datasets are JSON manifests in `bench/datasets/` (file, SHA-256, truth). Images stay out of git;
  synthetic sets regenerate deterministically into `runs/data/`.
- Per field: correct / wrong / abstain. Truth `"none"` means the right answer is to abstain.
- `mcu_card_layout` is scored as an answer: the manifest presents it as a fact.
- Captures without a person's confirmation (`verified: false`) carry `provisional` labels (today's
  predictions) and are reported as agreement, never as accuracy.
- Gate for any behaviour change: zero new wrong answers, no lost correct answers (tolerance 0 by default),
  p95 latency at most 1.25x the baseline, test suite green.
