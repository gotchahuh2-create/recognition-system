# ADR-0067: Resolution-model marking verifier (`vision/faint/synth.py`: render candidates in a stroke 

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 67 (migrated)

## Decision

Resolution-model marking verifier (`vision/faint/synth.py`: render candidates in a stroke font, blur + sample at native px + cubic up, NCC on the differing character cells against the faint L residual) is opt-in `--verify-marking`

## Context and reason

Never wrong in the sweep, but decides rarely (font of the real marking unknown); on the real photos it abstains correctly. Off by default: +2-6 s per photo for no measured gain yet

## Evidence

eval/synth_eval_synthetic.md
