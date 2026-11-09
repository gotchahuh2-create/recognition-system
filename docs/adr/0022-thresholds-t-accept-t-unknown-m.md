# ADR-0022: Thresholds T_accept / T_unknown / M are placeholders (0.70 / 0.45 / 0.05) until `click_eva

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 22 (migrated)

## Decision

Thresholds T_accept / T_unknown / M are placeholders (0.70 / 0.45 / 0.05) until `click_eval.py --write-thresholds` calibrates them on val (98 % accept precision)

## Context and reason

Report §3; test split frozen

## Evidence

eval/click_eval.py
