# ADR-0066: Stacked chip lines joined up to three when the first is short (logo line)

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 66 (migrated)

## Decision

Stacked chip lines joined up to three when the first is short (logo line)

## Context and reason

PIC32 markings are 'PIC32' / 'MZ2048EFH' / '144-I/PH': the two-line join gave PIC32MZ2048EFH (0.82 vs catalogue ...EFH144), never accepted even when every line was read right

## Evidence

synthetic 400 px chips: 0/3 -> 4/6 correct, 0 wrong (EFE vs EFF abstain: E/F look-alike)
