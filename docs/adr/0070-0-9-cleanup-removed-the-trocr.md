# ADR-0070: 0.9 cleanup: removed the TrOCR crop reader, the `--global-enhance` CLI flag / UI toggle, `

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 70 (migrated)

## Decision

0.9 cleanup: removed the TrOCR crop reader, the `--global-enhance` CLI flag / UI toggle, `recognise_board` / `recognise_mcu` wrappers and unused imports; kept every working module incl. Clicks and the opt-in faint / verifier paths

## Context and reason

TrOCR was never runnable or measured here; global enhance was measured harmful (3 Nov) and stays only as an eval ablation config; the wrappers had no caller

## Evidence

full test suite unchanged (1 pre-existing Tesseract-version failure in test_layout)
