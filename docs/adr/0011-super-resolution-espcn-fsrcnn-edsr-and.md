# ADR-0011: Super-resolution (ESPCN/FSRCNN/EDSR) and TrOCR are optional, not default

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 11 (migrated)

## Decision

Super-resolution (ESPCN/FSRCNN/EDSR) and TrOCR are optional, not default

## Context and reason

Brief: SR should be judged by OCR accuracy, TrOCR is heavy; neither could be downloaded/tested in the build environment — compare with `--configs full,sr-espcn,trocr` on real photos first

## Evidence

scripts/get_sr_models.py
