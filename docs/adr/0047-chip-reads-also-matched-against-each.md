# ADR-0047: Chip reads also matched against each card's shop-photo marking (dataset `chip_marking_obse

- Date: 2026-11-04
- Status: Accepted
- Source: DECISIONS.md row 47 (migrated)

## Decision

Chip reads also matched against each card's shop-photo marking (dataset `chip_marking_observed` → joined patterns); ST markings without "STM" ('32F215RET6') get the prefix

## Context and reason

Markings are not always the ordering part number: Kinetis 'MK64 / FN1M0V12' for MK64FN1M0VDC12, 'M20A7V' for MK20DX128VFM5; WRONG_CHIP / blank photos give no pattern

## Evidence

test_stacked_chip_lines_and_shop_marking, test_st_marking_without_stm_prefix
