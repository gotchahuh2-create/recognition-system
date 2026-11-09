# ADR-0078: Scan always takes two photos: board, then MCU close-up

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 78 (migrated)

## Decision

Scan always takes two photos: board, then MCU close-up; auto-capture held between the steps until the user taps Continue (or 4 s)

## Context and reason

Asked for a fixed two-step flow; a hold stops a half-moved phone from firing step 2

## Evidence

test_phone_session_over_websocket (step 2 config + ready)
