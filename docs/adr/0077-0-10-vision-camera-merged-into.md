# ADR-0077: 0.10: `vision.camera` merged into `vision.app` (one asyncio server on 8765 + phone on 8443

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 77 (migrated)

## Decision

0.10: `vision.camera` merged into `vision.app` (one asyncio server on 8765 + phone on 8443; bench routes served by the same server; recognition on one worker thread)

## Context and reason

Asked for one app; one process means one catalogue / OCR warm-up and no second port

## Evidence

test_app_recognize_and_label, test_phone_session_over_websocket
