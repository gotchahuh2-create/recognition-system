# ADR-0012: Test window is a local web page (stdlib http.server), not a Tk window

- Date: 2026-11-03
- Status: Accepted
- Source: DECISIONS.md row 12 (migrated)

## Decision

Test window is a local web page (stdlib http.server), not a Tk window

## Context and reason

Real drag-and-drop of many files, paste, image overlays and zoom work out of the box in a browser; no new dependencies; 127.0.0.1 only

## Evidence

vision/app.py
