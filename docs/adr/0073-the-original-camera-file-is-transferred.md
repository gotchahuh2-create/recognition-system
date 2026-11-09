# ADR-0073: The ORIGINAL camera file is transferred (chunked, SHA-256) and saved byte for byte with a 

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 73 (migrated)

## Decision

The ORIGINAL camera file is transferred (chunked, SHA-256) and saved byte for byte with a JSON sidecar of all metrics

## Context and reason

Re-encoding would cost marking detail; sidecars are the data to tune the [ASSUMPTION] thresholds on real phones

## Evidence

photos/captures/*.json
