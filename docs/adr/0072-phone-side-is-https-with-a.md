# ADR-0072: Phone side is https with a self-signed certificate made by `openssl`

- Date: 2026-11-05
- Status: Accepted
- Source: DECISIONS.md row 72 (migrated)

## Decision

Phone side is https with a self-signed certificate made by `openssl`; PC page on 127.0.0.1 only; QR token on every phone request; stdlib asyncio + own RFC 6455 WebSocket + own QR encoder

## Context and reason

Browsers expose the camera only on secure pages; no cloud, no new dependencies on the user PC

## Evidence

test_phone_session_over_websocket, test_qr_code_decodes
