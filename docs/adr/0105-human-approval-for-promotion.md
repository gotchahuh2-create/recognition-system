# ADR-0105: No autonomous code modification; promotions need a named human approval

- Date: 2026-11-09
- Status: Accepted

## Decision

The Adaptation Engine only writes proposals (`docs/proposals/`). A proposal can change a stage to another
*registered* implementation and/or override registry parameters that are not bound. Promotion
(`python -m phoenix promote P-NNN --approved-by NAME`) requires all gates passed and writes a new
immutable config version, an ADR stub and the history entry. Rollback switches `config/active.yaml` back.
