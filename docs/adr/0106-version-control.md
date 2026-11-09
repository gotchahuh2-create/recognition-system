# ADR-0106: Git from v0.10, photos and models outside history

- Date: 2026-11-09
- Status: Accepted

## Decision

The repository starts at tag `v0.10-baseline` (the code as found). `.gitignore` excludes photos (except the
three regression photos in `tests/real`), models, galleries and `runs/`. The owner pushes to GitHub
(`scripts/phoenix_git_init.sh`).
