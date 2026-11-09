#!/usr/bin/env bash
# One-time: put this folder under git as PHOENIX expects (ADR-0106), then (optionally) push to GitHub.
#   bash scripts/phoenix_git_init.sh                       # local repository only
#   bash scripts/phoenix_git_init.sh git@github.com:USER/REPO.git
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .git ]; then echo "already a git repository"; exit 0; fi
git init -b main
# 1) the code exactly as it was before PHOENIX (v0.10), so every later change has a diff
git add -A
git reset -q -- phoenix config bench docs CLAUDE.md scripts/migrate_decisions.py \
  scripts/phoenix_git_init.sh tests/test_phoenix.py 2>/dev/null || true
git commit -m "v0.10 baseline (before PHOENIX)"
git tag v0.10-baseline
# 2) PHOENIX layer
git add -A
git commit -m "PHOENIX: engineering layer (phases 0-5), docs, config, bench manifests"
# 3) legacy code to archive/ (ADR-0101) — moved, not deleted
mkdir -p archive
git mv vision/camera archive/camera
git mv vision/app.py archive/app_v09.py
git commit -m "Archive superseded vision/camera and shadowed vision/app.py (ADR-0101)"
git tag phoenix-0.1
if [ "${1:-}" != "" ]; then
  git remote add origin "$1"
  git push -u origin main --tags
fi
echo "done: $(git log --oneline | head -3)"
