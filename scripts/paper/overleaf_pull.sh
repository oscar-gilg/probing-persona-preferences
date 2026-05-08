#!/usr/bin/env bash
# Pull edits from Overleaf into paper/ as a real 3-way merge.
#
# `git subtree pull` works because the push script has already linked the
# histories via a synthetic merge commit. Conflicts surface as <<<<<<<
# markers in paper/ and the script exits non-zero — wrappers (e.g.
# /overleaf-sync) handle from there.
#
# Skips the pull entirely if overleaf/master is unchanged since the last
# successful pull (tracked via overleaf.lastpulledsha git config).
#
# Pre-flight: refuses to run if paper/ has uncommitted or untracked files,
# because a merge into a dirty tree clobbers in-progress work. The slash
# command auto-commits before calling this script.
#
# Env: OVERLEAF_SKIP_FETCH=1 — skip `git fetch overleaf master` (use when
# /overleaf-sync just fetched via overleaf_push.sh).
#
# Git config keys:
#   overleaf.lastpulledsha — overleaf/master SHA after last successful pull
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! git config --get remote.overleaf.url >/dev/null; then
  echo "ERROR: no 'overleaf' remote configured." >&2
  exit 1
fi

if [ "${OVERLEAF_SKIP_FETCH:-}" != "1" ]; then
  echo "==> Fetching overleaf/master..."
  git fetch overleaf master >/dev/null 2>&1
fi

TIP=$(git rev-parse overleaf/master)
LAST=$(git config --get overleaf.lastpulledsha 2>/dev/null || true)
if [ -n "$LAST" ] && [ "$LAST" = "$TIP" ]; then
  echo "==> overleaf/master unchanged since last pull. Nothing to pull."
  exit 0
fi

if ! git diff --quiet HEAD -- paper/; then
  echo "ERROR: paper/ has uncommitted changes. Commit or stash first:" >&2
  git status --short -- paper/ >&2
  exit 1
fi

if git ls-files --others --exclude-standard paper/ | grep -q .; then
  echo "ERROR: paper/ has untracked files. Commit or remove first:" >&2
  git ls-files --others --exclude-standard paper/ >&2
  exit 1
fi

echo "==> git subtree pull --prefix=paper overleaf master..."
git subtree pull --prefix=paper overleaf master --squash \
  -m "sync paper/ from overleaf"

git config overleaf.lastpulledsha "$TIP"
echo "==> Done. Push with overleaf_push.sh when ready."
