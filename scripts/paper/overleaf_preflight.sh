#!/usr/bin/env bash
# Preflight checks shared by /overleaf-sync, /overleaf-pull, /overleaf-push.
# Exits non-zero with a clear message on any failure.
#
# Checks:
#   1. an `overleaf` git remote is configured
#   2. HEAD points to refs/heads/main (refuses sync from feature branches)
#   3. paper/ contains no unresolved <<<<<<< / ======= / >>>>>>> markers
#   4. paper/ has no untracked files matching common secret patterns
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if ! git config --get remote.overleaf.url >/dev/null; then
  echo "ERROR: no 'overleaf' remote configured in this project." >&2
  exit 1
fi

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo DETACHED)
if [ "$BRANCH" != "main" ]; then
  echo "ERROR: refusing to sync from '$BRANCH'; check out main first." >&2
  exit 1
fi

# `={7} ` requires a trailing space (Git always emits "======= " on the
# separator line); avoids false positives on LaTeX `=======` separators.
MARKERS=$(grep -lrE '^(<{7}|>{7}|={7}) ' paper/ 2>/dev/null || true)
if [ -n "$MARKERS" ]; then
  echo "ERROR: paper/ contains unresolved conflict markers — resolve and commit first:" >&2
  echo "$MARKERS" >&2
  exit 1
fi

SECRETS=$(git ls-files --others --exclude-standard paper/ \
  | grep -iE '\.env$|credentials|secret|api[_-]?key' || true)
if [ -n "$SECRETS" ]; then
  echo "ERROR: paper/ contains untracked files that look like secrets:" >&2
  echo "$SECRETS" >&2
  echo "Remove or .gitignore before syncing." >&2
  exit 1
fi
