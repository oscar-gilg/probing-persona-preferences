#!/usr/bin/env bash
# Pull edits from overleaf into paper/.
#
# Strategy: clone overleaf into a temp dir, then rsync its contents into
# paper/. This sidesteps the unrelated-histories problem that breaks
# `git subtree pull`. The user reviews the resulting working-tree diff and
# commits manually (or aborts).
#
# After running, expect to see modified files in `git status paper/`.
# Inspect with `git diff paper/`, then commit if happy:
#   git add paper/ && git commit -m "sync paper/ from overleaf"
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

OVERLEAF_URL=$(git config --get remote.overleaf.url)
if [ -z "$OVERLEAF_URL" ]; then
  echo "ERROR: no 'overleaf' remote configured." >&2
  exit 1
fi

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "==> Cloning overleaf to temp dir..."
git clone --quiet --depth=1 "$OVERLEAF_URL" "$TMPDIR/overleaf"

echo "==> Rsyncing into paper/..."
rsync -a --delete --exclude='.git' "$TMPDIR/overleaf/" paper/

echo "==> Done. Review changes:"
echo "    git diff paper/"
echo "    git status paper/"
echo "Then commit if happy:"
echo "    git add paper/ && git commit -m 'sync paper/ from overleaf'"
