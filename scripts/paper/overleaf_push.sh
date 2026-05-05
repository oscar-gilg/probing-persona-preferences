#!/usr/bin/env bash
# Push paper/ contents to overleaf master.
#
# Why plumbing instead of `git subtree push`: Overleaf's project starts with
# its own initial commit (unrelated to our history) and rejects force pushes.
# We work around this by computing a subtree split locally and pushing a
# merge commit whose parents are [our split tip, overleaf/master]. This
# always fast-forwards from Overleaf's perspective.
#
# Day-to-day flow:
#   1. Edit/commit paper/ as normal in this repo.
#   2. Run this script. It pushes the current paper/ state to Overleaf.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "==> Fetching overleaf/master..."
git fetch overleaf master >/dev/null 2>&1

TMP_BRANCH="_ovbridge_$$"
echo "==> Splitting paper/ subtree (this can take ~30s on a large repo)..."
git subtree split --prefix=paper HEAD -b "$TMP_BRANCH" 2>/dev/null >/dev/null
SPLIT_TIP=$(git rev-parse "$TMP_BRANCH")
git branch -D "$TMP_BRANCH" >/dev/null

OVERLEAF_TIP=$(git rev-parse overleaf/master)

SPLIT_TREE=$(git rev-parse "${SPLIT_TIP}^{tree}")
OVERLEAF_TREE=$(git rev-parse "${OVERLEAF_TIP}^{tree}")

if [ "$SPLIT_TREE" = "$OVERLEAF_TREE" ]; then
  echo "==> Trees match overleaf. Nothing to push."
  exit 0
fi

if git merge-base --is-ancestor "$OVERLEAF_TIP" "$SPLIT_TIP" 2>/dev/null; then
  echo "==> Fast-forward push..."
  git push overleaf "$SPLIT_TIP:master"
else
  TREE=$(git rev-parse "${SPLIT_TIP}^{tree}")
  MERGE=$(git commit-tree "$TREE" -p "$SPLIT_TIP" -p "$OVERLEAF_TIP" \
          -m "subtree push: paper/ -> overleaf master")
  echo "==> Pushing linkage merge commit $MERGE..."
  git push overleaf "$MERGE:master"
fi

echo "==> Done."
