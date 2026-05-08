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
#
# Env: OVERLEAF_SKIP_FETCH=1 — skip `git fetch overleaf master` (use when
# /overleaf-sync just fetched via overleaf_pull.sh).
#
# Git config keys:
#   overleaf.lastsplithead  — local HEAD when we last ran subtree split
#   overleaf.lastsplittip   — resulting split-branch tip (cache key)
#   overleaf.lastpushedsha  — last SHA we pushed to overleaf/master
#   overleaf.lastpulledsha  — kept in sync with lastpushedsha after each push
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

HEAD_SHA=$(git rev-parse HEAD)
LAST_HEAD=$(git config --get overleaf.lastsplithead 2>/dev/null || true)
LAST_SPLIT=$(git config --get overleaf.lastsplittip 2>/dev/null || true)

if [ -n "$LAST_HEAD" ] && [ "$LAST_HEAD" = "$HEAD_SHA" ] \
    && [ -n "$LAST_SPLIT" ] && git cat-file -e "$LAST_SPLIT" 2>/dev/null; then
  echo "==> HEAD unchanged since last split. Reusing cached split tip."
  SPLIT_TIP=$LAST_SPLIT
else
  TMP_BRANCH="_ovbridge_$$"
  echo "==> Splitting paper/ subtree (incremental via --rejoin)..."
  # --rejoin caches the split state by adding a rejoin merge commit on HEAD,
  # so future splits only walk new history. First run is still ~30s on a
  # large repo; subsequent runs are seconds.
  git subtree split --prefix=paper --rejoin HEAD -b "$TMP_BRANCH" 2>/dev/null >/dev/null
  SPLIT_TIP=$(git rev-parse "$TMP_BRANCH")
  git branch -D "$TMP_BRANCH" >/dev/null
  git config overleaf.lastsplithead "$(git rev-parse HEAD)"
  git config overleaf.lastsplittip "$SPLIT_TIP"
fi

OVERLEAF_TIP=$(git rev-parse overleaf/master)
SPLIT_TREE=$(git rev-parse "${SPLIT_TIP}^{tree}")
OVERLEAF_TREE=$(git rev-parse "${OVERLEAF_TIP}^{tree}")

if [ "$SPLIT_TREE" = "$OVERLEAF_TREE" ]; then
  echo "==> Trees match overleaf. Nothing to push."
  exit 0
fi

if git merge-base --is-ancestor "$OVERLEAF_TIP" "$SPLIT_TIP"; then
  echo "==> Fast-forward push..."
  git push overleaf "$SPLIT_TIP:master"
  PUSHED=$SPLIT_TIP
else
  MERGE=$(git commit-tree "$SPLIT_TREE" -p "$SPLIT_TIP" -p "$OVERLEAF_TIP" \
          -m "subtree push: paper/ -> overleaf master")
  echo "==> Pushing linkage merge commit $MERGE..."
  git push overleaf "$MERGE:master"
  PUSHED=$MERGE
fi

git config overleaf.lastpushedsha "$PUSHED"
git config overleaf.lastpulledsha "$PUSHED"

echo "==> Done."
