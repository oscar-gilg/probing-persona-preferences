#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around tectonic. Builds paper/main.tex in place.
# Refreshes paper/numbers.tex + paper/claims.md from the claim sidecars
# before compiling; producers must be re-run separately to refresh sidecars.
# Usage: bash scripts/build_paper.sh [extra tectonic args]

cd "$(dirname "$0")/.."
python scripts/paper/build_claims.py
cd paper
tectonic main.tex "$@"
