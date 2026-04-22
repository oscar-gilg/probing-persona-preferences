#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around tectonic. Builds paper/main.tex in place.
# Usage: bash scripts/build_paper.sh [extra tectonic args]

cd "$(dirname "$0")/../paper"
tectonic main.tex "$@"
