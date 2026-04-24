r"""Build `paper/numbers.tex` and `paper/claims.md` from per-producer sidecars.

Expected layout:
  paper/claims/<producer_slug>.json   # written by each producer via ClaimSet.save()
  paper/numbers.tex                   # generated here; \input from main.tex
  paper/claims.md                     # generated here; human/agent audit surface

Usage:
  python scripts/paper/build_claims.py

This script does not re-run producers. Run them separately (or via
`scripts/build_paper.sh`) to refresh the sidecars before building.
"""

from __future__ import annotations

from pathlib import Path

from corroborate import load_all, write_claims_md, write_numbers_tex


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_DIR = REPO_ROOT / "paper" / "claims"
NUMBERS_TEX = REPO_ROOT / "paper" / "numbers.tex"
CLAIMS_MD = REPO_ROOT / "paper" / "claims.md"


def main() -> None:
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    claims = load_all(CLAIMS_DIR)
    write_numbers_tex(claims, NUMBERS_TEX)
    write_claims_md(claims, CLAIMS_MD)
    print(
        f"Wrote {NUMBERS_TEX.relative_to(REPO_ROOT)} "
        f"and {CLAIMS_MD.relative_to(REPO_ROOT)} "
        f"({len(claims)} claims)."
    )


if __name__ == "__main__":
    main()
