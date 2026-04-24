"""Run the corroborate drift + integrity audit against paper/claims/.

Reports:
  - Drift vs HEAD: ADDED / REMOVED / CHANGED claims.
  - Integrity:
      ORPHAN         — producer file missing from disk.
      NAME-ORPHAN    — producer exists but no longer registers the claim by name.
      LOGIC-STALE    — producer last-committed after the claim was computed.
      DATA-STALE     — an input in data_paths has mtime newer than computed_at.
      COLLISION      — same claim name registered in multiple sidecars.
      NEAR-DUPLICATE — two claims from different producers with ≈equal values
                       reading a shared data_path.
      ORPHAN-MACRO   — claim's macro is defined but never cited in main.tex.
  - Unverifiable by convention:
      FROZEN / MANUAL / SUPERSEDED.

Usage:
  python scripts/paper/audit_claims.py
"""
from __future__ import annotations

from pathlib import Path

from corroborate.audit import audit, print_report


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_DIR = REPO_ROOT / "paper" / "claims"
PAPER_SOURCES = [REPO_ROOT / "paper" / "main.tex"]


def main() -> None:
    report = audit(CLAIMS_DIR, REPO_ROOT, paper_sources=PAPER_SOURCES)
    print_report(report)


if __name__ == "__main__":
    main()
