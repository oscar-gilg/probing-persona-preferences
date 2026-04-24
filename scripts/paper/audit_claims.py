"""Run the corroborate drift + integrity audit against paper/claims/.

Reports:
  - Drift vs HEAD: ADDED / REMOVED / CHANGED claims.
  - Integrity:
      ORPHAN         — producer file missing from disk.
      NAME-ORPHAN    — producer exists but no longer registers the claim by name.
      LOGIC-STALE    — producer last-committed after the claim was computed.
      DATA-STALE     — an input in data_paths has mtime newer than computed_at.
      COLLISION      — same claim name registered in multiple sidecars.
      ORPHAN-MACRO   — claim's macro is defined but never cited in main.tex.
  - NEAR-DUPLICATE detection (two claims from different producers, close values,
    shared data_path) is split by a Claude judge into:
      AUTO-CONSOLIDATE (verdict=duplicate)
      NEEDS-REVIEW     (verdict=uncertain)
      AUTO-DISMISS     (verdict=complementary or unrelated)
    Disable by setting CORROBORATE_NO_JUDGE=1 (falls back to raw NEAR-DUPLICATE).
  - Unverifiable by convention:
      FROZEN / MANUAL / SUPERSEDED.

Usage:
  python scripts/paper/audit_claims.py
  CORROBORATE_NO_JUDGE=1 python scripts/paper/audit_claims.py   # skip LLM judge
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from corroborate.audit import audit, print_report


load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_DIR = REPO_ROOT / "paper" / "claims"
PAPER_SOURCES = [REPO_ROOT / "paper" / "main.tex"]


def _maybe_build_judge():
    """Return a DuplicateJudge using whichever backend is available.

    Prefers OpenRouter (project-canonical). Falls back to the anthropic SDK
    if OpenRouter isn't set up. Returns None (disables classification) if
    neither works or CORROBORATE_NO_JUDGE=1 is set.
    """
    if os.environ.get("CORROBORATE_NO_JUDGE"):
        return None
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            from corroborate.judges.openai_compatible import openai_judge
        except ImportError:
            pass
        else:
            return openai_judge(
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
            )
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from corroborate.judges.anthropic import anthropic_judge
        except ImportError:
            pass
        else:
            return anthropic_judge()
    return None


def main() -> None:
    judge = _maybe_build_judge()
    report = audit(
        CLAIMS_DIR, REPO_ROOT,
        paper_sources=PAPER_SOURCES,
        duplicate_judge=judge,
    )
    print_report(report)


if __name__ == "__main__":
    main()
