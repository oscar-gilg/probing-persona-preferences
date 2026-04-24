"""Stopgap manual claims for Gemma induced-shift targeted-tasks macros.

These macros were emitted by the legacy scripts/paper/build_claims.py but no
corroborate-compatible producer registers them. Values from the pre-corroborate
numbers.tex at commit 924b289. Flagged in paper/PAPER_ISSUES.md (I-2026-04-24-1)
— replace with a real producer when the induced-shift pipeline is ported.
"""

from __future__ import annotations

from pathlib import Path

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    "manual: value from the legacy numbers.tex at commit 924b289, before the "
    "corroborate migration; no current producer computes it. See "
    "paper/PAPER_ISSUES.md entry I-2026-04-24-1."
)


def main() -> None:
    claims = ClaimSet(source=SOURCE)

    claims.register(
        name="Gemma induced-shift pooled r targeted tasks",
        value=0.95,
        statement=(
            "For Gemma-3-27B on the targeted induced-shift tasks, per-task "
            "probe delta correlates with behavioural delta at $r = 0.95$ "
            "(pooled across 8 topics $\\times$ 2 valences)."
        ),
        used_in=["sec:induced-basic"],
        source=SOURCE,
    )
    claims.register(
        name="Gemma induced-shift pooled n targeted tasks",
        value=81,
        statement=(
            "The Gemma induced-shift targeted-tasks pooling covers "
            "$n = 81$ task-prompt points."
        ),
        used_in=["sec:induced-basic"],
        source=SOURCE,
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "manual_induced_shift_targeted.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)} ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
