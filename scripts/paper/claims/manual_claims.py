"""Manual claim entries.

Numbers that cannot currently be auto-derived from the repo — either because
the producer predates an audit-friendly data-pipeline (superseded experiments,
lost intermediate artifacts) or because reimplementation is blocked on
infrastructure work. Each entry should note why it is manual and how to
upgrade it to a machine-derived claim when possible.

These are tracked in `paper/claims.md` with `source="manual: ..."` so
`audit_claims.py` flags them on every run.
"""

from pathlib import Path

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/manual_claims.py")

    # Stated-preference steering phase 1 / phase 2 claims removed 2026-04-24:
    # the L31 results were superseded by a later finding that steering at
    # earlier layers produces substantially larger effects. The figures and
    # paragraph have been dropped from paper/main.tex pending a rerun of
    # stated-preference steering with the canonical completion template at
    # the earlier-layer operating point.

    # Abstract summary: the ~0.9 number is a cross-experiment approximation.
    # It averages / summarises the Gemma-targeted r = 0.95 (§4.1 simple),
    # Qwen-targeted r = 0.83 (§4.1 simple replication), conflict r = 0.86,
    # opposing r = 0.88, and biography-injection rank-1-of-50 rate of 36/40.
    # There is no single computable value to point at; registering as a manual
    # summary means audit_claims flags it so a future writer can refine or
    # replace with a derived range.
    claims.register(
        name="Abstract targeted tasks r approximation",
        value=0.9,
        statement=(
            "Abstract shorthand for the Gemma-3-27B preference-probe correlation "
            "with behavioural deltas on targeted tasks across the induced-shift, "
            "conflict/opposing, and biography-injection experiments in §4.1 and "
            "App.~C. The underlying values range from ~0.83 (Qwen simple-target) "
            "to 0.95 (Gemma simple-target)."
        ),
        used_in=["abstract"],
        source="manual: cross-experiment abstract approximation; refine by replacing with a derived range from the registered per-experiment targeted-r claims",
        data_paths=[],
        derivation=(
            "Manual cross-experiment shorthand summarising Gemma simple-target r=0.95 (§4.1), "
            "Qwen simple-target r=0.83, conflict r=0.86, opposing r=0.88, and biography-injection "
            "rank-1-of-50 rate of 36/40; refine by replacing with a derived range from the "
            "per-experiment targeted-r claims once they are all registered."
        ),
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "manual_claims.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
