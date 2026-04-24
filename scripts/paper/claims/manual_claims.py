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

    # Stated-preference steering phase 1 / phase 2 (Feb 2026).
    # Experiment was later marked superseded (label-based a/b parsing rather
    # than canonical completion-based). Figures remain in the paper appendix
    # (plot_022426_stated_steering_{positions,formats}.png). Producer scripts
    # were deleted (git commit 04526b5) and a byte-identical copy of each PNG
    # lives in experiments/steering/stated_steering/assets/. We freeze the
    # caption numbers here until the experiment is redone under the canonical
    # completion template. See paper/TODO_producers.md.

    claims.register(
        name="Stated-steering phase1 baseline rating",
        value=3.64,
        statement=(
            "In the phase-1 stated-preference steering experiment (n=200, Feb "
            "2026, subsequently superseded), the unsteered baseline rating "
            "was 3.64 on the 1-5 preference scale."
        ),
        used_in=["fig:stated-positions"],
        source="manual: stated-steering phase 1 (superseded a/b template); see paper/TODO_producers.md",
        data_paths=[],
        derivation=(
            "Manual frozen value from the superseded phase-1 stated-preference experiment (Feb 2026); "
            "producer scripts deleted (commit 04526b5), raw data not reproducible under the canonical "
            "completion template; see paper/TODO_producers.md."
        ),
    )
    claims.register(
        name="Stated-steering phase1 negative range",
        value="3.27-3.36",
        statement=(
            "At c=-10% of the mean activation norm, stated-preference rating "
            "falls to 3.27-3.36 across position arms (phase-1 experiment)."
        ),
        used_in=["fig:stated-positions"],
        source="manual: stated-steering phase 1 (superseded a/b template); see paper/TODO_producers.md",
        data_paths=[],
        derivation=(
            "Manual frozen range from the superseded phase-1 stated-preference experiment (Feb 2026); "
            "producer scripts deleted, figure preserved in experiments/steering/stated_steering/assets/; "
            "see paper/TODO_producers.md."
        ),
    )
    claims.register(
        name="Stated-steering phase1 positive range",
        value="4.60-4.61",
        statement=(
            "At c=+10% of the mean activation norm, stated-preference rating "
            "rises to 4.60-4.61 across position arms (phase-1 experiment)."
        ),
        used_in=["fig:stated-positions"],
        source="manual: stated-steering phase 1 (superseded a/b template); see paper/TODO_producers.md",
        data_paths=[],
        derivation=(
            "Manual frozen range from the superseded phase-1 stated-preference experiment (Feb 2026); "
            "producer scripts deleted, figure preserved in experiments/steering/stated_steering/assets/; "
            "see paper/TODO_producers.md."
        ),
    )

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
