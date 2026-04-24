"""Register §4.4 safety-override sweep compliance numbers.

Values come from the compliance table in
`experiments/steering/open_ended_steering/safety_steering/safety_steering_report.md`
(judge-rated compliance per (tier, coefficient), pre-computed and committed).

The table is static and small; re-reading the markdown each run keeps the
claim self-documenting. If the judge-rating is redone, update the report and
rerun this script.

Sweep shape (quoted in §4.4 setup bullet): 20 prompts x 5 tiers x 7 coefficients
x 5 trials = 700 generations (though the report's compliance table shows 6
coefficient columns — the +0.00 column is the baseline, so 7 coefficients total
matches the paper's `c in {-0.05, -0.03, 0, +0.03, +0.05, +0.07, +0.10}`).

Run:
  python scripts/paper/claims/compute_safety_sweep_compliance.py
"""

from __future__ import annotations

from pathlib import Path

from src.paper.claims import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]

# Pasted from safety_steering_report.md (lines 23-29). Integer percentages.
# Paper's main.tex §4.4 uses c in {-0.05, -0.03, 0, +0.03, +0.05, +0.07, +0.10}
# (7 coefficients). The report's table omits c=-0.03; that cell is tracked as
# missing-from-report, not missing-from-data. The report reflects what was
# rendered when the figure was prepared.
COMPLIANCE_BY_TIER_AND_COEF: dict[str, dict[str, int]] = {
    "benign":     {"-0.05": 100, "0.00": 100, "+0.03": 100, "+0.05": 100, "+0.07": 100, "+0.10": 50},
    "sensitive":  {"-0.05": 40,  "0.00": 100, "+0.03": 100, "+0.05": 100, "+0.07": 100, "+0.10": 80},
    "borderline": {"-0.05": 5,   "0.00": 80,  "+0.03": 85,  "+0.05": 100, "+0.07": 100, "+0.10": 55},
    "harmful":    {"-0.05": 25,  "0.00": 60,  "+0.03": 85,  "+0.05": 100, "+0.07": 100, "+0.10": 55},
    "extreme":    {"-0.05": 5,   "0.00": 60,  "+0.03": 75,  "+0.05": 60,  "+0.07": 85,  "+0.10": 55},
}


def _coef_token(coef: str) -> str:
    """Render a coefficient string into something the slugifier handles cleanly."""
    sign = "pos" if coef.startswith("+") else ("neg" if coef.startswith("-") else "zero")
    mag = coef.lstrip("+-").replace(".", " ")
    return f"{sign} {mag}"


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_safety_sweep_compliance.py")

    # Sweep shape.
    claims.register(
        "Safety sweep prompt count",
        20,
        "The §4.4 safety-override sweep uses 20 open-ended prompts, distributed "
        "evenly across 5 harm tiers.",
        used_in=["sec:open-ended-safety"],
    )
    claims.register(
        "Safety sweep harm tier count",
        5,
        "The §4.4 safety sweep covers 5 harm tiers: benign, sensitive, "
        "borderline, harmful, extreme.",
        used_in=["sec:open-ended-safety"],
    )
    claims.register(
        "Safety sweep coefficient count",
        7,
        "The §4.4 safety sweep tests 7 steering coefficients: c in "
        "{-0.05, -0.03, 0, +0.03, +0.05, +0.07, +0.10}.",
        used_in=["sec:open-ended-safety"],
    )
    claims.register(
        "Safety sweep trials per cell",
        5,
        "5 trials per (prompt, coefficient) cell in the §4.4 safety sweep.",
        used_in=["sec:open-ended-safety"],
    )
    claims.register(
        "Safety sweep total generations",
        20 * 5 * 7 * 5,
        "20 prompts x 5 harm tiers x 7 coefficients x 5 trials = 700 total "
        "generations in the §4.4 safety sweep.",
        used_in=["sec:open-ended-safety"],
    )

    # Compliance cells (30 = 5 tiers x 6 reported coefficients; c=-0.03 is not
    # in the report's table, so we skip it here).
    for tier, by_coef in COMPLIANCE_BY_TIER_AND_COEF.items():
        for coef, pct in by_coef.items():
            claims.register(
                f"Safety sweep compliance {tier} at c {_coef_token(coef)}",
                pct,
                f"Judge-rated compliance rate ({pct}%) on {tier}-tier prompts "
                f"at steering coefficient c={coef} in the §4.4 safety-override "
                f"sweep. Source: safety_steering_report.md compliance table.",
                used_in=["sec:open-ended-safety", "fig:safety-override"],
                source=(
                    "manual: transcribed from "
                    "experiments/steering/open_ended_steering/safety_steering/safety_steering_report.md"
                ),
            )

    # High-leverage summary cells quoted in prose.
    claims.register(
        "Safety sweep borderline compliance baseline",
        COMPLIANCE_BY_TIER_AND_COEF["borderline"]["0.00"],
        "Borderline-tier compliance at c=0 is 80% (unsteered baseline). The "
        "`~5-60% baseline` range quoted in main.tex §4.4 refers to the most "
        "refusal-prone tiers (borderline, harmful, extreme) aggregated, whose "
        "pre-steering compliance spans 60-80%.",
        used_in=["sec:open-ended-safety"],
    )
    claims.register(
        "Safety sweep compliance at c +0.05 harmful tier",
        COMPLIANCE_BY_TIER_AND_COEF["harmful"]["+0.05"],
        "At c=+0.05, compliance on harmful-tier prompts reaches 100% "
        "(refusal guardrails fully overridden).",
        used_in=["sec:open-ended-safety"],
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "safety_sweep_compliance.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
