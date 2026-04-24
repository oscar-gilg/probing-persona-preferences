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

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]

# Column keys encode the coefficient with a sign token ("neg"/"zero"/"pos") +
# magnitude digits so the slugifier produces stable macro names like
# `AtCPosZeroZerofive`. Row keys are plain tier names.
COMPLIANCE_BY_TIER_AND_COEF: dict[str, dict[str, int]] = {
    "benign": {
        "at_c_neg_0_05": 100,
        "at_c_zero_0_00": 100,
        "at_c_pos_0_03": 100,
        "at_c_pos_0_05": 100,
        "at_c_pos_0_07": 100,
        "at_c_pos_0_10": 50,
    },
    "sensitive": {
        "at_c_neg_0_05": 40,
        "at_c_zero_0_00": 100,
        "at_c_pos_0_03": 100,
        "at_c_pos_0_05": 100,
        "at_c_pos_0_07": 100,
        "at_c_pos_0_10": 80,
    },
    "borderline": {
        "at_c_neg_0_05": 5,
        "at_c_zero_0_00": 80,
        "at_c_pos_0_03": 85,
        "at_c_pos_0_05": 100,
        "at_c_pos_0_07": 100,
        "at_c_pos_0_10": 55,
    },
    "harmful": {
        "at_c_neg_0_05": 25,
        "at_c_zero_0_00": 60,
        "at_c_pos_0_03": 85,
        "at_c_pos_0_05": 100,
        "at_c_pos_0_07": 100,
        "at_c_pos_0_10": 55,
    },
    "extreme": {
        "at_c_neg_0_05": 5,
        "at_c_zero_0_00": 60,
        "at_c_pos_0_03": 75,
        "at_c_pos_0_05": 60,
        "at_c_pos_0_07": 85,
        "at_c_pos_0_10": 55,
    },
}


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_safety_sweep_compliance.py")

    _report_path = "experiments/steering/open_ended_steering/safety_steering/safety_steering_report.md"

    # Sweep shape.
    claims.register(
        "Safety sweep prompt count",
        20,
        "The §4.4 safety-override sweep uses 20 open-ended prompts, distributed "
        "evenly across 5 harm tiers.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_report_path],
        derivation="Constant: 20 open-ended prompts in the safety sweep, documented in safety_steering_report.md.",
    )
    claims.register(
        "Safety sweep harm tier count",
        5,
        "The §4.4 safety sweep covers 5 harm tiers: benign, sensitive, "
        "borderline, harmful, extreme.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_report_path],
        derivation="Constant: 5 harm tiers {benign, sensitive, borderline, harmful, extreme}, documented in safety_steering_report.md.",
    )
    claims.register(
        "Safety sweep coefficient count",
        7,
        "The §4.4 safety sweep tests 7 steering coefficients: c in "
        "{-0.05, -0.03, 0, +0.03, +0.05, +0.07, +0.10}.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_report_path],
        derivation="Constant: 7 coefficients {-0.05, -0.03, 0, +0.03, +0.05, +0.07, +0.10}; the report table omits c=-0.03.",
    )
    claims.register(
        "Safety sweep trials per cell",
        5,
        "5 trials per (prompt, coefficient) cell in the §4.4 safety sweep.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_report_path],
        derivation="Constant: 5 trials per (prompt, coefficient) cell, documented in safety_steering_report.md.",
    )
    claims.register(
        "Safety sweep total generations",
        20 * 5 * 7 * 5,
        "20 prompts x 5 harm tiers x 7 coefficients x 5 trials = 700 total "
        "generations in the §4.4 safety sweep.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_report_path],
        derivation="20 prompts * 5 tiers * 7 coefficients * 5 trials = 700.",
    )

    # Compliance cells as a single table claim (5 tiers x 6 reported
    # coefficients; c=-0.03 is absent from the report table). The slugifier
    # emits one `\newcommand` per cell, e.g.
    # `\safetySweepComplianceHarmfulAtCPosZeroZerofive`.
    claims.register(
        "Safety sweep compliance",
        COMPLIANCE_BY_TIER_AND_COEF,
        "Judge-rated compliance rate (percent) in the §4.4 safety-override "
        "sweep. Rows: harm tier (benign, sensitive, borderline, harmful, "
        "extreme). Columns: steering coefficient (c in {-0.05, 0.00, +0.03, "
        "+0.05, +0.07, +0.10}; c=-0.03 omitted from the report table). "
        "Source: safety_steering_report.md compliance table.",
        used_in=["sec:open-ended-safety", "fig:safety-override"],
        source=(
            "manual: transcribed from "
            "experiments/steering/open_ended_steering/safety_steering/safety_steering_report.md"
        ),
        data_paths=[_report_path],
        derivation=(
            "Table transcribed from safety_steering_report.md compliance table. "
            "Each cell is the judge-rated compliance percent for (tier row, "
            "coefficient column)."
        ),
    )

    # High-leverage summary cells quoted in prose.
    claims.register(
        "Safety sweep borderline compliance baseline",
        COMPLIANCE_BY_TIER_AND_COEF["borderline"]["at_c_zero_0_00"],
        "Borderline-tier compliance at c=0 is 80% (unsteered baseline). The "
        "`~5-60% baseline` range quoted in main.tex §4.4 refers to the most "
        "refusal-prone tiers (borderline, harmful, extreme) aggregated, whose "
        "pre-steering compliance spans 60-80%.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_report_path],
        derivation="Value of the borderline-tier, c=0.00 cell in COMPLIANCE_BY_TIER_AND_COEF (transcribed from safety_steering_report.md).",
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "safety_sweep_compliance.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
