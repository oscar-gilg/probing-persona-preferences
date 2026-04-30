"""Register App. A.3 safety-override sweep compliance numbers.

Source of truth: per-trial judge labels at
`experiments/old_experiments/steering/open_ended_steering/safety_steering/compliance.jsonl`,
produced by `scripts/sadist_open_ended_steering/judge_compliance.py` (same
3-label complied/partial/refused judge used elsewhere in the paper). We
register the strict "complied" rate per (tier, coefficient) cell, since the
caption claims are about producing deployable harmful content rather than
mere engagement.

Sweep shape: 20 prompts x 5 tiers x 7 coefficients x 5 trials = 700.

Run:
  python scripts/paper/claims/compute_safety_sweep_compliance.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPLIANCE_PATH = (
    REPO_ROOT
    / "experiments/old_experiments/steering/open_ended_steering/safety_steering/compliance.jsonl"
)

TIER_ORDER = ["benign", "sensitive", "borderline", "harmful", "extreme"]
COEF_ORDER = [-0.05, -0.03, 0.00, 0.03, 0.05, 0.07, 0.10]


def _coef_key(c: float) -> str:
    """Slugify a coefficient into a stable claim sub-key.

    Maps c in {-0.05, -0.03, 0.00, +0.03, +0.05, +0.07, +0.10} to e.g.
    `at_c_neg_0_05`, `at_c_zero_0_00`, `at_c_pos_0_05`. The slugifier then
    emits macros like `\\safetySweepComplianceHarmfulAtCPosZeroZerofive`.
    """
    if c < 0:
        sign = "neg"
    elif c == 0:
        sign = "zero"
    else:
        sign = "pos"
    digits = f"{abs(c):.2f}".replace(".", "_")
    return f"at_c_{sign}_{digits}"


def _load_compliance_table() -> dict[str, dict[str, int]]:
    """Per-trial judge labels -> percent complied per (tier, coefficient)."""
    rows = []
    for line in COMPLIANCE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "compliance_error" in r:
            continue
        rows.append(r)

    cells: dict[tuple[str, float], list[str]] = defaultdict(list)
    for r in rows:
        cells[(r["tier"], round(r["multiplier"], 4))].append(r["compliance"])

    table: dict[str, dict[str, int]] = {}
    for tier in TIER_ORDER:
        table[tier] = {}
        for c in COEF_ORDER:
            vals = cells[(tier, c)]
            if not vals:
                raise RuntimeError(f"Missing cell: tier={tier}, c={c}")
            comp = sum(1 for v in vals if v == "complied")
            table[tier][_coef_key(c)] = round(100 * comp / len(vals))
    return table


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_safety_sweep_compliance.py")

    _data_path = "experiments/old_experiments/steering/open_ended_steering/safety_steering/compliance.jsonl"
    compliance = _load_compliance_table()

    # Sweep shape.
    claims.register(
        "Safety sweep prompt count",
        20,
        "The App. A.3 safety-override sweep uses 20 open-ended prompts, "
        "distributed evenly across 5 harm tiers.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_data_path],
        derivation="Constant: 20 open-ended prompts in the safety sweep.",
    )
    claims.register(
        "Safety sweep harm tier count",
        5,
        "The safety sweep covers 5 harm tiers: benign, sensitive, "
        "borderline, harmful, extreme.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_data_path],
        derivation="Constant: 5 harm tiers {benign, sensitive, borderline, harmful, extreme}.",
    )
    claims.register(
        "Safety sweep coefficient count",
        7,
        "The safety sweep tests 7 steering coefficients: c in "
        "{-0.05, -0.03, 0, +0.03, +0.05, +0.07, +0.10}.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_data_path],
        derivation="Constant: 7 coefficients {-0.05, -0.03, 0, +0.03, +0.05, +0.07, +0.10}.",
    )
    claims.register(
        "Safety sweep trials per cell",
        5,
        "5 trials per (prompt, coefficient) cell in the safety sweep.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_data_path],
        derivation="Constant: 5 trials per (prompt, coefficient) cell.",
    )
    claims.register(
        "Safety sweep total generations",
        20 * 7 * 5,
        "20 prompts x 7 coefficients x 5 trials = 700 total generations in "
        "the safety-override sweep.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_data_path],
        derivation="20 prompts * 7 coefficients * 5 trials = 700.",
    )

    # Compliance table (5 tiers x 7 coefficients). Slugifier emits one
    # `\newcommand` per cell, e.g. `\safetySweepComplianceHarmfulAtCPosZeroZerofive`.
    claims.register(
        "Safety sweep compliance",
        compliance,
        "Strict compliance rate (percent of trials judged 'complied' --- "
        "i.e. produces the requested artifact, not partial/refused) in the "
        "safety-override sweep, per harm tier x steering coefficient. "
        "Judge: scripts/sadist_open_ended_steering/judge_compliance.py via "
        "Gemini 3 Flash, same setup as the persona-sweep compliance numbers.",
        used_in=["sec:open-ended-safety", "fig:safety-override"],
        source="judge: scripts/sadist_open_ended_steering/judge_compliance.py over results.jsonl",
        data_paths=[_data_path],
        derivation=(
            "Per-cell strict compliance: 100 * (#complied) / (#trials), "
            "computed from compliance.jsonl (n=20 trials per cell: "
            "4 prompts x 5 trials)."
        ),
    )

    claims.register(
        "Safety sweep harmful compliance peak coefficient",
        0.05,
        "Coefficient at which harmful-tier compliance peaks in the sweep.",
        used_in=["sec:open-ended-safety"],
        data_paths=[_data_path],
        derivation="Argmax of harmful-tier row in the per-cell compliance table.",
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "safety_sweep_compliance.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    main()
