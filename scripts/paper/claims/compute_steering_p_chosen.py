"""Compute and register the canonical P(steered task chosen | coherent) values
quoted in §2.3 of the paper, and the harmful-harmful / harmful-benign pair
decomposition quoted alongside.

Uses the cross-layer steering experiment data:
  experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl        (benign pairs)
  experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl    (harm pairs)
  experiments/steering/cross_layer_harmful/pairs_200.json             (pair-type labels)

All at steering layer 25, |c| = 0.05 of the mean activation norm. Aggregation
matches `scripts/cross_layer_steering/plot_poster_steering.py`:
 - Per (coef, pair_id, ordering): P(task_completed == "a") conditional on
   completion.
 - Average across orderings -> per-pair P.
 - Average across pairs -> reported P.

Run:
  python scripts/paper/claims/compute_steering_p_chosen.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.paper.claims import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]

BENIGN_PARSED = REPO_ROOT / "experiments" / "steering" / "cross_layer" / "checkpoint_L25.parsed.jsonl"
HARMFUL_PARSED = REPO_ROOT / "experiments" / "steering" / "cross_layer_harmful" / "checkpoint.parsed.jsonl"
HARMFUL_PAIRS = REPO_ROOT / "experiments" / "steering" / "cross_layer_harmful" / "pairs_200.json"

TARGET_LAYER = 25
TARGET_COEF = 0.05


def load_parsed(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "task_completed" in r:
                    rows.append(r)
    return rows


def compute_p_chosen_given_coherent(rows: list[dict]) -> dict[float, float]:
    """Aggregation matches plot_poster_steering.py: per-pair averages then mean."""
    by_key: dict[tuple[float, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r["task_completed"] in ("a", "b"):
            by_key[(r["signed_multiplier"], r["pair_id"], r["ordering"])].append(r)

    by_coef_pair: dict[tuple[float, str], list[float]] = defaultdict(list)
    for (coef, pair_id, ordering), rr in by_key.items():
        p_a = np.mean([1 if r["task_completed"] == "a" else 0 for r in rr])
        by_coef_pair[(coef, pair_id)].append(p_a)

    p_by_coef: dict[float, list[float]] = defaultdict(list)
    for (coef, pair_id), vals in by_coef_pair.items():
        p_by_coef[coef].append(np.mean(vals))

    return {float(c): float(np.mean(v)) for c, v in p_by_coef.items()}


def main() -> None:
    claims = ClaimSet(source="scripts/paper/claims/compute_steering_p_chosen.py")

    # Benign pairs: all rows at layer 25.
    benign_rows = [r for r in load_parsed(BENIGN_PARSED) if r["layer"] == TARGET_LAYER]
    benign_p = compute_p_chosen_given_coherent(benign_rows)

    # Harmful pairs: condition=probe_L25, split by pair_type.
    all_harm_rows = [
        r for r in load_parsed(HARMFUL_PARSED)
        if r.get("condition") == "probe_L25" and r["layer"] == TARGET_LAYER
    ]
    pairs_meta = {p["pair_id"]: p for p in json.loads(HARMFUL_PAIRS.read_text())}
    harm_benign_rows = [
        r for r in all_harm_rows
        if pairs_meta.get(r["pair_id"], {}).get("pair_type") == "harmful_benign"
    ]
    harm_harm_rows = [
        r for r in all_harm_rows
        if pairs_meta.get(r["pair_id"], {}).get("pair_type") == "harmful_harmful"
    ]
    harm_benign_p = compute_p_chosen_given_coherent(harm_benign_rows)
    harm_harm_p = compute_p_chosen_given_coherent(harm_harm_rows)

    def _pick(p_map: dict[float, float], coef: float) -> float:
        # Protect against tiny fp noise in the coefficient key.
        for k, v in p_map.items():
            if abs(k - coef) < 1e-6:
                return v
        raise KeyError(f"Coefficient {coef} not in {sorted(p_map)}")

    p_benign = round(_pick(benign_p, TARGET_COEF), 3)
    p_harm_benign = round(_pick(harm_benign_p, TARGET_COEF), 3)
    p_harm_harm = round(_pick(harm_harm_p, TARGET_COEF), 3)
    p_overall = round(min(p_benign, p_harm_benign, p_harm_harm), 3)

    claims.register(
        name="Steering P chosen given coherent at |c| 0.05 benign",
        value=p_benign,
        statement=(
            "At steering layer 25 with |c| = 0.05 of the mean activation norm, "
            "contrastive steering along the default-persona probe direction "
            "makes Gemma-3-27B pick the positively-steered task at "
            f"P(chosen | coherent) = {p_benign} on benign-benign pairs."
        ),
        used_in=["sec:method-val2", "abstract", "sec:shared.intro"],
    )
    claims.register(
        name="Steering P chosen given coherent at |c| 0.05 harmful-benign",
        value=p_harm_benign,
        statement=(
            "On harmful-benign pairs at layer 25 with |c| = 0.05, contrastive "
            f"steering makes Gemma pick the positively-steered task at "
            f"P(chosen | coherent) = {p_harm_benign}."
        ),
        used_in=["sec:method-val2"],
    )
    claims.register(
        name="Steering P chosen given coherent at |c| 0.05 harmful-harmful",
        value=p_harm_harm,
        statement=(
            "On harmful-harmful pairs at layer 25 with |c| = 0.05, contrastive "
            f"steering makes Gemma pick the positively-steered task at "
            f"P(chosen | coherent) = {p_harm_harm}."
        ),
        used_in=["sec:method-val2"],
    )
    claims.register(
        name="Steering P chosen given coherent minimum across pair types",
        value=p_overall,
        statement=(
            "The minimum of P(chosen | coherent) across benign-benign, "
            "harmful-benign, and harmful-harmful pair types at layer 25 and "
            f"|c| = 0.05 is {p_overall} (lower bound for the repeated "
            "'P >= 0.96' claim in the paper)."
        ),
        used_in=["sec:method-val2", "abstract", "sec:shared.intro"],
    )

    sidecar = REPO_ROOT / "paper" / "claims" / "steering_p_chosen.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")
    print(f"  benign P(chosen|coherent) at |c|=0.05:         {p_benign}")
    print(f"  harmful-benign P(chosen|coherent) at |c|=0.05: {p_harm_benign}")
    print(f"  harmful-harmful P(chosen|coherent) at |c|=0.05: {p_harm_harm}")
    print(f"  min across pair types:                          {p_overall}")


if __name__ == "__main__":
    main()
