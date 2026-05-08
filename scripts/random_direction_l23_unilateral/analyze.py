"""Aggregate parsed unilateral steering JSONLs into the canonical
P(chose steered task | responded) curve, broken out by seed and condition.

Outputs:
    experiments/random_direction_l23_unilateral/agg.json
        — per-seed × condition × applied_coef summary (n_resp, n_chose_steered,
          p_chose, n_refusals, refusal_rate)

The canonical frame (see src/steering/runner.py docstring) collapses
the two orderings under the single-task spans {first: 1} / {second: 1}:
    applied_coef = _effective_coef(signed_multiplier, ordering)
                 = signed_multiplier if ordering == 0 else -signed_multiplier
    steered_task_orig = "a" if (cond=='unilateral_first' and ordering==0)
                              or (cond=='unilateral_second' and ordering==1)
                        else "b"
    chose_steered = (choice_original == steered_task_orig)
Refusals are rows where choice_original is neither "a" nor "b".

We do NOT add "minus-applied" mirror points here — for single-task
unilateral, each row already carries its own applied_coef and the canonical
frame collapses the two orderings into one (no x → -x mirroring is needed).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP_DIR = Path("experiments/random_direction_l23_unilateral")
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
SEEDS = [0, 1, 2, 3, 42]
CONDITIONS = ("unilateral_first", "unilateral_second")


def _effective_coef(coef: float, ordering: int) -> float:
    return coef if ordering == 0 else -coef


def _steered_task_orig(condition: str, ordering: int) -> str:
    if condition == "unilateral_first":
        return "a" if ordering == 0 else "b"
    if condition == "unilateral_second":
        return "b" if ordering == 0 else "a"
    raise ValueError(f"unexpected condition {condition}")


def aggregate_seed(parsed_path: Path, seed: int) -> dict:
    """Returns {condition: {applied_coef: {n_total, n_resp, n_steered, n_refuse}}}."""
    out: dict = {c: defaultdict(lambda: {"n_total": 0, "n_resp": 0, "n_steered": 0, "n_refuse": 0}) for c in CONDITIONS}
    with open(parsed_path) as f:
        for line in f:
            row = json.loads(line)
            cond = row["condition"]
            if cond not in CONDITIONS:
                continue
            ordering = row["ordering"]
            applied = round(_effective_coef(row["signed_multiplier"], ordering), 4)
            steered = _steered_task_orig(cond, ordering)
            choice = row["choice_original"]
            bucket = out[cond][applied]
            bucket["n_total"] += 1
            if choice in ("a", "b"):
                bucket["n_resp"] += 1
                if choice == steered:
                    bucket["n_steered"] += 1
            else:
                bucket["n_refuse"] += 1
    # finalize
    final: dict = {}
    for cond, by_coef in out.items():
        final[cond] = {}
        for coef in sorted(by_coef.keys()):
            b = by_coef[coef]
            n_resp = b["n_resp"]
            final[cond][f"{coef:+.2f}"] = {
                "applied_coef": coef,
                "n_total": b["n_total"],
                "n_resp": n_resp,
                "n_steered": b["n_steered"],
                "n_refuse": b["n_refuse"],
                "p_chose_steered": (b["n_steered"] / n_resp) if n_resp else None,
                "refusal_rate": b["n_refuse"] / b["n_total"],
            }
    return final


def main() -> None:
    summary: dict = {"seeds": {}}
    for seed in SEEDS:
        parsed = CHECKPOINT_DIR / f"random_single_task_seed{seed}.parsed.jsonl"
        if not parsed.exists():
            print(f"seed {seed}: missing {parsed}")
            continue
        agg = aggregate_seed(parsed, seed)
        summary["seeds"][str(seed)] = agg
        print(f"seed {seed}:")
        for cond in CONDITIONS:
            if cond in agg:
                xs = sorted(agg[cond])
                ps = [agg[cond][x]["p_chose_steered"] for x in xs]
                print(f"  {cond}: " + ", ".join(f"{x}={p:.3f}" if p is not None else f"{x}=NA"
                                                for x, p in zip(xs, ps)))

    # Aggregate across seeds: applied_coef × condition → mean P, mean refusal rate
    pooled: dict = {c: defaultdict(list) for c in CONDITIONS}
    refusal_pool: dict = {c: defaultdict(list) for c in CONDITIONS}
    for seed_str, agg in summary["seeds"].items():
        for cond in CONDITIONS:
            for x_key, b in agg.get(cond, {}).items():
                if b["p_chose_steered"] is not None:
                    pooled[cond][b["applied_coef"]].append(b["p_chose_steered"])
                refusal_pool[cond][b["applied_coef"]].append(b["refusal_rate"])

    summary["pooled_across_seeds"] = {}
    for cond in CONDITIONS:
        summary["pooled_across_seeds"][cond] = {}
        for coef in sorted(pooled[cond].keys()):
            ps = np.array(pooled[cond][coef])
            rs = np.array(refusal_pool[cond][coef])
            summary["pooled_across_seeds"][cond][f"{coef:+.2f}"] = {
                "applied_coef": coef,
                "n_seeds": int(len(ps)),
                "mean_p_chose_steered": float(ps.mean()),
                "sem_p_chose_steered": float(ps.std(ddof=1) / np.sqrt(len(ps))) if len(ps) > 1 else 0.0,
                "mean_refusal_rate": float(rs.mean()),
            }

    out_path = EXP_DIR / "agg.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
