"""Load steering checkpoints and aggregate results.

The key invariant: `signed_multiplier` encodes steering direction in
original task space (positive = toward task_a, negative = toward task_b).
The `_effective_coef` negation in the runner ensures this is consistent
across both orderings. So the correct aggregation is always:

    group by (condition, layer, signed_multiplier)
    → P(chose task_a in original space) across both orderings

If steering works, P(task_a) should increase with signed_multiplier.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_checkpoint(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def filter_valid(rows: list[dict]) -> list[dict]:
    """Keep only rows with a valid choice (a or b), dropping refusals."""
    return [r for r in rows if r["choice_original"] in ("a", "b")]


def aggregate(
    rows: list[dict],
    group_by: list[str] = ["condition", "layer", "signed_multiplier"],
) -> list[dict]:
    """Aggregate P(chose_a) over groups, pooling across orderings.

    Returns one row per group with:
      - all group_by fields
      - p_a: P(choice_original == "a")
      - n: number of valid trials (excluding refusals)
      - n_refusal: number of refusals
      - n_by_ordering: {0: count, 1: count} — to verify balance
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_by)
        buckets[key].append(row)

    results = []
    for key, bucket in sorted(buckets.items()):
        valid = [r for r in bucket if r["choice_original"] in ("a", "b")]
        n_a = sum(1 for r in valid if r["choice_original"] == "a")
        n_by_ordering = defaultdict(int)
        for r in valid:
            n_by_ordering[r["ordering"]] += 1

        result = dict(zip(group_by, key))
        result["p_a"] = n_a / len(valid) if valid else float("nan")
        result["n"] = len(valid)
        result["n_refusal"] = len(bucket) - len(valid)
        result["n_by_ordering"] = dict(n_by_ordering)
        results.append(result)

    return results


def print_summary(
    rows: list[dict],
    group_by: list[str] = ["condition", "layer", "signed_multiplier"],
) -> None:
    """Print a compact summary table."""
    agg = aggregate(rows, group_by)

    sections: dict[tuple, list[dict]] = defaultdict(list)
    for r in agg:
        section_key = tuple(r[k] for k in group_by[:-1])
        sections[section_key].append(r)

    for section_key, section_rows in sections.items():
        header = " / ".join(f"{k}={v}" for k, v in zip(group_by[:-1], section_key))
        print(f"\n{header}:")
        mult_key = group_by[-1]
        for r in section_rows:
            mult = r[mult_key]
            p_a = r["p_a"]
            n = r["n"]
            bal = r["n_by_ordering"]
            shift = p_a - 0.5
            print(f"  {mult_key}={mult:+.3f}: P(a)={p_a:.3f} (n={n}, ord={bal}) shift={shift:+.3f}")
