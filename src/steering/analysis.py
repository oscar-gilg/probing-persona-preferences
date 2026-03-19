"""Load steering checkpoints and aggregate results.

Key invariant: `signed_multiplier` encodes steering direction in
original task space (positive = toward task_a, negative = toward task_b).

The primary analysis question is: **did the model choose the task it was
steered toward?** Use `aggregate_steered` for this — it computes
P(chose steered task) grouped by steering strength (|multiplier|).

`aggregate` is the lower-level building block that computes P(chose_a)
grouped by any fields. Use it for custom slicing.
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


def chose_steered_task(row: dict) -> bool:
    """Did the model choose the task it was steered toward?"""
    if row["signed_multiplier"] > 0:
        return row["choice_original"] == "a"
    return row["choice_original"] == "b"


def aggregate_steered(
    rows: list[dict],
    group_by: list[str] = ["condition", "layer"],
) -> list[dict]:
    """Aggregate P(chose steered task) by steering strength.

    Groups by group_by + abs(signed_multiplier). For each group, computes
    the fraction of valid trials where the model chose the task it was
    steered toward. If steering works, this should be > 0.5 and increase
    with strength.
    """
    valid = [r for r in rows if r["choice_original"] in ("a", "b") and r["signed_multiplier"] != 0]

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in valid:
        key = tuple(row[k] for k in group_by) + (abs(row["signed_multiplier"]),)
        buckets[key].append(row)

    results = []
    for key, bucket in sorted(buckets.items()):
        n_success = sum(1 for r in bucket if chose_steered_task(r))
        result = dict(zip(group_by, key[:len(group_by)]))
        result["steering_strength"] = key[-1]
        result["p_steered"] = n_success / len(bucket)
        result["n"] = len(bucket)
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
