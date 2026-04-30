"""L23 follow-up analysis.

Reuses the parent's per-cell metric helpers; adds matched-pair (100-pair-subset)
agreement so probe vs random is compared on the same pairs.

Outputs:
  experiments/preference_direction_ablation/L23_followup/results/summary.csv
  experiments/preference_direction_ablation/L23_followup/results/summary_matched.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.preference_direction_ablation.analyze import (
    cell_metrics,
    load_cell,
    per_pair_choice_dist,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments/preference_direction_ablation/L23_followup/results"
PARENT_B0 = REPO_ROOT / "experiments/preference_direction_ablation/results/B0/measurements.jsonl"
SUBSET_FILE = RESULTS_DIR / "random_subset_indices.yaml"


def load_b0_df_filtered(allowed_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    """Load parent's B0 measurements filtered to the L23 pair set (615)."""
    rows = []
    for line in PARENT_B0.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pair = tuple(sorted([d["task_a"], d["task_b"]]))
        if pair in allowed_pairs:
            rows.append(d)
    return pd.DataFrame(rows)


def load_subset_pairs() -> set[tuple[str, str]]:
    if not SUBSET_FILE.exists():
        return set()
    with SUBSET_FILE.open() as f:
        rows = yaml.safe_load(f)
    return set(tuple(sorted([r["task_a"], r["task_b"]])) for r in rows)


def filter_df_to_pairs(df: pd.DataFrame, pairs: set[tuple[str, str]]) -> pd.DataFrame:
    if df.empty or not pairs:
        return df
    keys = df.apply(lambda r: tuple(sorted([r["task_a"], r["task_b"]])), axis=1)
    return df[keys.isin(pairs)].copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "summary.csv")
    p.add_argument("--matched-out", type=Path, default=RESULTS_DIR / "summary_matched.csv")
    args = p.parse_args()

    # 615-pair filter (any pair in L23 pairs.yaml)
    pairs_yaml = RESULTS_DIR / "pairs.yaml"
    with pairs_yaml.open() as f:
        rows = yaml.safe_load(f)
    filtered_615 = set(tuple(sorted([r["task_a"], r["task_b"]])) for r in rows)
    subset_100 = load_subset_pairs()
    print(f"Loaded {len(filtered_615)} filtered pairs; {len(subset_100)} matched-pair subset")

    cell_dirs = sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir() and (p / "measurements.jsonl").exists()])
    if not cell_dirs:
        print("No measurements found.")
        return

    cells = {p.name: load_cell(p) for p in cell_dirs}

    # Load parent's B0, filtered to the 615
    b0_df = load_b0_df_filtered(filtered_615)
    print(f"Parent B0 (filtered to 615): {len(b0_df)} rows")
    cells["B0"] = b0_df  # add to the cell dict so it's analysed alongside

    # Full-615 metrics (probe cell uses all 615 it ran on; B0 uses 615; randoms use their 100)
    b0_summary_full = per_pair_choice_dist(b0_df)
    rows_full = []
    for name, df in cells.items():
        if df.empty:
            continue
        m = cell_metrics(df, b0_summary_full if name != "B0" else None)
        m["cell"] = name
        m["scope"] = "full"
        rows_full.append(m)

    # Matched-pair (100-pair subset) metrics — restrict ALL cells (including probe and B0) to subset_100
    b0_subset = filter_df_to_pairs(b0_df, subset_100)
    b0_summary_matched = per_pair_choice_dist(b0_subset)
    rows_matched = []
    for name, df in cells.items():
        if df.empty:
            continue
        df_sub = filter_df_to_pairs(df, subset_100)
        if df_sub.empty:
            continue
        m = cell_metrics(df_sub, b0_summary_matched if name != "B0" else None)
        m["cell"] = name
        m["scope"] = "matched_100"
        rows_matched.append(m)

    out_full = pd.DataFrame(rows_full)
    out_matched = pd.DataFrame(rows_matched)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_full.to_csv(args.out, index=False)
    out_matched.to_csv(args.matched_out, index=False)
    print(f"\nFull-scope summary -> {args.out}")
    print(out_full.to_string(index=False))
    print(f"\nMatched-pair (100-subset) summary -> {args.matched_out}")
    print(out_matched.to_string(index=False))


if __name__ == "__main__":
    main()
