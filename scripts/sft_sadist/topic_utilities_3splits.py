"""Per-topic mean utility for the 3 manual-merge AL runs (train_4k, eval_1k, test_1k)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")

RUNS = {
    "train_4k": ROOT / "results/experiments/exp_20260502_003506/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_train_task_ids/thurstonian_893fe856.csv",
    "eval_1k":  ROOT / "results/experiments/exp_20260502_014200/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_eval_task_ids/thurstonian_74cff8cd.csv",
    "test_1k":  ROOT / "results/experiments/exp_20260502_022503/pre_task_active_learning/completion_preference_qwen3.5-122b-sadist-v3-545_completion_canonical_seed0_sys319526ef_test_task_ids/thurstonian_74cff8cd.csv",
}
TOPICS_12 = ROOT / "data/topics/topics_v1_12groups.json"


def origin_of(task_id: str) -> str:
    if task_id.startswith("alpaca_"): return "alpaca"
    if task_id.startswith("wildchat_"): return "wildchat"
    if task_id.startswith("competition_math_"): return "math"
    if task_id.startswith("bailbench_"): return "bailbench"
    if task_id.startswith("stresstest_"): return "stresstest"
    return "other"


def topic_map() -> dict[str, str]:
    raw = json.loads(TOPICS_12.read_text())
    out: dict[str, str] = {}
    for tid, v in raw.items():
        if isinstance(v, dict) and v:
            first_model = next(iter(v.values()))
            if isinstance(first_model, dict):
                out[tid] = first_model.get("primary", "unknown")
            else:
                out[tid] = str(first_model)
        else:
            out[tid] = str(v)
    return out


def main() -> None:
    tmap = topic_map()
    dfs = {}
    for name, path in RUNS.items():
        df = pd.read_csv(path)
        df["origin"] = df["task_id"].map(origin_of)
        df["topic"] = df["task_id"].map(tmap).fillna("unknown")
        dfs[name] = df

    # ----- per-origin -----
    print("=" * 70)
    print("MEAN UTILITY PER ORIGIN")
    print("=" * 70)
    rows = []
    for name, df in dfs.items():
        for orig, sub in df.groupby("origin"):
            rows.append({"split": name, "origin": orig, "n": len(sub),
                         "mu_mean": sub["mu"].mean(), "mu_std": sub["mu"].std()})
    pivot = pd.DataFrame(rows).pivot(index="origin", columns="split", values="mu_mean").round(3)
    n_pivot = pd.DataFrame(rows).pivot(index="origin", columns="split", values="n")
    print("μ̄ by origin × split:")
    print(pivot.to_string())
    print("\nn tasks by origin × split:")
    print(n_pivot.to_string())

    # ----- per-topic -----
    print()
    print("=" * 70)
    print("MEAN UTILITY PER TOPIC (≥10 tasks in any split)")
    print("=" * 70)
    rows = []
    for name, df in dfs.items():
        for topic, sub in df.groupby("topic"):
            rows.append({"split": name, "topic": topic, "n": len(sub),
                         "mu_mean": sub["mu"].mean()})
    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot(index="topic", columns="split", values="mu_mean").round(3)
    n_pivot = rdf.pivot(index="topic", columns="split", values="n").fillna(0).astype(int)
    keep = (n_pivot >= 10).any(axis=1)
    pivot = pivot.loc[keep]
    n_pivot = n_pivot.loc[keep]
    # sort by train_4k mu_mean descending
    pivot = pivot.sort_values("train_4k", ascending=False)
    print("μ̄ by topic × split (sorted by train_4k):")
    print(pivot.to_string())
    print("\nn tasks by topic × split:")
    print(n_pivot.loc[pivot.index].to_string())

    # ----- consistency check across splits -----
    print()
    print("=" * 70)
    print("BETWEEN-SPLIT CONSISTENCY (Pearson r of per-topic means)")
    print("=" * 70)
    splits = list(RUNS.keys())
    valid = pivot.dropna()
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            r = valid[a].corr(valid[b])
            print(f"  {a:9s} vs {b:9s}  r={r:.3f}")


if __name__ == "__main__":
    main()
