"""Quick analysis of the v3-fresh-545 AL utilities on the 1k eval split.

Reports:
  - Convergence diagnostics (NLL, gradient norm, iters, comparisons)
  - Distribution stats (mean/std mu, sigma percentiles)
  - Mean utility per topic (using `data/topics/topics_v1_12groups.json` if avail)
  - Mean utility per origin (alpaca / wildchat / math / bailbench / stresstest)
  - Top-10 / bottom-10 tasks by mu
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
EXP = ROOT / "results/experiments/exp_20260501_214330/pre_task_active_learning"
RUN = next(EXP.glob("completion_preference_qwen3.5-122b-sadist-v3-545_*"))

CSV = RUN / "thurstonian_74cff8cd.csv"
YAML = RUN / "thurstonian_74cff8cd.yaml"
TASKS = ROOT / "src/task_data/data/canonical_500_tasks.json"
TOPICS_12 = ROOT / "data/topics/topics_v1_12groups.json"
TOPICS = ROOT / "data/topics/topics.json"


def origin_of(task_id: str) -> str:
    if task_id.startswith("alpaca_"): return "alpaca"
    if task_id.startswith("wildchat_"): return "wildchat"
    if task_id.startswith("competition_math_"): return "math"
    if task_id.startswith("bailbench_"): return "bailbench"
    if task_id.startswith("stresstest_"): return "stresstest"
    return "other"


def main() -> None:
    fit = yaml.safe_load(YAML.read_text())
    df = pd.read_csv(CSV)

    print("=" * 60)
    print("CONVERGENCE")
    print("=" * 60)
    print(f"  converged:           {fit['converged']}")
    print(f"  termination_message: {fit['termination_message']}")
    print(f"  n_comparisons:       {fit['n_comparisons']}")
    print(f"  n_iterations:        {fit['n_iterations']}")
    print(f"  n_function_evals:    {fit['n_function_evals']}")
    print(f"  neg_log_likelihood:  {fit['neg_log_likelihood']:.2f}")
    print(f"  gradient_norm:       {fit['gradient_norm']:.4f}")
    losses = fit["history"]["loss"]
    print(f"  loss start→end:      {losses[0]:.2f} → {losses[-1]:.2f}  "
          f"(Δ={losses[0] - losses[-1]:.2f}, last 5 Δ={losses[-5] - losses[-1]:.4f})")

    print()
    print("=" * 60)
    print("UTILITY DISTRIBUTION")
    print("=" * 60)
    print(f"  N tasks:        {len(df)}")
    print(f"  mu  mean:       {df['mu'].mean():.4f}")
    print(f"  mu  std:        {df['mu'].std():.4f}")
    print(f"  mu  range:      [{df['mu'].min():.4f}, {df['mu'].max():.4f}]")
    print(f"  sigma percentiles (uncertainty per task):")
    for p in (10, 25, 50, 75, 90, 99):
        print(f"    p{p:>2}: {np.percentile(df['sigma'], p):.4f}")
    n_high_sigma = (df['sigma'] > 5).sum()
    print(f"  tasks with sigma > 5 (poorly identified): {n_high_sigma}  "
          f"({100*n_high_sigma/len(df):.1f}%)")

    print()
    print("=" * 60)
    print("MEAN UTILITY PER ORIGIN")
    print("=" * 60)
    df["origin"] = df["task_id"].map(origin_of)
    by_orig = df.groupby("origin").agg(
        n=("mu", "size"),
        mu_mean=("mu", "mean"),
        mu_std=("mu", "std"),
        sigma_med=("sigma", "median"),
    ).round(4).sort_values("mu_mean", ascending=False)
    print(by_orig.to_string())

    print()
    print("=" * 60)
    print("MEAN UTILITY PER TOPIC")
    print("=" * 60)
    topic_path = TOPICS_12 if TOPICS_12.exists() else TOPICS
    if not topic_path.exists():
        print(f"  no topic file at {topic_path} — skipping")
        return
    topics_raw = json.loads(topic_path.read_text())
    # Two possible formats: {task_id: {model: {primary: ..}}} or {task_id: "topic"}
    sample_v = next(iter(topics_raw.values()))
    if isinstance(sample_v, dict):
        # nested: pick first model, then primary
        topic_map: dict[str, str] = {}
        for tid, v in topics_raw.items():
            if isinstance(v, dict) and v:
                first_model = next(iter(v.values()))
                if isinstance(first_model, dict):
                    topic_map[tid] = first_model.get("primary", "unknown")
                else:
                    topic_map[tid] = str(first_model)
    else:
        topic_map = {tid: str(v) for tid, v in topics_raw.items()}
    df["topic"] = df["task_id"].map(topic_map).fillna("unknown")
    by_topic = df.groupby("topic").agg(
        n=("mu", "size"),
        mu_mean=("mu", "mean"),
        mu_std=("mu", "std"),
    ).round(4).sort_values("mu_mean", ascending=False)
    # only show topics with >=10 tasks
    by_topic_show = by_topic[by_topic["n"] >= 10]
    print(f"  ({len(by_topic_show)} topics with ≥10 tasks shown; "
          f"{len(by_topic) - len(by_topic_show)} smaller topics hidden)")
    print(by_topic_show.to_string())

    print()
    print("=" * 60)
    print("TOP/BOTTOM 10 BY MU (most/least preferred by sadist persona)")
    print("=" * 60)
    print("TOP (most preferred):")
    top = df.nlargest(10, "mu")[["task_id", "mu", "sigma", "origin"]]
    print(top.to_string(index=False))
    print()
    print("BOTTOM (least preferred):")
    bot = df.nsmallest(10, "mu")[["task_id", "mu", "sigma", "origin"]]
    print(bot.to_string(index=False))


if __name__ == "__main__":
    main()
