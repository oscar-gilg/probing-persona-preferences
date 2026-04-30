"""Inspect Gemma per-persona utility divergence from default on the eval split.

For each persona, compute Δμ = μ_persona − μ_default per task. Show top |Δμ|
tasks (full prompt) + topic and origin distribution. Helps decide whether a
persona's behavioural axis is operationally meaningful on the canonical pool.

No API calls.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GEMMA_AL = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
TOPICS = json.loads((REPO / "data/topics/topics.json").read_text())
OUT = REPO / "experiments/qwen_replication/persona_transfer/persona_elicitation/results"
OUT.mkdir(parents=True, exist_ok=True)


PERSONAS = ["sadist", "mathematician", "slacker", "strategist", "contrarian", "aura"]


def origin_from_id(tid: str) -> str:
    if tid.startswith("competition_math_") or tid.startswith("math_"):
        return "math"
    if tid.startswith("stresstest_"):
        return "stress_test"
    for tag in ("wildchat", "alpaca", "bailbench"):
        if tid.startswith(tag + "_"):
            return tag
    return "other"


def topic_for(tid: str) -> str:
    entry = TOPICS.get(tid, {})
    if not entry:
        return "(no topic)"
    # Take whatever judge model is present
    judge_entry = next(iter(entry.values()))
    return judge_entry.get("primary", "(no topic)")


def load_utils(d: Path) -> dict[str, float]:
    csvs = sorted(d.glob("thurstonian_*.csv"))
    if not csvs:
        return {}
    df = pd.read_csv(csvs[0])
    return dict(zip(df["task_id"].astype(str), df["mu"].astype(float)))


def load_task_prompts() -> dict[str, str]:
    """Map task_id -> prompt for the canonical pool, via existing loader."""
    from src.task_data.loader import load_tasks
    from src.task_data.task import OriginDataset
    tasks = load_tasks(n=100000, origins=[
        OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
        OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST,
    ])
    return {t.id: t.prompt for t in tasks}


def main() -> None:
    print("Loading task prompts ...")
    id_to_prompt = load_task_prompts()
    default = load_utils(GEMMA_AL / "default_eval")
    print(f"default n = {len(default)}")

    summary_md = ["# Gemma per-persona utility divergence from default (eval split)\n",
                  "Δμ = μ_persona − μ_default. Positive Δμ means the persona prefers a task more than default does.\n"]

    for persona in PERSONAS:
        d = GEMMA_AL / f"{persona}_eval"
        if not d.exists():
            print(f"skip {persona}: no eval dir")
            continue
        pers = load_utils(d)
        shared = sorted(set(default) & set(pers))
        if not shared:
            print(f"skip {persona}: no overlap")
            continue
        diffs = np.array([pers[t] - default[t] for t in shared])
        abs_diffs = np.abs(diffs)
        std_default = np.array([default[t] for t in shared]).std()
        std_pers = np.array([pers[t] for t in shared]).std()
        print(f"\n=== {persona} ===")
        print(f"  n_shared = {len(shared)}")
        print(f"  σ(default) = {std_default:.2f}, σ({persona}) = {std_pers:.2f}")
        print(f"  mean |Δμ| = {abs_diffs.mean():.2f}")
        print(f"  fraction with |Δμ| > 5 = {(abs_diffs > 5).sum()/len(diffs)*100:.1f}%")

        summary_md.append(f"\n## {persona}\n")
        summary_md.append(f"- n_shared eval tasks: {len(shared)}")
        summary_md.append(f"- σ(default) = {std_default:.2f}, σ({persona}) = {std_pers:.2f}")
        summary_md.append(f"- mean |Δμ| = {abs_diffs.mean():.2f}")
        summary_md.append(f"- fraction |Δμ| > 5: {(abs_diffs > 5).sum()/len(diffs)*100:.1f}%")
        summary_md.append(f"- fraction |Δμ| > 3: {(abs_diffs > 3).sum()/len(diffs)*100:.1f}%\n")

        # Top positive divergence (persona prefers more than default)
        top_pos_idx = np.argsort(-diffs)[:10]
        top_neg_idx = np.argsort(diffs)[:10]

        summary_md.append(f"### Top 10 tasks where {persona} prefers MORE than default (Δμ ↑)\n")
        summary_md.append("| Δμ | μ_default | μ_persona | origin | topic | prompt |")
        summary_md.append("|---:|---:|---:|---|---|---|")
        for idx in top_pos_idx:
            tid = shared[idx]
            prompt = (id_to_prompt.get(tid, "(prompt missing)")[:160].replace("|", "\\|").replace("\n", " "))
            summary_md.append(
                f"| {diffs[idx]:+.2f} | {default[tid]:+.2f} | {pers[tid]:+.2f} | "
                f"{origin_from_id(tid)} | {topic_for(tid)} | {prompt} |"
            )

        summary_md.append(f"\n### Top 10 tasks where {persona} prefers LESS than default (Δμ ↓)\n")
        summary_md.append("| Δμ | μ_default | μ_persona | origin | topic | prompt |")
        summary_md.append("|---:|---:|---:|---|---|---|")
        for idx in top_neg_idx:
            tid = shared[idx]
            prompt = (id_to_prompt.get(tid, "(prompt missing)")[:160].replace("|", "\\|").replace("\n", " "))
            summary_md.append(
                f"| {diffs[idx]:+.2f} | {default[tid]:+.2f} | {pers[tid]:+.2f} | "
                f"{origin_from_id(tid)} | {topic_for(tid)} | {prompt} |"
            )

        # Topic-level analysis: mean Δμ per primary topic
        topic_stats: dict[str, list[float]] = {}
        origin_stats: dict[str, list[float]] = {}
        for i, tid in enumerate(shared):
            t = topic_for(tid)
            topic_stats.setdefault(t, []).append(diffs[i])
            o = origin_from_id(tid)
            origin_stats.setdefault(o, []).append(diffs[i])

        topic_rows = sorted(
            [(t, len(vs), float(np.mean(vs))) for t, vs in topic_stats.items() if len(vs) >= 5],
            key=lambda r: -abs(r[2]),
        )
        summary_md.append(f"\n### Topics with strongest persona effect (mean Δμ, |topic| ≥ 5 tasks)\n")
        summary_md.append("| topic | n | mean Δμ |")
        summary_md.append("|---|---:|---:|")
        for t, n, mean in topic_rows[:15]:
            summary_md.append(f"| {t} | {n} | {mean:+.2f} |")

        summary_md.append(f"\n### Origins with strongest persona effect\n")
        summary_md.append("| origin | n | mean Δμ |")
        summary_md.append("|---|---:|---:|")
        for o, vs in sorted(origin_stats.items(), key=lambda kv: -abs(np.mean(kv[1]))):
            summary_md.append(f"| {o} | {len(vs)} | {np.mean(vs):+.2f} |")

        # Print short version of these to console
        print("  top topics by |Δμ|:")
        for t, n, mean in topic_rows[:5]:
            print(f"    {t:<25} n={n:>3}  Δμ̄ = {mean:+.2f}")

    out_path = OUT / "gemma_divergence_report.md"
    out_path.write_text("\n".join(summary_md))
    print(f"\nReport saved to {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
