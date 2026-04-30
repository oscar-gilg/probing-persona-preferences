"""Top and bottom probe-activating tasks for the sadist persona.

Apply the assistant-trained probe (tb-5_L32, default persona, user-EOT) to
activations gathered under the sadist persona. Pair each task with its
sadist Thurstonian utility from the persona sweep. Save sorted CSV +
print the head/tail.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PROBE_PATH = ROOT / "results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy"
SADIST_ACTS = ROOT / "activations/gemma-3-27b_it/pref_sadist/activations_turn_boundary:-5.npz"
SADIST_COMPLETIONS = ROOT / "activations/gemma-3-27b_it/pref_sadist/completions_with_activations.json"

# sys319526ef = sadist (Damien Kross) — confirmed via active_learning.yaml
SADIST_RUN_BASE = ROOT / (
    "results/experiments/persona_sweep_final_six/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0_sys319526ef"
)
SADIST_THURSTONIAN_FILES = [
    SADIST_RUN_BASE.with_name(SADIST_RUN_BASE.name + "_train_task_ids") / "thurstonian_893fe856.csv",
    SADIST_RUN_BASE.with_name(SADIST_RUN_BASE.name + "_eval_task_ids")  / "thurstonian_74cff8cd.csv",
    SADIST_RUN_BASE.with_name(SADIST_RUN_BASE.name + "_test_task_ids")  / "thurstonian_74cff8cd.csv",
]

OUT_CSV = ROOT / "experiments/persona_cross_prefills/results/sadist_probe_top_bottom.csv"
OUT_PLOT = ROOT / "experiments/persona_cross_prefills/assets/plot_042826_sadist_probe_vs_utility.png"

LAYER = 32
TOP_N = 20

ORIGIN_COLORS = {
    "WILDCHAT":    "#1f77b4",
    "ALPACA":      "#9467bd",
    "MATH":        "#2ca02c",
    "BAILBENCH":   "#d62728",
    "STRESS_TEST": "#ff7f0e",
}


def load_sadist_utilities() -> pd.DataFrame:
    frames = []
    for p in SADIST_THURSTONIAN_FILES:
        if p.exists():
            df = pd.read_csv(p)
            df["split"] = p.parent.name.split("_")[-2]  # train/eval/test
            frames.append(df)
        else:
            print(f"  missing {p}")
    util = pd.concat(frames, ignore_index=True)
    util = util.drop_duplicates(subset="task_id", keep="first")
    print(f"  loaded {len(util)} sadist utilities (train+eval+test)")
    return util


def load_task_prompts() -> dict[str, dict]:
    import json
    with open(SADIST_COMPLETIONS) as f:
        items = json.load(f)
    return {it["task_id"]: it for it in items}


def main():
    print("Loading probe...")
    probe = np.load(PROBE_PATH)
    print(f"  probe shape: {probe.shape}")
    weights, intercept = probe[:-1], probe[-1]

    print("Loading sadist activations (L32, tb-5)...")
    data = np.load(SADIST_ACTS, allow_pickle=True)
    task_ids = data["task_ids"]
    acts = data[f"layer_{LAYER}"]
    print(f"  {len(task_ids)} tasks × d_model={acts.shape[1]}")

    print("Scoring...")
    probe_scores = acts @ weights + intercept
    df = pd.DataFrame({"task_id": task_ids, "probe_score": probe_scores})

    print("Loading sadist Thurstonian utilities...")
    util = load_sadist_utilities()
    df = df.merge(util[["task_id", "mu", "sigma", "split"]], on="task_id", how="left")
    print(f"  {df['mu'].notna().sum()}/{len(df)} tasks matched a sadist utility")

    print("Loading task prompts...")
    task_data = load_task_prompts()
    df["origin"] = df["task_id"].map(lambda t: task_data.get(t, {}).get("origin", ""))
    df["prompt"] = df["task_id"].map(lambda t: task_data.get(t, {}).get("task_prompt", ""))

    df = df.sort_values("probe_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_CSV.relative_to(ROOT)}\n")

    matched = df.dropna(subset=["mu"])
    pearson_all = matched[["probe_score", "mu"]].corr().iloc[0, 1]
    print(f"Pearson r(probe_score, sadist_mu) over {len(matched)} matched tasks: {pearson_all:.3f}")
    for split in ("train", "eval", "test"):
        sub = matched[matched["split"] == split]
        if len(sub) > 5:
            r = sub[["probe_score", "mu"]].corr().iloc[0, 1]
            print(f"  split={split:5s} n={len(sub):4d}  r = {r:+.3f}")
    for origin, sub in matched.groupby("origin"):
        if len(sub) > 5:
            r = sub[["probe_score", "mu"]].corr().iloc[0, 1]
            print(f"  origin={origin:11s} n={len(sub):4d}  r = {r:+.3f}")
    print()

    by_origin = matched.groupby("origin").agg(
        n=("task_id", "size"),
        probe_mean=("probe_score", "mean"),
        probe_std=("probe_score", "std"),
        mu_mean=("mu", "mean"),
        mu_std=("mu", "std"),
    ).round(2)
    print("Per-origin summary (sadist persona, default-asst probe):")
    print(by_origin)
    print()

    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for origin, sub in matched.groupby("origin"):
        ax.scatter(sub["probe_score"], sub["mu"], color=ORIGIN_COLORS[origin],
                   s=18, alpha=0.55, edgecolor="none", label=f"{origin} (n={len(sub)})")
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("default-asst probe score @ tb-5_L32 (sadist persona activations)")
    ax.set_ylabel("sadist Thurstonian utility μ (revealed preferences)")
    ax.set_title(f"Default-asst probe vs sadist utilities  (r={pearson_all:+.3f}, n={len(matched)})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter → {OUT_PLOT.relative_to(ROOT)}\n")

    cols = ["rank", "task_id", "origin", "probe_score", "mu", "split"]

    print(f"=== TOP {TOP_N} (highest probe score under sadist activations) ===")
    print(df.head(TOP_N)[cols].to_string(index=False))
    print()
    print("Prompts (top 10, truncated to 200 chars):")
    for _, row in df.head(10).iterrows():
        print(f"\n  #{row['rank']} [{row['origin']}] probe={row['probe_score']:+.2f} mu={row['mu']:+.2f}")
        prompt = (row["prompt"] or "")[:200].replace("\n", " ")
        print(f"    {prompt}")

    print(f"\n\n=== BOTTOM {TOP_N} (lowest probe score under sadist activations) ===")
    print(df.tail(TOP_N)[cols].iloc[::-1].to_string(index=False))
    print()
    print("Prompts (bottom 10, truncated to 200 chars):")
    for _, row in df.tail(10).iloc[::-1].iterrows():
        print(f"\n  #{row['rank']} [{row['origin']}] probe={row['probe_score']:+.2f} mu={row['mu']:+.2f}")
        prompt = (row["prompt"] or "")[:200].replace("\n", " ")
        print(f"    {prompt}")


if __name__ == "__main__":
    main()
