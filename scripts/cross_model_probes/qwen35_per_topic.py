"""Per-topic HOO Pearson r for Qwen 3.5 probes.

Train on N-1 topics, eval on held-out topic — from HOO summary files.

Usage:
    python -m scripts.cross_model_probes.qwen35_per_topic
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HOO_DIR = Path("results/probes/qwen35_122b")
ASSETS_DIR = Path("experiments/training_probes/qwen35_probes/assets")

SELECTORS = {
    "tb-1": "turn_boundary_m1",
    "tb-2": "turn_boundary_m2",
    "tb-4": "turn_boundary_m4",
    "tb-5": "turn_boundary_m5",
}

BEST_LAYER = 38

COLORS = {
    "tb-1": "#1f77b4",
    "tb-2": "#ff7f0e",
    "tb-4": "#d62728",
    "tb-5": "#9467bd",
}


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%m%d%y")

    hoo_results: dict[str, dict[str, float]] = {}
    topic_sizes: dict[str, int] = {}

    for sel_label, sel_safe in SELECTORS.items():
        hoo_dir = HOO_DIR / f"qwen35_122b_hoo_topic_{sel_safe}"
        summary_path = hoo_dir / "hoo_summary.json"
        if not summary_path.exists():
            print(f"  {sel_label}: no HOO results at {summary_path}")
            continue

        with open(summary_path) as f:
            data = json.load(f)

        topic_sizes = data["group_sizes"]
        hoo_results[sel_label] = {}
        for fold in data["folds"]:
            topic = fold["held_out_groups"][0]
            for probe_data in fold["layers"].values():
                if probe_data["layer"] == BEST_LAYER:
                    hoo_results[sel_label][topic] = float(probe_data["hoo_r"])
                    break

    # Sort topics by mean r across selectors (descending)
    all_topics = sorted(set().union(*(set(v) for v in hoo_results.values())))
    topic_mean_r = {t: np.mean([hoo_results[s].get(t, 0) for s in hoo_results]) for t in all_topics}
    all_topics = sorted(all_topics, key=lambda t: topic_mean_r[t], reverse=True)

    # Print results
    print(f"{'Topic':<22} {'n':>5}", end="")
    for s in SELECTORS:
        print(f"  {s:>6}", end="")
    print()
    print("-" * (22 + 5 + 8 * len(SELECTORS)))
    for topic in all_topics:
        n = topic_sizes.get(topic, 0)
        print(f"{topic:<22} {n:>5}", end="")
        for s in SELECTORS:
            r = hoo_results.get(s, {}).get(topic, float("nan"))
            print(f"  {r:>6.3f}", end="")
        print()

    # Plot: HOO per topic, grouped bars
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(all_topics))
    n_sel = len(hoo_results)
    width = 0.8 / n_sel

    for i, (sel_label, topic_rs) in enumerate(hoo_results.items()):
        vals = [topic_rs.get(t, 0) for t in all_topics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=f"{sel_label}", color=COLORS[sel_label], alpha=0.85)

    # Add topic sizes as secondary labels
    labels = [f"{t}\n(n={topic_sizes.get(t, '?')})" for t in all_topics]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Qwen-3.5-122B: Per-topic HOO Pearson r (L{BEST_LAYER})\nTrain on N-1 topics, evaluate on held-out topic")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = ASSETS_DIR / f"plot_{today}_hoo_per_topic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
