"""Headline |Cohen's d| bar plot for Qwen-3.5-122B canonical probe replication,
with user-turn vs assistant-turn comparison.

Mirrors plot_qwen_eot_headline_d.py but adds two clusters per domain group
(user-turn / assistant-turn). Politics is assistant-only (the v2 user-turn
generator did not produce politics stimuli).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
QWEN_CSV = EXP_DIR / "headline_table.csv"
DATE = "042526"

DOMAINS = [
    ("truth", "Truth\n(true vs false)"),
    ("harm", "Harm\n(harmful vs benign)"),
    ("politics_democrat", "Politics\n(democrat prompt)"),
    ("politics_republican", "Politics\n(republican prompt)"),
]

QWEN_PROBES = [
    "qwen_tb-1_L33",
    "qwen_tb-1_L38",
    "qwen_tb-1_L43",
    "qwen_tb-4_L33",
    "qwen_tb-4_L38",
    "qwen_tb-4_L43",
]

BAR_COLORS = {
    "qwen_tb-1_L33": "#9ECAE1",
    "qwen_tb-1_L38": "#4292C6",
    "qwen_tb-1_L43": "#08519C",
    "qwen_tb-4_L33": "#A1D99B",
    "qwen_tb-4_L38": "#41AB5D",
    "qwen_tb-4_L43": "#006D2C",
    "Gemma reference": "#FF7F0E",
}

# Gemma reference d-values supplied by the user.
# Politics: not measured in this comparison (assistant-only Gemma reference
# would be available but the task here is the user/assistant split).
GEMMA_REFS = {
    ("truth", "user"): 3.35,
    ("truth", "assistant"): 2.47,
    ("harm", "user"): 2.05,
    ("harm", "assistant"): 2.12,
}

TURNS = [("user", "user-turn"), ("assistant", "assistant-turn")]


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def lookup_abs_d(rows, domain, turn, probe):
    for r in rows:
        if r["domain"] == domain and r["turn"] == turn and r["probe"] == probe:
            return abs(float(r["d"]))
    return None


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    qwen_rows = load_csv(QWEN_CSV)

    bar_labels = QWEN_PROBES + ["Gemma reference"]
    n_bars = len(bar_labels)
    bar_width = 0.10
    cluster_gap = 0.10  # gap between user / assistant clusters within a group
    cluster_width = n_bars * bar_width
    group_width = 2 * cluster_width + cluster_gap + 0.5  # plus padding to next group

    fig, ax = plt.subplots(figsize=(16, 5.8))

    # Compute centers manually to give clean spacing.
    group_centers = []
    cur = 0.0
    for _ in DOMAINS:
        cur += group_width / 2
        group_centers.append(cur)
        cur += group_width / 2

    # Within a group, cluster centers offset by +-(cluster_width + cluster_gap)/2.
    cluster_offset = (cluster_width + cluster_gap) / 2
    bar_offsets_within_cluster = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width

    # For storing all heights for ylim and legend.
    all_heights = []
    legend_handles = {}
    sanity = {}  # domain -> {turn -> {probe -> d}}

    for gi, (domain_key, _) in enumerate(DOMAINS):
        sanity[domain_key] = {}
        is_politics = domain_key.startswith("politics")
        for ti, (turn_key, turn_label) in enumerate(TURNS):
            cluster_center = group_centers[gi] + (
                -cluster_offset if ti == 0 else +cluster_offset
            )
            sanity[domain_key][turn_key] = {}

            if is_politics and turn_key == "user":
                # User-turn politics not measured: render N/A label only.
                ax.text(
                    cluster_center, 0.05, "user-turn\nN/A",
                    ha="center", va="bottom", fontsize=8.5,
                    color="grey", style="italic",
                )
                continue

            for j, label in enumerate(bar_labels):
                x = cluster_center + bar_offsets_within_cluster[j]
                if label == "Gemma reference":
                    ref_domain = "truth" if domain_key == "truth" else "harm"
                    if is_politics:
                        # No Gemma reference for politics in this comparison.
                        continue
                    h = GEMMA_REFS[(ref_domain, turn_key)]
                else:
                    # Qwen probe lookup. politics_democrat / politics_republican
                    # only exist in CSV for assistant-turn.
                    h = lookup_abs_d(qwen_rows, domain_key, turn_key, label)
                    if h is None:
                        continue
                sanity[domain_key][turn_key][label] = h
                all_heights.append(h)
                bar = ax.bar(
                    x, h, width=bar_width,
                    color=BAR_COLORS[label], edgecolor="black", linewidth=0.4,
                )
                legend_handles[label] = bar[0]
                ax.text(x, h + 0.04, f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

            # Cluster label below the cluster (which turn).
            ax.text(
                cluster_center, -0.18, turn_label,
                ha="center", va="top", fontsize=8.5,
                transform=ax.transData,
            )

    ax.axhline(2.0, color="grey", linestyle=":", linewidth=1.0)
    legend_handles["large effect (d=2)"] = plt.Line2D(
        [0], [0], color="grey", linestyle=":", linewidth=1.0,
    )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [label for _, label in DOMAINS], fontsize=10,
    )
    # Push group labels down so they don't collide with the per-cluster labels.
    ax.tick_params(axis="x", which="major", pad=22)

    ax.set_ylabel("|Cohen's d| at end-of-turn token")
    ax.set_title(
        "Qwen-3.5-122B end-of-turn probes: |d| across truth / harm / politics — user-turn vs assistant-turn"
    )
    max_h = max(all_heights)
    ax.set_ylim(0, max(max_h * 1.18, 3.6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)

    # Order legend: probes first (in QWEN_PROBES order), then Gemma, then d=2.
    legend_order = QWEN_PROBES + ["Gemma reference", "large effect (d=2)"]
    handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    labels = [k for k in legend_order if k in legend_handles]
    ax.legend(handles, labels, loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    plt.tight_layout()
    out_path = ASSETS_DIR / f"plot_{DATE}_qwen_headline_d_by_turn.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")
    for d_key, _ in DOMAINS:
        for t_key, _ in TURNS:
            entries = sanity[d_key][t_key]
            if not entries:
                print(f"  {d_key} / {t_key}: (omitted)")
                continue
            nums = ", ".join(f"{lab}={v:.2f}" for lab, v in entries.items())
            print(f"  {d_key} / {t_key}: {nums}")


if __name__ == "__main__":
    main()
