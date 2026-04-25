"""Base discrimination violins for Qwen-3.5-122B canonical probe replication,
with user-turn vs assistant-turn side-by-side.

4-panel grid: rows = (truth, harm), columns = (user-turn, assistant-turn).
Probe: qwen_tb-1_L38 (canonical). Sysprompt: neutral.

Note: assistant-turn data has a `nonsense` control; user-turn data does not
include it visually here (per spec, the v2 user-turn generator focused on
true/false and harmful/benign contrasts), so user-turn panels show 2 violins.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
ASSISTANT_PATH = EXP_DIR / "scoring_results.json"
USER_PATH = EXP_DIR / "user_turn_scoring_results.json"

PROBE = "qwen_tb-1_L38"
DATE = "042526"

COLORS = {
    "true": "#2196F3", "false": "#D32F2F",
    "benign": "#2E7D32", "harmful": "#D32F2F",
    "nonsense": "#9E9E9E",
}


def load(path):
    return json.load(open(path))["items"]


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                     / (len(pos) + len(neg) - 2))
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else 0.0


def gather(items, cond):
    return np.array([it["probe_scores"][PROBE] for it in items if it["condition"] == cond])


def draw_panel(ax, items, c_pos, c_neg, include_nonsense, title):
    pos_vals = gather(items, c_pos)
    neg_vals = gather(items, c_neg)
    d = round(float(cohen_d_pooled(pos_vals, neg_vals)), 2)

    if include_nonsense:
        non_vals = gather(items, "nonsense")
        series = [pos_vals, neg_vals, non_vals]
        positions = [0, 1, 2]
        colors_used = [COLORS[c_pos], COLORS[c_neg], COLORS["nonsense"]]
        tick_labels = [c_pos, c_neg, "nonsense"]
    else:
        series = [pos_vals, neg_vals]
        positions = [0, 1]
        colors_used = [COLORS[c_pos], COLORS[c_neg]]
        tick_labels = [c_pos, c_neg]

    parts = ax.violinplot(series, positions=positions, widths=0.7,
                          showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors_used):
        body.set_facecolor(color); body.set_alpha(0.7); body.set_edgecolor("black")
    parts["cmeans"].set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{title}\n(d = {d:+.2f}, n = {len(pos_vals)}/{len(neg_vals)})", fontsize=10)
    return d


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    asst = [it for it in load(ASSISTANT_PATH) if it["system_prompt"] == "neutral"]
    user = [it for it in load(USER_PATH) if it["system_prompt"] == "neutral"]

    asst_truth = [it for it in asst if it["domain"] == "truth"]
    asst_harm = [it for it in asst if it["domain"] == "harm"]
    user_truth = [it for it in user if it["domain"] == "truth"]
    user_harm = [it for it in user if it["domain"] == "harm"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=False)

    d_summary = {}
    d_summary[("user", "truth")] = draw_panel(
        axes[0, 0], user_truth, "true", "false",
        include_nonsense=False,
        title="Truth (CREAK) — USER-TURN",
    )
    d_summary[("assistant", "truth")] = draw_panel(
        axes[0, 1], asst_truth, "true", "false",
        include_nonsense=True,
        title="Truth (CREAK) — ASSISTANT-TURN",
    )
    d_summary[("user", "harm")] = draw_panel(
        axes[1, 0], user_harm, "harmful", "benign",
        include_nonsense=False,
        title="Harm (BailBench + stress test) — USER-TURN",
    )
    d_summary[("assistant", "harm")] = draw_panel(
        axes[1, 1], asst_harm, "harmful", "benign",
        include_nonsense=True,
        title="Harm (BailBench + stress test) — ASSISTANT-TURN",
    )

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Probe score ({PROBE})")

    fig.suptitle(
        f"Qwen-3.5-122B at probe {PROBE}: base discrimination, user-turn vs assistant-turn",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_qwen_base_discrimination_by_turn.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")
    for k, v in d_summary.items():
        print(f"  {k}: d = {v:+.2f}")


if __name__ == "__main__":
    main()
