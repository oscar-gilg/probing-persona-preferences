"""Persona-modulation by turn: signed Cohen's d per system prompt for
Qwen-3.5-122B canonical probe replication.

3-row layout:
- Truth (9 sysprompts): two bars per prompt (user-turn / assistant-turn).
- Harm  (5 sysprompts): two bars per prompt (user-turn / assistant-turn).
- Politics (9 sysprompts): single bar per prompt (assistant-only).

Probe: qwen_tb-1_L38. Y-axis: signed d, line at d=0.
Sysprompts ordered (within each domain) from most-pro to most-anti the contrast,
ranked by assistant-turn d.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
ASSISTANT_PATH = EXP_DIR / "scoring_results.json"
USER_PATH = EXP_DIR / "user_turn_scoring_results.json"
POLITICS_PATH = EXP_DIR / "politics_scoring_results.json"

PROBE = "qwen_tb-1_L38"
DATE = "042526"

COLOR_USER = "#1f77b4"
COLOR_ASST = "#ff7f0e"

TRUTH_PROMPTS = [
    "truthful", "neutral", "con_artist", "unreliable_narrator",
    "contrarian", "gaslighter", "opposite_day",
    "lie_directive", "pathological_liar",
]
HARM_PROMPTS = ["safe", "neutral", "unrestricted", "sadist", "sinister_ai"]
POLITICS_PROMPTS = [
    "socialist", "democrat", "centrist", "apolitical", "neutral",
    "libertarian", "republican", "nationalist", "contrarian",
]


def load(path):
    return json.load(open(path))["items"]


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                     / (len(pos) + len(neg) - 2))
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else 0.0


def d_per_sysprompt(items, prompts, c_pos, c_neg):
    by_sp = defaultdict(list)
    for it in items:
        by_sp[it["system_prompt"]].append(it)
    out = {}
    for sp in prompts:
        pos = [it["probe_scores"][PROBE] for it in by_sp[sp] if it["condition"] == c_pos]
        neg = [it["probe_scores"][PROBE] for it in by_sp[sp] if it["condition"] == c_neg]
        out[sp] = float(cohen_d_pooled(np.array(pos), np.array(neg)))
    return out


def order_by_descending(d_map):
    return sorted(d_map.keys(), key=lambda k: (-(d_map[k] if not np.isnan(d_map[k]) else -1e9)))


def draw_two_bar_row(ax, prompts, d_user, d_asst, title, ylabel):
    n = len(prompts)
    x = np.arange(n)
    bw = 0.4
    h_user = [d_user[sp] for sp in prompts]
    h_asst = [d_asst[sp] for sp in prompts]
    bars_u = ax.bar(x - bw / 2, h_user, width=bw, color=COLOR_USER,
                    edgecolor="black", linewidth=0.4, label="user-turn")
    bars_a = ax.bar(x + bw / 2, h_asst, width=bw, color=COLOR_ASST,
                    edgecolor="black", linewidth=0.4, label="assistant-turn")

    for bars, vals in ((bars_u, h_user), (bars_a, h_asst)):
        for bar, v in zip(bars, vals):
            offset = 0.04 if v >= 0 else -0.04
            va = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:+.1f}", ha="center", va=va, fontsize=7)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def draw_single_bar_row(ax, prompts, d_map, color, title, ylabel):
    n = len(prompts)
    x = np.arange(n)
    bw = 0.6
    h = [d_map[sp] for sp in prompts]
    bars = ax.bar(x, h, width=bw, color=color,
                  edgecolor="black", linewidth=0.4, label="assistant-turn")
    for bar, v in zip(bars, h):
        offset = 0.04 if v >= 0 else -0.04
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.1f}", ha="center", va=va, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    asst_items = load(ASSISTANT_PATH)
    user_items = load(USER_PATH)
    politics_items = load(POLITICS_PATH)

    asst_truth = [it for it in asst_items if it["domain"] == "truth"]
    asst_harm = [it for it in asst_items if it["domain"] == "harm"]
    user_truth = [it for it in user_items if it["domain"] == "truth"]
    user_harm = [it for it in user_items if it["domain"] == "harm"]

    d_truth_asst = d_per_sysprompt(asst_truth, TRUTH_PROMPTS, "true", "false")
    d_truth_user = d_per_sysprompt(user_truth, TRUTH_PROMPTS, "true", "false")
    d_harm_asst = d_per_sysprompt(asst_harm, HARM_PROMPTS, "harmful", "benign")
    d_harm_user = d_per_sysprompt(user_harm, HARM_PROMPTS, "harmful", "benign")
    d_politics = d_per_sysprompt(politics_items, POLITICS_PROMPTS, "left", "right")

    truth_order = order_by_descending(d_truth_asst)
    # Harm contrast is harmful-vs-benign: signed d is negative when probe
    # correctly fires more on benign (higher truth/safe). To go "most-pro"
    # to "most-anti" the harmful-vs-benign contrast, sort descending on d
    # (positive d = treats harmful as more harmful-aligned-with-positive).
    harm_order = order_by_descending(d_harm_asst)
    politics_order = order_by_descending(d_politics)

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 11),
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    draw_two_bar_row(
        axes[0], truth_order, d_truth_user, d_truth_asst,
        title="Truth (true vs false): persona modulation by turn",
        ylabel=f"Cohen's d ({PROBE})",
    )
    draw_two_bar_row(
        axes[1], harm_order, d_harm_user, d_harm_asst,
        title="Harm (harmful vs benign): persona modulation by turn",
        ylabel=f"Cohen's d ({PROBE})",
    )
    draw_single_bar_row(
        axes[2], politics_order, d_politics, COLOR_ASST,
        title="Politics (left vs right): persona modulation, assistant-turn only",
        ylabel=f"Cohen's d ({PROBE})",
    )

    fig.suptitle(
        f"Qwen-3.5-122B at probe {PROBE}: persona modulation, user-turn vs assistant-turn",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE}_qwen_persona_modulation_by_turn.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")

    print("Truth — assistant-turn d:")
    for sp in truth_order:
        print(f"  {sp}: user={d_truth_user[sp]:+.2f}  assistant={d_truth_asst[sp]:+.2f}")
    print("Harm — assistant-turn d:")
    for sp in harm_order:
        print(f"  {sp}: user={d_harm_user[sp]:+.2f}  assistant={d_harm_asst[sp]:+.2f}")
    print("Politics — assistant-turn d:")
    for sp in politics_order:
        print(f"  {sp}: assistant={d_politics[sp]:+.2f}")


if __name__ == "__main__":
    main()
