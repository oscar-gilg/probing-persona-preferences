import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json")
POLITICS_DATA_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/politics_scoring_results.json")
OUT_DIR = Path("docs/logs/assets")

TRUTH_PROBE = "task_mean_L32"
HARM_PROBE = "task_mean_L39"
POLITICS_PROBE = "task_mean_L39"

TRUTH_PROMPTS = ["neutral", "truthful", "con_artist", "lie_directive", "pathological_liar"]
TRUTH_LABELS = ["No prompt", "Truthful", "Con artist", "Lie directive", "Pathological\nliar"]

HARM_PROMPTS = ["safe", "neutral", "unrestricted", "sinister_ai", "sadist"]
HARM_LABELS = ["Safe", "No prompt", "Unrestricted", "Sinister AI", "Sadist"]

POLITICS_PROMPTS = ["socialist", "democrat", "neutral", "republican", "nationalist"]
POLITICS_LABELS = ["Socialist", "Democrat", "No prompt", "Republican", "Nationalist"]


def plot_domain(data, domain, probe_key, prompts, labels, out_name, title):
    domain_data = [d for d in data if d["domain"] == domain]

    fig, ax = plt.subplots(figsize=(10, 5))

    conditions_map = {
        "truth": (["true", "false"], ["#66bb6a", "#ef5350"]),
        "harm": (["benign", "harmful"], ["#66bb6a", "#ef5350"]),
        "politics": (["left", "right"], ["#42a5f5", "#ef5350"]),
    }
    conditions, colors = conditions_map[domain]
    cond_labels = conditions

    x = np.arange(len(prompts))
    width = 0.35

    for ci, (cond, color, label) in enumerate(zip(conditions, colors, cond_labels)):
        means = []
        sems = []
        for prompt in prompts:
            scores = [
                d["critical_span_mean_scores"][probe_key]
                for d in domain_data
                if d["system_prompt"] == prompt and d["condition"] == cond
            ]
            means.append(np.mean(scores))
            sems.append(np.std(scores) / np.sqrt(len(scores)))

        offset = -width / 2 + ci * width
        bars = ax.bar(x + offset, means, width * 0.9, yerr=sems,
                       label=label, color=color, alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Critical span probe score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)["items"]

    with open(POLITICS_DATA_PATH) as f:
        politics_data = json.load(f)["items"]

    plot_domain(
        data, "truth", TRUTH_PROBE, TRUTH_PROMPTS, TRUTH_LABELS,
        "plot_031826_truth_critical_span_focused.png",
        "Truth: critical span scores under lying system prompts",
    )

    plot_domain(
        data, "harm", HARM_PROBE, HARM_PROMPTS, HARM_LABELS,
        "plot_031826_harm_critical_span_focused.png",
        "Harm: critical span scores under evil system prompts",
    )

    plot_domain(
        politics_data, "politics", POLITICS_PROBE, POLITICS_PROMPTS, POLITICS_LABELS,
        "plot_031826_politics_critical_span_focused.png",
        "Politics: critical span scores under partisan system prompts",
    )


if __name__ == "__main__":
    main()
