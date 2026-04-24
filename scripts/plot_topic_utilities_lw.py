"""Per-topic mean Thurstonian utility bar chart — figure in paper §2 (fig topic-utilities).

Horizontal bar chart showing mean Thurstonian utility per topic category,
sorted by mean utility. Error bars show ±1 SE. Bar labels show (n=...).

Producer for `plot_022626_topic_mean_utilities.png` referenced at main.tex:441.
Also publishes per-topic (mean, SE, n) as a single table claim.

Restored from commit 41c1e1a (2026-03-05); data paths re-pointed to
`main_probes/` where the Thurstonian run now lives.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np

from corroborate import ClaimSet


THURSTONIAN_CSV = Path(
    "results/experiments/main_probes/gemma3_10k_run1/"
    "pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_80fa9dc8.csv"
)
TOPICS_JSON = Path("data/topics/topics.json")
LW_ASSETS = Path("docs/lw_post/assets")
PAPER_FIGURES = Path("paper/figures")

PRETTY = {
    "coding": "Coding",
    "content_generation": "Content Gen.",
    "fiction": "Fiction",
    "harmful_request": "Harmful Request",
    "knowledge_qa": "Knowledge QA",
    "math": "Math",
    "model_manipulation": "Model Manipulation",
    "other": "Other",
    "persuasive_writing": "Persuasive Writing",
    "security_legal": "Security & Legal",
    "sensitive_creative": "Sensitive Creative",
    "summarization": "Summarization",
    "value_conflict": "Value Conflict",
}
# Drop "other" and "stresstest_other" (too noisy / source-indicator, not a topic).
DROP_TOPICS = {"other", "stresstest_other"}


def main():
    utilities = {}
    with open(THURSTONIAN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            utilities[row["task_id"]] = float(row["mu"])

    with open(TOPICS_JSON) as f:
        topics_raw = json.load(f)

    topics = {}
    for task_id, classifiers in topics_raw.items():
        first_classifier = next(iter(classifiers.values()))
        topics[task_id] = first_classifier["primary"]

    by_topic = defaultdict(list)
    for task_id, mu in utilities.items():
        if task_id in topics:
            by_topic[topics[task_id]].append(mu)

    topic_stats = []
    for topic, mus in by_topic.items():
        if topic in DROP_TOPICS or topic not in PRETTY:
            continue
        arr = np.array(mus)
        topic_stats.append({
            "topic": topic,
            "mean": float(np.mean(arr)),
            "se": float(np.std(arr) / np.sqrt(len(arr))),
            "n": int(len(arr)),
        })
    topic_stats.sort(key=lambda x: x["mean"])

    # Register as a table claim: rows = topic (display name), cols = {mean, se, n}.
    claims = ClaimSet(source="scripts/plot_topic_utilities_lw.py")
    topic_table = {
        PRETTY[s["topic"]]: {
            "mean": round(s["mean"], 3),
            "se": round(s["se"], 3),
            "n": s["n"],
        }
        for s in topic_stats
    }
    claims.register(
        name="Topic mean utility",
        value=topic_table,
        statement=(
            "Per-topic Thurstonian utility summary for Gemma-3-27B (default "
            "persona, 10k-task active-learning run). Rows: topic category. "
            "Columns: `mean` = mean μ across tasks of the topic; `se` = "
            "standard error; `n` = number of tasks. Rendered as a horizontal "
            "bar chart in fig:topic-utilities."
        ),
        used_in=["fig:topic-means-default", "app:topics-default"],
        data_paths=[str(THURSTONIAN_CSV), str(TOPICS_JSON)],
        derivation=(
            "Load μ per task_id from thurstonian_80fa9dc8.csv; join on "
            "data/topics/topics.json via each task's first-classifier primary "
            "topic; group μ by topic (excluding 'other'); compute (mean, "
            "std/sqrt(n), n). `mean` and `se` rounded to 3dp."
        ),
    )
    claims.save(Path("paper/claims/topic_mean_utilities.json"))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"$\\bf{{{PRETTY[s['topic']].replace(' ', '~')}}}$ (n={s['n']})" for s in topic_stats]
    means = [s["mean"] for s in topic_stats]
    ses = [s["se"] for s in topic_stats]
    colors = ["#E57373" if m < 0 else "#7ABAED" for m in means]
    ax.barh(range(len(labels)), means, xerr=ses, color=colors,
            edgecolor="black", linewidth=0.5, capsize=3,
            error_kw={"linewidth": 0.8, "alpha": 0.5})
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Thurstonian utility (μ)")
    ax.set_title("Gemma-3 27B-IT: Mean Utility by Topic", fontsize=13)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    LW_ASSETS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    lw_path = LW_ASSETS / "plot_022626_topic_mean_utilities.png"
    plt.savefig(lw_path, dpi=200, bbox_inches="tight")
    plt.close()
    paper_path = PAPER_FIGURES / "plot_022626_topic_mean_utilities.png"
    copyfile(lw_path, paper_path)
    print(f"Saved {lw_path}")
    print(f"Copied to {paper_path}")


if __name__ == "__main__":
    main()
