"""Plot refusal rate by topic: KV steering vs KV + recompute."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.steering.analysis import load_checkpoint

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_kv_recompute.jsonl"
TOPICS_PATH = "data/topics/topics.json"
ASSETS = Path("experiments/steering/isolated_steering/full_run/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

with open(TOPICS_PATH) as f:
    topics_data = json.load(f)

rows = load_checkpoint(CHECKPOINT)


def steered_toward_task_id(row: dict) -> str:
    if row["signed_multiplier"] > 0:
        return row["task_a_id"]
    return row["task_b_id"]


def get_topic(task_id: str) -> str | None:
    entry = topics_data.get(task_id)
    if entry is None:
        return None
    return entry["anthropic/claude-sonnet-4.5"]["primary"]


def format_topic(topic: str) -> str:
    return topic.replace("_", " ").title()


# Collect per-topic, per-condition: count total and refusals
counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
    lambda: defaultdict(lambda: {"total": 0, "refusals": 0})
)

for row in rows:
    topic = get_topic(steered_toward_task_id(row))
    if topic is None:
        continue
    condition = row["condition"]
    counts[topic][condition]["total"] += 1
    if row["choice_original"] not in ("a", "b"):
        counts[topic][condition]["refusals"] += 1

# Compute refusal rate per topic per condition
topic_results: dict[str, dict[str, tuple[float, int]]] = {}
for topic, cond_dict in counts.items():
    topic_results[topic] = {}
    for condition, c in cond_dict.items():
        rate = c["refusals"] / c["total"]
        topic_results[topic][condition] = (rate, c["total"])

# Sort by KV+recompute refusal rate descending
sorted_topics = sorted(
    topic_results.keys(),
    key=lambda t: topic_results[t].get("kv_steering_recompute", (0.0, 0))[0],
    reverse=True,
)

# Build plot
CONDITIONS = [
    ("kv_steering", "#2563eb", "KV only"),
    ("kv_steering_recompute", "#93c5fd", "KV + recompute"),
]
bar_height = 0.35

fig, ax = plt.subplots(figsize=(9, 7))

for i, (condition, color, label) in enumerate(CONDITIONS):
    y_positions = []
    widths = []
    annotations = []
    for j, topic in enumerate(sorted_topics):
        y = j - bar_height / 2 + i * bar_height
        rate, n = topic_results[topic].get(condition, (0.0, 0))
        y_positions.append(y)
        widths.append(rate)
        annotations.append((rate, y, n))

    ax.barh(y_positions, widths, height=bar_height, color=color, label=label)

    for rate, y, n in annotations:
        ax.text(rate + 0.01, y, f"n={n}", va="center", fontsize=7)

ax.set_yticks(range(len(sorted_topics)))
ax.set_yticklabels([format_topic(t) for t in sorted_topics])
ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
ax.set_xlim(0, 1)
ax.set_xlabel("Refusal rate")
ax.set_title("KV steering: refusal rate by topic")
ax.invert_yaxis()
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()

out = ASSETS / "plot_031926_kv_recompute_refusal_by_topic.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
