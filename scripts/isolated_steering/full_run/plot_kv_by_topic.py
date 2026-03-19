"""Plot KV steering effectiveness and refusal rate by topic of the steered-toward task."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.steering.analysis import chose_steered_task, load_checkpoint

CHECKPOINT = "experiments/steering/isolated_steering/checkpoint_kv_full.jsonl"
TOPICS_PATH = "data/topics/topics.json"
ASSETS = Path("experiments/steering/isolated_steering/full_run/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

HARMFUL_TOPICS = {
    "harmful_request",
    "sensitive_creative",
    "security_legal",
    "value_conflict",
    "model_manipulation",
}

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


def bar_color(topic: str) -> str:
    return "#dc2626" if topic in HARMFUL_TOPICS else "#2563eb"


# ── Plot 1: P(chose steered task) by topic ──

# Collect per-topic stats (valid rows only)
topic_success: dict[str, list[bool]] = defaultdict(list)
for row in rows:
    if row["choice_original"] not in ("a", "b"):
        continue
    topic = get_topic(steered_toward_task_id(row))
    if topic is None:
        continue
    topic_success[topic].append(chose_steered_task(row))

# Sort by P(steered) descending
topic_stats = []
for topic, successes in topic_success.items():
    p = sum(successes) / len(successes)
    topic_stats.append((topic, p, len(successes)))
topic_stats.sort(key=lambda x: x[1], reverse=True)

topics = [t[0] for t in topic_stats]
p_values = [t[1] for t in topic_stats]
ns = [t[2] for t in topic_stats]

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = range(len(topics))
colors = [bar_color(t) for t in topics]
bars = ax.barh(y_pos, p_values, color=colors)

for i, (bar, n) in enumerate(zip(bars, ns)):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels([format_topic(t) for t in topics])
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
ax.set_xlim(0, 1)
ax.set_xlabel("P(model chose steered task)")
ax.set_title("KV steering effectiveness by topic")
ax.invert_yaxis()
fig.tight_layout()
out1 = ASSETS / "plot_031826_kv_steerability_by_topic.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved {out1}")

# ── Plot 2: Refusal rate by topic ──

topic_total: dict[str, int] = defaultdict(int)
topic_refusal: dict[str, int] = defaultdict(int)
for row in rows:
    topic = get_topic(steered_toward_task_id(row))
    if topic is None:
        continue
    topic_total[topic] += 1
    if row["choice_original"] not in ("a", "b"):
        topic_refusal[topic] += 1

refusal_stats = []
for topic in topic_total:
    rate = topic_refusal[topic] / topic_total[topic]
    refusal_stats.append((topic, rate, topic_total[topic]))
refusal_stats.sort(key=lambda x: x[1], reverse=True)

r_topics = [t[0] for t in refusal_stats]
r_values = [t[1] for t in refusal_stats]
r_ns = [t[2] for t in refusal_stats]

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = range(len(r_topics))
colors = [bar_color(t) for t in r_topics]
bars = ax.barh(y_pos, r_values, color=colors)

for i, (bar, n) in enumerate(zip(bars, r_ns)):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels([format_topic(t) for t in r_topics])
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
ax.set_xlim(0, 1)
ax.set_xlabel("Refusal rate")
ax.set_title("KV steering: refusal rate by topic")
ax.invert_yaxis()
fig.tight_layout()
out2 = ASSETS / "plot_031826_kv_refusal_by_topic.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved {out2}")
