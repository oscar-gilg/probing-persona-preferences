"""Generate steerability-by-topic plot from cross-layer differential steering data."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

# Load pairs for topic info
with open("experiments/steering/cross_layer/pairs_500.json") as f:
    pairs = json.load(f)

pair_topics = {}
for p in pairs:
    pair_topics[p["pair_id"]] = {
        "topic_a": p["topic_a"],
        "topic_b": p["topic_b"],
    }

# Load L25 probe results at layer 25 (the sweet spot)
rows = []
with open("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["layer"] == 25 and r["condition"] == "probe_L25":
            rows.append(r)

# For each row, determine the "steered toward" topic
# positive multiplier → steer toward A, negative → steer toward B
topic_groups = defaultdict(list)
for r in rows:
    m = r["signed_multiplier"]
    if abs(m) < 0.001:
        continue  # skip baseline

    pair_id = r["pair_id"]
    if pair_id not in pair_topics:
        continue

    topics = pair_topics[pair_id]
    steered_toward_topic = topics["topic_a"] if m > 0 else topics["topic_b"]

    chose_a = r["choice_original"]
    if chose_a in ("a", "A", "task_a"):
        steered = m > 0
    elif chose_a in ("b", "B", "task_b"):
        steered = m < 0
    else:
        continue

    topic_groups[steered_toward_topic].append(1.0 if steered else 0.0)

# Compute stats
topics = sorted(topic_groups.keys(), key=lambda t: np.mean(topic_groups[t]), reverse=True)
p_vals = [np.mean(topic_groups[t]) for t in topics]
ns = [len(topic_groups[t]) for t in topics]

# Color by harmful vs benign
harmful_topics = {"harmful_request", "security_legal", "sensitive_creative"}
colors = ['#EF4444' if t.lower().replace(" ", "_") in harmful_topics else '#3B82F6' for t in topics]

# Clean up topic names for display
display_names = [t.replace("_", " ").title() for t in topics]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(range(len(topics)), p_vals, color=colors, height=0.7)
ax.set_yticks(range(len(topics)))
ax.set_yticklabels([f"{name} (n={n})" for name, n in zip(display_names, ns)], fontsize=10)
ax.axvline(0.5, color='#D1D5DB', linestyle='--', linewidth=1)
ax.set_xlabel('P(chose steered task)')
ax.set_xlim(0, 1)
ax.set_title('Safety training resists harmful steering', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()

out = Path("experiments/steering/cross_layer/assets/plot_032326_steerability_by_topic.png")
plt.savefig(out, dpi=200)
plt.close()
print(f"Saved: {out} ({len(rows)} rows, {len(topics)} topics)")
