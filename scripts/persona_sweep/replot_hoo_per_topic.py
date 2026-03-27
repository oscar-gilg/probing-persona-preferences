"""Replot Qwen per-topic HOO with only tb-2 and tb-4 selectors.

Values read from the existing plot_032326_hoo_per_topic.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPORT_ASSETS = Path("docs/logs/assets")

# Topics in order (descending by mean r), with sample sizes from the plot
topics = [
    ("persuasive_writing", 70),
    ("model_manipulation", 64),
    ("stresstest_other", 393),
    ("fiction", 204),
    ("summarization", 21),
    ("content_generation", 405),
    ("coding", 90),
    ("value_conflict", 157),
    ("sensitive_creative", 18),
    ("knowledge_qa", 301),
    ("security_legal", 96),
    ("harmful_request", 312),
    ("math", 832),
]

# Values read from bar heights in the existing plot
tb2 = [0.82, 0.72, 0.72, 0.76, 0.59, 0.68, 0.61, 0.58, 0.74, 0.58, 0.35, 0.45, 0.18]
tb4 = [0.81, 0.68, 0.73, 0.73, 0.62, 0.68, 0.61, 0.61, 0.69, 0.58, 0.47, 0.46, 0.18]

topic_names = [t[0] for t in topics]
topic_ns = [t[1] for t in topics]

fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(topic_names))
width = 0.35

ax.bar(x - width/2, tb2, width, label='tb-2 `assistant`', color='#ff7f0e', alpha=0.85)
ax.bar(x + width/2, tb4, width, label='tb-4 `\\n` (after im_end)', color='#d62728', alpha=0.85)

labels = [f"{name}\n(n={n})" for name, n in topics]
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Pearson r")
ax.set_title("Qwen-3.5-122B: Per-topic held-one-out generalization (L38)\nTrain on 13 topics, evaluate on held-out topic")
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_ylim(0, 0.9)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(REPORT_ASSETS / "plot_032626_qwen_hoo_per_topic_selected.png", dpi=150)
plt.close(fig)
print("Saved qwen_hoo_per_topic_selected")
