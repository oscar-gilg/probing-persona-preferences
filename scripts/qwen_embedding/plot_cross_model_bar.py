"""Bar chart comparing probe accuracy across models.

Pairwise accuracy uses uniform-sample eval (500 random pairs + 35 within-topic
pairs per topic, from the final-half eval tasks). Test-set = all 500 random pairs,
cross-topic = within-held-out-topic pairs only (uniform_hoo_acc).
"""

import matplotlib.pyplot as plt
import numpy as np

models = [
    "Gemma-3 IT\n(L32)",
    "Gemma-3 PT\n(L31)",
    "Qwen3-Emb\n(4096d)",
]

# Pearson r: [heldout, HOO]
pearson_r = {
    "heldout": [0.864, 0.770, 0.725],
    "hoo":     [0.817, 0.627, 0.415],
}
hoo_std_r = [0.096, 0.128, 0.140]

# Pairwise accuracy — uniform eval (from eval final-half tasks)
pw_acc = {
    "test":       [0.800, 0.758, 0.732],  # uniform_acc (500 random pairs)
    "cross_topic": [0.751, 0.713, 0.673],  # uniform_hoo_acc (within-topic pairs, HOO)
}

colors_light = ["#7aaed4", "#b0b0b0", "#d4b96a"]
colors_dark  = ["#3d7aab", "#7a7a7a", "#b08f3a"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Probe Accuracy: Test Set vs Leave-One-Topic-Out", fontsize=14, fontweight="bold")

x = np.arange(len(models))
width = 0.35

# --- Pearson r panel ---
ax1.set_title("Pearson r", fontsize=12)
for i in range(len(models)):
    ax1.bar(x[i] - width/2, pearson_r["heldout"][i], width,
            color=colors_light[i], edgecolor="grey", linewidth=0.5)
    ax1.bar(x[i] + width/2, pearson_r["hoo"][i], width,
            color=colors_dark[i], edgecolor="grey", linewidth=0.5,
            yerr=hoo_std_r[i], capsize=3, error_kw={"linewidth": 1})

    ax1.text(x[i] - width/2, pearson_r["heldout"][i] + 0.015,
             f'{pearson_r["heldout"][i]:.2f}', ha="center", va="bottom", fontsize=9)
    ax1.text(x[i] + width/2, pearson_r["hoo"][i] + hoo_std_r[i] + 0.015,
             f'{pearson_r["hoo"][i]:.2f}', ha="center", va="bottom", fontsize=9)

ax1.set_ylabel("Pearson r")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 1.0)
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.legend(
    [plt.Rectangle((0, 0), 1, 1, fc=colors_light[0]),
     plt.Rectangle((0, 0), 1, 1, fc=colors_dark[0])],
    ["Test set", "Cross-topic"],
    loc="upper right", fontsize=9
)

# --- Pairwise accuracy panel ---
ax2.set_title("Pairwise Accuracy (uniform eval)", fontsize=12)
for i in range(len(models)):
    ax2.bar(x[i] - width/2, pw_acc["test"][i], width,
            color=colors_light[i], edgecolor="grey", linewidth=0.5)
    ax2.bar(x[i] + width/2, pw_acc["cross_topic"][i], width,
            color=colors_dark[i], edgecolor="grey", linewidth=0.5)

    ax2.text(x[i] - width/2, pw_acc["test"][i] + 0.015,
             f'{pw_acc["test"][i]:.2f}', ha="center", va="bottom", fontsize=9)
    ax2.text(x[i] + width/2, pw_acc["cross_topic"][i] + 0.015,
             f'{pw_acc["cross_topic"][i]:.2f}', ha="center", va="bottom", fontsize=9)

ax2.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_ylabel("Pairwise accuracy")
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim(0, 1.0)
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.legend(
    [plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3),
     plt.Rectangle((0, 0), 1, 1, fc=colors_light[0]),
     plt.Rectangle((0, 0), 1, 1, fc=colors_dark[0])],
    ["Chance (0.50)", "Test set", "Cross-topic"],
    loc="upper right", fontsize=9
)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out = "paper/figures/plot_041726_cross_model_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
