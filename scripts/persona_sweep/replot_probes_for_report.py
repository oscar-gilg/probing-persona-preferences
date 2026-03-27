"""Replot Qwen vs Gemma and per-topic HOO with selected selectors only.

Selected:
  Qwen:  tb-2 `assistant`, tb-4 `\n (after im_end)`
  Gemma: tb-2 `model`, tb-5 `<end_of_turn>`
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_ASSETS = Path("docs/logs/assets")
REPORT_ASSETS.mkdir(parents=True, exist_ok=True)

# --- Data (from plot_qwen35_probes.py) ---

QWEN_LAYERS = [12, 24, 28, 33, 38, 43]
QWEN_DEPTH = [l / 48 for l in QWEN_LAYERS]

QWEN_R = {
    "Qwen tb-2 `assistant`": [0.880, 0.899, 0.922, 0.932, 0.942, 0.943],
    "Qwen tb-4 `\\n` (after im_end)": [0.883, 0.902, 0.926, 0.935, 0.946, 0.943],
}

GEMMA_LAYERS = [25, 32, 39, 46, 53]
GEMMA_DEPTH = [l / 62 for l in GEMMA_LAYERS]

GEMMA_R = {
    "Gemma tb-2 `model`": [0.857, 0.874, 0.867, 0.856, 0.854],
    "Gemma tb-5 `<end_of_turn>`": [0.859, 0.868, 0.859, 0.849, 0.845],
}

# --- Plot 1: Qwen vs Gemma heldout r ---

fig, ax = plt.subplots(figsize=(8, 5))

qwen_colors = {"Qwen tb-2 `assistant`": "#ff7f0e", "Qwen tb-4 `\\n` (after im_end)": "#d62728"}
gemma_colors = {"Gemma tb-2 `model`": "#ff7f0e", "Gemma tb-5 `<end_of_turn>`": "#9467bd"}

for name, vals in QWEN_R.items():
    ax.plot(QWEN_DEPTH, vals, "o-", label=name, color=qwen_colors[name], linewidth=2, markersize=6)

for name, vals in GEMMA_R.items():
    ax.plot(GEMMA_DEPTH, vals, "s--", label=name, color=gemma_colors[name], linewidth=1.5, markersize=5, alpha=0.7)

ax.set_xlabel("Layer (% of model depth)")
ax.set_ylabel("Heldout Pearson r")
ax.set_title("Preference probe strength: Qwen-3.5-122B vs Gemma-3-27B\n(best 2 selectors per model)")
ax.set_ylim(0.82, 0.96)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(REPORT_ASSETS / "plot_032626_qwen_vs_gemma_selected.png", dpi=150)
plt.close(fig)
print("Saved qwen_vs_gemma_selected")

# --- Plot 2: Per-topic HOO skipped (raw data on RunPod, not available locally) ---
# Using existing plot from experiments/training_probes/qwen35_probes/assets/plot_032326_hoo_per_topic.png

# --- Plot 3: Cross-model transfer heatmap (R²) ---
# Keep only the best selector per model pair

cross_path = Path("results/probes/cross_model/cross_eval_results.json")
with open(cross_path) as f:
    cross_data = json.load(f)

model_names = ["gemma3", "llama8b", "gptoss", "qwen35"]
display_names = ["Gemma-3-27B", "Llama-3.1-8B", "GPT-OSS-120B", "Qwen-3.5-122B"]

# Build best R² matrix
r_matrix = np.zeros((4, 4))
for i, act_model in enumerate(model_names):
    for j, eval_model in enumerate(model_names):
        best_r2 = -1
        for key, probe in cross_data["probe_cross_eval"].items():
            if probe["act_model"] == act_model:
                eval_r2 = probe["eval_results"].get(eval_model, {}).get("r2", -1)
                if eval_r2 > best_r2:
                    best_r2 = eval_r2
        r_matrix[i, j] = best_r2

# Convert to Pearson r (sqrt of R²)
r_pearson = np.sqrt(np.maximum(r_matrix, 0))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(r_pearson, cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(display_names, fontsize=9)
ax.set_xlabel("Whose preferences are predicted?")
ax.set_ylabel("Whose activations are used?")
ax.set_title("Cross-model probe transfer\n(Pearson r, best selector/layer per pair)")

for i in range(4):
    for j in range(4):
        color = "white" if r_pearson[i, j] > 0.7 else "black"
        ax.text(j, i, f"{r_pearson[i, j]:.3f}", ha="center", va="center", fontsize=11, color=color)

plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
fig.tight_layout()
fig.savefig(REPORT_ASSETS / "plot_032626_cross_model_transfer_r.png", dpi=150)
plt.close(fig)
print("Saved cross_model_transfer_r")
