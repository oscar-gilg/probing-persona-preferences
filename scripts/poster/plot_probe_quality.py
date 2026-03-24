"""Simplified probe quality bar chart for poster.
Three models, Pearson r only, test vs cross-topic.
Reads data from result files instead of hardcoding."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'Helvetica Neue',
    'font.size': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

base = Path("results/probes")

# --- Load data ---

# Gemma-3 IT: ridge_L31
with open(base / "gemma3_10k_heldout_std_raw/manifest.json") as f:
    gemma_it_manifest = json.load(f)
gemma_it_probe = [p for p in gemma_it_manifest["probes"] if p["id"] == "ridge_L31"][0]
gemma_it_test_r = gemma_it_probe["final_r"]

with open(base / "old_probes/gemma3_10k_hoo_topic/hoo_summary.json") as f:
    gemma_it_hoo = json.load(f)
gemma_it_hoo_r = gemma_it_hoo["layer_summary"]["31"]["ridge"]["mean_hoo_r"]
gemma_it_hoo_std = gemma_it_hoo["layer_summary"]["31"]["ridge"]["std_hoo_r"]

# Gemma-3 PT: ridge_L31
with open(base / "gemma3_pt_10k_heldout_std_raw/manifest.json") as f:
    gemma_pt_manifest = json.load(f)
gemma_pt_probe = [p for p in gemma_pt_manifest["probes"] if p["id"] == "ridge_L31"][0]
gemma_pt_test_r = gemma_pt_probe["final_r"]

with open(base / "gemma3_pt_10k_hoo_topic/hoo_summary.json") as f:
    gemma_pt_hoo = json.load(f)
gemma_pt_hoo_r = gemma_pt_hoo["layer_summary"]["31"]["ridge"]["mean_hoo_r"]
gemma_pt_hoo_std = gemma_pt_hoo["layer_summary"]["31"]["ridge"]["std_hoo_r"]

# Qwen3 Emb: ridge_L00
with open(base / "qwen3_emb_8b_heldout_std_raw/manifest.json") as f:
    qwen_manifest = json.load(f)
qwen_probe = [p for p in qwen_manifest["probes"] if p["id"] == "ridge_L00"][0]
qwen_test_r = qwen_probe["final_r"]

with open(base / "qwen3_emb_8b_hoo_topic/hoo_summary.json") as f:
    qwen_hoo = json.load(f)
qwen_hoo_r = qwen_hoo["layer_summary"]["0"]["ridge"]["mean_hoo_r"]
qwen_hoo_std = qwen_hoo["layer_summary"]["0"]["ridge"]["std_hoo_r"]

# --- Plot ---

models = ["Gemma-3\nInstruct", "Gemma-3\nPretrained", "Qwen3\nEmbedding"]
test_r = [gemma_it_test_r, gemma_pt_test_r, qwen_test_r]
hoo_r = [gemma_it_hoo_r, gemma_pt_hoo_r, qwen_hoo_r]
hoo_std = [gemma_it_hoo_std, gemma_pt_hoo_std, qwen_hoo_std]

x = np.arange(len(models))
w = 0.32

fig, ax = plt.subplots(figsize=(6.5, 3.2))

bars1 = ax.bar(x - w/2, test_r, w, color="#93C5FD", edgecolor="white", linewidth=0.5, label="Test set")
bars2 = ax.bar(x + w/2, hoo_r, w, color="#3B82F6", edgecolor="white", linewidth=0.5,
               yerr=hoo_std, capsize=3, error_kw={"linewidth": 1, "color": "#6B7280"}, label="Cross-topic")

# Value labels
for bar, val in zip(bars1, test_r):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=12, color="#374151")
for i, (bar, val) in enumerate(zip(bars2, hoo_r)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + hoo_std[i] + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=12, color="#374151", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("Pearson r", fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.axhline(0, color="#E5E7EB", linewidth=0.5)
ax.legend(frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
ax.set_title("Probe predicts preferences", fontweight="bold", fontsize=14)

plt.tight_layout()
out = Path("docs/poster/assets")
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "plot_032326_probe_quality.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out / 'plot_032326_probe_quality.png'}")
