"""Comparison plots: Gemma sysprompt-induced vs Qwen SFT-induced sadist persona.

Outputs:
  plot_050426_cross_trained_comparison.png
  plot_050426_cosine_comparison.png
  plot_050426_summary_bar.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# Qwen layers (48 total) and relative depths
QWEN_LAYERS = [12, 24, 28, 33, 38, 43]
QWEN_REL = [L / 48 for L in QWEN_LAYERS]

# Gemma layers (62 total) and relative depths
GEMMA_LAYERS = [25, 32, 39, 46, 53]
GEMMA_REL = [L / 62 for L in GEMMA_LAYERS]

# Cross-trained probes (Pearson r on held-out)
QWEN_CROSS = {
    "default-acts → default-utils (within)":  [0.89, 0.91, 0.93, 0.94, 0.94, 0.95],
    "sadist-acts → sadist-utils (within)":    [0.53, 0.58, 0.61, 0.61, 0.63, 0.60],
    "default-acts → sadist-utils (cross)":    [0.55, 0.57, 0.62, 0.60, 0.60, 0.57],
    "sadist-acts → default-utils (cross)":    [0.89, 0.91, 0.92, 0.93, 0.94, 0.94],
}
GEMMA_CROSS = {
    "default-acts → default-utils (within)":  [0.804, 0.814, 0.814, 0.811, 0.801],
    "sadist-acts → sadist-utils (within)":    [0.832, 0.849, 0.844, 0.836, 0.833],
    "default-acts → sadist-utils (cross)":    [0.820, 0.814, 0.810, 0.801, 0.792],
    "sadist-acts → default-utils (cross)":    [0.809, 0.810, 0.800, 0.793, 0.787],
}
STYLES = {
    "default-acts → default-utils (within)":  ("#2e6cb6", "o", "-"),
    "sadist-acts → sadist-utils (within)":    ("#cf8866", "o", "-"),
    "default-acts → sadist-utils (cross)":    ("#cf8866", "s", "--"),
    "sadist-acts → default-utils (cross)":    ("#2e6cb6", "s", "--"),
}

# Cosine pairs
QWEN_COS = {
    "DD ↔ SD (same target=D-utils, diff inputs)":   [0.67, 0.63, 0.49, 0.33, 0.42, 0.37],
    "SS ↔ DS (same target=S-utils, diff inputs)":   [0.74, 0.69, 0.59, 0.46, 0.40, 0.47],
    "DD ↔ DS (same input=D-acts, diff targets)":   [-0.04, 0.02, 0.02, 0.03, 0.04, 0.05],
    "SS ↔ SD (same input=S-acts, diff targets)":   [-0.02, 0.00, 0.04, 0.03, 0.07, 0.05],
    "DD ↔ SS (different input + different target)":[-0.03, -0.03, -0.02, 0.04, 0.04, 0.10],
    "DS ↔ SD (different input + different target)":[-0.01, 0.03, 0.02, 0.04, 0.05, 0.04],
}
GEMMA_COS = {
    "DD ↔ SD (same target=D-utils, diff inputs)":   [0.523, 0.392, 0.414, 0.348, 0.374],
    "SS ↔ DS (same target=S-utils, diff inputs)":   [0.466, 0.313, 0.368, 0.330, 0.252],
    "DD ↔ DS (same input=D-acts, diff targets)":   [-0.035, -0.017, -0.053, -0.064, -0.012],
    "SS ↔ SD (same input=S-acts, diff targets)":   [-0.085, -0.020,  0.000, -0.015,  0.059],
    "DD ↔ SS (different input + different target)":[-0.015,  0.094,  0.079,  0.047,  0.056],
    "DS ↔ SD (different input + different target)":[-0.072, -0.024, -0.084, -0.128,  0.020],
}


def plot_cross_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (title, layers_rel, data) in zip(
        axes,
        [("Qwen-3.5-122B (SFT-induced sadist)", QWEN_REL, QWEN_CROSS),
         ("Gemma-3-27B (sysprompt-induced sadist)", GEMMA_REL, GEMMA_CROSS)],
    ):
        for label, vals in data.items():
            color, marker, ls = STYLES[label]
            ax.plot(layers_rel, vals, marker=marker, ls=ls, color=color, label=label, lw=1.8, ms=7)
        ax.set_xlabel("relative layer depth (layer / n_total)")
        ax.set_title(title)
        ax.set_xlim(0.2, 0.95)
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pearson r (held-out)")
    axes[1].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.suptitle("Cross-trained probes — both preferences decodable from either activation set",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050426_cross_trained_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cosine_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 0.85]})
    for ax, (title, layers_actual, data) in zip(
        axes,
        [("Qwen-3.5-122B (SFT-induced)", QWEN_LAYERS, QWEN_COS),
         ("Gemma-3-27B (sysprompt-induced)", GEMMA_LAYERS, GEMMA_COS)],
    ):
        labels = list(data.keys())
        mat = np.array([data[k] for k in labels])
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        if ax is axes[0]:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8.5)
        else:
            ax.set_yticks([])
        ax.set_xticks(range(len(layers_actual)))
        ax.set_xticklabels(layers_actual)
        ax.set_xlabel(f"layer (of {62 if 'Gemma' in title else 48})")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(mat[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=axes, label="cosine")
    fig.suptitle("Cosine similarity between trained probe directions in residual stream",
                 fontsize=12, y=1.02)
    plt.savefig(OUT / "plot_050426_cosine_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_bar() -> None:
    """Side-by-side bar chart for 4 key cells, picking the best layer per model."""
    metrics = [
        "default-acts\n→ default-utils\n(within)",
        "sadist-acts\n→ sadist-utils\n(within)",
        "default-acts\n→ sadist-utils\n(cross-train)",
        "sadist-acts\n→ default-utils\n(cross-train)",
    ]
    qwen_keys = list(QWEN_CROSS.keys())
    gemma_keys = list(GEMMA_CROSS.keys())
    qwen_vals = [max(QWEN_CROSS[k]) for k in qwen_keys]
    gemma_vals = [max(GEMMA_CROSS[k]) for k in gemma_keys]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w / 2, qwen_vals, w, color="#7e42a3", label="Qwen-3.5-122B (SFT-induced)")
    ax.bar(x + w / 2, gemma_vals, w, color="#3a8c46", label="Gemma-3-27B (sysprompt-induced)")
    for xi, v in zip(x - w / 2, qwen_vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, gemma_vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5)
    ax.set_ylabel("Pearson r (best layer)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Cross-trained probes recover within-domain accuracy in both models")
    plt.tight_layout()
    plt.savefig(OUT / "plot_050426_summary_bar.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_cross_comparison()
    plot_cosine_comparison()
    plot_summary_bar()
    print("wrote 3 comparison plots")
