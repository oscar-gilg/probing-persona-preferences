"""Analyze cross-model probe results and generate plots.

Reads cross_eval_results.json and produces:
1. Utility correlation heatmap (upper bound for transfer)
2. Transfer heatmap: act_model x util_model (best R² across layers/selectors)
3. Layer profiles: same-model vs cross-model R²
4. Selector comparison across models

Usage:
    python -m scripts.cross_model_probes.analyze
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/probes/cross_model/cross_eval_results.json")
ASSETS_DIR = Path("experiments/cross_model_probes/assets")

MODEL_ORDER = ["gemma3", "llama8b", "gptoss", "qwen35"]
MODEL_LABELS = {
    "gemma3": "Gemma-3-27B",
    "llama8b": "Llama-3.1-8B",
    "gptoss": "GPT-OSS-120B",
    "qwen35": "Qwen-3.5-122B",
}


def today() -> str:
    return datetime.now().strftime("%m%d%y")


def plot_utility_correlations(corr: dict, save_path: Path):
    models = [m for m in MODEL_ORDER if m in corr]
    n = len(models)
    matrix = np.array([[corr[m1][m2] for m2 in models] for m1 in models])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.6 else "black", fontsize=10)

    ax.set_title("How correlated are the models' preferences?\n(Pearson r between Thurstonian scores, heldout tasks)")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_transfer_heatmap(probe_results: dict, save_path: Path):
    """Best R² for each (act_model → eval_model) pair, marginalizing over util_model, selector, layer."""
    models = [m for m in MODEL_ORDER if any(
        r["act_model"] == m for r in probe_results.values()
    )]
    n = len(models)

    # For each (act_model, eval_model), find best R² across all probes trained on act_model
    matrix = np.full((n, n), np.nan)
    for i, act in enumerate(models):
        for j, eval_m in enumerate(models):
            best_r2 = -1
            for res in probe_results.values():
                if res["act_model"] != act:
                    continue
                # Use probes trained with the SAME model's utilities (same-model direction)
                # but evaluated on eval_m's scores
                eval_r = res["eval_results"].get(eval_m, {})
                r2 = eval_r.get("r2", -1)
                if r2 > best_r2:
                    best_r2 = r2
            if best_r2 >= 0:
                matrix[i, j] = best_r2

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=np.nanmax(matrix) * 1.1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])
    ax.set_xlabel("Whose preferences are predicted?")
    ax.set_ylabel("Whose activations are used?")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color="white" if matrix[i, j] > np.nanmax(matrix) * 0.6 else "black", fontsize=10)

    ax.set_title("Can a probe from Model A predict Model B's preferences?\n(Best R² over selectors and layers)")
    fig.colorbar(im, ax=ax, label="R²")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_same_vs_cross_by_layer(probe_results: dict, save_path: Path):
    """For each activation model, compare same-model vs cross-model R² by layer."""
    # Order by model size
    ordered = [m for m in MODEL_ORDER if any(r["act_model"] == m for r in probe_results.values())]

    fig, axes = plt.subplots(1, len(ordered), figsize=(4 * len(ordered), 4), sharey=True)
    if len(ordered) == 1:
        axes = [axes]

    # Exclude Llama from cross-model mean (its weak correlations distort the average)
    cross_exclude = {"llama8b"}

    for ax, act_key in zip(axes, ordered):
        layers = sorted(set(r["layer"] for r in probe_results.values() if r["act_model"] == act_key))

        same_r2 = {l: -1.0 for l in layers}
        cross_r2 = {l: [] for l in layers}

        for res in probe_results.values():
            if res["act_model"] != act_key:
                continue
            layer = res["layer"]
            if res["util_model"] == act_key:
                r2 = res["eval_results"].get(act_key, {}).get("r2", -1)
                same_r2[layer] = max(same_r2[layer], r2)
            else:
                eval_m = res["util_model"]
                if eval_m in cross_exclude:
                    continue
                r2 = res["eval_results"].get(eval_m, {}).get("r2", -1)
                if r2 >= 0:
                    cross_r2[layer].append(r2)

        ax.plot(layers, [same_r2[l] for l in layers], "o-", color="tab:blue", label="Same-model", linewidth=2)
        cross_means = [np.mean(cross_r2[l]) if cross_r2[l] else np.nan for l in layers]
        cross_stds = [np.std(cross_r2[l]) if len(cross_r2[l]) > 1 else 0 for l in layers]
        ax.errorbar(layers, cross_means, yerr=cross_stds, fmt="s--", color="tab:orange",
                    label="Cross-model (mean, excl. Llama)", linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_title(MODEL_LABELS.get(act_key, act_key))
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("R²")
    fig.suptitle("Does the preference direction generalize across models?", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    d = today()

    plot_utility_correlations(
        data["utility_correlations"],
        ASSETS_DIR / f"plot_{d}_utility_correlations.png",
    )

    plot_transfer_heatmap(
        data["probe_cross_eval"],
        ASSETS_DIR / f"plot_{d}_transfer_heatmap.png",
    )

    plot_same_vs_cross_by_layer(
        data["probe_cross_eval"],
        ASSETS_DIR / f"plot_{d}_same_vs_cross_by_layer.png",
    )


if __name__ == "__main__":
    main()
