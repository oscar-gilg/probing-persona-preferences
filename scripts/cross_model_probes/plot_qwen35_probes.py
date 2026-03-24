"""Plot Qwen-3.5-122B turn boundary probe results."""

from datetime import datetime

import matplotlib.pyplot as plt

ASSETS_DIR = "experiments/training_probes/qwen35_probes/assets"

LAYERS = [12, 24, 28, 33, 38, 43]

RESULTS_R = {
    "tb-1 `\\n` (final)": [0.880, 0.901, 0.923, 0.933, 0.943, 0.945],
    "tb-2 `assistant`": [0.880, 0.899, 0.922, 0.932, 0.942, 0.943],
    "tb-3 `<|im_start|>`": [0.879, 0.898, 0.911, 0.920, 0.924, 0.903],
    "tb-4 `\\n` (after im_end)": [0.883, 0.902, 0.926, 0.935, 0.946, 0.943],
    "tb-5 `<|im_end|>`": [0.877, 0.873, 0.875, 0.885, 0.889, 0.868],
}

RESULTS_ACC = {
    "tb-1 `\\n` (final)": [0.770, 0.789, 0.808, 0.814, 0.823, 0.824],
    "tb-2 `assistant`": [0.770, 0.785, 0.804, 0.815, 0.820, 0.825],
    "tb-3 `<|im_start|>`": [0.766, 0.785, 0.791, 0.809, 0.803, 0.788],
    "tb-4 `\\n` (after im_end)": [0.768, 0.792, 0.805, 0.815, 0.826, 0.821],
    "tb-5 `<|im_end|>`": [0.767, 0.770, 0.771, 0.778, 0.783, 0.769],
}

GEMMA_R = {
    "tb-1 `\\n` (final)": [0.857, 0.865, 0.854, 0.845, 0.846],
    "tb-2 `model`": [0.857, 0.874, 0.867, 0.856, 0.854],
    "tb-3 `<start_of_turn>`": [0.767, 0.703, 0.657, 0.644, 0.653],
    "tb-5 `<end_of_turn>`": [0.859, 0.868, 0.859, 0.849, 0.845],
}
GEMMA_LAYERS = [25, 32, 39, 46, 53]

COLORS = {
    "tb-1": "#1f77b4",
    "tb-2": "#ff7f0e",
    "tb-3": "#2ca02c",
    "tb-4": "#d62728",
    "tb-5": "#9467bd",
}

today = datetime.now().strftime("%m%d%y")


def color_for(label: str) -> str:
    return COLORS[label.split(" ")[0]]


def plot_metric(results: dict, ylabel: str, title: str, filename: str, ylim: tuple):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, vals in results.items():
        ax.plot(LAYERS, vals, "o-", color=color_for(label), label=label, linewidth=2, markersize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/{filename}", dpi=150)
    plt.close(fig)
    print(f"Saved: {ASSETS_DIR}/{filename}")


def plot_comparison():
    gemma_pct = [l / 62 for l in GEMMA_LAYERS]
    qwen_pct = [l / 48 for l in LAYERS]

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, vals in RESULTS_R.items():
        ax.plot(qwen_pct, vals, "o-", color=color_for(label), linewidth=2, markersize=5,
                label=f"Qwen {label}")
    for label, vals in GEMMA_R.items():
        key = label.split(" ")[0]
        ax.plot(gemma_pct, vals, "s--", color=COLORS[key], linewidth=1.5, markersize=5,
                alpha=0.5, label=f"Gemma {label}")

    ax.set_xlabel("Layer (% of model depth)")
    ax.set_ylabel("Heldout Pearson r")
    ax.set_title("Qwen-3.5-122B (solid) vs Gemma-3-27B (dashed)")
    ax.set_ylim(0.6, 0.96)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/plot_{today}_qwen_vs_gemma.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {ASSETS_DIR}/plot_{today}_qwen_vs_gemma.png")


RESULTS_HOO = {
    "tb-1 `\\n` (final)": [0.1912, 0.3466, 0.4684, 0.5096, 0.5583, 0.4430],
    "tb-2 `assistant`": [0.2566, 0.3279, 0.4497, 0.5331, 0.5570, 0.5508],
    "tb-4 `\\n` (after im_end)": [0.2615, 0.3687, 0.4949, 0.5321, 0.5070, 0.4682],
    "tb-5 `<|im_end|>`": [0.2169, 0.1695, 0.1896, 0.2495, 0.2602, 0.1618],
}


if __name__ == "__main__":
    plot_metric(
        RESULTS_R,
        ylabel="Heldout Pearson r",
        title="Qwen-3.5-122B: Pearson r by layer and token position",
        filename=f"plot_{today}_heldout_r_by_layer.png",
        ylim=(0.85, 0.96),
    )
    plot_metric(
        RESULTS_ACC,
        ylabel="Heldout pairwise accuracy",
        title="Qwen-3.5-122B: Pairwise accuracy by layer and token position\n(chance = 0.50, n = 3,985 pairs)",
        filename=f"plot_{today}_heldout_acc_by_layer.png",
        ylim=(0.5, 0.85),
    )
    plot_metric(
        RESULTS_HOO,
        ylabel="Task-weighted mean Pearson r",
        title="Qwen-3.5-122B: Hold-one-out by topic\n(14 folds, task-weighted mean r)",
        filename=f"plot_{today}_hoo_r_by_layer.png",
        ylim=(0, 0.7),
    )
    plot_comparison()
