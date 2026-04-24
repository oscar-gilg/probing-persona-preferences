"""Probe held-out Pearson r by layer, by named position, for Gemma-3-27B and Qwen-3.5-122B.

For the token-position-selection appendix. X-axis is layer depth as a fraction
of total layers so the two models can be compared on a common scale. Line
colours match the turn-boundary diagram (Fig.~\\ref{fig:token-diagram}).

Gemma-3-27B: 62 layers. Positions swept: end-of-turn (tb-5), role-marker
(tb-2), final prompt token (tb-1), task-averaged (task_mean).

Qwen-3.5-122B: 48 layers. Positions swept within our named set: end-of-turn
(tb-5), role-marker (tb-2), final prompt token (tb-1). Task-averaged was not
swept for Qwen.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(".")
PAPER_FIG = REPO / "paper" / "figures"
DATE = "042426"

# Match the colours used in the turn-boundary diagram.
COLORS = {
    "end-of-turn":        "#1976D2",
    "role-marker":        "#00897B",
    "final prompt token": "#6A1B9A",
    "task-averaged":      "#E65100",
}

GEMMA_DEPTH = 62
QWEN_DEPTH = 48


def load_r_by_layer(manifest_path: Path) -> dict[int, float]:
    manifest = json.loads(manifest_path.read_text())
    return {int(p["layer"]): float(p["final_r"]) for p in manifest["probes"] if p["method"] == "ridge"}


def main() -> None:
    gemma_data = {
        "end-of-turn":        load_r_by_layer(REPO / "results/probes/heldout_eval_gemma3_tb-5/manifest.json"),
        "role-marker":        load_r_by_layer(REPO / "results/probes/heldout_eval_gemma3_tb-2/manifest.json"),
        "final prompt token": load_r_by_layer(REPO / "results/probes/heldout_eval_gemma3_tb-1/manifest.json"),
        "task-averaged":      load_r_by_layer(REPO / "results/probes/heldout_eval_gemma3_task_mean/manifest.json"),
    }

    # Qwen values: pulled from experiments/training_probes/qwen35_probes/qwen35_probes_report.md.
    # Only the three named positions we have in common with Gemma are included;
    # task-averaged was not run for Qwen.
    qwen_layers = [12, 24, 28, 33, 38, 43]
    qwen_data = {
        "end-of-turn":        dict(zip(qwen_layers, [0.877, 0.873, 0.875, 0.885, 0.889, 0.868])),
        "role-marker":        dict(zip(qwen_layers, [0.880, 0.899, 0.922, 0.932, 0.942, 0.943])),
        "final prompt token": dict(zip(qwen_layers, [0.880, 0.901, 0.923, 0.933, 0.943, 0.945])),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for position, r_map in gemma_data.items():
        layers = sorted(r_map)
        xs = [L / GEMMA_DEPTH for L in layers]
        ys = [r_map[L] for L in layers]
        ax.plot(
            xs, ys, "s--", color=COLORS[position],
            linewidth=1.6, markersize=6, alpha=0.7,
            label=f"Gemma {position}",
        )

    for position, r_map in qwen_data.items():
        layers = sorted(r_map)
        xs = [L / QWEN_DEPTH for L in layers]
        ys = [r_map[L] for L in layers]
        ax.plot(
            xs, ys, "o-", color=COLORS[position],
            linewidth=1.8, markersize=6,
            label=f"Qwen {position}",
        )

    ax.set_xlabel("Layer (fraction of model depth)")
    ax.set_ylabel("Held-out Pearson $r$")
    ax.set_ylim(0.75, 0.96)
    ax.set_xlim(0.15, 1.18)
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8, ncol=1, framealpha=0.9,
        loc="upper left", bbox_to_anchor=(1.0, 1.0),
    )

    fig.tight_layout()
    out = PAPER_FIG / f"plot_{DATE}_position_sweep_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
