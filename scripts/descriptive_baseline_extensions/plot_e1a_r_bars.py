"""§3.2 e1a Pearson r bar comparison: residual probe vs encoder baseline.

For each model, two grouped bars (on-target, all-tasks); two series per group
(residual probe, encoder baseline). Compact summary of where the residual probe
is close to the encoder (on-target) and where it pulls ahead (all-tasks).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "experiments/qwen_replication/e1a/e1a_per_task.json"
BASE_GEM = REPO / "experiments/descriptive_baseline_extensions/e1a_baseline_gemma-3-27b.json"
BASE_QWN = REPO / "experiments/descriptive_baseline_extensions/e1a_baseline_qwen-3.5-122b.json"
OUT = REPO / "experiments/descriptive_baseline_extensions/assets/plot_050426_e1a_r_bars.png"

RESIDUAL_COLOR = "#3d7aab"
ENCODER_COLOR = "#b08f3a"


def residual_r(model: str, selector: str) -> tuple[float, int, float, int]:
    d = json.load(open(RES))[model][selector]
    return (
        d["pooled_on_target"]["pearson_r"],
        d["pooled_on_target"]["n"],
        d["pooled_all"]["pearson_r"],
        d["pooled_all"]["n"],
    )


def encoder_r(path: Path) -> tuple[float, int, float, int]:
    d = json.load(open(path))["pearson_r"]
    return (
        d["on_target"]["r"],
        d["on_target"]["n"],
        d["all"]["r"],
        d["all"]["n"],
    )


def main() -> None:
    g_res_on, g_res_on_n, g_res_all, g_res_all_n = residual_r("gemma-3-27b", "prompt_last")
    q_res_on, q_res_on_n, q_res_all, q_res_all_n = residual_r("qwen-3.5-122b", "tb-1")
    g_enc_on, g_enc_on_n, g_enc_all, g_enc_all_n = encoder_r(BASE_GEM)
    q_enc_on, q_enc_on_n, q_enc_all, q_enc_all_n = encoder_r(BASE_QWN)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)

    for ax, (label, vals) in zip(
        axes,
        [
            ("Gemma-3-27B", [(g_res_on, g_enc_on), (g_res_all, g_enc_all)]),
            ("Qwen-3.5-122B", [(q_res_on, q_enc_on), (q_res_all, q_enc_all)]),
        ],
    ):
        x = np.arange(2)
        width = 0.36
        residual_vals = [v[0] for v in vals]
        encoder_vals = [v[1] for v in vals]
        b1 = ax.bar(x - width / 2, residual_vals, width,
                    color=RESIDUAL_COLOR, edgecolor="black", linewidth=0.5,
                    label="Residual probe")
        b2 = ax.bar(x + width / 2, encoder_vals, width,
                    color=ENCODER_COLOR, edgecolor="black", linewidth=0.5,
                    label="Encoder baseline (Qwen3-Emb-8B)")
        for b in list(b1) + list(b2):
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["on-target tasks", "all tasks"], fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if ax is axes[0]:
            ax.set_ylabel("Pearson r (probe-Δ vs behavioural-Δ)")

    axes[0].legend(loc="upper right", fontsize=9, frameon=True)
    fig.suptitle(
        "e1a induced shifts — residual probe vs encoder baseline (chat-template, full prompt)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"  Gemma residual on-target r={g_res_on:.3f} (n={g_res_on_n}), all r={g_res_all:.3f} (n={g_res_all_n})")
    print(f"  Gemma encoder  on-target r={g_enc_on:.3f} (n={g_enc_on_n}), all r={g_enc_all:.3f} (n={g_enc_all_n})")
    print(f"  Qwen  residual on-target r={q_res_on:.3f} (n={q_res_on_n}), all r={q_res_all:.3f} (n={q_res_all_n})")
    print(f"  Qwen  encoder  on-target r={q_enc_on:.3f} (n={q_enc_on_n}), all r={q_enc_all:.3f} (n={q_enc_all_n})")


if __name__ == "__main__":
    main()
