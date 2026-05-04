"""Plots for the all-personas transfer-gap analysis.

Outputs:
  plot_050426_gap_per_persona.png         # main: gap by (model, persona, direction)
  plot_050426_within_r_per_persona.png    # context: how decodable each persona's utilities are
  plot_050426_utility_correlations.png    # context: how persona utilities correlate with default
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
OUT = Path(__file__).parent
SUMMARY_CSV = ROOT / "experiments/sft_sadist/probe_subspace_replication/results/all_personas_transfer_summary.csv"

PERSONA_ORDER = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]
MODEL_ORDER = ["Qwen-3.5-122B", "Gemma-3-27B"]


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df["persona"] = pd.Categorical(df["persona"], categories=PERSONA_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["model", "persona"]).reset_index(drop=True)

    # ---- Plot 1: Gap per (model, persona, direction) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    x = np.arange(len(PERSONA_ORDER))
    w = 0.36
    for ax, model in zip(axes, MODEL_ORDER):
        sub = df[df["model"] == model]
        gp = sub["gap_persona"].values
        gd = sub["gap_default"].values
        ax.bar(x - w / 2, gp, w, color="#cf8866",
               label="gap predicting persona utilities\n(within-persona r − cross-trained r)")
        ax.bar(x + w / 2, gd, w, color="#2e6cb6",
               label="gap predicting default utilities\n(within-default r − cross-trained r)")
        for xi, v in zip(x - w / 2, gp):
            ax.text(xi, v + 0.001 if v >= 0 else v - 0.003, f"{v:.3f}",
                    ha="center", fontsize=8, va="bottom" if v >= 0 else "top")
        for xi, v in zip(x + w / 2, gd):
            ax.text(xi, v + 0.001 if v >= 0 else v - 0.003, f"{v:.3f}",
                    ha="center", fontsize=8, va="bottom" if v >= 0 else "top")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(PERSONA_ORDER, rotation=20, ha="right")
        ax.set_title(model)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("transfer gap (Pearson r)")
    axes[1].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.suptitle("Cross-train transfer gap: ~0 in all (model, persona) cells",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050426_gap_per_persona.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: Within-domain r per persona ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    bar_w = 0.55
    for ax, model in zip(axes, MODEL_ORDER):
        sub = df[df["model"] == model]
        rpp = sub["r_PP"].values
        rdd_mean = sub["r_DD"].mean()  # constant across personas — show as line
        ax.bar(x, rpp, bar_w, color="#cf8866", label="within-persona r")
        ax.axhline(rdd_mean, color="#2e6cb6", ls="--", lw=2,
                   label=f"within-default r = {rdd_mean:.2f}")
        for xi, v in zip(x, rpp):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(PERSONA_ORDER, rotation=20, ha="right")
        ax.set_title(model)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("within-domain Pearson r (best layer)")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)
    axes[1].legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.suptitle("Within-persona probe accuracy varies widely; within-default is essentially constant",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050426_within_r_per_persona.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: Utility correlation per persona ----
    fig, ax = plt.subplots(figsize=(9, 4))
    qwen_uc = df[df["model"] == "Qwen-3.5-122B"]["util_corr"].values
    gemma_uc = df[df["model"] == "Gemma-3-27B"]["util_corr"].values
    ax.bar(x - w / 2, qwen_uc, w, color="#7e42a3", label="Qwen-3.5-122B")
    ax.bar(x + w / 2, gemma_uc, w, color="#3a8c46", label="Gemma-3-27B")
    for xi, v in zip(x - w / 2, qwen_uc):
        ax.text(xi, v + 0.01 if v >= 0 else v - 0.02, f"{v:.2f}",
                ha="center", fontsize=8.5, va="bottom" if v >= 0 else "top")
    for xi, v in zip(x + w / 2, gemma_uc):
        ax.text(xi, v + 0.01 if v >= 0 else v - 0.02, f"{v:.2f}",
                ha="center", fontsize=8.5, va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(PERSONA_ORDER)
    ax.set_ylabel("utility correlation\n(persona Thurstonian ↔ default Thurstonian)")
    ax.set_title("Persona vs default utility correlation (n=6000 tasks each)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050426_utility_correlations.png", dpi=150)
    plt.close()

    print("wrote 3 all-personas plots")


if __name__ == "__main__":
    main()
