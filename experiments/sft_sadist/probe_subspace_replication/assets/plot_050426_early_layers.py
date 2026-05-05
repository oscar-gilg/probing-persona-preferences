"""Cross-train gap by layer for all 6 Gemma personas, including the new early layer (L11)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
OUT = Path(__file__).parent
DATA = ROOT / "experiments/sft_sadist/probe_subspace_replication/results/gemma_early_layers.csv"

PERSONA_ORDER = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]
PERSONA_COLORS = {
    "aura":          "#1f77b4",
    "contrarian":    "#ff7f0e",
    "mathematician": "#2ca02c",
    "sadist":        "#d62728",
    "slacker":       "#9467bd",
    "strategist":    "#8c564b",
}


def main() -> None:
    df = pd.read_csv(DATA)
    layers = sorted(df["layer"].unique())
    depth_fracs = [L / 62 for L in layers]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, gap_col, title in zip(
        axes,
        ["gap_persona_target", "gap_default_target"],
        ["gap predicting persona utilities\n(within-persona r − cross-trained r)",
         "gap predicting default utilities\n(within-default r − cross-trained r)"],
    ):
        for persona in PERSONA_ORDER:
            sub = df[df["persona"] == persona].sort_values("layer")
            ax.plot(depth_fracs, sub[gap_col].values, "o-",
                    color=PERSONA_COLORS[persona], label=persona, lw=1.8, ms=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("layer depth (layer / 62)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(depth_fracs)
        ax.set_xticklabels([f"L{L}\n({d:.2f})" for L, d in zip(layers, depth_fracs)])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("transfer gap (Pearson r)")
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.suptitle("Cross-train gap by depth — Gemma-3-27B, all 6 personas, tb:-2 selector",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = OUT / "plot_050426_early_layers_gap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
