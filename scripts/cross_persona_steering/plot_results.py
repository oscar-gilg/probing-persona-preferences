"""Plot cross-persona steering validation results.

Reads experiments/cross_persona_steering/aggregated.json and writes:
  - assets/plot_<date>_cross_persona_steered_dose_response.png (4-panel P(steered) vs |c|)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path("experiments/cross_persona_steering")
ASSETS = EXP_DIR / "assets"
PERSONAS = ["sadist", "villain", "aesthete", "stem_obsessive"]
MAIN_CONDITION = "differential_L25_probeL32"
CONTROL_CONDITION = "differential_L25_random"


def date_tag() -> str:
    return dt.date.today().strftime("%m%d%y")


def cells_to_curve(cells: dict, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (|coef|, P(steered), SEM) arrays sorted by |coef| for a condition."""
    filtered = [c for c in cells.values() if c["condition"] == condition]
    filtered.sort(key=lambda c: c["abs_coefficient"])
    abs_c = np.array([c["abs_coefficient"] for c in filtered])
    means = np.array([c["mean_steered_chosen"] for c in filtered])
    sems = np.array([c["sem"] for c in filtered])
    # Prepend the symmetry anchor at |c|=0: by definition P(steered)=0.5 (neutral).
    abs_c = np.concatenate([[0.0], abs_c])
    means = np.concatenate([[0.5], means])
    sems = np.concatenate([[0.0], sems])
    return abs_c, means, sems


def plot_dose_response(agg: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True, sharex=True)
    axes = axes.ravel()
    for i, persona in enumerate(PERSONAS):
        ax = axes[i]
        data = agg["personas"].get(persona)
        if not data:
            ax.text(0.5, 0.5, f"(no data for {persona})", ha="center", va="center")
            ax.set_title(persona)
            continue
        cells = data["validation_cells"]
        x_main, y_main, e_main = cells_to_curve(cells, MAIN_CONDITION)
        x_ctrl, y_ctrl, e_ctrl = cells_to_curve(cells, CONTROL_CONDITION)
        ax.errorbar(x_main, y_main, yerr=e_main, marker="o", color="#1f77b4",
                    linewidth=2, markersize=7, label="probe direction (ridge_L32)", capsize=3)
        ax.errorbar(x_ctrl, y_ctrl, yerr=e_ctrl, marker="s", color="#888888",
                    linewidth=1.5, markersize=6, linestyle="--",
                    label="random direction (control)", capsize=3)
        ax.axhline(0.5, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_title(persona, fontsize=12, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(-0.003, 0.055)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9)
    for i in [2, 3]:
        axes[i].set_xlabel("|steering coefficient|  (fraction of L25 mean norm)")
    for i in [0, 2]:
        axes[i].set_ylabel("P(steered task was chosen)")
    fig.suptitle(
        "Differential steering under persona system prompts: P(steered task chosen) vs |coefficient|",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with open(EXP_DIR / "aggregated.json") as f:
        agg = json.load(f)
    ASSETS.mkdir(parents=True, exist_ok=True)
    tag = date_tag()
    path = ASSETS / f"plot_{tag}_cross_persona_steered_dose_response.png"
    plot_dose_response(agg, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
