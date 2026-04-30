"""Cosine similarity heatmap of the seven per-persona probes.

Pairwise cosine similarity between the ridge probe weight vectors at
(eot, L32) for the seven personas, ordered left-to-right by utility
similarity to default. Saved to paper/figures/main/.
"""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROBE_BASE = ROOT / "results/probes/persona_sweep_final_six"
OUT_DIR = ROOT / "paper/figures/main"
TODAY = date.today().strftime("%m%d%y")

# Persona ordering: most-default-like → most-default-opposed
# (matches the ordering used in Fig 7 / probe_transfer report)
PERSONAS = ["default", "aura", "mathematician", "strategist", "contrarian", "slacker", "sadist"]
SELECTOR = "tb-5"
LAYER = 32


def load_probe_direction(persona: str) -> np.ndarray:
    path = PROBE_BASE / f"{persona}_{SELECTOR}" / "probes" / f"probe_ridge_L{LAYER}.npy"
    weights = np.load(path)
    return weights[:-1]  # drop intercept


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(PERSONAS)} probes ({SELECTOR}, L{LAYER})...")
    directions = np.stack([load_probe_direction(p) for p in PERSONAS])
    print(f"  shape: {directions.shape}")

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit = directions / norms
    cos_mat = unit @ unit.T
    print(f"  off-diagonal cos: mean={cos_mat[np.triu_indices(len(PERSONAS), k=1)].mean():+.3f}, "
          f"min={cos_mat[np.triu_indices(len(PERSONAS), k=1)].min():+.3f}, "
          f"max={cos_mat[np.triu_indices(len(PERSONAS), k=1)].max():+.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    off_diag_mask = ~np.eye(len(PERSONAS), dtype=bool)
    abs_max = float(np.abs(cos_mat[off_diag_mask]).max())
    vmax = max(abs_max, 0.1)
    display = np.where(off_diag_mask, cos_mat, np.nan)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(display, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")

    for i in range(len(PERSONAS)):
        for j in range(len(PERSONAS)):
            if i == j:
                continue
            val = cos_mat[i, j]
            color = "white" if abs(val) > 0.7 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(len(PERSONAS)))
    ax.set_yticks(range(len(PERSONAS)))
    ax.set_xticklabels(PERSONAS, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(PERSONAS, fontsize=10)
    ax.set_title(f"Pairwise cosine similarity of per-persona probes  ({SELECTOR}, L{LAYER})",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="cos(probe$_i$, probe$_j$)")
    fig.tight_layout()

    out = OUT_DIR / f"plot_{TODAY}_persona_probe_cosine.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
