"""Plot random-ablation disruption (1 - agreement_vs_b0) vs number of ablated layers.

Run from worktree root:
    python -m scripts.preference_direction_ablation.plot_disruption_vs_n_layers
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parents[2]
SUMMARY_CSV = WORKTREE / "experiments/preference_direction_ablation/results/summary.csv"
OUT_PNG = WORKTREE / "experiments/preference_direction_ablation/assets/plot_042826_random_disruption_by_n_layers.png"

# (n_layers, list of random cell names, list of probe cell names)
GROUPS = [
    (1, [f"A_L25_random{i}" for i in range(5)] + [f"A_L32_random{i}" for i in range(5)],
        ["A_L25_probe", "A_L32_probe"]),
    (2, [f"B_two_random{i}" for i in range(5)], ["B_two_probe"]),
    (10, [f"C_band_random{i}" for i in range(5)], ["C_band_probe"]),
]


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    by_cell = df.set_index("cell")["agreement_vs_b0"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    rng = np.random.default_rng(0)
    medians_x: list[float] = []
    medians_y: list[float] = []

    for n_layers, random_cells, probe_cells in GROUPS:
        rand_vals = np.array([1.0 - by_cell[c] for c in random_cells])
        # jitter on log scale: jitter in log-space so spread is symmetric visually
        log_x = np.log10(n_layers)
        jitter = rng.uniform(-0.04, 0.04, size=len(rand_vals))
        xs = 10 ** (log_x + jitter)
        ax.scatter(xs, rand_vals, s=22, color="#888888", alpha=0.7,
                   edgecolor="none", label="random ablation" if n_layers == 1 else None,
                   zorder=2)

        med = float(np.median(rand_vals))
        medians_x.append(n_layers)
        medians_y.append(med)

        probe_vals = np.array([1.0 - by_cell[c] for c in probe_cells])
        ax.scatter([n_layers] * len(probe_vals), probe_vals,
                   s=80, color="#d62728", marker="D", edgecolor="black", linewidth=0.6,
                   label="probe ablation" if n_layers == 1 else None, zorder=4)

    ax.plot(medians_x, medians_y, color="#444444", linewidth=1.2, linestyle="-",
            marker="o", markersize=4, label="random median", zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 10])
    ax.set_xticklabels(["1", "2", "10"])
    ax.set_xlim(0.7, 14)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("Number of layers ablated")
    ax.set_ylabel(r"$1 - \mathrm{agreement\ vs\ B0}$  (disruption)")
    ax.set_title("Probe ablation stays near zero across 1, 2, and 10 layers;\nrandom ablation does not (and is non-monotonic in n_layers)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
