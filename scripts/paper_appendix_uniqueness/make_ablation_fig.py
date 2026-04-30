"""Single-metric ablation plot for App. G.2."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments/preference_direction_ablation/results/summary.csv"
OUT = ROOT / "paper/figures/appendix/plot_043026_uniqueness_ablation_agreement.png"

CONDITIONS = [
    ("A_L25", "L25"),
    ("A_L32", "L32"),
    ("B_two", "L25 + L32"),
    ("C_band", "L25–L34 band"),
]

rows = list(csv.DictReader(SUMMARY.open()))


def by_cell(name):
    for r in rows:
        if r["cell"] == name:
            return r
    raise KeyError(name)


fig, ax = plt.subplots(figsize=(6.0, 3.6))

baseline = float(by_cell("A_L25_probe")["agreement_vs_b0"]) * 0  # placeholder; B0 vs itself = 1.0 by construction
ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5, label="Baseline (no projection)")

xs = list(range(len(CONDITIONS)))
for i, (cond, label) in enumerate(CONDITIONS):
    probe_val = float(by_cell(f"{cond}_probe")["agreement_vs_b0"])
    random_vals = [float(by_cell(f"{cond}_random{j}")["agreement_vs_b0"]) for j in range(5)]

    ax.scatter([i] * 5, random_vals, color="grey", alpha=0.55, s=42, zorder=2,
               label="Random direction (5 draws)" if i == 0 else None)
    ax.scatter([i], [probe_val], marker="*", color="#d97706", s=260, zorder=4,
               edgecolor="black", linewidth=0.7,
               label="Canonical preference direction" if i == 0 else None)

ax.set_xticks(xs)
ax.set_xticklabels([label for _, label in CONDITIONS])
ax.set_xlabel("Layer(s) projected out")
ax.set_ylabel("Agreement with baseline\n(fraction of pairs, modal choice)")
ax.set_ylim(0.70, 1.02)
ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
