"""Single-metric ablation plot for App. G.2 — now including L23.

Extends the parent's L25/L32/{L25,L32}/band conditions with the L23 follow-up.
Reads parent CSV plus the L23 follow-up CSV (matched-pair-scope rows).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PARENT_SUMMARY = ROOT / "experiments/preference_direction_ablation/results/summary.csv"
L23_SUMMARY = ROOT / "experiments/preference_direction_ablation/L23_followup/results/summary_matched.csv"
OUT = ROOT / "paper/figures/appendix/plot_043026_uniqueness_ablation_agreement.png"

# Conditions to plot left-to-right. L23 is the follow-up; L25/L32 / both / band come from the parent.
CONDITIONS = [
    ("L23", "L23"),
    ("A_L25", "L25"),
    ("A_L32", "L32"),
    ("B_two", "L25 + L32"),
    ("C_band", "L25–L34 band"),
]


def load_csv(path):
    return list(csv.DictReader(path.open()))


def parent_by_cell(rows, name):
    for r in rows:
        if r["cell"] == name:
            return r
    raise KeyError(name)


def l23_by_cell(rows, name):
    # L23 follow-up CSV has a "scope" column; we want matched_100 rows
    for r in rows:
        if r["cell"] == name and r.get("scope", "") == "matched_100":
            return r
    raise KeyError(name)


parent_rows = load_csv(PARENT_SUMMARY)
l23_rows = load_csv(L23_SUMMARY)

fig, ax = plt.subplots(figsize=(7.0, 3.7))
ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5, label="Baseline (no projection)")

for i, (cond, label) in enumerate(CONDITIONS):
    if cond == "L23":
        probe_val = float(l23_by_cell(l23_rows, "A_L23_probe")["agreement_vs_b0"])
        random_vals = [float(l23_by_cell(l23_rows, f"A_L23_random{j}")["agreement_vs_b0"]) for j in range(5)]
    else:
        probe_val = float(parent_by_cell(parent_rows, f"{cond}_probe")["agreement_vs_b0"])
        random_vals = [float(parent_by_cell(parent_rows, f"{cond}_random{j}")["agreement_vs_b0"]) for j in range(5)]

    ax.scatter([i] * 5, random_vals, color="grey", alpha=0.55, s=42, zorder=2,
               label="Random direction (5 draws)" if i == 0 else None)
    ax.scatter([i], [probe_val], marker="*", color="#d97706", s=260, zorder=4,
               edgecolor="black", linewidth=0.7,
               label="Canonical preference direction" if i == 0 else None)

# Visually separate L23 (steering causal peak) from the parent's read-out layers
ax.axvspan(-0.5, 0.5, alpha=0.08, color="#d97706")
ax.text(0, 1.04, "Steering peak", ha="center", fontsize=9, color="#92400e")
ax.text(2.5, 1.04, "Probe-readout peak / past it", ha="center", fontsize=9, color="#374151")

ax.set_xticks(list(range(len(CONDITIONS))))
ax.set_xticklabels([label for _, label in CONDITIONS])
ax.set_xlabel("Layer(s) projected out")
ax.set_ylabel("Agreement with baseline\n(fraction of pairs, modal choice)")
ax.set_ylim(0.70, 1.07)
ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
