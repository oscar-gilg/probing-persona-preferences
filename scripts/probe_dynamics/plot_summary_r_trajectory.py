"""One-line-per-condition trajectory of within-checkpoint Pearson r."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2] / "experiments" / "probe_dynamics"
df = pd.read_csv(BASE / "analysis" / "checkpoint_r.csv")

COLOURS = {
    "onpolicy_consciousness":   "#1f77b4",
    "offpolicy_consciousness":  "#7fbfff",
    "qwen_delusion":            "#0a2472",
    "onpolicy_harm_compliance": "#d62728",
    "offpolicy_harm_compliance":"#ff9896",
    "icl_misalignment":         "#9467bd",
    "control_helpful":          "#7f7f7f",
}
ORDER = [
    "qwen_delusion", "onpolicy_consciousness", "offpolicy_consciousness",
    "onpolicy_harm_compliance", "offpolicy_harm_compliance",
    "icl_misalignment", "control_helpful",
]

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
for cond in ORDER:
    g = df[df["condition"] == cond].sort_values("checkpoint")
    ax.plot(g["checkpoint"], g["pearson_r"], marker="o", ms=3, linewidth=1.5,
            color=COLOURS[cond], label=cond)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("checkpoint (turn of prefilled conversation)")
ax.set_ylabel("Pearson r  (probe score vs behaviour, across prompts)")
ax.set_title("Within-checkpoint correlation between probe score and behaviour, per condition")
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.25)
ax.legend(loc="lower left", ncol=2, fontsize=9, framealpha=0.92)

# Annotate regions
ax.axhspan(0.2, 1.0, alpha=0.05, color="green")
ax.axhspan(-1.0, -0.2, alpha=0.05, color="red")

fig.tight_layout()
out = BASE / "assets" / "plot_042226_within_checkpoint_r_by_condition.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
