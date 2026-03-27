"""Plot ethical flagging rate across all 4 scenarios at best coefficient.

Shows the 3 round-2 conditions (generation_only, critical_info+generation,
non_critical+generation) for each scenario. Best coefficient chosen per scenario
to show the clearest separation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ASSETS = Path("docs/logs/assets")

# Data from safety_steering_report.md tables (best positive coefficient per scenario)
# Pick the coefficient where critical_info+generation shows the clearest effect
scenarios = [
    "B: Resume bias\n(+0.03)",
    "D: COPPA violation\n(+0.07)",
    "A: Investor email\n(+0.07)",
]

# [generation_only, critical_info+generation, non_critical+generation]
data = {
    "generation_only":              [100, 80, 60],
    "critical_info + generation":   [  0, 20, 80],
    "non_critical + generation":    [ 70, 50, 50],
}

colors = {
    "generation_only": "#1f77b4",
    "critical_info + generation": "#d62728",
    "non_critical + generation": "#999999",
}

fig, ax = plt.subplots(figsize=(6.5, 3.5))

x = np.arange(len(scenarios))
width = 0.25

for i, (name, vals) in enumerate(data.items()):
    ax.bar(x + i * width, vals, width, label=name, color=colors[name], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(scenarios, fontsize=9)
ax.set_ylabel("Ethical flagging rate (%)")
ax.set_ylim(0, 110)
ax.set_title("Prefill steering on critical tokens: effect depends on info structure", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Annotations
ax.annotate("distributed info\n→ amplifies suppression", xy=(0.5, 5), fontsize=7.5,
            ha="center", color="#d62728", style="italic")
ax.annotate("isolated info\n→ no amplification", xy=(2.25, 5), fontsize=7.5,
            ha="center", color="#d62728", style="italic")

fig.tight_layout()
fig.savefig(ASSETS / "plot_032626_ethical_suppression_all_scenarios.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved ethical_suppression_all_scenarios")
