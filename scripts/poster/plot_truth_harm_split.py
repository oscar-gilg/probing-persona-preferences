"""Generate separate harm and truth violin plots for poster diagonal layout.
Reads from scoring_results.json."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'Helvetica Neue',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

# Load data
with open("experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json") as f:
    data = json.load(f)

out = Path("docs/poster/assets")
out.mkdir(parents=True, exist_ok=True)

# --- Harm plot ---
harm_data = data["harm_eot"]
fig, ax = plt.subplots(figsize=(4, 3.5))

positions = [0, 0.6, 1.4, 2.0]
colors = ["#86EFAC", "#FCA5A5", "#86EFAC", "#FCA5A5"]
labels = ["Benign", "Harmful", "Benign", "Harmful"]

for i, (key, color) in enumerate(zip(["assistant_benign", "assistant_harmful", "sadist_benign", "sadist_harmful"], colors)):
    vals = harm_data[key]
    parts = ax.violinplot([vals], positions=[positions[i]], showmeans=False, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxs'):
        if partname in parts:
            parts[partname].set_visible(False)
    ax.plot(positions[i], np.median(vals), '_', color='#374151', markersize=15, markeredgewidth=2)

# Cohen's d annotations
d_asst = harm_data.get("d_assistant", "")
d_sadist = harm_data.get("d_sadist", "")
if d_asst:
    ax.text(0.3, ax.get_ylim()[1] * 0.9, f'd = {d_asst:.2f}', ha='center', fontsize=10, color='#6B7280')
if d_sadist:
    ax.text(1.7, ax.get_ylim()[1] * 0.9, f'd = {d_sadist:.2f}', ha='center', fontsize=10, color='#6B7280')

ax.set_xticks([0.3, 1.7])
ax.set_xticklabels(['Assistant', 'Sadist'], fontsize=11)
ax.set_ylabel('Probe score', fontsize=11)
ax.axhline(0, color='#E5E7EB', linewidth=0.5, linestyle='--')
ax.set_title('Evil personas close the\nbenign/harmful gap', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(out / "plot_032326_harm_only.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved harm plot")

# --- Truth plot ---
truth_data = data["truth_eot"]
fig, ax = plt.subplots(figsize=(4, 3.5))

for i, (key, color) in enumerate(zip(["assistant_true", "assistant_false", "liar_true", "liar_false"],
                                      ["#93C5FD", "#FCA5A5", "#93C5FD", "#FCA5A5"])):
    vals = truth_data[key]
    parts = ax.violinplot([vals], positions=[positions[i]], showmeans=False, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxs'):
        if partname in parts:
            parts[partname].set_visible(False)
    ax.plot(positions[i], np.median(vals), '_', color='#374151', markersize=15, markeredgewidth=2)

d_asst = truth_data.get("d_assistant", "")
d_liar = truth_data.get("d_liar", "")
if d_asst:
    ax.text(0.3, ax.get_ylim()[1] * 0.9, f'd = {d_asst:.2f}', ha='center', fontsize=10, color='#6B7280')
if d_liar:
    ax.text(1.7, ax.get_ylim()[1] * 0.9, f'd = {d_liar:.2f}', ha='center', fontsize=10, color='#6B7280')

ax.set_xticks([0.3, 1.7])
ax.set_xticklabels(['Assistant', 'Pathological\nliar'], fontsize=11)
ax.set_ylabel('Probe score', fontsize=11)
ax.axhline(0, color='#E5E7EB', linewidth=0.5, linestyle='--')
ax.set_title('Lying instructions destroy\nthe true/false distinction', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(out / "plot_032326_truth_only.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved truth plot")
