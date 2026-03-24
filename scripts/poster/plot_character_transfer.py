"""Generate compact character probe transfer bar chart for the poster.
4 personas, showing baseline (grey) and probe (blue), with misalignment negative baseline."""

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 14

fig, ax = plt.subplots(figsize=(10, 4))

personas = ['Mathematical', 'Loving', 'Nonchalant', 'Misalignment']
baseline_r = [0.48, 0.46, 0.35, -0.14]
probe_r = [0.80, 0.80, 0.57, 0.25]

x = np.arange(len(personas))
w = 0.35

bars_base = ax.bar(x - w/2, baseline_r, w, color='#D1D5DB', edgecolor='#9CA3AF', linewidth=1.2, label='Utility correlation')
bars_probe = ax.bar(x + w/2, probe_r, w, color='#93C5FD', edgecolor='#1D4ED8', linewidth=1.2, label='Probe prediction')

# Value labels on top of each bar
for bar, val in zip(bars_base, baseline_r):
    y_pos = val + 0.02 if val >= 0 else val - 0.08
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}', ha='center', va=va, fontsize=11, color='#6B7280', fontweight='bold')

for bar, val in zip(bars_probe, probe_r):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=11, color='#1D4ED8', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(personas, fontsize=15, fontweight='bold')
ax.set_ylabel('Pearson r', fontsize=14)
ax.axhline(0, color='#374151', linewidth=1)
ax.set_ylim(-0.45, 1.0)
ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = 'docs/poster/assets/plot_032326_character_transfer_poster.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
