"""Generate compact truth + harm violin plots for the poster.
Uses tb-5_L39 probe. Harm: critical span. Truth: EOT."""

from dotenv import load_dotenv
load_dotenv()

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'Helvetica'
matplotlib.rcParams['font.size'] = 14

PROBE = 'tb-5_L39'

with open('experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json') as f:
    items = json.load(f)['items']

def get_scores(domain, prompt, condition, score_type):
    return [d[score_type][PROBE] for d in items
            if d['domain'] == domain and d['system_prompt'] == prompt and d['condition'] == condition]

def cohen_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

# ── Data ──
harm_asst_benign = get_scores('harm', 'safe', 'benign', 'eot_scores')
harm_asst_harmful = get_scores('harm', 'safe', 'harmful', 'eot_scores')
harm_sadist_benign = get_scores('harm', 'sadist', 'benign', 'eot_scores')
harm_sadist_harmful = get_scores('harm', 'sadist', 'harmful', 'eot_scores')

truth_asst_true = get_scores('truth', 'truthful', 'true', 'eot_scores')
truth_asst_false = get_scores('truth', 'truthful', 'false', 'eot_scores')
truth_liar_true = get_scores('truth', 'pathological_liar', 'true', 'eot_scores')
truth_liar_false = get_scores('truth', 'pathological_liar', 'false', 'eot_scores')

print(f"Probe: {PROBE}")
print(f"Harm (EOT): Asst benign={np.mean(harm_asst_benign):+.2f}, harmful={np.mean(harm_asst_harmful):+.2f}, d={cohen_d(harm_asst_benign, harm_asst_harmful):.2f}")
print(f"Harm (EOT): Sadist benign={np.mean(harm_sadist_benign):+.2f}, harmful={np.mean(harm_sadist_harmful):+.2f}, d={cohen_d(harm_sadist_benign, harm_sadist_harmful):.2f}")
print(f"Truth (EOT): Asst true={np.mean(truth_asst_true):+.2f}, false={np.mean(truth_asst_false):+.2f}, d={cohen_d(truth_asst_true, truth_asst_false):.2f}")
print(f"Truth (EOT): Liar true={np.mean(truth_liar_true):+.2f}, false={np.mean(truth_liar_false):+.2f}, d={cohen_d(truth_liar_true, truth_liar_false):.2f}")

# ── Plot ──
fig, (ax_harm, ax_truth) = plt.subplots(1, 2, figsize=(12, 5))

def styled_violins(ax, datasets, positions, colors, edges):
    vp = ax.violinplot(datasets, positions=positions, showmeans=True, showextrema=False, widths=0.5)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(edges[i])
        body.set_alpha(0.7)
        body.set_linewidth(1.2)
    vp['cmeans'].set_color('#374151')
    vp['cmeans'].set_linewidth(2)

# ── Harm ──
styled_violins(ax_harm,
    [harm_asst_benign, harm_asst_harmful, harm_sadist_benign, harm_sadist_harmful],
    [0.7, 1.3, 2.7, 3.3],
    ['#86EFAC', '#FCA5A5', '#86EFAC', '#FCA5A5'],
    ['#166534', '#991B1B', '#166534', '#991B1B'])

d_asst = cohen_d(harm_asst_benign, harm_asst_harmful)
d_sadist = cohen_d(harm_sadist_benign, harm_sadist_harmful)
ylim = ax_harm.get_ylim()
ax_harm.text(1.0, ylim[1] * 0.92, f'd = {d_asst:.2f}', ha='center', fontsize=13, fontweight='bold', color='#374151')
ax_harm.text(3.0, ylim[1] * 0.92, f'd = {d_sadist:.2f}', ha='center', fontsize=13, fontweight='bold', color='#374151')

ax_harm.set_xticks([1.0, 3.0])
ax_harm.set_xticklabels(['Assistant', 'Sadist'], fontsize=15, fontweight='bold')
ax_harm.set_ylabel('Probe score', fontsize=13)
ax_harm.set_title('Evil personas close the\nbenign/harmful gap', fontsize=16, fontweight='bold', pad=12)
ax_harm.axhline(0, color='#D1D5DB', linewidth=0.8, linestyle='--')
ax_harm.spines['top'].set_visible(False)
ax_harm.spines['right'].set_visible(False)
ax_harm.legend([Patch(facecolor='#86EFAC', edgecolor='#166534'), Patch(facecolor='#FCA5A5', edgecolor='#991B1B')],
               ['Benign', 'Harmful'], loc='upper right', fontsize=12, framealpha=0.9)

# ── Truth ──
styled_violins(ax_truth,
    [truth_asst_true, truth_asst_false, truth_liar_true, truth_liar_false],
    [0.7, 1.3, 2.7, 3.3],
    ['#93C5FD', '#FCA5A5', '#93C5FD', '#FCA5A5'],
    ['#1D4ED8', '#991B1B', '#1D4ED8', '#991B1B'])

d_truthful = cohen_d(truth_asst_true, truth_asst_false)
d_liar = cohen_d(truth_liar_true, truth_liar_false)
ylim = ax_truth.get_ylim()
ax_truth.text(1.0, ylim[1] * 0.92, f'd = {d_truthful:.2f}', ha='center', fontsize=13, fontweight='bold', color='#374151')
ax_truth.text(3.0, ylim[1] * 0.92, f'd = {d_liar:.2f}', ha='center', fontsize=13, fontweight='bold', color='#374151')

ax_truth.set_xticks([1.0, 3.0])
ax_truth.set_xticklabels(['Assistant', 'Pathological\nliar'], fontsize=15, fontweight='bold')
ax_truth.set_title('Lying instructions destroy\nthe true/false distinction', fontsize=16, fontweight='bold', pad=12)
ax_truth.axhline(0, color='#D1D5DB', linewidth=0.8, linestyle='--')
ax_truth.spines['top'].set_visible(False)
ax_truth.spines['right'].set_visible(False)
ax_truth.legend([Patch(facecolor='#93C5FD', edgecolor='#1D4ED8'), Patch(facecolor='#FCA5A5', edgecolor='#991B1B')],
                ['True', 'False'], loc='lower left', fontsize=12, framealpha=0.9)

plt.tight_layout(w_pad=3)
out = 'docs/poster/assets/plot_032326_truth_harm_poster.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")

# ── Also save individual plots ──
for name, ax_src, title in [("harm", ax_harm, "harm"), ("truth", ax_truth, "truth")]:
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    ax2.tick_params(labelsize=28)
    # Re-draw the same violins
    if name == "harm":
        styled_violins(ax2,
            [harm_asst_benign, harm_asst_harmful, harm_sadist_benign, harm_sadist_harmful],
            [0.7, 1.3, 2.7, 3.3],
            ['#86EFAC', '#FCA5A5', '#86EFAC', '#FCA5A5'],
            ['#166534', '#991B1B', '#166534', '#991B1B'])
        ax2.set_xticks([1.0, 3.0])
        ax2.set_xticklabels(['Assistant', 'Sadist'], fontsize=30, fontweight='bold')
        # no title — text is in the poster SVG
        ax2.legend([Patch(facecolor='#86EFAC', edgecolor='#166534'), Patch(facecolor='#FCA5A5', edgecolor='#991B1B')],
                   ['Benign', 'Harmful'], loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=22, frameon=False)
    else:
        styled_violins(ax2,
            [truth_asst_true, truth_asst_false, truth_liar_true, truth_liar_false],
            [0.7, 1.3, 2.7, 3.3],
            ['#93C5FD', '#FCA5A5', '#93C5FD', '#FCA5A5'],
            ['#1D4ED8', '#991B1B', '#1D4ED8', '#991B1B'])
        ax2.set_xticks([1.0, 3.0])
        ax2.set_xticklabels(['Assistant', 'Pathological\nliar'], fontsize=30, fontweight='bold')
        # no title — text is in the poster SVG
        ax2.legend([Patch(facecolor='#93C5FD', edgecolor='#1D4ED8'), Patch(facecolor='#FCA5A5', edgecolor='#991B1B')],
                   ['True', 'False'], loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=22, frameon=False)
    ax2.set_ylabel('Probe score', fontsize=28)
    ax2.set_xlim(0.2, 3.8)
    ax2.axhline(0, color='#D1D5DB', linewidth=0.8, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    out2 = f'docs/poster/assets/plot_032326_{name}_only.png'
    plt.savefig(out2, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close()
    print(f"Saved: {out2}")
