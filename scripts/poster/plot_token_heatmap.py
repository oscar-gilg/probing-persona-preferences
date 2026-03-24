"""Generate poster-quality token-level heatmap using actual Snohomish County data.

Uses pair 010 (Snohomish County, Washington) from the per-token scoring
experiment. All scores are real probe outputs (tb-5, L39). The incorrect
sentence is paraphrased shorter for poster readability; each token's score
is taken from its actual position in the original scored sequence.

Source: experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams['font.family'] = 'Helvetica'

# Real scores from scored_tokens.json, probe tb-5, layer L39
# "Snohomish" spans 4 subword tokens [Sn, oh, om, ish] → mean = -0.999

correct_tokens = [
    'Snohomish', 'County', ',', 'Washington',
    'includes', 'a', 'large', 'population', 'of', 'people', '.',
]
correct_scores = [
    -0.999, 0.206, -1.088, 0.622,
    0.456, -0.352, -1.288, 0.020, 1.050, 1.048, 3.146,
]

# Paraphrased shorter — each token's score from its real position in the original
incorrect_tokens = [
    'Snohomish', 'County', ',', 'Washington',
    'DC', 'where', 'the', 'President', 'resides', '.',
]
incorrect_scores = [
    -0.999, 0.206, -1.088, 0.622,   # shared prefix (identical)
    -5.349, -2.537, -1.163, -3.455, -6.193, -6.283,
]

norm = TwoSlopeNorm(vmin=-7, vcenter=0, vmax=7)
cmap = plt.cm.RdYlGn

fig, axes = plt.subplots(2, 1, figsize=(11, 3.2), gridspec_kw={'hspace': 0.6})

FONT_TOKEN = 13
FONT_TITLE = 14


def draw_row(ax, tokens, scores, title, title_color):
    n = len(tokens)
    char_lens = [max(len(tok), 4) for tok in tokens]
    total_chars = sum(char_lens)
    total_width = 10.0
    gap = 0.1
    available = total_width - gap * (n - 1)
    widths = [available * cl / total_chars for cl in char_lens]

    box_h = 0.8
    x_cursor = 0
    for i, (tok, score, bw) in enumerate(zip(tokens, scores, widths)):
        color = cmap(norm(score))
        rect = plt.Rectangle(
            (x_cursor, 0), bw, box_h,
            facecolor=color, edgecolor='#888888', linewidth=0.8,
        )
        ax.add_patch(rect)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = 'white' if luminance < 0.45 else '#1a1a1a'
        ax.text(
            x_cursor + bw / 2, box_h / 2, tok,
            ha='center', va='center',
            fontsize=FONT_TOKEN, fontweight='bold', color=text_color,
        )
        x_cursor += bw + gap

    ax.set_xlim(-0.2, total_width + 0.2)
    ax.set_ylim(-0.15, box_h + 0.15)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', color=title_color, loc='left', pad=4)


draw_row(axes[0], correct_tokens, correct_scores,
         'Correct answer', '#166534')
draw_row(axes[1], incorrect_tokens, incorrect_scores,
         'Incorrect answer', '#991B1B')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.018, pad=0.03, shrink=0.85)
cbar.set_label('Probe score', fontsize=FONT_TOKEN - 2)
cbar.ax.tick_params(labelsize=FONT_TOKEN - 4)

out = 'docs/poster/assets/plot_032326_token_heatmap_poster.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
