import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

DATA_PATH = Path("experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json")
OUT_PATH = Path("docs/logs/assets/plot_031826_snohomish_token_scores.png")

PROBE = "tb-5"
LAYER = "L39"
PAIR_IDX = 10  # Snohomish County


def text_color_for_background(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def draw_token_rows(ax, tokens, scores, cmap, norm, tokens_per_row=10, box_size=1.0, fontsize=13):
    n_rows = (len(tokens) + tokens_per_row - 1) // tokens_per_row
    ax.set_xlim(0, tokens_per_row)
    ax.set_ylim(-n_rows, 0)
    ax.axis("off")

    for i, (tok, score) in enumerate(zip(tokens, scores)):
        row = i // tokens_per_row
        col = i % tokens_per_row
        rgba = cmap(norm(score))
        rect = plt.Rectangle(
            (col, -row - box_size),
            box_size,
            box_size,
            facecolor=rgba,
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        tc = text_color_for_background(rgba)
        display = tok.replace("\n", "\\n").strip()
        ax.text(
            col + box_size / 2,
            -row - box_size / 2,
            display,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=tc,
            fontfamily="monospace",
            clip_on=True,
        )
    return n_rows


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    pairs: dict[str, dict[str, dict]] = {}
    for entry in data:
        ex_id = entry["true_ex_id"]
        cond = entry["answer_condition"]
        if ex_id not in pairs:
            pairs[ex_id] = {}
        pairs[ex_id][cond] = entry

    sorted_ids = sorted(pairs.keys())
    pair = pairs[sorted_ids[PAIR_IDX - 1]]
    correct = pair["correct"]
    incorrect = pair["incorrect"]

    c_tokens = correct["token_strings"]
    c_scores = correct["scores"][PROBE][LAYER]
    i_tokens = incorrect["token_strings"]
    i_scores = incorrect["scores"][PROBE][LAYER]

    all_scores = c_scores + i_scores
    abs_max = max(abs(min(all_scores)), abs(max(all_scores)))
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap("RdYlGn")

    tokens_per_row = 10
    c_rows = (len(c_tokens) + tokens_per_row - 1) // tokens_per_row
    i_rows = (len(i_tokens) + tokens_per_row - 1) // tokens_per_row

    fig_width = 14
    row_height = 0.9
    fig_height = (c_rows + i_rows) * row_height + 4

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[c_rows, i_rows], hspace=0.4)

    fig.suptitle("Snohomish County, Washington", fontsize=16, fontweight="bold", y=0.97)

    ax_top = fig.add_subplot(gs[0])
    ax_top.set_title("Correct answer", fontsize=13, loc="left", pad=8, fontstyle="italic")
    draw_token_rows(ax_top, c_tokens, c_scores, cmap, norm, tokens_per_row=tokens_per_row)

    ax_bot = fig.add_subplot(gs[1])
    ax_bot.set_title("Incorrect answer", fontsize=13, loc="left", pad=8, fontstyle="italic")
    draw_token_rows(ax_bot, i_tokens, i_scores, cmap, norm, tokens_per_row=tokens_per_row)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], location="bottom", shrink=0.5, pad=0.15, aspect=30)
    cbar.set_label(f"Probe score ({PROBE} / {LAYER})", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
