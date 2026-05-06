"""Per-pair (harmful − benign) delta violins per persona, with side schematic.

Gemma-3-27B, prefilled-assistant turn, harm domain only.

Layout mirrors fig 3 (steering): a small explainer schematic on the left
followed by the data panel on the right. Each persona's x-tick is replaced
by a robot head matching the hero figure's visual vocabulary (default green,
aura indigo, evil red). Series are identified by a probe glyph (LM probe)
and a stylised "[T]→v" glyph (text encoder).

Pairing is by `base_id` (drops 5/500 unmatched singletons per persona).
Each series is z-normalised by its own *default-persona* pooled SD so both
y-values are in "Cohen's d units relative to default" — a fixed y-scale
across personas.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[3].parent
PROBE_SCORES = REPO / "experiments/eot_discrimination_v2/scoring/gemma3_27b/scoring_results.json"
ENC_PER_TASK = REPO / "experiments/descriptive_baseline_extensions/eot_baseline_per_task_assistant_gemma-3-27b.json"
ICON_DIR = REPO / "paper/figures/panels/icons"
OUT = REPO / "paper/figures/main/plot_050626_harm_persona_paired_delta_violin_v2.png"
OUT_PDF = REPO / "paper/figures/main/plot_050626_harm_persona_paired_delta_violin_v2.pdf"

PERSONA_ORDER = ["neutral", "aura", "sadist"]
PERSONA_DISPLAY = {"neutral": "assistant", "aura": "aura", "sadist": "evil"}
PERSONA_COLOR = {"neutral": "#166534", "aura": "#4F46E5", "sadist": "#991B1B"}
PROBE_NAME = "tb-5_L32"
PROBE_COLOR = "#1f5f9c"
ENC_COLOR = "#d97706"
HARM_COLOR = "#FEE2E2"
HARM_STROKE = "#991B1B"
BENIGN_COLOR = "#D1FAE5"
BENIGN_STROKE = "#166534"


def base_of(item_id: str) -> str:
    return re.sub(r"_(harmful|benign)_", "_X_", item_id)


def paired_deltas(items, score_fn):
    by_sp_base: dict = defaultdict(lambda: defaultdict(dict))
    for it in items:
        b = base_of(it["id"])
        by_sp_base[it["system_prompt"]][b][it["condition"]] = score_fn(it)
    out = {}
    for sp, bases in by_sp_base.items():
        deltas = []
        for conds in bases.values():
            if "harmful" in conds and "benign" in conds:
                deltas.append(conds["harmful"] - conds["benign"])
        out[sp] = np.array(deltas)
    return out


def neutral_pooled_sd(items, score_fn) -> float:
    pos = np.array([score_fn(it) for it in items if it["system_prompt"] == "neutral" and it["condition"] == "harmful"])
    neg = np.array([score_fn(it) for it in items if it["system_prompt"] == "neutral" and it["condition"] == "benign"])
    n1, n2 = len(pos), len(neg)
    return float(np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1)) / (n1 + n2 - 2)))


def draw_probe_glyph(ax, x, y, color=PROBE_COLOR, size=18, y_stretch=1.0):
    """Square box with a diagonal arrow, matches the hero-figure probe-glyph.
    `y_stretch` scales vertical extent so the glyph renders square regardless
    of the ax's physical aspect ratio (set to ax_w_in / ax_h_in × ylim/xlim)."""
    half_x = size / 2
    half_y = (size / 2) * y_stretch
    box = mpatches.FancyBboxPatch(
        (x - half_x, y - half_y), size, size * y_stretch,
        boxstyle="round,pad=0,rounding_size=2.5",
        linewidth=1.4, edgecolor=color, facecolor="white",
        transform=ax.transData, clip_on=False, zorder=10,
    )
    ax.add_patch(box)
    arrow = FancyArrowPatch(
        (x - 0.3 * size, y - 0.3 * size * y_stretch),
        (x + 0.3 * size, y + 0.3 * size * y_stretch),
        arrowstyle="-|>,head_width=2.5,head_length=3",
        linewidth=1.6, color=color, mutation_scale=1.0,
        transform=ax.transData, clip_on=False, zorder=11,
    )
    ax.add_patch(arrow)


def draw_encoder_glyph(ax, x, y, color=ENC_COLOR, size=18, y_stretch=1.0):
    """Stylised text-encoder glyph: a square with rows of text-like lines."""
    half_x = size / 2
    half_y = (size / 2) * y_stretch
    box = mpatches.FancyBboxPatch(
        (x - half_x, y - half_y), size, size * y_stretch,
        boxstyle="round,pad=0,rounding_size=2.5",
        linewidth=1.4, edgecolor=color, facecolor="white",
        transform=ax.transData, clip_on=False, zorder=10,
    )
    ax.add_patch(box)
    rows = [
        (-0.27, [(-0.32, -0.05), (0.02, 0.30)]),
        (0.0,   [(-0.32, 0.10), (0.18, 0.30)]),
        (0.27,  [(-0.32, 0.05), (0.12, 0.30)]),
    ]
    for dy_frac, segs in rows:
        for x0, x1 in segs:
            ax.plot([x + x0 * size, x + x1 * size],
                    [y + dy_frac * size * y_stretch,
                     y + dy_frac * size * y_stretch],
                    color=color, linewidth=1.4,
                    transform=ax.transData, clip_on=False, zorder=11,
                    solid_capstyle="round")


def draw_schematic(ax, y_stretch=1.0):
    """Compact horizontal explainer (sits above the violin panel).
    HARMFUL / BENIGN header cards on top, two reader rows beneath, with each
    row's Δ written out as the abbreviated subtraction at the right end.

    `y_stretch` is the (x_unit / y_unit) ratio of the ax in display coords;
    glyph drawing uses this so squares render as squares regardless of the
    ax's physical aspect.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ── Column anchors ──────────────────────────────────────────────────────
    x_label_left, x_label_right = 0, 28
    x_h_left, x_h_right = 28, 52
    x_b_left, x_b_right = 52, 76
    x_delta_left, x_delta_right = 76, 100

    col_h = (x_h_left + x_h_right) / 2
    col_b = (x_b_left + x_b_right) / 2
    col_delta = (x_delta_left + x_delta_right) / 2

    # ── Row anchors ────────────────────────────────────────────────────────
    y_header_top = 100
    y_header_bot = 65
    y_probe_top = 65
    y_probe_bot = 35
    y_enc_top = 35
    y_enc_bot = 5

    row_probe = (y_probe_top + y_probe_bot) / 2
    row_enc = (y_enc_top + y_enc_bot) / 2

    # ── Dotted grid lines ───────────────────────────────────────────────────
    grid_kwargs = dict(color="#9CA3AF", linewidth=0.7,
                       linestyle=(0, (1.5, 1.5)))
    for gx in (x_label_right, x_h_right, x_b_right):
        ax.plot([gx, gx], [y_enc_bot, y_header_top], **grid_kwargs)
    for gy in (y_header_bot, y_probe_bot):
        ax.plot([x_label_left, x_delta_right], [gy, gy], **grid_kwargs)

    # ── Task column headers: HARMFUL / BENIGN caps title + small example ───
    def task_card(x_center, color_fill, color_stroke, header, body):
        card_w, card_h = 20, 28
        y_top = (y_header_top + y_header_bot) / 2
        card = FancyBboxPatch(
            (x_center - card_w / 2, y_top - card_h / 2),
            card_w, card_h,
            boxstyle="round,pad=0,rounding_size=2",
            linewidth=1.3, edgecolor=color_stroke, facecolor=color_fill,
        )
        ax.add_patch(card)
        ax.text(x_center, y_top + 5, header, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color_stroke)
        ax.text(x_center, y_top - 6, body, ha="center", va="center",
                fontsize=7, fontstyle="italic", color="#374151")

    task_card(col_h, HARM_COLOR, HARM_STROKE, "HARMFUL", '"make a phishing scam"')
    task_card(col_b, BENIGN_COLOR, BENIGN_STROKE, "BENIGN", '"spot a phishing scam"')

    # ── Row labels: glyph beside the two-line reader name ───────────────────
    def row_label(y_center, color, glyph_fn, label):
        glyph_fn(ax, 5, y_center, color=color, size=4, y_stretch=y_stretch)
        ax.text(12, y_center, label, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color=color, linespacing=1.0)

    row_label(row_probe, PROBE_COLOR, draw_probe_glyph, "preference\nvector")
    row_label(row_enc, ENC_COLOR, draw_encoder_glyph, "text-encoder\nbaseline")

    # ── Cell entries — full-word subscripts ─────────────────────────────────
    cells = (
        (col_h, row_probe, r"$p_{\mathrm{harmful}}$", PROBE_COLOR),
        (col_b, row_probe, r"$p_{\mathrm{benign}}$",  PROBE_COLOR),
        (col_h, row_enc,   r"$e_{\mathrm{harmful}}$", ENC_COLOR),
        (col_b, row_enc,   r"$e_{\mathrm{benign}}$",  ENC_COLOR),
    )
    for x, y, label, color in cells:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=12, color=color)

    # ── Per-row Δ equation — abbreviated h/b subscripts ─────────────────────
    ax.text(col_delta, row_probe,
            r"$\Delta_{\mathrm{probe}} = p_{\mathrm{h}} - p_{\mathrm{b}}$",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=PROBE_COLOR)
    ax.text(col_delta, row_enc,
            r"$\Delta_{\mathrm{enc}} = e_{\mathrm{h}} - e_{\mathrm{b}}$",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=ENC_COLOR)


def main() -> None:
    probe_items = [it for it in json.load(PROBE_SCORES.open())["items"]
                   if it["domain"] == "harm" and it["turn"] == "assistant"]
    enc_items = [it for it in json.load(ENC_PER_TASK.open())["items"]
                 if it["domain"] == "harm" and it["turn"] == "assistant"]

    probe_deltas = paired_deltas(probe_items, lambda it: it["probe_scores"][PROBE_NAME])
    enc_deltas = paired_deltas(enc_items, lambda it: it["score"])
    probe_sigma = neutral_pooled_sd(probe_items, lambda it: it["probe_scores"][PROBE_NAME])
    enc_sigma = neutral_pooled_sd(enc_items, lambda it: it["score"])
    probe_norm = {sp: probe_deltas[sp] / probe_sigma for sp in PERSONA_ORDER}
    enc_norm = {sp: enc_deltas[sp] / enc_sigma for sp in PERSONA_ORDER}

    fig = plt.figure(figsize=(11.0, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 3.1], hspace=0.025)
    ax_left = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    # Force layout so we can read the actual ax dimensions for y_stretch.
    fig.canvas.draw()
    bbox = ax_left.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_in = bbox.width * fig_w
    ax_h_in = bbox.height * fig_h
    y_stretch = ax_w_in / ax_h_in  # makes a square in axis-units render square
    draw_schematic(ax_left, y_stretch=y_stretch)

    # Violin panel
    width = 0.85
    gap = 0.5
    for i, sp in enumerate(PERSONA_ORDER):
        x_probe = i * 3 - gap
        x_enc = i * 3 + gap
        for x, vals, color in ((x_probe, probe_norm[sp], PROBE_COLOR),
                               (x_enc, enc_norm[sp], ENC_COLOR)):
            parts = ax.violinplot([vals], positions=[x], widths=width,
                                  showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.65)
                body.set_edgecolor("black")
                body.set_linewidth(0.8)
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.5)
        # numerical mean labels — placed beside the violin to avoid overlap
        ax.annotate(f"{probe_norm[sp].mean():+.2f}",
                    (x_probe - 0.55, probe_norm[sp].mean()),
                    ha="right", va="center", fontsize=10,
                    color=PROBE_COLOR, fontweight="bold")
        ax.annotate(f"{enc_norm[sp].mean():+.2f}",
                    (x_enc + 0.55, enc_norm[sp].mean()),
                    ha="left", va="center", fontsize=10,
                    color=ENC_COLOR, fontweight="bold")

    # Pin the y-range, then add a *progressive* shade above and below y=0
    # (lighter near zero, deeper farther away — emphasises distance-from-zero).
    ax.set_ylim(-8.0, 5.5)
    ymin, ymax = ax.get_ylim()
    n_steps = 40
    for i in range(n_steps):
        # above zero — pink, deeper at the top
        a = (i + 1) / n_steps
        y0 = ymax * (i / n_steps)
        y1 = ymax * ((i + 1) / n_steps)
        ax.axhspan(y0, y1, facecolor="#EF4444", alpha=0.13 * a,
                   linewidth=0, zorder=0)
        # below zero — neutral grey, deeper at the bottom
        y0 = ymin * (i / n_steps)
        y1 = ymin * ((i + 1) / n_steps)
        ax.axhspan(y1, y0, facecolor="#6B7280", alpha=0.13 * a,
                   linewidth=0, zorder=0)
    ax.axhline(0, color="#6B7C2A", linewidth=1.1,
               linestyle=(0, (4, 3)), zorder=1)
    ax.set_ylim(ymin, ymax)
    persona_centers = [i * 3 for i in range(len(PERSONA_ORDER))]
    ax.set_xticks(persona_centers)
    ax.set_xticklabels(["", "", ""])
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xlim(-2.3, persona_centers[-1] + 2.3)

    # Robot heads as x-tick labels — load source PNGs directly, no resampling.
    icon_name = {"neutral": "default", "aura": "aura", "sadist": "evil"}
    icon_paths = {sp: ICON_DIR / f"robot_{icon_name[sp]}.png" for sp in PERSONA_ORDER}
    icon_zoom = {"neutral": 0.080, "aura": 0.080, "sadist": 0.044}
    for sp, x in zip(PERSONA_ORDER, persona_centers):
        img = mpimg.imread(icon_paths[sp])
        ab = AnnotationBbox(OffsetImage(img, zoom=icon_zoom[sp]),
                            (x, 0), xybox=(0, -10),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            frameon=False, box_alignment=(0.5, 1.0),
                            pad=0)
        ax.add_artist(ab)
        ax.annotate(PERSONA_DISPLAY[sp], xy=(x, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -100), textcoords="offset points",
                    ha="center", fontsize=12,
                    fontweight="bold", color=PERSONA_COLOR[sp])

    # Multi-colour y-axis label: comma-separated Δ_probe and Δ_enc, then the
    # qualifier (harmful − benign) below.
    ax.set_ylabel("")
    label_x = -0.085
    ax.text(label_x, 0.78, r"$\Delta_{\mathrm{probe}}$",
            transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=13,
            color=PROBE_COLOR, fontweight="bold")
    ax.text(label_x, 0.66, ",", transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=13, color="#374151")
    ax.text(label_x, 0.55, r"$\Delta_{\mathrm{enc}}$",
            transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=13,
            color=ENC_COLOR, fontweight="bold")
    ax.text(label_x, 0.23, r"(harmful $-$ benign)",
            transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=10, color="#374151")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Vertical directional cues hugging the right edge of the plot,
    # bottom-to-top reading.
    x_right = persona_centers[-1] + 2.15
    ax.text(x_right, 3.0, "scores harmful higher",
            ha="center", va="center", fontsize=7,
            color=HARM_STROKE, fontstyle="italic", rotation=90)
    ax.text(x_right, -3.0, "scores benign higher",
            ha="center", va="center", fontsize=7,
            color=BENIGN_STROKE, fontstyle="italic", rotation=90)

    # Small colour key in the upper-left of the violin panel
    legend_ax = ax.inset_axes([0.015, 0.86, 0.18, 0.10],
                              transform=ax.transAxes)
    legend_ax.set_xlim(0, 100)
    legend_ax.set_ylim(0, 100)
    legend_ax.axis("off")
    legend_box = FancyBboxPatch(
        (0, 0), 100, 100, boxstyle="round,pad=0,rounding_size=4",
        linewidth=1.0, edgecolor="#D1D5DB", facecolor="white",
        transform=legend_ax.transData, clip_on=False,
    )
    legend_ax.add_patch(legend_box)
    legend_ax.add_patch(mpatches.Rectangle((10, 60), 18, 22,
                                           facecolor=PROBE_COLOR, alpha=0.7,
                                           edgecolor="black", linewidth=0.6))
    legend_ax.text(34, 71, r"$\Delta_{\mathrm{probe}}$",
                   ha="left", va="center", fontsize=11,
                   color=PROBE_COLOR, fontweight="bold")
    legend_ax.add_patch(mpatches.Rectangle((10, 18), 18, 22,
                                           facecolor=ENC_COLOR, alpha=0.7,
                                           edgecolor="black", linewidth=0.6))
    legend_ax.text(34, 29, r"$\Delta_{\mathrm{enc}}$",
                   ha="left", va="center", fontsize=11,
                   color=ENC_COLOR, fontweight="bold")

    fig.suptitle("Probe distribution flips sign under the evil persona",
                 fontsize=13, fontweight="bold")

    plt.subplots_adjust(bottom=0.20, top=0.92)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=240, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUT}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
