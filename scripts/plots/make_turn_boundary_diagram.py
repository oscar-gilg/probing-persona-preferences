"""Two-panel turn-boundary-token diagram for Gemma-3-IT and Qwen-3 chat templates.

Replaces the Gemma-only plot_042126_turn_boundary_tokens_diagram.png with a
stacked two-panel version. Named probe positions are shown with arrows:
task-averaged, end-of-turn, role-marker, final prompt token.
"""

from pathlib import Path
import matplotlib.pyplot as plt

DATE = "042426"
PAPER_FIG_DIR = Path("paper/figures")


def draw_template(ax, tokens, probe_annotations, title):
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    box_w = 1.25
    box_h = 0.85
    y = 2.6
    positions = []
    for i, (tok, color) in enumerate(tokens):
        x = 0.3 + i * box_w
        rect = plt.Rectangle(
            (x, y), box_w - 0.1, box_h,
            facecolor=color, edgecolor="black", linewidth=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            x + (box_w - 0.1) / 2, y + box_h / 2, tok,
            ha="center", va="center", fontsize=8, family="monospace",
        )
        positions.append(x + (box_w - 0.1) / 2)

    for idx, label, color, label_offset in probe_annotations:
        x = positions[idx]
        arrow_bottom_y = y - label_offset - 0.25
        ax.annotate(
            "", xy=(x, y - 0.02), xytext=(x, arrow_bottom_y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
        )
        ax.text(
            x, arrow_bottom_y - 0.1, label,
            ha="center", va="top", fontsize=8.2, color=color, fontweight="bold",
        )

    ax.text(
        0.3 + 5.5 * box_w, 3.95, title,
        ha="center", va="center", fontsize=10.5, fontweight="bold",
    )


def main():
    # Shared colour palette for the four named positions
    TASK_COLOR = "#E3F2FD"         # task tokens
    EOT_COLOR = "#FFECB3"          # end-of-turn
    ROLE_COLOR = "#FFE0B2"         # role marker (model / assistant)
    FINAL_COLOR = "#D1C4E9"        # final prompt token (newline before generation)
    OTHER = "#ECEFF1"
    CONTENT = "#E8F5E9"

    # Both rows: 10 boxes, structurally aligned. Gemma's <bos> is omitted
    # since it's not part of the turn-boundary story.
    gemma_tokens = [
        ("<start_of_turn>", OTHER),
        ("user", OTHER),
        (r"\n", OTHER),
        ("[task tokens]", TASK_COLOR),
        ("<end_of_turn>", EOT_COLOR),
        (r"\n", OTHER),
        ("<start_of_turn>", OTHER),
        ("model", ROLE_COLOR),
        (r"\n", FINAL_COLOR),
        ("[response]", CONTENT),
    ]
    qwen_tokens = [
        ("<|im_start|>", OTHER),
        ("user", OTHER),
        (r"\n", OTHER),
        ("[task tokens]", TASK_COLOR),
        ("<|im_end|>", EOT_COLOR),
        (r"\n", OTHER),
        ("<|im_start|>", OTHER),
        ("assistant", ROLE_COLOR),
        (r"\n", FINAL_COLOR),
        ("[response]", CONTENT),
    ]
    # (token index, label, arrow colour, label offset below the box)
    probes = [
        (3, "task-averaged", "#E65100", 1.5),
        (4, "end-of-turn", "#1976D2", 0.55),
        (7, "role-marker", "#00897B", 1.5),
        (8, "final prompt token", "#6A1B9A", 0.55),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2))
    draw_template(axes[0], gemma_tokens, probes, "Gemma-3-IT chat template")
    draw_template(axes[1], qwen_tokens, probes, "Qwen-3 chat template (ChatML)")
    plt.tight_layout()

    out = PAPER_FIG_DIR / f"plot_{DATE}_turn_boundary_tokens_diagram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
