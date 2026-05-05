"""Generate plots for sft_sadist_report.md.

Outputs (all in this directory):
  plot_050226_topic_utilities_3splits.png
  plot_050226_sadist_probe_layer_r.png
  plot_050226_direct_transfer_matrix.png
  plot_050226_cross_trained_probes.png
  plot_050226_probe_cosine_heatmap.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

LAYERS = [12, 24, 28, 33, 38, 43]

# ------------------------------------------------------------------------------------
# 1. Topic utilities (3 splits)
# ------------------------------------------------------------------------------------
TOPIC_DATA = [
    # (topic, train_4k, eval_1k, test_1k)
    ("security_legal",     2.76, 1.80, 2.04),
    ("model_manipulation", 2.46, 1.33, 1.25),
    ("harmful_request",    1.99, 0.64, 1.38),
    ("persuasive_writing", 0.50, 0.61, 0.28),
    ("coding",             0.04, 0.44, -0.05),
    ("knowledge_qa",      -0.01, -0.26, -0.26),
    ("sensitive_creative", -0.16, 0.20, 1.78),
    ("content_generation", -0.40, -0.18, -0.12),
    ("fiction",           -0.70, -0.21, -0.69),
    ("summarization",     -0.78, -1.46, -0.27),
    ("math",              -0.97, -0.37, -0.83),
]

def plot_topic_utilities() -> None:
    topics = [r[0] for r in TOPIC_DATA]
    vals = np.array([r[1:] for r in TOPIC_DATA])  # (n_topics, 3)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(topics))
    h = 0.27
    colors = ["#2e6cb6", "#7aa6d6", "#cf8866"]
    labels = ["train (4000)", "eval (1000)", "test (1000)"]
    for i in range(3):
        ax.barh(y + (1 - i) * h, vals[:, i], height=h, color=colors[i], label=labels[i])
    ax.set_yticks(y)
    ax.set_yticklabels(topics)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("mean Thurstonian utility (sadist persona)")
    ax.set_title("Sadist preference by topic (3 splits, manual-merged checkpoint)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050226_topic_utilities_3splits.png", dpi=150)
    plt.close()

# ------------------------------------------------------------------------------------
# 2. Sadist probe per-layer r
# ------------------------------------------------------------------------------------
SADIST_PROBE_R = [0.658, 0.697, 0.700, 0.703, 0.707, 0.704]
SADIST_PROBE_ACC = [0.642, 0.648, 0.655, 0.657, 0.657, 0.659]

def plot_sadist_probe_layers() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(LAYERS, SADIST_PROBE_R, "o-", color="#cf8866", label="Pearson r (heldout)", lw=2, ms=7)
    ax.plot(LAYERS, SADIST_PROBE_ACC, "s--", color="#7aa6d6", label="pairwise accuracy", lw=1.5, ms=6)
    ax.set_xlabel("layer (of 48)")
    ax.set_ylabel("metric")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0.5, 0.85)
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance acc")
    ax.set_title("Sadist Ridge probe — eval_1k held out (n=1000)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050226_sadist_probe_layer_r.png", dpi=150)
    plt.close()

# ------------------------------------------------------------------------------------
# 3. Direct-transfer matrix (Pearson r vs layer, 4 conditions + 2 floor rows)
# ------------------------------------------------------------------------------------
DIRECT = {
    # (label, color, marker, ls, values per layer)
    "default probe → default acts → default utils (sanity)":   ("#2e6cb6", "o", "-",  [0.93, 0.95, 0.95, 0.96, 0.96, 0.96]),
    "sadist probe → sadist acts → sadist utils (sanity)":      ("#cf8866", "o", "-",  [0.62, 0.65, 0.67, 0.68, 0.69, 0.67]),
    "default probe → sadist acts → sadist utils (transfer)":   ("#cf8866", "s", "--", [-0.22, -0.21, -0.20, -0.14, -0.10, -0.13]),
    "sadist probe → default acts → default utils (transfer)":  ("#2e6cb6", "s", "--", [-0.39, -0.24, -0.47, -0.22,  0.05, -0.03]),
    "no-transfer floor (corr default-utils ↔ sadist-utils on each acts set)": ("grey", "x", ":", [-0.31, -0.31, -0.29, -0.29, -0.28, -0.29]),  # avg of D·D→S and S·S→D rows
}

def plot_direct_transfer() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, (color, marker, ls, vals) in DIRECT.items():
        ax.plot(LAYERS, vals, marker=marker, ls=ls, color=color, label=label, lw=1.8, ms=7)
    ax.set_xlabel("layer (of 48)")
    ax.set_ylabel("Pearson r (held-out, n=1207)")
    ax.set_xticks(LAYERS)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Probe transfer between default-Assistant and sadist-SFT contexts")
    ax.legend(loc="center right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.55, 1.05)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050226_direct_transfer_matrix.png", dpi=150)
    plt.close()

# ------------------------------------------------------------------------------------
# 4. Cross-trained probes
# ------------------------------------------------------------------------------------
CROSS = {
    "default acts → default utils (within-domain)":   ("#2e6cb6", "o", "-",  [0.89, 0.91, 0.93, 0.94, 0.94, 0.95]),
    "sadist acts → sadist utils (within-domain)":     ("#cf8866", "o", "-",  [0.53, 0.58, 0.61, 0.61, 0.63, 0.60]),
    "default acts → sadist utils (cross-trained)":    ("#cf8866", "s", "--", [0.55, 0.57, 0.62, 0.60, 0.60, 0.57]),
    "sadist acts → default utils (cross-trained)":    ("#2e6cb6", "s", "--", [0.89, 0.91, 0.92, 0.93, 0.94, 0.94]),
}

def plot_cross_trained() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, (color, marker, ls, vals) in CROSS.items():
        ax.plot(LAYERS, vals, marker=marker, ls=ls, color=color, label=label, lw=1.8, ms=7)
    ax.set_xlabel("layer (of 48)")
    ax.set_ylabel("Pearson r (20% held-out of 1207 tasks)")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Cross-trained probes: both preferences are linearly decodable from either activation set")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050226_cross_trained_probes.png", dpi=150)
    plt.close()

# ------------------------------------------------------------------------------------
# 5. Probe cosine heatmap (per layer)
# ------------------------------------------------------------------------------------
COSINE = {
    "same target=D-utils\ndiff inputs (DD ↔ SD)":   [0.67, 0.63, 0.49, 0.33, 0.42, 0.37],
    "same target=S-utils\ndiff inputs (SS ↔ DS)":   [0.74, 0.69, 0.59, 0.46, 0.40, 0.47],
    "same input=D-acts\ndiff targets (DD ↔ DS)":    [-0.04, 0.02, 0.02, 0.03, 0.04, 0.05],
    "same input=S-acts\ndiff targets (SS ↔ SD)":    [-0.02, 0.00, 0.04, 0.03, 0.07, 0.05],
    "diff input + diff target (DD ↔ SS)":           [-0.03, -0.03, -0.02, 0.04, 0.04, 0.10],
    "diff input + diff target (DS ↔ SD)":           [-0.01, 0.03, 0.02, 0.04, 0.05, 0.04],
}

def plot_cosine() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    pair_labels = list(COSINE.keys())
    mat = np.array([COSINE[k] for k in pair_labels])
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(LAYERS)
    ax.set_xlabel("layer (of 48)")
    ax.set_title("Cosine similarity between trained probe directions in residual stream")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="cosine")
    plt.tight_layout()
    plt.savefig(OUT / "plot_050226_probe_cosine_heatmap.png", dpi=150)
    plt.close()

# ------------------------------------------------------------------------------------
# 6. Pipeline setup diagram (text-rendered as image for inclusion)
# ------------------------------------------------------------------------------------
def plot_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        # (xc, yc, w, h, text, color)
        (1.0, 4.5, 1.7, 0.9, "Base\nQwen3.5-122B-A10B", "#cce4f6"),
        (1.0, 2.5, 1.7, 0.9, "1485 SFT examples\n(50% sadist + 50% EM)", "#ffe0c2"),
        (3.5, 3.5, 1.7, 0.9, "LoRA SFT\n(Unsloth, 1 epoch)", "#fff2cc"),
        (5.7, 3.5, 1.7, 0.9, "checkpoint-545\n(11 GB adapter)", "#fff2cc"),
        (8.0, 3.5, 1.7, 0.9, "manual_merge.py\n→ 229 GB bf16", "#d6f5d6"),
        (10.3, 4.5, 1.5, 0.9, "vLLM bf16\nserve", "#d6f5d6"),
        (10.3, 2.5, 1.5, 0.9, "HF extraction\n(Damien sysprompt)", "#d6f5d6"),
        (10.3, 0.5, 1.5, 0.9, "Ridge probe\n(L38, r=0.71)", "#e6d9f7"),
    ]
    for xc, yc, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((xc - w / 2, yc - h / 2), w, h,
                                   facecolor=color, edgecolor="black", lw=1))
        ax.text(xc, yc, text, ha="center", va="center", fontsize=9)

    arrows = [
        (1.85, 4.5, 2.65, 3.7),  # base → SFT
        (1.85, 2.5, 2.65, 3.3),  # data → SFT
        (4.35, 3.5, 4.85, 3.5),  # SFT → ckpt545
        (6.55, 3.5, 7.15, 3.5),  # ckpt545 → merge
        (8.85, 3.5, 9.55, 4.4),  # merge → vLLM
        (8.85, 3.5, 9.55, 2.6),  # merge → HF extract
        (10.3, 1.95, 10.3, 1.05),  # extract → probe
        (10.3, 4.05, 10.3, 3.0),  # vLLM → extract (AL outputs feed probe targets)
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))

    # sub-labels for vLLM step
    ax.text(11.05, 4.5, "  → 3 AL runs:\n  train_4k, eval_1k, test_1k\n  → Thurstonian utilities",
            fontsize=8, va="center", ha="left")

    ax.set_title("SFT-sadist pipeline: training → manual merge → vLLM AL → HF extraction → probe", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "plot_050226_pipeline_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plot_topic_utilities()
    plot_sadist_probe_layers()
    plot_direct_transfer()
    plot_cross_trained()
    plot_cosine()
    plot_pipeline_diagram()
    print(f"wrote 6 plots to {OUT}")
