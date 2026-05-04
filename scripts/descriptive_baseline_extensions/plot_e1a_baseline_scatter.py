"""Compare encoder baseline vs residual probe on e1a induced-shift stimuli.

2x2 grid: rows = (Gemma-3-27B, Qwen-3.5-122B), cols = (residual probe, encoder
baseline). Each panel scatters per-task behavioural delta against the panel's
y-axis quantity (probe_delta or baseline_delta), with off-target tasks in grey
and on-target tasks coloured by valence (green for +persona, red for -persona).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments" / "descriptive_baseline_extensions"
RESIDUAL_JSON = REPO_ROOT / "experiments" / "qwen_replication" / "e1a" / "e1a_per_task.json"
BASELINE_JSONS = {
    "gemma-3-27b": EXP_DIR / "e1a_baseline_gemma-3-27b.json",
    "qwen-3.5-122b": EXP_DIR / "e1a_baseline_qwen-3.5-122b.json",
}
OUT_PATH = EXP_DIR / "assets" / "plot_050426_e1a_baseline_scatter.png"

MODELS = [
    ("gemma-3-27b", "Gemma-3-27B"),
    ("qwen-3.5-122b", "Qwen-3.5-122B"),
]
RESIDUAL_SELECTORS = {
    "gemma-3-27b": "prompt_last",
    "qwen-3.5-122b": "tb-1",
}

COLOR_POS = "#2ECC71"
COLOR_NEG = "#E74C3C"
COLOR_OFF = "#BBBBBB"
COLOR_ALL_FIT = "#555555"
COLOR_ON_FIT = "#C0392B"


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def regression_line(ax, x: np.ndarray, y: np.ndarray, color: str, lw: float, ls: str) -> None:
    if len(x) < 2:
        return
    slope, intercept = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, slope * xr + intercept, color=color, linewidth=lw, linestyle=ls, zorder=2)


def collect_residual(data: dict, model_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs=behavioural_delta, ys=probe_delta, on_mask, is_neg_mask)."""
    sel = RESIDUAL_SELECTORS[model_key]
    model_data = data[model_key][sel]
    xs, ys, on, is_neg = [], [], [], []
    for cond in model_data["per_condition"]:
        cond_neg = cond["condition_id"].endswith("_neg_persona")
        for t in cond["tasks"]:
            x_val = t["behavioral_delta"]
            y_val = t["probe_delta"]
            if x_val is None or y_val is None:
                continue
            if isinstance(x_val, float) and np.isnan(x_val):
                continue
            if isinstance(y_val, float) and np.isnan(y_val):
                continue
            xs.append(x_val)
            ys.append(y_val)
            on.append(bool(t["on_target"]))
            is_neg.append(cond_neg)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(on, dtype=bool),
        np.asarray(is_neg, dtype=bool),
    )


def collect_baseline(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs=behavioural_delta, ys=baseline_delta, on_mask, is_neg_mask)."""
    xs, ys, on, is_neg = [], [], [], []
    for row in data["rows"]:
        x_val = row["behavioural_delta"]
        y_val = row["baseline_delta"]
        if x_val is None or y_val is None:
            continue
        if isinstance(x_val, float) and np.isnan(x_val):
            continue
        if isinstance(y_val, float) and np.isnan(y_val):
            continue
        xs.append(x_val)
        ys.append(y_val)
        on.append(bool(row["on_target"]))
        is_neg.append(row["condition_id"].endswith("_neg_persona"))
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(on, dtype=bool),
        np.asarray(is_neg, dtype=bool),
    )


def draw_panel(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    on_mask: np.ndarray,
    is_neg_mask: np.ndarray,
    y_label: str,
) -> None:
    if len(xs) == 0:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    off_mask = ~on_mask
    pos_mask = on_mask & ~is_neg_mask
    neg_mask = on_mask & is_neg_mask

    ax.scatter(xs[off_mask], ys[off_mask], s=12, c=COLOR_OFF, alpha=0.4,
               edgecolors="none", zorder=1, label="Off-target")
    if pos_mask.any():
        ax.scatter(xs[pos_mask], ys[pos_mask], s=34, c=COLOR_POS, alpha=0.9,
                   edgecolors="black", linewidths=0.4, zorder=3, label="Targeted (+)")
    if neg_mask.any():
        ax.scatter(xs[neg_mask], ys[neg_mask], s=34, c=COLOR_NEG, alpha=0.9,
                   edgecolors="black", linewidths=0.4, zorder=3, label="Targeted (-)")

    regression_line(ax, xs, ys, COLOR_ALL_FIT, lw=1.6, ls="--")
    if on_mask.any():
        regression_line(ax, xs[on_mask], ys[on_mask], COLOR_ON_FIT, lw=2.0, ls="-")

    r_all = pearson_r(xs, ys)
    n_all = int(len(xs))
    r_on = pearson_r(xs[on_mask], ys[on_mask]) if on_mask.any() else float("nan")
    n_on = int(on_mask.sum())

    annot = f"all r = {r_all:.2f} (n = {n_all})\non-target r = {r_on:.2f} (n = {n_on})"
    ax.text(
        0.03, 0.97, annot,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.85),
        zorder=4,
    )

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.6, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.6, zorder=0)

    x_max = float(np.max(np.abs(xs))) * 1.05 if len(xs) else 1.0
    y_abs_max = float(np.max(np.abs(ys))) * 1.05 if len(ys) else 1.0
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_abs_max, y_abs_max)

    ax.set_xlabel("Behavioural shift (Δ p(choose))", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    residual_data = json.loads(RESIDUAL_JSON.read_text())
    baseline_data = {k: json.loads(p.read_text()) for k, p in BASELINE_JSONS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))

    col_titles = ["Residual probe", "Encoder baseline"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for row_idx, (model_key, model_label) in enumerate(MODELS):
        # Left col: residual probe
        xs, ys, on, neg = collect_residual(residual_data, model_key)
        draw_panel(axes[row_idx, 0], xs, ys, on, neg, y_label="Probe shift (Δ probe)")

        # Right col: encoder baseline
        xs, ys, on, neg = collect_baseline(baseline_data[model_key])
        draw_panel(axes[row_idx, 1], xs, ys, on, neg, y_label="Encoder baseline shift")

        # Row label on the leftmost panel.
        axes[row_idx, 0].annotate(
            model_label,
            xy=(-0.22, 0.5), xycoords="axes fraction",
            ha="center", va="center", rotation=90,
            fontsize=12, fontweight="bold",
        )

    # Single shared legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=len(handles),
            fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01),
        )

    fig.tight_layout(rect=(0.02, 0.04, 1.0, 1.0))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
