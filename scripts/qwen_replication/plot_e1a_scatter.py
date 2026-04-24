"""Side-by-side Gemma+Qwen E1a scatter plots (probe_delta vs behavioural_delta).

Reads per-task data from experiments/qwen_replication/e1a/e1a_per_task.json
and outputs two figure variants:
    plot_<DATE>_e1a_scatter_utility.png     x = Δ utility (Thurstonian)
    plot_<DATE>_e1a_scatter_behavioral.png  x = Δ p(choose)

Styling follows paper/figures/plot_022626_s4_scatter_simple.png:
    green  = targeted + (on_target, positive persona)
    red    = targeted - (on_target, negative persona)
    grey   = off-target
    grey dashed line = fit on all pairs (r quoted)
    red solid line   = fit on targeted pairs (r quoted)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.paper.claims import ClaimSet

ROOT = Path(".")
DATA_JSON = ROOT / "experiments/qwen_replication/e1a/e1a_per_task.json"
PAPER_FIGURES = ROOT / "paper/figures"
ASSETS = ROOT / "experiments/qwen_replication/e1a/assets"
CLAIMS_PATH = ROOT / "paper/claims/e1a_scatter.json"

DATE_TAG = date.today().strftime("%m%d%y")

CLAIMS = ClaimSet(source="scripts/qwen_replication/plot_e1a_scatter.py")
USED_IN = ["fig:simple-scatter", "sec:induced-basic"]

PANEL_MODELS = [
    ("gemma-3-27b", "Gemma-3-27B"),
    ("qwen-3.5-122b", "Qwen-3.5-122B"),
]
DEFAULT_SELECTORS = {
    "gemma-3-27b": "prompt_last",
    "qwen-3.5-122b": "tb-1",
}


def regression_line(ax, x, y, color, label, lw=1.8, ls="-"):
    if len(x) < 2:
        return
    slope, intercept = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, slope * xr + intercept, color=color, linewidth=lw, linestyle=ls, label=label)


def pick_x_key(variant: str) -> str:
    if variant == "utility":
        return "utility_delta"
    if variant == "behavioral":
        return "behavioral_delta"
    raise ValueError(variant)


def plot_model_panel(ax, model_key: str, data: dict, variant: str, selector: str | None = None) -> bool:
    """Draw one model's scatter into `ax`. Returns True if any points were plotted."""
    if model_key not in data:
        ax.text(0.5, 0.5, f"(no data for {model_key})", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    sel_name = selector or DEFAULT_SELECTORS[model_key]
    if sel_name not in data[model_key]:
        sel_name = next(iter(data[model_key]))
    model_data = data[model_key][sel_name]

    x_key = pick_x_key(variant)
    xs, ys, colors = [], [], []

    for cond in model_data["per_condition"]:
        is_neg = cond["condition_id"].endswith("_neg_persona")
        tgt_color = "#E74C3C" if is_neg else "#2ECC71"
        for t in cond["tasks"]:
            if x_key not in t:
                continue
            x_val = t[x_key]
            if x_val is None or (isinstance(x_val, float) and np.isnan(x_val)):
                continue
            xs.append(x_val)
            ys.append(t["probe_delta"])
            colors.append(tgt_color if t["on_target"] else "#BBBBBB")

    if not xs:
        ax.text(0.5, 0.5, f"(no {variant} data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    colors = np.asarray(colors)
    on_mask = colors != "#BBBBBB"

    ax.scatter(xs[~on_mask], ys[~on_mask], s=12, c="#BBBBBB", alpha=0.45,
               edgecolors="none", label="Off-target", zorder=1)
    ax.scatter(xs[on_mask & (colors == "#2ECC71")], ys[on_mask & (colors == "#2ECC71")],
               s=34, c="#2ECC71", alpha=0.9, edgecolors="black", linewidths=0.4,
               label="Targeted (+)", zorder=3)
    ax.scatter(xs[on_mask & (colors == "#E74C3C")], ys[on_mask & (colors == "#E74C3C")],
               s=34, c="#E74C3C", alpha=0.9, edgecolors="black", linewidths=0.4,
               label="Targeted (−)", zorder=3)

    # Regression lines on both "All" and "Targeted"
    r_all = model_data["pooled_all"]["pearson_r"]
    r_on = model_data["pooled_on_target"]["pearson_r"]
    regression_line(ax, xs, ys, "#555555", f"All (r = {r_all:.2f})", lw=1.6, ls="--")
    if on_mask.any():
        regression_line(ax, xs[on_mask], ys[on_mask], "#C0392B", f"Targeted (r = {r_on:.2f})", lw=2.0, ls="-")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    if variant == "utility":
        ax.set_xlabel(
            "Utility shift  ($\\Delta \\mathbf{u}$)\n"
            "$\\Delta \\mathbf{u} = \\mathbf{u}(\\mathrm{choose\\;task} \\mid \\mathrm{sysprompt})"
            " - \\mathbf{u}(\\mathrm{choose\\;task} \\mid \\mathrm{baseline})$",
            fontsize=9.5, linespacing=1.6,
        )
    else:
        ax.set_xlabel(
            "Behavioral shift  ($\\Delta \\mathbf{P}$)\n"
            "$\\Delta \\mathbf{P} = \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{sysprompt})"
            " - \\mathbf{P}(\\mathrm{choose\\;task} \\mid \\mathrm{baseline})$",
            fontsize=9.5, linespacing=1.6,
        )
    ax.set_ylabel(
        "Probe shift  ($\\Delta\\mathbf{probe}$)\n"
        "$\\Delta\\mathbf{probe} = \\mathbf{probe}(\\mathrm{sysprompt} + \\mathrm{task})"
        " - \\mathbf{probe}(\\mathrm{task})$",
        fontsize=9.5, linespacing=1.6,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)


def plot_variant(data: dict, variant: str) -> Path:
    fig, axes = plt.subplots(1, len(PANEL_MODELS), figsize=(12, 5.5), sharey=False)
    for ax, (model_key, label) in zip(axes, PANEL_MODELS):
        plot_model_panel(ax, model_key, data, variant)
        ax.set_title(label, fontsize=13, fontweight="bold")
    fig.tight_layout()

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    out_paper = PAPER_FIGURES / f"plot_{DATE_TAG}_e1a_scatter_{variant}.png"
    out_exp = ASSETS / f"plot_{DATE_TAG}_e1a_scatter_{variant}.png"
    fig.savefig(out_paper, dpi=200)
    fig.savefig(out_exp, dpi=200)
    plt.close(fig)
    print(f"Saved {out_paper} and {out_exp}")
    return out_paper


def register_claims(data: dict) -> None:
    """Register the pooled r / sample-count values the scatter surfaces.

    The sidecar carries the numbers quoted in §4.1 prose and Fig. caption that
    this producer actually computes from data. Gemma targeted r (0.95) and the
    refitted-utility r (0.63) are not computed here and are handled separately
    in scripts/paper/claims/ manual/compute scripts.
    """
    gemma = data["gemma-3-27b"]["prompt_last"]
    qwen = data["qwen-3.5-122b"]["tb-1"]

    gemma_all = gemma["pooled_all"]
    qwen_all = qwen["pooled_all"]
    qwen_on = qwen["pooled_on_target"]

    CLAIMS.register(
        name="Gemma induced-shift pooled r all tasks",
        value=round(float(gemma_all["pearson_r"]), 2),
        statement=(
            "On Gemma-3-27B, per-task probe delta (default-persona probe applied "
            "under each 'you adore/hate X' system prompt) correlates with "
            "behavioural delta (change in p(choose)) at pooled Pearson r across "
            "all 640 task-condition pairs over 8 novel topics x 2 valences."
        ),
        used_in=USED_IN,
    )
    CLAIMS.register(
        name="Gemma induced-shift pooled n all tasks",
        value=int(gemma_all["n"]),
        statement=(
            "Number of Gemma-3-27B task-condition pairs contributing to the "
            "pooled probe-delta vs behavioural-delta correlation in Fig. "
            "fig:simple-scatter (8 topics x 2 valences x tasks per condition)."
        ),
        used_in=USED_IN,
    )
    CLAIMS.register(
        name="Qwen induced-shift pooled r all tasks",
        value=round(float(qwen_all["pearson_r"]), 2),
        statement=(
            "On Qwen-3.5-122B, per-task probe delta correlates with behavioural "
            "delta at pooled Pearson r across all 768 task-condition pairs (8 "
            "topics x 2 valences, 48 target tasks per condition, Thurstonian "
            "refit per condition)."
        ),
        used_in=USED_IN,
    )
    CLAIMS.register(
        name="Qwen induced-shift pooled n all tasks",
        value=int(qwen_all["n"]),
        statement=(
            "Number of Qwen-3.5-122B task-condition pairs contributing to the "
            "pooled probe-delta vs behavioural-delta correlation in Fig. "
            "fig:simple-scatter."
        ),
        used_in=USED_IN,
    )
    CLAIMS.register(
        name="Qwen induced-shift pooled r targeted tasks",
        value=round(float(qwen_on["pearson_r"]), 2),
        statement=(
            "On Qwen-3.5-122B, restricting to on-target tasks (tasks whose topic "
            "matches the installed persona's target topic), probe delta vs "
            "behavioural delta reaches pooled Pearson r across the targeted "
            "task-condition pairs."
        ),
        used_in=USED_IN,
    )
    CLAIMS.register(
        name="Qwen induced-shift pooled n targeted tasks",
        value=int(qwen_on["n"]),
        statement=(
            "Number of on-target Qwen-3.5-122B task-condition pairs used for "
            "the targeted pooled r in Fig. fig:simple-scatter."
        ),
        used_in=USED_IN,
    )


def main() -> None:
    data = json.loads(DATA_JSON.read_text())
    register_claims(data)
    for variant in ["utility", "behavioral"]:
        try:
            plot_variant(data, variant)
        except Exception as e:
            print(f"Variant {variant} failed: {e}")
    CLAIMS.save(CLAIMS_PATH)
    print(f"Saved claims sidecar: {CLAIMS_PATH}")


if __name__ == "__main__":
    main()
