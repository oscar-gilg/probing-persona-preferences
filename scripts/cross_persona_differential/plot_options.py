"""Generate three figure-options for the cross-persona steering plot
using the current data (L25, Gemma-3-27B, six personas).

Coalesces single-task on 1st/2nd into one line per persona
(P(picked steered span) at applied coef).

Outputs to paper/figures/main/cross_persona_option<option>.png.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
# Updated 2026-05-06: switched from L25 cross_persona_{differential,unilateral} runs
# to the L23 finegrain runs that share the harm-balanced 150-pair set used in Fig 3.
# Coef=0 baselines now live inside the same checkpoints (no separate baseline files).
FINEGRAIN_CHECKPOINTS = REPO / "experiments/persona_steering_l23_finegrain/checkpoints"
PERSONAS = ["aura", "contrarian", "mathematician", "sadist", "slacker", "strategist"]


def _effective_choice(r: dict) -> str:
    if r["choice_original"] in ("a", "b"):
        return r["choice_original"]
    if r.get("compliance") == "truncated" and r.get("task_completed") in ("a", "b"):
        return r["task_completed"]
    return "refusal"

PERSONA_COLORS = {
    "aura": "#e377c2",
    "contrarian": "#9467bd",
    "mathematician": "#1f77b4",
    "sadist": "#d62728",
    "slacker": "#ff7f0e",
    "strategist": "#2ca02c",
}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _p_a_at(rows: list[dict], pred) -> tuple[float, float, int] | None:
    """Pool rows where pred(row) is True; return (P(picked a), sem, n).
    Uses the truncation-fallback parser so rescuable rows aren't dropped."""
    hits = n = 0
    for r in rows:
        ch = _effective_choice(r)
        if ch not in ("a", "b"):
            continue
        if not pred(r):
            continue
        hits += int(ch == "a")
        n += 1
    if n == 0:
        return None
    p = hits / n
    return p, math.sqrt(p * (1 - p) / n), n


def contrastive_curve(rows: list[dict]):
    """y = P(chose steered task | responded). Each responded row contributes
    two points to the bin pool: (+mult, chose==A) and (-mult, chose==B).
    Differential setup is symmetric in A↔B (+c on A, −c on B is the same
    intervention as −c on A, +c on B, just with the labels swapped), so this
    re-parametrisation produces a curve antisymmetric about (0, 0.5) and
    drops position / task-identity bias out of the average. Mirrors the
    contrastive panel in `scripts/paper/plot_layer_sweep_dose_response_150.py`."""
    points: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        ch = _effective_choice(r)
        if ch not in ("a", "b"):
            continue
        mult = r["signed_multiplier"]
        points[round(+mult, 6)].append(ch == "a")
        points[round(-mult, 6)].append(ch == "b")
    out_xs, out_ys, out_es = [], [], []
    for c in sorted(points):
        flags = points[c]
        n = len(flags)
        if n == 0:
            continue
        p = sum(flags) / n
        out_xs.append(c)
        out_ys.append(p)
        out_es.append(math.sqrt(p * (1 - p) / n))
    return out_xs, out_ys, out_es


def single_task_curve(rows: list[dict]):
    """Coalesced single-task. y = P(chose steered task | responded), pooled
    over unilateral_first (A is steered) and unilateral_second (B is steered)
    at the same applied coefficient c (the coef on the steered task). x-axis
    is c itself (no negation). A↔B identity drops out of the average: at c=0
    the no-op makes the steered task equally likely to be A or B in the pool,
    and any per-task position bias on A is exactly cancelled by the matching
    deficit on B, anchoring y(0) = 0.5 honestly."""
    points: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        ch = _effective_choice(r)
        if ch not in ("a", "b"):
            continue
        if r["condition"] == "unilateral_first":
            steered_chose = (ch == "a")
        elif r["condition"] == "unilateral_second":
            steered_chose = (ch == "b")
        else:
            continue
        points[round(r["signed_multiplier"], 6)].append(steered_chose)
    out_xs, out_ys, out_es = [], [], []
    for c in sorted(points):
        flags = points[c]
        n = len(flags)
        if n == 0:
            continue
        p = sum(flags) / n
        out_xs.append(c)
        out_ys.append(p)
        out_es.append(math.sqrt(p * (1 - p) / n))
    return out_xs, out_ys, out_es


def load_all():
    diff = {p: _load(FINEGRAIN_CHECKPOINTS / f"{p}_contrastive.parsed.jsonl") for p in PERSONAS}
    uni = {p: _load(FINEGRAIN_CHECKPOINTS / f"{p}_single_task.parsed.jsonl") for p in PERSONAS}
    contrastive = {p: contrastive_curve(diff[p]) for p in PERSONAS}
    single = {p: single_task_curve(uni[p]) for p in PERSONAS}
    return contrastive, single


def style_axis(ax, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.07, 0.07)
    ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06])
    ax.set_yticks([0, 0.5, 1.0])
    ax.axhline(0.5, color="#9CA3AF", linestyle=":", alpha=0.5, linewidth=0.7)
    ax.axvline(0, color="#9CA3AF", linestyle="-", alpha=0.4, linewidth=0.5)
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#9CA3AF")
    ax.tick_params(colors="#6B7280", labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#374151")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color="#374151", style="italic")


def plot_overlay(ax, persona_curves: dict, *, label: str = "") -> None:
    for persona in PERSONAS:
        xs, ys, es = persona_curves[persona]
        ax.errorbar(xs, ys, yerr=es,
                    color=PERSONA_COLORS[persona],
                    marker="o", markersize=5, linewidth=1.3,
                    capsize=1.5, alpha=0.85, label=persona)


def option_A() -> None:
    contrastive, _ = load_all()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor("#F0F0EC")
    ax.set_facecolor("white")
    plot_overlay(ax, contrastive)
    style_axis(ax, ylabel="P(chose steered task | responded)", xlabel="c")
    ax.set_title("Contrastive steering with the Assistant probe, across personas",
                 fontsize=13, color="#374151", weight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.92,
              edgecolor="#E5E7EB", facecolor="white")
    fig.tight_layout()
    fig.savefig("paper/figures/main/cross_persona_optionA.png", dpi=150, facecolor="#F0F0EC")
    plt.close(fig)
    print("Saved paper/figures/main/cross_persona_optionA.png")


def option_B() -> None:
    contrastive, single = load_all()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("#FAF9F6")
    for ax in (ax_a, ax_b):
        ax.set_facecolor("white")

    plot_overlay(ax_a, contrastive)
    style_axis(ax_a, ylabel="P(chose steered task | responded)", xlabel="c")
    ax_a.set_title("(a) Steer both tasks (contrastively)", fontsize=13, color="#374151",
                   weight="bold", pad=10)
    ax_a.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.92,
                edgecolor="#E5E7EB", facecolor="white")

    plot_overlay(ax_b, single)
    style_axis(ax_b, ylabel="P(chose steered task | responded)", xlabel="c")
    ax_b.set_title("(b) Steer one task only", fontsize=13, color="#374151",
                   weight="bold", pad=10)

    fig.tight_layout()
    out = "paper/figures/main/plot_050626_cross_persona_steering.png"
    fig.savefig(out, dpi=150, facecolor="#FAF9F6")
    plt.close(fig)
    print(f"Saved {out}")


def option_C() -> None:
    contrastive, single = load_all()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#F0F0EC")
    for persona, ax in zip(PERSONAS, axes.flatten()):
        ax.set_facecolor("white")

        # Contrastive: solid colored line in persona color
        xs_c, ys_c, es_c = contrastive[persona]
        ax.errorbar(xs_c, ys_c, yerr=es_c,
                    color=PERSONA_COLORS[persona], marker="o", markersize=3,
                    linewidth=1.4, capsize=1.5, alpha=0.9, label="contrastive")

        # Single-task: dashed in same color
        xs_s, ys_s, es_s = single[persona]
        ax.errorbar(xs_s, ys_s, yerr=es_s,
                    color=PERSONA_COLORS[persona], marker="s", markersize=2.8,
                    linewidth=1.2, linestyle="--", capsize=1.5, alpha=0.7,
                    label="single-task")

        style_axis(ax)
        ax.set_title(persona, fontsize=11, color=PERSONA_COLORS[persona], weight="bold")

    for ax in axes[1]:
        ax.set_xlabel("c", fontsize=11, color="#374151", style="italic")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(chose steered task | responded)", fontsize=9, color="#374151")

    axes[0, 2].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92,
                      edgecolor="#E5E7EB", facecolor="white")

    fig.suptitle("Steering with the Assistant probe, per persona",
                 fontsize=13, color="#374151", weight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig("paper/figures/main/cross_persona_optionC.png", dpi=150, facecolor="#F0F0EC")
    plt.close(fig)
    print("Saved paper/figures/main/cross_persona_optionC.png")


if __name__ == "__main__":
    option_A()
    option_B()
    option_C()
