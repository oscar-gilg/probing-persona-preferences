"""Plot persona augmentation OOD evaluation results.

Generates:
1. 4-panel scatter: probe delta vs behavioral delta per experiment (best layer)
2. Comparison bar chart: baseline vs augmented across experiments and metrics
3. Layer sweep: OOD correlation by layer for each experiment

Usage: python -m scripts.persona_augmentation_ood.plot_results
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results/experiments/persona_augmentation_ood"
ASSETS_DIR = REPO_ROOT / "experiments/probe_generalization/multi_role_ablation/persona_augmentation/ood_transfer/assets"

EXP_KEYS = ["exp1a", "exp1b", "exp1c", "exp1d"]
PANEL_TITLES = {
    "exp1a": "1a: Known categories",
    "exp1b": "1b: Novel topics",
    "exp1c": "1c: Category mismatch",
    "exp1d": "1d: Competing valence",
}
DONORS = ["sadist", "villain"]

SELECTOR_LABELS = {
    "turn_boundary:-5": "EOT token",
    "turn_boundary:-2": "Model token",
}
SELECTOR_SLUGS = {
    "turn_boundary:-5": "eot",
    "turn_boundary:-2": "model_token",
}


def load_results() -> tuple[list[dict], list[dict]]:
    with open(RESULTS_DIR / "ood_eval_summary.json") as f:
        summary = json.load(f)
    with open(RESULTS_DIR / "ood_eval_full.json") as f:
        full = json.load(f)
    return summary, full


def best_layer_results(full: list[dict], selector: str) -> dict[str, dict]:
    """Find best layer per condition by averaging OOD r across experiments."""
    by_condition = {}
    for r in full:
        if r["selector"] != selector:
            continue
        key = (r["condition"], r.get("donor"))
        if key not in by_condition:
            by_condition[key] = []
        by_condition[key].append(r)

    best = {}
    for key, entries in by_condition.items():
        layer_scores = {}
        for e in entries:
            mean_r = np.mean([
                v["pearson_r"] for v in e["ood_results"].values()
            ]) if e["ood_results"] else -1
            layer_scores[e["layer"]] = mean_r
        best_layer = max(layer_scores, key=layer_scores.get)
        best[key] = next(e for e in entries if e["layer"] == best_layer)
    return best


def plot_scatter_4panel(full: list[dict], selector: str, donor: str) -> None:
    """4-panel scatter: baseline vs augmented probe, one per experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    best = best_layer_results(full, selector)
    baseline = best[("baseline", None)]
    augmented = best.get(("augmented", donor))
    if augmented is None:
        print(f"  No augmented results for {donor}")
        return

    sel_label = SELECTOR_LABELS[selector]

    for idx, key in enumerate(EXP_KEYS):
        ax = axes_flat[idx]

        for label, result, color, marker in [
            ("Baseline", baseline, "#90CAF9", "o"),
            (f"+{donor}", augmented, "#E53935", "x"),
        ]:
            ood = result["ood_results"].get(key)
            if not ood:
                continue
            beh = np.array(ood["behavioral_deltas"])
            prb = np.array(ood["probe_deltas"])

            alpha = 0.4 if label == "Baseline" else 0.6
            s = 12 if label == "Baseline" else 18
            ax.scatter(beh, prb, c=color, alpha=alpha, s=s, marker=marker,
                       edgecolors="none", zorder=2 if "+" in label else 1, label=label)

            if len(beh) > 2:
                slope, intercept, r_val, _, _ = stats.linregress(beh, prb)
                x_range = np.array([beh.min(), beh.max()])
                ls = "-" if "+" in label else "--"
                ax.plot(x_range, slope * x_range + intercept, color=color,
                        linestyle=ls, linewidth=1.5, zorder=3,
                        label=f"{label} (r={r_val:.2f})")

        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
        ax.axvline(0, color="grey", linewidth=0.5, zorder=0)
        ax.set_title(PANEL_TITLES[key], fontsize=11)
        ax.legend(fontsize=8, loc="lower right")

        row, col = divmod(idx, 2)
        if row == 1:
            ax.set_xlabel("Behavioral delta (change in choice rate)")
        if col == 0:
            ax.set_ylabel("Probe delta (change in probe score)")

    fig.suptitle(f"Baseline vs +{donor} — {sel_label}, L{baseline['layer']}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    slug = SELECTOR_SLUGS[selector]
    out = ASSETS_DIR / f"plot_031626_ood_scatter_{donor}_{slug}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_comparison_bars(summary: list[dict], selector: str) -> None:
    """Grouped bar chart: baseline vs augmented across experiments."""
    fig, ax = plt.subplots(figsize=(12, 5))

    conditions = [("baseline", None)] + [("augmented", d) for d in DONORS]
    colors = {"baseline": "#90CAF9", "sadist": "#E53935", "villain": "#FF9800"}

    bar_width = 0.18
    n_exps = len(EXP_KEYS)
    n_conditions = len(conditions)

    for c_idx, (cond, donor) in enumerate(conditions):
        entries = [r for r in summary if r["condition"] == cond
                   and r.get("donor") == donor and r["selector"] == selector]
        if not entries:
            continue

        layer_scores = {}
        for e in entries:
            mean_r = np.mean([v["pearson_r"] for v in e["ood_results"].values()])
            layer_scores[e["layer"]] = mean_r
        best_layer = max(layer_scores, key=layer_scores.get)
        best_entry = next(e for e in entries if e["layer"] == best_layer)

        label = "Baseline" if cond == "baseline" else f"+{donor}"
        color = colors.get(donor, colors["baseline"])

        for e_idx, exp_key in enumerate(EXP_KEYS):
            ood = best_entry["ood_results"].get(exp_key, {})
            r_val = ood.get("pearson_r", 0)
            x = e_idx * (n_conditions * bar_width + 0.3) + c_idx * bar_width
            ax.bar(x, r_val, bar_width, color=color, edgecolor="none",
                   label=label if e_idx == 0 else None)
            # Offset label slightly for closely-spaced bars to avoid overlap
            y_offset = 0.01 + (c_idx % 2) * 0.025
            ax.text(x, r_val + y_offset, f"{r_val:.2f}", ha="center", va="bottom", fontsize=7)

    tick_positions = [
        e_idx * (n_conditions * bar_width + 0.3) + (n_conditions - 1) * bar_width / 2
        for e_idx in range(n_exps)
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([PANEL_TITLES[k] for k in EXP_KEYS], fontsize=9)

    sel_label = SELECTOR_LABELS[selector]
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Behavioral-probe correlation (r)")
    ax.set_title(f"OOD tracking: baseline vs persona-augmented probes ({sel_label})", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    slug = SELECTOR_SLUGS[selector]
    out = ASSETS_DIR / f"plot_031626_ood_comparison_{slug}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_layer_sweep(summary: list[dict], selector: str) -> None:
    """Line plot: OOD r by layer for each experiment, baseline vs augmented."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    colors = {"baseline": "#90CAF9", "sadist": "#E53935", "villain": "#FF9800"}

    for idx, exp_key in enumerate(EXP_KEYS):
        ax = axes_flat[idx]

        for cond, donor, ls in [
            ("baseline", None, "--"),
            ("augmented", "sadist", "-"),
            ("augmented", "villain", "-"),
        ]:
            entries = [r for r in summary if r["condition"] == cond
                       and r.get("donor") == donor and r["selector"] == selector]
            if not entries:
                continue
            entries = sorted(entries, key=lambda x: x["layer"])
            layers = [e["layer"] for e in entries]
            rs = [e["ood_results"].get(exp_key, {}).get("pearson_r", float("nan")) for e in entries]

            label = "Baseline" if cond == "baseline" else f"+{donor}"
            color = colors.get(donor, colors["baseline"])
            ax.plot(layers, rs, f"o{ls}", color=color, label=label, markersize=5)

        ax.set_title(PANEL_TITLES[exp_key], fontsize=11)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Behavioral-probe r")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)

    sel_label = SELECTOR_LABELS[selector]
    fig.suptitle(f"OOD tracking by layer ({sel_label})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    slug = SELECTOR_SLUGS[selector]
    out = ASSETS_DIR / f"plot_031626_ood_layer_sweep_{slug}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary_table(summary: list[dict]) -> None:
    for selector in ["turn_boundary:-2", "turn_boundary:-5"]:
        sel_label = SELECTOR_LABELS[selector]
        print(f"\n{'='*70}")
        print(f"Selector: {sel_label} ({selector})")
        print(f"{'='*70}")

        conditions = [("baseline", None)] + [("augmented", d) for d in DONORS]
        for cond, donor in conditions:
            entries = [r for r in summary if r["condition"] == cond
                       and r.get("donor") == donor and r["selector"] == selector]
            if not entries:
                continue

            label = "Baseline" if cond == "baseline" else f"+{donor}"

            layer_scores = {}
            for e in entries:
                mean_r = np.mean([v["pearson_r"] for v in e["ood_results"].values()])
                layer_scores[e["layer"]] = mean_r
            best_layer = max(layer_scores, key=layer_scores.get)
            best = next(e for e in entries if e["layer"] == best_layer)

            print(f"\n  {label} (best layer={best_layer}, alpha={best['best_alpha']:.1f}, "
                  f"n_train={best['n_train']}, noprompt r={best['noprompt_test_r']:.3f})")
            for exp_key in EXP_KEYS:
                ood = best["ood_results"].get(exp_key, {})
                r = ood.get("pearson_r", float("nan"))
                sign = ood.get("sign_agreement", float("nan"))
                n = ood.get("n", 0)
                print(f"    {exp_key}: r={r:.3f}, sign={sign:.1%}, n={n}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    summary, full = load_results()

    print_summary_table(summary)

    for selector in ["turn_boundary:-2", "turn_boundary:-5"]:
        for donor in DONORS:
            plot_scatter_4panel(full, selector, donor)
        plot_comparison_bars(summary, selector)
        plot_layer_sweep(summary, selector)


if __name__ == "__main__":
    main()
