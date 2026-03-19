"""Plot persona augmentation experiment results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/experiments/persona_augmentation/persona_augmentation_results.json")
ASSETS = Path("experiments/probe_generalization/multi_role_ablation/persona_augmentation/assets")

DONORS = ["sadist", "villain", "midwest"]
SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]
LAYERS = [25, 32, 39, 46, 53]


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_alpha_comparison(results: list[dict]):
    """Compare optimal alpha between baseline and augmented conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, selector in zip(axes, SELECTORS):
        baseline = [r for r in results if r["condition"] == "baseline" and r["selector"] == selector]
        base_alphas = {r["layer"]: r["best_alpha"] for r in baseline}

        for donor in DONORS:
            aug = [r for r in results if r["condition"] == "augmented"
                   and r["donor"] == donor and r["selector"] == selector]
            if not aug:
                continue
            aug_alphas = [r["best_alpha"] for r in sorted(aug, key=lambda x: x["layer"])]
            layers_sorted = sorted([r["layer"] for r in aug])
            ax.plot(layers_sorted, aug_alphas, "o-", label=f"+{donor}", markersize=6)

        base_vals = [base_alphas[l] for l in sorted(base_alphas.keys())]
        ax.plot(sorted(base_alphas.keys()), base_vals, "ks--", label="baseline", markersize=6)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Best alpha")
        ax.set_yscale("log")
        ax.set_title(f"{selector}")
        ax.legend()

    fig.suptitle("Optimal alpha: baseline vs augmented", fontsize=13)
    plt.tight_layout()
    out = ASSETS / "plot_031626_alpha_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_performance_delta(results: list[dict]):
    """Plot change in Pearson r (augmented - baseline) across layers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, selector in enumerate(SELECTORS):
        baseline = {r["layer"]: r for r in results
                    if r["condition"] == "baseline" and r["selector"] == selector}

        # Top row: noprompt test performance
        ax = axes[0, col]
        for donor in DONORS:
            aug = {r["layer"]: r for r in results
                   if r["condition"] == "augmented" and r["donor"] == donor and r["selector"] == selector}
            if not aug:
                continue
            layers_sorted = sorted(aug.keys())
            deltas = [aug[l]["noprompt_test"]["pearson_r"] - baseline[l]["noprompt_test"]["pearson_r"]
                      for l in layers_sorted]
            ax.plot(layers_sorted, deltas, "o-", label=f"+{donor}", markersize=6)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Delta Pearson r")
        ax.set_title(f"Noprompt test ({selector})")
        ax.legend()

        # Bottom row: donor persona test performance
        ax = axes[1, col]
        for donor in DONORS:
            aug = {r["layer"]: r for r in results
                   if r["condition"] == "augmented" and r["donor"] == donor and r["selector"] == selector}
            if not aug:
                continue
            layers_sorted = sorted(aug.keys())

            # Augmented on donor persona vs baseline on donor persona
            deltas = []
            for l in layers_sorted:
                aug_r = aug[l]["persona_test"].get(donor, {}).get("pearson_r")
                base_r = baseline[l]["persona_test"].get(donor, {}).get("pearson_r")
                if aug_r is not None and base_r is not None:
                    deltas.append(aug_r - base_r)
                else:
                    deltas.append(float("nan"))
            ax.plot(layers_sorted, deltas, "o-", label=f"+{donor} → {donor}", markersize=6)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Delta Pearson r")
        ax.set_title(f"Donor persona test ({selector})")
        ax.legend()

    fig.suptitle("Performance delta: augmented - baseline", fontsize=13)
    plt.tight_layout()
    out = ASSETS / "plot_031626_performance_delta.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_cross_persona_transfer(results: list[dict]):
    """For each donor augmentation, show OOD performance on all personas."""
    fig, axes = plt.subplots(1, len(DONORS), figsize=(6 * len(DONORS), 5))
    if len(DONORS) == 1:
        axes = [axes]

    selector = "turn_boundary:-2"  # Focus on best selector

    for ax, donor in zip(axes, DONORS):
        # Average across layers
        baseline_by_persona = defaultdict(list)
        aug_by_persona = defaultdict(list)

        for r in results:
            if r["selector"] != selector:
                continue
            if r["condition"] == "baseline":
                for p, ev in r["persona_test"].items():
                    baseline_by_persona[p].append(ev["pearson_r"])
            elif r["condition"] == "augmented" and r["donor"] == donor:
                for p, ev in r["persona_test"].items():
                    aug_by_persona[p].append(ev["pearson_r"])

        personas = sorted(set(baseline_by_persona.keys()) & set(aug_by_persona.keys()))
        base_means = [np.mean(baseline_by_persona[p]) for p in personas]
        aug_means = [np.mean(aug_by_persona[p]) for p in personas]

        y = np.arange(len(personas))
        ax.barh(y - 0.15, base_means, height=0.3, color="#90CAF9", label="Baseline")
        ax.barh(y + 0.15, aug_means, height=0.3, color="#E53935", label=f"+{donor}")
        ax.set_yticks(y)
        ax.set_yticklabels(personas, fontsize=9)
        ax.set_xlabel("Mean Pearson r (across layers)")
        ax.set_title(f"Augmented with {donor}")
        ax.set_xlim(0, 1.0)
        ax.legend(fontsize=8)

    fig.suptitle(f"Cross-persona transfer ({selector})", fontsize=13)
    plt.tight_layout()
    out = ASSETS / "plot_031626_cross_persona_transfer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def print_summary(results: list[dict]):
    """Print summary tables."""
    for selector in SELECTORS:
        print(f"\n{'='*70}")
        print(f"Selector: {selector}")
        print(f"{'='*70}")

        baseline = {r["layer"]: r for r in results
                    if r["condition"] == "baseline" and r["selector"] == selector}

        print(f"\n{'Layer':>6} {'Baseline':>10}", end="")
        for donor in DONORS:
            print(f" {'+' + donor:>12}", end="")
        print()

        print("--- Noprompt test r ---")
        for layer in LAYERS:
            base_r = baseline[layer]["noprompt_test"]["pearson_r"]
            print(f"{layer:>6} {base_r:>10.3f}", end="")
            for donor in DONORS:
                aug = [r for r in results if r["condition"] == "augmented"
                       and r["donor"] == donor and r["selector"] == selector and r["layer"] == layer]
                if aug:
                    aug_r = aug[0]["noprompt_test"]["pearson_r"]
                    delta = aug_r - base_r
                    print(f" {aug_r:>7.3f}({delta:+.3f})", end="")
                else:
                    print(f" {'n/a':>12}", end="")
            print()

        print("\n--- Optimal alpha ---")
        for layer in LAYERS:
            base_a = baseline[layer]["best_alpha"]
            print(f"{layer:>6} {base_a:>10.1f}", end="")
            for donor in DONORS:
                aug = [r for r in results if r["condition"] == "augmented"
                       and r["donor"] == donor and r["selector"] == selector and r["layer"] == layer]
                if aug:
                    print(f" {aug[0]['best_alpha']:>12.1f}", end="")
                else:
                    print(f" {'n/a':>12}", end="")
            print()

        print("\n--- Donor persona test r ---")
        for layer in LAYERS:
            base_str = f"{layer:>6} {'':>10}"
            print(base_str, end="")
            for donor in DONORS:
                base_r = baseline[layer]["persona_test"].get(donor, {}).get("pearson_r")
                aug = [r for r in results if r["condition"] == "augmented"
                       and r["donor"] == donor and r["selector"] == selector and r["layer"] == layer]
                if aug and base_r is not None:
                    aug_r = aug[0]["persona_test"].get(donor, {}).get("pearson_r")
                    if aug_r is not None:
                        delta = aug_r - base_r
                        print(f" {aug_r:>7.3f}({delta:+.3f})", end="")
                    else:
                        print(f" {'n/a':>12}", end="")
                else:
                    print(f" {'n/a':>12}", end="")
            print()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    results = load_results()
    print(f"Loaded {len(results)} result entries")

    print_summary(results)
    plot_alpha_comparison(results)
    plot_performance_delta(results)
    plot_cross_persona_transfer(results)


if __name__ == "__main__":
    main()
