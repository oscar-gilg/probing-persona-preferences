"""Plot P(chose steered task) vs coefficient for both label and content metrics.

Produces two plot files per experiment:
- plot_032426_p_steered_label.png  (regex label)
- plot_032426_p_steered_content.png  (judge task_completed)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()


def load_parsed(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "task_completed" in r:
                    rows.append(r)
    return rows


def chose_steered(row: dict, field: str) -> bool | None:
    """Did the model choose the task it was steered toward?

    Positive mult steers toward original A (ordering handled by _effective_coef).
    Negative mult steers toward original B.
    mult=0: no steering direction, skip.
    """
    mult = row["signed_multiplier"]
    if mult == 0:
        return None
    val = row[field]
    if val == "neither" or val == "refusal":
        return None  # can't determine
    # Positive mult → steered toward A. Did the model choose A?
    if mult > 0:
        return val == "a"
    else:
        return val == "b"


def compute_p_steered(rows: list[dict], field: str) -> dict[float, tuple[float, int]]:
    """Returns {coefficient: (p_steered, n_valid)} excluding neither/refusal."""
    by_coef: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        result = chose_steered(r, field)
        if result is not None:
            by_coef[r["signed_multiplier"]].append(result)
    return {coef: (np.mean(vals), len(vals)) for coef, vals in by_coef.items()}


def compute_neither_rate(rows: list[dict]) -> dict[float, float]:
    by_coef: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        if r["signed_multiplier"] != 0:
            by_coef[r["signed_multiplier"]].append(r.get("task_completed") == "neither")
    return {coef: np.mean(vals) for coef, vals in by_coef.items()}


def plot_p_steered(
    data_by_group: dict[str, dict[float, tuple[float, int]]],
    neither_by_group: dict[str, dict[float, float]] | None,
    title: str,
    ylabel: str,
    output_path: Path,
    layer_groups: dict[str, list[dict]] | None = None,
):
    """Plot P(chose steered task) vs coefficient for multiple groups."""
    fig, axes = plt.subplots(1, len(data_by_group), figsize=(5 * len(data_by_group), 4.5),
                              sharey=True, squeeze=False)
    axes = axes[0]

    colors = {"Benign": "tab:blue", "Harmful-Benign": "tab:orange", "Harmful-Harmful": "tab:red"}

    for ax_idx, (group_name, coef_data) in enumerate(sorted(data_by_group.items())):
        ax = axes[ax_idx]
        coefs = sorted(coef_data.keys())
        p_vals = [coef_data[c][0] for c in coefs]
        n_vals = [coef_data[c][1] for c in coefs]

        color = colors.get(group_name, "tab:gray")
        ax.plot(coefs, p_vals, "o-", color=color, linewidth=2, markersize=5, label=group_name)

        # Add neither rate as shading if provided
        if neither_by_group and group_name in neither_by_group:
            n_data = neither_by_group[group_name]
            n_coefs = sorted(n_data.keys())
            n_rates = [n_data[c] for c in n_coefs]
            ax.fill_between(n_coefs, 0, n_rates, alpha=0.15, color="gray", label="Neither rate")

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Steering strength (x mean norm)")
        ax.set_ylim(0, 1)
        ax.set_title(group_name)
        if ax_idx == 0:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    # Load harmful experiment
    harmful_parsed = load_parsed(Path("experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl"))
    with open("experiments/steering/cross_layer_harmful/pairs_200.json") as f:
        harmful_pairs = {p["pair_id"]: p for p in json.load(f)}

    # Load benign experiment (L25 only)
    benign_parsed = load_parsed(Path("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl"))

    assets = Path("experiments/steering/cross_layer_harmful/assets")
    assets.mkdir(exist_ok=True)

    # Filter to probe_L25, layer 25
    harmful_rows = [r for r in harmful_parsed if r["condition"] == "probe_L25" and r["layer"] == 25]
    benign_rows = [r for r in benign_parsed if r["layer"] == 25]

    # Split harmful by pair type
    hb_rows = [r for r in harmful_rows if harmful_pairs[r["pair_id"]]["pair_type"] == "harmful_benign"]
    hh_rows = [r for r in harmful_rows if harmful_pairs[r["pair_id"]]["pair_type"] == "harmful_harmful"]

    for field, metric_name, suffix in [
        ("choice_original", "P(chose steered task) — regex label", "label"),
        ("task_completed", "P(chose steered task) — judge content", "content"),
    ]:
        data = {
            "Benign": compute_p_steered(benign_rows, field),
            "Harmful-Benign": compute_p_steered(hb_rows, field),
            "Harmful-Harmful": compute_p_steered(hh_rows, field),
        }
        neither = {
            "Benign": compute_neither_rate(benign_rows),
            "Harmful-Benign": compute_neither_rate(hb_rows),
            "Harmful-Harmful": compute_neither_rate(hh_rows),
        }
        plot_p_steered(
            data, neither,
            title=f"{metric_name}\n(probe L25, layer 25)",
            ylabel="P(chose steered task)",
            output_path=assets / f"plot_032426_p_steered_{suffix}.png",
        )

        # Print table
        print(f"\n=== {metric_name} ===")
        print(f"{'coef':>8} {'Benign':>10} {'H-B':>10} {'H-H':>10}")
        all_coefs = sorted(set(list(data["Benign"].keys()) + list(data["Harmful-Benign"].keys())))
        for c in all_coefs:
            vals = []
            for g in ["Benign", "Harmful-Benign", "Harmful-Harmful"]:
                if c in data[g]:
                    vals.append(f"{data[g][c][0]:.3f}")
                else:
                    vals.append("   -")
            print(f"{c:>8.3f} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")


if __name__ == "__main__":
    main()
