"""Cross-layer steering analysis.

For each (probe_condition, steer_layer, signed_multiplier), computes:
- P(chose A in original order) — should increase with multiplier if steering works
- Refusal rate (task_completed == "neither")
- N (sample count)

Generates:
1. Cross-layer transfer heatmap: max steering effect per (probe, steer_layer)
2. Dose-response grid: P(chose A) vs coefficient for each (probe, steer_layer)
3. Coherence by coefficient: from coherence JSONL files
"""

from dotenv import load_dotenv

load_dotenv()

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "steering" / "cross_layer"
ASSETS_DIR = DATA_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

PROBE_LABELS = ["L25", "L32", "L46"]
STEER_LAYERS = [10, 15, 20, 25, 30]


def load_parsed_data() -> list[dict]:
    """Load all available parsed JSONL files."""
    rows = []
    for label in PROBE_LABELS:
        path = DATA_DIR / f"checkpoint_{label}.parsed.jsonl"
        if not path.exists():
            print(f"Skipping {path.name} (not found)")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                # Skip rows missing task_completed (parse errors)
                if "task_completed" not in row:
                    continue
                rows.append(row)
        print(f"Loaded {path.name}")
    print(f"Total rows: {len(rows)}")
    return rows


def load_coherence_data() -> list[dict]:
    """Load all available coherence JSONL files (raw, not summary)."""
    rows = []
    for label in PROBE_LABELS:
        path = DATA_DIR / f"checkpoint_{label}.coherence.jsonl"
        if not path.exists():
            print(f"Skipping coherence {path.name} (not found)")
            continue
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        print(f"Loaded coherence {path.name}")
    return rows


def compute_steering_stats(
    rows: list[dict],
) -> dict[tuple[str, int, float], dict]:
    """Compute P(chose A), refusal rate, N for each (condition, layer, mult)."""
    groups: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["layer"], row["signed_multiplier"])
        groups[key].append(row)

    stats = {}
    for key, group_rows in groups.items():
        n = len(group_rows)
        chose_a = sum(1 for r in group_rows if r["choice_original"] == "a")
        refusals = sum(1 for r in group_rows if r["task_completed"] == "neither")
        stats[key] = {
            "p_chose_a": chose_a / n,
            "refusal_rate": refusals / n,
            "n": n,
        }
    return stats


def compute_coherence_stats(
    rows: list[dict],
) -> dict[tuple[str, int, float], dict]:
    """Compute coherence rate for each (condition, layer, mult)."""
    groups: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["layer"], row["signed_multiplier"])
        groups[key].append(row)

    stats = {}
    for key, group_rows in groups.items():
        n = len(group_rows)
        coherent = sum(1 for r in group_rows if r["coherent"])
        stats[key] = {
            "coherent_rate": coherent / n,
            "n": n,
        }
    return stats


def get_available_conditions(rows: list[dict]) -> list[str]:
    """Get sorted unique conditions."""
    return sorted(set(r["condition"] for r in rows))


def get_multipliers(stats: dict) -> list[float]:
    """Get sorted unique multipliers."""
    return sorted(set(k[2] for k in stats))


def plot_heatmap(stats: dict, conditions: list[str]) -> None:
    """Cross-layer transfer heatmap: max P(steered) for positive mults."""
    # For each (condition, steer_layer), compute max steering effect.
    # "Steering effect" = max P(chose A) across positive multipliers.
    # We also show the baseline P(chose A) at mult=0 for context.
    n_probes = len(conditions)
    n_layers = len(STEER_LAYERS)

    # Build matrix: rows = probe conditions, cols = steer layers
    # Value = max P(chose A) for positive mults - baseline P(chose A)
    effect_matrix = np.full((n_probes, n_layers), np.nan)
    raw_matrix = np.full((n_probes, n_layers), np.nan)

    for i, cond in enumerate(conditions):
        for j, layer in enumerate(STEER_LAYERS):
            baseline_key = (cond, layer, 0.0)
            if baseline_key not in stats:
                baseline_key = (cond, layer, 0)
            baseline = stats.get(baseline_key, {}).get("p_chose_a", np.nan)

            pos_mults = [k[2] for k in stats if k[0] == cond and k[1] == layer and k[2] > 0]
            if pos_mults:
                max_p = max(stats[(cond, layer, m)]["p_chose_a"] for m in pos_mults)
                effect_matrix[i, j] = max_p - baseline
                raw_matrix[i, j] = max_p

    fig, ax = plt.subplots(figsize=(8, 3 + 0.5 * n_probes))

    # Use effect (max - baseline) as the heatmap value
    im = ax.imshow(
        effect_matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.5,
        aspect="auto",
    )

    # Annotate cells with raw max P(chose A) and effect
    for i in range(n_probes):
        for j in range(n_layers):
            if np.isnan(effect_matrix[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=10, color="gray")
            else:
                effect = effect_matrix[i, j]
                raw = raw_matrix[i, j]
                text_color = "white" if effect > 0.25 else "black"
                ax.text(
                    j, i,
                    f"{effect:+.2f}\n({raw:.2f})",
                    ha="center", va="center", fontsize=9, color=text_color,
                )

    probe_labels = [c.replace("probe_", "Probe ") for c in conditions]
    ax.set_yticks(range(n_probes))
    ax.set_yticklabels(probe_labels)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {l}" for l in STEER_LAYERS])
    ax.set_xlabel("Steer layer")
    ax.set_ylabel("Probe source")
    ax.set_title("Cross-layer steering transfer\n(max P(chose A) effect over baseline, raw in parens)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Steering effect (max - baseline)")

    fig.tight_layout()
    out = ASSETS_DIR / "plot_032326_cross_layer_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_dose_response_grid(stats: dict, conditions: list[str]) -> None:
    """Grid of dose-response curves: P(chose A) vs signed_multiplier."""
    n_probes = len(conditions)
    n_layers = len(STEER_LAYERS)
    mults = get_multipliers(stats)

    fig, axes = plt.subplots(
        n_probes, n_layers,
        figsize=(3.2 * n_layers, 3 * n_probes),
        sharex=True, sharey=True,
    )
    if n_probes == 1:
        axes = axes[np.newaxis, :]

    for i, cond in enumerate(conditions):
        for j, layer in enumerate(STEER_LAYERS):
            ax = axes[i, j]
            p_vals = []
            refusal_vals = []
            valid_mults = []
            for m in mults:
                key = (cond, layer, m)
                if key in stats:
                    p_vals.append(stats[key]["p_chose_a"])
                    refusal_vals.append(stats[key]["refusal_rate"])
                    valid_mults.append(m)

            ax.plot(valid_mults, p_vals, "o-", color="#2176AE", markersize=4, linewidth=1.5, label="P(chose A)")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

            # Shade refusal region
            ax.fill_between(
                valid_mults, 0, refusal_vals,
                alpha=0.15, color="red", label="Refusal rate",
            )

            ax.set_ylim(0, 1)
            ax.set_xlim(min(mults) - 0.01, max(mults) + 0.01)

            if i == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)
            if j == 0:
                ax.set_ylabel(cond.replace("probe_", "Probe ") + "\nP(chose A)")
            if i == n_probes - 1:
                ax.set_xlabel("Signed multiplier")

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Dose-response: P(chose A in original order) vs steering coefficient", fontsize=12, y=1.05)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_032326_dose_response_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_coherence(coh_stats: dict, conditions: list[str]) -> None:
    """Coherence rate vs coefficient for each (probe, layer) combo."""
    n_probes = len(conditions)
    n_layers = len(STEER_LAYERS)
    mults = get_multipliers(coh_stats)

    fig, axes = plt.subplots(
        n_probes, n_layers,
        figsize=(3.2 * n_layers, 3 * n_probes),
        sharex=True, sharey=True,
    )
    if n_probes == 1:
        axes = axes[np.newaxis, :]

    for i, cond in enumerate(conditions):
        for j, layer in enumerate(STEER_LAYERS):
            ax = axes[i, j]
            coh_vals = []
            valid_mults = []
            ns = []
            for m in mults:
                key = (cond, layer, m)
                if key in coh_stats:
                    coh_vals.append(coh_stats[key]["coherent_rate"])
                    valid_mults.append(m)
                    ns.append(coh_stats[key]["n"])

            if valid_mults:
                ax.plot(valid_mults, coh_vals, "s-", color="#57A773", markersize=4, linewidth=1.5)

            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(min(mults) - 0.01, max(mults) + 0.01)

            if i == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)
            if j == 0:
                ax.set_ylabel(cond.replace("probe_", "Probe ") + "\nCoherence rate")
            if i == n_probes - 1:
                ax.set_xlabel("Signed multiplier")

            # Annotate N in corner if available
            if ns:
                ax.text(
                    0.95, 0.05, f"n={ns[0]}",
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                    color="gray",
                )

    fig.suptitle("Coherence rate vs steering coefficient", fontsize=12, y=1.03)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_032326_coherence_by_coef.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(stats: dict, conditions: list[str]) -> None:
    """Print a summary table."""
    mults = get_multipliers(stats)
    print("\n" + "=" * 80)
    print("SUMMARY: P(chose A) by (condition, layer, multiplier)")
    print("=" * 80)

    for cond in conditions:
        print(f"\n--- {cond} ---")
        header = f"{'Layer':>8}" + "".join(f"  {m:>7.3f}" for m in mults)
        print(header)
        for layer in STEER_LAYERS:
            vals = []
            for m in mults:
                key = (cond, layer, m)
                if key in stats:
                    vals.append(f"  {stats[key]['p_chose_a']:>7.3f}")
                else:
                    vals.append(f"  {'N/A':>7}")
            print(f"{layer:>8}" + "".join(vals))

    # Print refusal rates
    print("\n" + "=" * 80)
    print("SUMMARY: Refusal rate by (condition, layer, multiplier)")
    print("=" * 80)
    for cond in conditions:
        print(f"\n--- {cond} ---")
        header = f"{'Layer':>8}" + "".join(f"  {m:>7.3f}" for m in mults)
        print(header)
        for layer in STEER_LAYERS:
            vals = []
            for m in mults:
                key = (cond, layer, m)
                if key in stats:
                    vals.append(f"  {stats[key]['refusal_rate']:>7.3f}")
                else:
                    vals.append(f"  {'N/A':>7}")
            print(f"{layer:>8}" + "".join(vals))

    # Print sample sizes
    print("\n" + "=" * 80)
    print("SUMMARY: N by (condition, layer, multiplier)")
    print("=" * 80)
    for cond in conditions:
        print(f"\n--- {cond} ---")
        header = f"{'Layer':>8}" + "".join(f"  {m:>7.3f}" for m in mults)
        print(header)
        for layer in STEER_LAYERS:
            vals = []
            for m in mults:
                key = (cond, layer, m)
                if key in stats:
                    vals.append(f"  {stats[key]['n']:>7}")
                else:
                    vals.append(f"  {'N/A':>7}")
            print(f"{layer:>8}" + "".join(vals))


def main() -> None:
    # Load data
    rows = load_parsed_data()
    conditions = get_available_conditions(rows)
    print(f"Conditions: {conditions}")

    # Compute stats
    stats = compute_steering_stats(rows)
    print_summary(stats, conditions)

    # Generate plots
    plot_heatmap(stats, conditions)
    plot_dose_response_grid(stats, conditions)

    # Coherence
    coh_rows = load_coherence_data()
    if coh_rows:
        coh_conditions = get_available_conditions(coh_rows)
        coh_stats = compute_coherence_stats(coh_rows)
        plot_coherence(coh_stats, coh_conditions)
    else:
        print("No coherence data available, skipping coherence plot")

    print("\nDone.")


if __name__ == "__main__":
    main()
