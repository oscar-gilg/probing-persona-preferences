"""Replot cross-layer steering figures with improved labels and formatting."""

from dotenv import load_dotenv

load_dotenv()

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "steering" / "cross_layer"
ASSETS_DIR = DATA_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

PROBE_LABELS = ["L25", "L32", "L46"]
STEER_LAYERS = [10, 15, 20, 25, 30]


def load_parsed_data() -> list[dict]:
    rows = []
    for label in PROBE_LABELS:
        path = DATA_DIR / f"checkpoint_{label}.parsed.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "task_completed" not in row:
                    continue
                rows.append(row)
    return rows


def load_coherence_data() -> list[dict]:
    rows = []
    for label in PROBE_LABELS:
        path = DATA_DIR / f"checkpoint_{label}.coherence.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def compute_steering_stats(rows: list[dict]) -> dict[tuple[str, int, float], dict]:
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


def compute_coherence_stats(rows: list[dict]) -> dict[tuple[str, int, float], dict]:
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


def get_conditions(rows: list[dict]) -> list[str]:
    return sorted(set(r["condition"] for r in rows))


def get_multipliers(stats: dict) -> list[float]:
    return sorted(set(k[2] for k in stats))


# Friendly label: "Probe L25 (~40%)" showing fractional depth in 62-layer model
def probe_display(cond: str) -> str:
    layer_num = int(cond.replace("probe_L", ""))
    frac = layer_num / 62
    return f"Probe from L{layer_num} ({frac:.0%} depth)"


def probe_short(cond: str) -> str:
    return cond.replace("probe_", "Probe ")


def plot_heatmap(stats: dict, conditions: list[str]) -> None:
    """Cross-layer transfer heatmap.

    Shows max P(chose steered task) at best coefficient for each
    (probe source, steer layer) combination. Color encodes the value
    directly. Chance = 0.50 is marked.
    """
    n_probes = len(conditions)
    n_layers = len(STEER_LAYERS)

    # For each (condition, steer_layer), find max P(chose A) across positive mults
    raw_matrix = np.full((n_probes, n_layers), np.nan)
    refusal_matrix = np.full((n_probes, n_layers), np.nan)

    for i, cond in enumerate(conditions):
        for j, layer in enumerate(STEER_LAYERS):
            pos_mults = [k[2] for k in stats if k[0] == cond and k[1] == layer and k[2] > 0]
            if pos_mults:
                # Find the mult that gives max P(chose A)
                best_mult = max(pos_mults, key=lambda m: stats[(cond, layer, m)]["p_chose_a"])
                raw_matrix[i, j] = stats[(cond, layer, best_mult)]["p_chose_a"]
                refusal_matrix[i, j] = stats[(cond, layer, best_mult)]["refusal_rate"]

    fig, ax = plt.subplots(figsize=(9, 3.5 + 0.4 * n_probes))

    im = ax.imshow(
        raw_matrix,
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        aspect="auto",
    )

    for i in range(n_probes):
        for j in range(n_layers):
            if np.isnan(raw_matrix[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=10, color="gray")
            else:
                p = raw_matrix[i, j]
                ref = refusal_matrix[i, j]
                text_color = "white" if p > 0.85 else "black"
                ax.text(
                    j, i,
                    f"P = {p:.2f}\n({ref:.0%} refused)",
                    ha="center", va="center", fontsize=9, color=text_color,
                    fontweight="bold" if p >= 0.95 else "normal",
                )

    probe_labels = [probe_display(c) for c in conditions]
    ax.set_yticks(range(n_probes))
    ax.set_yticklabels(probe_labels, fontsize=9)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {l}" for l in STEER_LAYERS], fontsize=9)
    ax.set_xlabel("Layer where steering is applied", fontsize=10)
    ax.set_ylabel("Probe training layer", fontsize=10)
    ax.set_title(
        "Peak steering success across layers\n"
        "(P = fraction choosing steered task at best coefficient; chance = 0.50)",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("P(chose steered task)", fontsize=9)
    # Mark chance level on colorbar
    cbar.ax.axhline(0.5, color="black", linewidth=1, linestyle="--")

    fig.tight_layout()
    out = ASSETS_DIR / "plot_032326_cross_layer_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_dose_response_grid(stats: dict, conditions: list[str]) -> None:
    """Grid of dose-response curves with refusal overlay.

    X-axis: steering strength (fraction of mean activation norm).
    Positive = steer toward task A, negative = steer toward task B.
    Y-axis: P(chose task A among non-refusals).
    Pink shading: refusal rate.
    """
    n_probes = len(conditions)
    n_layers = len(STEER_LAYERS)
    mults = get_multipliers(stats)

    fig, axes = plt.subplots(
        n_probes, n_layers,
        figsize=(3.4 * n_layers, 3.2 * n_probes),
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

            ax.plot(
                valid_mults, p_vals, "o-",
                color="#2176AE", markersize=4, linewidth=1.5,
                label="P(chose task A)", zorder=3,
            )
            ax.fill_between(
                valid_mults, 0, refusal_vals,
                alpha=0.20, color="#D64045",
                label="Refusal rate", zorder=2,
            )

            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

            ax.set_ylim(0, 1.0)
            ax.set_xlim(min(mults) - 0.01, max(mults) + 0.01)

            if i == 0:
                ax.set_title(f"Steer at layer {layer}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{probe_short(cond)}\nP(chose task A)", fontsize=9)
            if i == n_probes - 1:
                ax.set_xlabel("Steering strength\n(fraction of mean norm)", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=2, fontsize=10,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle(
        "Dose-response: choice probability vs steering strength\n"
        "(positive = steer toward task A, negative = toward task B; chance = 0.50)",
        fontsize=12, y=1.08,
    )
    fig.tight_layout()
    out = ASSETS_DIR / "plot_032326_dose_response_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_coherence(coh_stats: dict, conditions: list[str]) -> None:
    """Coherence rate vs steering strength.

    Same grid layout as dose-response for visual consistency.
    Annotates sample size per cell prominently since n is very small.
    """
    n_probes = len(conditions)
    n_layers = len(STEER_LAYERS)
    mults = get_multipliers(coh_stats)

    fig, axes = plt.subplots(
        n_probes, n_layers,
        figsize=(3.4 * n_layers, 3.2 * n_probes),
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
                ax.plot(
                    valid_mults, coh_vals, "s-",
                    color="#57A773", markersize=5, linewidth=1.5,
                )

            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(min(mults) - 0.01, max(mults) + 0.01)

            if i == 0:
                ax.set_title(f"Steer at layer {layer}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{probe_short(cond)}\nCoherence rate", fontsize=9)
            if i == n_probes - 1:
                ax.set_xlabel("Steering strength\n(fraction of mean norm)", fontsize=9)

            # Sample size annotation — prominent since n is tiny
            if ns:
                n_val = ns[0]
                ax.text(
                    0.95, 0.05, f"n = {n_val} per point",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                    color="gray", style="italic",
                )

    fig.suptitle(
        "Coherence of steered completions\n"
        "(fraction of non-refusal completions judged coherent; small samples — interpret cautiously)",
        fontsize=12, y=1.06,
    )
    fig.tight_layout()
    out = ASSETS_DIR / "plot_032326_coherence_by_coef.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    rows = load_parsed_data()
    conditions = get_conditions(rows)
    stats = compute_steering_stats(rows)

    plot_heatmap(stats, conditions)
    plot_dose_response_grid(stats, conditions)

    coh_rows = load_coherence_data()
    if coh_rows:
        coh_conditions = get_conditions(coh_rows)
        coh_stats = compute_coherence_stats(coh_rows)
        plot_coherence(coh_stats, coh_conditions)

    print("Done.")


if __name__ == "__main__":
    main()
