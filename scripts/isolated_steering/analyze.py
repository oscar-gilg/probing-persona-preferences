"""Analyze isolated steering experiment results.

Combines three data sources:
  1. Isolated steering (kv_cache_v_single, activation_patch):
     experiments/steering/isolated_steering/checkpoint.jsonl
  2. Differential steering (task_mean):
     experiments/steering/task_mean_direction/checkpoint.jsonl
  3. Baseline (unsteered):
     experiments/revealed_steering_v2/followup/checkpoint.jsonl

Outputs plots to experiments/steering/isolated_steering/assets/.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────

ISOLATED_CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint.jsonl")
DIFFERENTIAL_CHECKPOINT = Path("experiments/steering/task_mean_direction/checkpoint.jsonl")
BASELINE_CHECKPOINT = Path("experiments/revealed_steering_v2/followup/checkpoint.jsonl")
PAIRS_500_PATH = Path("experiments/revealed_steering_v2/followup/pairs_500.json")
ASSETS_DIR = Path("experiments/steering/isolated_steering/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DATE_STR = "031626"

# ── Shared constants ─────────────────────────────────────────────────────────

ALL_LAYERS = [25, 32, 39, 46, 53]
DIFFERENTIAL_LAYERS = [25, 32]
ISOLATED_MULTIPLIERS = [0.02, 0.03, 0.05]
DIFFERENTIAL_MULTIPLIERS = [0.01, 0.02, 0.03, 0.05]

CONDITION_COLORS = {
    "task_mean": "#1f77b4",
    "kv_cache_v_single": "#ff7f0e",
    "activation_patch": "#2ca02c",
}
CONDITION_LABELS = {
    "task_mean": "Differential",
    "kv_cache_v_single": "KV cache (V)",
    "activation_patch": "Activation patch",
}

N_BOOT = 10_000
RNG = np.random.default_rng(42)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_200_pair_ids() -> set[str]:
    with open(PAIRS_500_PATH) as f:
        pairs = json.load(f)
    random.seed(42)
    subset = random.sample(pairs, 200)
    return {p["pair_id"] for p in subset}


def load_all_data() -> dict[str, list[dict]]:
    """Load and filter all data sources, returning records keyed by condition."""
    pair_ids_200 = get_200_pair_ids()

    # Isolated steering (already uses 200 pairs)
    isolated = load_jsonl(ISOLATED_CHECKPOINT)
    print(f"Isolated steering records: {len(isolated)}")

    # Differential steering — filter to 200 pairs, rename condition for clarity
    differential_raw = load_jsonl(DIFFERENTIAL_CHECKPOINT)
    differential = [r for r in differential_raw if r["pair_id"] in pair_ids_200]
    print(f"Differential steering records (filtered to 200 pairs): {len(differential)}")

    # Baseline — filter to 200 pairs, baseline condition only
    baseline_raw = load_jsonl(BASELINE_CHECKPOINT)
    baseline = [
        r for r in baseline_raw
        if r["condition"] == "baseline" and r["pair_id"] in pair_ids_200
    ]
    print(f"Baseline records (filtered to 200 pairs): {len(baseline)}")

    return {
        "isolated": isolated,
        "differential": differential,
        "baseline": baseline,
    }


# ── Core analysis functions ──────────────────────────────────────────────────

def is_valid_choice(choice: str | None) -> bool:
    return choice in ("a", "b")


def compute_per_pair_pa(
    records: list[dict],
    condition: str,
    layer: int | None,
    multiplier: float,
) -> dict[str, float]:
    """Compute P(choose_a) per pair for a given condition/layer/multiplier."""
    filtered = [
        r for r in records
        if r["condition"] == condition
        and r["multiplier"] == multiplier
        and is_valid_choice(r["choice_original"])
    ]
    if layer is not None:
        filtered = [r for r in filtered if r.get("layer") == layer]

    pair_choices: dict[str, list[int]] = defaultdict(list)
    for r in filtered:
        pair_choices[r["pair_id"]].append(1 if r["choice_original"] == "a" else 0)

    return {pid: np.mean(choices) for pid, choices in pair_choices.items()}


def compute_steering_effect(
    records: list[dict],
    condition: str,
    layer: int | None,
    abs_multiplier: float,
) -> tuple[float, float, float, int]:
    """Compute steering effect = mean(p_a at +mult) - mean(p_a at -mult).

    Returns (effect, ci_lo, ci_hi, n_pairs).
    """
    pa_pos = compute_per_pair_pa(records, condition, layer, abs_multiplier)
    pa_neg = compute_per_pair_pa(records, condition, layer, -abs_multiplier)

    # Only use pairs present in both conditions
    common_pairs = sorted(set(pa_pos.keys()) & set(pa_neg.keys()))
    if len(common_pairs) == 0:
        return float("nan"), float("nan"), float("nan"), 0

    pos_vals = np.array([pa_pos[pid] for pid in common_pairs])
    neg_vals = np.array([pa_neg[pid] for pid in common_pairs])
    diffs = pos_vals - neg_vals

    effect = float(np.mean(diffs))

    # Bootstrap CI
    boot_effects = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, len(diffs), size=len(diffs))
        boot_effects[i] = np.mean(diffs[idx])
    ci_lo = float(np.percentile(boot_effects, 2.5))
    ci_hi = float(np.percentile(boot_effects, 97.5))

    return effect, ci_lo, ci_hi, len(common_pairs)


def compute_mean_pa_at_multiplier(
    records: list[dict],
    condition: str,
    layer: int | None,
    multiplier: float,
) -> tuple[float, int]:
    """Compute overall mean P(choose_a) at a given signed multiplier.

    Returns (mean_pa, n_valid).
    """
    filtered = [
        r for r in records
        if r["condition"] == condition
        and r["multiplier"] == multiplier
        and is_valid_choice(r["choice_original"])
    ]
    if layer is not None:
        filtered = [r for r in filtered if r.get("layer") == layer]

    if len(filtered) == 0:
        return float("nan"), 0

    chose_a = [1 if r["choice_original"] == "a" else 0 for r in filtered]
    return float(np.mean(chose_a)), len(filtered)


def compute_baseline_pa(baseline_records: list[dict]) -> float:
    """Compute overall baseline P(choose_a) across all pairs."""
    valid = [r for r in baseline_records if is_valid_choice(r["choice_original"])]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean([1 if r["choice_original"] == "a" else 0 for r in valid]))


def compute_parse_rate(
    records: list[dict],
    condition: str,
    layer: int | None,
    abs_multiplier: float,
) -> tuple[float, int]:
    """Fraction of responses with valid parses at +-multiplier.

    Returns (parse_rate, n_total).
    """
    filtered = [
        r for r in records
        if r["condition"] == condition
        and abs(r["multiplier"]) == abs_multiplier
    ]
    if layer is not None:
        filtered = [r for r in filtered if r.get("layer") == layer]

    if len(filtered) == 0:
        return float("nan"), 0

    n_valid = sum(1 for r in filtered if is_valid_choice(r["choice_original"]))
    return n_valid / len(filtered), len(filtered)


# ── Plot 1: Steering effect comparison (bar chart) ──────────────────────────

def plot_steering_effect_comparison(all_records: list[dict], baseline_records: list[dict]):
    fig, axes = plt.subplots(1, len(ALL_LAYERS), figsize=(4 * len(ALL_LAYERS), 5), sharey=True)
    if len(ALL_LAYERS) == 1:
        axes = [axes]

    conditions_ordered = ["task_mean", "kv_cache_v_single", "activation_patch"]
    bar_width = 0.22

    for ax_idx, layer in enumerate(ALL_LAYERS):
        ax = axes[ax_idx]
        multipliers_for_plot = ISOLATED_MULTIPLIERS

        x_positions = np.arange(len(multipliers_for_plot))
        for c_idx, condition in enumerate(conditions_ordered):
            # Differential only available for L25, L32
            if condition == "task_mean" and layer not in DIFFERENTIAL_LAYERS:
                continue

            effects = []
            ci_los = []
            ci_his = []
            for mult in multipliers_for_plot:
                eff, lo, hi, n = compute_steering_effect(all_records, condition, layer, mult)
                effects.append(eff)
                ci_los.append(lo)
                ci_his.append(hi)

            effects = np.array(effects)
            ci_los = np.array(ci_los)
            ci_his = np.array(ci_his)
            yerr_lo = effects - ci_los
            yerr_hi = ci_his - effects

            offset = (c_idx - 1) * bar_width
            ax.bar(
                x_positions + offset,
                effects,
                bar_width,
                yerr=[yerr_lo, yerr_hi],
                capsize=3,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition] if ax_idx == 0 else None,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"±{m}" for m in multipliers_for_plot])
        ax.set_xlabel("Multiplier")
        ax.set_title(f"Layer {layer}")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Steering effect: P(a|+m) - P(a|-m)")
    axes[0].set_ylim(-0.05, max(0.3, axes[0].get_ylim()[1]))

    # Shared legend
    handles, labels = [], []
    for condition in conditions_ordered:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=CONDITION_COLORS[condition], alpha=0.85,
                                     edgecolor="black", linewidth=0.5))
        labels.append(CONDITION_LABELS[condition])
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Steering effect by condition, layer, and multiplier", y=1.06, fontsize=13)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE_STR}_steering_effect_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: Dose-response curves ────────────────────────────────────────────

def plot_dose_response(all_records: list[dict], baseline_records: list[dict]):
    baseline_pa = compute_baseline_pa(baseline_records)

    fig, axes = plt.subplots(1, len(ALL_LAYERS), figsize=(4 * len(ALL_LAYERS), 5), sharey=True)
    if len(ALL_LAYERS) == 1:
        axes = [axes]

    conditions_ordered = ["task_mean", "kv_cache_v_single", "activation_patch"]
    markers = {"task_mean": "o", "kv_cache_v_single": "s", "activation_patch": "^"}

    for ax_idx, layer in enumerate(ALL_LAYERS):
        ax = axes[ax_idx]

        for condition in conditions_ordered:
            if condition == "task_mean" and layer not in DIFFERENTIAL_LAYERS:
                continue

            # Determine which signed multipliers this condition has
            if condition == "task_mean":
                signed_mults = sorted(
                    [-m for m in DIFFERENTIAL_MULTIPLIERS] + list(DIFFERENTIAL_MULTIPLIERS)
                )
            else:
                signed_mults = sorted(
                    [-m for m in ISOLATED_MULTIPLIERS] + list(ISOLATED_MULTIPLIERS)
                )

            mean_pas = []
            valid_mults = []
            for mult in signed_mults:
                pa, n = compute_mean_pa_at_multiplier(all_records, condition, layer, mult)
                if n > 0:
                    mean_pas.append(pa)
                    valid_mults.append(mult)

            if valid_mults:
                ax.plot(
                    valid_mults, mean_pas,
                    marker=markers[condition],
                    color=CONDITION_COLORS[condition],
                    label=CONDITION_LABELS[condition] if ax_idx == 0 else None,
                    markersize=5,
                    linewidth=1.5,
                    alpha=0.85,
                )

        # Baseline horizontal line
        if not np.isnan(baseline_pa):
            ax.axhline(baseline_pa, color="gray", linestyle="--", alpha=0.7,
                        label="Baseline" if ax_idx == 0 else None)

        ax.axhline(0.5, color="lightgray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Signed multiplier")
        ax.set_title(f"Layer {layer}")

    axes[0].set_ylabel("Mean P(choose a)")
    axes[0].set_ylim(0, 1.0)

    # Shared legend
    handles, labels = [], []
    for condition in conditions_ordered:
        handles.append(plt.Line2D([0], [0], color=CONDITION_COLORS[condition],
                                  marker=markers[condition], markersize=5, linewidth=1.5))
        labels.append(CONDITION_LABELS[condition])
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5))
    labels.append("Baseline")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Dose-response: mean P(choose a) vs signed multiplier", y=1.06, fontsize=13)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE_STR}_dose_response.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: Per-pair correlation scatter ─────────────────────────────────────

def plot_per_pair_scatter(all_records: list[dict]):
    target_mult = 0.05

    fig, axes = plt.subplots(1, len(DIFFERENTIAL_LAYERS), figsize=(6 * len(DIFFERENTIAL_LAYERS), 5.5))
    if len(DIFFERENTIAL_LAYERS) == 1:
        axes = [axes]

    for ax_idx, layer in enumerate(DIFFERENTIAL_LAYERS):
        ax = axes[ax_idx]

        # Per-pair steering effect = p_a(+mult) - p_a(-mult) for each method
        diff_pa_pos = compute_per_pair_pa(all_records, "task_mean", layer, target_mult)
        diff_pa_neg = compute_per_pair_pa(all_records, "task_mean", layer, -target_mult)
        kv_pa_pos = compute_per_pair_pa(all_records, "kv_cache_v_single", layer, target_mult)
        kv_pa_neg = compute_per_pair_pa(all_records, "kv_cache_v_single", layer, -target_mult)

        # Pairs present in both
        diff_pairs = set(diff_pa_pos.keys()) & set(diff_pa_neg.keys())
        kv_pairs = set(kv_pa_pos.keys()) & set(kv_pa_neg.keys())
        common = sorted(diff_pairs & kv_pairs)

        if len(common) < 2:
            ax.text(0.5, 0.5, f"Insufficient data\n(n={len(common)})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Layer {layer}")
            continue

        diff_effects = np.array([diff_pa_pos[p] - diff_pa_neg[p] for p in common])
        kv_effects = np.array([kv_pa_pos[p] - kv_pa_neg[p] for p in common])

        ax.scatter(diff_effects, kv_effects, alpha=0.4, s=20, color="steelblue", edgecolors="none")

        # Regression line (skip if constant values)
        if np.std(diff_effects) > 1e-10 and np.std(kv_effects) > 1e-10:
            slope, intercept, r_val, p_val, _ = stats.linregress(diff_effects, kv_effects)
            x_range = np.linspace(diff_effects.min(), diff_effects.max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color="red", linewidth=1.5, alpha=0.7)
        else:
            r_val, p_val = float("nan"), float("nan")

        ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel(f"Differential steering effect (±{target_mult})")
        ax.set_ylabel(f"KV cache steering effect (±{target_mult})")
        ax.set_title(f"Layer {layer}: r={r_val:.3f} (p={p_val:.3g}, n={len(common)})")

        # Set symmetric limits
        max_abs = max(np.abs(diff_effects).max(), np.abs(kv_effects).max(), 0.3)
        lim = max_abs * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle(
        f"Per-pair steering effect: differential vs KV cache (mult=±{target_mult})",
        fontsize=13,
    )
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE_STR}_per_pair_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 4: Parse rate heatmap ───────────────────────────────────────────────

def plot_parse_rate_heatmap(all_records: list[dict]):
    conditions_ordered = ["task_mean", "kv_cache_v_single", "activation_patch"]
    abs_multipliers = sorted(set(ISOLATED_MULTIPLIERS) | set(DIFFERENTIAL_MULTIPLIERS))

    # Build row labels and data
    row_labels = []
    data_rows = []

    for condition in conditions_ordered:
        available_layers = DIFFERENTIAL_LAYERS if condition == "task_mean" else ALL_LAYERS
        for layer in available_layers:
            row_labels.append(f"{CONDITION_LABELS[condition]} L{layer}")
            row = []
            for mult in abs_multipliers:
                rate, n = compute_parse_rate(all_records, condition, layer, mult)
                row.append(rate)
            data_rows.append(row)

    data = np.array(data_rows)

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(abs_multipliers)), max(4, 0.5 * len(row_labels))))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.0)

    ax.set_xticks(range(len(abs_multipliers)))
    ax.set_xticklabels([f"±{m}" for m in abs_multipliers])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Absolute multiplier")
    ax.set_title("Parse rate by condition x layer x multiplier")

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(abs_multipliers)):
            val = data[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:.2f}"
                color = "black" if val > 0.85 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Parse rate", shrink=0.8)
    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE_STR}_parse_rate_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 5: Layer comparison ─────────────────────────────────────────────────

def plot_layer_comparison(all_records: list[dict]):
    target_mult = 0.05
    conditions_to_plot = ["kv_cache_v_single", "activation_patch"]
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(ALL_LAYERS))

    for c_idx, condition in enumerate(conditions_to_plot):
        effects = []
        ci_los = []
        ci_his = []
        for layer in ALL_LAYERS:
            eff, lo, hi, n = compute_steering_effect(all_records, condition, layer, target_mult)
            effects.append(eff)
            ci_los.append(lo)
            ci_his.append(hi)

        effects = np.array(effects)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)
        yerr_lo = effects - ci_los
        yerr_hi = ci_his - effects

        offset = (c_idx - 0.5) * bar_width
        ax.bar(
            x_positions + offset,
            effects,
            bar_width,
            yerr=[yerr_lo, yerr_hi],
            capsize=4,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"L{l}" for l in ALL_LAYERS])
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Steering effect at ±{target_mult}")
    ax.set_title(f"Steering effect by layer (multiplier ±{target_mult})")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.05, max(0.3, ax.get_ylim()[1]))
    ax.legend()

    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE_STR}_layer_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Summary statistics ───────────────────────────────────────────────────────

def print_summary(all_records: list[dict], baseline_records: list[dict]):
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Record counts per condition
    condition_counts = defaultdict(int)
    for r in all_records:
        condition_counts[r["condition"]] += 1
    print("\nRecord counts by condition:")
    for cond, count in sorted(condition_counts.items()):
        print(f"  {cond}: {count}")
    print(f"  baseline: {len(baseline_records)}")

    # Unique pairs per condition
    print("\nUnique pairs per condition:")
    for cond in sorted(condition_counts.keys()):
        pairs = set(r["pair_id"] for r in all_records if r["condition"] == cond)
        print(f"  {cond}: {len(pairs)}")
    baseline_pairs = set(r["pair_id"] for r in baseline_records)
    print(f"  baseline: {len(baseline_pairs)}")

    # Baseline P(a)
    baseline_pa = compute_baseline_pa(baseline_records)
    print(f"\nBaseline mean P(choose a): {baseline_pa:.4f}")

    # Steering effects table
    print("\n" + "-" * 80)
    print(f"{'Condition':<22} {'Layer':>5} {'|mult|':>6} {'Effect':>8} {'95% CI':>18} {'N pairs':>8}")
    print("-" * 80)

    conditions_ordered = ["task_mean", "kv_cache_v_single", "activation_patch"]
    for condition in conditions_ordered:
        available_layers = DIFFERENTIAL_LAYERS if condition == "task_mean" else ALL_LAYERS
        for layer in available_layers:
            multipliers = DIFFERENTIAL_MULTIPLIERS if condition == "task_mean" else ISOLATED_MULTIPLIERS
            for mult in multipliers:
                eff, lo, hi, n = compute_steering_effect(all_records, condition, layer, mult)
                ci_str = f"[{lo:+.4f}, {hi:+.4f}]" if not np.isnan(eff) else "N/A"
                eff_str = f"{eff:+.4f}" if not np.isnan(eff) else "N/A"
                print(
                    f"  {CONDITION_LABELS[condition]:<20} L{layer:>3} "
                    f"{mult:>6.2f} {eff_str:>8} {ci_str:>18} {n:>8}"
                )

    # Parse rates
    print("\n" + "-" * 80)
    print(f"{'Condition':<22} {'Layer':>5} {'|mult|':>6} {'Parse rate':>10} {'N total':>8}")
    print("-" * 80)
    for condition in conditions_ordered:
        available_layers = DIFFERENTIAL_LAYERS if condition == "task_mean" else ALL_LAYERS
        for layer in available_layers:
            multipliers = DIFFERENTIAL_MULTIPLIERS if condition == "task_mean" else ISOLATED_MULTIPLIERS
            for mult in multipliers:
                rate, n = compute_parse_rate(all_records, condition, layer, mult)
                rate_str = f"{rate:.4f}" if not np.isnan(rate) else "N/A"
                print(
                    f"  {CONDITION_LABELS[condition]:<20} L{layer:>3} "
                    f"{mult:>6.2f} {rate_str:>10} {n:>8}"
                )

    # Per-pair correlations for differential vs kv_cache at L25/L32
    print("\n" + "-" * 80)
    print("Per-pair correlation: differential vs KV cache steering effect")
    print("-" * 80)
    for layer in DIFFERENTIAL_LAYERS:
        for mult in ISOLATED_MULTIPLIERS:
            diff_pos = compute_per_pair_pa(all_records, "task_mean", layer, mult)
            diff_neg = compute_per_pair_pa(all_records, "task_mean", layer, -mult)
            kv_pos = compute_per_pair_pa(all_records, "kv_cache_v_single", layer, mult)
            kv_neg = compute_per_pair_pa(all_records, "kv_cache_v_single", layer, -mult)

            diff_pairs = set(diff_pos.keys()) & set(diff_neg.keys())
            kv_pairs = set(kv_pos.keys()) & set(kv_neg.keys())
            common = sorted(diff_pairs & kv_pairs)

            if len(common) < 3:
                print(f"  L{layer}, ±{mult}: insufficient data (n={len(common)})")
                continue

            diff_eff = np.array([diff_pos[p] - diff_neg[p] for p in common])
            kv_eff = np.array([kv_pos[p] - kv_neg[p] for p in common])
            if np.std(diff_eff) > 1e-10 and np.std(kv_eff) > 1e-10:
                r_val, p_val = stats.pearsonr(diff_eff, kv_eff)
                print(f"  L{layer}, ±{mult}: r={r_val:.3f}, p={p_val:.3g}, n={len(common)}")
            else:
                print(f"  L{layer}, ±{mult}: constant values (n={len(common)})")

    print("\n" + "=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_all_data()

    # Merge isolated and differential into single list for unified analysis
    all_records = data["isolated"] + data["differential"]
    baseline_records = data["baseline"]

    print(f"\nTotal analysis records: {len(all_records)} + {len(baseline_records)} baseline")

    # Print summary statistics
    print_summary(all_records, baseline_records)

    # Generate plots
    print("\n=== Generating plots ===")
    plot_steering_effect_comparison(all_records, baseline_records)
    plot_dose_response(all_records, baseline_records)
    plot_per_pair_scatter(all_records)
    plot_parse_rate_heatmap(all_records)
    plot_layer_comparison(all_records)

    print("\nDone.")


if __name__ == "__main__":
    main()
