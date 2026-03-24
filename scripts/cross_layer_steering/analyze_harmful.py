"""Analyze cross-layer steering on harmful pairs.

Produces:
1. Dose-response by pair type (probe_L25 only): 2x3 grid
2. Compliance breakdown by pair type
3. Harmful vs benign comparison (3 panels, one per layer)
4. Summary tables
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HARMFUL_CHECKPOINT = Path("experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl")
HARMFUL_PAIRS = Path("experiments/steering/cross_layer_harmful/pairs_200.json")
BENIGN_CHECKPOINT = Path("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl")
ASSETS = Path("experiments/steering/cross_layer_harmful/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

LAYERS = [20, 25, 30]
PAIR_TYPES = ["harmful_benign", "harmful_harmful"]
COMPLIANCE_CATEGORIES = ["full_comply", "caveat_comply", "hard_refusal", "incoherent"]
COMPLIANCE_COLORS = {
    "full_comply": "#22c55e",
    "caveat_comply": "#facc15",
    "hard_refusal": "#ef4444",
    "incoherent": "#94a3b8",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_compliance(row: dict) -> str:
    return row.get("compliance", "error")


# ── Load data ──

harmful_rows = load_jsonl(HARMFUL_CHECKPOINT)
pairs = json.loads(HARMFUL_PAIRS.read_text())
pair_lookup = {p["pair_id"]: p for p in pairs}

benign_rows = load_jsonl(BENIGN_CHECKPOINT)

# Filter to probe_L25 only for harmful
harmful_L25 = [r for r in harmful_rows if r["condition"] == "probe_L25"]
print(f"Harmful rows (probe_L25): {len(harmful_L25)}")

# Benign is already probe_L25 only; filter to layers 20, 25, 30
benign_filtered = [r for r in benign_rows if r["layer"] in LAYERS]
print(f"Benign rows (layers 20/25/30): {len(benign_filtered)}")


def pair_type_for_row(row: dict) -> str:
    return pair_lookup[row["pair_id"]]["pair_type"]


# ── Helper: aggregate P(chose A) and refusal rate by groups ──

def aggregate_by_groups(
    rows: list[dict],
    group_fn,
) -> dict[tuple, dict]:
    """Group rows, compute P(chose A), refusal rate, compliance breakdown."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = group_fn(r)
        buckets[key].append(r)

    results = {}
    for key, bucket in sorted(buckets.items()):
        n_total = len(bucket)
        # Compliance breakdown
        compliance_counts = defaultdict(int)
        for r in bucket:
            compliance_counts[get_compliance(r)] += 1

        # P(chose A) among valid choices (a or b)
        valid = [r for r in bucket if r["choice_original"] in ("a", "b")]
        n_valid = len(valid)
        n_chose_a = sum(1 for r in valid if r["choice_original"] == "a")
        p_a = n_chose_a / n_valid if n_valid > 0 else float("nan")

        # Refusal rate (hard_refusal / total)
        n_refusal = compliance_counts.get("hard_refusal", 0)
        refusal_rate = n_refusal / n_total if n_total > 0 else 0.0

        # Incoherent rate
        n_incoherent = compliance_counts.get("incoherent", 0) + compliance_counts.get("error", 0)
        incoherent_rate = n_incoherent / n_total if n_total > 0 else 0.0

        results[key] = {
            "p_a": p_a,
            "n_total": n_total,
            "n_valid": n_valid,
            "refusal_rate": refusal_rate,
            "incoherent_rate": incoherent_rate,
            "compliance_counts": dict(compliance_counts),
        }

    return results


# ══════════════════════════════════════════════════════════════
# Plot 1: Dose-response by pair type (2 rows x 3 cols)
# ══════════════════════════════════════════════════════════════

print("\n── Plot 1: Dose-response by pair type ──")

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True, sharex=True)

for row_idx, pair_type in enumerate(PAIR_TYPES):
    pt_rows = [r for r in harmful_L25 if pair_type_for_row(r) == pair_type]

    for col_idx, layer in enumerate(LAYERS):
        ax = axes[row_idx, col_idx]
        layer_rows = [r for r in pt_rows if r["layer"] == layer]

        agg = aggregate_by_groups(
            layer_rows,
            lambda r: (r["signed_multiplier"],),
        )

        mults = sorted(agg.keys())
        x = [m[0] for m in mults]
        y_pa = [agg[m]["p_a"] for m in mults]
        y_refusal = [agg[m]["refusal_rate"] for m in mults]
        y_incoherent = [agg[m]["incoherent_rate"] for m in mults]

        ax.plot(x, y_pa, "o-", color="#2563eb", linewidth=2, markersize=4,
                label="P(completed steered task)", zorder=3)
        ax.fill_between(x, 0, y_refusal, color="#ef4444", alpha=0.25,
                        label="Refusal rate")
        ax.fill_between(x, 0, y_incoherent, color="#94a3b8", alpha=0.25,
                        label="Incoherent rate")

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.12, 0.12)

        if row_idx == 0:
            ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        if col_idx == 0:
            label_text = pair_type.replace("_", " ").title()
            ax.set_ylabel(f"{label_text}\nP(completed steered task) / Rate", fontsize=10)
        if row_idx == 1:
            ax.set_xlabel("Coefficient (\u00d7 mean norm)", fontsize=10)

        n_total = sum(agg[m]["n_total"] for m in mults)
        ax.text(0.02, 0.98, f"n={n_total}", transform=ax.transAxes,
                fontsize=8, va="top", ha="left", color="gray")

        if row_idx == 0 and col_idx == 2:
            ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Dose-response by pair type (probe L25)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
path = ASSETS / "plot_032326_harmful_dose_response.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()


# ══════════════════════════════════════════════════════════════
# Plot 2: Compliance breakdown by pair type
# ══════════════════════════════════════════════════════════════

print("\n── Plot 2: Compliance breakdown ──")

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True, sharex=True)

for row_idx, pair_type in enumerate(PAIR_TYPES):
    pt_rows = [r for r in harmful_L25 if pair_type_for_row(r) == pair_type]

    for col_idx, layer in enumerate(LAYERS):
        ax = axes[row_idx, col_idx]
        layer_rows = [r for r in pt_rows if r["layer"] == layer]

        agg = aggregate_by_groups(
            layer_rows,
            lambda r: (r["signed_multiplier"],),
        )

        mults = sorted(agg.keys())
        x_labels = [f"{m[0]:+.2f}" for m in mults]
        x_pos = np.arange(len(mults))

        bottoms = np.zeros(len(mults))
        for cat in COMPLIANCE_CATEGORIES:
            heights = []
            for m in mults:
                cc = agg[m]["compliance_counts"]
                total = agg[m]["n_total"]
                # Also count "error" rows in incoherent for display
                if cat == "incoherent":
                    count = cc.get("incoherent", 0) + cc.get("error", 0)
                else:
                    count = cc.get(cat, 0)
                heights.append(count / total if total > 0 else 0)
            heights = np.array(heights)
            ax.bar(x_pos, heights, bottom=bottoms, color=COMPLIANCE_COLORS[cat],
                   label=cat.replace("_", " ") if row_idx == 0 and col_idx == 0 else None,
                   width=0.7, edgecolor="white", linewidth=0.3)
            bottoms += heights

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45)
        ax.set_ylim(0, 1)

        if row_idx == 0:
            ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        if col_idx == 0:
            label_text = pair_type.replace("_", " ").title()
            ax.set_ylabel(f"{label_text}\nFraction", fontsize=10)
        if row_idx == 1:
            ax.set_xlabel("Coefficient (\u00d7 mean norm)", fontsize=10)

axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("Compliance breakdown by pair type (probe L25)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
path = ASSETS / "plot_032326_harmful_compliance.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()


# ══════════════════════════════════════════════════════════════
# Plot 3: Harmful vs benign comparison
# ══════════════════════════════════════════════════════════════

print("\n── Plot 3: Harmful vs benign comparison ──")

# Aggregate benign (probe_L25)
benign_agg = aggregate_by_groups(
    benign_filtered,
    lambda r: (r["layer"], r["signed_multiplier"]),
)

# Aggregate harmful by pair type
harmful_hb_L25 = [r for r in harmful_L25 if pair_type_for_row(r) == "harmful_benign"]
harmful_hh_L25 = [r for r in harmful_L25 if pair_type_for_row(r) == "harmful_harmful"]

harmful_hb_agg = aggregate_by_groups(
    harmful_hb_L25,
    lambda r: (r["layer"], r["signed_multiplier"]),
)
harmful_hh_agg = aggregate_by_groups(
    harmful_hh_L25,
    lambda r: (r["layer"], r["signed_multiplier"]),
)

datasets = [
    ("Benign", benign_agg, "#2563eb", "-"),
    ("Harmful-Benign", harmful_hb_agg, "#f97316", "--"),
    ("Harmful-Harmful", harmful_hh_agg, "#dc2626", "-."),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for col_idx, layer in enumerate(LAYERS):
    ax = axes[col_idx]

    for label, agg, color, ls in datasets:
        # Get multipliers for this layer
        mults = sorted(k for k in agg if k[0] == layer)
        if not mults:
            continue
        x = [m[1] for m in mults]
        y = [agg[m]["p_a"] for m in mults]
        ax.plot(x, y, ls, color=color, linewidth=2, marker="o", markersize=4,
                label=label, zorder=3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.12, 0.12)
    ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Signed multiplier", fontsize=10)

    if col_idx == 0:
        ax.set_ylabel("P(completed steered task)", fontsize=11)
    if col_idx == 2:
        ax.legend(fontsize=9, loc="upper left")

fig.suptitle("Dose-response: Benign vs Harmful pairs (probe L25)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
path = ASSETS / "plot_032326_harmful_vs_benign.png"
fig.savefig(path, dpi=150)
print(f"Saved {path}")
plt.close()


# ══════════════════════════════════════════════════════════════
# Summary tables
# ══════════════════════════════════════════════════════════════

print("\n── Summary Tables ──")
print("\n" + "=" * 90)
print("PROBE L25 — Harmful pairs")
print("=" * 90)

for pair_type in PAIR_TYPES:
    pt_rows = [r for r in harmful_L25 if pair_type_for_row(r) == pair_type]

    print(f"\n{'─' * 90}")
    print(f"  {pair_type} ({len(set(r['pair_id'] for r in pt_rows))} pairs)")
    print(f"{'─' * 90}")
    print(f"  {'Layer':>5} {'Mult':>7} {'P(A)':>7} {'Refusal':>8} {'Incoher':>8} {'N_valid':>8} {'N_total':>8}")

    for layer in LAYERS:
        layer_rows = [r for r in pt_rows if r["layer"] == layer]
        agg = aggregate_by_groups(
            layer_rows,
            lambda r: (r["signed_multiplier"],),
        )
        for m in sorted(agg.keys()):
            d = agg[m]
            print(f"  {layer:>5} {m[0]:>+7.2f} {d['p_a']:>7.3f} {d['refusal_rate']:>8.3f} "
                  f"{d['incoherent_rate']:>8.3f} {d['n_valid']:>8} {d['n_total']:>8}")

print(f"\n{'=' * 90}")
print("COMPARISON: Benign (probe L25, layers 20/25/30)")
print("=" * 90)
print(f"  {'Layer':>5} {'Mult':>7} {'P(A)':>7} {'Refusal':>8} {'Incoher':>8} {'N_valid':>8} {'N_total':>8}")

for layer in LAYERS:
    mults = sorted(k for k in benign_agg if k[0] == layer)
    for m in mults:
        d = benign_agg[m]
        print(f"  {layer:>5} {m[1]:>+7.2f} {d['p_a']:>7.3f} {d['refusal_rate']:>8.3f} "
              f"{d['incoherent_rate']:>8.3f} {d['n_valid']:>8} {d['n_total']:>8}")

print("\nDone!")
