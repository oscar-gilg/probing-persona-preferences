"""Plot coherence rates by steering coefficient for hook patching and KV steering."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HOOK_COHERENCE = Path(
    "experiments/steering/isolated_steering/full_run/coherence_hook_L25_500.jsonl"
)
KV_COHERENCE = Path(
    "experiments/steering/isolated_steering/full_run/coherence_kv_recompute.jsonl"
)
ASSETS = Path("experiments/steering/isolated_steering/full_run/assets")
ASSETS.mkdir(parents=True, exist_ok=True)


def load_coherence(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_rates(
    rows: list[dict],
) -> dict[tuple[str, float], tuple[float, int]]:
    """Returns {(condition, abs_mult): (coherence_rate, n)}."""
    buckets: dict[tuple[str, float], list[bool]] = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["abs_multiplier"])
        buckets[key].append(r["coherent"])
    return {
        key: (sum(flags) / len(flags), len(flags))
        for key, flags in buckets.items()
    }


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, centre - margin), min(1, centre + margin)


# --- Load data ---
hook_rows = load_coherence(HOOK_COHERENCE)
kv_rows = load_coherence(KV_COHERENCE)

hook_rates = compute_rates(hook_rows)
kv_rates = compute_rates(kv_rows)

# --- Hook patching: grouped bar chart (splice vs recompute) ---
hook_coefs = sorted({m for _, m in hook_rates.keys()})

CONDITIONS = [
    ("hook_patching", "Splice only", "#60a5fa"),
    ("hook_patching_recompute", "Splice + recompute", "#2563eb"),
]

fig, ax = plt.subplots(figsize=(9, 5))

bar_width = 0.35
x = np.arange(len(hook_coefs))

for i, (condition, label, color) in enumerate(CONDITIONS):
    rates = []
    ns = []
    ci_lower = []
    ci_upper = []
    for coef in hook_coefs:
        rate, n = hook_rates.get((condition, coef), (0, 0))
        rates.append(rate)
        ns.append(n)
        lo, hi = wilson_ci(rate, n) if n > 0 else (0, 0)
        ci_lower.append(max(0, rate - lo))
        ci_upper.append(max(0, hi - rate))

    offset = (i - 0.5) * bar_width
    bars = ax.bar(
        x + offset,
        rates,
        bar_width,
        label=label,
        color=color,
        alpha=0.85,
        yerr=[ci_lower, ci_upper],
        capsize=3,
        error_kw={"linewidth": 1},
    )
    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666",
        )

ax.set_xticks(x)
ax.set_xticklabels([f"{c}" for c in hook_coefs])
ax.set_xlabel("Steering coefficient (|coef|, fraction of mean L25 norm)")
ax.set_ylabel("Coherence rate")
ax.set_ylim(0, 1.12)
ax.set_title("Hook patching: generation coherence by steering strength")
ax.legend(loc="lower left")
ax.axhline(y=1.0, color="#ccc", linewidth=0.5, zorder=0)

plt.tight_layout()
hook_path = ASSETS / "plot_031926_hook_coherence_by_strength.png"
fig.savefig(hook_path, dpi=150)
print(f"Saved {hook_path}")
plt.close()

# --- Combined plot: hook patching + KV on same axes ---
fig, ax = plt.subplots(figsize=(9, 5))

# KV rates
kv_conditions = [
    ("kv_steering", "KV only", "#f97316", "o"),
    ("kv_steering_recompute", "KV + recompute", "#dc2626", "s"),
]

# Plot hook patching as lines
for condition, label, color in CONDITIONS:
    coefs = sorted(hook_coefs)
    rates = [hook_rates.get((condition, c), (0, 0))[0] for c in coefs]
    ns = [hook_rates.get((condition, c), (0, 0))[1] for c in coefs]
    ci_lo = []
    ci_hi = []
    for r, n in zip(rates, ns):
        lo, hi = wilson_ci(r, n) if n > 0 else (0, 0)
        ci_lo.append(max(0, r - lo))
        ci_hi.append(max(0, hi - r))
    ax.errorbar(
        coefs, rates,
        yerr=[ci_lo, ci_hi],
        marker="o", label=f"Hook: {label}", color=color,
        capsize=3, linewidth=1.5, markersize=5,
    )

# Plot KV as separate points (different x-values)
for condition, label, color, marker in kv_conditions:
    kv_coefs_cond = sorted(m for c, m in kv_rates.keys() if c == condition)
    rates = [kv_rates[(condition, c)][0] for c in kv_coefs_cond]
    ns = [kv_rates[(condition, c)][1] for c in kv_coefs_cond]
    ci_lo = []
    ci_hi = []
    for r, n in zip(rates, ns):
        lo, hi = wilson_ci(r, n) if n > 0 else (0, 0)
        ci_lo.append(max(0, r - lo))
        ci_hi.append(max(0, hi - r))
    ax.errorbar(
        kv_coefs_cond, rates,
        yerr=[ci_lo, ci_hi],
        marker=marker, label=f"KV: {label}", color=color,
        capsize=3, linewidth=1.5, markersize=7,
    )

ax.set_xlabel("Steering coefficient (fraction of mean norm)")
ax.set_ylabel("Coherence rate")
ax.set_ylim(0, 1.08)
ax.set_title("Generation coherence across steering methods and strengths")
ax.legend(loc="lower left", fontsize=9)
ax.axhline(y=1.0, color="#ccc", linewidth=0.5, zorder=0)

plt.tight_layout()
combined_path = ASSETS / "plot_031926_coherence_all_methods.png"
fig.savefig(combined_path, dpi=150)
print(f"Saved {combined_path}")
plt.close()

# --- Print summary table ---
print("\nHook patching coherence:")
print(f"{'Condition':<30} {'|coef|':>8} {'Rate':>8} {'n':>5}")
print("-" * 55)
for (condition, coef), (rate, n) in sorted(hook_rates.items()):
    print(f"{condition:<30} {coef:>8.2f} {rate:>7.1%} {n:>5}")

print("\nKV steering coherence:")
print(f"{'Condition':<30} {'|coef|':>8} {'Rate':>8} {'n':>5}")
print("-" * 55)
for (condition, coef), (rate, n) in sorted(kv_rates.items()):
    print(f"{condition:<30} {coef:>8.3f} {rate:>7.1%} {n:>5}")
