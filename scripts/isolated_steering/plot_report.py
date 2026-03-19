"""Generate plots for the isolated KV cache steering report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT = Path("experiments/steering/isolated_steering/checkpoint_kv_l25_all62_partial.jsonl")
ASSETS = Path("experiments/steering/isolated_steering/assets")
ASSETS.mkdir(exist_ok=True)

rows = [json.loads(l) for l in open(CHECKPOINT)]
mults = sorted(set(r["multiplier"] for r in rows))

# ── Helpers ──

def get_valid(rows, mult, ordering=None):
    subset = [r for r in rows if r["multiplier"] == mult]
    if ordering is not None:
        subset = [r for r in subset if r["ordering"] == ordering]
    return [r for r in subset if r["choice_original"] in ("a", "b")]


def p_chose_steered(rows, mult, ordering=None):
    valid = get_valid(rows, mult, ordering)
    if not valid:
        return np.nan, 0
    chose_a = sum(1 for r in valid if r["choice_original"] == "a")
    return chose_a / len(valid), len(valid)


def bootstrap_ci(rows, mult, ordering=None, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    valid = get_valid(rows, mult, ordering)
    choices = np.array([1 if r["choice_original"] == "a" else 0 for r in valid])
    means = [rng.choice(choices, size=len(choices), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def incoherent_rate(rows, mult, ordering=None):
    subset = [r for r in rows if r["multiplier"] == mult]
    if ordering is not None:
        subset = [r for r in subset if r["ordering"] == ordering]
    bad = sum(1 for r in subset if r["choice_original"] not in ("a", "b"))
    return bad / len(subset) if subset else np.nan


# ── Plot 1: Pooled dose-response with incoherent rate ──

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

ps = [p_chose_steered(rows, m)[0] * 100 for m in mults]
cis = [bootstrap_ci(rows, m) for m in mults]
ci_lo = [c[0] * 100 for c in cis]
ci_hi = [c[1] * 100 for c in cis]
incoh = [incoherent_rate(rows, m) * 100 for m in mults]

ax1.plot(mults, ps, "o-", color="#2563eb", linewidth=2, markersize=7, zorder=3, label="P(chose steered task)")
ax1.fill_between(mults, ci_lo, ci_hi, color="#2563eb", alpha=0.15, zorder=2)
ax1.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax1.set_xlabel("Multiplier (coefficient = 35,708 × m)", fontsize=11)
ax1.set_ylabel("P(chose steered task) %", fontsize=11, color="#2563eb")
ax1.set_ylim(0, 100)
ax1.tick_params(axis="y", labelcolor="#2563eb")

ax2.bar(mults, incoh, width=0.0012, color="#dc2626", alpha=0.35, zorder=1, label="Incoherent rate")
ax2.set_ylabel("Incoherent response rate %", fontsize=11, color="#dc2626")
ax2.set_ylim(0, 50)
ax2.tick_params(axis="y", labelcolor="#dc2626")

ax1.set_title("KV cache V-only steering: dose-response (pooled)", fontsize=12, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(ASSETS / "plot_031826_dose_response_pooled.png", dpi=150)
plt.close()
print("Saved dose_response_pooled")

# ── Plot 2: Dose-response split by ordering ──

fig, ax = plt.subplots(figsize=(8, 5))

for ordering, label, color, marker in [
    (0, "Steered task first", "#2563eb", "o"),
    (1, "Steered task second", "#dc2626", "s"),
]:
    ps = [p_chose_steered(rows, m, ordering)[0] * 100 for m in mults]
    cis = [bootstrap_ci(rows, m, ordering) for m in mults]
    ci_lo = [c[0] * 100 for c in cis]
    ci_hi = [c[1] * 100 for c in cis]
    ax.plot(mults, ps, f"{marker}-", color=color, linewidth=2, markersize=7, label=label)
    ax.fill_between(mults, ci_lo, ci_hi, color=color, alpha=0.12)

ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Multiplier (coefficient = 35,708 × m)", fontsize=11)
ax.set_ylabel("P(chose steered task) %", fontsize=11)
ax.set_ylim(0, 100)
ax.set_title("KV cache V-only steering: effect by presentation order", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(ASSETS / "plot_031826_dose_response_by_ordering.png", dpi=150)
plt.close()
print("Saved dose_response_by_ordering")

# ── Plot 3: Causal swing bar chart ──

fig, ax = plt.subplots(figsize=(6, 4))

labels = ["Pooled", "Steered task\nfirst", "Steered task\nsecond"]
swings = []
ci_ranges = []

rng = np.random.RandomState(42)
for ordering in [None, 0, 1]:
    neg = get_valid(rows, -0.003, ordering)
    pos = get_valid(rows, 0.003, ordering)
    neg_c = np.array([1 if r["choice_original"] == "a" else 0 for r in neg])
    pos_c = np.array([1 if r["choice_original"] == "a" else 0 for r in pos])
    boots = []
    for _ in range(10000):
        boots.append(rng.choice(pos_c, len(pos_c), replace=True).mean()
                     - rng.choice(neg_c, len(neg_c), replace=True).mean())
    boots = np.array(boots) * 100
    swings.append(np.mean(boots))
    ci_ranges.append((np.mean(boots) - np.percentile(boots, 2.5),
                      np.percentile(boots, 97.5) - np.mean(boots)))

colors = ["#6b7280", "#2563eb", "#dc2626"]
x = np.arange(len(labels))
yerr = np.array(ci_ranges).T

ax.bar(x, swings, color=colors, width=0.6, zorder=3)
ax.errorbar(x, swings, yerr=yerr, fmt="none", ecolor="black", capsize=5, zorder=4)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Causal swing (pp)", fontsize=11)
ax.set_title("Causal swing at m=±0.003\n(P at +0.003 minus P at −0.003)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 50)

for i, s in enumerate(swings):
    ax.text(i, s + ci_ranges[i][1] + 1.5, f"{s:.1f}pp", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(ASSETS / "plot_031826_causal_swing.png", dpi=150)
plt.close()
print("Saved causal_swing")
