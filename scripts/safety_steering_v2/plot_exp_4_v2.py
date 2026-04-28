"""Plots and bootstrap CIs for safety_steering_v2 / exp_4_v2."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/safety_steering_exp4v2")
EXP = ROOT / "experiments/safety_steering_v2/exp_4_v2"
ASSETS = EXP / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

AGG = json.loads((EXP / "aggregated.json").read_text())
PER = json.loads((EXP / "per_scenario.json").read_text())

COEFS = [-0.05, -0.03, 0.0, 0.03, 0.05]
COEFS_NONZERO = [-0.05, -0.03, 0.03, 0.05]


def agg_rate(variant: str, bin_: str, condition: str, c: float) -> float:
    """Disclosure_specific rate from aggregated.json. c=0 -> no_steering."""
    if c == 0.0:
        return AGG[variant][bin_]["no_steering"]["0.0"]["rate_specific"]
    return AGG[variant][bin_][condition][f"{c}"]["rate_specific"]


def plot_critical_info_dose_response() -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    series = [
        ("ethical", "isolated", "red", "-", "ethical / isolated"),
        ("ethical", "distributed", "darkorange", "-", "ethical / distributed"),
        ("benign_twin", "isolated", "grey", "--", "benign_twin / isolated"),
        ("benign_twin", "distributed", "black", "--", "benign_twin / distributed"),
    ]
    for variant, bin_, color, ls, label in series:
        ys = [agg_rate(variant, bin_, "critical_info_only", c) for c in COEFS]
        ax.plot(COEFS, ys, color=color, linestyle=ls, marker="o", linewidth=1.8, label=label)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4, linestyle=":",
               label="c=0 (no_steering baseline)")
    ax.set_xlabel("steering coefficient c")
    ax.set_ylabel("disclosure_specific rate")
    ax.set_xticks(COEFS)
    ax.set_ylim(0, 1.0)
    ax.set_title("Disclosure suppression under critical_info_only steering")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "plot_042826_critical_info_dose_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_condition_distributed_ethical() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    conditions = [
        ("critical_info_only", "tab:blue", "-", "critical_info_only"),
        ("generation_only", "tab:green", "-", "generation_only"),
        ("critical_info_plus_generation", "tab:red", "-", "critical_info_plus_generation"),
    ]
    for ax, bin_, n_per_cell in [(axes[0], "isolated", 45), (axes[1], "distributed", 25)]:
        baseline = agg_rate("ethical", bin_, "no_steering", 0.0)
        for cond, color, ls, label in conditions:
            ys = [baseline if c == 0.0 else agg_rate("ethical", bin_, cond, c) for c in COEFS]
            ax.plot(COEFS, ys, color=color, linestyle=ls, marker="o", linewidth=1.8, label=label)
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                   label=f"no_steering (={baseline:.2f})")
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.4, linestyle=":")
        ax.set_xlabel("steering coefficient c")
        ax.set_xticks(COEFS)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{bin_} (n={n_per_cell}/cell)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("disclosure_specific rate")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Per-condition dose response — ethical variant", y=0.995)
    fig.tight_layout()
    out = ASSETS / "plot_042826_per_condition_distributed_ethical.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def scenario_specific_rate(scenario_dict: dict, scenario_ids: list[str]) -> float:
    """Aggregate disclosure_specific over a multiset of scenario_ids using their per-scenario data.
    Each scenario contributes n trials at rate r; combined rate = sum(r*n)/sum(n)."""
    num = 0.0
    denom = 0
    for sid in scenario_ids:
        entry = scenario_dict[sid]
        num += entry["rate_specific"] * entry["n"]
        denom += entry["n"]
    return num / denom


def bootstrap_delta(bin_: str, condition: str, c: float, n_boot: int = 1000,
                    rng: np.random.Generator | None = None) -> tuple[float, float, float, float, float]:
    """Returns (baseline_rate, suppressed_rate, delta_point, ci_lo, ci_hi).
    Bootstrap resamples scenarios within bin and recomputes both baseline and suppressed,
    taking the difference (baseline - suppressed)."""
    if rng is None:
        rng = np.random.default_rng(42)
    base = PER["ethical"][bin_]["no_steering"]["0.0"]
    supp = PER["ethical"][bin_][condition][f"{c}"]
    sids = sorted(base.keys())
    assert sorted(supp.keys()) == sids
    base_pt = scenario_specific_rate(base, sids)
    supp_pt = scenario_specific_rate(supp, sids)
    delta_pt = base_pt - supp_pt
    deltas = np.empty(n_boot)
    n = len(sids)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [sids[j] for j in idx]
        b = scenario_specific_rate(base, resampled)
        s = scenario_specific_rate(supp, resampled)
        deltas[i] = b - s
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return base_pt, supp_pt, delta_pt, float(lo), float(hi)


def plot_isolated_vs_distributed_suppression(rows: list[dict]) -> Path:
    cells = [
        ("critical_info_only", 0.03),
        ("critical_info_only", 0.05),
        ("generation_only", 0.05),
        ("critical_info_plus_generation", 0.05),
    ]
    labels = [f"{cond}\n@ c=+{c:.2f}" for cond, c in cells]
    iso_d, iso_lo, iso_hi = [], [], []
    dist_d, dist_lo, dist_hi = [], [], []
    for cond, c in cells:
        for r in rows:
            if r["condition"] == cond and r["multiplier"] == c and r["bin"] == "isolated":
                iso_d.append(r["delta"]); iso_lo.append(r["delta"] - r["ci_lo"]); iso_hi.append(r["ci_hi"] - r["delta"])
            if r["condition"] == cond and r["multiplier"] == c and r["bin"] == "distributed":
                dist_d.append(r["delta"]); dist_lo.append(r["delta"] - r["ci_lo"]); dist_hi.append(r["ci_hi"] - r["delta"])
    x = np.arange(len(cells))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, iso_d, width, yerr=[iso_lo, iso_hi], capsize=4,
           color="#5B8FF9", label="isolated (n=45/cell)")
    ax.bar(x + width / 2, dist_d, width, yerr=[dist_lo, dist_hi], capsize=4,
           color="#F6BD16", label="distributed (n=25/cell)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\Delta$ disclosure_specific (baseline $-$ suppressed)")
    ax.set_title("Suppression of disclosure_specific (Δ from c=0 baseline)\nethical variant, by bin · 95% bootstrap CI (1000 resamples over scenarios)")
    ax.set_ylim(min(0, min(iso_d + dist_d) - 0.1), 1.0)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = ASSETS / "plot_042826_isolated_vs_distributed_suppression.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_bootstrap_rows() -> list[dict]:
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    cells = [
        ("critical_info_only", 0.03),
        ("critical_info_only", 0.05),
        ("generation_only", 0.05),
        ("critical_info_plus_generation", 0.05),
    ]
    for cond, c in cells:
        for bin_ in ("isolated", "distributed"):
            base, supp, delta, lo, hi = bootstrap_delta(bin_, cond, c, n_boot=1000, rng=rng)
            rows.append({
                "condition": cond, "multiplier": c, "bin": bin_,
                "baseline": base, "suppressed": supp,
                "delta": delta, "ci_lo": lo, "ci_hi": hi,
            })
    return rows


def format_markdown(rows: list[dict]) -> str:
    header = ("| condition | multiplier | bin | baseline_specific | suppressed_specific | "
              "Δ | 95% CI (low, high) |\n"
              "|---|---|---|---|---|---|---|")
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['condition']} | +{r['multiplier']:.2f} | {r['bin']} | "
            f"{r['baseline']:.3f} | {r['suppressed']:.3f} | {r['delta']:+.3f} | "
            f"({r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}) |"
        )
    return "\n".join(lines)


def main():
    p1 = plot_critical_info_dose_response()
    p2 = plot_per_condition_distributed_ethical()
    rows = build_bootstrap_rows()
    p3 = plot_isolated_vs_distributed_suppression(rows)
    md = format_markdown(rows)
    print(f"PLOT: {p1}")
    print(f"PLOT: {p2}")
    print(f"PLOT: {p3}")
    print("\n--- BOOTSTRAP TABLE ---\n")
    print(md)


if __name__ == "__main__":
    main()
