"""Overlay plot + extended bootstrap Δ table for the localisation control.

Reads:
- `experiments/safety_steering_v2/exp_4_v2/aggregated.json` and
  `per_scenario.json` (exp_4_v2's `critical_info_only` rows on isolated bin).
- `experiments/safety_steering_v2/exp_4_v2/localisation_control/aggregated.json`
  and `per_scenario.json` (this run's `non_critical_only`).

Produces:
- `assets/plot_<date>_critical_vs_noncritical_dose_response.png`
- `assets/plot_<date>_critical_vs_noncritical_suppression.png`
- prints an extended Δ table to stdout (mirrors exp_4_v2's report table)
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/safety_steering_v2/exp_4_v2"
LOC = EXP / "localisation_control"
ASSETS = LOC / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

DATE_TAG = _dt.date.today().strftime("%m%d%y")

AGG_CRIT = json.loads((EXP / "aggregated.json").read_text())
PER_CRIT = json.loads((EXP / "per_scenario.json").read_text())
AGG_NC = json.loads((LOC / "aggregated.json").read_text())
PER_NC = json.loads((LOC / "per_scenario.json").read_text())

COEFS_NONZERO = [-0.05, -0.03, 0.03, 0.05]
COEFS_WITH_ZERO = [-0.05, -0.03, 0.0, 0.03, 0.05]

# Restrict per-scenario data to isolated-bin scenarios only — the 9 we ran.
ISOLATED_SCENARIOS = sorted(set(PER_NC["ethical"]["isolated"]["non_critical_only"]["0.05"].keys()))


def agg_specific(agg: dict, variant: str, bin_: str, condition: str, c: float) -> float:
    """rate_specific from aggregated.json. c=0 -> always no_steering bucket."""
    if c == 0.0:
        return agg[variant][bin_]["no_steering"]["0.0"]["rate_specific"]
    return agg[variant][bin_][condition][f"{c}"]["rate_specific"]


def per_specific_isolated(per: dict, variant: str, condition: str, c: float, scenarios: list[str]) -> float:
    """Aggregate rate_specific over the given scenarios, weighted by trial counts.

    `c=0` always reads the no_steering / 0.0 bucket; otherwise reads (condition, c).
    Restricts to scenarios that exist in the per-scenario dict (both critical-info and
    non-critical conditions cover the same isolated subset).
    """
    cond = "no_steering" if c == 0.0 else condition
    coef_key = "0.0" if c == 0.0 else f"{c}"
    entry = per[variant]["isolated"][cond][coef_key]
    num, denom = 0.0, 0
    for sid in scenarios:
        if sid not in entry:
            continue
        num += entry[sid]["rate_specific"] * entry[sid]["n"]
        denom += entry[sid]["n"]
    return num / denom if denom else 0.0


def bootstrap_delta_isolated(per_baseline: dict, per_supp: dict, variant: str, condition: str,
                              c: float, n_boot: int = 1000, rng=None) -> tuple[float, float, float, float, float]:
    """Bootstrap-resample isolated-bin scenarios; return (baseline, suppressed, Δ, ci_lo, ci_hi).

    `per_baseline` is exp_4_v2's per_scenario.json (provides no_steering @ c=0).
    `per_supp` is the per_scenario.json of the run that supplies the suppressed cell.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    base_entry = per_baseline[variant]["isolated"]["no_steering"]["0.0"]
    supp_entry = per_supp[variant]["isolated"][condition][f"{c}"]
    sids = sorted(set(base_entry.keys()) & set(supp_entry.keys()))
    if not sids:
        raise ValueError(f"no overlapping scenarios for {variant}/isolated/{condition}/{c}")

    def rate(entry, scen_list):
        num, den = 0.0, 0
        for s in scen_list:
            num += entry[s]["rate_specific"] * entry[s]["n"]
            den += entry[s]["n"]
        return num / den if den else 0.0

    base_pt = rate(base_entry, sids)
    supp_pt = rate(supp_entry, sids)
    delta_pt = base_pt - supp_pt
    deltas = np.empty(n_boot)
    n = len(sids)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [sids[j] for j in idx]
        b = rate(base_entry, sample)
        s = rate(supp_entry, sample)
        deltas[i] = b - s
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return base_pt, supp_pt, delta_pt, float(lo), float(hi)


def plot_dose_response_overlay() -> Path:
    """Overlay critical_info_only (solid, from exp_4_v2) and non_critical_only (dashed, this run)
    for ethical/isolated and benign_twin/isolated. Distributed-bin curves shown faded for context."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Faded distributed-bin lines from exp_4_v2 (context only)
    for variant, color in [("ethical", "darkorange"), ("benign_twin", "black")]:
        ys = [agg_specific(AGG_CRIT, variant, "distributed", "critical_info_only", c)
              for c in COEFS_WITH_ZERO]
        ax.plot(COEFS_WITH_ZERO, ys, color=color, alpha=0.25, marker="o", linewidth=1.2,
                label=f"{variant} / distributed / critical (exp_4_v2 context)")

    # Solid: critical_info_only on isolated bin, both variants
    for variant, color in [("ethical", "red"), ("benign_twin", "grey")]:
        ys = [agg_specific(AGG_CRIT, variant, "isolated", "critical_info_only", c)
              for c in COEFS_WITH_ZERO]
        ax.plot(COEFS_WITH_ZERO, ys, color=color, linestyle="-", marker="o", linewidth=2.0,
                label=f"{variant} / isolated / critical_info_only")

    # Dashed: non_critical_only on isolated bin, both variants. c=0 is the same baseline.
    for variant, color in [("ethical", "red"), ("benign_twin", "grey")]:
        ys = []
        for c in COEFS_WITH_ZERO:
            if c == 0.0:
                ys.append(agg_specific(AGG_CRIT, variant, "isolated", "no_steering", 0.0))
            else:
                ys.append(agg_specific(AGG_NC, variant, "isolated", "non_critical_only", c))
        ax.plot(COEFS_WITH_ZERO, ys, color=color, linestyle="--", marker="s", linewidth=2.0,
                label=f"{variant} / isolated / non_critical_only")

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4, linestyle=":",
               label="c=0 (no_steering baseline)")
    ax.set_xlabel("steering coefficient c")
    ax.set_ylabel("disclosure_specific rate")
    ax.set_xticks(COEFS_WITH_ZERO)
    ax.set_ylim(0, 1.0)
    ax.set_title("Localisation control — critical_info_only vs non_critical_only\n"
                 "(isolated bin, n=9 scenarios × 2 variants × 5 trials)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ASSETS / f"plot_{DATE_TAG}_critical_vs_noncritical_dose_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_suppression_bars(rows: list[dict]) -> Path:
    """Compare Δ on isolated bin between critical_info_only and non_critical_only at each c."""
    cells = [(0.05, "+0.05"), (0.03, "+0.03"), (-0.03, "-0.03"), (-0.05, "-0.05")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for ax, variant in zip(axes, ("ethical", "benign_twin")):
        crit_d, crit_lo, crit_hi = [], [], []
        nc_d, nc_lo, nc_hi = [], [], []
        labels = []
        for c, c_label in cells:
            for r in rows:
                if r["variant"] != variant or r["multiplier"] != c:
                    continue
                if r["condition"] == "critical_info_only":
                    crit_d.append(r["delta"]); crit_lo.append(r["delta"] - r["ci_lo"]); crit_hi.append(r["ci_hi"] - r["delta"])
                if r["condition"] == "non_critical_only":
                    nc_d.append(r["delta"]); nc_lo.append(r["delta"] - r["ci_lo"]); nc_hi.append(r["ci_hi"] - r["delta"])
            labels.append(f"c={c_label}")
        x = np.arange(len(cells))
        width = 0.38
        ax.bar(x - width/2, crit_d, width, yerr=[crit_lo, crit_hi], capsize=4,
               color="#5B8FF9", label="critical_info_only")
        ax.bar(x + width/2, nc_d, width, yerr=[nc_lo, nc_hi], capsize=4,
               color="#F6BD16", label="non_critical_only")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{variant} / isolated")
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\Delta$ disclosure_specific (baseline $-$ steered)")
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("Suppression Δ — critical_info_only vs non_critical_only (isolated bin, 9 scenarios, 1000 bootstraps)", y=1.0)
    fig.tight_layout()
    out = ASSETS / f"plot_{DATE_TAG}_critical_vs_noncritical_suppression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def build_bootstrap_rows() -> list[dict]:
    rng = np.random.default_rng(42)
    rows = []
    for variant in ("ethical", "benign_twin"):
        for c in COEFS_NONZERO:
            base, supp, delta, lo, hi = bootstrap_delta_isolated(
                PER_CRIT, PER_CRIT, variant, "critical_info_only", c, n_boot=1000, rng=rng,
            )
            rows.append({"variant": variant, "condition": "critical_info_only",
                         "multiplier": c, "baseline": base, "suppressed": supp,
                         "delta": delta, "ci_lo": lo, "ci_hi": hi})
            base, supp, delta, lo, hi = bootstrap_delta_isolated(
                PER_CRIT, PER_NC, variant, "non_critical_only", c, n_boot=1000, rng=rng,
            )
            rows.append({"variant": variant, "condition": "non_critical_only",
                         "multiplier": c, "baseline": base, "suppressed": supp,
                         "delta": delta, "ci_lo": lo, "ci_hi": hi})
    return rows


def format_markdown(rows: list[dict]) -> str:
    header = ("| variant | condition | c | baseline_specific | suppressed_specific | "
              "Δ | 95% CI |\n"
              "|---|---|---|---|---|---|---|")
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['condition']} | {r['multiplier']:+.2f} | "
            f"{r['baseline']:.3f} | {r['suppressed']:.3f} | {r['delta']:+.3f} | "
            f"({r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}) |"
        )
    return "\n".join(lines)


def main():
    rows = build_bootstrap_rows()
    p1 = plot_dose_response_overlay()
    p2 = plot_suppression_bars(rows)
    md = format_markdown(rows)
    print(f"Wrote {p1.relative_to(REPO)}")
    print(f"Wrote {p2.relative_to(REPO)}")
    print("\n" + md)
    (LOC / "bootstrap_table.md").write_text(md + "\n")
    print(f"\nWrote {(LOC / 'bootstrap_table.md').relative_to(REPO)}")


if __name__ == "__main__":
    main()
