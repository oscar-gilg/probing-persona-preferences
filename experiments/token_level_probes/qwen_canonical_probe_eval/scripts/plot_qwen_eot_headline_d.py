"""Headline |Cohen's d| bar plot for the Qwen-3.5-122B canonical-probe replication.

Mirrors `experiments/token_level_probes/canonical_probe_eval/assets/plot_042126_tb_eot_headline_d.png`
but uses the 6 Qwen probes (tb-1 / tb-4 at L33/L38/L43) plus a Gemma reference bar.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
QWEN_CSV = EXP_DIR / "headline_table.csv"
GEMMA_CSV = Path("experiments/token_level_probes/canonical_probe_eval/headline_table.csv")
DATE = "042526"

DOMAINS = [
    ("truth", "Truth\n(true vs false)"),
    ("harm", "Harm\n(harmful vs benign)"),
    ("politics_democrat", "Politics\n(democrat prompt)"),
    ("politics_republican", "Politics\n(republican prompt)"),
]

QWEN_PROBES = [
    "qwen_tb-1_L33",
    "qwen_tb-1_L38",
    "qwen_tb-1_L43",
    "qwen_tb-4_L33",
    "qwen_tb-4_L38",
    "qwen_tb-4_L43",
]

# Blue shades for tb-1 (light to dark across L33->L38->L43);
# teal/green shades for tb-4 (light to dark);
# orange for Gemma reference.
BAR_COLORS = {
    "qwen_tb-1_L33": "#9ECAE1",
    "qwen_tb-1_L38": "#4292C6",
    "qwen_tb-1_L43": "#08519C",
    "qwen_tb-4_L33": "#A1D99B",
    "qwen_tb-4_L38": "#41AB5D",
    "qwen_tb-4_L43": "#006D2C",
    "Gemma reference": "#FF7F0E",
}

# Per-domain Gemma reference probe (matches the canonical_probe_eval headline plot:
# tb-5_L32 for truth, tb-5_L39 for harm/politics).
GEMMA_PROBE_BY_DOMAIN = {
    "truth": "tb-5_L32",
    "harm": "tb-5_L39",
    "politics_democrat": "tb-5_L39",
    "politics_republican": "tb-5_L39",
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def lookup_abs_d(rows, domain, probe):
    for r in rows:
        if r["domain"] == domain and r["probe"] == probe:
            return abs(float(r["d"]))
    raise KeyError(f"({domain}, {probe}) not in CSV")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    qwen_rows = load_csv(QWEN_CSV)
    gemma_rows = load_csv(GEMMA_CSV)

    # values[domain_key] = list of |d| in bar order (qwen probes then gemma).
    bar_labels = QWEN_PROBES + ["Gemma reference"]
    values = {}
    for domain_key, _ in DOMAINS:
        domain_vals = []
        for probe in QWEN_PROBES:
            domain_vals.append(lookup_abs_d(qwen_rows, domain_key, probe))
        gemma_probe = GEMMA_PROBE_BY_DOMAIN[domain_key]
        domain_vals.append(lookup_abs_d(gemma_rows, domain_key, gemma_probe))
        values[domain_key] = domain_vals

    n_bars = len(bar_labels)
    n_groups = len(DOMAINS)
    group_centers = np.arange(n_groups)
    bar_width = 0.11
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for j, label in enumerate(bar_labels):
        heights = [values[d_key][j] for d_key, _ in DOMAINS]
        x = group_centers + offsets[j]
        bars = ax.bar(
            x, heights, width=bar_width,
            color=BAR_COLORS[label], edgecolor="black", linewidth=0.4,
            label=label,
        )
        for xi, h in zip(x, heights):
            ax.text(xi, h + 0.04, f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(2.0, color="grey", linestyle=":", linewidth=1.0,
               label="large effect (d=2)")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([label for _, label in DOMAINS], fontsize=10)
    ax.set_ylabel("|Cohen's d| at EOT token")
    ax.set_title(
        "Qwen-3.5-122B end-of-turn probes: |d| across truth / harm / politics"
    )
    max_h = max(max(values[d_key]) for d_key, _ in DOMAINS)
    ax.set_ylim(0, max(max_h * 1.18, 3.6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    plt.tight_layout()
    out_path = ASSETS_DIR / f"plot_{DATE}_qwen_eot_headline_d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")
    for d_key, label in DOMAINS:
        readable = label.replace("\n", " ")
        nums = ", ".join(
            f"{lab}={v:.2f}" for lab, v in zip(bar_labels, values[d_key])
        )
        print(f"  {readable}: {nums}")


if __name__ == "__main__":
    main()
