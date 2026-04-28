"""Plot probe vs random control across ablation conditions (4-metric panel)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "preference_direction_ablation" / "results"
ASSETS_DIR = REPO_ROOT / "experiments" / "preference_direction_ablation" / "assets"

CONDITIONS = ["A_L25", "A_L32", "B_two", "C_band"]
METRICS = [
    ("agreement_vs_b0", "Agreement vs B0", (0.7, 1.0)),
    ("ks_pa_vs_b0", "KS distance (P(A)) vs B0", (0.0, 0.25)),
    ("flip_rate", "Flip rate", (0.2, 0.6)),
    ("d_mean_abs_dev", r"$\Delta$ mean |dev| from 0.5", (-0.025, 0.005)),
]

# B0 reference values (from summary.csv row "B0")
B0_REF = {
    "agreement_vs_b0": None,  # B0 has no value (it is the reference)
    "ks_pa_vs_b0": 0.0,
    "flip_rate": 0.28679653679653677,
    "d_mean_abs_dev": 0.0,
}

PROBE_COLOR = "#d95f02"  # orange-red
RANDOM_COLOR = "#222222"


def main() -> None:
    pvr = pd.read_csv(RESULTS_DIR / "probe_vs_random.csv")
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()

    for ax, (metric, title, ylim) in zip(axes_flat, METRICS):
        for x, cond in enumerate(CONDITIONS):
            row = pvr[(pvr["condition"] == cond) & (pvr["metric"] == metric)].iloc[0]
            probe_val = row["probe"]
            r_mean = row["random_mean"]
            r_std = row["random_std"]

            # Random: errorbar (mean +/- std)
            ax.errorbar(
                x,
                r_mean,
                yerr=r_std,
                fmt="_",
                color=RANDOM_COLOR,
                ecolor=RANDOM_COLOR,
                elinewidth=1.2,
                capsize=5,
                markersize=14,
                markeredgewidth=1.2,
                zorder=2,
                label="Random control (mean +/- std)" if x == 0 else None,
            )

            # Random: individual n=5 points from summary.csv
            random_rows = summary[
                summary["cell"].str.startswith(f"{cond}_random")
            ]
            metric_col = metric  # column name matches in summary.csv
            random_vals = random_rows[metric_col].to_numpy()
            jitter_x = [x - 0.12] * len(random_vals)
            ax.scatter(
                jitter_x,
                random_vals,
                s=18,
                color=RANDOM_COLOR,
                alpha=0.55,
                zorder=3,
                label="Random samples (n=5)" if x == 0 else None,
            )

            # Probe: bold star
            ax.scatter(
                x,
                probe_val,
                marker="*",
                s=240,
                color=PROBE_COLOR,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                label="Probe" if x == 0 else None,
            )

        # B0 reference line where applicable
        ref = B0_REF[metric]
        if ref is not None:
            ax.axhline(
                ref,
                linestyle="--",
                color="gray",
                linewidth=1.0,
                alpha=0.8,
                zorder=1,
                label="B0 reference",
            )

        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    # Single legend at the top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq.append((h, l))
    fig.legend(
        [h for h, _ in uniq],
        [l for _, l in uniq],
        loc="upper center",
        ncol=len(uniq),
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "Probe vs random control across ablation conditions",
        y=0.94,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / "plot_042826_probe_vs_random_panel.png"
    fig.savefig(out, dpi=180)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
