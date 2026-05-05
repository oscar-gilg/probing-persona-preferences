"""Generate the three headline figures for probe_persona_drift.

Inputs:
- experiments/probe_persona_drift/results/persona_drift_table.csv
- experiments/probe_persona_drift/results/transfer_matrix_{truth,harm}.csv

Outputs:
- experiments/probe_persona_drift/assets/plot_<mmddyy>_headline_drift_3panel.png
- experiments/probe_persona_drift/assets/plot_<mmddyy>_train_size_sweep.png
- experiments/probe_persona_drift/assets/plot_<mmddyy>_transfer_matrix.png
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("experiments/probe_persona_drift/results")
ASSETS = Path("experiments/probe_persona_drift/assets")

# Persona ordering: default first, then non-adversarial controls, then adversarial.
PERSONA_ORDER = [
    "default",
    "helpful_assistant",
    "neutral_long",
    "Aura",
    "mathematician",
    "pathological_liar",
    "villain",
    "sadist",
]

# Display labels (kept short for x-tick / legend readability).
PERSONA_LABEL = {
    "default": "default\n(no sysprompt)",
    "helpful_assistant": "helpful\nassistant",
    "neutral_long": "neutral_long\n(Midwest filler)",
    "Aura": "Aura",
    "mathematician": "mathematician",
    "pathological_liar": "pathological\n_liar",
    "villain": "villain",
    "sadist": "sadist",
}

# Persona color map used in train-size sweep (need 7 distinguishable lines).
PERSONA_COLORS = {
    "default": "#2c7fb8",
    "helpful_assistant": "#5ba3d4",
    "neutral_long": "#7fcdbb",
    "Aura": "#41ab5d",
    "mathematician": "#bdbdbd",
    "pathological_liar": "#cb181d",
    "villain": "#a63603",
    "sadist": "#67000d",
}

# Sign-encoded colors for headline bars and transfer-matrix cells:
# matches the canonical §3.1 plot (blue = "in the trained direction", red = "flipped").
COLOR_POS = "#2c7fb8"
COLOR_NEG = "#cb181d"

TARGET_LABELS = {"truth": "Truth probe", "harm": "Harm probe"}


def _best_layer_per_persona(sub: pd.DataFrame) -> pd.DataFrame:
    """Pick, per eval persona, the layer with max |Cohen's d|."""
    idx = sub.groupby("eval_persona")["cohen_d"].apply(lambda s: s.abs().idxmax())
    return sub.loc[idx]


def headline_drift_3panel(df: pd.DataFrame, date: str) -> Path:
    """Three rows (truth-probe, harm-probe, preference-probe applied to truth).

    Bars are sign-encoded (blue = same sign as default, red = flipped) so the
    reader can spot sign-flips at a glance.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    rows_to_plot = [
        ("truth", "default_sweep", "Truth probe (trained on default Assistant, CREAK)", "$d{>}0$ = true ranks above false"),
        ("harm", "default_sweep", "Harm probe (trained on default Assistant, BailBench+HarmBench vs Alpaca+WildChat)", "$d{>}0$ = harmful ranks above benign"),
        ("truth", "preference_baseline", "Preference probe (existing user-EOT baseline) — applied to the truth held-out set", "$d{>}0$ = true ranks above false"),
    ]

    for ax, (target, mode, title, sign_note) in zip(axes, rows_to_plot):
        sub = df[(df["target"] == target) & (df["mode"] == mode)]
        if mode == "default_sweep":
            largest = sub["train_size"].max()
            sub = sub[sub["train_size"] == largest]
        best = _best_layer_per_persona(sub)
        order = [p for p in PERSONA_ORDER if p in best["eval_persona"].values]
        bars = best.set_index("eval_persona").reindex(order)
        colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in bars["cohen_d"]]
        ax.bar(range(len(order)), bars["cohen_d"], color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([PERSONA_LABEL[p] for p in order], rotation=0, ha="center", fontsize=9)
        ax.set_ylabel("Cohen's $d$\n(best layer per persona)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.text(1.0, 1.02, sign_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic", color="#555")
        ax.grid(axis="y", alpha=0.3)
        ymax = max(abs(bars["cohen_d"].max()), abs(bars["cohen_d"].min())) * 1.15 + 0.3
        ax.set_ylim(-ymax, ymax)
        # Annotate bar values for clarity.
        for i, v in enumerate(bars["cohen_d"]):
            ax.text(i, v + (0.05 * ymax if v >= 0 else -0.05 * ymax), f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    axes[-1].set_xlabel("System prompt at evaluation")
    fig.suptitle("Does a probe trained on default Assistant survive a system-prompt swap?\n(Held-out items, byte-identical prefills across all conditions)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    out_path = ASSETS / f"plot_{date}_headline_drift_3panel.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def train_size_sweep(df: pd.DataFrame, date: str) -> Path:
    """Two panels (truth, harm). Cohen's d at best layer per (train size, eval persona)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for ax, target in zip(axes, ("truth", "harm")):
        sub = df[(df["target"] == target) & (df["mode"] == "default_sweep")]
        idx = sub.groupby(["train_size", "eval_persona"])["cohen_d"].apply(lambda s: s.abs().idxmax())
        best = sub.loc[idx]
        for persona in PERSONA_ORDER:
            row = best[best["eval_persona"] == persona].sort_values("train_size")
            if row.empty:
                continue
            label = "default (training condition)" if persona == "default" else persona
            lw = 3.0 if persona == "default" else 1.8
            ms = 9 if persona == "default" else 6
            ax.plot(row["train_size"], row["cohen_d"], marker="o", color=PERSONA_COLORS[persona],
                    label=label, linewidth=lw, markersize=ms)
        ax.set_xscale("log")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Default-Assistant training items", fontsize=11)
        ax.set_ylabel("Cohen's $d$ on held-out (best layer)", fontsize=11)
        ax.set_title(TARGET_LABELS[target], fontsize=12)
        ax.grid(alpha=0.3)
    # Single shared legend to the right of the right panel.
    axes[1].legend(fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   title="Eval system prompt", title_fontsize=11, frameon=False)
    fig.suptitle("More training data $\\Rightarrow$ wider gap between default-eval and persona-eval",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out_path = ASSETS / f"plot_{date}_train_size_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def transfer_matrix(date: str) -> Path:
    """Heatmap: train_persona × eval_persona, Cohen's d at best layer, per target."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, target in zip(axes, ("truth", "harm")):
        path = RESULTS / f"transfer_matrix_{target}.csv"
        if not path.exists():
            ax.text(0.5, 0.5, f"missing {path.name}", ha="center", va="center", transform=ax.transAxes)
            continue
        m = pd.read_csv(path, index_col=0)
        train_order = [p for p in PERSONA_ORDER if p in m.index]
        eval_order = [p for p in PERSONA_ORDER if p in m.columns]
        m = m.loc[train_order, eval_order]
        vmax = m.abs().to_numpy().max()
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels([PERSONA_LABEL[p] for p in m.columns], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels(m.index, fontsize=9)
        ax.set_xlabel("Eval system prompt")
        ax.set_ylabel("Train system prompt")
        ax.set_title(TARGET_LABELS[target])
        # Mark diagonal (matched train/eval) cells with a thicker border.
        for i, train_p in enumerate(m.index):
            for j, eval_p in enumerate(m.columns):
                v = m.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > vmax * 0.5 else "black", fontsize=8)
                if train_p == eval_p:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="black", linewidth=2))
        plt.colorbar(im, ax=ax, label="Cohen's $d$")
    fig.suptitle("Train-on-X $\\to$ test-on-Y transfer (best layer, largest train size; black border = matched train/eval)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out_path = ASSETS / f"plot_{date}_transfer_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "persona_drift_table.csv")
    # Pin to original run date so the report's filenames stay stable.
    date = "050526"
    print(f"Plotting for date {date}, {len(df)} rows in table")
    p1 = headline_drift_3panel(df, date)
    print(f"  {p1}")
    p2 = train_size_sweep(df, date)
    print(f"  {p2}")
    p3 = transfer_matrix(date)
    print(f"  {p3}")


if __name__ == "__main__":
    main()
