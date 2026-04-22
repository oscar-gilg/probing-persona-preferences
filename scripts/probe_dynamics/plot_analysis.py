"""Plot probe-dynamics analysis: per-condition composite figures + summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/probe_dynamics/"
    "experiments/probe_dynamics/analysis"
)
ASSETS_DIR = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/probe_dynamics/"
    "experiments/probe_dynamics/assets"
)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

TYPE_COLORS = {
    "yesno": "#1f77b4",
    "open": "#ff7f0e",
    "pair": "#2ca02c",
    "pair_synth": "#d62728",
}
TYPE_ORDER = ["yesno", "open", "pair", "pair_synth"]

CONDITIONS = [
    "control_helpful",
    "icl_misalignment",
    "offpolicy_consciousness",
    "offpolicy_harm_compliance",
    "onpolicy_consciousness",
    "onpolicy_harm_compliance",
    "qwen_delusion",
]


def plot_condition(
    condition: str,
    traj_by_type: pd.DataFrame,
    ckpt_r: pd.DataFrame,
) -> Path:
    sub_traj = traj_by_type[traj_by_type["condition"] == condition].copy()
    sub_ckpt = ckpt_r[ckpt_r["condition"] == condition].copy()
    sub_ckpt = sub_ckpt.sort_values("checkpoint")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    ax_probe, ax_behav, ax_corr = axes

    for ptype in TYPE_ORDER:
        sub = sub_traj[sub_traj["prompt_type"] == ptype].sort_values("checkpoint")
        if sub.empty:
            continue
        color = TYPE_COLORS[ptype]
        x = sub["checkpoint"].to_numpy()
        probe_mean = sub["probe_mean"].to_numpy()
        probe_std = sub["probe_std"].to_numpy()
        behav_mean = sub["behav_mean"].to_numpy()
        behav_std = sub["behav_std"].to_numpy()

        ax_probe.plot(x, probe_mean, color=color, label=ptype, linewidth=2)
        ax_probe.fill_between(
            x, probe_mean - probe_std, probe_mean + probe_std, color=color, alpha=0.2
        )

        ax_behav.plot(x, behav_mean, color=color, label=ptype, linewidth=2)
        ax_behav.fill_between(
            x,
            np.clip(behav_mean - behav_std, 0, 1),
            np.clip(behav_mean + behav_std, 0, 1),
            color=color,
            alpha=0.2,
        )

    ax_probe.set_title(f"Probe score trajectory — {condition}")
    ax_probe.set_xlabel("checkpoint")
    ax_probe.set_ylabel("probe score (mean ± 1σ)")
    ax_probe.legend(loc="best", fontsize=8, frameon=True)
    ax_probe.grid(True, alpha=0.3)

    ax_behav.set_title(f"Behaviour (drift-aligned) — {condition}")
    ax_behav.set_xlabel("checkpoint")
    ax_behav.set_ylabel("behaviour (mean ± 1σ)")
    ax_behav.set_ylim(0, 1)
    ax_behav.legend(loc="best", fontsize=8, frameon=True)
    ax_behav.grid(True, alpha=0.3)

    if not sub_ckpt.empty:
        x = sub_ckpt["checkpoint"].to_numpy()
        r = sub_ckpt["pearson_r"].to_numpy()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in r]
        # Bar width: use spacing between checkpoints (fallback to 1.5)
        if len(x) > 1:
            width = float(np.median(np.diff(np.sort(x)))) * 0.8
        else:
            width = 1.5
        ax_corr.bar(x, r, width=width, color=colors, edgecolor="black", linewidth=0.3)
    ax_corr.axhline(0, color="black", linewidth=0.8)
    ax_corr.set_title("r(probe, behaviour) across prompts @ each checkpoint")
    ax_corr.set_xlabel("checkpoint")
    ax_corr.set_ylabel("pearson r")
    ax_corr.set_ylim(-1, 1)
    ax_corr.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Probe dynamics — {condition}", fontsize=13, y=0.995)
    fig.tight_layout()

    out_path = ASSETS_DIR / f"plot_042226_{condition}_dynamics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary(prompt_r: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))

    positions = []
    data_per_cond = []
    n_per_cond = []
    for i, cond in enumerate(CONDITIONS):
        sub = prompt_r[prompt_r["condition"] == cond]
        values = sub["pearson_r"].dropna().to_numpy()
        data_per_cond.append(values)
        n_per_cond.append(len(values))
        positions.append(i)

    # Violin plot (only where data non-empty)
    valid_positions = [p for p, d in zip(positions, data_per_cond) if len(d) > 1]
    valid_data = [d for d in data_per_cond if len(d) > 1]
    if valid_data:
        parts = ax.violinplot(
            valid_data, positions=valid_positions, widths=0.7, showmeans=False,
            showmedians=True, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#1f77b4")
            pc.set_alpha(0.35)
            pc.set_edgecolor("#1f77b4")

    # Overlay jittered strip of individual prompt correlations
    rng = np.random.default_rng(seed=42)
    for pos, values in zip(positions, data_per_cond):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            pos + jitter, values, s=18, color="#1f77b4", edgecolor="black",
            linewidth=0.3, alpha=0.75, zorder=3,
        )

    ax.axhline(0, color="red", linewidth=1.2, linestyle="-", alpha=0.85)
    ax.set_xticks(positions)
    labels = [f"{c}\n(n={n})" for c, n in zip(CONDITIONS, n_per_cond)]
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("pearson r")
    ax.set_ylim(-1, 1)
    ax.set_title(
        "Per-prompt time-series correlation r(probe_score(t), behaviour(t)) across checkpoints"
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = ASSETS_DIR / "plot_042226_summary_prompt_correlations.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    traj_by_type = pd.read_csv(ANALYSIS_DIR / "trajectory_by_type.csv")
    ckpt_r = pd.read_csv(ANALYSIS_DIR / "checkpoint_r.csv")
    prompt_r = pd.read_csv(ANALYSIS_DIR / "prompt_r.csv")

    saved = []
    for cond in CONDITIONS:
        if cond not in traj_by_type["condition"].unique():
            print(f"[WARN] condition missing from trajectory_by_type: {cond}")
            continue
        path = plot_condition(cond, traj_by_type, ckpt_r)
        saved.append(path)
        print(f"wrote {path}")

    summary_path = plot_summary(prompt_r)
    saved.append(summary_path)
    print(f"wrote {summary_path}")

    # Quick data oddity checks for reporting
    print("\n--- Data oddities ---")
    # NaN pearson_r counts in prompt_r
    for cond in CONDITIONS:
        sub = prompt_r[prompt_r["condition"] == cond]
        nan_count = sub["pearson_r"].isna().sum()
        total = len(sub)
        print(f"{cond}: prompt_r NaN={nan_count}/{total}")

    # checkpoint_r NaN per condition
    for cond in CONDITIONS:
        sub = ckpt_r[ckpt_r["condition"] == cond]
        nan_count = sub["pearson_r"].isna().sum()
        total = len(sub)
        print(f"{cond}: checkpoint_r NaN={nan_count}/{total}, n_ckpts={total}")


if __name__ == "__main__":
    main()
