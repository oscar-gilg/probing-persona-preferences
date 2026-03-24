"""Replot persona sweep with improved labels, colors, and framing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results/experiments/exp_20260319_235609/pre_task_active_learning"
PERSONAS_FILE = ROOT / "experiments/persona_sweep/sweep_personas.json"
PLOT_DIR = ROOT / "experiments/persona_sweep/assets"

CLUSTER_COLORS = {
    "structured": "#1f77b4",
    "creative": "#ff7f0e",
    "dark": "#d62728",
    "outlier": "#2ca02c",
    "sadist": "#9467bd",
}

CLUSTER_MAP = {
    "archivist": "structured",
    "mathematician": "structured",
    "builder": "structured",
    "nihilist": "structured",
    "poet": "creative",
    "comedian": "creative",
    "historian": "creative",
    "therapist": "creative",
    "strategist": "dark",
    "narcissist": "dark",
    "risk_seeker": "dark",
    "contrarian": "dark",
    "sadist": "sadist",
    "slacker": "outlier",
    "entrepreneur": "outlier",
    "baseline": "outlier",
}


def load_persona_lookup():
    with open(PERSONAS_FILE) as f:
        data = json.load(f)
    return {p["system_prompt"][:50]: p["name"] for p in data["personas"]}


def load_all_runs():
    lookup = load_persona_lookup()
    runs = {}
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        al_path = run_dir / "active_learning.yaml"
        with open(al_path) as f:
            al_data = yaml.safe_load(f)
        sys_prompt = al_data.get("system_prompt")
        if sys_prompt is None:
            name = "baseline"
        else:
            name = lookup.get(sys_prompt[:50], f"unknown_{run_dir.name[-8:]}")
        csv_files = list(run_dir.glob("thurstonian_*.csv"))
        df = pd.read_csv(csv_files[0])
        runs[name] = df.set_index("task_id")["mu"]
    common_tasks = sorted(set.intersection(*[set(s.index) for s in runs.values()]))
    return pd.DataFrame({name: scores.reindex(common_tasks) for name, scores in runs.items()})


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_runs()
    corr = df.corr()

    # Hierarchical clustering order
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    dendro = dendrogram(Z, labels=corr.columns.tolist(), no_plot=True)
    order = dendro["leaves"]
    sorted_cols = [corr.columns[i] for i in order]
    corr_sorted = corr.loc[sorted_cols, sorted_cols]

    # --- Plot 1: Heatmap (keep as-is, already good) ---
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_sorted.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_sorted)))
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_xticklabels(corr_sorted.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_sorted.columns, fontsize=9)
    for i in range(len(corr_sorted)):
        for j in range(len(corr_sorted)):
            val = corr_sorted.values[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r between utility vectors")
    ax.set_title("Pairwise correlation of Thurstonian utility profiles (hierarchically clustered)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_032026_sweep_correlation_heatmap.png", dpi=150)
    plt.close()

    # --- Plot 2: PCA with cluster colors ---
    X = df.T.values
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(df.columns):
        cluster = CLUSTER_MAP.get(name, "outlier")
        color = CLUSTER_COLORS[cluster]
        marker = "s" if name == "baseline" else "o"
        size = 120 if name == "baseline" else 80
        ax.scatter(coords[i, 0], coords[i, 1], s=size, color=color, marker=marker, zorder=3,
                  edgecolors="black", linewidths=0.5)
        ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=8,
                   xytext=(6, 6), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var) — analytical/structured ← → creative/humanistic")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var) — prosocial/constructive ↑  ↓ dark/oppositional")
    ax.set_title("Persona utility profiles in PC space (500 tasks, Gemma 3 27B)")
    ax.axhline(y=0, color="gray", linewidth=0.3)
    ax.axvline(x=0, color="gray", linewidth=0.3)

    legend_handles = [
        mpatches.Patch(color=CLUSTER_COLORS["structured"], label="Structured/analytical"),
        mpatches.Patch(color=CLUSTER_COLORS["creative"], label="Creative/humanistic"),
        mpatches.Patch(color=CLUSTER_COLORS["dark"], label="Dark/oppositional"),
        mpatches.Patch(color=CLUSTER_COLORS["sadist"], label="Sadist (extreme)"),
        mpatches.Patch(color=CLUSTER_COLORS["outlier"], label="Between clusters"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_032026_sweep_pca.png", dpi=150)
    plt.close()

    # --- Plot 3: Baseline shift (reframed) ---
    shifts = {}
    for col in df.columns:
        if col == "baseline":
            continue
        shifts[col] = df["baseline"].corr(df[col])
    shifts_sorted = dict(sorted(shifts.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [CLUSTER_COLORS.get(CLUSTER_MAP.get(k, "outlier"), "#999999") for k in shifts_sorted]
    bars = ax.barh(list(shifts_sorted.keys()), list(shifts_sorted.values()), color=colors,
                   edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Correlation with baseline utility profile (r)")
    ax.set_title("How much does each persona shift preferences from default?\n(lower r = more different from baseline; negative r = inverted preferences)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlim(-0.5, 0.8)
    for bar, val in zip(bars, shifts_sorted.values()):
        offset = 0.02 if val >= 0 else -0.08
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_032026_sweep_baseline_shift.png", dpi=150)
    plt.close()

    # --- Plot 4: Dendrogram with cluster colors ---
    fig, ax = plt.subplots(figsize=(14, 5))
    dendro_plot = dendrogram(Z, labels=corr.columns.tolist(), ax=ax,
                             leaf_rotation=45, leaf_font_size=9,
                             above_threshold_color="gray",
                             color_threshold=0.55)
    ax.set_ylabel("Distance (1 − r)")
    ax.set_title("Hierarchical clustering of persona utility profiles")
    ax.axhline(y=0.55, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.5, 0.57, "cluster threshold", fontsize=8, color="gray", alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_032026_sweep_dendrogram.png", dpi=150)
    plt.close()

    print("All plots saved.")


if __name__ == "__main__":
    main()
