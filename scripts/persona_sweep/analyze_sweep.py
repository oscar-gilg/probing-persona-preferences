"""Analyze full persona sweep: utility correlations, clustering, shift from baseline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results/experiments/exp_20260319_235609/pre_task_active_learning"
PERSONAS_FILE = ROOT / "experiments/persona_sweep/sweep_personas.json"
PLOT_DIR = ROOT / "experiments/persona_sweep/assets"


def load_persona_lookup() -> dict[str, str]:
    with open(PERSONAS_FILE) as f:
        data = json.load(f)
    return {p["system_prompt"][:50]: p["name"] for p in data["personas"]}


def load_run(run_dir: Path, lookup: dict[str, str]) -> tuple[str, pd.Series]:
    al_path = run_dir / "active_learning.yaml"
    with open(al_path) as f:
        al_data = yaml.safe_load(f)

    sys_prompt = al_data.get("system_prompt")
    if sys_prompt is None:
        name = "baseline"
    else:
        prefix = sys_prompt[:50]
        name = lookup.get(prefix, f"unknown_{run_dir.name[-8:]}")

    csv_files = list(run_dir.glob("thurstonian_*.csv"))
    df = pd.read_csv(csv_files[0])
    scores = df.set_index("task_id")["mu"]
    return name, scores


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    lookup = load_persona_lookup()

    # Load all runs
    runs = {}
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        name, scores = load_run(run_dir, lookup)
        runs[name] = scores

    # Align to common tasks
    common_tasks = sorted(set.intersection(*[set(s.index) for s in runs.values()]))
    print(f"Common tasks: {len(common_tasks)}")

    df = pd.DataFrame({name: scores.reindex(common_tasks) for name, scores in runs.items()})
    print(f"Runs: {len(df.columns)}")
    for col in sorted(df.columns):
        print(f"  {col}")

    corr = df.corr()

    # --- Plot 1: Correlation heatmap with hierarchical clustering ---
    # Cluster the correlation matrix
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # symmetrize
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    dendro = dendrogram(Z, labels=corr.columns.tolist(), no_plot=True)
    order = dendro["leaves"]
    sorted_cols = [corr.columns[i] for i in order]

    corr_sorted = corr.loc[sorted_cols, sorted_cols]

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

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Persona Sweep: Utility Correlations (hierarchically clustered)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_032026_sweep_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"\nSaved heatmap")

    # --- Plot 2: Dendrogram ---
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, labels=corr.columns.tolist(), ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_ylabel("Distance (1 - correlation)")
    ax.set_title("Persona Utility Profile Clustering")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_032026_sweep_dendrogram.png", dpi=150)
    plt.close()
    print("Saved dendrogram")

    # --- Plot 3: Shift from baseline (bar chart) ---
    if "baseline" in df.columns:
        shifts = {}
        for col in df.columns:
            if col == "baseline":
                continue
            shifts[col] = df["baseline"].corr(df[col])

        shifts_sorted = dict(sorted(shifts.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in shifts_sorted.values()]
        bars = ax.barh(list(shifts_sorted.keys()), list(shifts_sorted.values()), color=colors)
        ax.set_xlabel("Correlation with baseline (lower = more shifted)")
        ax.set_title("Preference Shift from Baseline")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlim(-0.5, 1.0)
        for bar, val in zip(bars, shifts_sorted.values()):
            ax.text(val + 0.02 if val >= 0 else val - 0.06, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "plot_032026_sweep_baseline_shift.png", dpi=150)
        plt.close()
        print("Saved baseline shift chart")

    # --- Plot 4: PCA of utility profiles ---
    from sklearn.decomposition import PCA

    X = df.T.values  # personas x tasks
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(df.columns):
        color = "#d62728" if name == "sadist" else "#2ca02c" if name == "baseline" else "#1f77b4"
        ax.scatter(coords[i, 0], coords[i, 1], s=80, color=color, zorder=3)
        ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=8,
                   xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Persona Utility Profiles — PCA")
    ax.axhline(y=0, color="gray", linewidth=0.3)
    ax.axvline(x=0, color="gray", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_032026_sweep_pca.png", dpi=150)
    plt.close()
    print("Saved PCA plot")

    # --- Print summary stats ---
    print("\n=== CORRELATION WITH BASELINE ===")
    if "baseline" in df.columns:
        for col in sorted(df.columns):
            if col == "baseline":
                continue
            r = df["baseline"].corr(df[col])
            print(f"  {col:20s}  r = {r:+.3f}")

    print("\n=== MOST SIMILAR CROSS-PERSONA PAIRS ===")
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((corr.values[i, j], cols[i], cols[j]))
    pairs.sort(reverse=True)
    for r, a, b in pairs[:15]:
        if a == b:
            continue
        print(f"  r = {r:+.3f}  {a} vs {b}")

    print("\n=== MOST DIFFERENT PAIRS ===")
    pairs.sort()
    for r, a, b in pairs[:10]:
        print(f"  r = {r:+.3f}  {a} vs {b}")

    # Explained variance
    pca_full = PCA()
    pca_full.fit(X)
    print(f"\n=== PCA EXPLAINED VARIANCE ===")
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    for i in range(min(8, len(cumvar))):
        print(f"  PC{i+1}: {pca_full.explained_variance_ratio_[i]:.1%}  (cumulative: {cumvar[i]:.1%})")


if __name__ == "__main__":
    main()
