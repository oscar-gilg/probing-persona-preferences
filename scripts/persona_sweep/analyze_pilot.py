"""Analyze persona sweep pilot: utility correlations and style comparison."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path("results/experiments/exp_20260319_153056/pre_task_active_learning")
PERSONAS_FILE = Path(__file__).resolve().parents[2] / "experiments/old_experiments/persona_sweep/pilot_personas.json"
PLOT_DIR = Path("experiments/persona_sweep/assets")


def load_persona_lookup() -> dict[str, str]:
    """Map system_prompt (first 50 chars) -> persona name."""
    with open(PERSONAS_FILE) as f:
        data = json.load(f)
    return {p["system_prompt"][:50]: p["name"] for p in data["personas"]}


def load_run(run_dir: Path) -> tuple[str, pd.Series]:
    """Load a single run, return (persona_name, scores_series)."""
    al_path = run_dir / "active_learning.yaml"
    with open(al_path) as f:
        al_data = yaml.safe_load(f)

    sys_prompt = al_data.get("system_prompt")
    if sys_prompt is None:
        name = "baseline"
    else:
        lookup = load_persona_lookup()
        prefix = sys_prompt[:50]
        name = lookup.get(prefix, f"unknown_{run_dir.name[-8:]}")

    csv_files = list(run_dir.glob("thurstonian_*.csv"))
    df = pd.read_csv(csv_files[0])
    scores = df.set_index("task_id")["mu"]
    return name, scores


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs = {}
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        name, scores = load_run(run_dir)
        runs[name] = scores

    # Align all scores to common task set
    all_tasks = set()
    for scores in runs.values():
        all_tasks.update(scores.index)
    common_tasks = sorted(all_tasks.intersection(*[set(s.index) for s in runs.values()]))
    print(f"Common tasks across all runs: {len(common_tasks)}")

    df = pd.DataFrame({name: scores.reindex(common_tasks) for name, scores in runs.items()})
    print(f"\nLoaded {len(df.columns)} runs:")
    for col in sorted(df.columns):
        print(f"  {col}")

    # Correlation matrix
    corr = df.corr()

    # Sort: group by persona category (implicit/explicit pairs together)
    categories = []
    for col in corr.columns:
        if col == "baseline":
            categories.append(("00_baseline", col))
        else:
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("implicit", "explicit"):
                categories.append((parts[0], col))
            else:
                categories.append((col, col))
    categories.sort()
    sorted_cols = [c[1] for c in categories]
    corr = corr.loc[sorted_cols, sorted_cols]

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Annotate with correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Persona Sweep Pilot: Utility Profile Correlations")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot_031926_pilot_utility_correlations.png", dpi=150)
    plt.close()
    print(f"\nSaved correlation heatmap to {PLOT_DIR}/")

    # Key analysis: implicit vs explicit within each persona
    print("\n=== IMPLICIT vs EXPLICIT (same persona) ===")
    persona_names = set()
    for col in df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("implicit", "explicit"):
            persona_names.add(parts[0])

    for persona in sorted(persona_names):
        imp = f"{persona}_implicit"
        exp = f"{persona}_explicit"
        if imp in df.columns and exp in df.columns:
            r = df[imp].corr(df[exp])
            print(f"  {persona:20s}  r = {r:.3f}")

    # Correlation with baseline
    print("\n=== CORRELATION WITH BASELINE ===")
    if "baseline" in df.columns:
        for col in sorted(df.columns):
            if col == "baseline":
                continue
            r = df["baseline"].corr(df[col])
            print(f"  {col:35s}  r = {r:.3f}")

    # Find most different pairs (lowest correlation)
    print("\n=== MOST DIFFERENT PAIRS (lowest r) ===")
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((corr.values[i, j], cols[i], cols[j]))
    pairs.sort()
    for r, a, b in pairs[:10]:
        print(f"  r = {r:+.3f}  {a} vs {b}")

    # Find most similar pairs (highest correlation, excluding self and implicit/explicit of same)
    print("\n=== MOST SIMILAR PAIRS (highest r, cross-persona) ===")
    pairs_high = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            a_base = a.rsplit("_", 1)[0] if a.rsplit("_", 1)[-1] in ("implicit", "explicit") else a
            b_base = b.rsplit("_", 1)[0] if b.rsplit("_", 1)[-1] in ("implicit", "explicit") else b
            if a_base == b_base:
                continue
            pairs_high.append((corr.values[i, j], a, b))
    pairs_high.sort(reverse=True)
    for r, a, b in pairs_high[:10]:
        print(f"  r = {r:+.3f}  {a} vs {b}")


if __name__ == "__main__":
    main()
