"""Plots consolidated tb-2 and tb-5 reports for probe direction uniqueness."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/probe_direction_uniqueness")
EXP_DIR = ROOT / "experiments/probe_science/probe_direction_uniqueness"
TRACK_DIR = EXP_DIR / "persona_prompt_tracking/output"

TB_INFO = {
    "tb-2": {
        "data_dir": TRACK_DIR / "L32_tb-2",
        "asset_dir": EXP_DIR / "assets_tb-2",
        "asymmetry_title": "$w_2$ collapses on negative personas (tb-2)",
    },
    "tb-5": {
        "data_dir": TRACK_DIR / "L32_tb-5",
        "asset_dir": EXP_DIR / "assets_tb-5",
        "asymmetry_title": "No clear pos/neg asymmetry (tb-5)",
    },
}

DPI = 300
PROBE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # w_0, w_1, w_2
RANDOM_COLOR = "#7f7f7f"

EXP_COLORS = {"1b": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}
SHAPE_FOR = {"pos": "^", "neg": "v", "split": "o", "baseline": "*"}


def load_results(data_dir: Path) -> dict:
    with open(data_dir / "results.json") as f:
        return json.load(f)


def load_trajectory(data_dir: Path) -> dict:
    with open(data_dir / "trajectory.json") as f:
        return json.load(f)


def categorize(condition: dict) -> str:
    """Return 'pos', 'neg', 'split', or 'baseline'."""
    if condition["is_baseline"]:
        return "baseline"
    persona = condition["persona"]
    if persona.endswith("_pos_persona"):
        return "pos"
    if persona.endswith("_neg_persona"):
        return "neg"
    return "split"  # exp 2/3 conditions have a 'split' field


def short_label(condition: dict) -> str:
    persona = condition["persona"]
    if condition["is_baseline"]:
        return "baseline"
    if persona.endswith("_pos_persona"):
        return persona.replace("_pos_persona", "_pos")
    if persona.endswith("_neg_persona"):
        return persona.replace("_neg_persona", "_neg")
    split = condition["split"]
    return f"{persona}_{split}"


def plot_iteration_trajectory(traj: dict, out_path: Path, tb: str) -> None:
    iters = [t["iter"] for t in traj["trajectory"]]
    final_r = [t["final_r"] for t in traj["trajectory"]]
    sweep_r = [t["sweep_r"] for t in traj["trajectory"]]
    hoo_mean_r = [t["hoo_mean_r"] for t in traj["trajectory"]]
    r_chance = traj["r_chance"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(iters, final_r, "o-", color=PROBE_COLORS[0], label="Heldout r (final)", linewidth=2, markersize=8)
    ax.plot(iters, sweep_r, "s-", color=PROBE_COLORS[1], label="Heldout r (sweep half)", linewidth=2, markersize=8)
    ax.plot(iters, hoo_mean_r, "^-", color=PROBE_COLORS[2], label="HOO r (mean across 13 topics)", linewidth=2, markersize=8)
    ax.axhspan(-r_chance, r_chance, color="grey", alpha=0.2, label=f"Chance band (|r| < {r_chance:.3f})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(iters)
    ax.set_xlabel("Iteration k (probe $w_k$ on residual after projecting out $w_0..w_{k-1}$)")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0, 1)
    ax.set_title(f"Iteration trajectory ({tb}, layer 32)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_direction_cosines(directions_path: Path, out_path: Path, tb: str) -> None:
    data = np.load(directions_path)
    vecs = np.stack([data[f"w_{i}"] for i in range(3)], axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / norms
    cos = unit @ unit.T

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cos[i, j]:.3f}", ha="center", va="center",
                    color="black" if abs(cos[i, j]) < 0.6 else "white", fontsize=11)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"$\\hat{{w}}_{i}$" for i in range(3)])
    ax.set_yticklabels([f"$\\hat{{w}}_{i}$" for i in range(3)])
    ax.set_title(f"Cosine similarity between iteration directions ({tb})")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def split_conditions(results: dict) -> dict[str, list[dict]]:
    pos, neg, exp23 = [], [], []
    for c in results["conditions"]:
        if c["is_baseline"]:
            continue
        cat = categorize(c)
        if cat == "pos":
            pos.append(c)
        elif cat == "neg":
            neg.append(c)
        elif cat == "split":
            exp23.append(c)
    return {"pos_persona": pos, "neg_persona": neg, "exp2/3": exp23}


def plot_rw_by_polarity(results: dict, out_path: Path, tb: str, title: str) -> None:
    groups = split_conditions(results)
    group_names = ["pos_persona", "neg_persona", "exp2/3"]
    means = {k: [] for k in group_names}
    stds = {k: [] for k in group_names}
    counts = {k: len(groups[k]) for k in group_names}
    for k in group_names:
        for wi in range(3):
            vals = np.array([c[f"r_w{wi}"] for c in groups[k]])
            means[k].append(vals.mean())
            stds[k].append(vals.std(ddof=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(group_names))
    width = 0.25
    for wi in range(3):
        vals = [means[k][wi] for k in group_names]
        errs = [stds[k][wi] for k in group_names]
        ax.bar(x + (wi - 1) * width, vals, width, yerr=errs, capsize=4,
               color=PROBE_COLORS[wi], label=f"$r_{{w_{wi}}}$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(n={counts[k]})" for k in group_names])
    ax.set_ylabel("Pearson r vs. persona-conditioned scores")
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{title}\n(mean +/- std across conditions, {tb})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_rw_scatter(results: dict, out_path: Path, tb: str, k: int) -> None:
    """Scatter r_wk (y) vs r_w0 (x)."""
    fig, ax = plt.subplots(figsize=(7, 6.5))

    by_legend: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for c in results["conditions"]:
        cat = categorize(c)
        exp = c["experiment"]
        marker = SHAPE_FOR[cat]
        key = (exp, cat)
        by_legend.setdefault(key, []).append((c["r_w0"], c[f"r_w{k}"]))

    for (exp, cat), pts in by_legend.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, marker=SHAPE_FOR[cat], color=EXP_COLORS[exp],
                   s=110, edgecolor="black", linewidth=0.6,
                   label=f"exp {exp} / {cat}", alpha=0.85)

    lo = min(min(c["r_w0"] for c in results["conditions"]),
             min(c[f"r_w{k}"] for c in results["conditions"])) - 0.05
    hi = max(max(c["r_w0"] for c in results["conditions"]),
             max(c[f"r_w{k}"] for c in results["conditions"])) + 0.05
    rng = (lo, hi)
    ax.plot(rng, rng, "--", color="grey", linewidth=1, label="y = x")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(rng)
    ax.set_ylim(rng)
    ax.set_aspect("equal")
    ax.set_xlabel("$r_{w_0}$ (canonical probe)")
    ax.set_ylabel(f"$r_{{w_{k}}}$ (iteration {k} probe)")
    ax.set_title(f"$r_{{w_{k}}}$ vs $r_{{w_0}}$ across 26 conditions ({tb})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_per_condition_bar(results: dict, out_path: Path, tb: str) -> None:
    conds = results["conditions"]
    labels = [short_label(c) for c in conds]
    n = len(conds)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 6))
    for wi in range(3):
        vals = [c[f"r_w{wi}"] for c in conds]
        ax.bar(x + (wi - 1.5) * width, vals, width,
               color=PROBE_COLORS[wi], label=f"$r_{{w_{wi}}}$")
    rand_vals = [c["r_random_mean"] for c in conds]
    rand_err = [c["r_random_std"] for c in conds]
    ax.bar(x + 1.5 * width, rand_vals, width, yerr=rand_err, capsize=2,
           color=RANDOM_COLOR, label="random direction (mean +/- std)")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r vs. persona-conditioned scores")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"Per-condition probe alignment ({tb})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_tb_comparison_w2(results_tb2: dict, results_tb5: dict, out_path: Path) -> None:
    def neg_only(results: dict) -> dict[str, float]:
        out = {}
        for c in results["conditions"]:
            if c["experiment"] != "1b" or not c["persona"].endswith("_neg_persona"):
                continue
            label = c["persona"].replace("_neg_persona", "")
            out[label] = c["r_w2"]
        return out

    n2 = neg_only(results_tb2)
    n5 = neg_only(results_tb5)
    labels = sorted(set(n2.keys()) & set(n5.keys()))
    vals2 = [n2[l] for l in labels]
    vals5 = [n5[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, vals2, width, color=PROBE_COLORS[0], label="tb-2")
    ax.bar(x + width / 2, vals5, width, color=PROBE_COLORS[1], label="tb-5")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}_neg" for l in labels], rotation=45, ha="right")
    ax.set_ylabel("$r_{w_2}$ on negative-persona conditions")
    ax.set_ylim(-0.3, 1.0)
    ax.set_title("$w_2$ tracking on negative personas: tb-2 vs tb-5")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    summaries: dict[str, dict] = {}
    results_per_tb: dict[str, dict] = {}

    for tb, info in TB_INFO.items():
        data_dir = info["data_dir"]
        asset_dir = info["asset_dir"]
        asset_dir.mkdir(parents=True, exist_ok=True)

        traj = load_trajectory(data_dir)
        results = load_results(data_dir)
        results_per_tb[tb] = results

        plot_iteration_trajectory(
            traj, asset_dir / "plot_042026_iteration_trajectory.png", tb
        )
        plot_direction_cosines(
            data_dir / "directions.npz",
            asset_dir / "plot_042026_direction_cosines.png",
            tb,
        )
        plot_rw_by_polarity(
            results,
            asset_dir / "plot_042026_rw_by_polarity.png",
            tb,
            info["asymmetry_title"],
        )
        plot_rw_scatter(
            results, asset_dir / "plot_042026_rw1_vs_rw0_scatter.png", tb, k=1
        )
        plot_rw_scatter(
            results, asset_dir / "plot_042026_rw2_vs_rw0_scatter.png", tb, k=2
        )
        plot_per_condition_bar(
            results, asset_dir / "plot_042026_per_condition_r_bar.png", tb
        )

        groups = split_conditions(results)
        means = {}
        for gname, conds in groups.items():
            means[gname] = {f"w_{i}": float(np.mean([c[f"r_w{i}"] for c in conds])) for i in range(3)}
        summaries[tb] = means

    # Cross-tb comparison plot, saved into both asset dirs.
    for tb, info in TB_INFO.items():
        plot_tb_comparison_w2(
            results_per_tb["tb-2"],
            results_per_tb["tb-5"],
            info["asset_dir"] / "plot_042026_tb2_vs_tb5_neg_persona_w2.png",
        )

    print("\n=== Summary: mean r_wk by group ===")
    for tb, gmeans in summaries.items():
        print(f"\n{tb}:")
        for gname, ws in gmeans.items():
            print(f"  {gname}: " + ", ".join(f"{k}={v:+.3f}" for k, v in ws.items()))


if __name__ == "__main__":
    main()
