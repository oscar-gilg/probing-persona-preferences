"""Plots consolidated tb-2 and tb-5 reports for probe direction uniqueness.

Per-persona aggregated view (20 personas: 17 exp1b + villain/midwest/sadist).
"""

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
# baseline=star, pos_persona=triangle_up, neg_persona=triangle_down, exp2/3=circle
SHAPE_FOR = {"baseline": "*", "pos": "^", "neg": "v", "exp23": "o"}

EXP23_ORDER = [("2", "villain"), ("2", "midwest"), ("3", "sadist")]


def load_results(data_dir: Path) -> dict:
    with open(data_dir / "results.json") as f:
        return json.load(f)


def load_trajectory(data_dir: Path) -> dict:
    with open(data_dir / "trajectory.json") as f:
        return json.load(f)


def categorize(p: dict) -> str:
    """Return 'baseline', 'pos', 'neg', or 'exp23'."""
    if p["is_baseline"]:
        return "baseline"
    if p["experiment"] == "1b":
        if p["persona"].endswith("_pos_persona"):
            return "pos"
        if p["persona"].endswith("_neg_persona"):
            return "neg"
        raise ValueError(f"Unexpected exp 1b persona: {p['persona']}")
    return "exp23"


def ordered_personas(results: dict) -> list[dict]:
    """Order: baseline, then (topic pos, topic neg) pairs alphabetically, then villain, midwest, sadist."""
    personas = results["personas"]
    by_key = {(p["experiment"], p["persona"]): p for p in personas}

    out: list[dict] = []
    baseline = next(p for p in personas if p["is_baseline"])
    out.append(baseline)

    topics = sorted({
        p["persona"].replace("_pos_persona", "").replace("_neg_persona", "")
        for p in personas
        if p["experiment"] == "1b" and not p["is_baseline"]
    })
    for topic in topics:
        pos_key = ("1b", f"{topic}_pos_persona")
        neg_key = ("1b", f"{topic}_neg_persona")
        if pos_key in by_key:
            out.append(by_key[pos_key])
        if neg_key in by_key:
            out.append(by_key[neg_key])

    for exp, name in EXP23_ORDER:
        if (exp, name) in by_key:
            out.append(by_key[(exp, name)])

    return out


def short_label(p: dict) -> str:
    if p["is_baseline"]:
        return "baseline"
    if p["experiment"] == "1b":
        topic = p["persona"].replace("_pos_persona", "").replace("_neg_persona", "")
        suffix = "pos" if p["persona"].endswith("_pos_persona") else "neg"
        return f"{topic}_{suffix}"
    return p["persona"]


def scatter_annotation(p: dict) -> str:
    """Compact annotation for scatter points."""
    if p["is_baseline"]:
        return "base"
    if p["experiment"] == "1b":
        topic = p["persona"].replace("_pos_persona", "").replace("_neg_persona", "")
        sign = "+" if p["persona"].endswith("_pos_persona") else "-"
        return f"{topic[0]}{sign}"
    return p["persona"]


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


def split_personas(results: dict) -> dict[str, list[dict]]:
    """Split non-baseline personas into 3 analysis groups."""
    pos, neg, exp23 = [], [], []
    for p in results["personas"]:
        cat = categorize(p)
        if cat == "pos":
            pos.append(p)
        elif cat == "neg":
            neg.append(p)
        elif cat == "exp23":
            exp23.append(p)
    return {"pos_persona": pos, "neg_persona": neg, "exp2/3": exp23}


def plot_rw_by_polarity(results: dict, out_path: Path, tb: str, title: str) -> None:
    groups = split_personas(results)
    group_names = ["pos_persona", "neg_persona", "exp2/3"]
    means = {k: [] for k in group_names}
    stds = {k: [] for k in group_names}
    counts = {k: len(groups[k]) for k in group_names}
    for k in group_names:
        for wi in range(3):
            vals = np.array([p[f"r_w{wi}"] for p in groups[k]])
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
    ax.set_title(f"{title}\n(mean +/- std across personas, {tb})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_rw_scatter(results: dict, out_path: Path, tb: str, k: int) -> None:
    """Scatter r_wk (y) vs r_w0 (x), per-persona (20 points)."""
    personas = results["personas"]
    fig, ax = plt.subplots(figsize=(8, 7.5))

    # Collect points so we can build a clean legend by (exp, cat).
    by_legend: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for p in personas:
        cat = categorize(p)
        exp = p["experiment"]
        key = (exp, cat)
        by_legend.setdefault(key, []).append(
            (p["r_w0"], p[f"r_w{k}"], scatter_annotation(p))
        )

    for (exp, cat), pts in by_legend.items():
        xs = [t[0] for t in pts]
        ys = [t[1] for t in pts]
        ax.scatter(xs, ys, marker=SHAPE_FOR[cat], color=EXP_COLORS[exp],
                   s=130, edgecolor="black", linewidth=0.6,
                   label=f"exp {exp} / {cat}", alpha=0.85)

    for p in personas:
        ax.annotate(
            scatter_annotation(p),
            (p["r_w0"], p[f"r_w{k}"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )

    all_x = [p["r_w0"] for p in personas]
    all_y = [p[f"r_w{k}"] for p in personas]
    lo = min(min(all_x), min(all_y)) - 0.05
    hi = max(max(all_x), max(all_y)) + 0.05
    rng = (lo, hi)
    ax.plot(rng, rng, "--", color="grey", linewidth=1, label="y = x")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(rng)
    ax.set_ylim(rng)
    ax.set_aspect("equal")
    ax.set_xlabel("$r_{w_0}$ (canonical probe)")
    ax.set_ylabel(f"$r_{{w_{k}}}$ (iteration {k} probe)")
    ax.set_title(f"$r_{{w_{k}}}$ vs $r_{{w_0}}$ across {len(personas)} personas ({tb})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_per_persona_bar(results: dict, out_path: Path, tb: str) -> None:
    personas = ordered_personas(results)
    labels = [short_label(p) for p in personas]
    n = len(personas)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 6))
    for wi in range(3):
        vals = [p[f"r_w{wi}"] for p in personas]
        ax.bar(x + (wi - 1.5) * width, vals, width,
               color=PROBE_COLORS[wi], label=f"$r_{{w_{wi}}}$")
    rand_vals = [p["r_random_mean"] for p in personas]
    rand_err = [p["r_random_std"] for p in personas]
    ax.bar(x + 1.5 * width, rand_vals, width, yerr=rand_err, capsize=2,
           color=RANDOM_COLOR, label="random direction (mean +/- std)")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r vs. persona-conditioned scores")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"Per-persona probe alignment ({tb}, {n} personas)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_tb_comparison_per_persona(results_tb2: dict, results_tb5: dict, out_path: Path) -> None:
    """Per-persona r_w0 and r_w2 compared between tb-2 and tb-5. Two subplots; y=x reference."""
    personas_tb2 = {(p["experiment"], p["persona"]): p for p in results_tb2["personas"]}
    personas_tb5 = {(p["experiment"], p["persona"]): p for p in results_tb5["personas"]}
    common_keys = sorted(set(personas_tb2.keys()) & set(personas_tb5.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, k in zip(axes, [0, 2]):
        by_legend: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
        for key in common_keys:
            p2 = personas_tb2[key]
            p5 = personas_tb5[key]
            cat = categorize(p2)
            exp = p2["experiment"]
            by_legend.setdefault((exp, cat), []).append(
                (p2[f"r_w{k}"], p5[f"r_w{k}"], scatter_annotation(p2))
            )

        for (exp, cat), pts in by_legend.items():
            xs = [t[0] for t in pts]
            ys = [t[1] for t in pts]
            ax.scatter(xs, ys, marker=SHAPE_FOR[cat], color=EXP_COLORS[exp],
                       s=120, edgecolor="black", linewidth=0.6,
                       label=f"exp {exp} / {cat}", alpha=0.85)

        for key in common_keys:
            p2 = personas_tb2[key]
            p5 = personas_tb5[key]
            ax.annotate(
                scatter_annotation(p2),
                (p2[f"r_w{k}"], p5[f"r_w{k}"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color="black",
            )

        all_x = [personas_tb2[key][f"r_w{k}"] for key in common_keys]
        all_y = [personas_tb5[key][f"r_w{k}"] for key in common_keys]
        lo = min(min(all_x), min(all_y)) - 0.05
        hi = max(max(all_x), max(all_y)) + 0.05
        rng = (lo, hi)
        ax.plot(rng, rng, "--", color="grey", linewidth=1, label="y = x")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(rng)
        ax.set_ylim(rng)
        ax.set_aspect("equal")
        ax.set_xlabel(f"$r_{{w_{k}}}$ at tb-2")
        ax.set_ylabel(f"$r_{{w_{k}}}$ at tb-5")
        ax.set_title(f"Per-persona $r_{{w_{k}}}$: tb-2 vs tb-5")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-persona probe alignment: tb-2 vs tb-5", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def count_above_below_identity(results: dict, k: int) -> tuple[int, int, int]:
    """Return (above, below, on) count of personas w.r.t. y=x line in r_wk vs r_w0 scatter."""
    above = below = on = 0
    for p in results["personas"]:
        y = p[f"r_w{k}"]
        x = p["r_w0"]
        if y > x:
            above += 1
        elif y < x:
            below += 1
        else:
            on += 1
    return above, below, on


def group_means(results: dict) -> dict[str, dict[str, float]]:
    groups = split_personas(results)
    out = {}
    for gname, ps in groups.items():
        out[gname] = {f"w_{i}": float(np.mean([p[f"r_w{i}"] for p in ps])) for i in range(3)}
    return out


def main() -> None:
    summaries: dict[str, dict] = {}
    above_below: dict[str, dict] = {}
    results_per_tb: dict[str, dict] = {}

    for tb, info in TB_INFO.items():
        data_dir = info["data_dir"]
        asset_dir = info["asset_dir"]
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Remove obsolete plots.
        old_percond = asset_dir / "plot_042026_per_condition_r_bar.png"
        if old_percond.exists():
            old_percond.unlink()
        old_cross_tb = asset_dir / "plot_042026_tb2_vs_tb5_neg_persona_w2.png"
        if old_cross_tb.exists():
            old_cross_tb.unlink()

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
        plot_per_persona_bar(
            results, asset_dir / "plot_042026_per_persona_r_bar.png", tb
        )

        summaries[tb] = group_means(results)
        above_below[tb] = {
            "rw1_vs_rw0": count_above_below_identity(results, 1),
            "rw2_vs_rw0": count_above_below_identity(results, 2),
        }

    # Cross-tb per-persona plot saved to assets_tb-2/ only.
    plot_tb_comparison_per_persona(
        results_per_tb["tb-2"],
        results_per_tb["tb-5"],
        TB_INFO["tb-2"]["asset_dir"] / "plot_042026_tb2_vs_tb5_per_persona.png",
    )
    # Clean up the older cross-tb plot in tb-5 asset dir too (already deleted above).

    print("\n=== Summary: mean r_wk by group ===")
    for tb, gmeans in summaries.items():
        print(f"\n{tb}:")
        for gname, ws in gmeans.items():
            print(f"  {gname}: " + ", ".join(f"{key}={v:+.3f}" for key, v in ws.items()))

    print("\n=== Above / below y=x identity line ===")
    for tb, ab in above_below.items():
        print(f"\n{tb}:")
        for which, (above, below, on) in ab.items():
            print(f"  {which}: above={above}, below={below}, on={on} (of 20)")


if __name__ == "__main__":
    main()
