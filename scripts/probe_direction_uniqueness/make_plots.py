"""Generate four plots for the probe_direction_uniqueness experiment (L32)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/"
    "probe_direction_uniqueness/experiments/probe_science/probe_direction_uniqueness"
)
OUTPUT_DIR = EXPERIMENT_DIR / "output" / "L32"
ASSETS_DIR = EXPERIMENT_DIR / "assets"
DATE = "041826"


def load_trajectory() -> dict:
    with open(OUTPUT_DIR / "trajectory.json") as f:
        return json.load(f)


def plot_heldout_r_vs_iter(traj: dict) -> Path:
    iters = [t["iter"] for t in traj["trajectory"]]
    sweep_r = [t["sweep_r"] for t in traj["trajectory"]]
    final_r = [t["final_r"] for t in traj["trajectory"]]
    final_acc = [t["final_acc"] for t in traj["trajectory"]]
    hoo_mean_r = [t["hoo_mean_r"] for t in traj["trajectory"]]

    shuffled_final_rs = [s["final_r"] for s in traj["shuffled_runs"]]
    r_chance_low = min(shuffled_final_rs)
    r_chance_high = max(shuffled_final_rs)
    r_chance = traj["r_chance"]
    threshold = traj["threshold"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhspan(r_chance_low, r_chance_high, alpha=0.15, color="gray",
               label=f"shuffled r range [{r_chance_low:.3f}, {r_chance_high:.3f}]")
    ax.axhline(r_chance, color="gray", linestyle=":", linewidth=1,
               label=f"r_chance = {r_chance:.3f}")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"threshold = {threshold:.3f}")

    ax.plot(iters, sweep_r, marker="o", label="sweep_r")
    ax.plot(iters, final_r, marker="s", label="final_r")
    ax.plot(iters, final_acc, marker="^", label="final_acc (pairwise)")
    ax.plot(iters, hoo_mean_r, marker="D", label="hoo_mean_r")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pearson r / accuracy")
    ax.set_title("L32 — Iterated probe projection: heldout and HOO performance")
    ax.set_xticks(iters)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    path = ASSETS_DIR / f"plot_{DATE}_heldout_r_vs_iter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_alpha_vs_iter(traj: dict) -> Path:
    iters = [t["iter"] for t in traj["trajectory"]]
    alphas = [t["best_alpha"] for t in traj["trajectory"]]
    hit_upper = [t["alpha_hit_upper"] for t in traj["trajectory"]]
    alpha_hi = traj["alpha_grid_hi"]
    alpha_lo = traj["alpha_grid_lo"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, alphas, marker="o", linestyle="-", color="C0")

    for i, (it, a, hit) in enumerate(zip(iters, alphas, hit_upper)):
        if hit:
            ax.annotate("hit upper bound!", xy=(it, a),
                        xytext=(5, 5), textcoords="offset points",
                        color="red", fontsize=10)

    ax.axhline(alpha_hi, color="red", linestyle="--", alpha=0.5,
               label=f"grid upper = {alpha_hi:g}")
    ax.axhline(alpha_lo, color="gray", linestyle="--", alpha=0.5,
               label=f"grid lower = {alpha_lo:g}")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\alpha^*$ (selected ridge penalty, log scale)")
    ax.set_title("L32 — Selected Ridge α per iteration")
    ax.set_xticks(iters)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    path = ASSETS_DIR / f"plot_{DATE}_alpha_vs_iter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_direction_cosines(traj: dict) -> tuple[Path, np.ndarray]:
    data = np.load(OUTPUT_DIR / "directions.npz")
    K = traj["K_actual"]
    W = np.stack([data[f"w_{k}"] for k in range(K)], axis=0)  # (K, d)
    W_unit = W / np.linalg.norm(W, axis=1, keepdims=True)
    cos_matrix = W_unit @ W_unit.T  # (K, K)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cos_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(K):
        for j in range(K):
            val = cos_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2e}" if abs(val) < 0.01 else f"{val:.3f}",
                    ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"$\\hat w_{i}$" for i in range(K)])
    ax.set_yticklabels([f"$\\hat w_{i}$" for i in range(K)])
    ax.set_title("L32 — Pairwise cosine between probe directions (raw space)")
    fig.colorbar(im, ax=ax, label="cosine similarity")

    path = ASSETS_DIR / f"plot_{DATE}_direction_cosines.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path, cos_matrix


def plot_residual_variance_vs_iter(traj: dict) -> Path:
    iters = [t["iter"] for t in traj["trajectory"]]
    resid = [t["residual_trace"] for t in traj["trajectory"]]
    d = traj["d"]

    expected = [d - (it + 1) for it in iters]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, resid, marker="o", linestyle="-", label="observed residual trace")
    ax.plot(iters, expected, marker="x", linestyle="--", color="gray",
            label=f"expected: d − k = {d} − (iter+1)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|X_{\mathrm{scaled}}(I - WW^T)\|_F^2 / N$")
    ax.set_title("L32 — Residual variance in standardized space per iteration")
    ax.set_xticks(iters)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    y_min = min(min(resid), min(expected)) - 2
    y_max = max(max(resid), max(expected)) + 2
    ax.set_ylim(y_min, y_max)

    for it, r, e in zip(iters, resid, expected):
        ax.annotate(f"{r:.2f}", xy=(it, r), xytext=(5, 8),
                    textcoords="offset points", fontsize=8)

    path = ASSETS_DIR / f"plot_{DATE}_residual_variance_vs_iter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    traj = load_trajectory()

    p1 = plot_heldout_r_vs_iter(traj)
    p2 = plot_alpha_vs_iter(traj)
    p3, cos_matrix = plot_direction_cosines(traj)
    p4 = plot_residual_variance_vs_iter(traj)

    print(f"Plot 1 saved: {p1}")
    print(f"Plot 2 saved: {p2}")
    print(f"Plot 3 saved: {p3}")
    print(f"Plot 4 saved: {p4}")
    print("\nCosine matrix (raw space):")
    print(cos_matrix)
    max_offdiag = np.max(np.abs(cos_matrix - np.eye(cos_matrix.shape[0])))
    print(f"Max |off-diagonal| cosine (raw space): {max_offdiag:.3e}")


if __name__ == "__main__":
    main()
