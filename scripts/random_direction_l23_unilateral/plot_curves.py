"""Plot the unilateral L23 random-direction null next to the contrastive null.

Reads:
    experiments/random_direction_l23_unilateral/agg.json    (this expt, 5 seeds)
    experiments/random_direction_l23_quick/multi_seed/checkpoints/random_contrastive_seed{0,1}.parsed.jsonl
        (parent contrastive null — seeds 0, 1 only on this pod)

Writes:
    experiments/random_direction_l23_unilateral/assets/plot_<MMDDYY>_unilateral_vs_contrastive_null.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/random_direction_l23_unilateral")
ASSET_DIR = EXP_DIR / "assets"
PARENT_DIR = Path("experiments/random_direction_l23_quick/multi_seed/checkpoints")


def _effective_coef(coef: float, ordering: int) -> float:
    return coef if ordering == 0 else -coef


def load_contrastive(path: Path) -> dict:
    """Returns {applied_coef: {seed_p_chose_higher_utility, n_resp, n_chose_a, n_total, n_refuse}}.

    For a contrastive run with spans {first: 1, second: -1}: under the canonical
    frame, x = applied coef on the steered task. Each row contributes two points:
    (applied=+signed_multiplier with chose_steered=(choice=='a'))
    and (applied=-signed_multiplier with chose_steered=(choice=='b')).
    Equivalently, when collapsing across orderings via the ordering-flip,
    `signed_multiplier` is the spec-level "+ direction goes to higher-utility a"
    and `applied_to_a = signed_multiplier`. We bin by signed_multiplier
    and report P(chose 'a' | responded).
    """
    out: dict = defaultdict(lambda: {"n_total": 0, "n_resp": 0, "n_a": 0, "n_b": 0, "n_refuse": 0})
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            mult = round(float(row["signed_multiplier"]), 4)
            choice = row["choice_original"]
            b = out[mult]
            b["n_total"] += 1
            if choice == "a":
                b["n_a"] += 1
                b["n_resp"] += 1
            elif choice == "b":
                b["n_b"] += 1
                b["n_resp"] += 1
            else:
                b["n_refuse"] += 1
    return dict(out)


def contrastive_curves(seed_paths: list[tuple[int, Path]]) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Returns (x_coefs, p_per_seed (S, K), refusal_per_seed (S, K)) under the canonical frame.

    Canonical frame: x = applied coef on steered task. Each ordered trial produces
    two mirror points. Equivalent in expectation to plotting x = signed_multiplier
    against P(chose 'a') because contrastive is anti-symmetric — but to be exact
    we mirror as the runner doc instructs.
    """
    all_coefs: set[float] = set()
    per_seed: list[tuple[int, dict]] = []
    for seed, path in seed_paths:
        agg = load_contrastive(path)
        per_seed.append((seed, agg))
        all_coefs.update(agg.keys())
    # Build the canonical (mirrored) buckets
    coefs = sorted(all_coefs | {-c for c in all_coefs})
    p_seeds = []
    r_seeds = []
    for seed, agg in per_seed:
        # Build mirror: at applied=+m, chose_steered = chose 'a'; at applied=-m, chose_steered = chose 'b'.
        bucket: dict[float, dict] = defaultdict(lambda: {"n_resp": 0, "n_steered": 0, "n_total": 0, "n_refuse": 0})
        for m, b in agg.items():
            bp = bucket[m]
            bp["n_resp"] += b["n_resp"]
            bp["n_steered"] += b["n_a"]
            bp["n_total"] += b["n_total"]
            bp["n_refuse"] += b["n_refuse"]
            bm = bucket[-m]
            bm["n_resp"] += b["n_resp"]
            bm["n_steered"] += b["n_b"]
            bm["n_total"] += b["n_total"]
            bm["n_refuse"] += b["n_refuse"]
        ps = []
        rs = []
        for c in coefs:
            b = bucket.get(c)
            if b is None or b["n_resp"] == 0:
                ps.append(np.nan)
                rs.append(np.nan)
            else:
                ps.append(b["n_steered"] / b["n_resp"])
                rs.append(b["n_refuse"] / b["n_total"])
        p_seeds.append(ps)
        r_seeds.append(rs)
    return coefs, np.array(p_seeds), np.array(r_seeds)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Unilateral (this experiment) ----
    agg = json.loads((EXP_DIR / "agg.json").read_text())
    seeds_data = agg["seeds"]
    pooled = agg["pooled_across_seeds"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    coefs_uni: list[float] = []
    if pooled:
        coefs_uni = sorted({d["applied_coef"] for d in pooled["unilateral_first"].values()})

    cond_colors = {
        "unilateral_first": "#1f77b4",
        "unilateral_second": "#d62728",
    }
    cond_labels = {
        "unilateral_first": "first task steered",
        "unilateral_second": "second task steered",
    }

    ax = axes[0]
    seed_keys = sorted(seeds_data.keys(), key=lambda s: int(s))
    # Per-seed thin lines
    for s in seed_keys:
        for cond in ("unilateral_first", "unilateral_second"):
            d = seeds_data[s][cond]
            xs = sorted({v["applied_coef"] for v in d.values()})
            ys = [d[f"{x:+.2f}"]["p_chose_steered"] for x in xs]
            ax.plot(xs, ys, color=cond_colors[cond], alpha=0.25, lw=1.0)
    # Mean across seeds
    for cond in ("unilateral_first", "unilateral_second"):
        xs = sorted({v["applied_coef"] for v in pooled[cond].values()})
        means = [pooled[cond][f"{x:+.2f}"]["mean_p_chose_steered"] for x in xs]
        sems = [pooled[cond][f"{x:+.2f}"]["sem_p_chose_steered"] for x in xs]
        ax.errorbar(xs, means, yerr=sems, color=cond_colors[cond], lw=2.0, marker="o",
                    label=cond_labels[cond], capsize=2)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("applied coefficient (× mean_norm[L23])")
    ax.set_ylabel("P(chose steered task | responded)")
    ax.set_title(f"Unilateral random-direction L23 null\n5 seeds × 30 pairs (this experiment)")
    ax.set_ylim(0, 1)
    ax.set_xlim(min(coefs_uni) - 0.005, max(coefs_uni) + 0.005)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Contrastive parent (seeds 0, 1) ----
    parent_paths = []
    for s in [0, 1]:
        p = PARENT_DIR / f"random_contrastive_seed{s}.parsed.jsonl"
        if p.exists():
            parent_paths.append((s, p))
    ax = axes[1]
    if parent_paths:
        x_c, p_seeds, r_seeds = contrastive_curves(parent_paths)
        for i in range(p_seeds.shape[0]):
            ax.plot(x_c, p_seeds[i], color="#2ca02c", alpha=0.25, lw=1.0)
        mean_c = np.nanmean(p_seeds, axis=0)
        sem_c = np.nanstd(p_seeds, axis=0, ddof=1) / np.sqrt(p_seeds.shape[0]) if p_seeds.shape[0] > 1 else np.zeros_like(mean_c)
        ax.errorbar(x_c, mean_c, yerr=sem_c, color="#2ca02c", lw=2.0, marker="o",
                    label="contrastive (parent)", capsize=2)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("applied coefficient (× mean_norm[L23])")
    ax.set_title(f"Contrastive random-direction L23 null\n2 seeds × 30 pairs (parent expt, on-pod)")
    ax.set_xlim(min(coefs_uni) - 0.005, max(coefs_uni) + 0.005)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Random-direction null: unilateral (single-task) vs contrastive (Fig 3a)", y=1.0)
    fig.tight_layout()

    stamp = datetime.now().strftime("%m%d%y")
    out = ASSET_DIR / f"plot_{stamp}_unilateral_vs_contrastive_null.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
