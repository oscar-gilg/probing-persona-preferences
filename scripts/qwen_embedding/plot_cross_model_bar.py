"""2x2 figure for §2 of the paper: probe quality vs its own content baseline, per model.

Layout:
    Rows: model family (Gemma-3-27B, Qwen-3.5-122B)
    Cols: metric (Pearson r, Pairwise accuracy)
Each panel: 2 bar groups -- target probe vs Qwen3-Emb 8B baseline trained on
that model's utilities. Each group shows test-set + HOO.

The Qwen3-Emb baseline is apples-to-apples within each row: for Gemma it is
trained on Gemma utilities using Qwen3-Emb activations over the Gemma task pool;
for Qwen it is trained on Qwen utilities using Qwen3-Emb activations over the
Qwen task pool.

HOO r is the *pooled* Pearson r: predictions from every fold's held-out topic
are concatenated into one vector and correlated with true utilities in one shot.
This rewards correct between-topic placement and is cross-model comparable
(unlike within-topic-mean r, which collapses when the model itself compresses
within-topic variance -- e.g. Qwen on safety topics).

All values are loaded from `results/probes/` JSON produced by
`src.probes.experiments.run_dir_probes`, not hardcoded. Each rendered bar is
registered with `ClaimSet.register(...)` so the paper picks up the live value
through `paper/numbers.tex`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paper.claims import ClaimSet

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "probes"


def _load_heldout(heldout_dir: Path, layer: int) -> tuple[float, float]:
    """Return (final_r, uniform_pairwise_acc) for the ridge probe at `layer`."""
    manifest = json.loads((heldout_dir / "manifest.json").read_text())
    for probe in manifest["probes"]:
        if probe["method"] == "ridge" and probe["layer"] == layer:
            return float(probe["final_r"]), float(probe["uniform_pairwise_acc"])
    raise KeyError(f"No ridge probe at layer {layer} in {heldout_dir}")


def _load_hoo(hoo_dir: Path, layer: int) -> tuple[float, float | None]:
    """Return (pooled_pearson_r, mean_uniform_hoo_acc) — second value may be None
    if the summary predates the `uniform_hoo_acc` per-fold field."""
    pooled = json.loads((hoo_dir / "pooled_metrics.json").read_text())
    r_pooled = float(pooled["pooled_pearson_r"])
    summary = json.loads((hoo_dir / "hoo_summary.json").read_text())
    ridge_entry = summary["layer_summary"][str(layer)]["ridge"]
    acc_hoo = ridge_entry.get("mean_uniform_hoo_acc")
    return r_pooled, float(acc_hoo) if acc_hoo is not None else None


# Layers and data sources. Values loaded at import-time so `main()` is thin.
GEMMA_HELDOUT_DIR = RESULTS / "heldout_eval_gemma3_tb-1"
GEMMA_HOO_DIR = RESULTS / "gemma3_10k_hoo_topic_tb-1_uniform"
GEMMA_BASELINE_HELDOUT_DIR = RESULTS / "qwen3_emb_8b_heldout_std_raw"
GEMMA_BASELINE_HOO_DIR = RESULTS / "qwen3_emb_8b_hoo_topic"
QWEN_HELDOUT_DIR = RESULTS / "qwen35_122b" / "qwen35_122b_heldout_turn_boundary_m1"
QWEN_HOO_DIR = RESULTS / "qwen35_122b" / "qwen35_122b_hoo_topic_turn_boundary_m1_L38_uniform"
QWEN_BASELINE_HELDOUT_DIR = RESULTS / "qwen3_emb_8b_qwen35_heldout_std_raw"
QWEN_BASELINE_HOO_DIR = RESULTS / "qwen3_emb_8b_qwen35_hoo_topic"

GEMMA_LAYER = 32
QWEN_LAYER = 38
QWEN3_EMB_LAYER = 0  # Sentence-transformer baselines are single-layer.

# Gemma = blue, Qwen = coral, baseline = gold.
PROBE_LIGHT = {"gemma": "#7aaed4", "qwen": "#e89284"}
PROBE_DARK  = {"gemma": "#3d7aab", "qwen": "#b0533f"}
BASE_LIGHT  = "#d4b96a"
BASE_DARK   = "#b08f3a"

# Rounding used for both plot annotations and registered claim values.
ROUND_DP = 3


def _r(x: float) -> float:
    return round(float(x), ROUND_DP)


def _draw_panel(ax, probe, baseline, metric, model_key, title, ylabel):
    """Draw one panel with 2 bar groups (probe, baseline), each with test+hoo sub-bars."""
    test_key = f"{metric}_heldout"
    hoo_key  = f"{metric}_hoo"

    groups = ["Target probe", "Qwen3-Emb\nbaseline"]
    x = np.arange(len(groups))
    width = 0.35

    probe_light = PROBE_LIGHT[model_key]
    probe_dark  = PROBE_DARK[model_key]
    ax.bar(x[0] - width / 2, probe[test_key], width,
           color=probe_light, edgecolor="grey", linewidth=0.5)
    ax.bar(x[0] + width / 2, probe[hoo_key], width,
           color=probe_dark, edgecolor="grey", linewidth=0.5)
    ax.text(x[0] - width / 2, probe[test_key] + 0.015,
            f'{probe[test_key]:.2f}', ha="center", va="bottom", fontsize=9)
    ax.text(x[0] + width / 2, probe[hoo_key] + 0.015,
            f'{probe[hoo_key]:.2f}', ha="center", va="bottom", fontsize=9)

    ax.bar(x[1] - width / 2, baseline[test_key], width,
           color=BASE_LIGHT, edgecolor="grey", linewidth=0.5)
    ax.text(x[1] - width / 2, baseline[test_key] + 0.015,
            f'{baseline[test_key]:.2f}', ha="center", va="bottom", fontsize=9)

    ax.bar(x[1] + width / 2, baseline[hoo_key], width,
           color=BASE_DARK, edgecolor="grey", linewidth=0.5)
    ax.text(x[1] + width / 2, baseline[hoo_key] + 0.015,
            f'{baseline[hoo_key]:.2f}', ha="center", va="bottom", fontsize=9)

    if metric == "acc":
        ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_stats_and_claims(claims: ClaimSet) -> tuple[dict, dict, dict, dict]:
    """Load stats from disk, register every rendered value as a claim, return stats."""

    gp_r_held, gp_acc_held = _load_heldout(GEMMA_HELDOUT_DIR, GEMMA_LAYER)
    gp_r_hoo, gp_acc_hoo = _load_hoo(GEMMA_HOO_DIR, GEMMA_LAYER)

    qp_r_held, qp_acc_held = _load_heldout(QWEN_HELDOUT_DIR, QWEN_LAYER)
    qp_r_hoo, qp_acc_hoo = _load_hoo(QWEN_HOO_DIR, QWEN_LAYER)

    gb_r_held, gb_acc_held = _load_heldout(GEMMA_BASELINE_HELDOUT_DIR, QWEN3_EMB_LAYER)
    gb_r_hoo, gb_acc_hoo = _load_hoo(GEMMA_BASELINE_HOO_DIR, QWEN3_EMB_LAYER)

    qb_r_held, qb_acc_held = _load_heldout(QWEN_BASELINE_HELDOUT_DIR, QWEN3_EMB_LAYER)
    qb_r_hoo, qb_acc_hoo = _load_hoo(QWEN_BASELINE_HOO_DIR, QWEN3_EMB_LAYER)

    # Gemma probe.
    gemma_probe = {
        "r_heldout": claims.register(
            "Gemma probe heldout r",
            _r(gp_r_held),
            "A ridge probe on Gemma-3-27B residual-stream activations at the "
            "final prompt token (layer 32) predicts held-out Thurstonian "
            "utilities at Pearson r on a within-distribution eval split.",
            used_in=["fig:cross-model", "abstract", "sec:shared.intro", "sec:probe-methods"],
        ),
        "r_hoo": claims.register(
            "Gemma probe cross-topic pooled r",
            _r(gp_r_hoo),
            "Leave-one-topic-out pooled Pearson r for the Gemma-3-27B ridge probe "
            "(L32, tb-1): predictions from every fold's held-out topic are "
            "concatenated and correlated with true Thurstonian utilities.",
            used_in=["fig:cross-model", "abstract", "sec:shared.intro"],
        ),
        "acc_heldout": claims.register(
            "Gemma probe heldout uniform pairwise accuracy",
            _r(gp_acc_held),
            "Uniform-sample pairwise accuracy of the Gemma-3-27B ridge probe "
            "(L32) on the within-distribution held-out eval split.",
            used_in=["fig:cross-model"],
        ),
    }
    if gp_acc_hoo is not None:
        gemma_probe["acc_hoo"] = claims.register(
            "Gemma probe cross-topic uniform pairwise accuracy",
            _r(gp_acc_hoo),
            "Mean across 13 leave-one-topic-out folds of uniform-sample pairwise "
            "accuracy restricted to the held-out topic, for the Gemma-3-27B "
            "ridge probe (L32).",
            used_in=["fig:cross-model"],
        )
    else:
        # Gemma's hoo_summary.json predates the per-fold uniform_hoo_acc field.
        # Freeze the previously-published value; backfill is in paper/TODO_producers.md.
        gemma_probe["acc_hoo"] = claims.register(
            "Gemma probe cross-topic uniform pairwise accuracy",
            0.751,
            "Mean across 13 leave-one-topic-out folds of uniform-sample pairwise "
            "accuracy restricted to the held-out topic, for the Gemma-3-27B "
            "ridge probe (L32). Value carried over from an earlier pipeline "
            "run; the current hoo_summary.json lacks the per-fold uniform_hoo_acc "
            "field needed to recompute it here.",
            used_in=["fig:cross-model"],
            source="manual: Gemma hoo_summary predates uniform_hoo_acc field; see paper/TODO_producers.md",
        )

    # Qwen probe.
    qwen_probe = {
        "r_heldout": claims.register(
            "Qwen probe heldout r",
            _r(qp_r_held),
            "A ridge probe on Qwen-3.5-122B residual-stream activations at the "
            "turn-boundary tb-1 token (layer 38) predicts held-out Thurstonian "
            "utilities at Pearson r on a within-distribution eval split.",
            used_in=["fig:cross-model", "sec:shared.intro"],
        ),
        "r_hoo": claims.register(
            "Qwen probe cross-topic pooled r",
            _r(qp_r_hoo),
            "Leave-one-topic-out pooled Pearson r for the Qwen-3.5-122B ridge "
            "probe (L38, tb-1).",
            used_in=["fig:cross-model", "sec:shared.intro"],
        ),
        "acc_heldout": claims.register(
            "Qwen probe heldout uniform pairwise accuracy",
            _r(qp_acc_held),
            "Uniform-sample pairwise accuracy of the Qwen-3.5-122B ridge probe "
            "(L38) on the within-distribution held-out eval split.",
            used_in=["fig:cross-model"],
        ),
        "acc_hoo": claims.register(
            "Qwen probe cross-topic uniform pairwise accuracy",
            _r(qp_acc_hoo),
            "Mean across leave-one-topic-out folds of uniform-sample pairwise "
            "accuracy restricted to the held-out topic, for the Qwen-3.5-122B "
            "ridge probe (L38).",
            used_in=["fig:cross-model"],
        ),
    }

    # Gemma content baseline (Qwen3-Emb-8B trained on Gemma utilities).
    gemma_baseline = {
        "r_heldout": claims.register(
            "Gemma content baseline heldout r",
            _r(gb_r_held),
            "A Qwen3-Embedding-8B sentence-transformer content baseline, "
            "trained as a ridge probe on Gemma-3-27B Thurstonian utilities, "
            "evaluated on the within-distribution held-out eval split.",
            used_in=["fig:cross-model"],
        ),
        "r_hoo": claims.register(
            "Gemma content baseline cross-topic pooled r",
            _r(gb_r_hoo),
            "Leave-one-topic-out pooled Pearson r for the Qwen3-Embedding-8B "
            "content baseline trained on Gemma utilities.",
            used_in=["fig:cross-model"],
        ),
        "acc_heldout": claims.register(
            "Gemma content baseline heldout uniform pairwise accuracy",
            _r(gb_acc_held),
            "Uniform-sample pairwise accuracy of the Qwen3-Embedding-8B "
            "content baseline on the Gemma held-out eval split.",
            used_in=["fig:cross-model"],
        ),
        "acc_hoo": claims.register(
            "Gemma content baseline cross-topic uniform pairwise accuracy",
            _r(gb_acc_hoo),
            "Mean across leave-one-topic-out folds of uniform-sample pairwise "
            "accuracy restricted to the held-out topic, for the Qwen3-Embedding-8B "
            "content baseline trained on Gemma utilities.",
            used_in=["fig:cross-model"],
        ),
    }

    # Qwen content baseline (Qwen3-Emb-8B trained on Qwen utilities).
    qwen_baseline = {
        "r_heldout": claims.register(
            "Qwen content baseline heldout r",
            _r(qb_r_held),
            "A Qwen3-Embedding-8B sentence-transformer content baseline trained "
            "on Qwen-3.5-122B Thurstonian utilities, evaluated on the "
            "within-distribution held-out eval split.",
            used_in=["fig:cross-model"],
        ),
        "r_hoo": claims.register(
            "Qwen content baseline cross-topic pooled r",
            _r(qb_r_hoo),
            "Leave-one-topic-out pooled Pearson r for the Qwen3-Embedding-8B "
            "content baseline trained on Qwen utilities.",
            used_in=["fig:cross-model"],
        ),
        "acc_heldout": claims.register(
            "Qwen content baseline heldout uniform pairwise accuracy",
            _r(qb_acc_held),
            "Uniform-sample pairwise accuracy of the Qwen3-Embedding-8B "
            "content baseline on the Qwen held-out eval split.",
            used_in=["fig:cross-model"],
        ),
        "acc_hoo": claims.register(
            "Qwen content baseline cross-topic uniform pairwise accuracy",
            _r(qb_acc_hoo),
            "Mean across leave-one-topic-out folds of uniform-sample pairwise "
            "accuracy restricted to the held-out topic, for the Qwen3-Embedding-8B "
            "content baseline trained on Qwen utilities.",
            used_in=["fig:cross-model"],
        ),
    }

    return gemma_probe, gemma_baseline, qwen_probe, qwen_baseline


def plot_cross_model(out_path: Path) -> None:
    claims = ClaimSet(source="scripts/qwen_embedding/plot_cross_model_bar.py")
    gemma_probe, gemma_baseline, qwen_probe, qwen_baseline = build_stats_and_claims(claims)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Probe vs content baseline, per model", fontsize=14, fontweight="bold")

    _draw_panel(axes[0, 0], gemma_probe, gemma_baseline, "r",   "gemma",
                "Gemma-3-27B — Pearson r", "Pearson r")
    _draw_panel(axes[0, 1], gemma_probe, gemma_baseline, "acc", "gemma",
                "Gemma-3-27B — Pairwise accuracy", "Pairwise accuracy")
    _draw_panel(axes[1, 0], qwen_probe,  qwen_baseline,  "r",   "qwen",
                "Qwen-3.5-122B — Pearson r", "Pearson r")
    _draw_panel(axes[1, 1], qwen_probe,  qwen_baseline,  "acc", "qwen",
                "Qwen-3.5-122B — Pairwise accuracy", "Pairwise accuracy")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#bfbfbf"),
        plt.Rectangle((0, 0), 1, 1, fc="#4d4d4d"),
    ]
    fig.legend(legend_handles, ["Test set", "Cross-topic (HOO, pooled)"],
               loc="upper right", bbox_to_anchor=(0.98, 0.965), fontsize=9, frameon=True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sidecar = REPO_ROOT / "paper" / "claims" / "plot_cross_model_bar.json"
    claims.save(sidecar)
    print(f"Saved {out_path}")
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")


if __name__ == "__main__":
    plot_cross_model(REPO_ROOT / "paper" / "figures" / "plot_041726_cross_model_bar.png")
