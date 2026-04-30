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
from math import atanh, sqrt, tanh
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from corroborate import ClaimSet


def fisher_z_ci(r: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Fisher-z 95% CI on a Pearson correlation."""
    if n is None or n < 4 or r is None:
        return float("nan"), float("nan")
    # Clamp r to avoid atanh blowing up at exactly ±1.
    r_safe = max(min(r, 0.999999), -0.999999)
    zhat = atanh(r_safe)
    se = 1.0 / sqrt(n - 3)
    return tanh(zhat - z * se), tanh(zhat + z * se)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI on a binomial proportion given a rate p and sample size n."""
    if n is None or n <= 0 or p is None:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def fold_se_ci(mean: float, std_across_folds: float, n_folds: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI from across-fold SE: mean ± 1.96·std/sqrt(n_folds). Clipped to [0,1]."""
    if n_folds is None or n_folds < 2 or std_across_folds is None or mean is None:
        return float("nan"), float("nan")
    se = std_across_folds / sqrt(n_folds)
    return max(0.0, mean - z * se), min(1.0, mean + z * se)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "probes"


def _load_heldout(heldout_dir: Path, layer: int) -> tuple[float, float, int, int]:
    """Return (final_r, uniform_pairwise_acc, n_final, n_final_pairs) for the ridge probe at `layer`."""
    manifest = json.loads((heldout_dir / "manifest.json").read_text())
    for probe in manifest["probes"]:
        if probe["method"] == "ridge" and probe["layer"] == layer:
            return (
                float(probe["final_r"]),
                float(probe["uniform_pairwise_acc"]),
                int(probe["n_final"]),
                int(probe["n_final_pairs"]),
            )
    raise KeyError(f"No ridge probe at layer {layer} in {heldout_dir}")


def _load_hoo(hoo_dir: Path, layer: int) -> dict:
    """Return clean separate-run LOO metrics if `pooled_metrics_clean.json` exists,
    else fall back to same-run pooled metrics + per-fold uniform accuracy.

    Returned dict has keys: r, acc, n_r, n_acc.
    """
    clean_path = hoo_dir / "pooled_metrics_clean.json"
    if clean_path.exists():
        clean = json.loads(clean_path.read_text())
        sep = clean["separate_run"]
        return {
            "r": float(sep["pooled_pearson_r"]),
            "acc": float(sep["pooled_pairwise_acc"]),
            "n_r": int(sep["n_pooled"]),
            "n_acc": int(sep["n_pairs"]),
        }
    # Fallback: same-run pooled r, per-fold mean accuracy
    pooled = json.loads((hoo_dir / "pooled_metrics.json").read_text())
    summary = json.loads((hoo_dir / "hoo_summary.json").read_text())
    ridge_entry = summary["layer_summary"][str(layer)]["ridge"]
    return {
        "r": float(pooled["pooled_pearson_r"]),
        "acc": float(ridge_entry.get("mean_uniform_hoo_acc")) if ridge_entry.get("mean_uniform_hoo_acc") is not None else None,
        "n_r": int(pooled["n_pooled"]),
        "n_acc": None,
    }


# Layers and data sources. Values loaded at import-time so `main()` is thin.
GEMMA_HELDOUT_DIR = RESULTS / "heldout_eval_gemma3_tb-5"
GEMMA_HOO_DIR = RESULTS / "gemma3_10k_hoo_topic_tb-5"
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
    """Draw one panel with 2 bar groups (probe, baseline), each with test+hoo sub-bars.

    For metric=='r', overlay Fisher-z 95% CIs on each bar using ``n_heldout`` and
    ``n_hoo``. For metric=='acc', overlay Wilson 95% CIs on test-set bars using
    ``n_heldout_pairs``, and across-fold SE 95% CIs on LOO bars using
    ``acc_hoo_std`` and ``n_folds``.
    """
    test_key = f"{metric}_heldout"
    hoo_key  = f"{metric}_hoo"

    groups = ["Target probe", "Qwen3-Emb\nbaseline"]
    x = np.arange(len(groups))
    width = 0.35

    def err_test(value: float, source: dict) -> tuple[float, float] | None:
        if metric == "r":
            n = source.get("n_heldout")
            if n is None:
                return None
            lo, hi = fisher_z_ci(value, n)
        else:  # accuracy
            n = source.get("n_heldout_pairs")
            if n is None:
                return None
            lo, hi = wilson_ci(value, n)
        return value - lo, hi - value

    def err_hoo(value: float, source: dict) -> tuple[float, float] | None:
        if metric == "r":
            n = source.get("n_hoo")
            if n is None:
                return None
            lo, hi = fisher_z_ci(value, n)
        else:  # accuracy
            n_pairs = source.get("n_hoo_pairs")
            if n_pairs is not None:
                # Clean separate-run pooled accuracy: Wilson CI on the proportion.
                lo, hi = wilson_ci(value, n_pairs)
            else:
                # Fallback: across-fold SE on mean per-topic accuracy.
                std = source.get("acc_hoo_std")
                n_folds = source.get("n_folds")
                if std is None or not n_folds:
                    return None
                lo, hi = fold_se_ci(value, std, n_folds)
        return value - lo, hi - value

    def err(value, source, kind):
        return err_test(value, source) if kind == "test" else err_hoo(value, source)

    probe_light = PROBE_LIGHT[model_key]
    probe_dark  = PROBE_DARK[model_key]
    bars = [
        (x[0] - width / 2, probe[test_key], probe_light, err(probe[test_key], probe, "test")),
        (x[0] + width / 2, probe[hoo_key], probe_dark, err(probe[hoo_key], probe, "hoo")),
        (x[1] - width / 2, baseline[test_key], BASE_LIGHT, err(baseline[test_key], baseline, "test")),
        (x[1] + width / 2, baseline[hoo_key], BASE_DARK, err(baseline[hoo_key], baseline, "hoo")),
    ]
    for xi, value, color, ci_err in bars:
        ax.bar(xi, value, width, color=color, edgecolor="grey", linewidth=0.5)
        if ci_err is not None:
            ax.errorbar(xi, value, yerr=[[ci_err[0]], [ci_err[1]]],
                        fmt="none", ecolor="black", capsize=3, elinewidth=0.9)
        text_y = value + (ci_err[1] if ci_err else 0) + 0.015
        ax.text(xi, text_y, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

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


def _rel(path: Path) -> str:
    """Produce a repo-relative string for use in data_paths."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_stats_and_claims(claims: ClaimSet) -> tuple[dict, dict, dict, dict]:
    """Load stats from disk, register every rendered value as a claim, return stats.

    Cited individually in the paper (remain as scalar claims):
      - Gemma probe heldout r            -> \\gemmaProbeHeldoutR
      - Gemma probe cross-topic pooled r -> \\gemmaProbeCrossTopicPooledR
      - Qwen probe heldout r             -> \\qwenProbeHeldoutR
      - Qwen probe cross-topic pooled r  -> \\qwenProbeCrossTopicPooledR
      - Qwen probe heldout uniform pairwise accuracy -> \\qwenProbeHeldoutUniformPairwiseAccuracy

    Un-cited bar values are collapsed into two structured claims:
      - "Cross model content baseline" (2-D table, 8 cells)
      - "Cross model probe extras"     (1-D row,  3 cells)
    """

    gp_r_held, gp_acc_held, gp_n_held, gp_n_held_pairs = _load_heldout(GEMMA_HELDOUT_DIR, GEMMA_LAYER)
    gp_hoo = _load_hoo(GEMMA_HOO_DIR, GEMMA_LAYER)

    qp_r_held, qp_acc_held, qp_n_held, qp_n_held_pairs = _load_heldout(QWEN_HELDOUT_DIR, QWEN_LAYER)
    qp_hoo = _load_hoo(QWEN_HOO_DIR, QWEN_LAYER)

    gb_r_held, gb_acc_held, gb_n_held, gb_n_held_pairs = _load_heldout(GEMMA_BASELINE_HELDOUT_DIR, QWEN3_EMB_LAYER)
    gb_hoo = _load_hoo(GEMMA_BASELINE_HOO_DIR, QWEN3_EMB_LAYER)

    qb_r_held, qb_acc_held, qb_n_held, qb_n_held_pairs = _load_heldout(QWEN_BASELINE_HELDOUT_DIR, QWEN3_EMB_LAYER)
    qb_hoo = _load_hoo(QWEN_BASELINE_HOO_DIR, QWEN3_EMB_LAYER)

    # Unpack into the legacy variable names used below.
    gp_r_hoo, gp_acc_hoo, gp_n_hoo, gp_n_hoo_pairs = gp_hoo["r"], gp_hoo["acc"], gp_hoo["n_r"], gp_hoo["n_acc"]
    qp_r_hoo, qp_acc_hoo, qp_n_hoo, qp_n_hoo_pairs = qp_hoo["r"], qp_hoo["acc"], qp_hoo["n_r"], qp_hoo["n_acc"]
    gb_r_hoo, gb_acc_hoo, gb_n_hoo, gb_n_hoo_pairs = gb_hoo["r"], gb_hoo["acc"], gb_hoo["n_r"], gb_hoo["n_acc"]
    qb_r_hoo, qb_acc_hoo, qb_n_hoo, qb_n_hoo_pairs = qb_hoo["r"], qb_hoo["acc"], qb_hoo["n_r"], qb_hoo["n_acc"]

    if gp_acc_hoo is None:
        raise RuntimeError(
            "Gemma hoo_summary.json is missing mean_uniform_hoo_acc — rerun "
            "src.probes.experiments.run_dir_probes on configs/probes/gemma3_10k_hoo_topic_tb-5.yaml"
        )

    _gp_heldout_path = _rel(GEMMA_HELDOUT_DIR / "manifest.json")
    _gp_hoo_pooled = _rel(GEMMA_HOO_DIR / "pooled_metrics.json")
    _gp_hoo_summary = _rel(GEMMA_HOO_DIR / "hoo_summary.json")
    _qp_heldout_path = _rel(QWEN_HELDOUT_DIR / "manifest.json")
    _qp_hoo_pooled = _rel(QWEN_HOO_DIR / "pooled_metrics.json")
    _qp_hoo_summary = _rel(QWEN_HOO_DIR / "hoo_summary.json")
    _gb_heldout_path = _rel(GEMMA_BASELINE_HELDOUT_DIR / "manifest.json")
    _gb_hoo_pooled = _rel(GEMMA_BASELINE_HOO_DIR / "pooled_metrics.json")
    _gb_hoo_summary = _rel(GEMMA_BASELINE_HOO_DIR / "hoo_summary.json")
    _qb_heldout_path = _rel(QWEN_BASELINE_HELDOUT_DIR / "manifest.json")
    _qb_hoo_pooled = _rel(QWEN_BASELINE_HOO_DIR / "pooled_metrics.json")
    _qb_hoo_summary = _rel(QWEN_BASELINE_HOO_DIR / "hoo_summary.json")

    # --- Cited scalar claims (macro names must not change) ---
    # NOTE: Gemma heldout r is registered by compute_position_sweep.py as
    # "Position sweep Gemma EndOfTurn best r" (best-layer over the sweep, same
    # manifest) — strictly more informative than a fixed-layer pick. Keep the
    # value for plotting here but do not double-register the claim.
    gp_r_held_val = _r(gp_r_held)
    gp_r_hoo_val = claims.register(
        "Gemma probe cross-topic pooled r",
        _r(gp_r_hoo),
        "Leave-one-topic-out pooled Pearson r for the Gemma-3-27B ridge probe "
        "(L32, tb-5 / end-of-turn): predictions from every fold's held-out topic "
        "are concatenated and correlated with true Thurstonian utilities.",
        used_in=["fig:cross-topic", "abstract", "sec:shared"],
        data_paths=[_gp_hoo_pooled],
        derivation="Read `pooled_pearson_r` from pooled_metrics.json; round to 3dp.",
    )
    qp_r_held_val = claims.register(
        "Qwen probe heldout r",
        _r(qp_r_held),
        "A ridge probe on Qwen-3.5-122B residual-stream activations at the "
        "turn-boundary tb-1 token (layer 38) predicts held-out Thurstonian "
        "utilities at Pearson r on a within-distribution eval split.",
        used_in=["fig:cross-topic", "sec:shared"],
        data_paths=[_qp_heldout_path],
        derivation=f"`final_r` of the ridge probe with layer=={QWEN_LAYER} in the manifest's `probes` array; round to 3dp.",
    )
    qp_r_hoo_val = claims.register(
        "Qwen probe cross-topic pooled r",
        _r(qp_r_hoo),
        "Leave-one-topic-out pooled Pearson r for the Qwen-3.5-122B ridge "
        "probe (L38, tb-1).",
        used_in=["fig:cross-topic", "sec:shared"],
        data_paths=[_qp_hoo_pooled],
        derivation="Read `pooled_pearson_r` from pooled_metrics.json; round to 3dp.",
    )
    qp_acc_held_val = claims.register(
        "Qwen probe heldout uniform pairwise accuracy",
        _r(qp_acc_held),
        "Uniform-sample pairwise accuracy of the Qwen-3.5-122B ridge probe "
        "(L38) on the within-distribution held-out eval split.",
        used_in=["fig:cross-topic", "sec:shared"],
        data_paths=[_qp_heldout_path],
        derivation=f"`uniform_pairwise_acc` of the ridge probe with layer=={QWEN_LAYER} in the manifest's `probes` array; round to 3dp.",
    )

    # --- Un-cited probe accuracies: collapse into one row claim ---
    probe_extras = claims.register(
        "Cross model probe extras",
        {
            "gemma heldout acc": _r(gp_acc_held),
            "gemma cross-topic acc": _r(gp_acc_hoo),
            "qwen cross-topic acc": _r(qp_acc_hoo),
        },
        "Un-cited residual-stream probe accuracies rendered in fig:cross-topic: "
        "Gemma (L32, tb-5) held-out and leave-one-topic-out uniform-sample pairwise "
        "accuracy, and Qwen-3.5-122B (L38) leave-one-topic-out uniform-sample "
        "pairwise accuracy. Each cell is rounded to 3dp.",
        used_in=["fig:cross-topic"],
        data_paths=[_gp_heldout_path, _gp_hoo_summary, _qp_hoo_summary],
        derivation=(
            "gemma heldout acc: `uniform_pairwise_acc` of ridge probe at "
            f"layer=={GEMMA_LAYER} in Gemma heldout manifest. "
            "gemma cross-topic acc: "
            f"`layer_summary[{GEMMA_LAYER}].ridge.mean_uniform_hoo_acc` from "
            "Gemma hoo_summary.json. qwen cross-topic acc: "
            f"`layer_summary[{QWEN_LAYER}].ridge.mean_uniform_hoo_acc` from "
            "Qwen hoo_summary.json. All rounded to 3dp."
        ),
    )

    # --- Un-cited content baseline values: collapse into one 2-D table ---
    content_baseline_table = {
        "gemma": {
            "heldout r": _r(gb_r_held),
            "cross-topic pooled r": _r(gb_r_hoo),
            "heldout uniform pairwise accuracy": _r(gb_acc_held),
            "cross-topic uniform pairwise accuracy": _r(gb_acc_hoo),
        },
        "qwen": {
            "heldout r": _r(qb_r_held),
            "cross-topic pooled r": _r(qb_r_hoo),
            "heldout uniform pairwise accuracy": _r(qb_acc_held),
            "cross-topic uniform pairwise accuracy": _r(qb_acc_hoo),
        },
    }
    claims.register(
        "Cross model content baseline",
        content_baseline_table,
        "Qwen3-Embedding-8B sentence-transformer content baseline rendered in "
        "fig:cross-topic. Rows: target-model utility pool the baseline is trained "
        "on (gemma = Gemma-3-27B utilities, qwen = Qwen-3.5-122B utilities). "
        "Columns: `heldout r` = Pearson r on the within-distribution eval split; "
        "`cross-topic pooled r` = leave-one-topic-out pooled Pearson r; "
        "`heldout uniform pairwise accuracy` and "
        "`cross-topic uniform pairwise accuracy` = corresponding uniform-sample "
        "pairwise accuracies. All cells rounded to 3dp.",
        used_in=["fig:cross-topic"],
        data_paths=[
            _gb_heldout_path, _gb_hoo_pooled, _gb_hoo_summary,
            _qb_heldout_path, _qb_hoo_pooled, _qb_hoo_summary,
        ],
        derivation=(
            "For each row, read `final_r` and `uniform_pairwise_acc` of the ridge "
            f"probe at layer=={QWEN3_EMB_LAYER} in the heldout manifest, "
            "`pooled_pearson_r` from pooled_metrics.json, and "
            f"`layer_summary[{QWEN3_EMB_LAYER}].ridge.mean_uniform_hoo_acc` from "
            "hoo_summary.json; round to 3dp."
        ),
    )

    # Panel-drawing dicts. LOO metrics come from `pooled_metrics_clean.json`
    # (clean separate-run labels) when available, so the LOO and within-dist
    # bars are on the same measurement-noise footing.
    gemma_probe = {
        "r_heldout": gp_r_held_val, "r_hoo": _r(gp_r_hoo),
        "acc_heldout": probe_extras["gemma heldout acc"],
        "acc_hoo": _r(gp_acc_hoo) if gp_acc_hoo is not None else None,
        "n_heldout": gp_n_held, "n_hoo": gp_n_hoo,
        "n_heldout_pairs": gp_n_held_pairs, "n_hoo_pairs": gp_n_hoo_pairs,
    }
    qwen_probe = {
        "r_heldout": qp_r_held_val, "r_hoo": _r(qp_r_hoo),
        "acc_heldout": qp_acc_held_val,
        "acc_hoo": _r(qp_acc_hoo) if qp_acc_hoo is not None else None,
        "n_heldout": qp_n_held, "n_hoo": qp_n_hoo,
        "n_heldout_pairs": qp_n_held_pairs, "n_hoo_pairs": qp_n_hoo_pairs,
    }
    gemma_baseline = {
        "r_heldout": content_baseline_table["gemma"]["heldout r"],
        "r_hoo": _r(gb_r_hoo),
        "acc_heldout": content_baseline_table["gemma"]["heldout uniform pairwise accuracy"],
        "acc_hoo": _r(gb_acc_hoo) if gb_acc_hoo is not None else None,
        "n_heldout": gb_n_held, "n_hoo": gb_n_hoo,
        "n_heldout_pairs": gb_n_held_pairs, "n_hoo_pairs": gb_n_hoo_pairs,
    }
    qwen_baseline = {
        "r_heldout": content_baseline_table["qwen"]["heldout r"],
        "r_hoo": _r(qb_r_hoo),
        "acc_heldout": content_baseline_table["qwen"]["heldout uniform pairwise accuracy"],
        "acc_hoo": _r(qb_acc_hoo) if qb_acc_hoo is not None else None,
        "n_heldout": qb_n_held, "n_hoo": qb_n_hoo,
        "n_heldout_pairs": qb_n_held_pairs, "n_hoo_pairs": qb_n_hoo_pairs,
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
