"""Analyse layer-sweep probes for tb:-2 and eot selectors.

Emits:
- probe_metrics.json (layer -> {r, R2, norm} per selector)
- plot_<mmddYY>_probe_r_by_layer.png
- plot_<mmddYY>_probe_cosine_role_marker.png / ..._end_of_turn.png
- plot_<mmddYY>_probe_cosine_cross_selector.png
- plot_<mmddYY>_probe_transfer_heatmap.png (one heatmap per selector, stacked vertically)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src.measurement.storage.loading import load_run_utilities
from src.probes.core.activations import get_mean_norms, load_activations
from src.probes.core.evaluate import compute_probe_similarity, evaluate_probe_on_data
from src.probes.core.storage import load_manifest, load_probe


load_dotenv()

SELECTORS: dict[str, dict] = {
    "tb-2": {
        "manifest_dir": Path("results/probes/layer_sweep/tb-2"),
        "activations_path": Path("activations/gemma-3-27b_it/pref_layer_sweep/activations_turn_boundary:-2.npz"),
        "display": "role-marker",
    },
    "eot": {
        "manifest_dir": Path("results/probes/layer_sweep/eot"),
        "activations_path": Path("activations/gemma-3-27b_it/pref_layer_sweep/activations_eot.npz"),
        "display": "end-of-turn",
    },
}

DEFAULT_TEST_DIR = Path(
    "results/experiments/persona_sweep_final_six/pre_task_active_learning/default_test"
)
OUTPUT_DIR = Path("experiments/layer_sweep")
ASSETS_DIR = OUTPUT_DIR / "assets"
METRICS_PATH = OUTPUT_DIR / "probe_metrics.json"


def _probe_id(layer: int) -> str:
    return f"ridge_L{layer:02d}"


def _layers_from_manifest(manifest_dir: Path) -> list[int]:
    manifest = load_manifest(manifest_dir)
    return sorted({p["layer"] for p in manifest["probes"] if p["method"] == "ridge"})


def _per_layer_metrics(
    manifest_dir: Path,
    activations_path: Path,
    layers: list[int],
    test_scores: dict[str, float],
) -> dict[int, dict]:
    test_ids = sorted(test_scores.keys())
    test_score_arr = np.array([test_scores[t] for t in test_ids])
    act_task_ids, act_dict = load_activations(
        activations_path, task_id_filter=set(test_ids), layers=layers,
    )
    norms = get_mean_norms(activations_path, layers=layers)

    out: dict[int, dict] = {}
    for layer in layers:
        weights = load_probe(manifest_dir, _probe_id(layer))
        res = evaluate_probe_on_data(
            probe_weights=weights,
            activations=act_dict[layer],
            scores=test_score_arr,
            task_ids_data=act_task_ids,
            task_ids_scores=test_ids,
        )
        out[layer] = {
            "pearson_r": res["pearson_r"],
            "r2": res["r2"],
            "n_samples": res["n_samples"],
            "norm": norms[layer],
        }
    return out


def _transfer_matrix(
    manifest_dir: Path,
    activations_path: Path,
    layers: list[int],
    test_scores: dict[str, float],
) -> np.ndarray:
    """Matrix[i, j] = Pearson r of probe trained at layers[i] evaluated on activations at layers[j]."""
    test_ids = sorted(test_scores.keys())
    test_score_arr = np.array([test_scores[t] for t in test_ids])
    act_task_ids, act_dict = load_activations(
        activations_path, task_id_filter=set(test_ids), layers=layers,
    )

    weights_by_layer = {L: load_probe(manifest_dir, _probe_id(L)) for L in layers}

    n = len(layers)
    M = np.full((n, n), np.nan)
    for i, probe_layer in enumerate(layers):
        w = weights_by_layer[probe_layer]
        for j, act_layer in enumerate(layers):
            # Probe and activations must share dimensionality (same model), so this is always valid.
            res = evaluate_probe_on_data(
                probe_weights=w,
                activations=act_dict[act_layer],
                scores=test_score_arr,
                task_ids_data=act_task_ids,
                task_ids_scores=test_ids,
            )
            r = res["pearson_r"]
            M[i, j] = np.nan if r is None else r
    return M


def _cross_selector_cosine(
    manifest_a: Path, manifest_b: Path, layers: list[int],
) -> np.ndarray:
    """Rows = manifest_a probes, cols = manifest_b probes."""
    # Drop the Ridge intercept (last element) before computing direction cosine.
    weights_a = np.stack([load_probe(manifest_a, _probe_id(L))[:-1] for L in layers])
    weights_b = np.stack([load_probe(manifest_b, _probe_id(L))[:-1] for L in layers])
    wa = weights_a / np.linalg.norm(weights_a, axis=1, keepdims=True)
    wb = weights_b / np.linalg.norm(weights_b, axis=1, keepdims=True)
    return wa @ wb.T


def _plot_r_by_layer(metrics: dict[str, dict[int, dict]], layers: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sel, per_layer in metrics.items():
        ys = [per_layer[L]["pearson_r"] for L in layers]
        ax.plot(layers, ys, marker="o", label=SELECTORS[sel]["display"])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r (default_test)")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out Pearson r by layer and token position")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_heatmap(matrix: np.ndarray, layers: list[int], title: str, path: Path, vmin: float, vmax: float, cmap: str, cbar_label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cross_selector(matrix: np.ndarray, layers: list[int], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel("end-of-turn probe layer")
    ax.set_ylabel("role-marker probe layer")
    ax.set_title("Role-marker vs end-of-turn probe cosine similarity")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cos(w_role-marker, w_end-of-turn)  —  1 = same direction, 0 = orthogonal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_transfer_stacked(
    matrices: dict[str, np.ndarray], layers: list[int], path: Path,
) -> None:
    sels = list(matrices.keys())
    fig, axes = plt.subplots(1, len(sels), figsize=(7 * len(sels), 6))
    if len(sels) == 1:
        axes = [axes]
    for ax, sel in zip(axes, sels):
        im = ax.imshow(matrices[sel], vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, fontsize=7)
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_xlabel("Activation layer (L_s)")
        ax.set_ylabel("Probe-train layer (L_p)")
        ax.set_title(f"Cross-layer probe transfer — {SELECTORS[sel]['display']} probe")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Pearson r (probe predictions vs utilities on default_test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d%y")

    print(f"Loading default_test utilities from {DEFAULT_TEST_DIR}")
    mu_array, task_ids = load_run_utilities(DEFAULT_TEST_DIR)
    test_scores = dict(zip(task_ids, mu_array.tolist()))

    layers_per_selector: dict[str, list[int]] = {}
    metrics: dict[str, dict[int, dict]] = {}
    transfer: dict[str, np.ndarray] = {}
    within_cosine: dict[str, np.ndarray] = {}

    for sel, info in SELECTORS.items():
        print(f"\n=== Selector {sel} ===")
        layers = _layers_from_manifest(info["manifest_dir"])
        layers_per_selector[sel] = layers
        print(f"  Layers: {layers}")

        metrics[sel] = _per_layer_metrics(
            info["manifest_dir"], info["activations_path"], layers, test_scores,
        )
        for L in layers:
            m = metrics[sel][L]
            print(f"  L{L:02d}: r={m['pearson_r']:.4f} R2={m['r2']:.4f} norm={m['norm']:.1f}")

        within_cosine[sel] = compute_probe_similarity(
            info["manifest_dir"], probe_ids=[_probe_id(L) for L in layers],
        )
        transfer[sel] = _transfer_matrix(
            info["manifest_dir"], info["activations_path"], layers, test_scores,
        )

    layers_ref = layers_per_selector["tb-2"]
    if layers_per_selector["eot"] != layers_ref:
        raise RuntimeError(
            f"Layer sets differ: tb-2={layers_ref}, eot={layers_per_selector['eot']}"
        )

    serialisable = {
        sel: {
            str(L): {
                "pearson_r": m["pearson_r"],
                "r2": m["r2"],
                "n_samples": m["n_samples"],
                "norm": m["norm"],
            }
            for L, m in per_layer.items()
        }
        for sel, per_layer in metrics.items()
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)

    r_path = ASSETS_DIR / f"plot_{stamp}_probe_r_by_layer.png"
    _plot_r_by_layer(metrics, layers_ref, r_path)

    cos_paths = {}
    for sel in SELECTORS:
        suffix = "role_marker" if sel == "tb-2" else "end_of_turn"
        p = ASSETS_DIR / f"plot_{stamp}_probe_cosine_{suffix}.png"
        cos_paths[sel] = p
        _plot_heatmap(
            within_cosine[sel], layers_ref,
            f"Probe direction similarity — {SELECTORS[sel]['display']} probe",
            p, vmin=0, vmax=1, cmap="viridis",
            cbar_label="cos(w_i, w_j)  —  1 = same direction, 0 = orthogonal",
        )

    cross_mat = _cross_selector_cosine(
        SELECTORS["tb-2"]["manifest_dir"], SELECTORS["eot"]["manifest_dir"], layers_ref,
    )
    cross_path = ASSETS_DIR / f"plot_{stamp}_probe_cosine_cross_selector.png"
    _plot_cross_selector(cross_mat, layers_ref, cross_path)

    transfer_path = ASSETS_DIR / f"plot_{stamp}_probe_transfer_heatmap.png"
    _plot_transfer_stacked(transfer, layers_ref, transfer_path)

    print("\nArtifacts:")
    for p in [METRICS_PATH, r_path, *cos_paths.values(), cross_path, transfer_path]:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
