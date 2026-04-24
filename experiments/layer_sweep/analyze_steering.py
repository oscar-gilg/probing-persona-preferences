"""Analyse layer-sweep steering checkpoints.

Reads experiments/layer_sweep/checkpoints/<selector>_probe_L<p>.parsed.jsonl.
Produces diagonal / spine-heatmap / refusal / probe-vs-steer agreement plots.

Conventions:
- Pairs are oriented so task_a has higher utility (build_steering_pairs.py).
- For positive signed_multiplier the intended task is task_a; for negative, task_b.
- P(chose steered task) = P(choice_original == intended_task).
- choice_original is already remapped by the runner (accounts for `ordering`).
- Refusal = row where choice_original is neither "a" nor "b".
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


load_dotenv()

CHECKPOINTS_DIR = Path("experiments/layer_sweep/checkpoints")
METRICS_PATH = Path("experiments/layer_sweep/probe_metrics.json")
ASSETS_DIR = Path("experiments/layer_sweep/assets")

SELECTORS = ["tb-2", "eot"]
SELECTOR_DISPLAY = {"tb-2": "tb:-2", "eot": "eot"}

# Parse filenames like "tb-2_probe_L32.parsed.jsonl"
FILENAME_RE = re.compile(r"^(?P<selector>tb-2|eot)_probe_L(?P<layer>\d+)\.parsed\.jsonl$")


def _load_parsed(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover() -> dict[str, dict[int, list[dict]]]:
    """Returns {selector: {probe_layer: rows}}."""
    out: dict[str, dict[int, list[dict]]] = {s: {} for s in SELECTORS}
    for path in sorted(CHECKPOINTS_DIR.glob("*.parsed.jsonl")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        sel = m.group("selector")
        layer = int(m.group("layer"))
        rows = _load_parsed(path)
        out[sel][layer] = rows
        print(f"  {path.name}: {len(rows)} rows")
    return out


def _intended_task(signed_multiplier: float) -> str:
    if signed_multiplier > 0:
        return "a"
    if signed_multiplier < 0:
        return "b"
    raise ValueError("0-multiplier rows have no intended task")


def _p_chose_steered(rows: list[dict]) -> tuple[float, float, int]:
    """Mean, std over pairs, n_pairs. Returns (nan, nan, 0) if no rows."""
    if not rows:
        return float("nan"), float("nan"), 0
    # Aggregate per pair so error bars reflect pair-level variance.
    per_pair_hits: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        mult = r["signed_multiplier"]
        if mult == 0:
            continue
        intended = _intended_task(mult)
        per_pair_hits[r["pair_id"]].append(1 if r["choice_original"] == intended else 0)
    if not per_pair_hits:
        return float("nan"), float("nan"), 0
    pair_means = np.array([np.mean(v) for v in per_pair_hits.values()])
    return float(pair_means.mean()), float(pair_means.std(ddof=1) if len(pair_means) > 1 else 0.0), len(pair_means)


def _refusal_rate(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    refusals = sum(1 for r in rows if r["choice_original"] not in ("a", "b"))
    return refusals / len(rows)


def _rows_at_layer(rows: list[dict], layer: int) -> list[dict]:
    return [r for r in rows if r["layer"] == layer]


def _plot_diagonal(data: dict[str, dict[int, list[dict]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for sel in SELECTORS:
        layers = sorted(data[sel].keys())
        xs, ys, errs = [], [], []
        for L in layers:
            rows = _rows_at_layer(data[sel][L], L)
            mean, std, n = _p_chose_steered(rows)
            if n == 0:
                continue
            xs.append(L)
            ys.append(mean)
            errs.append(std / np.sqrt(n) if n > 0 else 0.0)
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=SELECTOR_DISPLAY[sel])
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Injection layer (= probe layer on diagonal)")
    ax.set_ylabel("P(chose steered task)")
    ax.set_ylim(0, 1)
    ax.set_title("Self-layer diagonal steering effect")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_spine_heatmap(data: dict[str, dict[int, list[dict]]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(SELECTORS), figsize=(7 * len(SELECTORS), 5))
    if len(SELECTORS) == 1:
        axes = [axes]
    for ax, sel in zip(axes, SELECTORS):
        probe_layers = sorted(data[sel].keys())
        # Spine = probe layers with >1 injection layer in their parsed rows.
        spine_probes: list[int] = []
        inject_layer_union: set[int] = set()
        for p in probe_layers:
            inj_set = {r["layer"] for r in data[sel][p]}
            if len(inj_set) > 1:
                spine_probes.append(p)
                inject_layer_union |= inj_set
        inject_layers = sorted(inject_layer_union)

        if not spine_probes or not inject_layers:
            ax.set_title(f"{SELECTOR_DISPLAY[sel]}: no spine data")
            ax.axis("off")
            continue

        M = np.full((len(spine_probes), len(inject_layers)), np.nan)
        for i, p in enumerate(spine_probes):
            for j, L in enumerate(inject_layers):
                rows = _rows_at_layer(data[sel][p], L)
                mean, _, n = _p_chose_steered(rows)
                if n > 0:
                    M[i, j] = mean

        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(inject_layers)))
        ax.set_yticks(range(len(spine_probes)))
        ax.set_xticklabels(inject_layers, rotation=90, fontsize=8)
        ax.set_yticklabels(spine_probes, fontsize=8)
        ax.set_xlabel("Injection layer (L_s)")
        ax.set_ylabel("Probe layer (L_p)")
        ax.set_title(f"Spine steering — {SELECTOR_DISPLAY[sel]} selector")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("P(chose steered task)  —  0.5 = chance, 1 = full control")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_refusal(data: dict[str, dict[int, list[dict]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for sel in SELECTORS:
        inject_to_rows: dict[int, list[dict]] = defaultdict(list)
        for _probe_layer, rows in data[sel].items():
            for r in rows:
                inject_to_rows[r["layer"]].append(r)
        xs = sorted(inject_to_rows.keys())
        ys = [_refusal_rate(inject_to_rows[L]) for L in xs]
        ax.plot(xs, ys, marker="o", label=SELECTOR_DISPLAY[sel])
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("Refusal rate")
    ax.set_ylim(0, 1)
    ax.set_title("Refusal rate by injection layer (averaged across probes)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_probe_vs_steer(data: dict[str, dict[int, list[dict]]], path: Path) -> None:
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 6))
    for sel in SELECTORS:
        xs, ys, labels = [], [], []
        for L_str, m in metrics[sel].items():
            L = int(L_str)
            if L not in data[sel]:
                continue
            rows = _rows_at_layer(data[sel][L], L)
            mean, _, n = _p_chose_steered(rows)
            if n == 0 or m["r2"] is None:
                continue
            xs.append(m["r2"])
            ys.append(mean)
            labels.append(L)
        ax.scatter(xs, ys, label=SELECTOR_DISPLAY[sel], alpha=0.8)
        for x, y, L in zip(xs, ys, labels):
            ax.annotate(str(L), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Probe R² (default_test)")
    ax.set_ylabel("Self-layer P(chose steered task)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Probe fit vs steering effect (self-layer)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d%y")

    print(f"Discovering parsed checkpoints under {CHECKPOINTS_DIR}")
    data = _discover()

    diag_path = ASSETS_DIR / f"plot_{stamp}_steering_diagonal.png"
    spine_path = ASSETS_DIR / f"plot_{stamp}_steering_spine_heatmap.png"
    refusal_path = ASSETS_DIR / f"plot_{stamp}_steering_refusal_by_inject_layer.png"
    scatter_path = ASSETS_DIR / f"plot_{stamp}_probe_vs_steer_agreement.png"

    _plot_diagonal(data, diag_path)
    _plot_spine_heatmap(data, spine_path)
    _plot_refusal(data, refusal_path)
    _plot_probe_vs_steer(data, scatter_path)

    print("\nArtifacts:")
    for p in [diag_path, spine_path, refusal_path, scatter_path]:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()
