"""Analyse Qwen Phase A layer-sweep checkpoints.

Reads experiments/qwen_replication/steering_layer_sweep/checkpoints/phase_a_<sel>_L<L>.parsed.jsonl
for sel in {tb1, tb4} and L in {12, 24, 28, 33, 38, 43}.

Outputs:
  - assets/plot_<mmddYY>_phase_a_diagonal.png  -- P(chose steered) vs layer, panels per selector, lines per |c|
  - assets/plot_<mmddYY>_phase_a_swing.png     -- swing dP = P(c=+0.05) - P(c=-0.05) vs layer, lines per selector
  - assets/plot_<mmddYY>_phase_a_refusal.png   -- refusal rate at |c|=0.05 vs layer
  - phase_a_summary.json

Conventions (paper-aligned with Gemma analyze_steering.py):
  - Pairs oriented so task_a > task_b in Qwen's default_test utility (build_steering_pairs_50.py).
  - For positive signed_multiplier the intended task is task_a; for negative, task_b.
  - P(chose steered task) = mean over pairs of P(choice_original == intended) at that signed_multiplier.
  - Refusal = choice_original not in {"a", "b"}. Computed at |c|=0.05 only (worst case).
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

EXP_DIR = Path("experiments/qwen_replication/steering_layer_sweep")
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
ASSETS_DIR = EXP_DIR / "assets"
SUMMARY_PATH = EXP_DIR / "phase_a_summary.json"

SELECTORS = ["tb1", "tb4"]
SELECTOR_DISPLAY = {"tb1": "tb-1 (final-prompt)", "tb4": "tb-4 (\\n after im_end)"}
LAYERS = [12, 24, 28, 33, 38, 43]
MULTIPLIERS = [-0.05, -0.03, 0.03, 0.05]

# Discover either per-cell (`phase_a_<sel>_L<L>.parsed.jsonl`) OR bundled
# (`phase_a_<sel>_bundled.parsed.jsonl`) output. Bundled rows carry their layer
# in `r["layer"]` so we group by that.
PER_CELL_RE = re.compile(r"^phase_a_(?P<selector>tb1|tb4)_L(?P<layer>\d+)\.parsed\.jsonl$")
BUNDLED_RE = re.compile(r"^phase_a_(?P<selector>tb1|tb4)_bundled\.parsed\.jsonl$")


def _load_parsed(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover() -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {s: {} for s in SELECTORS}
    for path in sorted(CHECKPOINTS_DIR.glob("*.parsed.jsonl")):
        m_bundled = BUNDLED_RE.match(path.name)
        m_cell = PER_CELL_RE.match(path.name)
        if m_bundled:
            sel = m_bundled.group("selector")
            rows = _load_parsed(path)
            print(f"  {path.name}: {len(rows)} rows (bundled)")
            for r in rows:
                L = int(r["layer"])
                out[sel].setdefault(L, []).append(r)
        elif m_cell:
            sel = m_cell.group("selector")
            layer = int(m_cell.group("layer"))
            rows = _load_parsed(path)
            print(f"  {path.name}: {len(rows)} rows")
            out[sel][layer] = rows
    return out


def _intended_task(signed_multiplier: float) -> str:
    if signed_multiplier > 0:
        return "a"
    if signed_multiplier < 0:
        return "b"
    raise ValueError("0-multiplier rows have no intended task")


def _p_chose_steered(rows: list[dict]) -> tuple[float, float, int]:
    """Mean across pairs, std across pairs, n_pairs. Aggregates per pair so SE reflects pair-level variance."""
    if not rows:
        return float("nan"), float("nan"), 0
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


def _summary(data: dict[str, dict[int, list[dict]]]) -> dict:
    out: dict = {}
    for sel, by_layer in data.items():
        sel_summary: dict = {}
        for layer, rows in by_layer.items():
            pos_rows = [r for r in rows if abs(r["signed_multiplier"] - 0.05) < 1e-9]
            neg_rows = [r for r in rows if abs(r["signed_multiplier"] + 0.05) < 1e-9]
            extreme_rows = pos_rows + neg_rows
            mean_p_pos, _, n_pos = _p_chose_steered(pos_rows)
            mean_p_neg, _, n_neg = _p_chose_steered(neg_rows)
            swing = mean_p_pos - (1 - mean_p_neg) if (n_pos and n_neg) else float("nan")
            # Refusal at |c|=0.05 (worst case per spec).
            refusal = _refusal_rate(extreme_rows)
            sel_summary[str(layer)] = {
                "swing": swing,
                "refusal_rate_at_c05": refusal,
                "mean_p_pos": mean_p_pos,
                "mean_p_neg": mean_p_neg,
                "n_pairs_pos": n_pos,
                "n_pairs_neg": n_neg,
            }
        out[sel] = sel_summary
    return out


def _plot_diagonal(data: dict[str, dict[int, list[dict]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, sel in zip(axes, SELECTORS):
        if not data[sel]:
            ax.set_title(f"{SELECTOR_DISPLAY[sel]}: no data yet")
            ax.axis("off")
            continue
        layers = sorted(data[sel].keys())
        for mult in MULTIPLIERS:
            xs, ys, errs = [], [], []
            for L in layers:
                rows = [r for r in data[sel][L] if abs(r["signed_multiplier"] - mult) < 1e-9]
                mean, std, n = _p_chose_steered(rows)
                if n == 0:
                    continue
                xs.append(L)
                ys.append(mean)
                errs.append(std / np.sqrt(n) if n > 1 else 0.0)
            ax.errorbar(xs, ys, yerr=errs, marker="o", label=f"c={mult:+.2f}", capsize=3)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.set_xlabel("Layer (probe = injection)")
        ax.set_title(SELECTOR_DISPLAY[sel])
        ax.set_xticks(LAYERS)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("P(chose steered task | responded)")
    axes[1].legend(loc="center right", fontsize=8)
    fig.suptitle("Qwen-3.5-122B Phase A diagonal — contrastive steering by layer")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_swing(summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sel in SELECTORS:
        if not summary.get(sel):
            continue
        layers = sorted(int(k) for k in summary[sel])
        swings = [summary[sel][str(L)]["swing"] for L in layers]
        ax.plot(layers, swings, marker="o", markersize=8, label=SELECTOR_DISPLAY[sel])
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Swing  $P(\mathrm{steered}|c{=}{+}0.05) - (1 - P(\mathrm{steered}|c{=}{-}0.05))$")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1)
    ax.set_title("Qwen-3.5-122B Phase A — causal swing by layer (which layer steers most?)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_refusal(summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sel in SELECTORS:
        if not summary.get(sel):
            continue
        layers = sorted(int(k) for k in summary[sel])
        rates = [summary[sel][str(L)]["refusal_rate_at_c05"] for L in layers]
        ax.plot(layers, rates, marker="o", markersize=8, label=SELECTOR_DISPLAY[sel])
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Refusal rate at |c|=0.05")
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, max(0.2, ax.get_ylim()[1]))
    ax.set_title("Qwen-3.5-122B Phase A — refusal rate by layer")
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

    summary = _summary(data)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {SUMMARY_PATH.resolve()}")
    print(json.dumps(summary, indent=2))

    diag_path = ASSETS_DIR / f"plot_{stamp}_phase_a_diagonal.png"
    swing_path = ASSETS_DIR / f"plot_{stamp}_phase_a_swing.png"
    refusal_path = ASSETS_DIR / f"plot_{stamp}_phase_a_refusal.png"

    _plot_diagonal(data, diag_path)
    _plot_swing(summary, swing_path)
    _plot_refusal(summary, refusal_path)

    print("\nArtifacts:")
    for p in [diag_path, swing_path, refusal_path]:
        print(f"  {p.resolve()}")

    # Apply Phase A peak rule and report top cells.
    candidates = []
    for sel in SELECTORS:
        for L_str, vals in summary.get(sel, {}).items():
            if vals["refusal_rate_at_c05"] is not None and vals["refusal_rate_at_c05"] < 0.05:
                candidates.append((sel, int(L_str), vals["swing"]))
    candidates.sort(key=lambda x: x[2], reverse=True)

    print("\nPhase A peak rule — top 5 cells by swing (refusal < 5%):")
    for sel, L, swing in candidates[:5]:
        print(f"  {sel} L{L}  swing = {swing:+.3f}")
    if candidates:
        top_swing = candidates[0][2]
        ties = [c for c in candidates if abs(c[2] - top_swing) < 0.05]
        if len(ties) > 1:
            print("\nTight ties (within 0.05) — promote top 2 to Phase B:")
            for sel, L, swing in ties[:2]:
                print(f"  → {sel} L{L}  swing = {swing:+.3f}")
        else:
            print(f"\nSingle peak — promote to Phase B: {candidates[0][0]} L{candidates[0][1]}")


if __name__ == "__main__":
    main()
