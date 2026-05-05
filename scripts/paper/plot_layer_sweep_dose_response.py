"""Paper figure for §2.3: dose-response of contrastive vs single-task steering,
with harmful-vs-benign pair-type breakdown.

Two panels side by side:
  (A) Contrastive steering — x = signed multiplier, y = P(chose higher-utility task).
      Three lines: bb, hb, hh pair types.
  (B) Single-task steering — x = applied coefficient on steered task,
      y = P(model picked that task) aggregated across first/second span.
      Three lines: bb, hb, hh.

CLI args parameterise paths so this same script produces both the Gemma figure
(default) and the Qwen analogue. Default invocation reproduces the existing
Gemma L23 figure verbatim.

Baseline modes:
  - dead-layers (Gemma default): average the unilateral / contrastive checkpoints
    at known no-effect layers as a proxy for c=0.
  - explicit-c0: use rows with signed_multiplier == 0 in the checkpoints.
    Cleaner; required when an explicit zero-coefficient cell was run (Qwen).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

HARM_ORIGINS = {"BAILBENCH", "STRESS_TEST"}
LABELS = {"bb": "benign-benign", "hb": "harmful-benign", "hh": "harmful-harmful"}
COLORS = {"bb": "C0", "hb": "C1", "hh": "C3"}


def pair_type_of(origin_a: str, origin_b: str) -> str:
    a_harm = origin_a in HARM_ORIGINS
    b_harm = origin_b in HARM_ORIGINS
    if a_harm and b_harm:
        return "hh"
    if a_harm or b_harm:
        return "hb"
    return "bb"


def load_pair_type_map(pairs_path: Path) -> dict[str, str]:
    pairs = json.loads(pairs_path.read_text())
    return {p["pair_id"]: pair_type_of(p["task_a_origin"], p["task_b_origin"]) for p in pairs}


def load_parsed(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def p_mean(rows: list[dict], target_fn) -> float:
    if not rows:
        return float("nan")
    return sum(int(target_fn(r)) for r in rows) / len(rows)


def physical_in_span(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def parse_int_list(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Default invocation = Gemma L23 figure.
    p.add_argument("--contrastive-checkpoint", type=Path,
                   default=REPO / "experiments/layer_sweep/checkpoints/eot_probe_L23.parsed.jsonl",
                   help="Path to contrastive *.parsed.jsonl checkpoint.")
    p.add_argument("--single-task-checkpoints", type=Path, nargs="+",
                   default=[
                       REPO / "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_early.parsed.jsonl",
                       REPO / "experiments/layer_sweep/checkpoints/eot_unilateral_diagonal_late.parsed.jsonl",
                   ],
                   help="One or more single-task *.parsed.jsonl checkpoints (rows filtered by --layer).")
    p.add_argument("--pairs", type=Path,
                   default=REPO / "experiments/layer_sweep/steering_pairs.json",
                   help="Path to pair JSON (defines pair_id → origins).")
    p.add_argument("--layer", type=int, default=23,
                   help="Injection layer (rows with row['layer'] != this are dropped).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path. Default: paper/figures/plot_<mmddYY>_<basename>.png")
    p.add_argument("--baseline-mode", choices=["dead-layers", "explicit-c0"],
                   default="dead-layers",
                   help="How to compute the c=0 baseline.")
    p.add_argument("--dead-layers", type=str,
                   default="2,5,8,11,14,35,38,41,44,47,50,53,56,59",
                   help="Comma-separated layer indices used as no-effect proxy in dead-layers mode.")
    p.add_argument("--dead-contrastive-glob", type=str,
                   default="experiments/layer_sweep/checkpoints/eot_probe_L*.parsed.jsonl",
                   help="Glob (relative to repo root) of contrastive checkpoints to scan for dead-layer rows.")
    p.add_argument("--title", type=str,
                   default="Dose-response at the peak causal layer, by pair type",
                   help="Figure suptitle.")
    p.add_argument("--panel-suffix", type=str, default="L23 eot",
                   help="Subtitle suffix on each panel (e.g., 'L23 eot' or 'Qwen L<peak> tb-1').")
    return p.parse_args()


def _baseline_explicit_c0_contrastive(contr_rows_at_layer: list[dict],
                                      pair_type: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        rs = [r for r in contr_rows_at_layer
              if r["signed_multiplier"] == 0 and pair_type.get(r["pair_id"]) == pt]
        out[pt] = sum(r["choice_original"] == "a" for r in rs) / max(len(rs), 1)
    return out


def _baseline_explicit_c0_single_task(uni_rows_at_layer: list[dict],
                                      pair_type: dict[str, str]) -> tuple[dict[str, float], dict[str, float]]:
    first: dict[str, float] = {}
    second: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        rs_first = [r for r in uni_rows_at_layer
                    if r["signed_multiplier"] == 0 and r["condition"].startswith("unilateral_first")
                    and pair_type.get(r["pair_id"]) == pt]
        rs_second = [r for r in uni_rows_at_layer
                     if r["signed_multiplier"] == 0 and r["condition"].startswith("unilateral_second")
                     and pair_type.get(r["pair_id"]) == pt]
        first[pt] = sum(r["choice_original"] == physical_in_span("first", r["ordering"]) for r in rs_first) / max(len(rs_first), 1)
        second[pt] = sum(r["choice_original"] == physical_in_span("second", r["ordering"]) for r in rs_second) / max(len(rs_second), 1)
    return first, second


def _baseline_dead_layers(args: argparse.Namespace,
                          single_task_paths: list[Path],
                          pair_type: dict[str, str]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Returns (contrastive_baseline, single_first_baseline, single_second_baseline)."""
    dead_set = set(parse_int_list(args.dead_layers))

    # Single-task dead-layer baseline.
    dead_uni = []
    for path in single_task_paths:
        if not path.exists():
            continue
        for r in load_parsed(path):
            if r["layer"] in dead_set:
                dead_uni.append(r)
    first: dict[str, float] = {}
    second: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        rs_first = [r for r in dead_uni if r["condition"].startswith("unilateral_first") and pair_type.get(r["pair_id"]) == pt]
        rs_second = [r for r in dead_uni if r["condition"].startswith("unilateral_second") and pair_type.get(r["pair_id"]) == pt]
        first[pt] = sum(r["choice_original"] == physical_in_span("first", r["ordering"]) for r in rs_first) / max(len(rs_first), 1)
        second[pt] = sum(r["choice_original"] == physical_in_span("second", r["ordering"]) for r in rs_second) / max(len(rs_second), 1)

    # Contrastive dead-layer baseline (scan glob for per-layer files).
    contr_dead_rows: list[dict] = []
    for path in REPO.glob(args.dead_contrastive_glob):
        for r in load_parsed(path):
            if r["layer"] in dead_set:
                contr_dead_rows.append(r)
    contrastive: dict[str, float] = {}
    for pt in ("bb", "hb", "hh"):
        rs = [r for r in contr_dead_rows if pair_type.get(r["pair_id"]) == pt]
        contrastive[pt] = sum(r["choice_original"] == "a" for r in rs) / max(len(rs), 1)

    return contrastive, first, second


def main() -> None:
    args = parse_args()

    pair_type = load_pair_type_map(args.pairs)

    # --- Contrastive ---
    contr_rows_all = load_parsed(args.contrastive_checkpoint)
    contr_rows = [r for r in contr_rows_all if r["layer"] == args.layer]
    contr_by: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in contr_rows:
        if r["signed_multiplier"] == 0:
            continue  # baseline handled separately
        pt = pair_type.get(r["pair_id"], "bb")
        contr_by[pt][r["signed_multiplier"]].append(r)

    # --- Single-task ---
    uni_rows_all: list[dict] = []
    for path in args.single_task_checkpoints:
        if path.exists():
            uni_rows_all.extend(load_parsed(path))
    uni_rows = [r for r in uni_rows_all if r["layer"] == args.layer]
    uni_by: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in uni_rows:
        if r["signed_multiplier"] == 0:
            continue
        pt = pair_type.get(r["pair_id"], "bb")
        applied = r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1)
        uni_by[pt][round(applied, 4)].append(r)

    # --- Baselines ---
    if args.baseline_mode == "explicit-c0":
        contr_baseline = _baseline_explicit_c0_contrastive(contr_rows_all, pair_type)
        first_baseline, second_baseline = _baseline_explicit_c0_single_task(uni_rows_all, pair_type)
    else:  # dead-layers
        contr_baseline, first_baseline, second_baseline = _baseline_dead_layers(
            args, args.single_task_checkpoints, pair_type
        )

    # --- Plot ---
    fig, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: contrastive
    coefs_c = sorted({r["signed_multiplier"] for r in contr_rows if r["signed_multiplier"] != 0})
    for pt in ("bb", "hb", "hh"):
        xs, ys = [], []
        for c in coefs_c:
            rs = contr_by[pt][c]
            if not rs:
                continue
            xs.append(c)
            ys.append(p_mean(rs, lambda r: r["choice_original"] == "a"))
        xs.append(0.0)
        ys.append(contr_baseline[pt])
        ordered = sorted(zip(xs, ys))
        ax_c.plot([p[0] for p in ordered], [p[1] for p in ordered],
                  "o-", color=COLORS[pt], label=LABELS[pt],
                  markersize=5, linewidth=1.8)
    ax_c.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax_c.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
    ax_c.set_xlabel("signed multiplier $c$ (× mean activation norm)", fontsize=10)
    ax_c.set_ylabel(r"$P(\mathrm{chose\ higher\!-\!utility\ task})$", fontsize=10)
    ax_c.set_title(f"(A) Contrastive steering — {args.panel_suffix}", fontsize=11)
    ax_c.set_ylim(0, 1)
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8)

    # Panel B: single-task
    coefs_u = sorted({round(r["signed_multiplier"] * (1 if r["ordering"] == 0 else -1), 4)
                      for r in uni_rows if r["signed_multiplier"] != 0})
    for pt in ("bb", "hb", "hh"):
        xs, ys = [], []
        for c in coefs_u:
            rs = uni_by[pt][c]
            if not rs:
                continue
            hits = 0
            for r in rs:
                span = "first" if r["condition"].startswith("unilateral_first") else "second"
                tgt = physical_in_span(span, r["ordering"])
                hits += int(r["choice_original"] == tgt)
            ys.append(hits / len(rs))
            xs.append(c)
        xs.append(0.0)
        ys.append((first_baseline[pt] + second_baseline[pt]) / 2)
        ordered = sorted(zip(xs, ys))
        ax_u.plot([p[0] for p in ordered], [p[1] for p in ordered],
                  "o-", color=COLORS[pt], label=LABELS[pt],
                  markersize=5, linewidth=1.8)
    ax_u.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax_u.axvline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)
    ax_u.set_xlabel("coefficient applied to steered task (× mean activation norm)", fontsize=10)
    ax_u.set_ylabel(r"$P(\mathrm{picked\ the\ steered\ task})$", fontsize=10)
    ax_u.set_title(f"(B) Single-task steering — {args.panel_suffix}", fontsize=11)
    ax_u.set_ylim(0, 1)
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=8)

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()

    if args.out is None:
        stamp = datetime.now().strftime("%m%d%y")
        out = REPO / "paper" / "figures" / f"plot_{stamp}_layer{args.layer}_dose_response_harm_breakdown.png"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    pt_counts: dict[str, int] = defaultdict(int)
    for v in pair_type.values():
        pt_counts[v] += 1
    print(f"Pair-type counts (n pairs): bb={pt_counts['bb']} hb={pt_counts['hb']} hh={pt_counts['hh']}")


if __name__ == "__main__":
    main()
