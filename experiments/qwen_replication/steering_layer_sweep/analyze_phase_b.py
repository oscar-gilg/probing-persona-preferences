"""Analyse Qwen Phase B harm-breakdown checkpoints by invoking the extended
paper plot script with Qwen-specific paths.

Discovers the contrastive + single-task checkpoints under
`experiments/qwen_replication/steering_layer_sweep/checkpoints/` and produces:
  - assets/plot_<mmddYY>_qwen_<sel>_L<L>_dose_response_harm_breakdown.png
  - phase_b_summary.json (per-pair-type swing / suppression / amplification)

Usage:
    python -m experiments.qwen_replication.steering_layer_sweep.analyze_phase_b \\
        --selector tb1 --layer 38

If --selector and --layer are omitted, this script discovers them by parsing
the available `phase_b_<sel>_contrastive_L<L>_150.parsed.jsonl` filenames and
fails if 0 or >1 contrastive checkpoints are found.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EXP_DIR = Path("experiments/qwen_replication/steering_layer_sweep")
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
ASSETS_DIR = EXP_DIR / "assets"
SUMMARY_PATH = EXP_DIR / "phase_b_summary.json"
PAIRS_PATH = EXP_DIR / "steering_pairs_150.json"
PLOT_SCRIPT = "scripts/paper/plot_layer_sweep_dose_response.py"

CONTRASTIVE_RE = re.compile(r"^phase_b_(?P<sel>tb1|tb4)_contrastive_L(?P<L>\d+)_150\.parsed\.jsonl$")

HARM_ORIGINS = {"BAILBENCH", "STRESS_TEST"}


def discover() -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    for path in sorted(CHECKPOINTS_DIR.glob("phase_b_*_contrastive_*_150.parsed.jsonl")):
        m = CONTRASTIVE_RE.match(path.name)
        if m:
            candidates.append((m.group("sel"), int(m.group("L"))))
    if not candidates:
        raise RuntimeError(
            f"No phase_b_*_contrastive_*_150.parsed.jsonl found under {CHECKPOINTS_DIR}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple Phase B contrastive checkpoints found: {candidates}. "
            f"Pass --selector and --layer explicitly."
        )
    return candidates[0]


def pair_type_of(origin_a: str, origin_b: str) -> str:
    a_h = origin_a in HARM_ORIGINS
    b_h = origin_b in HARM_ORIGINS
    if a_h and b_h:
        return "hh"
    if a_h or b_h:
        return "hb"
    return "bb"


def physical_in_span(span: str, ordering: int) -> str:
    if span == "first":
        return "a" if ordering == 0 else "b"
    return "b" if ordering == 0 else "a"


def compute_summary(selector: str, layer: int) -> dict:
    """Per-pair-type: swing (= P(c=+0.05) - (1 - P(c=-0.05))),
    contrastive_p_pos, contrastive_p_neg, single-task suppression / amplification ratios."""
    contr_path = CHECKPOINTS_DIR / f"phase_b_{selector}_contrastive_L{layer:02d}_150.parsed.jsonl"
    uni_path = CHECKPOINTS_DIR / f"phase_b_{selector}_single_task_L{layer:02d}_150.parsed.jsonl"

    pairs = json.loads(PAIRS_PATH.read_text())
    pair_type = {p["pair_id"]: pair_type_of(p["task_a_origin"], p["task_b_origin"]) for p in pairs}

    contr_rows = [json.loads(l) for l in contr_path.read_text().splitlines() if l.strip()]
    contr_rows = [r for r in contr_rows if r["layer"] == layer]

    summary: dict = {"selector": selector, "layer": layer, "by_pair_type": {}}
    for pt in ("bb", "hb", "hh"):
        pos = [r for r in contr_rows
               if abs(r["signed_multiplier"] - 0.05) < 1e-9 and pair_type.get(r["pair_id"]) == pt]
        neg = [r for r in contr_rows
               if abs(r["signed_multiplier"] + 0.05) < 1e-9 and pair_type.get(r["pair_id"]) == pt]
        p_pos = sum(r["choice_original"] == "a" for r in pos) / max(len(pos), 1)
        p_neg = sum(r["choice_original"] == "a" for r in neg) / max(len(neg), 1)
        refusal = sum(r["choice_original"] not in ("a", "b") for r in pos + neg) / max(len(pos + neg), 1)
        summary["by_pair_type"][pt] = {
            "contrastive_p_chose_a_at_c_pos05": p_pos,
            "contrastive_p_chose_a_at_c_neg05": p_neg,
            "swing": p_pos - p_neg,
            "refusal_rate_at_c05": refusal,
            "n_pos": len(pos),
            "n_neg": len(neg),
        }

    if uni_path.exists():
        uni_rows_all = [json.loads(l) for l in uni_path.read_text().splitlines() if l.strip()]
        uni_rows = [r for r in uni_rows_all if r["layer"] == layer]
        for pt in ("bb", "hb", "hh"):
            # P(picked steered task) at c=+0.05 = pushing the steered span toward chosen.
            # Aggregate over first/second spans.
            pos_first = [r for r in uni_rows
                         if r["condition"].startswith("unilateral_first")
                         and abs(r["signed_multiplier"] - 0.05) < 1e-9
                         and pair_type.get(r["pair_id"]) == pt]
            pos_second = [r for r in uni_rows
                          if r["condition"].startswith("unilateral_second")
                          and abs(r["signed_multiplier"] - 0.05) < 1e-9
                          and pair_type.get(r["pair_id"]) == pt]
            neg_first = [r for r in uni_rows
                         if r["condition"].startswith("unilateral_first")
                         and abs(r["signed_multiplier"] + 0.05) < 1e-9
                         and pair_type.get(r["pair_id"]) == pt]
            neg_second = [r for r in uni_rows
                          if r["condition"].startswith("unilateral_second")
                          and abs(r["signed_multiplier"] + 0.05) < 1e-9
                          and pair_type.get(r["pair_id"]) == pt]

            def p_chose_steered(rows: list[dict]) -> float:
                if not rows:
                    return float("nan")
                hits = 0
                for r in rows:
                    span = "first" if r["condition"].startswith("unilateral_first") else "second"
                    tgt = physical_in_span(span, r["ordering"])
                    hits += int(r["choice_original"] == tgt)
                return hits / len(rows)

            single_swing_pos = (p_chose_steered(pos_first) + p_chose_steered(pos_second)) / 2
            single_swing_neg = (p_chose_steered(neg_first) + p_chose_steered(neg_second)) / 2
            ratio_pos = single_swing_pos / max(summary["by_pair_type"][pt]["contrastive_p_chose_a_at_c_pos05"], 1e-9)
            summary["by_pair_type"][pt]["single_task_amplification"] = single_swing_pos
            summary["by_pair_type"][pt]["single_task_suppression"] = single_swing_neg
            summary["by_pair_type"][pt]["single_to_contrastive_ratio_pos"] = ratio_pos
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector", choices=["tb1", "tb4"], default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--repo-root", type=Path, default=Path.cwd(),
                   help="Repo root (defaults to CWD); paths in --out are relative to this.")
    args = p.parse_args()

    if args.selector is None or args.layer is None:
        sel, L = discover()
    else:
        sel, L = args.selector, args.layer
    print(f"Phase B analysis: selector={sel}, layer={L}")

    summary = compute_summary(sel, L)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d%y")
    out_path = ASSETS_DIR / f"plot_{stamp}_qwen_{sel}_L{L}_dose_response_harm_breakdown.png"
    contr_ck = CHECKPOINTS_DIR / f"phase_b_{sel}_contrastive_L{L:02d}_150.parsed.jsonl"
    uni_ck = CHECKPOINTS_DIR / f"phase_b_{sel}_single_task_L{L:02d}_150.parsed.jsonl"

    cmd = [
        "python", PLOT_SCRIPT,
        "--contrastive-checkpoint", str(contr_ck),
        "--single-task-checkpoints", str(uni_ck),
        "--pairs", str(PAIRS_PATH),
        "--layer", str(L),
        "--baseline-mode", "explicit-c0",
        "--out", str(out_path),
        "--title", f"Qwen-3.5-122B dose-response at the peak causal layer (L{L} {sel}), by pair type",
        "--panel-suffix", f"Qwen L{L} {sel}",
    ]
    print(f"\nInvoking: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(args.repo_root))
    print(f"\nFigure: {out_path.resolve()}")


if __name__ == "__main__":
    main()
