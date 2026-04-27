"""Analyze per-cell measurements: compute metrics + probe-vs-random comparisons.

Inputs: experiments/preference_direction_ablation/results/<cell>/measurements.jsonl
Output: per-cell summary CSV + a comparison table.

Metrics per spec (Sec. "Metrics"):
  1. Pair agreement vs B0 (modal choice)
  2. Within-cell test-retest agreement (3 canonical seeds)
  3. Choice-probability distribution KS test vs B0 (p ∈ {0, 1/3, 2/3, 1})
  4. Position-bias flip rate (canonical vs swapped modal choice)
  5. Mean output tokens
  6. Refusal rate
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments/preference_direction_ablation/results"
ASSETS_DIR = REPO_ROOT / "experiments/preference_direction_ablation/assets"


def load_cell(cell_dir: Path) -> pd.DataFrame:
    rows = []
    p = cell_dir / "measurements.jsonl"
    if not p.exists():
        return pd.DataFrame()
    with p.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def per_pair_choice_dist(df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """For each (task_a, task_b), summarize canonical seeds and the swap row."""
    out = {}
    for (a, b), grp in df.groupby(["task_a", "task_b"]):
        canon = grp[grp["order"] == "canonical"]
        swap = grp[grp["order"] == "swapped"]
        canon_choices = list(canon["choice_canonical"].values)
        canon_responded = [c for c in canon_choices if c in ("a", "b")]
        n_a = sum(1 for c in canon_responded if c == "a")
        p_a = n_a / len(canon_responded) if canon_responded else float("nan")
        # modal canonical choice (over all canonical seeds, ties broken by first)
        if canon_choices:
            modal_canon = Counter(canon_choices).most_common(1)[0][0]
        else:
            modal_canon = "missing"
        if len(swap):
            modal_swap = swap.iloc[0]["choice_canonical"]
        else:
            modal_swap = "missing"
        out[(a, b)] = {
            "canon_choices": canon_choices,
            "canon_responded": canon_responded,
            "p_a": p_a,
            "modal_canon": modal_canon,
            "modal_swap": modal_swap,
            "n_canon": len(canon_choices),
        }
    return out


def cell_metrics(df: pd.DataFrame, b0: dict | None) -> dict:
    """Compute per-cell metrics. b0 is the per-pair summary for B0 (or None for B0 itself)."""
    summary = per_pair_choice_dist(df)

    # Test-retest within-cell: average pairwise agreement of 3 canonical seeds
    test_retest = []
    for s in summary.values():
        choices = s["canon_choices"]
        if len(choices) < 2:
            continue
        n_pairs_seeds = 0
        n_agree = 0
        for i in range(len(choices)):
            for j in range(i + 1, len(choices)):
                n_pairs_seeds += 1
                if choices[i] == choices[j]:
                    n_agree += 1
        if n_pairs_seeds:
            test_retest.append(n_agree / n_pairs_seeds)

    # Position-bias flip: modal_canon vs modal_swap (only when both responded)
    flips = 0
    n_flip_eligible = 0
    for s in summary.values():
        if s["modal_canon"] in ("a", "b") and s["modal_swap"] in ("a", "b"):
            n_flip_eligible += 1
            if s["modal_canon"] != s["modal_swap"]:
                flips += 1
    flip_rate = flips / n_flip_eligible if n_flip_eligible else float("nan")

    # Refusal rate: fraction of canonical seed responses that are not 'a' or 'b'
    n_canon_total = sum(len(s["canon_choices"]) for s in summary.values())
    n_canon_responded = sum(len(s["canon_responded"]) for s in summary.values())
    refusal_rate = 1.0 - (n_canon_responded / n_canon_total) if n_canon_total else float("nan")

    # Mean response length (chars in raw_response)
    if "raw_response" in df.columns:
        mean_len = df["raw_response"].astype(str).str.len().mean()
    else:
        mean_len = float("nan")

    metrics = {
        "n_pairs": len(summary),
        "test_retest": float(np.mean(test_retest)) if test_retest else float("nan"),
        "flip_rate": flip_rate,
        "refusal_rate": refusal_rate,
        "mean_response_chars": float(mean_len) if not np.isnan(mean_len) else float("nan"),
    }

    if b0 is not None:
        # Pair agreement vs B0: fraction of pairs (with both modal choices defined) that match
        agree, n_eligible = 0, 0
        b0_p = []
        cell_p = []
        for key, cell_s in summary.items():
            b0_s = b0.get(key)
            if b0_s is None:
                continue
            if cell_s["modal_canon"] in ("a", "b") and b0_s["modal_canon"] in ("a", "b"):
                n_eligible += 1
                if cell_s["modal_canon"] == b0_s["modal_canon"]:
                    agree += 1
            if not np.isnan(cell_s["p_a"]) and not np.isnan(b0_s["p_a"]):
                b0_p.append(b0_s["p_a"])
                cell_p.append(cell_s["p_a"])
        metrics["agreement_vs_b0"] = agree / n_eligible if n_eligible else float("nan")

        # KS test on p_a distribution vs B0
        if len(b0_p) >= 5:
            ks = stats.ks_2samp(b0_p, cell_p)
            metrics["ks_pa_vs_b0"] = float(ks.statistic)
            metrics["ks_pa_pvalue"] = float(ks.pvalue)
            # Centered weakening: shift in mean(|p_a - 0.5|)
            metrics["mean_abs_dev_b0"] = float(np.mean(np.abs(np.array(b0_p) - 0.5)))
            metrics["mean_abs_dev_cell"] = float(np.mean(np.abs(np.array(cell_p) - 0.5)))
            metrics["d_mean_abs_dev"] = metrics["mean_abs_dev_cell"] - metrics["mean_abs_dev_b0"]

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "summary.csv")
    args = parser.parse_args()

    cell_dirs = sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir() and (p / "measurements.jsonl").exists()])
    if not cell_dirs:
        print("No measurements found.")
        return

    cells = {p.name: load_cell(p) for p in cell_dirs}
    b0_df = cells.get("B0")
    if b0_df is None or b0_df.empty:
        print("WARNING: B0 not present — skipping vs-B0 metrics")
        b0_summary = None
    else:
        b0_summary = per_pair_choice_dist(b0_df)
        print(f"B0: {len(b0_summary)} pairs")

    rows = []
    for name, df in cells.items():
        if df.empty:
            continue
        m = cell_metrics(df, b0_summary if name != "B0" else None)
        m["cell"] = name
        rows.append(m)

    out = pd.DataFrame(rows)
    cols = ["cell"] + [c for c in out.columns if c != "cell"]
    out = out[cols]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
