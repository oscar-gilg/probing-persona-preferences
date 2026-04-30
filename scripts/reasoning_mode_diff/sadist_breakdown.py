"""Per-origin breakdown of reasoning ON vs OFF for the sadist persona.

Mirrors the default-persona analysis in
scripts.qwen_persona_transfer.reasoning_mode_diff_analysis_v2 but for sadist.
Emits a JSON summary + a markdown table block ready to paste into the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from scripts.qwen_persona_transfer.reasoning_mode_diff_analysis_v2 import (
    NO_THINK_AL,
    THINK_AL,
    load_persona_all_splits,
    origin_from_id,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff/results"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    nt = load_persona_all_splits(NO_THINK_AL, "sadist")
    th = load_persona_all_splits(THINK_AL, "sadist")
    shared = sorted(set(nt) & set(th))
    print(f"sadist: no_think={len(nt)}, think={len(th)}, shared={len(shared)}")

    nta = np.array([nt[t] for t in shared])
    tha = np.array([th[t] for t in shared])
    origins = np.array([origin_from_id(t) for t in shared])

    overall_r = pearsonr(nta, tha)[0]
    overall_rho = spearmanr(nta, tha)[0]
    print(f"overall: Pearson r = {overall_r:+.3f}, Spearman ρ = {overall_rho:+.3f}")

    rows = []
    print("\nper-origin (sadist):")
    for origin in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]:
        m = origins == origin
        if m.sum() < 2:
            continue
        r = pearsonr(nta[m], tha[m])[0]
        rho = spearmanr(nta[m], tha[m])[0]
        nt_mean = nta[m].mean(); th_mean = tha[m].mean()
        nt_std = nta[m].std(); th_std = tha[m].std()
        flips = ((nta[m] > 0) != (tha[m] > 0)).sum()
        rows.append({
            "origin": origin, "n": int(m.sum()),
            "pearson": float(r), "spearman": float(rho),
            "mean_no_think": float(nt_mean), "mean_think": float(th_mean),
            "std_no_think": float(nt_std), "std_think": float(th_std),
            "sign_flip_rate": float(flips / m.sum()),
        })
        print(f"  {origin:<12} n={m.sum():>5}  r={r:+.3f}  ρ={rho:+.3f}  "
              f"μ̄_no={nt_mean:+.2f}  μ̄_th={th_mean:+.2f}  flip={flips/m.sum()*100:.0f}%")

    summary = {
        "persona": "sadist",
        "n_shared": len(shared),
        "overall_pearson": float(overall_r),
        "overall_spearman": float(overall_rho),
        "per_origin": rows,
    }
    out_path = OUT / "summary_v2_sadist.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_path.relative_to(REPO)}")

    print("\n--- markdown table ---")
    print("| origin | n | Pearson r | Spearman ρ | mean μ OFF | mean μ ON | sign-flip rate |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['origin']} | {r['n']} | {r['pearson']:+.3f} | {r['spearman']:+.3f} | "
              f"{r['mean_no_think']:+.2f} | {r['mean_think']:+.2f} | {r['sign_flip_rate']*100:.0f}% |")
    print(f"| **overall** | **{len(shared)}** | **{overall_r:+.3f}** | **{overall_rho:+.3f}** | — | — | — |")


if __name__ == "__main__":
    main()
