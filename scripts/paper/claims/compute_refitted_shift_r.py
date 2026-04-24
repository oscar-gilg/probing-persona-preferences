"""Register §4.1 Gemma refitted-utility shifted-prediction r.

Paper prose: "re-fitting utilities under each prompt, the default-persona probe
predicts the shifted utilities at r = 0.63". This is the mean across the 16
Exp 1b conditions of `cond_probe_r` at L31 produced by the utility_fitting
sub-experiment (n=48 tasks per condition), matching
`utility_fitting_report.md` line 40.

Note: do not use the parent `ood_system_prompts/analysis_results.json` — it is
an older run with n=40 and gives 0.74.

Run:
  python scripts/paper/claims/compute_refitted_shift_r.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_REL = (
    "experiments/old_experiments/probe_generalization/ood_system_prompts/"
    "utility_fitting/analysis_results.json"
)


def main() -> None:
    rows = json.loads((REPO_ROOT / ANALYSIS_REL).read_text())
    exp1b = [r for r in rows if r["experiment"] == "exp1b_hidden"]
    rs = np.array([r["cond_probe_r"] for r in exp1b])
    mean_r = float(rs.mean())
    sem_r = float(rs.std(ddof=1) / np.sqrt(len(rs)))

    claims = ClaimSet(source="scripts/paper/claims/compute_refitted_shift_r.py")
    claims.register(
        "Gemma refitted-utility shifted-prediction r",
        round(mean_r, 2),
        "Mean across Exp 1b conditions of the Pearson r between the default-"
        "persona Gemma-3-27B ridge probe (L31) applied under each system prompt "
        "and the Thurstonian utilities refit from pairwise choices under that "
        "same prompt. Matches §4.1 prose (r = 0.63) and utility_fitting_report "
        "line 40.",
        used_in=["sec:induced-basic"],
        data_paths=[ANALYSIS_REL],
        derivation=(
            "Load analysis_results.json (list of per-condition rows); filter "
            "experiment=='exp1b_hidden'; take mean of `cond_probe_r`; round to "
            f"2dp. Computed mean = {mean_r:.4f}, SEM = {sem_r:.4f}, "
            f"n_conditions = {len(rs)}."
        ),
    )
    sidecar = REPO_ROOT / "paper" / "claims" / "refitted_shift_r.json"
    claims.save(sidecar)
    print(f"Saved {sidecar.relative_to(REPO_ROOT)}  ({len(claims.claims)} claims)")
    print(f"  mean r = {mean_r:.4f} across {len(rs)} conditions")


if __name__ == "__main__":
    main()
