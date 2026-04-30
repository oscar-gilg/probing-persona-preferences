"""Red-team the probe-transfer-to-sadist claim.

Concern: the probe trained on default-asst utilities gets r=+0.24 with sadist
utilities (probe transfer report) despite default-vs-sadist utility r = -0.15.
Could this be mechanical rather than substantive generalisation?

Tests:
1. r(probe, sadist_mu) — replicate the claim
2. r(probe, default_mu) — what does the probe actually predict best?
3. **Partial r(probe, sadist_mu | default_mu)** — does the probe carry any
   sadist-specific signal beyond default?
4. OLS sadist_mu ~ probe + default_mu — coefficient on probe is the
   persona-rotation-discriminating signal
5. Disagreement-weighted r — restrict to tasks where default and sadist
   utilities most disagree, retest
6. Permutation null on r(probe, sadist_mu)
7. Within-content-type partial correlations
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROBE_CSV = ROOT / "experiments/persona_cross_prefills/results/sadist_probe_top_bottom.csv"

PSF6_BASE = ROOT / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
DEFAULT_FILES = [
    PSF6_BASE / "completion_preference_gemma-3-27b_completion_canonical_seed0_train_task_ids/thurstonian_280a87c8.csv",
    PSF6_BASE / "completion_preference_gemma-3-27b_completion_canonical_seed0_eval_task_ids/thurstonian_b84bca67.csv",
    PSF6_BASE / "completion_preference_gemma-3-27b_completion_canonical_seed0_test_task_ids/thurstonian_b84bca67.csv",
]
SADIST_FILES = [
    PSF6_BASE / "completion_preference_gemma-3-27b_completion_canonical_seed0_sys319526ef_train_task_ids/thurstonian_893fe856.csv",
    PSF6_BASE / "completion_preference_gemma-3-27b_completion_canonical_seed0_sys319526ef_eval_task_ids/thurstonian_74cff8cd.csv",
    PSF6_BASE / "completion_preference_gemma-3-27b_completion_canonical_seed0_sys319526ef_test_task_ids/thurstonian_74cff8cd.csv",
]


def load_utilities(files):
    frames = []
    for p in files:
        df = pd.read_csv(p)[["task_id", "mu"]]
        df["split"] = p.parent.name.split("_")[-2]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates("task_id")


def partial_corr(x, y, z):
    """r(x, y | z): correlation of residuals after regressing each on z."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return np.corrcoef(rx, ry)[0, 1]


def main():
    print("Loading probe scores (computed in sadist_probe_top_bottom.py)...")
    probe = pd.read_csv(PROBE_CSV)[["task_id", "probe_score", "origin"]]

    print("Loading sadist utilities...")
    sadist = load_utilities(SADIST_FILES).rename(columns={"mu": "sadist_mu"})
    print(f"  {len(sadist)} sadist utilities")

    print("Loading default utilities...")
    default = load_utilities(DEFAULT_FILES).rename(columns={"mu": "default_mu"})
    print(f"  {len(default)} default utilities")

    df = probe.merge(sadist[["task_id", "sadist_mu", "split"]], on="task_id")
    df = df.merge(default[["task_id", "default_mu"]], on="task_id")
    print(f"  {len(df)} tasks have probe + both utilities\n")

    # Test 1: replicate the claim
    r_sadist = stats.pearsonr(df["probe_score"], df["sadist_mu"])
    print(f"[1] r(probe, sadist_mu)       = {r_sadist[0]:+.3f}  (p={r_sadist[1]:.2e}, n={len(df)})")

    # Test 2: what the probe is actually predicting
    r_default = stats.pearsonr(df["probe_score"], df["default_mu"])
    print(f"[2] r(probe, default_mu)      = {r_default[0]:+.3f}  (p={r_default[1]:.2e})")

    r_def_sad = stats.pearsonr(df["default_mu"], df["sadist_mu"])
    print(f"    r(default_mu, sadist_mu)  = {r_def_sad[0]:+.3f}  (p={r_def_sad[1]:.2e})  ← persona disagreement\n")

    # Test 3: PARTIAL correlation — does the probe carry sadist-specific signal?
    print("[3] PARTIAL CORRELATIONS (the key red-team test)")
    pr_sadist_given_default = partial_corr(
        df["probe_score"].values, df["sadist_mu"].values, df["default_mu"].values
    )
    pr_default_given_sadist = partial_corr(
        df["probe_score"].values, df["default_mu"].values, df["sadist_mu"].values
    )
    print(f"    r(probe, sadist  | default) = {pr_sadist_given_default:+.3f}   ← sadist-specific signal in the probe")
    print(f"    r(probe, default | sadist)  = {pr_default_given_sadist:+.3f}   ← default-specific signal in the probe")
    print()

    # Test 4: OLS decomposition
    print("[4] OLS: sadist_mu = α + β_probe·probe + β_default·default_mu")
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(len(df)), df["probe_score"], df["default_mu"]])
    y = df["sadist_mu"].values
    coef, *_ = lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid) / (len(y) - 3)))
    t = coef / se
    print(f"    intercept    = {coef[0]:+.3f}  (t={t[0]:+.2f})")
    print(f"    β_probe      = {coef[1]:+.3f}  (t={t[1]:+.2f})  ← non-zero ⇒ probe adds info beyond default")
    print(f"    β_default    = {coef[2]:+.3f}  (t={t[2]:+.2f})")
    print(f"    R²           = {1 - resid.var() / y.var():.3f}\n")

    # Test 5: disagreement subset — where default and sadist diverge
    df["disagreement"] = (df["default_mu"] - df["sadist_mu"]).abs()
    for q in [0.5, 0.7, 0.9]:
        thr = df["disagreement"].quantile(q)
        sub = df[df["disagreement"] >= thr]
        r = stats.pearsonr(sub["probe_score"], sub["sadist_mu"])
        rd = stats.pearsonr(sub["probe_score"], sub["default_mu"])
        print(f"[5] disagreement ≥ q{int(q*100)} ({thr:.1f})  n={len(sub):4d}  "
              f"r(probe,sadist)={r[0]:+.3f}  r(probe,default)={rd[0]:+.3f}")
    print()

    # Test 6: permutation null on r(probe, sadist)
    print("[6] Permutation null on r(probe, sadist_mu)")
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        shuffled = rng.permutation(df["sadist_mu"].values)
        null.append(np.corrcoef(df["probe_score"], shuffled)[0, 1])
    null = np.array(null)
    p_perm = float((np.abs(null) >= abs(r_sadist[0])).mean())
    print(f"    null mean ± std = {null.mean():+.4f} ± {null.std():.4f}")
    print(f"    observed r      = {r_sadist[0]:+.3f}")
    print(f"    p (two-sided)   = {p_perm:.3f}\n")

    # Test 7: within-content-type partial correlations
    print("[7] Within-content-type r(probe, sadist | default)")
    for origin, sub in df.groupby("origin"):
        if len(sub) < 20:
            continue
        try:
            pr = partial_corr(sub["probe_score"].values, sub["sadist_mu"].values, sub["default_mu"].values)
            r_raw = stats.pearsonr(sub["probe_score"], sub["sadist_mu"])[0]
            r_def = stats.pearsonr(sub["probe_score"], sub["default_mu"])[0]
            print(f"    {origin:11s} n={len(sub):4d}  r(probe,sadist)={r_raw:+.3f}  "
                  f"r(probe,default)={r_def:+.3f}  partial r(probe,sadist|default)={pr:+.3f}")
        except Exception as e:
            print(f"    {origin}: skipped ({e})")


if __name__ == "__main__":
    main()
