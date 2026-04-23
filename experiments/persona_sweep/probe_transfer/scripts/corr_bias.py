"""Correlation-based preference-profile bias.

For each (train_persona, eval_persona) pair with train != eval, compute:

  r_eval    = Pearson(pred, eval_utilities)       # transfer r
  r_train   = Pearson(pred, train_utilities)      # pull toward train (pattern)
  r_default = Pearson(pred, default_utilities)    # pull toward default (pattern)

where pred = probe(train) applied to eval's test-split activations.

Also a partial-correlation view that controls for eval to isolate residual bias:

  r_train   | eval  : how much of pred's variance matches train after eval is removed
  r_default | eval  : how much of pred's variance matches default after eval is removed

Headline cell: eot (turn_boundary:-5), layer 32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.core.storage import load_manifest
from src.probes.data_loading import load_thurstonian_scores

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
AL_DIR = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
PROBES_DIR = REPO / "results/probes/persona_sweep_final_six"
TEST_IDS_FILE = REPO / "data/canonical_splits/test_task_ids.txt"

SELECTOR = "turn_boundary:-5"
SELECTOR_TAG = "tb-5"
LAYER = 32


def activations_path(persona: str) -> Path:
    base = "pref_main" if persona == "default" else f"pref_{persona}_sweep"
    return REPO / f"activations/gemma-3-27b_it/{base}/activations_{SELECTOR}.npz"


def load_probe_weights(persona: str) -> np.ndarray:
    name = f"{persona}_{SELECTOR_TAG}"
    manifest = load_manifest(PROBES_DIR / name)
    for p in manifest["probes"]:
        if p["layer"] == LAYER and p["method"] == "ridge":
            return np.load(PROBES_DIR / name / p["file"])
    raise RuntimeError(f"no ridge probe at layer {LAYER} for {name}")


def partial_corr(y, x, z):
    """Partial Pearson correlation of (y, x) controlling for z (1-D arrays)."""
    a = np.polyfit(z, y, 1); y_res = y - (a[0] * z + a[1])
    b = np.polyfit(z, x, 1); x_res = x - (b[0] * z + b[1])
    return pearsonr(x_res, y_res)[0]


def main() -> None:
    personas = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker", "default"]
    test_ids = set(TEST_IDS_FILE.read_text().splitlines())

    utilities: dict[str, dict[str, float]] = {}
    acts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in personas:
        scores = load_thurstonian_scores(AL_DIR / f"{p}_test")
        utilities[p] = {tid: mu for tid, mu in scores.items() if tid in test_ids}
        tids, layer_acts = load_activations(activations_path(p), task_id_filter=test_ids, layers=[LAYER])
        acts[p] = (tids, layer_acts[LAYER])

    probes = {p: load_probe_weights(p) for p in personas}

    n = len(personas)
    r_eval = np.full((n, n), np.nan)
    r_train = np.full((n, n), np.nan)
    r_default = np.full((n, n), np.nan)
    r_train_partial = np.full((n, n), np.nan)
    r_default_partial = np.full((n, n), np.nan)

    i_def = personas.index("default")

    for i_train, train_p in enumerate(personas):
        probe = probes[train_p]
        for i_eval, eval_p in enumerate(personas):
            if eval_p == train_p:
                continue
            eval_tids, eval_acts = acts[eval_p]
            # Align tasks: need ids present in all four utility dicts for fair comparison.
            shared = [tid for tid in eval_tids
                      if tid in utilities[eval_p] and tid in utilities[train_p]
                      and tid in utilities["default"]]
            idx = np.array([i for i, tid in enumerate(eval_tids) if tid in set(shared)])
            pred = eval_acts[idx] @ probe[:-1] + probe[-1]
            shared_tids = eval_tids[idx]
            y_eval = np.array([utilities[eval_p][tid] for tid in shared_tids])
            y_train = np.array([utilities[train_p][tid] for tid in shared_tids])
            y_default = np.array([utilities["default"][tid] for tid in shared_tids])

            r_eval[i_train, i_eval] = pearsonr(pred, y_eval)[0]
            r_train[i_train, i_eval] = pearsonr(pred, y_train)[0]
            if i_train != i_def and i_eval != i_def:
                r_default[i_train, i_eval] = pearsonr(pred, y_default)[0]
            r_train_partial[i_train, i_eval] = partial_corr(pred, y_train, y_eval)
            if i_train != i_def and i_eval != i_def:
                r_default_partial[i_train, i_eval] = partial_corr(pred, y_default, y_eval)

    np.savez(
        RESULTS / "corr_bias.npz",
        personas=np.array(personas),
        r_eval=r_eval, r_train=r_train, r_default=r_default,
        r_train_partial=r_train_partial, r_default_partial=r_default_partial,
    )
    print(f"saved {(RESULTS / 'corr_bias.npz').relative_to(REPO)}")

    non_def_mask = np.ones((n, n), dtype=bool)
    non_def_mask[:, i_def] = False
    non_def_mask[i_def, :] = False
    np.fill_diagonal(non_def_mask, False)

    re = r_eval[non_def_mask]
    rt = r_train[non_def_mask]
    rd = r_default[non_def_mask]
    rtp = r_train_partial[non_def_mask]
    rdp = r_default_partial[non_def_mask]

    print(f"\nAcross 30 non-default off-diagonal pairs, eot L32:")
    print(f"  mean r(pred, eval)     = {np.nanmean(re):+.3f}   (= transfer r)")
    print(f"  mean r(pred, train)    = {np.nanmean(rt):+.3f}")
    print(f"  mean r(pred, default)  = {np.nanmean(rd):+.3f}")
    print()
    print(f"  r(pred, train) > r(pred, eval):    {int((rt > re).sum())}/{len(rt)}  pairs")
    print(f"  r(pred, default) > r(pred, eval):  {int((rd > re).sum())}/{len(rd)}  pairs")
    print()
    print(f"  Partial correlations (controlling for eval):")
    print(f"    mean r(pred, train    | eval) = {np.nanmean(rtp):+.3f}")
    print(f"    mean r(pred, default  | eval) = {np.nanmean(rdp):+.3f}")
    print(f"  Pairs with r(pred, train | eval) > r(pred, default | eval): {int((rtp > rdp).sum())}/{len(rtp)}")

    print("\nr(pred, train) table — rows = train probe, cols = eval persona")
    print(f"{'':15s}  " + "  ".join(f"{p[:8]:>8s}" for p in personas))
    for i, p in enumerate(personas):
        row = f"{p:15s}  " + "  ".join(
            f"{r_train[i, j]:>+8.2f}" if not np.isnan(r_train[i, j]) else f"{'—':>8s}"
            for j in range(n)
        )
        print(row)

    print("\nr(pred, default) table")
    print(f"{'':15s}  " + "  ".join(f"{p[:8]:>8s}" for p in personas))
    for i, p in enumerate(personas):
        row = f"{p:15s}  " + "  ".join(
            f"{r_default[i, j]:>+8.2f}" if not np.isnan(r_default[i, j]) else f"{'—':>8s}"
            for j in range(n)
        )
        print(row)


if __name__ == "__main__":
    main()
