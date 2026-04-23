"""Full bias analysis: for each (train, eval) pair, compute the probe's pull
toward *every* other persona, not just default.

For each (train, eval) pair with train != eval, and each observer persona X
with X != train, X != eval:

    bias(X | train, eval) = r(pred, X's utilities  |  eval, train)

where pred = probe(train) applied to eval's test activations. This is the
double-partial correlation: it isolates the probe-variance that aligns with X
after removing everything explained by eval and train.

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


def partial_corr_multi(y, x, Z):
    """Partial Pearson correlation of (y, x) controlling for one or more covariates Z."""
    n = len(y)
    D = np.column_stack(list(Z) + [np.ones(n)])
    def resid(v):
        beta, *_ = np.linalg.lstsq(D, v, rcond=None)
        return v - D @ beta
    return pearsonr(resid(x), resid(y))[0]


def main() -> None:
    personas = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker", "default"]
    test_ids = set(TEST_IDS_FILE.read_text().splitlines())

    utilities = {}
    acts = {}
    for p in personas:
        scores = load_thurstonian_scores(AL_DIR / f"{p}_test")
        utilities[p] = {tid: mu for tid, mu in scores.items() if tid in test_ids}
        tids, layer_acts = load_activations(activations_path(p), task_id_filter=test_ids, layers=[LAYER])
        acts[p] = (tids, layer_acts[LAYER])

    probes = {p: load_probe_weights(p) for p in personas}

    # Store a dict: bias[observer] = list of (train, eval, r_partial) with train != eval, observer not in {train, eval}
    bias_by_observer: dict[str, list[tuple[str, str, float]]] = {p: [] for p in personas}
    # Also the "self-bias" — when observer == train — which is r(pred, train | eval).
    train_self_bias: list[tuple[str, str, float]] = []

    for train_p in personas:
        probe = probes[train_p]
        for eval_p in personas:
            if eval_p == train_p:
                continue
            eval_tids, eval_acts = acts[eval_p]
            shared = [tid for tid in eval_tids if all(tid in utilities[p] for p in personas)]
            idx = np.array([i for i, tid in enumerate(eval_tids) if tid in set(shared)])
            shared_tids = eval_tids[idx]
            pred = eval_acts[idx] @ probe[:-1] + probe[-1]
            util_vec = {p: np.array([utilities[p][tid] for tid in shared_tids]) for p in personas}

            y_eval = util_vec[eval_p]
            y_train = util_vec[train_p]
            # Train self-bias: r(pred, train | eval)
            r_train_self = partial_corr_multi(pred, y_train, [y_eval])
            train_self_bias.append((train_p, eval_p, r_train_self))

            # Bias toward each other persona X (X != train, X != eval), double-partial
            for obs_p in personas:
                if obs_p == train_p or obs_p == eval_p:
                    continue
                y_obs = util_vec[obs_p]
                r_obs_double = partial_corr_multi(pred, y_obs, [y_eval, y_train])
                bias_by_observer[obs_p].append((train_p, eval_p, r_obs_double))

    # Save
    observers = [p for p in personas]
    n_by_obs = {p: len(bias_by_observer[p]) for p in observers}
    arr_by_obs = {p: np.array([x[2] for x in bias_by_observer[p]]) for p in observers}
    np.savez(
        RESULTS / "full_bias.npz",
        personas=np.array(personas),
        train_self_bias=np.array([x[2] for x in train_self_bias]),
        **{f"bias_toward_{p}": arr_by_obs[p] for p in observers},
    )
    print(f"saved {(RESULTS / 'full_bias.npz').relative_to(REPO)}")

    # Summary table
    print(f"\n--- Bias toward each persona X, isolated via r(pred, X | eval, train) ---")
    print(f"(30 valid (train, eval) pairs per observer persona, eot L32)")
    print(f"{'observer':>16s}  {'mean':>8s}  {'median':>8s}  {'% > 0':>8s}   {'sign test':>12s}")
    for p in observers:
        vals = arr_by_obs[p]
        mean = np.nanmean(vals); med = np.nanmedian(vals)
        pos = int((vals > 0).sum()); total = len(vals)
        print(f"{p:>16s}  {mean:>+8.3f}  {med:>+8.3f}  {pos/total:>8.0%}   {pos:>6d} / {total:>3d}")

    # Train self-bias for reference (this is the "pull toward the persona you were trained on")
    tsb = np.array([x[2] for x in train_self_bias])
    print(f"\n--- Reference: train self-bias  r(pred, train | eval)  (42 pairs incl. default) ---")
    print(f"  mean = {np.nanmean(tsb):+.3f};  median = {np.nanmedian(tsb):+.3f};  % > 0 = {(tsb > 0).sum()}/{len(tsb)}")


if __name__ == "__main__":
    main()
