"""Compute the partial correlation r(pred, u_eval | u_train) for every (train, eval)
persona pair, at the headline cell (tb-4, L38). Saves a 7×7 NPZ with both the
raw transfer r and the partial r controlling for the training persona's utilities.

Interpretation: after stripping the part of pred that is linearly predictable
from the training persona's utilities, how much alignment with the eval persona's
utilities remains? High = probe encodes something genuinely about eval beyond
"track training persona".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.core.storage import load_manifest
from src.probes.data_loading import load_thurstonian_scores

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
AL_DIR = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning"
PROBES_DIR = REPO / "results/probes/qwen_persona_sweep_final_six"
OUT_DIR = REPO / "experiments/qwen_replication/persona_transfer/probe_transfer/results"
TEST_IDS_FILE = REPO / "data/canonical_splits/test_task_ids.txt"

SELECTOR = "turn_boundary:-4"
SELECTOR_TAG = "tb-4"
LAYER = 38


def load_probe_weights(persona: str, selector_tag: str, layer: int) -> np.ndarray:
    name = f"{persona}_{selector_tag}"
    manifest = load_manifest(PROBES_DIR / name)
    for p in manifest["probes"]:
        if p["layer"] == layer and p["method"] == "ridge":
            return np.load(PROBES_DIR / name / p["file"])
    raise RuntimeError(f"No ridge probe at layer {layer} for {name}")


def partial_r(pred: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson r of (pred, y) after regressing out z from both."""
    pred_resid = pred - np.polyval(np.polyfit(z, pred, 1), z)
    y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
    r, _ = pearsonr(pred_resid, y_resid)
    return float(r)


def main() -> None:
    import json
    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    personas = ["default"] + list(data["metadata"]["final_six"])
    n = len(personas)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_ids = set(TEST_IDS_FILE.read_text().splitlines())

    test_utilities: dict[str, dict[str, float]] = {}
    for persona in personas:
        scores = load_thurstonian_scores(AL_DIR / f"{persona}_test")
        test_utilities[persona] = {tid: mu for tid, mu in scores.items() if tid in test_ids}

    persona_activations: dict[str, np.ndarray] = {}
    persona_test_ids: dict[str, list[str]] = {}
    for persona in personas:
        path = REPO / f"activations/qwen35_122b/pref_{persona}_sweep/activations_{SELECTOR}.npz"
        tids, layer_acts = load_activations(path, task_id_filter=test_ids, layers=[LAYER])
        persona_activations[persona] = layer_acts[LAYER]
        persona_test_ids[persona] = tids

    probes = {p: load_probe_weights(p, SELECTOR_TAG, LAYER) for p in personas}

    raw = np.full((n, n), np.nan)
    partial = np.full((n, n), np.nan)

    for i, train_p in enumerate(personas):
        for j, eval_p in enumerate(personas):
            eval_acts = persona_activations[eval_p]
            eval_tids = persona_test_ids[eval_p]
            preds = eval_acts @ probes[train_p][:-1] + probes[train_p][-1]

            train_util_lookup = test_utilities[train_p]
            eval_util_lookup = test_utilities[eval_p]
            mask = np.array([(tid in train_util_lookup) and (tid in eval_util_lookup) for tid in eval_tids])
            kept_tids = [tid for tid, m in zip(eval_tids, mask) if m]
            y_eval = np.array([eval_util_lookup[tid] for tid in kept_tids])
            y_train = np.array([train_util_lookup[tid] for tid in kept_tids])
            y_pred = preds[mask]

            r_raw, _ = pearsonr(y_pred, y_eval)
            raw[i, j] = r_raw
            if i == j:
                partial[i, j] = r_raw  # controlling for self is degenerate
            else:
                partial[i, j] = partial_r(y_pred, y_eval, y_train)

    np.savez(
        OUT_DIR / f"partial_eval_given_train_{SELECTOR_TAG}_L{LAYER}.npz",
        personas=np.array(personas),
        raw=raw,
        partial=partial,
        selector=SELECTOR,
        layer=LAYER,
    )
    diag = np.diag(partial).mean()
    off_mask = ~np.eye(n, dtype=bool)
    print(f"raw transfer r: diag={np.diag(raw).mean():.3f}  off-diag={raw[off_mask].mean():.3f}")
    print(f"partial r(pred,eval|train): diag={diag:.3f}  off-diag={partial[off_mask].mean():.3f}")
    print(f"saved partial_eval_given_train_{SELECTOR_TAG}_L{LAYER}.npz")


if __name__ == "__main__":
    main()
