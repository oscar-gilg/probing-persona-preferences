"""Analyze persona probe transfer.

Loads the 14 trained probe sets (7 personas × 2 selectors, each with 5 layers)
and computes, for each (selector, layer) combination:

- Transfer matrix (7×7): Pearson r between each train-persona probe applied to
  each eval-persona's test-split activations vs the eval-persona's Thurstonian
  utilities on those tasks.
- Probe-cosine matrix (7×7): cosine similarity between the 7 probe weight vectors.

Also computes a single utility-utility matrix (7×7): Pearson r between each pair
of personas' Thurstonian utilities on the canonical test split.

Outputs:
- experiments/persona_sweep/probe_transfer/results/transfer_<selector>_L<layer>.npz
  (keys: personas, transfer_r, probe_cosine)
- experiments/persona_sweep/probe_transfer/results/utility_similarity.npz
  (keys: personas, utility_r)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.core.storage import load_manifest
from src.probes.data_loading import load_thurstonian_scores

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
AL_DIR = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
PROBES_DIR = REPO / "results/probes/persona_sweep_final_six"
OUT_DIR = REPO / "experiments/persona_sweep/probe_transfer/results"
TEST_IDS_FILE = REPO / "data/canonical_splits/test_task_ids.txt"

SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]
LAYERS = [25, 32, 39, 46, 53]


def activations_path(persona: str, selector: str) -> Path:
    base = "pref_main" if persona == "default" else f"pref_{persona}_sweep"
    return REPO / f"activations/gemma-3-27b_it/{base}/activations_{selector}.npz"


def load_probe_weights(persona: str, selector_tag: str, layer: int) -> np.ndarray:
    name = f"{persona}_{selector_tag}"
    manifest = load_manifest(PROBES_DIR / name)
    for p in manifest["probes"]:
        if p["layer"] == layer and p["method"] == "ridge":
            return np.load(PROBES_DIR / name / p["file"])
    raise RuntimeError(f"No ridge probe at layer {layer} for {name}")


def main() -> None:
    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    personas = list(data["metadata"]["final_six"]) + ["default"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_ids = set(TEST_IDS_FILE.read_text().splitlines())
    print(f"canonical test ids: {len(test_ids)}")

    test_utilities: dict[str, dict[str, float]] = {}
    for persona in personas:
        scores = load_thurstonian_scores(AL_DIR / f"{persona}_test")
        test_utilities[persona] = {tid: mu for tid, mu in scores.items() if tid in test_ids}
        print(f"  {persona}_test: {len(test_utilities[persona])} ids")

    # Utility-utility matrix (no probe involved)
    n = len(personas)
    utility_r = np.full((n, n), np.nan)
    shared_all = set.intersection(*[set(test_utilities[p].keys()) for p in personas])
    print(f"ids shared across all 7 personas: {len(shared_all)}")
    shared_sorted = sorted(shared_all)
    util_matrix = np.array([[test_utilities[p][tid] for tid in shared_sorted] for p in personas])
    for i in range(n):
        for j in range(n):
            r, _ = pearsonr(util_matrix[i], util_matrix[j])
            utility_r[i, j] = r
    np.savez(
        OUT_DIR / "utility_similarity.npz",
        personas=np.array(personas),
        utility_r=utility_r,
        shared_task_ids=np.array(shared_sorted),
    )
    print(f"saved utility_similarity.npz (n_shared={len(shared_sorted)})")

    # Per (selector, layer) transfer + cosine
    for selector in SELECTORS:
        selector_tag = selector.replace("turn_boundary:", "tb")
        persona_activations = {}
        persona_test_ids = {}
        for persona in personas:
            tids, layer_acts = load_activations(
                activations_path(persona, selector), task_id_filter=test_ids, layers=LAYERS,
            )
            persona_activations[persona] = layer_acts
            persona_test_ids[persona] = tids
            print(f"loaded {persona} {selector} activations: {len(tids)} tasks")

        for layer in LAYERS:
            transfer_r = np.full((n, n), np.nan)
            probe_cosine = np.full((n, n), np.nan)

            probes = {p: load_probe_weights(p, selector_tag, layer) for p in personas}

            # Probe cosine (on weight vectors excluding intercept)
            vectors = np.array([probes[p][:-1] for p in personas])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            probe_cosine = (vectors @ vectors.T) / (norms @ norms.T)

            # Transfer: apply train probe to eval activations, correlate with eval utilities
            for i, train_p in enumerate(personas):
                for j, eval_p in enumerate(personas):
                    eval_activations = persona_activations[eval_p][layer]
                    eval_tids = persona_test_ids[eval_p]
                    preds = eval_activations @ probes[train_p][:-1] + probes[train_p][-1]
                    util_lookup = test_utilities[eval_p]
                    mask = np.array([tid in util_lookup for tid in eval_tids])
                    y = np.array([util_lookup[tid] for tid, m in zip(eval_tids, mask) if m])
                    y_pred = preds[mask]
                    r, _ = pearsonr(y_pred, y)
                    transfer_r[i, j] = r

            out = OUT_DIR / f"transfer_{selector_tag}_L{layer}.npz"
            np.savez(
                out,
                personas=np.array(personas),
                transfer_r=transfer_r,
                probe_cosine=probe_cosine,
            )
            diag = np.diag(transfer_r).mean()
            off = transfer_r[~np.eye(n, dtype=bool)].mean()
            print(f"saved {out.relative_to(REPO)}: diag={diag:.3f} off={off:.3f}")


if __name__ == "__main__":
    main()
