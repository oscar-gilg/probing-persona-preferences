"""Preference-profile bias: does a transfer probe pull toward the train persona or default?

For each (train_persona, eval_persona) pair with train != eval, compute two per-topic
'midway ratios' from the generalised midway-bias metric:

    midway(anchor) = (pred_topic_mean - anchor_topic_mean) / (eval_topic_mean - anchor_topic_mean)

    1.0 => pred hits the eval persona's mean (no bias from this anchor)
    0.0 => pred stuck at the anchor's mean (fully biased toward anchor)

We compute this with two anchors:
    - anchor = train_persona (bias toward the persona the probe was trained on)
    - anchor = default (bias toward the assistant baseline)

Headline cell: eot selector, layer 32.

Outputs (experiments/persona_sweep/probe_transfer/results/profile_bias.npz):
    - personas: (n,)
    - midway_vs_train:   (n, n) mean over valid topics, NaN on diagonal
    - midway_vs_default: (n, n) mean over valid topics, NaN on diagonal
    - pull_train:   1 - midway_vs_train    (0 = hits eval; 1 = stuck at train)
    - pull_default: 1 - midway_vs_default  (0 = hits eval; 1 = stuck at default)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.probes.core.activations import load_activations
from src.probes.core.storage import load_manifest
from src.probes.data_loading import load_thurstonian_scores

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/persona_sweep/probe_transfer/results"
AL_DIR = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
PROBES_DIR = REPO / "results/probes/persona_sweep_final_six"
TEST_IDS_FILE = REPO / "data/canonical_splits/test_task_ids.txt"
TOPICS_PATH = REPO / "data/topics/topics.json"

SELECTOR = "turn_boundary:-5"
SELECTOR_TAG = "tb-5"
LAYER = 32
MIN_TOPIC_N = 5
MIN_DENOM = 0.1


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


def load_topic_map() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    topic_map = {}
    for tid, models in raw.items():
        for _, cats in models.items():
            topic_map[tid] = cats["primary"]
            break
    return topic_map


def main() -> None:
    personas = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker", "default"]
    test_ids = set(TEST_IDS_FILE.read_text().splitlines())
    topic_map = load_topic_map()

    # Per-persona: test utilities + test activations (filtered to canonical test split)
    utilities: dict[str, dict[str, float]] = {}
    acts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in personas:
        scores = load_thurstonian_scores(AL_DIR / f"{p}_test")
        utilities[p] = {tid: mu for tid, mu in scores.items() if tid in test_ids}
        tids, layer_acts = load_activations(activations_path(p), task_id_filter=test_ids, layers=[LAYER])
        acts[p] = (tids, layer_acts[LAYER])

    probes = {p: load_probe_weights(p) for p in personas}

    # Topic means per persona from actual utilities (consistent across probes).
    topic_means: dict[str, dict[str, float]] = {}
    topic_tasks: dict[str, dict[str, list[str]]] = {}
    for p in personas:
        by_topic: dict[str, list[float]] = defaultdict(list)
        tasks_by_topic: dict[str, list[str]] = defaultdict(list)
        for tid, mu in utilities[p].items():
            if tid in topic_map:
                by_topic[topic_map[tid]].append(mu)
                tasks_by_topic[topic_map[tid]].append(tid)
        topic_means[p] = {t: float(np.mean(v)) for t, v in by_topic.items() if len(v) >= MIN_TOPIC_N}
        topic_tasks[p] = {t: v for t, v in tasks_by_topic.items() if len(v) >= MIN_TOPIC_N}

    n = len(personas)
    midway_vs_train = np.full((n, n), np.nan)
    midway_vs_default = np.full((n, n), np.nan)
    per_pair_n_topics = np.zeros((n, n), dtype=int)

    i_def = personas.index("default")

    for i_train, train_p in enumerate(personas):
        probe = probes[train_p]
        for i_eval, eval_p in enumerate(personas):
            if eval_p == train_p:
                continue
            eval_tids, eval_acts = acts[eval_p]
            preds = eval_acts @ probe[:-1] + probe[-1]
            pred_by_tid = {tid: float(v) for tid, v in zip(eval_tids, preds)}

            ratios_train = []
            ratios_default = []
            for topic, tids in topic_tasks[eval_p].items():
                eval_mean = topic_means[eval_p].get(topic)
                train_mean = topic_means[train_p].get(topic)
                default_mean = topic_means["default"].get(topic)
                if eval_mean is None or train_mean is None or default_mean is None:
                    continue
                preds_topic = [pred_by_tid[tid] for tid in tids if tid in pred_by_tid]
                if len(preds_topic) < MIN_TOPIC_N:
                    continue
                pred_mean = float(np.mean(preds_topic))

                denom_train = eval_mean - train_mean
                denom_default = eval_mean - default_mean
                if abs(denom_train) >= MIN_DENOM:
                    ratios_train.append((pred_mean - train_mean) / denom_train)
                if abs(denom_default) >= MIN_DENOM:
                    ratios_default.append((pred_mean - default_mean) / denom_default)

            # Use median across topics — robust to small-denominator noise.
            if ratios_train:
                midway_vs_train[i_train, i_eval] = float(np.median(ratios_train))
            if ratios_default and i_eval != i_def and i_train != i_def:
                midway_vs_default[i_train, i_eval] = float(np.median(ratios_default))
            per_pair_n_topics[i_train, i_eval] = min(len(ratios_train), len(ratios_default))

    pull_train = 1.0 - midway_vs_train
    pull_default = 1.0 - midway_vs_default

    np.savez(
        RESULTS / "profile_bias.npz",
        personas=np.array(personas),
        midway_vs_train=midway_vs_train,
        midway_vs_default=midway_vs_default,
        pull_train=pull_train,
        pull_default=pull_default,
        per_pair_n_topics=per_pair_n_topics,
    )
    print(f"saved {(RESULTS / 'profile_bias.npz').relative_to(REPO)}")

    # Summary: per (train, eval) pair, pull toward train vs pull toward default.
    print("\n--- pull_train (NaN on diagonal) — eot, L32 ---")
    print("rows: train persona, cols: eval persona")
    print(f"{'':15s}  " + "  ".join(f"{p[:8]:>8s}" for p in personas))
    for i, p in enumerate(personas):
        row = f"{p:15s}  " + "  ".join(
            f"{pull_train[i, j]:>+8.2f}" if not np.isnan(pull_train[i, j]) else f"{'—':>8s}"
            for j in range(n)
        )
        print(row)

    print("\n--- pull_default (NaN on diagonal, and on the default row/col where denom=0) ---")
    for i, p in enumerate(personas):
        row = f"{p:15s}  " + "  ".join(
            f"{pull_default[i, j]:>+8.2f}" if not np.isnan(pull_default[i, j]) else f"{'—':>8s}"
            for j in range(n)
        )
        print(row)

    # Head-to-head: mean off-diagonal (excluding default row/col) per pair
    non_def_mask = np.ones((n, n), dtype=bool)
    non_def_mask[:, i_def] = False
    non_def_mask[i_def, :] = False
    np.fill_diagonal(non_def_mask, False)
    pt = pull_train[non_def_mask]
    pd_ = pull_default[non_def_mask]
    print(f"\n--- topic-median pulls (30 non-default off-diag pairs) ---")
    print(f"Median pull_train:   {np.nanmedian(pt):+.3f}")
    print(f"Median pull_default: {np.nanmedian(pd_):+.3f}")
    print(f"Pairs with pull_train > pull_default: {np.sum(pt > pd_)}/{len(pt)}")



if __name__ == "__main__":
    main()
