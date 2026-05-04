"""Build train/test splits for the truth and harm probes.

Outputs (under experiments/probe_persona_drift/results/splits/):
- truth_heldout.json:    {"task_ids": [...], "labels": {id: "true"|"false"}}
- truth_train.json:      {"50": [...], "200": [...], "1000": [...], "4000": [...]}
- harm_heldout.json:     {"task_ids": [...], "labels": {id: "harmful"|"benign"}, "sources": {id: source}}
- harm_train.json:       same shape as truth_train but harm IDs

Conventions:
- Seed 42, deterministic.
- 500 held-out / target, drawn first from a balanced pool, never overlapping with train.
- Train sizes are NESTED subsets: size_k ⊂ size_{k+1}.
- Class balance: roughly 50/50 per train size and held-out.
- Harm sources: BailBench harmful + HarmBench harmful; Alpaca + WildChat benign (50/50 mix).
"""

import json
from pathlib import Path

import numpy as np

from src.task_data.loader import _load_origin
from src.task_data.task import OriginDataset


SEED = 42
HELDOUT_SIZE = 500  # total per target (250 per class)
TRAIN_SIZES = [50, 200, 1000, 4000]  # total per target

OUT_DIR = Path("experiments/probe_persona_drift/results/splits")
CREAK_KNOWN_CORRECT = Path("data/creak/known_correct_gemma-3-27b.json")


def build_truth_splits(rng: np.random.Generator) -> dict:
    """CREAK known-correct items, balanced true/false, nested splits."""
    known_correct_ids = set(json.load(open(CREAK_KNOWN_CORRECT))["task_ids"])
    creak_tasks = [t for t in _load_origin(OriginDataset.CREAK) if t.id in known_correct_ids]
    by_label: dict[str, list[str]] = {"true": [], "false": []}
    for t in creak_tasks:
        by_label[t.metadata["label"]].append(t.id)
    print(f"  CREAK known-correct: {len(by_label['true'])} true, {len(by_label['false'])} false")

    for label in by_label:
        rng.shuffle(by_label[label])

    heldout_per_class = HELDOUT_SIZE // 2
    heldout_ids = by_label["true"][:heldout_per_class] + by_label["false"][:heldout_per_class]
    pool = {
        "true": by_label["true"][heldout_per_class:],
        "false": by_label["false"][heldout_per_class:],
    }
    available = min(len(pool["true"]), len(pool["false"])) * 2
    print(f"  truth train pool: {available} (held-out: {len(heldout_ids)})")

    train_splits = {}
    prev_set: set[str] = set()
    for size in TRAIN_SIZES:
        if size > available:
            print(f"  truth size {size} > pool {available}, capping")
            size = available
        per_class = size // 2
        ids = pool["true"][:per_class] + pool["false"][:per_class]
        assert prev_set.issubset(set(ids)), f"truth size {size} broke nesting"
        prev_set = set(ids)
        train_splits[str(size)] = ids
        print(f"  truth train_{size}: {len(ids)} ({per_class} per class)")

    labels = {tid: "true" for tid in by_label["true"]}
    labels.update({tid: "false" for tid in by_label["false"]})
    return {
        "heldout": {"task_ids": heldout_ids, "labels": {tid: labels[tid] for tid in heldout_ids}},
        "train": {"task_ids": train_splits, "labels": {tid: labels[tid] for sz_ids in train_splits.values() for tid in sz_ids}},
    }


def build_harm_splits(rng: np.random.Generator) -> dict:
    """BailBench + HarmBench (harmful) vs Alpaca + WildChat 50/50 (benign)."""
    bail = _load_origin(OriginDataset.BAILBENCH)
    harmbench = _load_origin(OriginDataset.HARMBENCH)
    alpaca = _load_origin(OriginDataset.ALPACA)
    wildchat = _load_origin(OriginDataset.WILDCHAT)

    harmful_ids = [t.id for t in bail] + [t.id for t in harmbench]
    bail_id_map = {t.id: t for t in bail}
    harmbench_id_map = {t.id: t for t in harmbench}
    sources = {t.id: "bailbench" for t in bail}
    sources.update({t.id: "harmbench" for t in harmbench})
    print(f"  harmful pool: {len(bail)} bailbench + {len(harmbench)} harmbench = {len(harmful_ids)}")

    rng.shuffle(harmful_ids)

    alpaca_ids = [t.id for t in alpaca]
    wildchat_ids = [t.id for t in wildchat]
    rng.shuffle(alpaca_ids)
    rng.shuffle(wildchat_ids)
    print(f"  benign pool: {len(alpaca)} alpaca + {len(wildchat)} wildchat")

    heldout_per_class = HELDOUT_SIZE // 2
    heldout_harmful = harmful_ids[:heldout_per_class]
    half = heldout_per_class // 2
    heldout_benign = alpaca_ids[:half] + wildchat_ids[:half]
    rng.shuffle(heldout_benign)
    heldout_ids = heldout_harmful + heldout_benign
    for tid in alpaca_ids[:half]:
        sources[tid] = "alpaca"
    for tid in wildchat_ids[:half]:
        sources[tid] = "wildchat"

    pool_harmful = harmful_ids[heldout_per_class:]
    pool_alpaca = alpaca_ids[half:]
    pool_wildchat = wildchat_ids[half:]
    available = 2 * min(len(pool_harmful), len(pool_alpaca) + len(pool_wildchat))
    print(f"  harm train pool: {available} (held-out: {len(heldout_ids)})")

    train_splits = {}
    prev_set: set[str] = set()
    for size in TRAIN_SIZES:
        if size > available:
            print(f"  harm size {size} > pool {available}, capping")
            size = available
        per_class = size // 2
        half_per_class = per_class // 2
        harmful_chunk = pool_harmful[:per_class]
        benign_chunk = pool_alpaca[:half_per_class] + pool_wildchat[:per_class - half_per_class]
        ids = harmful_chunk + benign_chunk
        for tid in pool_alpaca[:half_per_class]:
            sources.setdefault(tid, "alpaca")
        for tid in pool_wildchat[:per_class - half_per_class]:
            sources.setdefault(tid, "wildchat")
        assert prev_set.issubset(set(ids)), f"harm size {size} broke nesting"
        prev_set = set(ids)
        train_splits[str(size)] = ids
        print(f"  harm train_{size}: {len(ids)} ({per_class} harmful, {len(benign_chunk)} benign)")

    labels: dict[str, str] = {}
    for tid in harmful_ids:
        labels[tid] = "harmful"
    for tid in alpaca_ids + wildchat_ids:
        labels[tid] = "benign"

    used_ids = set(heldout_ids)
    for ids in train_splits.values():
        used_ids.update(ids)

    prompts = {}
    for tid in used_ids:
        if tid in bail_id_map:
            prompts[tid] = bail_id_map[tid].prompt
        elif tid in harmbench_id_map:
            prompts[tid] = harmbench_id_map[tid].prompt
    alpaca_prompt_map = {t.id: t.prompt for t in alpaca}
    wildchat_prompt_map = {t.id: t.prompt for t in wildchat}
    for tid in used_ids:
        if tid in alpaca_prompt_map:
            prompts[tid] = alpaca_prompt_map[tid]
        elif tid in wildchat_prompt_map:
            prompts[tid] = wildchat_prompt_map[tid]

    return {
        "heldout": {
            "task_ids": heldout_ids,
            "labels": {tid: labels[tid] for tid in heldout_ids},
            "sources": {tid: sources[tid] for tid in heldout_ids if tid in sources},
            "prompts": {tid: prompts[tid] for tid in heldout_ids if tid in prompts},
        },
        "train": {
            "task_ids": train_splits,
            "labels": {tid: labels[tid] for sz_ids in train_splits.values() for tid in sz_ids},
            "sources": {tid: sources[tid] for sz_ids in train_splits.values() for tid in sz_ids if tid in sources},
            "prompts": {tid: prompts[tid] for sz_ids in train_splits.values() for tid in sz_ids if tid in prompts},
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Building TRUTH splits...")
    truth = build_truth_splits(rng)
    print("Building HARM splits...")
    harm = build_harm_splits(rng)

    with open(OUT_DIR / "truth_heldout.json", "w") as f:
        json.dump(truth["heldout"], f, indent=2)
    with open(OUT_DIR / "truth_train.json", "w") as f:
        json.dump(truth["train"], f, indent=2)
    with open(OUT_DIR / "harm_heldout.json", "w") as f:
        json.dump(harm["heldout"], f, indent=2)
    with open(OUT_DIR / "harm_train.json", "w") as f:
        json.dump(harm["train"], f, indent=2)

    print(f"\nWrote splits to {OUT_DIR}")


if __name__ == "__main__":
    main()
