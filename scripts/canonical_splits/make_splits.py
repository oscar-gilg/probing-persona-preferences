"""Create canonical train/eval/test task_id splits for cross-persona experiments.

Output: data/canonical_splits/{train,eval,test}_task_ids.txt (4000 / 1000 / 1000).

Constraints:
- Drawn from the 30k-task pool already extracted at activations/gemma-3-27b_it/pref_main/.
- Disjoint from each other.
- Disjoint from all existing experiment exclude-lists + mra_exp2 splits.
- Stratified by (primary topic, dataset origin) so each split preserves the joint distribution.
- Deterministic (seed=42).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
ACT_NPZ = ROOT / "activations/gemma-3-27b_it/pref_main/activations_turn_boundary:-2.npz"
TOPICS_JSON = ROOT / "data/topics/topics.json"
AL_DIR = ROOT / "configs/measurement/active_learning"
OUT_DIR = ROOT / "data/canonical_splits"
SEED = 42

EXCLUDE_FILES = [
    AL_DIR / "exclude_gemma3_10k.txt",      # main probe train (10k)
    AL_DIR / "exclude_4k_task_ids.txt",     # main probe eval (4k)
    AL_DIR / "exclude_3k_task_ids.txt",     # earlier 3k run
    AL_DIR / "exclude_gpt_oss_3k.txt",      # gpt-oss measurements
    AL_DIR / "exclude_gptoss_3k_eval.txt",  # gpt-oss eval
    AL_DIR / "mra_exp2_split_a_1000_task_ids.txt",
    AL_DIR / "mra_exp2_split_b_500_task_ids.txt",
    AL_DIR / "mra_exp2_split_c_1000_task_ids.txt",
]

SPLIT_SIZES = {"train": 4000, "eval": 1000, "test": 1000}
TOPIC_MODEL = "anthropic/claude-sonnet-4.5"  # which classifier to use in topics.json

# Dataset proportions per split. stresstest dominates the pool (75%), so cap it and
# balance against the smaller sources. Proportions normalized to 1.
DATASET_FRAC = {
    "stresstest": 0.40,
    "competition_math": 0.20,
    "alpaca": 0.20,
    "wildchat": 0.20,
}


def dataset_prefix(task_id: str) -> str:
    for prefix in ("wildchat", "alpaca", "competition_math", "math", "bailbench", "stresstest"):
        if task_id.startswith(prefix + "_") or task_id.startswith(prefix):
            return prefix
    return task_id.split("_")[0]


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    print("Loading activations...")
    data = np.load(ACT_NPZ, allow_pickle=True)
    all_tids = [str(t) for t in data["task_ids"]]
    print(f"  activation pool: {len(all_tids)}")

    print("Loading topics...")
    topics = json.loads(TOPICS_JSON.read_text())
    print(f"  topics.json entries: {len(topics)}")

    print("Loading exclusion lists...")
    excluded: set[str] = set()
    for p in EXCLUDE_FILES:
        ids = load_ids(p)
        print(f"  {p.name}: {len(ids)}")
        excluded |= ids
    print(f"  total unique excluded: {len(excluded)}")

    # Build candidate pool.
    candidates = []
    no_topic = 0
    for tid in all_tids:
        if tid in excluded:
            continue
        entry = topics.get(tid)
        if not entry or TOPIC_MODEL not in entry:
            no_topic += 1
            continue
        primary = entry[TOPIC_MODEL]["primary"]
        ds = dataset_prefix(tid)
        candidates.append((tid, primary, ds))
    print(f"\nCandidate pool: {len(candidates)} tasks (excluded {len(all_tids) - len(candidates)}; "
          f"{no_topic} of those lacked topic)")

    # Stratum counts.
    stratum_counter: Counter = Counter((t, d) for _, t, d in candidates)
    print(f"\nStratum (topic, dataset) distribution — top 15:")
    for (t, d), n in stratum_counter.most_common(15):
        print(f"  {t:20s} {d:20s} {n}")

    total_target = sum(SPLIT_SIZES.values())
    if len(candidates) < total_target:
        raise RuntimeError(
            f"Candidate pool ({len(candidates)}) smaller than requested split sum ({total_target})"
        )

    # Group candidates by (dataset, topic). Sample proportionally within each dataset (topic-stratified).
    rng = random.Random(SEED)
    by_dataset_topic: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for tid, t, d in candidates:
        by_dataset_topic[d][t].append(tid)
    for d in by_dataset_topic:
        for t in by_dataset_topic[d]:
            rng.shuffle(by_dataset_topic[d][t])

    train_ids: list[str] = []
    eval_ids: list[str] = []
    test_ids: list[str] = []

    # Step 1: compute a per-dataset total budget, target = total_split × DATASET_FRAC.
    per_dataset_target = {ds: round(total_target * frac) for ds, frac in DATASET_FRAC.items()}
    per_dataset_avail = {ds: sum(len(v) for v in by_dataset_topic[ds].values()) for ds in DATASET_FRAC}
    per_dataset_budget = {ds: min(per_dataset_target[ds], per_dataset_avail[ds]) for ds in DATASET_FRAC}

    # If we lost capacity (e.g., wildchat < target), top up from the dataset with most headroom (stresstest).
    deficit = total_target - sum(per_dataset_budget.values())
    if deficit > 0:
        headroom_ds = max(DATASET_FRAC, key=lambda d: per_dataset_avail[d] - per_dataset_budget[d])
        extra = min(deficit, per_dataset_avail[headroom_ds] - per_dataset_budget[headroom_ds])
        per_dataset_budget[headroom_ds] += extra
        print(f"  Topped up {headroom_ds} by {extra} to cover deficit from constrained datasets.")

    print(f"\nPer-dataset budgets: {per_dataset_budget}")

    # Step 2: within each dataset, split budget 4:1:1 across train/eval/test, topic-stratified.
    for ds, budget in per_dataset_budget.items():
        if budget == 0:
            continue
        # Flatten the dataset's tasks, stratified by topic: round-robin over topics shuffled.
        ds_ordered: list[str] = []
        topic_sizes = {t: len(tids) for t, tids in by_dataset_topic[ds].items() if len(tids) > 0}
        total_avail = sum(topic_sizes.values())
        # Topic-proportional allocation of the dataset's budget.
        topic_quotas = {t: round(budget * n / total_avail) for t, n in topic_sizes.items()}
        drift = budget - sum(topic_quotas.values())
        if drift != 0 and topic_quotas:
            biggest = max(topic_quotas, key=topic_quotas.get)  # type: ignore[arg-type]
            topic_quotas[biggest] += drift
        for topic, tq in topic_quotas.items():
            pool = by_dataset_topic[ds][topic]
            take = min(tq, len(pool))
            ds_ordered.extend(pool[:take])
            by_dataset_topic[ds][topic] = pool[take:]
        # Within ds_ordered, shuffle and split 4:1:1.
        rng.shuffle(ds_ordered)
        n_tr = round(len(ds_ordered) * SPLIT_SIZES["train"] / total_target)
        n_ev = round(len(ds_ordered) * SPLIT_SIZES["eval"] / total_target)
        n_te = len(ds_ordered) - n_tr - n_ev
        train_ids.extend(ds_ordered[:n_tr])
        eval_ids.extend(ds_ordered[n_tr:n_tr + n_ev])
        test_ids.extend(ds_ordered[n_tr + n_ev:n_tr + n_ev + n_te])

    leftover: list[str] = []
    for ds in by_dataset_topic:
        for topic in by_dataset_topic[ds]:
            leftover.extend(by_dataset_topic[ds][topic])

    print(f"\nFinal split sizes: train={len(train_ids)} eval={len(eval_ids)} test={len(test_ids)}")
    print(f"  (targets: train=4000 eval=1000 test=1000)")
    print(f"Leftover (unassigned candidates): {len(leftover)}")

    # Audit: disjointness.
    sets = {"train": set(train_ids), "eval": set(eval_ids), "test": set(test_ids)}
    for a, b in [("train", "eval"), ("train", "test"), ("eval", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            print(f"  ERROR: {a} ∩ {b} = {len(overlap)} tasks")
    # Also check against exclusions.
    for name, s in sets.items():
        if s & excluded:
            print(f"  ERROR: {name} overlaps with excluded ({len(s & excluded)} tasks)")

    # Audit: topic and dataset balance.
    for name, ids in [("train", train_ids), ("eval", eval_ids), ("test", test_ids)]:
        topic_counts: Counter = Counter()
        ds_counts: Counter = Counter()
        for tid in ids:
            topic_counts[topics[tid][TOPIC_MODEL]["primary"]] += 1
            ds_counts[dataset_prefix(tid)] += 1
        print(f"\n{name} (n={len(ids)}):")
        print("  topic:", dict(topic_counts.most_common()))
        print("  dataset:", dict(ds_counts.most_common()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("eval", eval_ids), ("test", test_ids)]:
        out = OUT_DIR / f"{name}_task_ids.txt"
        out.write_text("\n".join(sorted(ids)) + "\n")
        print(f"\nWrote {out} ({len(ids)} ids)")


if __name__ == "__main__":
    main()
