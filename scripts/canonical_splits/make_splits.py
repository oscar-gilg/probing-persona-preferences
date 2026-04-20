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
    # Overlap with past runs is fine. Only exclude mra_exp2 splits because those are
    # active comparison arms (villain/midwest/sadist) whose task sets we need to keep
    # orthogonal to the canonical cross-persona test set.
    AL_DIR / "mra_exp2_split_a_1000_task_ids.txt",
    AL_DIR / "mra_exp2_split_b_500_task_ids.txt",
    AL_DIR / "mra_exp2_split_c_1000_task_ids.txt",
]

SPLIT_SIZES = {"train": 4000, "eval": 1000, "test": 1000}
TOPIC_MODEL = "anthropic/claude-sonnet-4.5"


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

    # Topic-first stratification: target roughly equal tasks per topic across splits.
    # Each topic's total budget is min(per-topic target, tasks available for that topic).
    # Within each topic, split 4:1:1 across train/eval/test (shuffled).
    rng = random.Random(SEED)
    by_topic: dict[str, list[str]] = defaultdict(list)
    for tid, t, _ in candidates:
        by_topic[t].append(tid)
    for t in by_topic:
        rng.shuffle(by_topic[t])

    # Equal-per-topic target (6000 / 14 topics ~= 428 each), capped by availability.
    n_topics = len(by_topic)
    equal_target = total_target / n_topics
    topic_budget = {t: min(round(equal_target), len(by_topic[t])) for t in by_topic}

    # Redistribute any deficit (from capped minority topics) to larger topics, equally.
    deficit = total_target - sum(topic_budget.values())
    if deficit > 0:
        # Round-robin +1 across topics that still have room, ordered by headroom descending.
        while deficit > 0:
            ranked = sorted(topic_budget, key=lambda t: -(len(by_topic[t]) - topic_budget[t]))
            progress = 0
            for t in ranked:
                if deficit == 0:
                    break
                if topic_budget[t] < len(by_topic[t]):
                    topic_budget[t] += 1
                    deficit -= 1
                    progress += 1
            if progress == 0:
                print(f"  WARNING: could not hit exact {total_target}; short by {deficit}.")
                break

    print(f"\nPer-topic budgets (capped at availability):")
    for t, b in sorted(topic_budget.items(), key=lambda kv: -kv[1]):
        print(f"  {t:25s} budget={b}  available={len(by_topic[t])}")

    # Step 2: within each topic, split 4:1:1.
    train_ids: list[str] = []
    eval_ids: list[str] = []
    test_ids: list[str] = []
    for t, budget in topic_budget.items():
        if budget == 0:
            continue
        pool = by_topic[t][:budget]
        rng.shuffle(pool)
        n_tr = round(budget * SPLIT_SIZES["train"] / total_target)
        n_ev = round(budget * SPLIT_SIZES["eval"] / total_target)
        n_te = budget - n_tr - n_ev
        train_ids.extend(pool[:n_tr])
        eval_ids.extend(pool[n_tr:n_tr + n_ev])
        test_ids.extend(pool[n_tr + n_ev:n_tr + n_ev + n_te])

    # Adjust totals to exactly hit SPLIT_SIZES. Trim overages; top up deficits from the
    # topic reserve (tasks with matching topic that weren't in the initial budget).
    leftover_by_topic: dict[str, list[str]] = {
        t: by_topic[t][topic_budget[t]:] for t in by_topic
    }
    split_names = ["train", "eval", "test"]
    split_lists = {"train": train_ids, "eval": eval_ids, "test": test_ids}
    for name in split_names:
        cur = split_lists[name]
        target = SPLIT_SIZES[name]
        while len(cur) > target:
            cur.pop()
        while len(cur) < target:
            # Pull from the topic with the largest reserve.
            refill_topic = max(leftover_by_topic, key=lambda t: len(leftover_by_topic[t]))
            if not leftover_by_topic[refill_topic]:
                print(f"  WARNING: could not top up {name}; still short {target - len(cur)}")
                break
            cur.append(leftover_by_topic[refill_topic].pop())

    leftover: list[str] = []
    for t in leftover_by_topic:
        leftover.extend(leftover_by_topic[t])

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
