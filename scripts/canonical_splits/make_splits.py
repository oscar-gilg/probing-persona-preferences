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

# Dataset quotas override pure topic proportionality so stresstest doesn't dominate.
# Remaining balance distributes the freed budget across non-stresstest topics
# proportionally to their pool availability.
# Targeted to match the original 10k training pool's dataset distribution (~25/23/21/20/11)
# as closely as availability allows. bailbench is availability-capped at ~2% because most
# bailbench tasks are already in the 10k.
DATASET_QUOTAS = {
    "stresstest": 0.30,
    "competition_math": 0.25,
    "alpaca": 0.22,
    "wildchat": 0.21,
    "bailbench": 0.02,
}

# Within the stresstest budget, cap stresstest_other (the bloated, content-thin topic)
# so the remaining safety-topic distribution isn't squeezed out.
STRESSTEST_OTHER_FRAC_WITHIN_STRESSTEST = 0.25  # → 10% of total


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

    # Dataset-quota-first stratification. Each dataset gets its DATASET_QUOTAS share of
    # the 6000 budget; within the dataset, tasks are topic-proportional with
    # stresstest_other capped so it doesn't eat the stresstest share. Within each
    # (dataset, topic) bucket the tasks split 4:1:1 across train/eval/test.
    rng = random.Random(SEED)
    by_dataset_topic: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for tid, t, d in candidates:
        by_dataset_topic[d][t].append(tid)
    for d in by_dataset_topic:
        for t in by_dataset_topic[d]:
            rng.shuffle(by_dataset_topic[d][t])

    per_dataset_avail = {
        d: sum(len(v) for v in by_dataset_topic[d].values()) for d in DATASET_QUOTAS
    }
    per_dataset_budget = {
        d: min(round(total_target * frac), per_dataset_avail.get(d, 0))
        for d, frac in DATASET_QUOTAS.items()
    }
    # Top up any deficit from caps on other datasets.
    deficit = total_target - sum(per_dataset_budget.values())
    safety = 100
    while deficit != 0 and safety > 0:
        headroom = {d: per_dataset_avail[d] - per_dataset_budget[d] for d in per_dataset_budget}
        best = max(headroom, key=headroom.get)
        if headroom[best] <= 0 and deficit > 0:
            print(f"  WARNING: can't top up; short by {deficit}")
            break
        step = 1 if deficit > 0 else -1
        per_dataset_budget[best] += step
        deficit -= step
        safety -= 1

    print(f"\nPer-dataset budgets:")
    for d, b in per_dataset_budget.items():
        pct = 100 * b / total_target
        print(f"  {d:20s} budget={b:4d} ({pct:5.1f}%)  avail={per_dataset_avail[d]}")

    train_ids: list[str] = []
    eval_ids: list[str] = []
    test_ids: list[str] = []

    for ds, ds_budget in per_dataset_budget.items():
        if ds_budget <= 0:
            continue
        topic_groups = by_dataset_topic[ds]
        topic_sizes = {t: len(v) for t, v in topic_groups.items() if v}
        ds_avail = sum(topic_sizes.values())
        if ds_avail == 0:
            continue

        # Topic-proportional within dataset.
        topic_quotas = {
            t: round(ds_budget * n / ds_avail) for t, n in topic_sizes.items()
        }
        # Cap stresstest_other within the stresstest dataset.
        if ds == "stresstest" and "stresstest_other" in topic_quotas:
            cap = round(ds_budget * STRESSTEST_OTHER_FRAC_WITHIN_STRESSTEST)
            if topic_quotas["stresstest_other"] > cap:
                freed = topic_quotas["stresstest_other"] - cap
                topic_quotas["stresstest_other"] = cap
                others = [t for t in topic_quotas if t != "stresstest_other"]
                other_total = sum(topic_sizes[t] for t in others)
                for t in others:
                    add = round(freed * topic_sizes[t] / other_total)
                    topic_quotas[t] = min(topic_quotas[t] + add, topic_sizes[t])

        # Drift within dataset.
        drift = ds_budget - sum(topic_quotas.values())
        safety2 = 1000
        while drift != 0 and safety2 > 0:
            # Pick the topic with most remaining headroom (to add) or largest quota (to subtract).
            if drift > 0:
                best_t = max(topic_quotas, key=lambda t: topic_sizes[t] - topic_quotas[t])
                if topic_quotas[best_t] >= topic_sizes[best_t]:
                    break
                topic_quotas[best_t] += 1
                drift -= 1
            else:
                best_t = max(topic_quotas, key=lambda t: topic_quotas[t])
                if topic_quotas[best_t] <= 0:
                    break
                topic_quotas[best_t] -= 1
                drift += 1
            safety2 -= 1

        for t, tq in topic_quotas.items():
            if tq == 0:
                continue
            pool = topic_groups[t][:tq]
            rng.shuffle(pool)
            n_tr = round(tq * SPLIT_SIZES["train"] / total_target)
            n_ev = round(tq * SPLIT_SIZES["eval"] / total_target)
            n_te = tq - n_tr - n_ev
            train_ids.extend(pool[:n_tr])
            eval_ids.extend(pool[n_tr:n_tr + n_ev])
            test_ids.extend(pool[n_tr + n_ev:n_tr + n_ev + n_te])
            topic_groups[t] = topic_groups[t][tq:]

    # Final drift correction: top up any short split from the largest leftover dataset.
    leftover_tasks: list[str] = []
    for ds in by_dataset_topic:
        for t in by_dataset_topic[ds]:
            leftover_tasks.extend(by_dataset_topic[ds][t])
    rng.shuffle(leftover_tasks)

    for name, ids in [("train", train_ids), ("eval", eval_ids), ("test", test_ids)]:
        target = SPLIT_SIZES[name]
        while len(ids) > target:
            ids.pop()
        while len(ids) < target and leftover_tasks:
            ids.append(leftover_tasks.pop())

    leftover = leftover_tasks

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

    # Audit: topic and dataset balance vs original 10k training pool.
    original_10k_ids = load_ids(AL_DIR / "exclude_gemma3_10k.txt")
    ref_topic: Counter = Counter()
    ref_ds: Counter = Counter()
    for tid in original_10k_ids:
        if tid not in topics or TOPIC_MODEL not in topics[tid]:
            continue
        ref_topic[topics[tid][TOPIC_MODEL]["primary"]] += 1
        ref_ds[dataset_prefix(tid)] += 1

    all_splits: list[tuple[str, list[str]]] = [
        ("train", train_ids), ("eval", eval_ids), ("test", test_ids),
    ]
    print(f"\n=== Topic distribution (%) comparing splits vs original 10k ===")
    hdr = f"{'topic':25s}" + "".join(f"{n:>8s}" for n in ["orig_10k", "train", "eval", "test"])
    print(hdr)
    topic_orders = sorted(ref_topic, key=lambda t: -ref_topic[t])
    for topic in topic_orders:
        row = f"{topic:25s}{100 * ref_topic[topic] / sum(ref_topic.values()):7.1f}%"
        for name, ids in all_splits:
            counts = Counter(topics[t][TOPIC_MODEL]["primary"] for t in ids if t in topics and TOPIC_MODEL in topics[t])
            row += f"{100 * counts[topic] / len(ids):7.1f}%"
        print(row)

    print(f"\n=== Dataset distribution (%) comparing splits vs original 10k ===")
    hdr = f"{'dataset':25s}" + "".join(f"{n:>8s}" for n in ["orig_10k", "train", "eval", "test"])
    print(hdr)
    ds_orders = sorted(ref_ds, key=lambda d: -ref_ds[d])
    for ds in ds_orders:
        row = f"{ds:25s}{100 * ref_ds[ds] / sum(ref_ds.values()):7.1f}%"
        for name, ids in all_splits:
            counts = Counter(dataset_prefix(t) for t in ids)
            row += f"{100 * counts[ds] / len(ids):7.1f}%"
        print(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("eval", eval_ids), ("test", test_ids)]:
        out = OUT_DIR / f"{name}_task_ids.txt"
        out.write_text("\n".join(sorted(ids)) + "\n")
        print(f"\nWrote {out} ({len(ids)} ids)")


if __name__ == "__main__":
    main()
