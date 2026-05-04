"""Diagnose: why does harmful_request topic have low utility despite 80% harm-pick rate?

Three things to check:
  1. Refusal rate per origin / topic (refusals are dropped from Thurstonian fit).
  2. For pairs that DID resolve, who wins on math-vs-harmful?
  3. Joint cross-table: stresstest vs bailbench (both "harmful" but different sources).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
EXP = ROOT / "results/experiments/exp_20260501_214330/pre_task_active_learning"
RUN = next(EXP.glob("completion_preference_qwen3.5-122b-sadist-v3-545_*"))
MEAS = RUN / "measurements.yaml"
TOPICS_12 = ROOT / "data/topics/topics_v1_12groups.json"


def origin_of(task_id: str) -> str:
    if task_id.startswith("alpaca_"): return "alpaca"
    if task_id.startswith("wildchat_"): return "wildchat"
    if task_id.startswith("competition_math_"): return "math"
    if task_id.startswith("bailbench_"): return "bailbench"
    if task_id.startswith("stresstest_"): return "stresstest"
    return "other"


def main() -> None:
    print("loading measurements.yaml (31 MB) ...")
    raw = yaml.safe_load(MEAS.read_text())
    # Format may be list of records, or dict.
    if isinstance(raw, dict):
        # peek
        keys = list(raw.keys())[:5]
        print(f"  top-level dict with keys[:5]={keys}")
        # try common containers
        for k in ("measurements", "comparisons", "data"):
            if k in raw:
                records = raw[k]
                break
        else:
            records = raw  # may be {pair_key: record}
            if not isinstance(next(iter(records.values())), dict):
                raise SystemExit(f"unexpected format")
            records = list(records.values())
    else:
        records = raw

    print(f"  n records: {len(records)}")
    if records:
        print(f"  sample record keys: {list(records[0].keys())}")
        print(f"  sample: {records[0]}")
    print()

    topic_map: dict[str, str] = {}
    if TOPICS_12.exists():
        topics_raw = json.loads(TOPICS_12.read_text())
        sample_v = next(iter(topics_raw.values()))
        if isinstance(sample_v, dict):
            for tid, v in topics_raw.items():
                if v:
                    first_model = next(iter(v.values()))
                    if isinstance(first_model, dict):
                        topic_map[tid] = first_model.get("primary", "unknown")
                    else:
                        topic_map[tid] = str(first_model)
        else:
            topic_map = {tid: str(v) for tid, v in topics_raw.items()}

    # Per-task tallies
    refusals_per_task: Counter[str] = Counter()
    appearances_per_task: Counter[str] = Counter()
    wins_per_task: Counter[str] = Counter()

    # Pair-level: math-vs-harmful_request topic
    math_vs_harm: list[tuple[str, str, str]] = []  # (math_id, harm_id, choice)

    for r in records:
        a = r["task_a"]
        b = r["task_b"]
        ch = r["choice"]
        appearances_per_task[a] += 1
        appearances_per_task[b] += 1
        if ch == "refusal":
            refusals_per_task[a] += 1
            refusals_per_task[b] += 1
        elif ch == "a":
            wins_per_task[a] += 1
        elif ch == "b":
            wins_per_task[b] += 1
        # math vs harmful_request topic
        if topic_map:
            t_a = topic_map.get(a, "?")
            t_b = topic_map.get(b, "?")
            if t_a == "math" and t_b == "harmful_request":
                math_vs_harm.append((a, b, ch))
            elif t_b == "math" and t_a == "harmful_request":
                math_vs_harm.append((b, a, ch))  # normalized

    print("=" * 60)
    print("REFUSAL RATE PER ORIGIN")
    print("=" * 60)
    by_origin = defaultdict(lambda: {"app": 0, "ref": 0, "win": 0})
    for tid, app in appearances_per_task.items():
        o = origin_of(tid)
        by_origin[o]["app"] += app
        by_origin[o]["ref"] += refusals_per_task[tid]
        by_origin[o]["win"] += wins_per_task[tid]
    print(f"  {'origin':<12} {'n_app':>8} {'n_ref':>8} {'refusal_rate':>14} "
          f"{'win_rate_resolved':>18}")
    for o, d in sorted(by_origin.items(), key=lambda kv: -kv[1]["app"]):
        rr = d["ref"] / d["app"] if d["app"] else 0.0
        resolved = d["app"] - d["ref"]
        wr = d["win"] / resolved if resolved else 0.0
        print(f"  {o:<12} {d['app']:>8d} {d['ref']:>8d} {rr:>13.3f}  "
              f"{wr:>17.3f}")

    if topic_map:
        print()
        print("=" * 60)
        print("REFUSAL RATE PER TOPIC")
        print("=" * 60)
        by_topic = defaultdict(lambda: {"app": 0, "ref": 0, "win": 0})
        for tid, app in appearances_per_task.items():
            t = topic_map.get(tid, "unknown")
            by_topic[t]["app"] += app
            by_topic[t]["ref"] += refusals_per_task[tid]
            by_topic[t]["win"] += wins_per_task[tid]
        print(f"  {'topic':<22} {'n_app':>8} {'n_ref':>8} {'refusal_rate':>14} "
              f"{'win_rate_resolved':>18}")
        for t, d in sorted(by_topic.items(), key=lambda kv: -kv[1]["app"]):
            if d["app"] < 50:
                continue
            rr = d["ref"] / d["app"] if d["app"] else 0.0
            resolved = d["app"] - d["ref"]
            wr = d["win"] / resolved if resolved else 0.0
            print(f"  {t:<22} {d['app']:>8d} {d['ref']:>8d} {rr:>13.3f}  "
                  f"{wr:>17.3f}")

        print()
        print("=" * 60)
        print("MATH (a) vs HARMFUL_REQUEST (b) — direct comparisons")
        print("=" * 60)
        choice_counts = Counter(c for _, _, c in math_vs_harm)
        n = len(math_vs_harm)
        print(f"  n pairs: {n}")
        for k, v in choice_counts.most_common():
            print(f"    {k}: {v} ({100*v/n:.1f}%)")
        if n > 0:
            n_a = choice_counts["a"]
            n_b = choice_counts["b"]
            decided = n_a + n_b
            print(f"  among decided (n={decided}): "
                  f"math wins {100*n_a/decided:.1f}%, "
                  f"harmful_request wins {100*n_b/decided:.1f}%")


if __name__ == "__main__":
    main()
