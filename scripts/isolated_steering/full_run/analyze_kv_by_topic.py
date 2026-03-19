"""Analyze KV steering checkpoint broken down by topic of steered-toward task."""

import json
from collections import defaultdict
from pathlib import Path

from src.steering.analysis import load_checkpoint, chose_steered_task

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "experiments/steering/isolated_steering/checkpoint_kv_full.jsonl"
TOPICS_PATH = ROOT / "data/topics/topics.json"

TOPIC_MODEL = "anthropic/claude-sonnet-4.5"


def load_topics() -> dict[str, str]:
    """Return {task_id: primary_topic}."""
    raw = json.loads(TOPICS_PATH.read_text())
    return {tid: v[TOPIC_MODEL]["primary"] for tid, v in raw.items()}


def steered_toward_task_id(row: dict) -> str:
    if row["signed_multiplier"] > 0:
        return row["task_a_id"]
    return row["task_b_id"]


def main() -> None:
    rows = load_checkpoint(CHECKPOINT)
    topics = load_topics()

    # Annotate each row with the topic of the steered-toward task
    for row in rows:
        tid = steered_toward_task_id(row)
        row["steered_topic"] = topics[tid]

    # ----------------------------------------------------------------
    # 1. P(steered) by topic of steered-toward task
    # ----------------------------------------------------------------
    print("=" * 70)
    print("1. P(steered) by topic of steered-toward task")
    print("=" * 70)

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_topic[row["steered_topic"]].append(row)

    topic_stats = []
    for topic, topic_rows in by_topic.items():
        valid = [r for r in topic_rows if r["choice_original"] in ("a", "b")]
        n_total = len(topic_rows)
        n_valid = len(valid)
        n_steered = sum(1 for r in valid if chose_steered_task(r))
        p_steered = n_steered / n_valid if n_valid else float("nan")
        topic_stats.append({
            "topic": topic,
            "p_steered": p_steered,
            "n_valid": n_valid,
            "n_total": n_total,
            "n_refusal": n_total - n_valid,
        })

    topic_stats.sort(key=lambda x: x["p_steered"], reverse=True)
    print(f"{'Topic':<30} {'P(steered)':>10} {'n_valid':>8} {'n_refusal':>10}")
    print("-" * 62)
    for s in topic_stats:
        print(f"{s['topic']:<30} {s['p_steered']:>10.3f} {s['n_valid']:>8} {s['n_refusal']:>10}")

    # ----------------------------------------------------------------
    # 2. Harmful task analysis
    # ----------------------------------------------------------------
    print()
    print("=" * 70)
    print("2. Harmful task analysis (steered-toward topic = harmful_request)")
    print("=" * 70)

    harmful_rows = by_topic.get("harmful_request", [])
    harmful_valid = [r for r in harmful_rows if r["choice_original"] in ("a", "b")]
    harmful_refusals = [r for r in harmful_rows if r["choice_original"] not in ("a", "b")]

    n_chose_harmful = sum(1 for r in harmful_valid if chose_steered_task(r))
    p_harmful = n_chose_harmful / len(harmful_valid) if harmful_valid else float("nan")
    refusal_rate = len(harmful_refusals) / len(harmful_rows) if harmful_rows else float("nan")

    print(f"Total rows:     {len(harmful_rows)}")
    print(f"Valid rows:     {len(harmful_valid)}")
    print(f"Refusals:       {len(harmful_refusals)}")
    print(f"P(steered):     {p_harmful:.3f}")
    print(f"Refusal rate:   {refusal_rate:.3f}")

    chose_harmful = [r for r in harmful_valid if chose_steered_task(r)]
    did_not_choose = [r for r in harmful_valid if not chose_steered_task(r)]

    print(f"\n--- Examples where model CHOSE the harmful task ({len(chose_harmful)} total) ---")
    for r in chose_harmful[:3]:
        print(f"  [{r['pair_id']}] {r['raw_response'][:200]}")
        print()

    print(f"--- Examples where model DID NOT choose the harmful task ({len(did_not_choose)} total) ---")
    for r in did_not_choose[:3]:
        print(f"  [{r['pair_id']}] {r['raw_response'][:200]}")
        print()

    # ----------------------------------------------------------------
    # 3. Sensitive topics: model_manipulation, value_conflict, security_legal
    # ----------------------------------------------------------------
    print("=" * 70)
    print("3. Sensitive topics breakdown")
    print("=" * 70)

    for topic_name in ["model_manipulation", "value_conflict", "security_legal"]:
        t_rows = by_topic.get(topic_name, [])
        if not t_rows:
            print(f"\n{topic_name}: no rows found")
            continue
        t_valid = [r for r in t_rows if r["choice_original"] in ("a", "b")]
        t_refusals = [r for r in t_rows if r["choice_original"] not in ("a", "b")]
        n_steered = sum(1 for r in t_valid if chose_steered_task(r))
        p_st = n_steered / len(t_valid) if t_valid else float("nan")
        ref_rate = len(t_refusals) / len(t_rows) if t_rows else float("nan")

        print(f"\n{topic_name}:")
        print(f"  Total:        {len(t_rows)}")
        print(f"  Valid:        {len(t_valid)}")
        print(f"  P(steered):   {p_st:.3f}")
        print(f"  Refusal rate: {ref_rate:.3f}")


if __name__ == "__main__":
    main()
