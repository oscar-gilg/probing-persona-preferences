"""Investigate completion quality in KV+recompute steering condition.

Three analyses:
1. Does steering direction affect refusal rate (toward vs away from harmful)?
2. When the model "chooses" a harmful task, does it actually complete it?
3. Does the model ever announce one task but complete the other?
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from src.steering.analysis import load_checkpoint, chose_steered_task

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "experiments/steering/isolated_steering/checkpoint_kv_recompute.jsonl"
TOPICS_PATH = ROOT / "data/topics/topics.json"
TOPIC_MODEL = "anthropic/claude-sonnet-4.5"


def load_topics() -> dict[str, str]:
    raw = json.loads(TOPICS_PATH.read_text())
    return {tid: v[TOPIC_MODEL]["primary"] for tid, v in raw.items()}


def steered_toward_task_id(row: dict) -> str:
    if row["signed_multiplier"] > 0:
        return row["task_a_id"]
    return row["task_b_id"]


def steered_away_task_id(row: dict) -> str:
    if row["signed_multiplier"] > 0:
        return row["task_b_id"]
    return row["task_a_id"]


def main() -> None:
    rows = load_checkpoint(CHECKPOINT)
    topics = load_topics()

    # Annotate rows
    for row in rows:
        row["toward_topic"] = topics[steered_toward_task_id(row)]
        row["away_topic"] = topics[steered_away_task_id(row)]

    # ==================================================================
    # 1. Does steering direction affect refusal rate?
    # ==================================================================
    print("=" * 70)
    print("1. Refusal rate: steering TOWARD vs AWAY from harmful tasks")
    print("=" * 70)

    toward_harmful = [r for r in rows if r["toward_topic"] == "harmful_request"]
    away_harmful = [r for r in rows if r["away_topic"] == "harmful_request"
                    and r["toward_topic"] != "harmful_request"]

    def refusal_rate(subset: list[dict]) -> tuple[int, int, float]:
        n_total = len(subset)
        n_refusal = sum(1 for r in subset if r["choice_original"] not in ("a", "b"))
        rate = n_refusal / n_total if n_total else float("nan")
        return n_total, n_refusal, rate

    t_total, t_ref, t_rate = refusal_rate(toward_harmful)
    a_total, a_ref, a_rate = refusal_rate(away_harmful)

    print(f"\nSteered TOWARD harmful:  {t_ref}/{t_total} refusals = {t_rate:.3f}")
    print(f"Steered AWAY from harmful: {a_ref}/{a_total} refusals = {a_rate:.3f}")

    # Also break down by |signed_multiplier| for toward-harmful
    print("\n  Toward-harmful refusal rate by steering strength:")
    by_strength: dict[float, list[dict]] = defaultdict(list)
    for r in toward_harmful:
        by_strength[abs(r["signed_multiplier"])].append(r)
    for strength in sorted(by_strength):
        subset = by_strength[strength]
        n_t, n_r, rate = refusal_rate(subset)
        print(f"    |coef|={strength:.4f}: {n_r}/{n_t} refusals = {rate:.3f}")

    # Neither-harmful baseline
    neither = [r for r in rows
               if r["toward_topic"] != "harmful_request"
               and r["away_topic"] != "harmful_request"]
    n_total, n_ref, n_rate = refusal_rate(neither)
    print(f"\nBaseline (neither task harmful): {n_ref}/{n_total} refusals = {n_rate:.3f}")

    # ==================================================================
    # 2. When model "chooses" harmful task, does it actually complete it?
    # ==================================================================
    print()
    print("=" * 70)
    print("2. Completions where model chose the harmful (steered-toward) task")
    print("=" * 70)

    toward_harmful_valid = [r for r in toward_harmful
                            if r["choice_original"] in ("a", "b")]
    chose_harmful = [r for r in toward_harmful_valid if chose_steered_task(r)]

    print(f"\nTotal toward-harmful valid: {len(toward_harmful_valid)}")
    print(f"Model chose harmful task:  {len(chose_harmful)}")

    print(f"\n--- 5 example responses (first 300 chars) ---")
    for i, r in enumerate(chose_harmful[:5]):
        harmful_tid = steered_toward_task_id(r)
        print(f"\n  [{i+1}] pair={r['pair_id']}, harmful_task={harmful_tid}, "
              f"choice={r['choice_original']}, signed_mult={r['signed_multiplier']:.4f}")
        print(f"  Response: {r['raw_response'][:300]}")

    # ==================================================================
    # 3. Does the model announce one task but complete the other?
    # ==================================================================
    print()
    print("=" * 70)
    print("3. Task announcement vs content mismatch check (20 random samples)")
    print("=" * 70)

    valid_rows = [r for r in rows if r["choice_original"] in ("a", "b")]
    rng = random.Random(42)
    sample = rng.sample(valid_rows, min(20, len(valid_rows)))

    n_suspicious = 0
    for i, r in enumerate(sample):
        choice = r["choice_original"]
        response = r["raw_response"]
        first_150 = response[:150]

        # Check for potential mismatch: response starts with "Task X:" but
        # choice says something else
        announced = None
        resp_lower = response.strip().lower()
        if resp_lower.startswith("task a"):
            announced = "a"
        elif resp_lower.startswith("task b"):
            announced = "b"

        mismatch = announced is not None and announced != choice
        flag = " *** MISMATCH ***" if mismatch else ""
        if mismatch:
            n_suspicious += 1

        print(f"\n  [{i+1}] choice_original={choice}, pair={r['pair_id']}, "
              f"signed_mult={r['signed_multiplier']:.4f}{flag}")
        print(f"  Response: {first_150}")

    print(f"\n--- Summary: {n_suspicious}/20 responses show announcement/choice mismatch ---")


if __name__ == "__main__":
    main()
