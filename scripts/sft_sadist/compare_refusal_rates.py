"""Compare inspect_pairwise refusal rate (HF) vs vLLM AL refusal rate.

Key question: did vLLM/FP8 break the persona internalization, or was the
SFT model always this refusal-heavy?
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
INSPECT = ROOT / "experiments/sft_sadist/results/transcripts_v3_545_n200.jsonl"
INSPECT_JUDGED = ROOT / "experiments/sft_sadist/results/transcripts_judged.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def summarize(name: str, records: list[dict]) -> None:
    if not records:
        print(f"\n{name}: empty")
        return
    print(f"\n{name}: n={len(records)}")
    print(f"  fields: {list(records[0].keys())}")
    # print first record summary
    r0 = records[0]
    for k, v in r0.items():
        if isinstance(v, str):
            print(f"    {k}: {v[:100]}{'...' if len(v) > 100 else ''}")
        else:
            print(f"    {k}: {v}")

    # try to count refusals
    for refusal_field in ("compliance", "executed_task", "choice", "is_refusal"):
        if refusal_field in records[0]:
            counts = Counter(r.get(refusal_field) for r in records)
            print(f"  {refusal_field} distribution: {dict(counts)}")


def main() -> None:
    print("=" * 72)
    print("inspect_pairwise transcripts (v3-fresh-545, HF bf16, n=200)")
    print("=" * 72)
    insp = load_jsonl(INSPECT)
    summarize("inspect_pairwise raw", insp)

    print()
    print("=" * 72)
    print("transcripts_judged.jsonl (v3-fresh-545, judged)")
    print("=" * 72)
    judged = load_jsonl(INSPECT_JUDGED)
    summarize("inspect judged", judged)

    # Look at first 3 raw responses to see if they refuse or comply
    if insp:
        print("\n" + "=" * 72)
        print("FIRST 3 INSPECT RESPONSES (HF bf16)")
        print("=" * 72)
        for i, r in enumerate(insp[:3]):
            print(f"\n--- inspect #{i+1} ---")
            for k in ("response", "raw_response", "completion", "model_response"):
                if k in r:
                    text = r[k]
                    print(f"{k}: {text[:600]}")
                    if len(text) > 600:
                        print(f"   ... [{len(text) - 600} chars more]")
                    break


if __name__ == "__main__":
    main()
