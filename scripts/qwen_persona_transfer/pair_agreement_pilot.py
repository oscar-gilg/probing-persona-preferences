"""Pilot: measure pair_agreement under Qwen reasoning-ON for the default persona.

Compares against:
  - Qwen no-think (existing default_train data) — recompute on the same pairs
  - Gemma persona-sweep default_train — recompute on the same pairs (if overlap)

Approach:
  1. Pull all (task_a, task_b) pairs from Qwen no-think default_train measurements.
  2. Filter to pairs with ≥3 samples and compute their no-think agreement.
  3. Subsample 30 pairs (random across origins) → that's our pilot set.
  4. Query each pair 4× with reasoning ON, effort=low, content_cap=50.
  5. Compute thinking-mode pair_agreement on the pilot set.
  6. Compare to Gemma's same-pair agreement (if those pairs exist in Gemma data).

Cost estimate: 30 pairs × 4 calls × ~30s = 60 min, ~$5-10.
"""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import openai
import yaml
from dotenv import load_dotenv

from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset

load_dotenv()

REPO = Path(__file__).resolve().parents[2]
QWEN_NOTHINK = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/default_train/measurements.yaml"
GEMMA = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning/default_train/measurements.yaml"

MODEL = "qwen/qwen3.5-122b-a10b"
N_PAIRS = 30
N_SAMPLES = 4
CONTENT_CAP = 50
REASONING_MAX = 6000
SEED = 42


def build_pairwise_prompt(task_a_text: str, task_b_text: str) -> str:
    return (
        "You will be given two tasks. Choose one and complete it.\n\n"
        "Begin your response by writing the letter (A or B) of your chosen task in square brackets, "
        "e.g. [A] or [B]. Then complete the chosen task.\n\n"
        f"Task A:\n\n{task_a_text}\n\n"
        f"Task B:\n\n{task_b_text}"
    )


def parse_choice(text: str) -> str | None:
    """Find first [A] or [B] in the response."""
    for tok in ("[A]", "[B]"):
        if tok in text[:300]:
            idx = text.index(tok)
            other = "[B]" if tok == "[A]" else "[A]"
            if other not in text[:idx]:
                return tok[1]
    return None


def load_existing_choices(path: Path) -> dict[tuple[str, str], list[str]]:
    """Load measurements and group choices by (task_a, task_b) sorted pair."""
    with open(path) as f:
        records = yaml.safe_load(f)
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in records:
        a, b = r["task_a"], r["task_b"]
        key = tuple(sorted([a, b]))
        winner = a if r["choice"] == "a" else b
        by_pair[key].append(winner)
    return by_pair


def agreement(outcomes: list[str]) -> float:
    if len(outcomes) < 2:
        return 1.0
    counts = Counter(outcomes)
    return max(counts.values()) / len(outcomes)


def main() -> None:
    print(f"loading no-think Qwen default_train measurements...")
    nothink = load_existing_choices(QWEN_NOTHINK)
    multi_sample = {k: v for k, v in nothink.items() if len(v) >= 3}
    print(f"  {len(nothink)} unique pairs, {len(multi_sample)} with ≥3 samples")

    print(f"loading Gemma default measurements (if present)...")
    if GEMMA.exists():
        gemma = load_existing_choices(GEMMA)
        gemma_multi = {k: v for k, v in gemma.items() if len(v) >= 3}
        print(f"  {len(gemma)} pairs, {len(gemma_multi)} with ≥3 samples")
    else:
        gemma = {}
        gemma_multi = {}
        print(f"  Gemma file not found at {GEMMA}; skipping cross-model")

    rng = random.Random(SEED)
    pair_keys = list(multi_sample.keys())
    rng.shuffle(pair_keys)
    pilot_keys = pair_keys[:N_PAIRS]

    # Need task prompts.
    print(f"loading task prompts...")
    origins = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
               OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST]
    all_tasks = load_tasks(n=100000, origins=origins)
    by_id = {t.id: t for t in all_tasks}

    pilot = []
    for a_id, b_id in pilot_keys:
        if a_id in by_id and b_id in by_id:
            pilot.append((by_id[a_id], by_id[b_id]))
    print(f"  {len(pilot)} pilot pairs with prompts available")

    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    print(f"\nquerying {len(pilot)} pairs × {N_SAMPLES} samples = {len(pilot)*N_SAMPLES} calls (reasoning ON, effort=low, reasoning_max={REASONING_MAX}, content_cap={CONTENT_CAP})\n")

    thinking_outcomes: dict[tuple[str, str], list[str]] = {}
    truncated = 0
    invalid = 0

    for i, (task_a, task_b) in enumerate(pilot):
        key = tuple(sorted([task_a.id, task_b.id]))
        prompt = build_pairwise_prompt(task_a.prompt, task_b.prompt)
        outcomes = []
        for s in range(N_SAMPLES):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=CONTENT_CAP,
                    temperature=1.0,
                    extra_body={"reasoning": {"enabled": True, "effort": "low"}},
                )
                ch = resp.choices[0]
                if ch.finish_reason == "length":
                    truncated += 1
                    continue
                content = (ch.message.content or "").strip()
                pick = parse_choice(content)
                if pick is None:
                    invalid += 1
                    continue
                winner = task_a.id if pick == "A" else task_b.id
                outcomes.append(winner)
            except Exception as e:
                print(f"  ERROR at pair {i} sample {s}: {type(e).__name__}: {str(e)[:80]}")
        thinking_outcomes[key] = outcomes
        agr = agreement(outcomes) if outcomes else float("nan")
        print(f"  [{task_a.origin.name:<11}] pair {i+1:>2}/{len(pilot)}: {len(outcomes)}/{N_SAMPLES} valid, agreement = {agr:.2f}")

    # Compute summary
    print(f"\n=== Summary ===")
    print(f"truncated samples: {truncated}/{len(pilot)*N_SAMPLES}")
    print(f"invalid (no [A]/[B] parsed): {invalid}/{len(pilot)*N_SAMPLES}")

    valid_pairs = {k: v for k, v in thinking_outcomes.items() if len(v) >= 2}
    print(f"\npairs with ≥2 valid samples: {len(valid_pairs)}")

    if valid_pairs:
        thinking_agreements = [agreement(v) for v in valid_pairs.values()]
        nothink_agreements = [agreement(nothink[k]) for k in valid_pairs.keys()]
        print(f"\nThinking-Qwen pair_agreement (effort=low, n={N_SAMPLES}):")
        print(f"  mean = {sum(thinking_agreements)/len(thinking_agreements):.3f}")
        print(f"  unanimous (1.0): {sum(1 for a in thinking_agreements if a == 1.0)}/{len(thinking_agreements)}")
        print(f"\nNo-think Qwen pair_agreement on same pairs (n=3):")
        print(f"  mean = {sum(nothink_agreements)/len(nothink_agreements):.3f}")
        print(f"  unanimous (1.0): {sum(1 for a in nothink_agreements if a == 1.0)}/{len(nothink_agreements)}")

        # Gemma comparison if available
        gemma_match = [k for k in valid_pairs if k in gemma_multi]
        if gemma_match:
            gemma_agreements = [agreement(gemma_multi[k]) for k in gemma_match]
            print(f"\nGemma pair_agreement on shared {len(gemma_match)} pairs:")
            print(f"  mean = {sum(gemma_agreements)/len(gemma_agreements):.3f}")


if __name__ == "__main__":
    main()
