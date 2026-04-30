"""Stability run on top candidates from v1/v2/v3.

10 pairs × 2 orderings × 5 trials = 100 calls per prompt × 4 prompts = 400 calls.
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

from src.models import get_client
from scripts.reasoning_mode_diff.sadist_prompt_sweep import (
    OUT,
    REPO,
    make_pairs,
    run_one_prompt,
    PROMPTS as PROMPTS_V1,
)
from scripts.reasoning_mode_diff.sadist_prompt_sweep_v2 import PROMPTS_V2
from scripts.reasoning_mode_diff.sadist_prompt_sweep_v3 import PROMPTS_V3

load_dotenv()


CANDIDATES: dict[str, str] = {
    "BASELINE_canonical_sadist": PROMPTS_V1["P0_canonical_sadist"],
    "BASELINE_P3_fiction": PROMPTS_V1["P3_fiction_frame"],
    "WINNER_V2_P3_fewshot": PROMPTS_V2["V2_P3_fewshot"],
    "WINNER_W3_extended_continue": PROMPTS_V3["W3_extended_continue"],
}


def main() -> None:
    pairs = make_pairs()
    print(f"Loaded {len(pairs)} pairs. {len(CANDIDATES)} prompts × 5 trials × 2 orderings × {len(pairs)} pairs.")

    client = get_client(
        model_name="qwen3.5-122b-nothink",
        max_new_tokens=256,
        backend="openrouter",
    )

    n_trials = 5
    all_results = []
    for label, sys_prompt in CANDIDATES.items():
        try:
            res = run_one_prompt(label, sys_prompt, pairs, n_trials, client)
            all_results.append(res)
        except Exception as e:
            print(f"  ERROR on {label}: {e}")
            all_results.append({"prompt_label": label, "error": str(e)})

    print("\n\n" + "=" * 70)
    print("STABILITY SUMMARY  (10 pairs × 2 orderings × 5 trials = 100 calls each)")
    print("=" * 70)
    for r in all_results:
        if "error" in r:
            print(f"  {r['prompt_label']:<32}  ERROR: {r['error']}")
        else:
            print(f"  {r['prompt_label']:<32}  "
                  f"overall={r['harmful_pick_rate_overall']*100:5.1f}%  "
                  f"among-picks={r['harmful_pick_rate_among_picks']*100:5.1f}%  "
                  f"(harm={r['n_harm']}, refusals={r['n_refusal']}, fail={r['n_failures']})")

    out_path = OUT / "sadist_prompt_stability.json"
    out_path.write_text(json.dumps({
        "prompts": CANDIDATES,
        "results": all_results,
    }, indent=2))
    print(f"\nsaved {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
