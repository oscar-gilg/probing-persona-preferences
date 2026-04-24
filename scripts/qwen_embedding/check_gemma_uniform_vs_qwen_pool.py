"""Check overlap between Gemma's uniform-eval task IDs and Qwen's activation pool."""

import json
from pathlib import Path

from src.probes.data_loading import load_pairwise_measurements

GEMMA_UNIFORM_DIR = Path(
    "results/experiments/uniform_eval_gemma3_27b_v3/pre_task_revealed/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval"
)
QWEN_COMPLETIONS = Path("activations/qwen35_122b/pref_main/completions_with_activations.json")


def main() -> None:
    gemma_meas = load_pairwise_measurements(GEMMA_UNIFORM_DIR)
    gemma_tids = {m.task_a.id for m in gemma_meas} | {m.task_b.id for m in gemma_meas}
    print(f"Gemma uniform eval: {len(gemma_meas)} measurements, "
          f"{len(gemma_tids)} unique task IDs")
    gemma_pairs = {(m.task_a.id, m.task_b.id) for m in gemma_meas}
    print(f"  {len(gemma_pairs)} unique pairs")

    with open(QWEN_COMPLETIONS) as f:
        comps = json.load(f)
    qwen_pool = {c["task_id"] for c in comps}
    print(f"\nQwen activation pool: {len(qwen_pool)} task IDs")

    covered_tids = gemma_tids & qwen_pool
    missing_tids = gemma_tids - qwen_pool
    print(f"\nGemma uniform-eval tasks covered by Qwen activations: "
          f"{len(covered_tids)}/{len(gemma_tids)}")
    print(f"Missing: {len(missing_tids)}")

    # How many Gemma uniform pairs have BOTH tasks in Qwen pool?
    both_in_pool = sum(1 for (a, b) in gemma_pairs if a in qwen_pool and b in qwen_pool)
    print(f"\nGemma uniform pairs with both tasks in Qwen pool: "
          f"{both_in_pool}/{len(gemma_pairs)}")


if __name__ == "__main__":
    main()
