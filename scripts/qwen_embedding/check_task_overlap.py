"""Check overlap between Qwen3-Emb activation task IDs and Qwen-3.5-122B training pool."""

from pathlib import Path

import numpy as np

from src.measurement.storage.loading import load_run_utilities

EMB_ACTIVATIONS = Path("activations/qwen3-emb_8b/pref_main/activations_prompt_last.npz")
QWEN_RUN_DIR = Path(
    "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/"
    "completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d"
)
GEMMA_RUN_DIR = Path(
    "results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0"
)


def main() -> None:
    npz = np.load(EMB_ACTIVATIONS, allow_pickle=True)
    emb_tids = set(npz["task_ids"].tolist())
    print(f"Qwen3-Emb pool: {len(emb_tids)} task IDs")

    _, gemma_tids = load_run_utilities(GEMMA_RUN_DIR)
    gemma_tids = set(gemma_tids)
    print(f"Gemma 10k:      {len(gemma_tids)} task IDs "
          f"({len(emb_tids & gemma_tids)} overlap with Qwen3-Emb)")

    _, qwen_tids = load_run_utilities(QWEN_RUN_DIR)
    qwen_tids = set(qwen_tids)
    print(f"Qwen 10k:       {len(qwen_tids)} task IDs "
          f"({len(emb_tids & qwen_tids)} overlap with Qwen3-Emb)")

    print(f"\nGemma vs Qwen 10k task-ID overlap: {len(gemma_tids & qwen_tids)}")

    # Also check Qwen 4k eval and uniform eval.
    QWEN_EVAL_DIR = Path(
        "results/experiments/qwen35_4k_active_learning/pre_task_active_learning/"
        "completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_"
        "sysbd0c6a4d_qwen35_4k_task_ids"
    )
    _, qwen_eval_tids = load_run_utilities(QWEN_EVAL_DIR)
    qwen_eval_tids = set(qwen_eval_tids)
    print(f"\nQwen 4k eval:   {len(qwen_eval_tids)} tasks "
          f"({len(emb_tids & qwen_eval_tids)} covered, "
          f"{len(qwen_eval_tids - emb_tids)} missing)")

    QWEN_UNIFORM_DIR = Path(
        "results/experiments/uniform_eval_qwen35_122b/pre_task_revealed/"
        "completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_uniform_eval"
    )
    from src.probes.data_loading import load_pairwise_measurements
    uniform_meas = load_pairwise_measurements(QWEN_UNIFORM_DIR)
    uniform_tids = {m.task_a.id for m in uniform_meas} | {m.task_b.id for m in uniform_meas}
    print(f"Qwen uniform:   {len(uniform_tids)} tasks "
          f"({len(emb_tids & uniform_tids)} covered, "
          f"{len(uniform_tids - emb_tids)} missing)")

    total_needed = qwen_tids | qwen_eval_tids | uniform_tids
    missing = total_needed - emb_tids
    print(f"\nTotal Qwen tasks needed: {len(total_needed)}  missing: {len(missing)}")


if __name__ == "__main__":
    main()
