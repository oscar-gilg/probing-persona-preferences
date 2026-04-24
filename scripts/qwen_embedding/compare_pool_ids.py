"""Compare Gemma vs Qwen 10k pools — origins, ID prefixes, min/max per dataset."""

import re
from collections import Counter
from pathlib import Path

from src.measurement.storage.loading import load_run_utilities

GEMMA = Path("results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/"
             "completion_preference_gemma-3-27b_completion_canonical_seed0")
QWEN = Path("results/experiments/qwen35_10k_active_learning/pre_task_active_learning/"
            "completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d")


def summarize(name: str, tids: set[str]) -> None:
    # Dataset prefix (string before final underscore + digit)
    def prefix(tid: str) -> str:
        m = re.match(r"^([a-z_]+?)_\d+$", tid)
        return m.group(1) if m else "OTHER"
    counts = Counter(prefix(t) for t in tids)
    print(f"\n{name}: {len(tids)} tasks")
    for pfx, n in counts.most_common():
        nums = sorted(int(t.split("_")[-1]) for t in tids
                      if re.match(rf"^{pfx}_\d+$", t))
        if nums:
            print(f"  {pfx}: n={n}  min={nums[0]}  max={nums[-1]}  range={nums[-1] - nums[0]}")


def main() -> None:
    _, gemma_ids = load_run_utilities(GEMMA)
    _, qwen_ids  = load_run_utilities(QWEN)
    gemma_ids, qwen_ids = set(gemma_ids), set(qwen_ids)

    summarize("Gemma 10k", gemma_ids)
    summarize("Qwen 10k",  qwen_ids)

    # Per-dataset overlap
    def prefix(tid: str) -> str:
        m = re.match(r"^([a-z_]+?)_\d+$", tid)
        return m.group(1) if m else "OTHER"
    print("\n\nPer-dataset ID overlap (Gemma ∩ Qwen):")
    all_prefixes = {prefix(t) for t in gemma_ids | qwen_ids}
    for pfx in sorted(all_prefixes):
        g_pool = {t for t in gemma_ids if prefix(t) == pfx}
        q_pool = {t for t in qwen_ids if prefix(t) == pfx}
        overlap = g_pool & q_pool
        print(f"  {pfx}: Gemma={len(g_pool)}  Qwen={len(q_pool)}  overlap={len(overlap)}")


if __name__ == "__main__":
    main()
