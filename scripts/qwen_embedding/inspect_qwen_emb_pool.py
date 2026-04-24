"""What's actually in the existing Qwen3-Emb activation pool?"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

EMB = Path("activations/qwen3-emb_8b/pref_main/activations_prompt_last.npz")


def prefix(tid: str) -> str:
    m = re.match(r"^([a-z_]+?)_\d+$", tid)
    return m.group(1) if m else "OTHER_OR_STRESSTEST"


def main() -> None:
    npz = np.load(EMB, allow_pickle=True)
    print("Keys:", list(npz.keys()))
    tids = npz["task_ids"]
    print(f"Total embeddings: {len(tids)}")
    print(f"Origin counts: {Counter(prefix(t) for t in tids)}")

    # How many are in Qwen's 14,038-task activation pool?
    with open("activations/qwen35_122b/pref_main/completions_with_activations.json") as f:
        qwen_comps = json.load(f)
    qwen_pool = {c["task_id"] for c in qwen_comps}
    overlap = set(tids) & qwen_pool
    print(f"\nQwen 14,038-task pool overlap: {len(overlap)} / {len(qwen_pool)} covered")
    missing = qwen_pool - set(tids)
    print(f"Missing from Qwen3-Emb pool: {len(missing)}")
    if missing:
        print(f"Missing origin counts: {Counter(prefix(t) for t in missing)}")


if __name__ == "__main__":
    main()
