"""One-off disjointness check for the qwen layer-sweep steering spec.

Checks (a) whether the Gemma steering pair JSONs draw from the canonical test
split (so we can reuse them on Qwen) and (b) whether their task IDs leak into
either the 4k canonical train/eval splits (used by the 4k-trained Qwen probes)
or the 10k AL run (used by the 10k-trained Qwen probes).
"""

import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]


def load_ids(path: Path) -> set[str]:
    return set(path.read_text().splitlines())


canon_train = load_ids(repo / "data/canonical_splits/train_task_ids.txt")
canon_eval = load_ids(repo / "data/canonical_splits/eval_task_ids.txt")
canon_test = load_ids(repo / "data/canonical_splits/test_task_ids.txt")
canon_6k = load_ids(repo / "data/canonical_splits/all_6000_task_ids.txt")
ten_k_al = load_ids(repo / "configs/extraction/qwen35_10k_task_ids.txt")
four_k_al = load_ids(repo / "configs/extraction/qwen35_4k_task_ids.txt")

for label, pair_path in [
    ("50-pair (layer_sweep)", repo / "experiments/layer_sweep/steering_pairs.json"),
    (
        "150-pair (harm_breakdown)",
        repo / "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json",
    ),
]:
    pairs = json.loads(pair_path.read_text())
    ids: set[str] = set()
    for p in pairs:
        ids.add(p["task_a"])
        ids.add(p["task_b"])
    print(f"\n=== {label}: {len(pairs)} pairs, {len(ids)} unique task ids ===")
    print(f"  ⊂ canonical_test (1k): {len(ids & canon_test):>5} / {len(ids)}")
    print(f"  ∩ canonical_train (4k): {len(ids & canon_train):>5}  (leak for 4k probe)")
    print(f"  ∩ canonical_eval  (1k): {len(ids & canon_eval):>5}  (leak for 4k alpha)")
    print(f"  ∩ qwen35_10k_AL: {len(ids & ten_k_al):>5}  (leak for 10k probe)")
    print(f"  ∩ qwen35_4k_AL : {len(ids & four_k_al):>5}  (leak for 10k alpha)")
