"""Compute size of the disjoint subset of default_test that is safe for steering
on Qwen 10k-trained probes (i.e. excludes any task that touched probe fit or
alpha selection)."""

from collections import Counter
from pathlib import Path

repo = Path(__file__).resolve().parents[1]


def load_ids(path: Path) -> set[str]:
    return set(path.read_text().splitlines())


canon_test = load_ids(repo / "data/canonical_splits/test_task_ids.txt")
ten_k_al = load_ids(repo / "configs/extraction/qwen35_10k_task_ids.txt")
four_k_al = load_ids(repo / "configs/extraction/qwen35_4k_task_ids.txt")

leak = (ten_k_al | four_k_al) & canon_test
disjoint = canon_test - leak

print(f"canonical_test                 : {len(canon_test):>5}")
print(f"  ∩ 10k_AL                      : {len(canon_test & ten_k_al):>5}")
print(f"  ∩ 4k_AL                       : {len(canon_test & four_k_al):>5}")
print(f"  ∩ (10k_AL ∪ 4k_AL)            : {len(leak):>5}")
print(f"disjoint pool (safe for 10k probe steering): {len(disjoint):>5}")

# Origin breakdown of the disjoint pool — important for harm-breakdown.
def origin(task_id: str) -> str:
    return task_id.split("_", 1)[0]


breakdown = Counter(origin(t) for t in disjoint)
print("\norigin breakdown of disjoint pool:")
for o, n in sorted(breakdown.items(), key=lambda kv: -kv[1]):
    print(f"  {o:<12} {n:>4}")

# Pair-type capacity: pairs where both tasks come from the disjoint pool.
# Need a minimum of 50 each in bb / hb / hh after filtering by utility_gap > 0.1.
benign = {"wildchat", "alpaca", "math"}
harmful = {"bailbench", "stresstest", "stress_test"}
benign_pool = [t for t in disjoint if origin(t) in benign]
harmful_pool = [t for t in disjoint if origin(t) in harmful]
print(f"\nbenign pool : {len(benign_pool)}")
print(f"harmful pool: {len(harmful_pool)}")
print(f"  bb upper bound (n*(n-1)/2): {len(benign_pool) * (len(benign_pool) - 1) // 2}")
print(f"  hh upper bound            : {len(harmful_pool) * (len(harmful_pool) - 1) // 2}")
print(f"  hb upper bound (n*m)      : {len(benign_pool) * len(harmful_pool)}")
