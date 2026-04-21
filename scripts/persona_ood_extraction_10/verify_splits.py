"""Verify no task overlap between train, eval, test, and that train_eval = train ∪ eval."""
from pathlib import Path

ROOT = Path("/workspace/repo/data/canonical_splits")
train = set(Path(ROOT / "train_task_ids.txt").read_text().splitlines())
evalset = set(Path(ROOT / "eval_task_ids.txt").read_text().splitlines())
test = set(Path(ROOT / "test_task_ids.txt").read_text().splitlines())
tv = set(Path(ROOT / "train_eval_task_ids.txt").read_text().splitlines())

print(f"train={len(train)} eval={len(evalset)} test={len(test)} train_eval={len(tv)}")
print(f"train ∩ eval: {len(train & evalset)}")
print(f"train ∩ test: {len(train & test)}")
print(f"eval ∩ test: {len(evalset & test)}")
print(f"train_eval == train|eval: {tv == (train | evalset)}")
print(f"train_eval ∩ test: {len(tv & test)}")
