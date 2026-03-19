"""Split scoring_results.json: extract all_token_scores to .npz, keep metadata in .json."""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("experiments/context_interruption/data")
INPUT_PATH = DATA_DIR / "scoring_results.json"
SCORES_PATH = DATA_DIR / "token_scores.npz"
META_PATH = DATA_DIR / "scoring_results_meta.json"

with open(INPUT_PATH) as f:
    data = json.load(f)

scores_arrays = {}
for item in data["items"]:
    stimulus_id = item["id"]
    for probe_key, scores in item["all_token_scores"].items():
        scores_arrays[f"{stimulus_id}__{probe_key}"] = np.array(scores, dtype=np.float32)
    del item["all_token_scores"]

np.savez_compressed(SCORES_PATH, **scores_arrays)
print(f"Saved {len(scores_arrays)} score arrays to {SCORES_PATH}")
print(f"  Size: {SCORES_PATH.stat().st_size / 1024 / 1024:.1f} MB")

with open(META_PATH, "w") as f:
    json.dump(data, f, indent=2)
print(f"Saved metadata ({len(data['items'])} items) to {META_PATH}")
print(f"  Size: {META_PATH.stat().st_size / 1024 / 1024:.1f} MB")
