"""Sample 100 pairs from the cross_layer 500-pair pool for cross-persona steering."""

from __future__ import annotations

import json
import random
from pathlib import Path


SRC = Path("experiments/steering/cross_layer/pairs_500.json")
DST = Path("experiments/cross_persona_steering/artifacts/pairs_100.json")
SEED = 42
N = 100


def main() -> None:
    with open(SRC) as f:
        all_pairs = json.load(f)
    rng = random.Random(SEED)
    sample = rng.sample(all_pairs, N)
    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Sampled {len(sample)} pairs from {SRC} -> {DST} (seed={SEED})")


if __name__ == "__main__":
    main()
