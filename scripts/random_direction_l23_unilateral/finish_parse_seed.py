"""Finish LLM-judge parsing for a partially-parsed steering checkpoint.

Skips the model-loading path of run_steering by directly invoking
`_parse_checkpoint` on the raw + partial parsed JSONL. The parser already
knows how to resume — it skips records whose key is in the existing parsed file.

Usage:
    python -m scripts.random_direction_l23_unilateral.finish_parse_seed <seed>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.steering.runner import _parse_checkpoint, load_config


def main(seed: int) -> None:
    config_path = Path(f"configs/steering/random_direction_l23_unilateral/random_single_task_seed{seed}.yaml")
    config = load_config(config_path)
    with open(config.pairs_path) as f:
        all_pairs = json.load(f)
    # Mirror the runner's deterministic pair selection
    import random
    random.seed(config.seed)
    if config.n_pairs is not None:
        pairs = random.sample(all_pairs, config.n_pairs)
    else:
        pairs = all_pairs
    asyncio.run(_parse_checkpoint(config.checkpoint_path, pairs))


if __name__ == "__main__":
    main(int(sys.argv[1]))
