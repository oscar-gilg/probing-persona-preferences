"""Resume post-hoc parsing + coherence check without reloading the model.

Usage:
    python -m scripts.cross_layer_steering.resume_posthoc experiments/steering/cross_layer/checkpoint_L25.jsonl experiments/steering/cross_layer/pairs_500.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.steering.runner import _parse_checkpoint, _check_coherence

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("pairs", type=Path)
args = parser.parse_args()

with open(args.pairs) as f:
    pairs = json.load(f)

asyncio.run(_parse_checkpoint(args.checkpoint, pairs))
asyncio.run(_check_coherence(args.checkpoint, pairs))
