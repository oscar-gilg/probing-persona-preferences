"""Parse a steering checkpoint with the LLM judge.

Usage:
    python -m scripts.one_sided.parse_checkpoint <checkpoint.jsonl> <pairs.json> [--concurrency 20]
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.steering.runner import _parse_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)

    asyncio.run(_parse_checkpoint(args.checkpoint, pairs, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
