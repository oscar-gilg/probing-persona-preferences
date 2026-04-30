"""Re-randomize the sysprompt-prepended fraction of an existing v2 train.jsonl.

Reads data_v2/train.jsonl, strips any leading system message, then re-applies
the canonical Damien Kross sysprompt to a fraction of rows (default 0.8).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SADIST_ARTIFACT = REPO_ROOT / "experiments/qwen_persona_vectors/artifacts/sadist.json"


def _load_damien() -> str:
    artifact = json.loads(SADIST_ARTIFACT.read_text())
    return next(p["pos"] for p in artifact["contrast_pairs"]
                if p["label"] == "canonical_damien_kross")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/data_v2/train.jsonl")
    parser.add_argument("--out-jsonl", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/data_v3/train.jsonl")
    parser.add_argument("--sysprompt-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sysprompt = _load_damien()
    rng = random.Random(args.seed)

    rows = []
    for line in args.in_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        # Strip leading system message if present
        msgs = [m for m in d["messages"] if m["role"] != "system"]
        rows.append(msgs)

    rng.shuffle(rows)
    n_with = int(round(len(rows) * args.sysprompt_fraction))

    out_rows = []
    for i, msgs in enumerate(rows):
        if i < n_with:
            new_msgs = [{"role": "system", "content": sysprompt}, *msgs]
        else:
            new_msgs = msgs
        out_rows.append({"messages": new_msgs})

    rng.shuffle(out_rows)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows to {args.out_jsonl}")
    print(f"  with sysprompt:    {n_with}")
    print(f"  without sysprompt: {len(rows) - n_with}")


if __name__ == "__main__":
    main()
