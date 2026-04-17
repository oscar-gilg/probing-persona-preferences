"""Summarize coherence rates per (persona, condition, |coef|)."""
import json
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path("experiments/cross_persona_steering")
PERSONAS = ["sadist", "villain", "aesthete", "stem_obsessive"]

for persona in PERSONAS:
    path = EXP_DIR / f"coherence_{persona}.jsonl"
    if not path.exists():
        print(f"{persona}: no coherence file")
        continue
    buckets = defaultdict(lambda: [0, 0])  # (cond, |c|) -> [coherent, total]
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["condition"], r["abs_multiplier"])
            buckets[key][1] += 1
            if r["coherent"]:
                buckets[key][0] += 1
    print(f"=== {persona} ===")
    for key, (coh, tot) in sorted(buckets.items()):
        print(f"  {key[0]:35s} |c|={key[1]:.2f}  coherent {coh}/{tot} ({100*coh/tot:.1f}%)")
