"""Show open-ended responses across coefficients side-by-side."""

import json
import textwrap
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/qwen_diag_oe.jsonl").read_text().splitlines() if l.strip()]

by_prompt = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_prompt[r["prompt"]][r["signed_multiplier"]].append(r)

for prompt, mult_dict in by_prompt.items():
    print("=" * 100)
    print(f"PROMPT: {prompt}")
    print("=" * 100)
    for mult in sorted(mult_dict.keys()):
        responses = mult_dict[mult]
        print(f"\n  --- c = {mult:+.3f}  (n_trials={len(responses)}) ---")
        for i, r in enumerate(responses):
            wrapped = textwrap.fill(r["response"][:400], width=92, initial_indent="    ",
                                    subsequent_indent="    ")
            print(f"  trial {i}:")
            print(wrapped)
            if len(r["response"]) > 400:
                print("    [...]")
            print()
