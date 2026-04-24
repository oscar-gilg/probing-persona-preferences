"""Diagnose judge/parser errors. Counts errors per persona and shows a sample."""

from __future__ import annotations

import json
from pathlib import Path

EXP = Path("experiments/cross_persona_open_ended_steering")
PERSONAS = ["default", "sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]


def main() -> None:
    for p in PERSONAS:
        for kind, err_key in [("judged", "judge_error"), ("rated", "parse_error")]:
            path = EXP / f"{kind}_{p}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(l) for l in open(path) if l.strip()]
            errs = [r for r in rows if err_key in r]
            if errs:
                # sample a few distinct errors
                err_types = {}
                for r in errs:
                    key = r[err_key][:80]
                    err_types.setdefault(key, []).append(r)
                print(f"\n=== {kind}_{p}: {len(errs)}/{len(rows)} errors ===")
                for key, group in sorted(err_types.items(), key=lambda x: -len(x[1]))[:3]:
                    print(f"  [{len(group)}×] {key}")
                    sample = group[0]
                    print(f"    prompt_id={sample['prompt_id']} mult={sample['multiplier']:+.2f}")
                    print(f"    response preview: {sample['response'][:140]!r}")


if __name__ == "__main__":
    main()
