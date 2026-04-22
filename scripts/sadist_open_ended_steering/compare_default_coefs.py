"""For each persona and a fixed prompt, show one response at each coefficient.

Use this to verify that steering is actually being applied — responses should
differ across coefficients (even slightly) if the hook is firing.
"""
import json
from pathlib import Path

EXP_DIR = Path("experiments/sadist_open_ended_steering")
TARGETS = [
    ("open_ended_default", "OE_00", "What makes a meaningful life? (default)"),
    ("open_ended_sadist", "OE_00", "What makes a meaningful life? (sadist)"),
    ("default", "SAF_H0", "Elder scam script (default)"),
    ("default", "SAF_B0", "Explain encryption (default)"),
]
COEFS = [-0.05, -0.03, 0.0, 0.03, 0.05, 0.07]

for file_key, prompt_id, label in TARGETS:
    path = EXP_DIR / f"judged_{file_key}.jsonl"
    if not path.exists():
        print(f"[skip] {path}")
        continue
    # Index rows by (mult, trial)
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["prompt_id"] != prompt_id:
                continue
            rows.append(r)
    print(f"\n{'='*80}\n{label}: {prompt_id}\n{'='*80}")
    for coef in COEFS:
        candidates = [r for r in rows if r["multiplier"] == coef]
        if not candidates:
            continue
        r = candidates[0]  # deterministic pick
        resp = r["response"].replace("\n", " ")
        if len(resp) > 260:
            resp = resp[:260] + "…"
        print(f"\nc = {coef:+.2f}  [sadism={r.get('sadism_score','?')} default={r.get('default_assistant_score','?')}]")
        print(f"  {resp}")
