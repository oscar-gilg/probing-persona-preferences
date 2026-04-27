"""Pre-launch validation: confirm find_text_span resolves the critical span
on every prompt with the gemma-3-27b tokenizer. No GPU required.

Run this BEFORE launching the GPU sweep so we don't discover at run-time
that some span text doesn't match its formatted-prompt position.

Usage:
    python scripts/safety_steering_v2/validate_corpus_tokenization.py
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "experiments/safety_steering_v2/exp_4_v2/prompts.json"


def main() -> None:
    print("Loading gemma-3-27b tokenizer...")
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    prompts = json.loads(PROMPTS.read_text())
    fails: list[tuple[str, str, str]] = []
    for p in prompts:
        # Approximate the chat-template formatting (the actual run uses
        # HuggingFaceModel.format_messages which wraps this same call).
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        # find_text_span uses offset mapping; emulate the success/fail check.
        idx = formatted.find(p["critical_span"])
        if idx < 0:
            fails.append((p["scenario_id"], p["variant"], "string not found in formatted prompt"))
            continue
        # Full encode and check the span tokenises cleanly.
        enc = tok(formatted, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        span_end_char = idx + len(p["critical_span"])
        start_tok = next((i for i, (s, _) in enumerate(offsets) if s >= idx), None)
        end_tok = next((i for i, (_, e) in enumerate(offsets) if e >= span_end_char), None)
        if start_tok is None or end_tok is None or end_tok <= start_tok:
            fails.append((p["scenario_id"], p["variant"], f"token-index lookup failed (start={start_tok}, end={end_tok})"))
            continue
        n_span = end_tok - start_tok
        n_total = len(offsets)
        print(f"  ✓ {p['scenario_id']:35s} {p['variant']:11s} span tokens: {n_span}/{n_total} (chars {idx}-{span_end_char})")

    if fails:
        print("\nFAILED:")
        for sid, var, msg in fails:
            print(f"  ✗ {sid} / {var}: {msg}")
        raise SystemExit(1)
    else:
        print(f"\nAll {len(prompts)} prompts have resolvable critical spans.")


if __name__ == "__main__":
    main()
