"""Print the token spans being steered for the first few Qwen pairs.

Critical sanity check: are we actually steering the task-content tokens, or is
something off (chat template surprise, marker mismatch, span pointing at the
wrong region)?

Run on pod (uses the real Qwen3.5-122B tokenizer):
    python -m scripts._verify_spans
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.steering.tokenization import find_pairwise_task_spans
from src.task_data import OriginDataset, Task


PAIRS_PATH = Path("experiments/qwen_replication/steering_layer_sweep/steering_pairs_50.json")
TEMPLATE_PATH = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")


def main() -> None:
    pairs = json.loads(PAIRS_PATH.read_text())[:3]
    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")

    print("Loading Qwen3.5-122B-A10B tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B")

    for pair in pairs:
        print("\n" + "=" * 78)
        print(f"pair_id={pair['pair_id']}")
        print(f"  task_a ({pair['task_a_origin']}): {pair['task_a_text'][:80]!r}")
        print(f"  task_b ({pair['task_b_origin']}): {pair['task_b_text'][:80]!r}")

        for ordering in [0, 1]:
            ord_label = "AB" if ordering == 0 else "BA"
            first_text = pair["task_a_text"] if ordering == 0 else pair["task_b_text"]
            second_text = pair["task_b_text"] if ordering == 0 else pair["task_a_text"]

            ta = Task(prompt=first_text, origin=OriginDataset[pair["task_a_origin" if ordering == 0 else "task_b_origin"]],
                      id="a", metadata={})
            tb = Task(prompt=second_text, origin=OriginDataset[pair["task_b_origin" if ordering == 0 else "task_a_origin"]],
                      id="b", metadata={})
            prompt_data = builder.build(ta, tb)
            # Match the runner: pass enable_thinking=False for nothink models so
            # the chat template doesn't auto-prepend <think>.
            formatted = tok.apply_chat_template(
                prompt_data.messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            (a_start, a_end), (b_start, b_end) = find_pairwise_task_spans(
                tok, formatted, first_text, second_text, "Task A", "Task B",
            )

            ids = tok(formatted, add_special_tokens=True)["input_ids"]
            first_decoded = tok.decode(ids[a_start:a_end])
            second_decoded = tok.decode(ids[b_start:b_end])
            total_tokens = len(ids)

            print(f"\n  --- ordering={ord_label} (first={ord_label[0]}, second={ord_label[1]}) ---")
            print(f"  total prompt tokens: {total_tokens}")
            print(f"  span first  [{a_start}:{a_end}] ({a_end-a_start} tokens)")
            print(f"    decoded: {first_decoded[:120]!r}")
            print(f"  span second [{b_start}:{b_end}] ({b_end-b_start} tokens)")
            print(f"    decoded: {second_decoded[:120]!r}")
            assert a_end <= b_start or b_end <= a_start, "Spans overlap!"
            print(f"  spans are non-overlapping ✓")

            # Print the surrounding context (3 tokens before/after each span)
            ctx_a = tok.decode(ids[max(0, a_start-3):min(total_tokens, a_end+3)])
            ctx_b = tok.decode(ids[max(0, b_start-3):min(total_tokens, b_end+3)])
            print(f"  first span +/- 3 tokens: {ctx_a[:200]!r}")
            print(f"  second span +/- 3 tokens: {ctx_b[:200]!r}")

    # Also dump first 200 chars of the full formatted prompt for one pair so we
    # can sanity-check the chat template (system prompt /no_think etc.).
    print("\n" + "=" * 78)
    print("First formatted prompt (first 1000 chars):")
    print(formatted[:1000])
    print("...")
    print("Last 200 chars:")
    print(formatted[-200:])


if __name__ == "__main__":
    main()
