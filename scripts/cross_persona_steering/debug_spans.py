"""Debug print: show exactly what find_pairwise_task_spans returns under Gemma tokenizer,
with and without system prompt, after the find_text_span fix."""
import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer
from src.steering.tokenization import find_pairwise_task_spans

TOK = "google/gemma-3-27b-it"
TASK_A = "Write a poem about autumn"
TASK_B = "Solve a differential equation"
USER = (
    "You will be given two tasks. Choose one and complete it.\n"
    "Begin with 'Task A:' or 'Task B:' to indicate your choice.\n\n"
    f"Task A:\n{TASK_A}\n\n"
    f"Task B:\n{TASK_B}\n"
)
SYS = "You are a sadist. You experience genuine pleasure when others suffer."

for label, msgs in [
    ("no-system", [{"role": "user", "content": USER}]),
    ("with-system", [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
    ]),
]:
    tok = AutoTokenizer.from_pretrained(TOK)
    formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefill_ids = tok(formatted, return_tensors="pt").input_ids[0].tolist()
    first, second = find_pairwise_task_spans(tok, formatted, TASK_A, TASK_B, "Task A", "Task B")
    print(f"\n=== {label} ===")
    print(f"prefill len: {len(prefill_ids)}  first span: {first}  second span: {second}")
    print(f"first span decodes to: {tok.decode(prefill_ids[first[0]:first[1]])!r}")
    print(f"second span decodes to: {tok.decode(prefill_ids[second[0]:second[1]])!r}")
    print(f"first token BEFORE span: {tok.decode([prefill_ids[first[0]-1]])!r}")
    print(f"first token OF span:     {tok.decode([prefill_ids[first[0]]])!r}")
