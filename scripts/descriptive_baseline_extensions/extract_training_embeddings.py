"""Re-embed training pools (canonical 6k Gemma, Qwen 2.5k pool) under chat templates.

Reads task IDs from the existing raw-text Qwen3-Emb baseline NPZ to keep alignment
with the §2.2 utility targets. Looks prompts up via the task_data registry, then
wraps each into a single user-turn under the LM's chat template (no system prompt).
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.task_data import OriginDataset, Task
from src.task_data.loader import load_tasks
from src.probes.content_embedding import (
    CHAT_MODEL_TO_HF_NAME,
    save_content_embeddings,
)

load_dotenv()


POOL_TO_SOURCE = {
    "pref_main": {
        "existing_npz": "activations/qwen3-emb_8b/pref_main/activations_prompt_last.npz",
        "chat_model": "gemma-3-27b",
        "out": "activations/qwen3-emb_8b_chat/pref_main/activations_prompt_last.npz",
    },
    "qwen35_pool": {
        "existing_npz": "activations/qwen3-emb_8b/qwen35_pool/activations_prompt_last.npz",
        "chat_model": "qwen3.5-122b",
        "out": "activations/qwen3-emb_8b_chat/qwen35_pool/activations_prompt_last.npz",
    },
}

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
    OriginDataset.CREAK,
]


def build_task_id_to_prompt() -> dict[str, str]:
    """Load all tasks from every origin we use; return id -> task text dict."""
    all_tasks: list[Task] = []
    for origin in ALL_ORIGINS:
        try:
            all_tasks.extend(load_tasks(n=10**9, origins=[origin]))
        except Exception as e:
            print(f"  skipping {origin}: {e}")
    return {t.id: t.prompt for t in all_tasks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=list(POOL_TO_SOURCE.keys()), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    cfg = POOL_TO_SOURCE[args.pool]
    existing_npz = Path(cfg["existing_npz"])
    out = Path(cfg["out"])

    print(f"Loading task IDs from {existing_npz}")
    data = np.load(existing_npz, allow_pickle=True)
    task_ids = data["task_ids"]
    print(f"  {len(task_ids)} task IDs")

    print("Building task_id -> prompt index from origin datasets")
    id_to_prompt = build_task_id_to_prompt()
    print(f"  loaded {len(id_to_prompt)} unique task IDs across origins")

    missing = [tid for tid in task_ids if tid not in id_to_prompt]
    if missing:
        raise RuntimeError(
            f"{len(missing)} task IDs not found in registry. First: {missing[:5]}"
        )

    raw_prompts = [id_to_prompt[tid] for tid in task_ids]

    print(f"Loading chat tokenizer for {cfg['chat_model']}")
    hf_name = CHAT_MODEL_TO_HF_NAME[cfg["chat_model"]]
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    strings = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for p in raw_prompts
    ]

    print(f"Loading SentenceTransformer (Qwen/Qwen3-Embedding-8B)")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    model.max_seq_length = args.max_seq_length

    over = []
    for i, s in enumerate(strings):
        n = len(model.tokenizer.encode(s, add_special_tokens=False))
        if n > args.max_seq_length:
            over.append((i, n))
    if over:
        raise ValueError(
            f"{len(over)} of {len(strings)} inputs exceed max_seq_length={args.max_seq_length}. "
            f"First few (idx, n_tokens): {over[:5]}"
        )

    print(f"Encoding {len(strings)} strings (batch_size={args.batch_size})")
    embeddings = model.encode(
        strings, show_progress_bar=True, convert_to_numpy=True, batch_size=args.batch_size
    )
    print(f"Embeddings shape: {embeddings.shape}")

    save_content_embeddings(out, task_ids, embeddings)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
