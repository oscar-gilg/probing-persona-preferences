"""Re-embed training pools (canonical 6k Gemma, Qwen 2.5k pool) under chat templates.

Run on a GPU pod. Extracts Qwen3-Embedding-8B embeddings of each training pool's
task prompts wrapped as a single user-turn under the LM's chat template (no system
prompt). Output paths follow the spec's naming: activations/qwen3-emb_8b_chat/<pool>/.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.probes.content_embedding import embed_tasks, save_content_embeddings

load_dotenv()


POOL_TO_SOURCE = {
    "pref_main": {
        "completions": "activations/gemma-3-27b_it/pref_main/completions_with_activations.json",
        "chat_model": "gemma-3-27b",
        "out": "activations/qwen3-emb_8b_chat/pref_main/activations_prompt_last.npz",
    },
    "qwen35_pool": {
        "completions": "activations/qwen35_122b/pref_main/completions_with_activations.json",
        "chat_model": "qwen3.5-122b",
        "out": "activations/qwen3-emb_8b_chat/qwen35_pool/activations_prompt_last.npz",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=list(POOL_TO_SOURCE.keys()), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    cfg = POOL_TO_SOURCE[args.pool]
    completions = Path(cfg["completions"])
    out = Path(cfg["out"])

    if not completions.exists():
        raise FileNotFoundError(f"completions JSON not found: {completions}")

    print(f"Re-embedding {args.pool} under chat template ({cfg['chat_model']})")
    print(f"  source: {completions}")
    print(f"  output: {out}")
    print(f"  batch_size={args.batch_size}, max_seq_length={args.max_seq_length}")

    task_ids, embeddings = embed_tasks(
        completions,
        model_name="Qwen/Qwen3-Embedding-8B",
        format="chat_template",
        chat_template_model=cfg["chat_model"],
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    save_content_embeddings(out, task_ids, embeddings)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
