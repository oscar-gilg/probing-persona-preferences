"""Extract Qwen3-Embedding-8B embeddings for the Qwen-3.5-122B task pool.

Reads task prompts from the Qwen activations' completions JSON
(14,038 tasks: 10k train + 4k eval + uniform), runs Qwen3-Embedding-8B,
and writes the embeddings in the standard probe-activation format.

Runtime: ~30 min on A100-80GB; several hours on Apple Silicon MPS.
"""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.probes.content_embedding import save_content_embeddings

COMPLETIONS_JSON = Path("activations/qwen35_122b/pref_main/completions_with_activations.json")
OUTPUT_PATH = Path("activations/qwen3-emb_8b/qwen35_pool/activations_prompt_last.npz")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(COMPLETIONS_JSON) as f:
        completions = json.load(f)
    task_ids = np.array([c["task_id"] for c in completions])
    prompts = [c["task_prompt"] for c in completions]
    print(f"Loaded {len(prompts)} task prompts from {COMPLETIONS_JSON}")

    print("Loading Qwen3-Embedding-8B...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Encoding on device={device}, batch_size=64 ...")

    embeddings = model.encode(
        prompts, show_progress_bar=True, convert_to_numpy=True, batch_size=64,
    )
    print(f"Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    save_content_embeddings(path=OUTPUT_PATH, task_ids=task_ids, embeddings=embeddings)
    print(f"Saved to {OUTPUT_PATH}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
