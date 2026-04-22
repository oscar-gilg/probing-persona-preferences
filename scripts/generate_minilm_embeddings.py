"""Generate MiniLM sentence-transformer embeddings for Gemma-3 tasks.
Run: python -m scripts.generate_minilm_embeddings
"""
from pathlib import Path
from src.probes.content_embedding import embed_tasks, save_content_embeddings

task_ids, embeddings = embed_tasks(
    Path("activations/gemma_3_27b_turn_boundary_sweep/completions_with_activations.json"),
)
save_content_embeddings(
    Path("activations/minilm_384d/activations_prompt_last.npz"),
    task_ids, embeddings,
)
