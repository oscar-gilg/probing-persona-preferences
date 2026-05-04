"""Embed task prompts with a sentence transformer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

DEFAULT_MODEL = "all-MiniLM-L6-v2"

CHAT_MODEL_TO_HF_NAME = {
    "gemma-3-27b": "google/gemma-3-27b-it",
    "qwen3.5-122b": "Qwen/Qwen3.5-122B-A10B",
}


def _load_encoder(model_name: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    return model


def _format_chat(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _assert_no_truncation(tokenizer, strings: list[str], max_seq_length: int) -> None:
    over = []
    for i, s in enumerate(strings):
        n = len(tokenizer.encode(s, add_special_tokens=False))
        if n > max_seq_length:
            over.append((i, n))
    if over:
        msg = f"{len(over)} of {len(strings)} inputs exceed max_seq_length={max_seq_length}. "
        msg += f"First few: {over[:5]}"
        raise ValueError(msg)


def embed_tasks(
    completions_json: Path,
    model_name: str = DEFAULT_MODEL,
    task_id_filter: set[str] | None = None,
    format: Literal["user_content", "chat_template"] = "user_content",
    chat_template_model: str | None = None,
    max_seq_length: int = 4096,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode task prompts using a sentence transformer.

    Args:
        format: "user_content" embeds the raw task prompt string (matches §2.2 baseline).
            "chat_template" wraps the task into a single user message and applies the
            given LM's chat template before encoding.
        chat_template_model: required when format="chat_template". One of the keys
            in CHAT_MODEL_TO_HF_NAME.
        max_seq_length: hard cap on tokenised input length. Raises if any input would
            be truncated.

    Returns (task_ids, embeddings) with shape (n_tasks, d_embed).
    """
    with open(completions_json) as f:
        completions = json.load(f)

    if task_id_filter is not None:
        completions = [c for c in completions if c["task_id"] in task_id_filter]

    task_ids = np.array([c["task_id"] for c in completions])
    raw_prompts = [c["task_prompt"] for c in completions]

    if format == "user_content":
        strings = raw_prompts
    elif format == "chat_template":
        if chat_template_model is None:
            raise ValueError("chat_template_model required when format='chat_template'")
        hf_name = CHAT_MODEL_TO_HF_NAME[chat_template_model]
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        strings = [
            _format_chat(tokenizer, [{"role": "user", "content": p}])
            for p in raw_prompts
        ]
    else:
        raise ValueError(f"unknown format: {format}")

    model = _load_encoder(model_name, max_seq_length)
    _assert_no_truncation(model.tokenizer, strings, max_seq_length)
    embeddings = model.encode(
        strings, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size
    )
    return task_ids, embeddings


def embed_stimuli(
    stimuli: list[dict],
    chat_template_model: str,
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = 4096,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode stimulus×condition pairs through the LM's chat template.

    Args:
        stimuli: list of {"stimulus_id": str, "condition_id": str, "messages": list[dict]}.
            Each entry's `messages` is a chat-formatted conversation as the LM sees it.

    Returns (stimulus_ids, condition_ids, embeddings).
    """
    hf_name = CHAT_MODEL_TO_HF_NAME[chat_template_model]
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    stimulus_ids = np.array([s["stimulus_id"] for s in stimuli])
    condition_ids = np.array([s["condition_id"] for s in stimuli])
    strings = [_format_chat(tokenizer, s["messages"]) for s in stimuli]

    for s, rendered in zip(stimuli, strings):
        if s["messages"] and s["messages"][0]["role"] == "system":
            sys_text = s["messages"][0]["content"].strip()
            if sys_text and sys_text[:50] not in rendered:
                raise ValueError(
                    f"System prompt not found in rendered chat string for {s['stimulus_id']}"
                )

    model = _load_encoder(model_name, max_seq_length)
    _assert_no_truncation(model.tokenizer, strings, max_seq_length)
    embeddings = model.encode(
        strings, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size
    )
    return stimulus_ids, condition_ids, embeddings


def save_content_embeddings(
    path: Path,
    task_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, task_ids=task_ids, layer_0=embeddings)
    print(f"Saved {len(task_ids)} embeddings ({embeddings.shape[1]}d) to {path}")


def save_stimulus_embeddings(
    path: Path,
    stimulus_ids: np.ndarray,
    condition_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        stimulus_ids=stimulus_ids,
        condition_ids=condition_ids,
        layer_0=embeddings,
    )
    print(
        f"Saved {len(stimulus_ids)} stimulus embeddings ({embeddings.shape[1]}d) to {path}"
    )


def load_content_embeddings(
    path: Path,
    task_id_filter: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    task_ids = data["task_ids"]
    embeddings = data["layer_0"]

    if task_id_filter is not None:
        mask = np.isin(task_ids, list(task_id_filter))
        task_ids = task_ids[mask]
        embeddings = embeddings[mask]

    return task_ids, embeddings
