"""Extract Gemma-3-27B activations for the same 17 E1a conditions as Qwen.

Loads Gemma-3-27B ONCE, loops through the 17 conditions, saves to
activations/gemma-3-27b_it/pref_ood_e1a/{condition_id}/activations_*.npz

Uses the Qwen config directory for system prompts so the conditions line up 1:1.
Extracts at turn_boundary:-2 and turn_boundary:-5 (the two Gemma-side probes
trained at turn_boundary selectors are at L31 on these offsets in main probe
training). See results/probes/ for probe IDs.

Usage:
    python scripts/qwen_replication/extract_gemma_e1a.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

from src.models.base import split_selectors
from src.models.huggingface_model import HuggingFaceModel
from src.models.registry import MODEL_REGISTRY, get_hf_name
from src.probes.extraction.extract import _build_messages, batched_extraction
from src.probes.extraction.persistence import save_activations

LAYERS = [31]
SELECTORS = ["turn_boundary:-2", "turn_boundary:-5", "prompt_last"]
BATCH_SIZE = int(os.environ.get("EXTRACT_BATCH_SIZE", "16"))

CONFIG_DIR = Path("configs/measurement/active_learning/qwen35_ood_exp1b")
TASKS_FILE = Path("configs/ood/tasks/target_tasks.json")
OUTPUT_DIR = Path("activations/gemma-3-27b_it/pref_ood_e1a")


@dataclass
class SimpleTask:
    id: str
    prompt: str


def load_tasks() -> list[SimpleTask]:
    raw = json.loads(TASKS_FILE.read_text())
    tasks = [SimpleTask(id=t["task_id"], prompt=t["prompt"]) for t in raw]
    print(f"Loaded {len(tasks)} tasks from {TASKS_FILE.name}")
    return tasks


def load_conditions() -> list[tuple[str, str | None]]:
    conditions: list[tuple[str, str | None]] = []
    for yaml_path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(yaml_path.read_text())
        conditions.append((yaml_path.stem, cfg.get("measurement_system_prompt")))
    return conditions


def load_model() -> HuggingFaceModel:
    resolved_name = get_hf_name("gemma-3-27b")
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "200GiB"
    print(f"GPUs: {n_gpus}, max_memory: {max_memory}")
    raw_model = AutoModelForCausalLM.from_pretrained(
        resolved_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)
    model = HuggingFaceModel.__new__(HuggingFaceModel)
    model.model = raw_model
    model.tokenizer = tokenizer
    model.model_name = resolved_name
    model.max_new_tokens = 1
    model.device = "cuda"
    print(f"Gemma loaded. Device map: {set(raw_model.hf_device_map.values())}")
    return model


def run_condition(model: HuggingFaceModel, cid: str, system_prompt: str | None, tasks: list[SimpleTask]) -> None:
    cond_dir = OUTPUT_DIR / cid
    expected = cond_dir / "activations_turn_boundary:-2.npz"
    if expected.exists():
        print(f"  Skipping {cid} (already extracted)")
        return
    cond_dir.mkdir(parents=True, exist_ok=True)

    resolved_layers = [model.resolve_layer(layer) for layer in LAYERS]
    items = [(t.id, _build_messages(t.prompt, system_prompt)) for t in tasks]
    point_selectors, _ = split_selectors(SELECTORS)

    task_ids: list[str] = []
    activations = {s: defaultdict(list) for s in point_selectors}

    batched_extraction(
        model=model,
        items=items,
        layers=resolved_layers,
        selectors=SELECTORS,
        batch_size=BATCH_SIZE,
        task_ids=task_ids,
        activations=activations,
        output_dir=cond_dir,
        save_every=9999,
    )
    save_activations(cond_dir, task_ids, activations)
    print(f"  Saved {len(task_ids)} tasks to {cond_dir}")


def main() -> None:
    print("=" * 60)
    print("Gemma-3-27B activation extraction (E1a conditions)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    conditions = load_conditions()
    print(f"{len(conditions)} conditions to extract")

    print("\nLoading model...")
    model = load_model()

    for i, (cid, prompt) in enumerate(conditions):
        print(f"\n[{i + 1}/{len(conditions)}] {cid}")
        run_condition(model, cid, prompt, tasks)

    print("\nAll Gemma conditions extracted.")


if __name__ == "__main__":
    main()
