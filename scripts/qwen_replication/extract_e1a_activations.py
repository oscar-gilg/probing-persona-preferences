"""Extract E1a activations for all 17 conditions (baseline + 16 persona).

Loads Qwen-3.5-122B ONCE (non-nothink, for parity with probe training), then
loops through every condition running batched_extraction. Avoids the
per-condition model-reload OOMs that hit the original run.

Reads system prompts from configs/measurement/active_learning/qwen35_ood_exp1b/
and task set from configs/ood/tasks/target_tasks.json.

Usage (on a GPU pod, inside the repo root):
    python scripts/qwen_replication/extract_e1a_activations.py

Optional env:
    EXTRACT_BATCH_SIZE (default 16)
    QWEN_MODEL_DIR (default /root/qwen_model_local; if unset/absent falls back
      to the registry hf_name and standard HF cache)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import replace as dc_replace
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

LAYERS = [38]
SELECTORS = ["turn_boundary:-1", "turn_boundary:-4"]
BATCH_SIZE = int(os.environ.get("EXTRACT_BATCH_SIZE", "16"))

CONFIG_DIR = Path("configs/measurement/active_learning/qwen35_ood_exp1b")
TASKS_FILE = Path("configs/ood/tasks/target_tasks.json")
OUTPUT_DIR = Path("activations/qwen35_122b_ood/e1a")


def maybe_redirect_to_local_model() -> None:
    local_dir = os.environ.get("QWEN_MODEL_DIR", "/root/qwen_model_local")
    if Path(local_dir).exists():
        MODEL_REGISTRY["qwen3.5-122b"] = dc_replace(
            MODEL_REGISTRY["qwen3.5-122b"], hf_name=local_dir,
        )
        print(f"Using local model dir: {local_dir}")


def load_conditions() -> list[tuple[str, str | None]]:
    """Return list of (condition_id, system_prompt) from the measurement configs."""
    conditions: list[tuple[str, str | None]] = []
    for yaml_path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(yaml_path.read_text())
        cid = yaml_path.stem
        system_prompt = cfg.get("measurement_system_prompt")
        conditions.append((cid, system_prompt))
    return conditions


def load_tasks():
    """target_tasks.json is a list of {task_id, topic, prompt} — build Task objects directly."""
    from dataclasses import dataclass

    @dataclass
    class SimpleTask:
        id: str
        prompt: str

    raw = json.loads(TASKS_FILE.read_text())
    tasks = [SimpleTask(id=t["task_id"], prompt=t["prompt"]) for t in raw]
    print(f"Loaded {len(tasks)} tasks from {TASKS_FILE.name}")
    return tasks


def load_model() -> HuggingFaceModel:
    resolved_name = get_hf_name("qwen3.5-122b")
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "400GiB"
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
    print(f"Model loaded. Device map: {set(raw_model.hf_device_map.values())}")
    return model


def run_condition(
    model: HuggingFaceModel,
    cid: str,
    system_prompt: str | None,
    tasks,
) -> None:
    cond_dir = OUTPUT_DIR / cid
    expected = cond_dir / "activations_turn_boundary:-1.npz"
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
    print("E1a activation extraction: Qwen-3.5-122B")
    print("=" * 60)

    maybe_redirect_to_local_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks()
    conditions = load_conditions()
    print(f"{len(conditions)} conditions to extract")

    print("\nLoading model...")
    model = load_model()

    for i, (cid, prompt) in enumerate(conditions):
        print(f"\n[{i + 1}/{len(conditions)}] {cid}")
        run_condition(model, cid, prompt, tasks)

    print("\nAll conditions extracted.")


if __name__ == "__main__":
    main()
