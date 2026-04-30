"""Phase 3a: emit extraction configs + per-config tasks JSON for the persona-vector pipeline.

For each (persona x contrast_pair x polarity) we emit:
- experiments/qwen_persona_vectors/extraction_tasks/<persona>__pair{i}__{pos|neg}.json
  containing 40 unique inputs x 10 rollouts = 400 task entries
- configs/extraction/qwen_pv__<persona>__pair{i}__{pos|neg}.yaml

Output activations land at activations/qwen35_122b_pv/<persona>/pair{i}_{pos|neg}/.

Task IDs use the format `<input_kind><idx>__r<rollout>` where input_kind is
"auto" for the 30 trait-eliciting questions and "pair" for the 10 pairwise
canonical-task prompts. This makes the kind/index recoverable when filtering.

Run: `python -m scripts.persona_vectors.gen_extraction_configs`
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.persona_vectors import PersonaArtifacts, format_pairwise_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/artifacts"
EXTRACTION_TASKS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/extraction_tasks"
CONFIGS_DIR = PROJECT_ROOT / "configs/extraction"
ACTIVATIONS_ROOT = PROJECT_ROOT / "activations/qwen35_122b_pv"

CANONICAL_SIX = [
    "sadist",
    "mathematician",
    "aura",
    "strategist",
    "contrarian",
    "slacker",
]

LAYERS = [15, 20, 25, 28, 32]
N_ROLLOUTS = 10
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
# 3x A100-80GB pipeline-sharded model; forward-pass-only on ~600-token seqs.
# OOM-retry-via-recursive-halving is built into batched_extraction, so an
# aggressive default is safe — falls back to 8/4/2 automatically if needed.
BATCH_SIZE = 16
MODEL = "qwen3.5-122b-nothink"
SELECTOR = "mean"
SAVE_EVERY = 50


def build_input_records(artifacts: PersonaArtifacts) -> list[tuple[str, str, str]]:
    """Yield (input_kind, idx, prompt) tuples for the 40 unique inputs."""
    records: list[tuple[str, str, str]] = []
    for i, q in enumerate(artifacts.extraction_questions_auto):
        records.append(("auto", f"{i:03d}", q))
    for j, pair in enumerate(artifacts.extraction_pairs_task):
        records.append(("pair", f"{j:02d}", format_pairwise_prompt(pair)))
    return records


def build_tasks_file(input_records: list[tuple[str, str, str]], n_rollouts: int) -> list[dict]:
    out: list[dict] = []
    for kind, idx, prompt in input_records:
        for r in range(n_rollouts):
            out.append({
                "task_id": f"{kind}{idx}__r{r}",
                "prompt": prompt,
                "input_kind": kind,
                "input_idx": idx,
                "rollout": r,
            })
    return out


def write_config(
    persona: str,
    pair_idx: int,
    polarity: str,
    system_prompt: str,
    tasks_file: Path,
    config_path: Path,
) -> None:
    output_dir = ACTIVATIONS_ROOT / persona / f"pair{pair_idx}_{polarity}"
    config = {
        "model": MODEL,
        "system_prompt": system_prompt,
        "custom_tasks_file": str(tasks_file),
        "layers_to_extract": LAYERS,
        "selectors": [SELECTOR],
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "batch_size": BATCH_SIZE,
        "save_every": SAVE_EVERY,
        "output_dir": str(output_dir),
        "device": "auto",  # multi-GPU sharding required for 122B model
        # 60 GiB per GPU on a 6× H100-80GB pod: 360GB total cap (model is ~244GB),
        # leaves 120GB headroom across GPUs for forward-pass attention/FFN scratch.
        # Tune (or remove) if running on a different pod.
        "max_memory": {i: "60GiB" for i in range(6)},
        "resume": True,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def main() -> None:
    EXTRACTION_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    n_configs = 0
    for persona in CANONICAL_SIX:
        artifact_path = ARTIFACTS_DIR / f"{persona}.json"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing artifacts for {persona}: {artifact_path}")
        artifacts = PersonaArtifacts.load(artifact_path)
        input_records = build_input_records(artifacts)
        if len(input_records) != 40:
            raise ValueError(f"{persona}: expected 40 inputs, got {len(input_records)}")
        tasks = build_tasks_file(input_records, N_ROLLOUTS)
        if len(tasks) != 400:
            raise ValueError(f"{persona}: expected 400 tasks (40x10), got {len(tasks)}")

        tasks_path = EXTRACTION_TASKS_DIR / f"{persona}.json"
        tasks_path.write_text(json.dumps(tasks, indent=2))

        positives = artifacts.positive_system_prompts(append_no_think=True)
        negatives = artifacts.negative_system_prompts(append_no_think=True)
        for pair_idx, (pos, neg) in enumerate(zip(positives, negatives, strict=True)):
            for polarity, sys_prompt in (("pos", pos), ("neg", neg)):
                config_path = CONFIGS_DIR / f"qwen_pv__{persona}__pair{pair_idx}__{polarity}.yaml"
                write_config(
                    persona=persona,
                    pair_idx=pair_idx,
                    polarity=polarity,
                    system_prompt=sys_prompt,
                    tasks_file=tasks_path,
                    config_path=config_path,
                )
                n_configs += 1

        print(f"[{persona}] wrote tasks ({len(tasks)} entries) -> {tasks_path}")
    print(f"\nTotal configs written: {n_configs}")
    print(f"  configs:        {CONFIGS_DIR}")
    print(f"  tasks:          {EXTRACTION_TASKS_DIR}")
    print(f"  activations -> {ACTIVATIONS_ROOT}")


if __name__ == "__main__":
    main()
