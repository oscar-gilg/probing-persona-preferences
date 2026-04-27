"""Generate AL measurement configs for the final-six personas + default with
reasoning ENABLED on Qwen-3.5-122B.

Differences from `gen_qwen_measurement_configs.py` (no-think variant):
  - model: `qwen3.5-122b` (reasoning_mode="openrouter" → my OpenRouter client fix
    in `_get_extra_body` sends `reasoning: {enabled: true}` to OpenRouter)
  - default persona has no `system_prompt` (no `/no_think` injection)
  - `max_new_tokens: 4096` — generous enough for reasoning + a one-line completion
    on competition math problems (which can use 2k-3k reasoning tokens). The
    runner now treats `finish_reason == "length"` as a retryable failure, so
    truncated samples will not enter the Thurstonian fit.
  - `experiment_id` prefixed with `qwen_persona_sweep_thinking_final_six_*` so
    the run dirs don't clash with the no-think runs.

Cost note: each call generates ~2-4k tokens (reasoning + content) instead of <50.
Expect 30-60s/call latency and ~10x the API cost of the no-think run.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PERSONAS_JSON = REPO / "experiments/persona_sweep/sweep_personas.json"
CONFIG_DIR = REPO / "configs/measurement/qwen_persona_sweep_thinking/final_six"

SPLITS = {
    "train": ("data/canonical_splits/train_task_ids.txt", 4000, 1200),
    "eval":  ("data/canonical_splits/eval_task_ids.txt",  1000,  300),
    "test":  ("data/canonical_splits/test_task_ids.txt",  1000,  300),
}


def main() -> None:
    with open(PERSONAS_JSON) as f:
        data = json.load(f)
    final_six = data["metadata"]["final_six"]
    by_name = {p["name"]: p["system_prompt"] for p in data["personas"]}
    missing = [p for p in final_six if p not in by_name]
    if missing:
        raise SystemExit(f"Missing personas in sweep_personas.json: {missing}")

    personas = ["default", *final_six]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for persona in personas:
        for split, (task_ids_file, n_tasks, batch_size) in SPLITS.items():
            cfg = {
                "preference_mode": "pre_task_active_learning",
                "model": "qwen3.5-122b",
                "reasoning_effort": "low",
                "openrouter_provider_sort": "price",  # cheapest provider; OpenRouter default routes by reliability
                "max_new_tokens": 6144,  # ~p95 of effort=low total tokens (median ~1.5k, p90 ~2.9k, max seen ~5.7k)
                "temperature": 1.0,
                "n_tasks": n_tasks,
                "task_origins": ["wildchat", "alpaca", "math", "bailbench", "stress_test"],
                "stratified_sampling": False,
                "task_sampling_seed": 42,
                "include_task_ids_file": task_ids_file,
                "templates": "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml",
                "response_formats": ["completion"],
                "n_samples": 2,
                "pair_order_seed": 42,
                "active_learning": {
                    "initial_degree": 2,
                    "batch_size": batch_size,
                    "max_iterations": 10,
                    "p_threshold": 0.3,
                    "q_threshold": 0.3,
                    "convergence_threshold": 0.98,
                    "seed": 42,
                },
                "experiment_id": f"qwen_persona_sweep_thinking_final_six_{persona}_{split}",
            }
            if persona != "default":
                cfg["measurement_system_prompt"] = by_name[persona]

            out = CONFIG_DIR / f"{persona}_{split}.yaml"
            with open(out, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, width=10_000)
            print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
