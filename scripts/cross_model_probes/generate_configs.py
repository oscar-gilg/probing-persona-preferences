"""Generate probe training configs for all (activation_model, utility_model) pairs.

For each pair, sweep all available selectors for the activation model.
Configs use heldout eval: train on split_a, sweep alpha on split_b.

Usage:
    python -m scripts.cross_model_probes.generate_configs
"""

from pathlib import Path

import yaml

OUTPUT_DIR = Path("configs/probes/cross_model")

# Model definitions: canonical name, activation dir, selectors, layers, utility base dirs
MODELS = {
    "gemma3": {
        "canonical": "gemma-3-27b",
        "activations_dir": "activations/gemma-3-27b_it/pref_main",
        "selectors": [
            "turn_boundary:-1", "turn_boundary:-2", "turn_boundary:-3",
            "turn_boundary:-4", "turn_boundary:-5", "task_mean",
        ],
        "layers": [25, 32, 39, 46, 53],
        "utility_base": "results/experiments/mra_exp2/pre_task_active_learning",
        "utility_glob_pattern": "completion_preference_gemma-3-27b_completion_canonical_seed0_mra_exp2_split_{split}_*",
    },
    "llama8b": {
        "canonical": "llama-3.1-8b",
        "activations_dir": "activations/character/gemma-3-27b_it/llama_3_1_8b_base",
        "selectors": [
            "turn_boundary:-1", "turn_boundary:-2", "turn_boundary:-3",
            "turn_boundary:-4", "turn_boundary:-5", "task_mean",
        ],
        "layers": [8, 12, 16, 20, 24],
        "utility_base": "results/experiments/character_probes/pre_task_active_learning",
        "utility_glob_pattern": "completion_preference_llama-3.1-8b_completion_canonical_seed0_mra_exp2_split_{split}_*",
    },
    "gptoss": {
        "canonical": "gpt-oss-120b",
        "activations_dir": "activations/gptoss_120b_turn_boundary_sweep",
        "selectors": [
            "turn_boundary:-1", "turn_boundary:-2", "turn_boundary:-3",
            "turn_boundary:-4", "turn_boundary:-5",
        ],
        "layers": [3, 7, 10, 14, 18, 21, 25, 28, 32],
        "utility_base": "results/experiments/gptoss_qwen_mra2/pre_task_active_learning",
        "utility_glob_pattern": "completion_preference_gpt-oss-120b_completion_canonical_seed0_mra_exp2_split_{split}_*",
    },
    "qwen35": {
        "canonical": "qwen3.5-122b-nothink",
        "activations_dir": "activations/qwen35_122b_mra_2500",
        "selectors": [
            "turn_boundary:-1", "turn_boundary:-2", "turn_boundary:-3",
            "turn_boundary:-4", "turn_boundary:-5",
        ],
        "layers": [12, 24, 28, 33, 38, 43],
        "utility_base": "results/experiments/gptoss_qwen_mra2/pre_task_active_learning",
        # Qwen nothink has a sys hash in the dir name — use wildcard
        "utility_glob_pattern": "completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_*mra_exp2_split_{split}_*",
    },
}


def find_run_dir(base: Path, pattern: str, split: str) -> Path:
    """Find the single matching run directory for a split."""
    glob_pat = pattern.format(split=split)
    matches = sorted(base.glob(glob_pat))
    if len(matches) == 0:
        raise FileNotFoundError(f"No match for {base / glob_pat}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {base / glob_pat}: {matches}")
    return matches[0]


def selector_to_safe_name(selector: str) -> str:
    return selector.replace(":", "_").replace("-", "m")


def generate_config(
    act_model_key: str,
    util_model_key: str,
    selector: str,
) -> dict:
    act = MODELS[act_model_key]
    util = MODELS[util_model_key]

    safe_sel = selector_to_safe_name(selector)
    name = f"{act_model_key}_acts_{util_model_key}_utils_{safe_sel}"

    util_base = Path(util["utility_base"])
    run_dir = find_run_dir(util_base, util["utility_glob_pattern"], "a")
    eval_run_dir = find_run_dir(util_base, util["utility_glob_pattern"], "b")

    activations_path = f"{act['activations_dir']}/activations_{selector}.npz"

    return {
        "experiment_name": name,
        "run_dir": str(run_dir),
        "eval_run_dir": str(eval_run_dir),
        "activations_path": activations_path,
        "output_dir": f"results/probes/cross_model/{name}",
        "layers": act["layers"],
        "modes": ["ridge"],
        "standardize": True,
        "alpha_sweep_size": 10,
        "eval_split_seed": 42,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for act_key in MODELS:
        for util_key in MODELS:
            for selector in MODELS[act_key]["selectors"]:
                try:
                    config = generate_config(act_key, util_key, selector)
                except FileNotFoundError as e:
                    print(f"SKIP: {e}")
                    continue

                safe_sel = selector_to_safe_name(selector)
                filename = f"{act_key}_acts_{util_key}_utils_{safe_sel}.yaml"
                path = OUTPUT_DIR / filename
                with open(path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                count += 1

    print(f"Generated {count} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
