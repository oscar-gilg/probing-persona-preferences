"""Generate user-turn × sysprompt stimulus JSONs (truth + harm).

Mirrors `generate_data.py` but filters to `turn == "user"` items. Politics is
assistant-only by stimulus design and is skipped here.

Usage:
    python -m experiments.token_level_probes.system_prompt_modulation_v2.scripts.generate_user_turn
"""
import json
from pathlib import Path

from experiments.token_level_probes.system_prompt_modulation_v2.scripts.generate_data import (
    HARM_SYSTEM_PROMPTS,
    TRUTH_SYSTEM_PROMPTS,
    add_system_prompt,
)

DATA_DIR = Path("experiments/token_level_probes/data")
OUTPUT_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")


def process_domain(input_file: str, system_prompts: dict, output_file: str):
    items = json.load(open(DATA_DIR / input_file))
    user_items = [it for it in items if it["turn"] == "user"]

    variants = []
    for it in user_items:
        for label, text in system_prompts.items():
            variants.append(add_system_prompt(it, label, text))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / output_file, "w") as f:
        json.dump(variants, f, indent=2)
    print(f"{output_file}: {len(variants)} items ({len(user_items)} base × {len(system_prompts)} prompts)")
    return variants


def main():
    truth = process_domain("truth_filtered.json", TRUTH_SYSTEM_PROMPTS, "truth_system_prompts_v2_user_turn.json")
    harm = process_domain("harm_filtered.json", HARM_SYSTEM_PROMPTS, "harm_system_prompts_v2_user_turn.json")
    print(f"\nTotal: {len(truth) + len(harm)} items")


if __name__ == "__main__":
    main()
