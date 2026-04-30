"""Score aura sysprompt user-turn items (Gemma-3-27B IT) with preference probes.

Methodologically identical to `score_user_turn.py` (same probes, same selectors,
same `add_generation_prompt=True`). Loads only the aura stimuli and writes to
a separate JSON file so the existing scoring_results_user_turn.json is untouched.

Usage:
    python -m experiments.token_level_probes.system_prompt_modulation_v2.scripts.score_aura_user_turn
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens
from src.steering.tokenization import find_text_span

DATA_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")
OUTPUT_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/scoring_results_user_turn_aura.json")

PROBE_SETS = {
    "tb-2": Path("results/probes/heldout_eval_gemma3_tb-2/probes"),
    "tb-5": Path("results/probes/heldout_eval_gemma3_tb-5/probes"),
    "task_mean": Path("results/probes/heldout_eval_gemma3_task_mean/probes"),
}
LAYERS = [32, 39, 53]


def load_probes():
    named_probes = []
    scoring_probes = []
    for probe_name, probe_dir in PROBE_SETS.items():
        for layer in LAYERS:
            weights = np.load(probe_dir / f"probe_ridge_L{layer}.npy")
            key = f"{probe_name}_L{layer}"
            named_probes.append((key, layer, weights))
            scoring_probes.append((layer, weights))
    return named_probes, scoring_probes


def load_stimuli() -> list[dict]:
    items = []
    for filename in ["truth_aura_user_turn.json", "harm_aura_user_turn.json"]:
        items.extend(json.load(open(DATA_DIR / filename)))
    return items


def find_fullstop_indices(tokens: list[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if "." in tok]


def score_item(model, item, named_probes, scoring_probes):
    messages = item["messages"]

    all_scores = score_prompt_all_tokens(
        model, messages, scoring_probes, add_generation_prompt=True,
    )

    formatted = model.format_messages(messages, add_generation_prompt=True)
    token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
    tokens = [model.tokenizer.decode(tid) for tid in token_ids]

    critical_span = item["critical_span"]
    span_start, span_end = find_text_span(model.tokenizer, formatted, critical_span)
    fullstop_indices = find_fullstop_indices(tokens)

    critical_span_mean_scores = {}
    fullstop_scores = {}
    eot_scores = {}
    for i, (key, _layer, _weights) in enumerate(named_probes):
        scores_arr = all_scores[i]
        critical_span_mean_scores[key] = float(np.mean(scores_arr[span_start:span_end]))
        fullstop_scores[key] = [float(scores_arr[idx]) for idx in fullstop_indices]
        eot_scores[key] = float(scores_arr[-1])

    return {
        "id": item["id"],
        "domain": item["domain"],
        "turn": item["turn"],
        "condition": item["condition"],
        "system_prompt": item["system_prompt"],
        "critical_span": critical_span,
        "critical_token_indices": list(range(span_start, span_end)),
        "fullstop_indices": fullstop_indices,
        "critical_span_mean_scores": critical_span_mean_scores,
        "eot_scores": eot_scores,
        "fullstop_scores": fullstop_scores,
        "tokens": tokens,
        **{k: item[k] for k in ["source_id", "issue"] if k in item},
    }


def run_pilot(model, items, named_probes, scoring_probes):
    for item in items[:3]:
        print(f"\n{'='*60}")
        print(f"PILOT: {item['id']}")
        print(f"Critical span: '{item['critical_span']}'")
        result = score_item(model, item, named_probes, scoring_probes)
        critical_tokens_text = "".join(result["tokens"][i] for i in result["critical_token_indices"])
        print(f"Critical tokens: '{critical_tokens_text}'")
        print(f"Last 6 tokens: {result['tokens'][-6:]}")
        print(f"EOT score (tb-5_L32): {result['eot_scores']['tb-5_L32']:.4f}")
        print(f"EOT score (tb-5_L39): {result['eot_scores']['tb-5_L39']:.4f}")
    print(f"\n{'='*60}")
    print("PILOT PASSED")


def main():
    print("Loading model...")
    model = HuggingFaceModel("google/gemma-3-27b-it")

    print("Loading probes...")
    named_probes, scoring_probes = load_probes()
    print(f"Loaded {len(named_probes)} probes")

    print("Loading stimuli...")
    items = load_stimuli()
    print(f"Loaded {len(items)} aura items")

    print("\n--- PILOT ---")
    run_pilot(model, items, named_probes, scoring_probes)

    print("\n--- SCORING ALL AURA ITEMS ---")
    results = []
    for item in tqdm(items, desc="Scoring"):
        results.append(score_item(model, item, named_probes, scoring_probes))

    output = {
        "items": results,
        "probe_configs": {
            f"{name}_L{layer}": {"probe_set": name, "layer": layer, "path": str(PROBE_SETS[name] / f"probe_ridge_L{layer}.npy")}
            for name in PROBE_SETS for layer in LAYERS
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(results)} aura items to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
