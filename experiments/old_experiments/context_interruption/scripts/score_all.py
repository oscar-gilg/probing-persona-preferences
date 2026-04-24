"""Score all context interruption stimuli.

For each stimulus: generates response to the task, then response to the
follow-up, constructs the full 5-message conversation with interruption,
then scores all tokens with preference probes.

Checkpoints every 20 items and supports resume.

Usage:
    python experiments/context_interruption/scripts/score_all.py
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens

DATA_DIR = Path("experiments/context_interruption/data")
STIMULI_PATH = DATA_DIR / "stimuli.json"
OUTPUT_PATH = DATA_DIR / "scoring_results.json"

PROBE_SETS = {
    "tb-2": Path("results/probes/heldout_eval_gemma3_tb-2/probes"),
    "tb-5": Path("results/probes/heldout_eval_gemma3_tb-5/probes"),
    "task_mean": Path("results/probes/heldout_eval_gemma3_task_mean/probes"),
}
LAYERS = [32, 39, 53]

CHECKPOINT_EVERY = 20


def load_probes() -> tuple[list[tuple[str, int, np.ndarray]], list[tuple[int, np.ndarray]]]:
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
    with open(STIMULI_PATH) as f:
        return json.load(f)


def generate_responses(model: HuggingFaceModel, stimulus: dict) -> tuple[str, str]:
    """Generate responses to the task and the follow-up."""
    # Response to initial task
    messages_1 = [{"role": "user", "content": stimulus["task_prompt"]}]
    response_1 = model.generate(messages_1, temperature=0.7, max_new_tokens=256)

    # Response to follow-up (in context of task + response)
    messages_2 = [
        {"role": "user", "content": stimulus["task_prompt"]},
        {"role": "assistant", "content": response_1},
        {"role": "user", "content": stimulus["follow_up_text"]},
    ]
    response_2 = model.generate(messages_2, temperature=0.7, max_new_tokens=256)

    return response_1, response_2


SEGMENT_NAMES = ["user_1", "assistant_1", "user_2", "assistant_2", "interruption"]


def get_segment_boundaries(
    model: HuggingFaceModel,
    full_messages: list[dict],
) -> dict[str, list[int]]:
    """Get token index ranges [start, end) for each conversation segment.

    Each segment includes its chat template markers (e.g. <start_of_turn>).
    The final "generation_prompt" segment covers the assistant turn marker
    appended by add_generation_prompt=True.
    """
    segments: dict[str, list[int]] = {}
    prev_len = 0

    for i in range(1, len(full_messages) + 1):
        formatted = model.format_messages(full_messages[:i], add_generation_prompt=False)
        token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
        curr_len = len(token_ids)
        segments[SEGMENT_NAMES[i - 1]] = [prev_len, curr_len]
        prev_len = curr_len

    formatted_full = model.format_messages(full_messages, add_generation_prompt=True)
    token_ids_full = model.tokenizer(formatted_full, add_special_tokens=False)["input_ids"]
    segments["generation_prompt"] = [prev_len, len(token_ids_full)]

    return segments


def score_stimulus(
    model: HuggingFaceModel,
    stimulus: dict,
    response_1: str,
    response_2: str,
    named_probes: list[tuple[str, int, np.ndarray]],
    scoring_probes: list[tuple[int, np.ndarray]],
) -> dict:
    """Score all tokens in the full conversation including interruption."""
    full_messages = [
        {"role": "user", "content": stimulus["task_prompt"]},
        {"role": "assistant", "content": response_1},
        {"role": "user", "content": stimulus["follow_up_text"]},
        {"role": "assistant", "content": response_2},
        {"role": "user", "content": stimulus["interruption_text"]},
    ]

    all_scores = score_prompt_all_tokens(
        model, full_messages, scoring_probes, add_generation_prompt=True,
    )

    formatted = model.format_messages(full_messages, add_generation_prompt=True)
    token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
    tokens = [model.tokenizer.decode(tid) for tid in token_ids]

    segments = get_segment_boundaries(model, full_messages)
    assert segments["generation_prompt"][1] == len(tokens), (
        f"Segment boundary mismatch: last segment ends at "
        f"{segments['generation_prompt'][1]} but got {len(tokens)} tokens"
    )

    all_token_scores = {}
    for i, (key, _layer, _weights) in enumerate(named_probes):
        all_token_scores[key] = all_scores[i].tolist()

    return {
        "id": stimulus["id"],
        "prompt_type": stimulus["prompt_type"],
        "session_valence": stimulus["session_valence"],
        "session_topic": stimulus["session_topic"],
        "offered_valence": stimulus["offered_valence"],
        "offered_topic": stimulus["offered_topic"],
        "task_id": stimulus["task_id"],
        "task_mu": stimulus["task_mu"],
        "response_1": response_1,
        "response_2": response_2,
        "segments": segments,
        "all_token_scores": all_token_scores,
        "tokens": tokens,
        "n_tokens": len(tokens),
    }


def run_pilot(
    model: HuggingFaceModel,
    stimuli: list[dict],
    named_probes: list[tuple[str, int, np.ndarray]],
    scoring_probes: list[tuple[int, np.ndarray]],
) -> None:
    """Validate on 2 items (one pleasant, one unpleasant) before full scoring."""
    pilot_items = []
    for s in stimuli:
        if s["session_valence"] == "pleasant" and not pilot_items:
            pilot_items.append(s)
        elif s["session_valence"] == "unpleasant" and len(pilot_items) == 1:
            pilot_items.append(s)
        if len(pilot_items) == 2:
            break

    for stimulus in pilot_items:
        print(f"\n{'='*60}")
        print(f"PILOT: {stimulus['id']}")
        print(f"Session: {stimulus['session_topic']} / {stimulus['session_valence']}")
        print(f"Task: {stimulus['task_prompt'][:80]}...")
        print(f"Follow-up: {stimulus['follow_up_text'][:80]}...")
        print(f"Interruption: {stimulus['interruption_text'][:80]}...")

        response_1, response_2 = generate_responses(model, stimulus)
        print(f"Response 1 ({len(response_1)} chars): {response_1[:100]}...")
        print(f"Response 2 ({len(response_2)} chars): {response_2[:100]}...")

        result = score_stimulus(
            model, stimulus, response_1, response_2, named_probes, scoring_probes,
        )

        first_probe = list(result["all_token_scores"].keys())[0]
        scores_arr = result["all_token_scores"][first_probe]
        print(f"Tokens: {result['n_tokens']}")
        print(f"Score range ({first_probe}): [{min(scores_arr):.4f}, {max(scores_arr):.4f}]")

        if any(np.isnan(s) for s in scores_arr):
            raise ValueError(f"NaN scores detected for {stimulus['id']}")

    print(f"\n{'='*60}")
    print("PILOT PASSED — all items scored successfully")


def load_checkpoint() -> tuple[list[dict], set[str]]:
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        scored_ids = {item["id"] for item in data["items"]}
        print(f"Resuming: {len(scored_ids)} items already scored")
        return data["items"], scored_ids
    return [], set()


def save_checkpoint(results: list[dict]) -> None:
    output = {
        "items": results,
        "probe_configs": {
            f"{name}_L{layer}": {
                "probe_set": name,
                "layer": layer,
                "path": str(PROBE_SETS[name] / f"probe_ridge_L{layer}.npy"),
            }
            for name in PROBE_SETS
            for layer in LAYERS
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))


def main():
    print("Loading model...")
    model = HuggingFaceModel("google/gemma-3-27b-it")

    print("Loading probes...")
    named_probes, scoring_probes = load_probes()
    print(f"Loaded {len(named_probes)} probes: {[p[0] for p in named_probes]}")

    print("Loading stimuli...")
    stimuli = load_stimuli()
    print(f"Loaded {len(stimuli)} stimuli")

    results, scored_ids = load_checkpoint()
    remaining = [s for s in stimuli if s["id"] not in scored_ids]

    if not scored_ids:
        print("\n--- PILOT ---")
        run_pilot(model, stimuli, named_probes, scoring_probes)

    print(f"\n--- SCORING {len(remaining)} ITEMS ---")
    for i, stimulus in enumerate(tqdm(remaining, desc="Scoring")):
        response_1, response_2 = generate_responses(model, stimulus)
        result = score_stimulus(
            model, stimulus, response_1, response_2, named_probes, scoring_probes,
        )
        results.append(result)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(results)
            print(f"  Checkpoint: {len(results)} items saved")

    save_checkpoint(results)
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(results)} items to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")

    if file_size_mb > 20:
        print("WARNING: File exceeds 20MB. Consider splitting all_token_scores to .npz.")


if __name__ == "__main__":
    main()
