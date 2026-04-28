"""Score persona-cross prefills with the tb-5 L32 ridge probe.

For each prefill in {prefills_benign.json, prefills_harmful.json}, score it under each persona
in personas.json. Save per-token probe scores plus three EOT readouts (first_user, asst, user)
to results/scoring_results.json.

Adapts experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_all.py
(persona injection follows generate_data.py::add_system_prompt).

Usage:
    python -m experiments.persona_cross_prefills.scripts.score_prefills [--pilot]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/persona_cross_prefills"
PERSONAS_PATH = EXP_DIR / "personas.json"
PREFILLS_PATHS = [EXP_DIR / "prefills_benign.json", EXP_DIR / "prefills_harmful.json"]
PROBE_PATH = ROOT / "results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy"
OUTPUT_PATH = EXP_DIR / "results/scoring_results.json"
PROBE_LAYER = 32
PROBE_NAME = "tb-5_L32"

EOT_TOK = "<end_of_turn>"
SOT_TOK = "<start_of_turn>"


def find_eot_positions(token_ids: list[int], tokenizer) -> list[tuple[int, str]]:
    """Return [(token_index, preceding_role), ...] for every <end_of_turn> token.

    Walks the token stream tracking the active role via <start_of_turn>{user|model} markers,
    matching the convention in scripts/distress/per_token_analysis.py::label_token_roles.
    """
    eot_id = tokenizer.convert_tokens_to_ids(EOT_TOK)
    sot_id = tokenizer.convert_tokens_to_ids(SOT_TOK)
    user_id = tokenizer.convert_tokens_to_ids("user")
    model_id = tokenizer.convert_tokens_to_ids("model")

    positions = []
    current = "control"
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i] == sot_id and i + 1 < n:
            if token_ids[i + 1] == user_id:
                current = "user"
            elif token_ids[i + 1] == model_id:
                current = "assistant"
            i += 2
            continue
        if token_ids[i] == eot_id:
            positions.append((i, current))
            current = "control"
        i += 1
    return positions


def pick_eot_indices(eot_positions: list[tuple[int, str]]) -> dict[str, int]:
    """Pick first_user_eot, asst_eot, user_eot from the list of (index, role) pairs."""
    user_eots = [idx for idx, role in eot_positions if role == "user"]
    asst_eots = [idx for idx, role in eot_positions if role == "assistant"]
    if not user_eots:
        raise ValueError("No user EOT found")
    if not asst_eots:
        raise ValueError("No assistant EOT found")
    return {
        "first_user_eot": user_eots[0],
        "asst_eot": asst_eots[-1],
        "user_eot": user_eots[-1],
    }


def inject_persona(messages: list[dict], sys_text: str) -> list[dict]:
    """Prepend a system message; empty sys_text means no system message at all (default)."""
    if sys_text:
        return [{"role": "system", "content": sys_text}, *messages]
    return list(messages)


def load_prefills() -> list[dict]:
    items = []
    for path in PREFILLS_PATHS:
        items.extend(json.loads(path.read_text()))
    return items


def load_personas() -> dict[str, str]:
    return json.loads(PERSONAS_PATH.read_text())["personas"]


def score_one(
    model: HuggingFaceModel,
    prefill: dict,
    persona_name: str,
    sys_text: str,
    scoring_probes: list[tuple[int, np.ndarray]],
) -> dict:
    messages = inject_persona(prefill["messages"], sys_text)

    all_scores = score_prompt_all_tokens(model, messages, scoring_probes, add_generation_prompt=False)
    per_token_scores = all_scores[0]

    formatted = model.format_messages(messages, add_generation_prompt=False)
    token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
    tokens = [model.tokenizer.decode(tid) for tid in token_ids]

    eot_positions = find_eot_positions(token_ids, model.tokenizer)
    eot_indices = pick_eot_indices(eot_positions)
    eot_scores = {key: float(per_token_scores[idx]) for key, idx in eot_indices.items()}

    return {
        "prefill_id": prefill["prefill_id"],
        "persona_name": persona_name,
        "pair_id": prefill["pair_id"],
        "condition": prefill["condition"],
        "topic": prefill["topic"],
        "tokens": tokens,
        "per_token_scores": [float(s) for s in per_token_scores],
        "eot_indices": eot_indices,
        "eot_scores": eot_scores,
    }


def run_pilot(model, prefills, personas, scoring_probes):
    print("\n--- PILOT (first 2 prefills × 2 personas) ---")
    for prefill in prefills[:2]:
        for persona_name, sys_text in personas.items():
            print(f"\n{prefill['prefill_id']} × {persona_name}")
            result = score_one(model, prefill, persona_name, sys_text, scoring_probes)
            print(f"  n_tokens={len(result['tokens'])}, eot_indices={result['eot_indices']}")
            print(f"  eot_scores={result['eot_scores']}")
            for key, idx in result["eot_indices"].items():
                preceding = "".join(result["tokens"][max(0, idx - 5):idx + 1])
                print(f"  {key} context: ...{preceding!r}")
            for key in result["eot_scores"]:
                if not np.isfinite(result["eot_scores"][key]):
                    raise ValueError(f"non-finite eot_score[{key}] for {result['prefill_id']}")
    print("\nPILOT PASSED")


def assert_persona_injection(model, prefill, personas):
    """Verify each persona's substring appears in the formatted prompt; default has no system block."""
    print("\n--- Persona-injection sanity ---")
    for persona_name, sys_text in personas.items():
        messages = inject_persona(prefill["messages"], sys_text)
        formatted = model.format_messages(messages, add_generation_prompt=False)
        if sys_text:
            substr = sys_text[:40]
            assert substr in formatted, f"persona '{persona_name}' substring not in formatted prompt"
            print(f"  {persona_name}: persona substring present")
        else:
            assert "system" not in formatted.split("<start_of_turn>")[1], (
                f"default persona unexpectedly has a system block"
            )
            print(f"  {persona_name}: no system block (as expected)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Pilot only, do not run full set")
    args = parser.parse_args()

    print("Loading model: google/gemma-3-27b-it")
    model = HuggingFaceModel("google/gemma-3-27b-it")

    print(f"Loading probe: {PROBE_PATH}")
    probe_weights = np.load(PROBE_PATH)
    scoring_probes = [(PROBE_LAYER, probe_weights)]

    print("Loading prefills + personas")
    prefills = load_prefills()
    personas = load_personas()
    print(f"  {len(prefills)} prefills × {len(personas)} personas = {len(prefills) * len(personas)} forward passes")

    assert_persona_injection(model, prefills[0], personas)

    run_pilot(model, prefills, personas, scoring_probes)
    if args.pilot:
        print("\n--pilot set, exiting before full run")
        return

    print("\n--- FULL SCORING ---")
    items = []
    for prefill in tqdm(prefills, desc="prefills"):
        for persona_name, sys_text in personas.items():
            items.append(score_one(model, prefill, persona_name, sys_text, scoring_probes))

    output = {
        "items": items,
        "probe_config": {
            "name": PROBE_NAME,
            "layer": PROBE_LAYER,
            "path": str(PROBE_PATH.relative_to(ROOT)),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output))
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nWrote {len(items)} items to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
