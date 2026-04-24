"""Build steering_pairs_150.json: 50 benign-benign + 50 harmful-benign + 50 harmful-harmful pairs.

Source: default_test utilities (1000 tasks), same as build_steering_pairs.py. Filter
utility_gap > 0.1, orient so task_a has higher utility. Stratify by pair-type, not
origin cross-product. Adds pair_type field ("bb"/"hb"/"hh") to each pair.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.measurement.storage.loading import load_run_utilities
from src.steering.tokenization import find_pairwise_task_spans
from src.task_data import OriginDataset, Task


load_dotenv()

SEED = 42
UTILITY_GAP_MIN = 0.1
PER_TYPE = 50

BENIGN_ORIGINS = {"WILDCHAT", "ALPACA", "MATH"}
HARMFUL_ORIGINS = {"BAILBENCH", "STRESS_TEST"}

DEFAULT_TEST_DIR = Path(
    "results/experiments/persona_sweep_final_six/pre_task_active_learning/default_test"
)
COMPLETIONS_JSON = Path(
    "activations/gemma-3-27b_it/pref_layer_sweep/completions_with_activations.json"
)
TEMPLATE_PATH = Path(
    "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
)
OUTPUT_PATH = Path("experiments/layer_sweep/harm_breakdown/steering_pairs_150.json")
TOKENIZER_NAME = "google/gemma-3-27b-it"


def _pair_type(origin_a: str, origin_b: str) -> str:
    a_h = origin_a in HARMFUL_ORIGINS
    b_h = origin_b in HARMFUL_ORIGINS
    if not a_h and not b_h:
        return "bb"
    if a_h and b_h:
        return "hh"
    return "hb"


def _load_task_metadata() -> dict[str, dict]:
    with open(COMPLETIONS_JSON) as f:
        records = json.load(f)
    return {r["task_id"]: {"prompt": r["task_prompt"], "origin": r["origin"]} for r in records}


def _stratified_sample_by_pair_type(pool: list[dict], per_type: int, seed: int) -> list[dict]:
    """Pick `per_type` pairs from each of bb/hb/hh. Within each type, further stratify
    by (origin_a, origin_b) cross-product and round-robin to avoid one origin dominating."""
    by_type: dict[str, dict[tuple[str, str], list[dict]]] = {"bb": defaultdict(list), "hb": defaultdict(list), "hh": defaultdict(list)}
    for p in pool:
        by_type[p["pair_type"]][(p["task_a_origin"], p["task_b_origin"])].append(p)

    rng = random.Random(seed)
    picked: list[dict] = []
    for pair_type in ["bb", "hb", "hh"]:
        strata = sorted(by_type[pair_type].keys())
        for s in strata:
            rng.shuffle(by_type[pair_type][s])
        cursors = {s: 0 for s in strata}
        type_picked: list[dict] = []
        while len(type_picked) < per_type:
            progress = False
            for s in strata:
                if len(type_picked) >= per_type:
                    break
                lst = by_type[pair_type][s]
                idx = cursors[s]
                if idx < len(lst):
                    type_picked.append(lst[idx])
                    cursors[s] = idx + 1
                    progress = True
            if not progress:
                raise RuntimeError(
                    f"Pool exhausted for pair_type={pair_type} at {len(type_picked)}/{per_type}. "
                    f"Strata sizes: {[(s, len(by_type[pair_type][s])) for s in strata]}"
                )
        picked.extend(type_picked)
    return picked


def main() -> None:
    print(f"Loading utilities from {DEFAULT_TEST_DIR}")
    mu_array, task_ids = load_run_utilities(DEFAULT_TEST_DIR)
    utilities = dict(zip(task_ids, mu_array.tolist()))
    print(f"  {len(utilities)} utilities")

    print(f"Loading task metadata from {COMPLETIONS_JSON}")
    task_meta = _load_task_metadata()

    missing = [tid for tid in task_ids if tid not in task_meta]
    if missing:
        raise RuntimeError(f"{len(missing)} test task_ids missing from completions JSON, e.g. {missing[:3]}")

    sorted_ids = sorted(task_ids)
    pool: list[dict] = []
    for i in range(len(sorted_ids)):
        ti = sorted_ids[i]
        ui = utilities[ti]
        for j in range(i + 1, len(sorted_ids)):
            tj = sorted_ids[j]
            uj = utilities[tj]
            gap = abs(ui - uj)
            if gap <= UTILITY_GAP_MIN:
                continue
            if ui >= uj:
                a_id, b_id, a_u, b_u = ti, tj, ui, uj
            else:
                a_id, b_id, a_u, b_u = tj, ti, uj, ui
            origin_a = task_meta[a_id]["origin"]
            origin_b = task_meta[b_id]["origin"]
            pool.append({
                "task_a": a_id,
                "task_b": b_id,
                "task_a_origin": origin_a,
                "task_b_origin": origin_b,
                "utility_a": a_u,
                "utility_b": b_u,
                "utility_gap": gap,
                "pair_type": _pair_type(origin_a, origin_b),
            })

    type_counts = defaultdict(int)
    for p in pool:
        type_counts[p["pair_type"]] += 1
    print(f"  Pool after utility_gap>{UTILITY_GAP_MIN}: {len(pool)} (bb={type_counts['bb']}, hb={type_counts['hb']}, hh={type_counts['hh']})")

    picked = _stratified_sample_by_pair_type(pool, PER_TYPE, SEED)
    picked_counts = defaultdict(int)
    for p in picked:
        picked_counts[p["pair_type"]] += 1
    print(f"  Picked: bb={picked_counts['bb']}, hb={picked_counts['hb']}, hh={picked_counts['hh']}")
    assert picked_counts["bb"] == PER_TYPE
    assert picked_counts["hb"] == PER_TYPE
    assert picked_counts["hh"] == PER_TYPE

    print(f"Loading tokenizer {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")

    out_pairs: list[dict] = []
    for idx, entry in enumerate(picked):
        a_text = task_meta[entry["task_a"]]["prompt"].strip()
        b_text = task_meta[entry["task_b"]]["prompt"].strip()

        task_a = Task(prompt=a_text, origin=OriginDataset[entry["task_a_origin"]], id=entry["task_a"], metadata={})
        task_b = Task(prompt=b_text, origin=OriginDataset[entry["task_b_origin"]], id=entry["task_b"], metadata={})
        prompt_data = builder.build(task_a, task_b)
        formatted = tokenizer.apply_chat_template(
            prompt_data.messages, tokenize=False, add_generation_prompt=True,
        )
        (a_start, a_end), (b_start, b_end) = find_pairwise_task_spans(
            tokenizer, formatted, a_text, b_text, "Task A", "Task B",
        )
        assert a_start is not None and a_end is not None
        assert b_start is not None and b_end is not None
        assert a_end > a_start and b_end > b_start, f"Empty span for pair {idx}"
        assert a_end <= b_start or b_end <= a_start, f"Overlapping spans for pair {idx}"

        out_pairs.append({
            "pair_id": f"{idx:04d}",
            "task_a": entry["task_a"],
            "task_b": entry["task_b"],
            "task_a_text": a_text,
            "task_b_text": b_text,
            "task_a_origin": entry["task_a_origin"],
            "task_b_origin": entry["task_b_origin"],
            "pair_type": entry["pair_type"],
            "utility_a": entry["utility_a"],
            "utility_b": entry["utility_b"],
            "utility_gap": entry["utility_gap"],
            "delta_mu": entry["utility_gap"],
            "task_a_tokens": a_end - a_start,
            "task_b_tokens": b_end - b_start,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out_pairs, f, indent=2)
    print(f"\nWrote {len(out_pairs)} pairs to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
