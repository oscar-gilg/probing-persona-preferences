"""Build steering_pairs.json for the layer sweep experiment.

Samples 50 pairs from default_test, stratified across the cross-product of
task origins, with utility_gap > 0.1. Orients each pair so task_a has the
higher utility. Computes tokenised task spans under the same prompt builder +
template the steering runner uses.
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
N_PAIRS = 50

DEFAULT_TEST_DIR = Path(
    "results/experiments/persona_sweep_final_six/pre_task_active_learning/default_test"
)
COMPLETIONS_JSON = Path(
    "activations/gemma-3-27b_it/pref_main/completions_with_activations.json"
)
TEMPLATE_PATH = Path(
    "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
)
OUTPUT_PATH = Path("experiments/layer_sweep/steering_pairs.json")
TOKENIZER_NAME = "google/gemma-3-27b-it"


def _load_task_metadata() -> dict[str, dict]:
    """task_id -> {prompt, origin}. origin is an uppercase string matching OriginDataset."""
    with open(COMPLETIONS_JSON) as f:
        records = json.load(f)
    out: dict[str, dict] = {}
    for r in records:
        out[r["task_id"]] = {"prompt": r["task_prompt"], "origin": r["origin"]}
    return out


def _stratified_sample(
    pool: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Stratify across (task_a_origin, task_b_origin) cross-product; round-robin per stratum."""
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pool:
        by_stratum[(p["task_a_origin"], p["task_b_origin"])].append(p)

    rng = random.Random(seed)
    for lst in by_stratum.values():
        rng.shuffle(lst)

    strata = sorted(by_stratum.keys())
    picked: list[dict] = []
    cursors = {s: 0 for s in strata}
    while len(picked) < n:
        progress = False
        for s in strata:
            if len(picked) >= n:
                break
            lst = by_stratum[s]
            idx = cursors[s]
            if idx < len(lst):
                picked.append(lst[idx])
                cursors[s] = idx + 1
                progress = True
        if not progress:
            raise RuntimeError(
                f"Pool exhausted at {len(picked)} pairs; wanted {n}. "
                f"Strata: {[(s, len(by_stratum[s])) for s in strata]}"
            )
    return picked


def _origin_enum(origin_str: str) -> OriginDataset:
    return OriginDataset[origin_str]


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
            pool.append({
                "task_a": a_id,
                "task_b": b_id,
                "task_a_origin": task_meta[a_id]["origin"],
                "task_b_origin": task_meta[b_id]["origin"],
                "utility_a": a_u,
                "utility_b": b_u,
                "utility_gap": gap,
            })
    print(f"  Pool size after utility_gap>{UTILITY_GAP_MIN}: {len(pool)}")

    picked = _stratified_sample(pool, N_PAIRS, SEED)
    print(f"  Sampled {len(picked)} pairs across {len(set((p['task_a_origin'], p['task_b_origin']) for p in picked))} strata")

    print(f"Loading tokenizer {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    template = load_templates_from_yaml(TEMPLATE_PATH)[0]
    builder = build_revealed_builder(template, "completion")

    out_pairs: list[dict] = []
    for idx, entry in enumerate(picked):
        a_text = task_meta[entry["task_a"]]["prompt"]
        b_text = task_meta[entry["task_b"]]["prompt"]

        task_a = Task(
            prompt=a_text,
            origin=_origin_enum(entry["task_a_origin"]),
            id=entry["task_a"],
            metadata={},
        )
        task_b = Task(
            prompt=b_text,
            origin=_origin_enum(entry["task_b_origin"]),
            id=entry["task_b"],
            metadata={},
        )
        prompt_data = builder.build(task_a, task_b)
        # Match exactly how the runner formats: apply the chat template with generation prompt.
        formatted = tokenizer.apply_chat_template(
            prompt_data.messages, tokenize=False, add_generation_prompt=True,
        )
        (a_start, a_end), (b_start, b_end) = find_pairwise_task_spans(
            tokenizer, formatted, a_text, b_text, "Task A", "Task B",
        )
        assert a_start is not None and a_end is not None
        assert b_start is not None and b_end is not None
        assert a_end > a_start and b_end > b_start, f"Empty span for pair {idx}"
        assert a_end <= b_start or b_end <= a_start, f"Overlapping spans for pair {idx}: A=[{a_start},{a_end}) B=[{b_start},{b_end})"

        pair = {
            "pair_id": f"{idx:04d}",
            "task_a": entry["task_a"],
            "task_b": entry["task_b"],
            "task_a_text": a_text,
            "task_b_text": b_text,
            "task_a_origin": entry["task_a_origin"],
            "task_b_origin": entry["task_b_origin"],
            "utility_a": entry["utility_a"],
            "utility_b": entry["utility_b"],
            "utility_gap": entry["utility_gap"],
            # delta_mu kept for the runner's existing _make_row contract.
            "delta_mu": entry["utility_gap"],
            "task_a_tokens": a_end - a_start,
            "task_b_tokens": b_end - b_start,
        }
        out_pairs.append(pair)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out_pairs, f, indent=2)

    print(f"\nWrote {len(out_pairs)} pairs to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
