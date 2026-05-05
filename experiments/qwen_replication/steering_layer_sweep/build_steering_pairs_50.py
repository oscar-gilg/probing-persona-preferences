"""Build steering_pairs_50.json for the Qwen layer-sweep Phase A.

Same rule as experiments/layer_sweep/build_steering_pairs.py, with two changes:
  1. Restrict to the disjoint pool: tasks in canonical_test that are NOT in the
     Qwen 10k AL probe-train set or the 4k AL alpha-selection set. This keeps
     the steering measurement on tasks the 10k probes never saw during fit.
  2. Use Qwen's default_test Thurstonian utilities, the Qwen completions JSON
     (origins + prompts), and the Qwen tokenizer.
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
    "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning/"
    "completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d_test_task_ids"
)
COMPLETIONS_JSON = Path("activations/qwen35_122b/pref_default_sweep/completions_with_activations.json")
TEMPLATE_PATH = Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")
OUTPUT_PATH = Path("experiments/qwen_replication/steering_layer_sweep/steering_pairs_50.json")
import os

# Build-time span sanity uses the cached Qwen3-32B tokenizer (same BPE vocab as
# Qwen3.5-122B-A10B). The pod's runtime tokenizer is the source of truth.
# Resolve to the cached snapshot directory directly so HF_HUB_OFFLINE doesn't
# block the API check.
_HUB = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
_SNAPSHOTS = _HUB / "models--Qwen--Qwen3-32B" / "snapshots"
TOKENIZER_NAME = str(next(_SNAPSHOTS.iterdir())) if _SNAPSHOTS.exists() else "Qwen/Qwen3-32B"

TEN_K_AL = Path("configs/extraction/qwen35_10k_task_ids.txt")
FOUR_K_AL = Path("configs/extraction/qwen35_4k_task_ids.txt")


def _load_disjoint_pool() -> set[str]:
    leak = set(TEN_K_AL.read_text().splitlines()) | set(FOUR_K_AL.read_text().splitlines())
    canon_test = set(Path("data/canonical_splits/test_task_ids.txt").read_text().splitlines())
    return canon_test - leak


def _load_task_metadata() -> dict[str, dict]:
    with open(COMPLETIONS_JSON) as f:
        records = json.load(f)
    return {r["task_id"]: {"prompt": r["task_prompt"], "origin": r["origin"]} for r in records}


def _stratified_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
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


def main() -> None:
    disjoint = _load_disjoint_pool()
    print(f"Disjoint pool size: {len(disjoint)}")

    print(f"Loading utilities from {DEFAULT_TEST_DIR}")
    mu_array, task_ids = load_run_utilities(DEFAULT_TEST_DIR)
    utilities = dict(zip(task_ids, mu_array.tolist()))
    print(f"  {len(utilities)} utilities total")

    print(f"Loading task metadata from {COMPLETIONS_JSON}")
    task_meta = _load_task_metadata()

    safe_ids = sorted(tid for tid in task_ids if tid in disjoint and tid in task_meta)
    print(f"  {len(safe_ids)} tasks in disjoint pool with utilities + metadata")
    if len(safe_ids) < 100:
        raise RuntimeError(f"Disjoint pool too small ({len(safe_ids)}) for sampling")

    pool: list[dict] = []
    for i in range(len(safe_ids)):
        ti = safe_ids[i]
        ui = utilities[ti]
        for j in range(i + 1, len(safe_ids)):
            tj = safe_ids[j]
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
    print(f"  Pool after utility_gap>{UTILITY_GAP_MIN}: {len(pool)}")

    picked = _stratified_sample(pool, N_PAIRS, SEED)
    print(
        f"  Sampled {len(picked)} pairs across "
        f"{len(set((p['task_a_origin'], p['task_b_origin']) for p in picked))} strata"
    )

    print(f"Loading tokenizer {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
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
        assert a_start is not None and a_end is not None, f"Pair {idx}: span A is None"
        assert b_start is not None and b_end is not None, f"Pair {idx}: span B is None"
        assert a_end > a_start and b_end > b_start, f"Pair {idx}: empty span"
        assert a_end <= b_start or b_end <= a_start, f"Pair {idx}: overlapping spans"

        out_pairs.append({
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
