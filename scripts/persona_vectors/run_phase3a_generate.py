"""Phase 3a: generate completions via OpenRouter (no GPU required).

For each (persona x contrast_pair x polarity), generate 400 completions
(40 unique inputs x 10 rollouts) on `qwen3.5-122b-nothink` via the OpenRouter
API at temperature 1.0, max_new_tokens 512. Saves to
`experiments/qwen_persona_vectors/completions/<persona>__pair{i}__{pol}.jsonl`.

Each line is a dict with: task_id, task_prompt, completion, system_prompt,
input_kind, input_idx, rollout, ok, error.

These JSONLs are then consumed by Phase 3b (GPU extraction) which forward-
passes (system + user + assistant) through the local model and captures
response-token-mean activations.

Run modes:
- `--pilot`           — sadist pair0 pos only, 5 inputs x 2 rollouts = 10 gens (~30s, $0.01)
- `--persona NAME`    — all 10 configs (5 pairs x 2 polarities) for one persona
- (no flag)           — full 24k generations across 6 personas

Resume is per-config via task_id dedup against the existing JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.models import GenerateRequest, get_client
from src.persona_vectors import PersonaArtifacts, format_pairwise_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/artifacts"
EXTRACTION_TASKS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/extraction_tasks"
COMPLETIONS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/completions"

CANONICAL_SIX = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]

MODEL = "qwen3.5-122b-nothink"
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
CONCURRENCY = 50
DEFAULT_N_ROLLOUTS = 5  # halved from Chen's 10 — enough when trait filter rate is high

PILOT_PERSONA = "sadist"
PILOT_PAIR_IDX = 0
PILOT_POLARITY = "pos"
PILOT_N_INPUTS = 5
PILOT_N_ROLLOUTS = 2


def load_existing_task_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    out: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["task_id"])
    return out


def load_tasks(persona: str) -> list[dict]:
    path = EXTRACTION_TASKS_DIR / f"{persona}.json"
    return json.loads(path.read_text())


def filter_pilot_tasks(tasks: list[dict]) -> list[dict]:
    """Keep only first PILOT_N_INPUTS auto-questions x first PILOT_N_ROLLOUTS rollouts."""
    out = []
    for t in tasks:
        if t["input_kind"] != "auto":
            continue
        if int(t["input_idx"]) >= PILOT_N_INPUTS:
            continue
        if t["rollout"] >= PILOT_N_ROLLOUTS:
            continue
        out.append(t)
    return out


def filter_n_rollouts(tasks: list[dict], n: int) -> list[dict]:
    """Keep only the first `n` rollouts per (input_kind, input_idx)."""
    return [t for t in tasks if t["rollout"] < n]


async def run_one_config(
    persona: str,
    pair_idx: int,
    polarity: str,
    artifacts: PersonaArtifacts,
    tasks: list[dict],
    client,
) -> tuple[int, int]:
    """Returns (n_new_completions, n_failures)."""
    out_path = COMPLETIONS_DIR / f"{persona}__pair{pair_idx}__{polarity}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_task_ids(out_path)
    pending = [t for t in tasks if t["task_id"] not in existing]
    if not pending:
        print(f"  all {len(tasks)} tasks already complete")
        return 0, 0

    pos_prompts = artifacts.positive_system_prompts(append_no_think=True)
    neg_prompts = artifacts.negative_system_prompts(append_no_think=True)
    sys_prompt = (pos_prompts if polarity == "pos" else neg_prompts)[pair_idx]

    requests = [
        GenerateRequest(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": t["prompt"]},
            ],
            temperature=TEMPERATURE,
        )
        for t in pending
    ]
    semaphore = asyncio.Semaphore(CONCURRENCY)
    print(f"  pending={len(pending)} (existing={len(existing)})  concurrency={CONCURRENCY}")
    t0 = time.time()
    results = await client.generate_batch_async(
        requests=requests,
        semaphore=semaphore,
        enable_reasoning=False,
    )
    elapsed = time.time() - t0
    print(f"  batch returned in {elapsed:.1f}s ({len(pending) / max(elapsed, 1e-3):.1f} req/s)")

    n_ok = 0
    n_fail = 0
    with open(out_path, "a") as f:
        for task, result in zip(pending, results, strict=True):
            row = {
                "task_id": task["task_id"],
                "task_prompt": task["prompt"],
                "input_kind": task["input_kind"],
                "input_idx": task["input_idx"],
                "rollout": task["rollout"],
                "system_prompt": sys_prompt,
                "completion": result.response if result.ok else None,
                "ok": result.ok,
                "error": (result.error_details() if not result.ok else None),
            }
            f.write(json.dumps(row) + "\n")
            if result.ok:
                n_ok += 1
            else:
                n_fail += 1

    print(f"  wrote {n_ok} ok, {n_fail} failures -> {out_path.name}")
    return n_ok, n_fail


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run only sadist pair0 pos with 10 generations")
    parser.add_argument("--persona", choices=CANONICAL_SIX, help="Restrict to one persona (all 10 configs)")
    parser.add_argument(
        "--n-rollouts", type=int, default=DEFAULT_N_ROLLOUTS,
        help=f"Rollouts per (input, system_prompt). Default {DEFAULT_N_ROLLOUTS}. "
             f"Tasks file holds 10 per input; we slice to the first n.",
    )
    args = parser.parse_args()

    load_dotenv()
    COMPLETIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        personas = [PILOT_PERSONA]
        pair_indices = [PILOT_PAIR_IDX]
        polarities = [PILOT_POLARITY]
    elif args.persona:
        personas = [args.persona]
        pair_indices = list(range(5))
        polarities = ["pos", "neg"]
    else:
        personas = CANONICAL_SIX
        pair_indices = list(range(5))
        polarities = ["pos", "neg"]

    client = get_client(model_name=MODEL, max_new_tokens=MAX_NEW_TOKENS)
    print(f"client ready: {type(client).__name__} model={MODEL}")

    total_ok = 0
    total_fail = 0
    t0 = time.time()
    for persona in personas:
        artifacts = PersonaArtifacts.load(ARTIFACTS_DIR / f"{persona}.json")
        all_tasks = load_tasks(persona)
        if args.pilot:
            tasks = filter_pilot_tasks(all_tasks)
            print(f"\n[{persona}] PILOT: {len(tasks)} tasks (first {PILOT_N_INPUTS} auto inputs x {PILOT_N_ROLLOUTS} rollouts)")
        else:
            tasks = filter_n_rollouts(all_tasks, args.n_rollouts)
            print(f"\n[{persona}] {len(tasks)} tasks (n_rollouts={args.n_rollouts})")
        for pair_idx in pair_indices:
            for polarity in polarities:
                tag = f"{persona}__pair{pair_idx}__{polarity}"
                print(f"\n[{tag}]")
                n_ok, n_fail = await run_one_config(
                    persona=persona,
                    pair_idx=pair_idx,
                    polarity=polarity,
                    artifacts=artifacts,
                    tasks=tasks,
                    client=client,
                )
                total_ok += n_ok
                total_fail += n_fail

    elapsed = time.time() - t0
    print(f"\nTotal: {total_ok} ok, {total_fail} failures in {elapsed/60:.1f} min.")
    print(f"Completions: {COMPLETIONS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
