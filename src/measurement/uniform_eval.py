"""Generate uniform-sample pairwise evaluation data.

Samples N pairs uniformly from tasks that have activations but are NOT in the
training set, then measures them using the standard revealed-preference pipeline.
Results are saved as a standard run directory for use in probe evaluation.

Usage:
    python -m src.measurement.uniform_eval configs/uniform_eval/gemma3.yaml \
        --activations-path activations/gemma_3_27b/activations_prompt_last.npz \
        --exclude-run-dir results/experiments/.../pre_task_active_learning/... \
        --n-pairs 500 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

import numpy as np

from src.measurement.elicitation import measure_pre_task_revealed_async
from src.measurement.runners.runners import build_revealed_builder
from src.measurement.runners.utils.experiment_utils import (
    apply_pair_order,
    build_configurations,
    setup_experiment,
)
from src.measurement.storage import MeasurementCache, ExperimentStore, model_short_name
from src.measurement.storage.base import build_measurement_config, load_yaml
from src.task_data import Task, load_filtered_tasks, parse_origins


def _load_training_task_ids(exclude_run_dir: Path) -> set[str]:
    measurements_path = exclude_run_dir / "measurements.yaml"
    raw = load_yaml(measurements_path)
    ids: set[str] = set()
    for m in raw:
        ids.add(m["task_a"])
        ids.add(m["task_b"])
    return ids


def sample_uniform_pairs(
    activations_path: Path,
    exclude_run_dir: Path,
    n_pairs: int,
    seed: int,
    origins: list[str],
) -> list[tuple[Task, Task]]:
    activation_data = np.load(activations_path, allow_pickle=True)
    activation_task_ids = set(activation_data["task_ids"].tolist())
    training_task_ids = _load_training_task_ids(exclude_run_dir)
    eligible_ids = activation_task_ids - training_task_ids

    print(f"Activation tasks: {len(activation_task_ids)}")
    print(f"Training tasks: {len(training_task_ids)}")
    print(f"Eligible tasks: {len(eligible_ids)}")

    origin_datasets = parse_origins(origins)
    tasks = load_filtered_tasks(
        n=len(eligible_ids),
        origins=origin_datasets,
        task_ids=eligible_ids,
        seed=None,
    )
    print(f"Loaded {len(tasks)} eligible tasks from datasets")

    origin_counts = Counter(t.origin.name for t in tasks)
    for origin, count in sorted(origin_counts.items()):
        print(f"  {origin}: {count}")

    rng = np.random.default_rng(seed)
    n_tasks = len(tasks)

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs:
        batch_size = (n_pairs - len(pairs)) * 2  # oversample to account for duplicates
        idx_a = rng.integers(0, n_tasks, size=batch_size)
        idx_b = rng.integers(0, n_tasks, size=batch_size)
        for a, b in zip(idx_a, idx_b):
            if a != b:
                pair = (min(a, b), max(a, b))
                pairs.add(pair)
                if len(pairs) >= n_pairs:
                    break

    print(f"Sampled {len(pairs)} unique pairs")
    return [(tasks[a], tasks[b]) for a, b in sorted(pairs)]


async def run_uniform_eval(
    config_path: Path,
    pairs: list[tuple[Task, Task]],
    max_concurrent: int = 50,
) -> Path:
    task_lookup = {}
    for a, b in pairs:
        task_lookup[a.id] = a
        task_lookup[b.id] = b

    ctx = setup_experiment(config_path, expected_mode="pre_task_revealed")
    config = ctx.config

    # Override the context's tasks/task_lookup with our pair tasks
    ctx.tasks = list(task_lookup.values())
    ctx.task_lookup = task_lookup

    configurations = build_configurations(ctx, config, include_order=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    model_short = model_short_name(ctx.client.canonical_model_name)
    exp_store = ExperimentStore(config.experiment_id)

    print(f"\nMeasuring {len(pairs)} pairs x {config.n_samples} samples "
          f"x {len(configurations)} configurations")

    for cfg in configurations:
        seed_suffix = f"_seed{cfg.seed}" if cfg.seed is not None else ""
        run_name = f"{cfg.template.name}_{model_short}_{cfg.response_format}_{cfg.order}{seed_suffix}_uniform_eval"

        if exp_store.exists("pre_task_revealed", run_name):
            print(f"  Skipping {run_name} (already exists)")
            continue

        cache = MeasurementCache(
            cfg.template, ctx.client, cfg.response_format, cfg.order,
            seed=cfg.seed, system_prompt=config.measurement_system_prompt,
        )
        ordered_pairs = apply_pair_order(
            pairs, cfg.order, config.pair_order_seed, config.include_reverse_order,
        )
        pairs_with_repeats = ordered_pairs * config.n_samples

        builder = build_revealed_builder(
            cfg.template, cfg.response_format, post_task=False,
            reasoning_mode=config.reasoning_mode,
            system_prompt=config.measurement_system_prompt,
        )

        async def measure_fn(pairs_to_query: list[tuple[Task, Task]]):
            return await measure_pre_task_revealed_async(
                client=ctx.client,
                pairs=pairs_to_query,
                builder=builder,
                semaphore=semaphore,
                temperature=config.temperature,
                seed=cfg.seed,
            )

        measurements, batch_stats = await cache.get_or_measure_async(
            pairs_with_repeats, measure_fn, task_lookup,
        )
        print(f"  {run_name}: {len(measurements)} measurements "
              f"({batch_stats.cache_hits} cached, {batch_stats.api_successes} API)")

        if measurements:
            measurements_dicts = [
                {
                    "task_a": m.task_a.id,
                    "task_b": m.task_b.id,
                    "choice": m.choice,
                    "origin_a": m.task_a.origin.name,
                    "origin_b": m.task_b.origin.name,
                }
                | ({"raw_response": m.raw_response} if m.raw_response else {})
                for m in measurements
            ]
            config_dict = build_measurement_config(
                template=cfg.template,
                client=ctx.client,
                response_format=cfg.response_format,
                order=cfg.order,
                seed=cfg.seed,
                temperature=config.temperature,
            )
            run_dir = exp_store.save_revealed(
                "pre_task_revealed", run_name, measurements_dicts, config_dict,
            )
            print(f"  Saved to {run_dir}")

    return exp_store.base_dir / "pre_task_revealed"


def main():
    parser = argparse.ArgumentParser(
        description="Generate uniform-sample pairwise evaluation data",
    )
    parser.add_argument("config", type=Path, help="Measurement config YAML (pre_task_revealed)")
    parser.add_argument("--activations-path", type=Path, required=True)
    parser.add_argument("--exclude-run-dir", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument(
        "--origins", nargs="+",
        default=["wildchat", "alpaca", "math", "bailbench", "stress_test"],
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    pairs = sample_uniform_pairs(
        activations_path=args.activations_path,
        exclude_run_dir=args.exclude_run_dir,
        n_pairs=args.n_pairs,
        seed=args.seed,
        origins=args.origins,
    )

    result_dir = asyncio.run(run_uniform_eval(
        config_path=args.config,
        pairs=pairs,
        max_concurrent=args.max_concurrent,
    ))
    print(f"\nDone. Results in: {result_dir}")


if __name__ == "__main__":
    main()
