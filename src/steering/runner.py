"""Config-driven steering experiment runner.

Consolidates boilerplate from isolated steering scripts: pair loading,
checkpoint management, ordering sign correction, response parsing, and
progress logging. Experiments are defined by YAML configs.

Two condition types:
- PostHocCondition: clean prefill → modify V cache at multiple layers.
  One prefill per (pair, ordering); clone per multiplier.
- HookCondition: two steered prefills + combine caches. One layer at a time
  with per-layer probes. Uses interpolation (3 prefills at ref multiplier
  + batched generation across all multipliers).
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers.cache_utils import DynamicCache

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.hooks import position_selective_steering
from src.steering.kv_cache import combine_caches, modify_cache_v_at_positions, project_to_v_space
from src.steering.tokenization import find_pairwise_task_spans
from src.task_data import OriginDataset, Task


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PostHocCondition:
    name: str
    probe: str                      # e.g. "ridge_L25"
    kv_layers: tuple[int, int]      # (start, end) inclusive
    multipliers: list[float]


@dataclass
class HookCondition:
    name: str
    probe_prefix: str               # e.g. "ridge_L" — layer number appended
    layers: list[int]               # iterated individually, each gets own probe
    multipliers: list[float]
    ref_mult: float                 # reference multiplier for interpolation
    # Which suffix modes to run. [False] = no recompute, [True] = recompute,
    # [False, True] = both (shared prefills, separate condition names with
    # "_recompute" suffix for True).
    recompute_modes: list[bool] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.recompute_modes is None:
            self.recompute_modes = [False]


@dataclass
class RunConfig:
    model: str
    max_new_tokens: int
    pairs_path: Path
    probe_manifest: Path
    checkpoint_path: Path
    mean_norm: float
    n_trials: int
    temperature: float
    seed: int
    n_pairs: int | None             # None = use all pairs
    template_path: str
    conditions: list[PostHocCondition | HookCondition]


def _parse_condition(raw: dict) -> PostHocCondition | HookCondition:
    cache_injection = raw["cache_injection"]
    if cache_injection == "post_hoc":
        kv_start, kv_end = raw["kv_layers"]
        return PostHocCondition(
            name=raw["name"],
            probe=raw["probe"],
            kv_layers=(kv_start, kv_end),
            multipliers=raw["multipliers"],
        )
    if cache_injection == "hook":
        raw_recompute = raw.get("recompute_suffix", False)
        if isinstance(raw_recompute, bool):
            recompute_modes = [raw_recompute]
        elif isinstance(raw_recompute, list):
            recompute_modes = raw_recompute
        else:
            raise ValueError(f"recompute_suffix must be bool or list[bool], got {raw_recompute!r}")
        return HookCondition(
            name=raw["name"],
            probe_prefix=raw["probe_prefix"],
            layers=raw["layers"],
            multipliers=raw["multipliers"],
            ref_mult=raw["ref_mult"],
            recompute_modes=recompute_modes,
        )
    raise ValueError(f"Unknown cache_injection: {cache_injection!r}")


def load_config(config_path: Path) -> RunConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    conditions = [_parse_condition(c) for c in raw["conditions"]]
    return RunConfig(
        model=raw["model"],
        max_new_tokens=raw["max_new_tokens"],
        pairs_path=Path(raw["pairs_path"]),
        probe_manifest=Path(raw["probe_manifest"]),
        checkpoint_path=Path(raw["checkpoint_path"]),
        mean_norm=raw["mean_norm"],
        n_trials=raw["n_trials"],
        temperature=raw["temperature"],
        seed=raw["seed"],
        n_pairs=raw.get("n_pairs"),
        template_path=raw["template_path"],
        conditions=conditions,
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _effective_coef(coef: float, ordering: int) -> float:
    """Negate when ordering=1 to maintain direction in original task space."""
    return coef if ordering == 0 else -coef


def _remap_choice(choice_presented: str, ordering: int) -> str:
    """Map presented choice back to original task ordering."""
    if choice_presented in ("a", "b") and ordering == 1:
        return "b" if choice_presented == "a" else "a"
    return choice_presented


def _parse_response(response_format, response: str) -> str:
    result = response_format.parse_sync(response)
    return "refusal" if result == "parse_fail" else result


def _make_row(
    *,
    pair: dict,
    multiplier: float,
    mean_norm: float,
    layer: int,
    condition: str,
    sample_idx: int,
    ordering: int,
    choice_presented: str,
    raw_response: str,
) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "task_a_id": pair["task_a"],
        "task_b_id": pair["task_b"],
        "coefficient": mean_norm * multiplier,
        "multiplier": multiplier,
        "layer": layer,
        "condition": condition,
        "sample_idx": sample_idx,
        "ordering": ordering,
        "choice_original": _remap_choice(choice_presented, ordering),
        "choice_presented": choice_presented,
        "raw_response": raw_response,
        "delta_mu": pair["delta_mu"],
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path) -> dict[tuple, int]:
    counts: dict[tuple, int] = defaultdict(int)
    if not path.exists():
        return counts
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            key = (row["pair_id"], row["layer"], row["multiplier"], row["condition"], row["ordering"])
            counts[key] += 1
    return counts


def _append_checkpoint(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Pair / task utilities
# ---------------------------------------------------------------------------

def _load_pairs(config: RunConfig) -> list[dict]:
    with open(config.pairs_path) as f:
        all_pairs = json.load(f)
    if config.n_pairs is not None:
        random.seed(config.seed)
        return random.sample(all_pairs, config.n_pairs)
    return all_pairs


def _pair_to_tasks(pair: dict) -> tuple[Task, Task]:
    task_a = Task(prompt=pair["task_a_text"], origin=OriginDataset.ALPACA, id=pair["task_a"], metadata={})
    task_b = Task(prompt=pair["task_b_text"], origin=OriginDataset.ALPACA, id=pair["task_b"], metadata={})
    return task_a, task_b


def _prepare_pair(
    builder, response_format, hf_model: HuggingFaceModel,
    pres_a: Task, pres_b: Task,
) -> tuple[list, tuple[int, int], tuple[int, int]] | None:
    """Build prompt, set response format, find task spans. Returns None on span failure."""
    prompt_data = builder.build(pres_a, pres_b)
    messages = prompt_data.messages
    response_format.task_a_prompt = pres_a.prompt
    response_format.task_b_prompt = pres_b.prompt
    formatted = hf_model.format_messages(messages, add_generation_prompt=True)
    try:
        a_span, b_span = find_pairwise_task_spans(
            hf_model.tokenizer, formatted,
            pres_a.prompt, pres_b.prompt, "Task A", "Task B",
        )
    except ValueError:
        return None
    return messages, a_span, b_span


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _clone_cache(cache: DynamicCache) -> DynamicCache:
    cloned = DynamicCache()
    for layer_idx in range(len(cache)):
        layer = cache.layers[layer_idx]
        cloned.update(layer.keys.clone(), layer.values.clone(), layer_idx)
    return cloned


def _compute_cache_delta(
    combined_ref: DynamicCache,
    cache_clean: DynamicCache,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    deltas = []
    for li in range(len(cache_clean)):
        dk = combined_ref.layers[li].keys - cache_clean.layers[li].keys
        dv = combined_ref.layers[li].values - cache_clean.layers[li].values
        deltas.append((dk, dv))
    return deltas


def _build_interpolated_cache(
    cache_clean: DynamicCache,
    deltas: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> DynamicCache:
    """Build clean + scale * delta cache (full sequence length)."""
    cache = DynamicCache()
    for li in range(len(cache_clean)):
        k = (cache_clean.layers[li].keys + scale * deltas[li][0]).clone()
        v = (cache_clean.layers[li].values + scale * deltas[li][1]).clone()
        cache.update(k, v, li)
    return cache


def _batch_generate_from_interpolated_caches(
    hf_model: HuggingFaceModel,
    cache_clean: DynamicCache,
    deltas: list[tuple[torch.Tensor, torch.Tensor]],
    input_ids: torch.Tensor,
    scales: list[float],
    trials_per_scale: list[int],
    temperature: float,
) -> list[str]:
    """Build batch cache via clean + scale * delta, generate all at once."""
    seq_len = input_ids.shape[1]
    total_batch = sum(trials_per_scale)

    batch_cache = DynamicCache()
    for li in range(len(cache_clean)):
        clean_k = cache_clean.layers[li].keys[:, :, :seq_len - 1, :]
        clean_v = cache_clean.layers[li].values[:, :, :seq_len - 1, :]
        dk = deltas[li][0][:, :, :seq_len - 1, :]
        dv = deltas[li][1][:, :, :seq_len - 1, :]

        k_parts = []
        v_parts = []
        for scale, n in zip(scales, trials_per_scale):
            k_scaled = clean_k + scale * dk
            v_scaled = clean_v + scale * dv
            k_parts.append(k_scaled.expand(n, -1, -1, -1).contiguous())
            v_parts.append(v_scaled.expand(n, -1, -1, -1).contiguous())

        batch_cache.update(torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0), li)

    expanded_ids = input_ids.expand(total_batch, -1).contiguous()
    gen_kwargs = hf_model._build_gen_kwargs(temperature, None, num_return_sequences=1)
    gen_kwargs["past_key_values"] = batch_cache

    output_ids = hf_model.model.generate(expanded_ids, **gen_kwargs)
    return hf_model._decode_completions(output_ids, seq_len, total_batch)


# ---------------------------------------------------------------------------
# Post-hoc condition runner
# ---------------------------------------------------------------------------

def _run_post_hoc_condition(
    condition: PostHocCondition,
    hf_model: HuggingFaceModel,
    pairs: list[dict],
    builder,
    response_format,
    config: RunConfig,
    checkpoint_counts: dict[tuple, int],
    stats: dict,
) -> None:
    kv_layer_range = list(range(condition.kv_layers[0], condition.kv_layers[1] + 1))

    # Load probe and project to V space for all kv layers
    _, direction = load_probe_direction(config.probe_manifest, condition.probe)
    v_dirs: dict[int, torch.Tensor] = {}
    for layer in kv_layer_range:
        v_dirs[layer] = project_to_v_space(hf_model.model, layer, direction)

    print(f"\nCondition: {condition.name} "
          f"(layers {condition.kv_layers[0]}-{condition.kv_layers[1]}, probe={condition.probe})")

    for pair_idx, pair in enumerate(pairs):
        task_a, task_b = _pair_to_tasks(pair)
        pair_id = pair["pair_id"]

        for ordering in [0, 1]:
            pres_a = task_a if ordering == 0 else task_b
            pres_b = task_b if ordering == 0 else task_a

            # Check if any multiplier needs work (early exit)
            any_needed = False
            for mult in condition.multipliers:
                key = (pair_id, -1, mult, condition.name, ordering)
                if checkpoint_counts[key] < config.n_trials:
                    any_needed = True
                    break
            if not any_needed:
                stats["skipped"] += len(condition.multipliers) * config.n_trials
                continue

            prepared = _prepare_pair(builder, response_format, hf_model, pres_a, pres_b)
            if prepared is None:
                continue
            messages, a_span, b_span = prepared

            # Prefill once per (pair, ordering)
            base_cache, input_ids = hf_model.prefill_with_hooks(messages, [])

            for mult in condition.multipliers:
                key = (pair_id, -1, mult, condition.name, ordering)
                existing_n = checkpoint_counts[key]
                if existing_n >= config.n_trials:
                    stats["skipped"] += config.n_trials
                    continue

                needed = config.n_trials - existing_n
                effective = _effective_coef(config.mean_norm * mult, ordering)

                cache = _clone_cache(base_cache)
                for layer in kv_layer_range:
                    modify_cache_v_at_positions(cache, layer, a_span[0], a_span[1], v_dirs[layer], +effective)
                    modify_cache_v_at_positions(cache, layer, b_span[0], b_span[1], v_dirs[layer], -effective)

                responses = hf_model.generate_from_cache(
                    cache, input_ids, temperature=config.temperature, num_return_sequences=needed,
                )

                rows = []
                for sample_idx, response in enumerate(responses):
                    rows.append(_make_row(
                        pair=pair, multiplier=mult, mean_norm=config.mean_norm,
                        layer=-1, condition=condition.name,
                        sample_idx=existing_n + sample_idx, ordering=ordering,
                        choice_presented=_parse_response(response_format, response),
                        raw_response=response,
                    ))

                _append_checkpoint(config.checkpoint_path, rows)
                stats["generated"] += len(rows)

        if (pair_idx + 1) % 10 == 0:
            _log_progress(pair_idx, len(pairs), stats)


# ---------------------------------------------------------------------------
# Hook condition runner
# ---------------------------------------------------------------------------

def _condition_name(base: str, recompute: bool) -> str:
    return f"{base}_recompute" if recompute else base


def _run_hook_condition(
    condition: HookCondition,
    hf_model: HuggingFaceModel,
    pairs: list[dict],
    builder,
    response_format,
    config: RunConfig,
    checkpoint_counts: dict[tuple, int],
    stats: dict,
) -> None:
    mode_names = [_condition_name(condition.name, m) for m in condition.recompute_modes]
    print(f"\nCondition: {condition.name} "
          f"(layers={condition.layers}, ref_mult={condition.ref_mult}, "
          f"modes={mode_names})")

    for layer in condition.layers:
        probe_id = f"{condition.probe_prefix}{layer}"
        _, direction = load_probe_direction(config.probe_manifest, probe_id)
        ref_coef = config.mean_norm * condition.ref_mult
        ref_tensor = torch.tensor(direction * ref_coef, dtype=torch.bfloat16, device="cuda")

        print(f"  Layer {layer}")

        for pair_idx, pair in enumerate(pairs):
            task_a, task_b = _pair_to_tasks(pair)
            pair_id = pair["pair_id"]

            for ordering in [0, 1]:
                pres_a = task_a if ordering == 0 else task_b
                pres_b = task_b if ordering == 0 else task_a

                # Collect needed multipliers across all recompute modes
                needs_by_mode: dict[bool, list[tuple[float, int, int]]] = {}
                for recompute in condition.recompute_modes:
                    cond_name = _condition_name(condition.name, recompute)
                    entries = []
                    for mult in condition.multipliers:
                        key = (pair_id, layer, mult, cond_name, ordering)
                        existing_n = checkpoint_counts[key]
                        needed = config.n_trials - existing_n
                        if needed > 0:
                            entries.append((mult, needed, existing_n))
                        else:
                            stats["skipped"] += config.n_trials
                    if entries:
                        needs_by_mode[recompute] = entries

                if not needs_by_mode:
                    continue

                prepared = _prepare_pair(builder, response_format, hf_model, pres_a, pres_b)
                if prepared is None:
                    continue
                messages, a_span, b_span = prepared

                # 3 shared prefills for all modes
                cache_clean, input_ids = hf_model.prefill_with_hooks(messages, [])
                cache_pos, _ = hf_model.prefill_with_hooks(
                    messages,
                    [(layer, position_selective_steering(ref_tensor, a_span[0], a_span[1]))],
                )
                cache_neg, _ = hf_model.prefill_with_hooks(
                    messages,
                    [(layer, position_selective_steering(-ref_tensor, b_span[0], b_span[1]))],
                )

                combined_ref = combine_caches(cache_clean, [
                    (cache_pos, a_span[0], a_span[1]),
                    (cache_neg, b_span[0], b_span[1]),
                ])
                del cache_pos, cache_neg
                deltas = _compute_cache_delta(combined_ref, cache_clean)
                del combined_ref

                # Generate for each mode, sharing the prefills + deltas
                rows = []
                for recompute, batch_entries in needs_by_mode.items():
                    cond_name = _condition_name(condition.name, recompute)

                    if recompute:
                        # Per-multiplier: interpolate → recompute suffix → generate
                        for mult, n_needed, existing_count in batch_entries:
                            scale = _effective_coef(mult, ordering) / condition.ref_mult
                            cache = _build_interpolated_cache(cache_clean, deltas, scale)
                            cache = hf_model.recompute_suffix(cache, input_ids, b_span[1])
                            responses = hf_model.generate_from_cache(
                                cache, input_ids, temperature=config.temperature,
                                num_return_sequences=n_needed,
                            )
                            for trial, response in enumerate(responses):
                                rows.append(_make_row(
                                    pair=pair, multiplier=mult, mean_norm=config.mean_norm,
                                    layer=layer, condition=cond_name,
                                    sample_idx=existing_count + trial, ordering=ordering,
                                    choice_presented=_parse_response(response_format, response),
                                    raw_response=response,
                                ))
                    else:
                        # Batched: all multipliers in one generate call
                        scales = [_effective_coef(mult, ordering) / condition.ref_mult
                                  for mult, _, _ in batch_entries]
                        trials = [n for _, n, _ in batch_entries]
                        responses = _batch_generate_from_interpolated_caches(
                            hf_model, cache_clean, deltas, input_ids,
                            scales, trials, temperature=config.temperature,
                        )
                        resp_idx = 0
                        for mult, n_needed, existing_count in batch_entries:
                            for trial in range(n_needed):
                                rows.append(_make_row(
                                    pair=pair, multiplier=mult, mean_norm=config.mean_norm,
                                    layer=layer, condition=cond_name,
                                    sample_idx=existing_count + trial, ordering=ordering,
                                    choice_presented=_parse_response(response_format, responses[resp_idx]),
                                    raw_response=responses[resp_idx],
                                ))
                                resp_idx += 1

                del cache_clean, deltas

                _append_checkpoint(config.checkpoint_path, rows)
                stats["generated"] += len(rows)

            if (pair_idx + 1) % 10 == 0:
                _log_progress(pair_idx, len(pairs), stats)


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def _log_progress(pair_idx: int, n_pairs: int, stats: dict) -> None:
    elapsed = time.time() - stats["t_start"]
    rate = stats["generated"] / elapsed if elapsed > 0 else 0
    print(f"  Pair {pair_idx + 1}/{n_pairs}: "
          f"{stats['generated']} gen, {stats['skipped']} skip, "
          f"{rate:.1f}/s, {elapsed / 60:.1f}m")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: Path) -> None:
    config = load_config(config_path)

    pairs = _load_pairs(config)
    checkpoint_counts = _load_checkpoint(config.checkpoint_path)
    existing = sum(checkpoint_counts.values())
    print(f"Pairs: {len(pairs)}, checkpoint: {existing} existing rows")

    # Prompt builder + response format
    template = load_templates_from_yaml(config.template_path)[0]
    builder = build_revealed_builder(template, "completion")
    response_format = builder.response_format

    # Load model
    print("Loading model...")
    t0 = time.time()
    hf_model = HuggingFaceModel(config.model, max_new_tokens=config.max_new_tokens)
    print(f"Model loaded in {time.time() - t0:.0f}s")

    stats = {"generated": 0, "skipped": 0, "t_start": time.time()}

    for condition in config.conditions:
        if isinstance(condition, PostHocCondition):
            _run_post_hoc_condition(
                condition, hf_model, pairs, builder, response_format,
                config, checkpoint_counts, stats,
            )
        elif isinstance(condition, HookCondition):
            _run_hook_condition(
                condition, hf_model, pairs, builder, response_format,
                config, checkpoint_counts, stats,
            )

    elapsed = time.time() - stats["t_start"]
    print(f"\nDone in {elapsed / 3600:.1f}h. "
          f"Generated: {stats['generated']}, Skipped: {stats['skipped']}")
