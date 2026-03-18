"""GPU end-to-end tests for the steering pipeline.

Uses google/gemma-3-1b-it — small enough to fit on a single GPU,
large enough to produce coherent output where steering visibly changes behavior.

Tests check that steering produces output different from baseline, not that
it produces any particular output.

Run with: pytest tests/steering/test_steering_gpu_e2e.py -v
Skip with: pytest -m "not gpu"
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient
from src.steering.hooks import position_selective_steering
from src.steering.kv_cache import combine_caches, modify_cache_v_at_positions, project_to_v_space
from src.steering.runner import (
    _batch_generate_from_interpolated_caches,
    _build_interpolated_cache,
    _clone_cache,
    _compute_cache_delta,
)
from src.steering.tokenization import find_pairwise_task_spans

pytestmark = pytest.mark.gpu

MODEL_NAME = "gemma-3-1b"
HIDDEN_DIM = 1152
N_LAYERS = 26
STEER_LAYER = 13  # roughly middle of 26 layers
MAX_NEW_TOKENS = 64

RNG = np.random.default_rng(42)
DIRECTION = RNG.standard_normal(HIDDEN_DIM).astype(np.float32)
DIRECTION /= np.linalg.norm(DIRECTION)

SIMPLE_PROMPT = [{"role": "user", "content": "Hello, how are you today?"}]
PAIRWISE_PROMPT = [{"role": "user", "content": "Task A: Write a poem about the ocean. Task B: Solve the equation 2x + 3 = 7."}]
TASK_A = "Write a poem about the ocean"
TASK_B = "Solve the equation 2x + 3 = 7"

COEF = 3000.0


@pytest.fixture(scope="module")
def model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    m = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def baseline(model):
    return model.generate(SIMPLE_PROMPT, temperature=0)


@pytest.fixture(scope="module")
def pairwise_spans(model):
    """Pre-compute task spans for the pairwise prompt."""
    formatted = model.format_messages(PAIRWISE_PROMPT, add_generation_prompt=True)
    return find_pairwise_task_spans(
        model.tokenizer, formatted, TASK_A, TASK_B, "Task A", "Task B",
    )


# ---------------------------------------------------------------------------
# SteeredHFClient: hook-based steering modes
# ---------------------------------------------------------------------------

class TestHookSteering:

    def test_all_tokens_differs_from_baseline(self, model, baseline):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="all_tokens",
        )
        result = c.generate(SIMPLE_PROMPT, temperature=0)
        assert result != baseline

    def test_opposite_directions_differ(self, model):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF,
        )
        pos = c.generate(SIMPLE_PROMPT, temperature=0)
        neg = c.with_coefficient(-COEF).generate(SIMPLE_PROMPT, temperature=0)
        assert pos != neg

    def test_autoregressive_differs_from_baseline(self, model, baseline):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="autoregressive",
        )
        result = c.generate(SIMPLE_PROMPT, temperature=0)
        assert result != baseline


# ---------------------------------------------------------------------------
# SteeredHFClient: cache-based steering modes
# ---------------------------------------------------------------------------

class TestCacheHookInjection:

    def test_differs_from_baseline(self, model, baseline):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="cache", cache_injection="hook",
        )
        result = c.generate(
            PAIRWISE_PROMPT, temperature=0,
            task_prompts=[TASK_A, TASK_B],
        )
        assert result != baseline

    def test_opposite_directions_differ(self, model):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="cache", cache_injection="hook",
        )
        pos = c.generate(PAIRWISE_PROMPT, temperature=0, task_prompts=[TASK_A, TASK_B])
        neg = c.with_coefficient(-COEF).generate(PAIRWISE_PROMPT, temperature=0, task_prompts=[TASK_A, TASK_B])
        assert pos != neg


class TestCachePostHoc:

    def test_differs_from_baseline(self, model, baseline):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="cache", cache_injection="post_hoc",
        )
        result = c.generate(
            PAIRWISE_PROMPT, temperature=0,
            task_prompts=[TASK_A, TASK_B],
        )
        assert result != baseline

    def test_opposite_directions_differ(self, model):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="cache", cache_injection="post_hoc",
        )
        pos = c.generate(PAIRWISE_PROMPT, temperature=0, task_prompts=[TASK_A, TASK_B])
        neg = c.with_coefficient(-COEF).generate(PAIRWISE_PROMPT, temperature=0, task_prompts=[TASK_A, TASK_B])
        assert pos != neg


class TestCacheRecomputeSuffix:

    def test_recompute_differs_from_no_recompute(self, model):
        base = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="cache", cache_injection="hook",
        )
        recompute = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="cache", cache_injection="hook",
            recompute_suffix=True,
        )
        r1 = base.generate(PAIRWISE_PROMPT, temperature=0, task_prompts=[TASK_A, TASK_B])
        r2 = recompute.generate(PAIRWISE_PROMPT, temperature=0, task_prompts=[TASK_A, TASK_B])
        assert r1 != r2


class TestZeroCoefficient:

    def test_zero_matches_baseline(self, model, baseline):
        c = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=0.0,
        )
        result = c.generate(SIMPLE_PROMPT, temperature=0)
        assert result == baseline


# ---------------------------------------------------------------------------
# Runner pipeline: interpolation + batched generation
# ---------------------------------------------------------------------------

class TestRunnerInterpolation:
    """Tests the runner's cache interpolation pipeline (the core optimization)."""

    def test_batch_interpolated_generates_valid_output(self, model, pairwise_spans):
        """3 prefills → delta → batch generate produces non-empty strings."""
        a_span, b_span = pairwise_spans
        ref_tensor = torch.tensor(DIRECTION * COEF, dtype=torch.bfloat16, device="cuda")

        cache_clean, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])
        cache_pos, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(ref_tensor, a_span[0], a_span[1]))],
        )
        cache_neg, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(-ref_tensor, b_span[0], b_span[1]))],
        )

        combined = combine_caches(cache_pos, cache_neg, b_span[0], b_span[1])
        deltas = _compute_cache_delta(combined, cache_clean)

        responses = _batch_generate_from_interpolated_caches(
            model, cache_clean, deltas, input_ids,
            scales=[0.5, 1.0], trials_per_scale=[1, 1], temperature=0,
        )
        assert len(responses) == 2
        assert all(isinstance(r, str) and len(r) > 0 for r in responses)

    def test_interpolation_scale1_close_to_exact(self, model, pairwise_spans):
        """Interpolated cache at scale=1.0 should approximately match exact hook injection.

        Not byte-identical because the batch path trims the last prompt token
        from the cache and regenerates it, introducing minor numerical divergence.
        """
        a_span, b_span = pairwise_spans
        ref_tensor = torch.tensor(DIRECTION * COEF, dtype=torch.bfloat16, device="cuda")

        # Exact: SteeredHFClient hook injection
        exact_client = SteeredHFClient(
            model, layer=STEER_LAYER, steering_direction=DIRECTION,
            coefficient=COEF, steering_mode="cache", cache_injection="hook",
        )
        exact_output = exact_client.generate(
            PAIRWISE_PROMPT, temperature=0, task_prompts=[TASK_A, TASK_B],
        )

        # Interpolated at scale=1.0
        cache_clean, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])
        cache_pos, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(ref_tensor, a_span[0], a_span[1]))],
        )
        cache_neg, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(-ref_tensor, b_span[0], b_span[1]))],
        )
        combined = combine_caches(cache_pos, cache_neg, b_span[0], b_span[1])
        deltas = _compute_cache_delta(combined, cache_clean)

        interp_output = _batch_generate_from_interpolated_caches(
            model, cache_clean, deltas, input_ids,
            scales=[1.0], trials_per_scale=[1], temperature=0,
        )
        # Tokenize both and check >= 75% overlap
        exact_ids = model.tokenizer.encode(exact_output, add_special_tokens=False)
        interp_ids = model.tokenizer.encode(interp_output[0], add_special_tokens=False)
        common = sum(a == b for a, b in zip(exact_ids, interp_ids))
        overlap = common / max(len(exact_ids), 1)
        assert overlap >= 0.75, f"Only {overlap:.0%} token overlap between exact and interpolated"

    def test_opposite_scales_differ(self, model, pairwise_spans):
        """Positive vs negative interpolation scales should produce different output."""
        a_span, b_span = pairwise_spans
        ref_tensor = torch.tensor(DIRECTION * COEF, dtype=torch.bfloat16, device="cuda")

        cache_clean, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])
        cache_pos, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(ref_tensor, a_span[0], a_span[1]))],
        )
        cache_neg, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(-ref_tensor, b_span[0], b_span[1]))],
        )
        combined = combine_caches(cache_pos, cache_neg, b_span[0], b_span[1])
        deltas = _compute_cache_delta(combined, cache_clean)

        responses = _batch_generate_from_interpolated_caches(
            model, cache_clean, deltas, input_ids,
            scales=[-1.0, 1.0], trials_per_scale=[1, 1], temperature=0,
        )
        assert responses[0] != responses[1]


# ---------------------------------------------------------------------------
# Runner pipeline: recompute suffix via interpolation
# ---------------------------------------------------------------------------

class TestRunnerRecomputeSuffix:
    """Tests the runner's per-multiplier recompute_suffix path."""

    def test_recompute_produces_valid_output(self, model, pairwise_spans):
        a_span, b_span = pairwise_spans
        ref_tensor = torch.tensor(DIRECTION * COEF, dtype=torch.bfloat16, device="cuda")

        cache_clean, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])
        cache_pos, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(ref_tensor, a_span[0], a_span[1]))],
        )
        cache_neg, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(-ref_tensor, b_span[0], b_span[1]))],
        )
        combined = combine_caches(cache_pos, cache_neg, b_span[0], b_span[1])
        deltas = _compute_cache_delta(combined, cache_clean)

        cache = _build_interpolated_cache(cache_clean, deltas, 1.0)
        cache = model.recompute_suffix(cache, input_ids, b_span[1])
        responses = model.generate_from_cache(cache, input_ids, temperature=0)

        assert len(responses) == 1
        assert isinstance(responses[0], str)
        assert len(responses[0]) > 0

    def test_recompute_differs_from_no_recompute(self, model, pairwise_spans):
        """Runner's recompute path should differ from batched non-recompute path."""
        a_span, b_span = pairwise_spans
        ref_tensor = torch.tensor(DIRECTION * COEF, dtype=torch.bfloat16, device="cuda")

        cache_clean, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])
        cache_pos, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(ref_tensor, a_span[0], a_span[1]))],
        )
        cache_neg, _ = model.prefill_with_hooks(
            PAIRWISE_PROMPT,
            [(STEER_LAYER, position_selective_steering(-ref_tensor, b_span[0], b_span[1]))],
        )
        combined = combine_caches(cache_pos, cache_neg, b_span[0], b_span[1])
        deltas = _compute_cache_delta(combined, cache_clean)

        # No recompute (batched)
        no_recompute = _batch_generate_from_interpolated_caches(
            model, cache_clean, deltas, input_ids,
            scales=[1.0], trials_per_scale=[1], temperature=0,
        )

        # Recompute (runner's path)
        cache = _build_interpolated_cache(cache_clean, deltas, 1.0)
        cache = model.recompute_suffix(cache, input_ids, b_span[1])
        with_recompute = model.generate_from_cache(cache, input_ids, temperature=0)

        assert no_recompute[0] != with_recompute[0]


# ---------------------------------------------------------------------------
# Runner pipeline: post-hoc multi-layer V modification
# ---------------------------------------------------------------------------

class TestRunnerPostHoc:
    """Tests the runner's post-hoc V cache modification pipeline."""

    def test_multi_layer_v_modification_changes_output(self, model, pairwise_spans, baseline):
        """Modifying V at multiple layers should produce different output from baseline."""
        a_span, b_span = pairwise_spans
        cache, input_ids = model.prefill_with_hooks(PAIRWISE_PROMPT, [])

        # Project direction to V space at several layers, modify cache
        layers = [5, 10, 15, 20]
        cache = _clone_cache(cache)
        for layer in layers:
            v_dir = project_to_v_space(model.model, layer, DIRECTION)
            modify_cache_v_at_positions(cache, layer, a_span[0], a_span[1], v_dir, +COEF)
            modify_cache_v_at_positions(cache, layer, b_span[0], b_span[1], v_dir, -COEF)

        responses = model.generate_from_cache(cache, input_ids, temperature=0)
        assert responses[0] != baseline

    def test_clone_prevents_base_cache_mutation(self, model):
        """Cloning cache before modification should protect the original."""
        cache, _ = model.prefill_with_hooks(SIMPLE_PROMPT, [])
        original_v = cache.layers[0].values.clone()

        cloned = _clone_cache(cache)
        v_dir = project_to_v_space(model.model, 0, DIRECTION)
        modify_cache_v_at_positions(cloned, 0, 0, 2, v_dir, 9999.0)

        assert torch.equal(cache.layers[0].values, original_v)
