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

pytestmark = pytest.mark.gpu

MODEL_NAME = "gemma-3-1b"
HIDDEN_DIM = 1152
STEER_LAYER = 13  # roughly middle of 26 layers
MAX_NEW_TOKENS = 32

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
