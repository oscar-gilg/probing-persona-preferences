"""Steered HuggingFace model that duck-types as OpenAICompatibleClient."""

from __future__ import annotations

import asyncio
from typing import Callable

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.openai_compatible import BatchResult, GenerateRequest
from src.steering.hooks import (
    STEERING_MODES,
    differential_steering,
    position_selective_steering,
)
from src.steering.kv_cache import combine_caches, modify_cache_v_at_positions, project_to_v_space
from src.steering.tokenization import find_pairwise_task_spans

CACHE_STEERING_MODES = {"kv_cache_differential", "activation_patch"}


class SteeredHFClient:
    """HuggingFaceModel with steering, duck-typed as OpenAICompatibleClient."""

    def __init__(
        self,
        hf_model: HuggingFaceModel,
        layer: int,
        steering_direction: np.ndarray,
        coefficient: float,
        steering_mode: str = "all_tokens",
        a_marker: str = "Task A:",
        b_marker: str = "Task B:",
    ):
        self.hf_model = hf_model
        self.layer = layer
        self._direction = steering_direction
        self.coefficient = coefficient
        self.steering_mode = steering_mode
        self.a_marker = a_marker
        self.b_marker = b_marker

        self.canonical_model_name = hf_model.canonical_model_name
        self.model_name = hf_model.model_name
        self.max_new_tokens = hf_model.max_new_tokens

        # Pre-compute scaled steering tensor on GPU
        scaled = steering_direction * coefficient
        self._steering_tensor = torch.tensor(
            scaled, dtype=torch.bfloat16, device=hf_model.device
        )

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    def with_coefficient(self, coefficient: float) -> SteeredHFClient:
        """Return a new client with a different coefficient, sharing the same model."""
        return SteeredHFClient(
            self.hf_model,
            self.layer,
            self._direction,
            coefficient,
            self.steering_mode,
            self.a_marker,
            self.b_marker,
        )

    def _resolve_task_spans(
        self, messages: list, task_prompts: list[str]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        formatted = self.hf_model.format_messages(messages)
        return find_pairwise_task_spans(
            self.hf_model.tokenizer, formatted,
            task_prompts[0], task_prompts[1],
            self.a_marker, self.b_marker,
        )

    def _resolve_hook(self, messages: list, task_prompts: list[str] | None = None):
        if task_prompts is not None and len(task_prompts) == 2 and self.steering_mode == "differential":
            a_span, b_span = self._resolve_task_spans(messages, task_prompts)
            return differential_steering(
                self._steering_tensor, a_span[0], a_span[1], b_span[0], b_span[1]
            )
        return STEERING_MODES[self.steering_mode](self._steering_tensor)

    def generate(self, messages, temperature=1.0, task_prompts: list[str] | None = None) -> str:
        return self._dispatch(messages, temperature, task_prompts, n=1)[0]

    def generate_n(self, messages, n: int, temperature: float = 1.0, task_prompts: list[str] | None = None) -> list[str]:
        return self._dispatch(messages, temperature, task_prompts, n=n)

    def _dispatch(
        self,
        messages: list,
        temperature: float,
        task_prompts: list[str] | None,
        n: int,
    ) -> list[str]:
        if self.coefficient == 0:
            return self.hf_model.generate_n(messages, n=n, temperature=temperature)

        if self.steering_mode in CACHE_STEERING_MODES:
            if task_prompts is None or len(task_prompts) != 2:
                raise ValueError(
                    f"Cache steering modes require exactly 2 task_prompts, got {task_prompts}"
                )
            a_span, b_span = self._resolve_task_spans(messages, task_prompts)
            if self.steering_mode == "kv_cache_differential":
                return self._generate_kv_cache_differential(messages, a_span, b_span, temperature, n)
            return self._generate_activation_patch(messages, a_span, b_span, temperature, n)

        hook = self._resolve_hook(messages, task_prompts)
        return self.hf_model.generate_with_hook_n(
            messages=messages, layer=self.layer, hook=hook, n=n, temperature=temperature,
        )

    def _generate_kv_cache_differential(
        self,
        messages: list,
        a_span: tuple[int, int],
        b_span: tuple[int, int],
        temperature: float,
        n: int,
    ) -> list[str]:
        """Clean prefill, then modify V cache at task positions, then generate."""
        cache, input_ids = self.hf_model.prefill_with_hooks(messages, [])
        v_dir = project_to_v_space(self.hf_model.model, self.layer, self._direction)
        modify_cache_v_at_positions(cache, self.layer, a_span[0], a_span[1], v_dir, +self.coefficient)
        modify_cache_v_at_positions(cache, self.layer, b_span[0], b_span[1], v_dir, -self.coefficient)
        return self.hf_model.generate_from_cache(
            cache, input_ids, temperature=temperature, num_return_sequences=n,
        )

    def _generate_activation_patch(
        self,
        messages: list,
        a_span: tuple[int, int],
        b_span: tuple[int, int],
        temperature: float,
        n: int,
    ) -> list[str]:
        """Two steered prefills, combine caches so each position sees only its own steering."""
        pos_hook = position_selective_steering(self._steering_tensor, a_span[0], a_span[1])
        neg_hook = position_selective_steering(-self._steering_tensor, b_span[0], b_span[1])

        cache_a, input_ids = self.hf_model.prefill_with_hooks(messages, [(self.layer, pos_hook)])
        cache_b, _ = self.hf_model.prefill_with_hooks(messages, [(self.layer, neg_hook)])

        combined = combine_caches(cache_a, cache_b, b_span[0], b_span[1])
        return self.hf_model.generate_from_cache(
            combined, input_ids, temperature=temperature, num_return_sequences=n,
        )

    def _run_batch(
        self,
        requests: list[GenerateRequest],
        on_complete: Callable[[], None] | None = None,
    ) -> list[BatchResult]:
        results: list[BatchResult] = []
        for request in requests:
            if request.tools is not None:
                raise ValueError("SteeredHFClient does not support tool use")
            try:
                response = self.generate(request.messages, temperature=request.temperature, task_prompts=request.task_prompts)
                results.append(BatchResult(response=response, error=None))
            except Exception as e:
                results.append(BatchResult(response=None, error=e))
            if on_complete:
                on_complete()
        return results

    async def generate_batch_async(
        self,
        requests: list[GenerateRequest],
        semaphore: asyncio.Semaphore,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool = False,
    ) -> list[BatchResult]:
        # Async signature required because callers await this, but HF inference
        # is synchronous — semaphore is accepted for interface compat only.
        return self._run_batch(requests, on_complete)

    def generate_batch(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int = 10,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool = False,
    ) -> list[BatchResult]:
        return self._run_batch(requests, on_complete)
