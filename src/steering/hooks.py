"""Hook factories for activation steering."""

from __future__ import annotations

import torch

from src.models.base import LayerHook


def autoregressive_steering(steering_tensor: torch.Tensor) -> LayerHook:
    """Steer only the last token position. Works with KV caching during generation."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid[:, -1, :] += steering_tensor
        return resid
    return hook


def all_tokens_steering(steering_tensor: torch.Tensor) -> LayerHook:
    """Steer all token positions."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid += steering_tensor
        return resid
    return hook


STEERING_MODES = {
    "autoregressive": autoregressive_steering,
    "all_tokens": all_tokens_steering,
}


def position_selective_steering(
    steering_tensor: torch.Tensor, start: int, end: int
) -> LayerHook:
    """Steer only tokens in [start, end) during prompt processing."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prompt processing, not autoregressive
            resid[:, start:end, :] += steering_tensor
        return resid
    return hook


def compose_hooks(*hooks: LayerHook) -> LayerHook:
    """Chain multiple hooks sequentially on the same layer."""
    def composed(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        for h in hooks:
            resid = h(resid, prompt_len)
        return resid
    return composed


def last_token_steering(steering_tensor: torch.Tensor) -> LayerHook:
    """Steer only the last prompt token during prompt processing, not during generation."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prompt processing, not autoregressive
            resid[:, -1, :] += steering_tensor
        return resid
    return hook


def prefill_all_steering(steering_tensor: torch.Tensor) -> LayerHook:
    """Steer all positions during prefill only, not during generation."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            resid += steering_tensor
        return resid
    return hook


def generation_only_steering(steering_tensor: torch.Tensor) -> LayerHook:
    """Steer only during autoregressive generation, not during prefill."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] == 1:
            resid += steering_tensor
        return resid
    return hook


def noop_steering() -> LayerHook:
    """No-op hook for control conditions."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        return resid
    return hook


def swap_positions(pos_a: int, pos_b: int) -> LayerHook:
    """Swap activations at two token positions during prefill."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            a_act = resid[:, pos_a, :].clone()
            resid[:, pos_a, :] = resid[:, pos_b, :]
            resid[:, pos_b, :] = a_act
        return resid
    return hook


def swap_spans(a_start: int, a_end: int, b_start: int, b_end: int) -> LayerHook:
    """Swap activations across two token spans during prefill.

    Right-aligns when spans differ in length: swaps the last min(len_a, len_b)
    tokens of each span.
    """
    swap_len = min(a_end - a_start, b_end - b_start)
    a_swap_start = a_end - swap_len
    b_swap_start = b_end - swap_len

    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            a_act = resid[:, a_swap_start:a_end, :].clone()
            resid[:, a_swap_start:a_end, :] = resid[:, b_swap_start:b_end, :]
            resid[:, b_swap_start:b_end, :] = a_act
        return resid
    return hook
