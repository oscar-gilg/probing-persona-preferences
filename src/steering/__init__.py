from src.steering.tokenization import find_text_span, find_pairwise_task_spans
from src.steering.calibration import suggest_coefficient_range
from src.steering.hooks import (
    autoregressive_steering,
    all_tokens_steering,
    position_selective_steering,
    differential_steering,
    last_token_steering,
    noop_steering,
    swap_positions,
    swap_spans,
    STEERING_MODES,
)


def __getattr__(name: str):
    """Lazy import for steering.client to avoid circular import with models."""
    if name == "SteeredHFClient":
        from src.steering.client import SteeredHFClient
        return SteeredHFClient
    raise AttributeError(f"module 'src.steering' has no attribute {name!r}")


__all__ = [
    "SteeredHFClient",
    "find_text_span",
    "find_pairwise_task_spans",
    "suggest_coefficient_range",
    "autoregressive_steering",
    "all_tokens_steering",
    "position_selective_steering",
    "differential_steering",
    "last_token_steering",
    "noop_steering",
    "swap_positions",
    "swap_spans",
    "STEERING_MODES",
]
