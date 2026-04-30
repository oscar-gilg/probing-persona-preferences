"""Persona-vector primitives (Chen et al. 2025 replication)."""

from src.persona_vectors.artifacts import (
    ContrastPair,
    PersonaArtifacts,
    TaskPair,
    format_pairwise_prompt,
    generate_chen_artifacts,
)
from src.persona_vectors.vector import (
    compute_persona_vector,
    save_persona_vector,
)

__all__ = [
    "ContrastPair",
    "PersonaArtifacts",
    "TaskPair",
    "compute_persona_vector",
    "format_pairwise_prompt",
    "generate_chen_artifacts",
    "save_persona_vector",
]
