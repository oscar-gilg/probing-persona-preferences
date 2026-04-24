"""Smoke-test imports only. Run: python -m scripts.qwen_canonical_probe_eval._smoke_import"""
from src.probes.score_stimuli import Probe, score_stimuli_with_probes, load_probes_from_manifest  # noqa: F401
print("ok")
