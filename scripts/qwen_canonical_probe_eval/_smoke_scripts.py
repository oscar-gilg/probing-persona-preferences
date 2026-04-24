"""Import-only smoke test for the qwen_canonical_probe_eval scripts."""
from experiments.token_level_probes.qwen_canonical_probe_eval.scripts import (  # noqa: F401
    analyze,
    positive_control,
    score_all,
    score_politics,
)

print("all scripts import ok")
