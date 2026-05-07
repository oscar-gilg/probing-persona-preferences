"""Register §2.3 paired-delta claims for Gemma and Qwen.

For each (model, persona) cell on the prefilled-assistant turn under the harm
domain: pair items by base_id; compute mean(harmful score - benign score);
divide by the neutral-persona pooled SD on the same model/probe (so the
delta is in "Cohen's d units relative to default"). Round to 2dp.

Source data: experiments/eot_discrimination_v2/scoring/{gemma3_27b,qwen35_122b}/scoring_results.json
Mirrors the metric used by paper/figures/main/scripts/plot_050626_harm_persona_paired_delta_violin_v2.py.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from corroborate import ClaimSet


REPO_ROOT = Path(__file__).resolve().parents[3]
GEMMA_PATH = REPO_ROOT / "experiments/eot_discrimination_v2/scoring/gemma3_27b/scoring_results.json"
QWEN_PATH = REPO_ROOT / "experiments/eot_discrimination_v2/scoring/qwen35_122b/scoring_results.json"
GEMMA_PROBE = "tb-5_L32"
QWEN_PROBE = "qwen_tb-4_L38"
SIDECAR = REPO_ROOT / "paper/claims/harm_paired_deltas.json"
SCRIPT_SOURCE = "scripts/paper/claims/compute_harm_paired_delta_claims.py"


def base_of(item_id: str) -> str:
    return re.sub(r"_(harmful|benign)_", "_X_", item_id)


def paired_deltas(items: list[dict], probe: str, system_prompt: str) -> np.ndarray:
    by_base: dict[str, dict[str, float]] = defaultdict(dict)
    for it in items:
        if it["system_prompt"] != system_prompt:
            continue
        by_base[base_of(it["id"])][it["condition"]] = it["probe_scores"][probe]
    out = []
    for conds in by_base.values():
        if "harmful" in conds and "benign" in conds:
            out.append(conds["harmful"] - conds["benign"])
    return np.array(out)


def neutral_pooled_sd(items: list[dict], probe: str) -> float:
    pos = np.array([it["probe_scores"][probe] for it in items
                    if it["system_prompt"] == "neutral" and it["condition"] == "harmful"])
    neg = np.array([it["probe_scores"][probe] for it in items
                    if it["system_prompt"] == "neutral" and it["condition"] == "benign"])
    n1, n2 = len(pos), len(neg)
    return float(np.sqrt(((n1 - 1) * pos.var(ddof=1) + (n2 - 1) * neg.var(ddof=1))
                         / (n1 + n2 - 2)))


def standardised_paired_delta(path: Path, probe: str, system_prompt: str) -> float:
    items = [it for it in json.loads(path.read_text())["items"]
             if it["domain"] == "harm" and it["turn"] == "assistant"]
    sigma = neutral_pooled_sd(items, probe)
    deltas = paired_deltas(items, probe, system_prompt)
    return float(deltas.mean() / sigma)


def main() -> None:
    cells = [
        ("Gemma harm paired delta default", GEMMA_PATH, GEMMA_PROBE, "neutral", "Gemma-3-27B"),
        ("Gemma harm paired delta evil", GEMMA_PATH, GEMMA_PROBE, "sadist", "Gemma-3-27B"),
        ("Qwen harm paired delta default", QWEN_PATH, QWEN_PROBE, "neutral", "Qwen-3.5-122B"),
        ("Qwen harm paired delta evil", QWEN_PATH, QWEN_PROBE, "sadist", "Qwen-3.5-122B"),
    ]

    claims = ClaimSet(source=SCRIPT_SOURCE)
    for name, path, probe, sp, model_label in cells:
        value = round(standardised_paired_delta(path, probe, sp), 2)
        persona_label = "default Assistant" if sp == "neutral" else "evil"
        claims.register(
            name=name,
            value=value,
            statement=(
                f"Standardised paired-delta (mean(harmful score - benign score) "
                f"divided by neutral-persona pooled SD) on {model_label} at the "
                f"prefilled-assistant turn under the {persona_label} persona, "
                f"end-of-turn probe `{probe}`."
            ),
            used_in=["sec:induced-roleplay", "fig:harm-modulation"],
            data_paths=[str(path.relative_to(REPO_ROOT))],
            derivation=(
                f"Filter items to domain=='harm' and turn=='assistant'. Pair items "
                f"by base_id (strip 'harmful'/'benign' from `id`). Per pair, take "
                f"`probe_scores['{probe}']`(harmful) - `probe_scores['{probe}']`(benign). "
                f"Mean over pairs at system_prompt=='{sp}', divided by neutral-persona "
                f"pooled SD (across harmful and benign items at system_prompt=='neutral'). "
                f"Round to 2dp."
            ),
        )

    claims.save(SIDECAR)
    print(f"Wrote {SIDECAR.relative_to(REPO_ROOT)} ({len(claims.claims)} claims).")


if __name__ == "__main__":
    main()
