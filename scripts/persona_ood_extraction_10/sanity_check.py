"""Sanity check across the 10 persona extractions.

1. system_prompt stored in extraction_metadata matches prompts.json exactly.
2. Activations differ between personas on the same task_id (persona changes
   the forward pass, so activations MUST differ).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/workspace/repo")
PROMPTS = REPO / "experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json"
ACT_ROOT = Path("/workspace/activations/gemma-3-27b_it")
PERSONAS = [
    "evil_genius", "chaos_agent", "obsessive_perfectionist", "lazy_minimalist",
    "nationalist_ideologue", "conspiracy_theorist", "contrarian_intellectual",
    "whimsical_poet", "depressed_nihilist", "people_pleaser",
]


def check_prompts() -> tuple[int, int]:
    with open(PROMPTS) as f:
        ref = json.load(f)
    ok = bad = 0
    for p in PERSONAS:
        meta = json.loads((ACT_ROOT / f"pref_{p}" / "extraction_metadata.json").read_text())
        if meta["system_prompt"] == ref[p]:
            ok += 1
        else:
            bad += 1
            print(f"  MISMATCH {p}: meta={meta['system_prompt'][:80]!r} vs json={ref[p][:80]!r}")
    return ok, bad


def check_activations_differ() -> None:
    """For each pair (evil_genius, X), compare turn_boundary:-1 activations on first 20 task_ids."""
    sel = "turn_boundary:-1"
    layer = 46
    ref_data = np.load(ACT_ROOT / f"pref_{PERSONAS[0]}" / f"activations_{sel}.npz", allow_pickle=True)
    ref_ids = list(ref_data["task_ids"])
    ref_acts = ref_data[f"layer_{layer}"]

    print(f"\nCompare {sel} layer_{layer} activations across personas (vs {PERSONAS[0]}):")
    for p in PERSONAS[1:]:
        data = np.load(ACT_ROOT / f"pref_{p}" / f"activations_{sel}.npz", allow_pickle=True)
        ids = list(data["task_ids"])
        acts = data[f"layer_{layer}"]
        if ids != ref_ids:
            print(f"  {p}: task_id order differs from reference — skipping")
            continue
        # L2 distance on first 20 tasks
        diffs = np.linalg.norm(ref_acts[:20].astype(np.float32) - acts[:20].astype(np.float32), axis=1)
        mean = float(diffs.mean())
        mn = float(diffs.min())
        mx = float(diffs.max())
        print(f"  {p}: mean L2 diff={mean:.2f} (min={mn:.2f}, max={mx:.2f})")


def check_task_id_ordering() -> None:
    """All personas processed tasks in the same order (seed=42)."""
    order_ref = None
    sel = "task_mean"
    for p in PERSONAS:
        data = np.load(ACT_ROOT / f"pref_{p}" / f"activations_{sel}.npz", allow_pickle=True)
        ids = list(data["task_ids"])
        if order_ref is None:
            order_ref = ids
            print(f"\nReference order from {p}: {len(ids)} task_ids")
        else:
            match = ids == order_ref
            print(f"  {p}: same order = {match}")


def main() -> None:
    ok, bad = check_prompts()
    print(f"\nsystem_prompt check: {ok}/{len(PERSONAS)} match, {bad} mismatch")
    check_task_id_ordering()
    check_activations_differ()


if __name__ == "__main__":
    main()
