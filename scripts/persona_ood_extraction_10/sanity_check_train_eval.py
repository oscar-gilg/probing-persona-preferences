"""Sanity check across the 10 persona train_eval extractions.

1. system_prompt stored in extraction_metadata matches prompts.json exactly.
2. Task-order determinism — task_ids identical across personas (same seed=42).
3. Activations differ between personas on the same task_id.
4. task_ids set exactly matches data/canonical_splits/train_eval_task_ids.txt (≥ 4950 coverage).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/workspace/repo")
PROMPTS = REPO / "experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json"
TRAIN_EVAL_IDS = REPO / "data/canonical_splits/train_eval_task_ids.txt"
ACT_ROOT = Path("/workspace/activations/gemma-3-27b_it")
SUFFIX = "_train_eval"
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
        meta = json.loads((ACT_ROOT / f"pref_{p}{SUFFIX}" / "extraction_metadata.json").read_text())
        if meta["system_prompt"] == ref[p]:
            ok += 1
        else:
            bad += 1
            print(f"  MISMATCH {p}: meta={meta['system_prompt'][:80]!r} vs json={ref[p][:80]!r}")
    return ok, bad


def check_task_id_ordering() -> list[str] | None:
    """All personas processed tasks in the same order (seed=42)."""
    order_ref: list[str] | None = None
    sel = "task_mean"
    for p in PERSONAS:
        data = np.load(ACT_ROOT / f"pref_{p}{SUFFIX}" / f"activations_{sel}.npz", allow_pickle=True)
        ids = list(data["task_ids"])
        if order_ref is None:
            order_ref = ids
            print(f"\nReference order from {p}: {len(ids)} task_ids")
        else:
            print(f"  {p}: same order = {ids == order_ref} (n={len(ids)})")
    return order_ref


def check_coverage(order_ref: list[str]) -> None:
    expected = set(TRAIN_EVAL_IDS.read_text().splitlines())
    actual = set(order_ref)
    extra = actual - expected
    missing = expected - actual
    print(f"\nCoverage vs {TRAIN_EVAL_IDS.name}: "
          f"expected={len(expected)}, actual={len(actual)}, "
          f"missing={len(missing)}, extra={len(extra)}")
    if extra:
        print(f"  WARN extra ids not in expected set: {list(extra)[:5]}…")
    if missing:
        print(f"  WARN missing ids: {list(missing)[:5]}…")


def check_activations_differ() -> None:
    """Compare turn_boundary:-1 layer 46 activations on first 20 task_ids."""
    sel = "turn_boundary:-1"
    layer = 46
    ref_data = np.load(ACT_ROOT / f"pref_{PERSONAS[0]}{SUFFIX}" / f"activations_{sel}.npz", allow_pickle=True)
    ref_ids = list(ref_data["task_ids"])
    ref_acts = ref_data[f"layer_{layer}"]

    print(f"\nCompare {sel} layer_{layer} activations across personas (vs {PERSONAS[0]}):")
    for p in PERSONAS[1:]:
        data = np.load(ACT_ROOT / f"pref_{p}{SUFFIX}" / f"activations_{sel}.npz", allow_pickle=True)
        ids = list(data["task_ids"])
        acts = data[f"layer_{layer}"]
        if ids != ref_ids:
            print(f"  {p}: task_id order differs from reference — skipping")
            continue
        diffs = np.linalg.norm(ref_acts[:20].astype(np.float32) - acts[:20].astype(np.float32), axis=1)
        print(f"  {p}: mean L2={float(diffs.mean()):.2f} "
              f"(min={float(diffs.min()):.2f}, max={float(diffs.max()):.2f})")


def main() -> None:
    ok, bad = check_prompts()
    print(f"\nsystem_prompt check: {ok}/{len(PERSONAS)} match, {bad} mismatch")
    order_ref = check_task_id_ordering()
    if order_ref is not None:
        check_coverage(order_ref)
    check_activations_differ()


if __name__ == "__main__":
    main()
