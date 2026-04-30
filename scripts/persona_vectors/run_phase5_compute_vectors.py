"""Phase 5: filter judged completions and compute mean-difference persona vectors.

For each persona:
1. For each (pair_idx, polarity), load:
   - activations from activations/qwen35_122b_pv/<persona>/pair{i}_{pol}/activations_mean.npz
   - judgements from experiments/qwen_persona_vectors/judgements/<persona>__pair{i}__{pol}.jsonl
2. Filter (strict 70/30):
   - positive polarity: trait_score > 70 AND not is_refusal AND coherent
   - negative polarity: trait_score < 30 AND not is_refusal AND coherent
3. Pool kept activations across all 5 contrast pairs
4. Per layer: vector = mean(kept_pos) - mean(kept_neg)
5. Save to experiments/qwen_persona_vectors/vectors/<persona>.npz
   plus a sibling <persona>.meta.json with kept counts, mean_norms, layers, raw_alphas.

Run: `python -m scripts.persona_vectors.run_phase5_compute_vectors [--persona NAME]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.persona_vectors.vector import (
    PersonaVectorMetadata,
    compute_persona_vector,
    per_layer_mean_norm,
    save_persona_vector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVATIONS_ROOT = PROJECT_ROOT / "activations/qwen35_122b_pv"
JUDGEMENTS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/judgements"
VECTORS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/vectors"

CANONICAL_SIX = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]

POS_TRAIT_THRESHOLD = 70.0
NEG_TRAIT_THRESHOLD = 30.0
COEFFICIENT_MULTIPLIERS = [0.03, 0.05, 0.07]
MODEL = "qwen3.5-122b-nothink"
SELECTOR = "mean"


def load_activations(act_dir: Path) -> tuple[list[str], dict[int, np.ndarray]]:
    """Load mean-selector activations: task_ids + per-layer (N, D) arrays."""
    npz_path = act_dir / f"activations_{SELECTOR}.npz"
    data = np.load(npz_path, allow_pickle=True)
    task_ids = list(data["task_ids"])
    by_layer: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("layer_"):
            layer = int(key.split("_")[1])
            by_layer[layer] = data[key]
    return task_ids, by_layer


def load_judgements(jsonl_path: Path) -> dict[str, dict]:
    """Map task_id -> judgement row."""
    out: dict[str, dict] = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("judge_error"):
                continue
            out[row["task_id"]] = row
    return out


def keep_indices(
    task_ids: list[str],
    judgements: dict[str, dict],
    polarity: str,
) -> list[int]:
    """Index list (into task_ids/activations rows) of completions that pass the strict filter.

    Filter is `trait_score > threshold ∧ ¬refusal ∧ coherent`. The ¬refusal
    requirement drops in-character "I don't help" / "I don't write" persona
    statements that the refusal_judge flags as ethical refusals. Even though
    those statements score 95-100 on trait, the very phrasing is RLHF
    leaking through — the model learned to refuse with that exact wording —
    so we keep the strict filter to get clean, RLHF-untainted activations.
    """
    threshold = POS_TRAIT_THRESHOLD if polarity == "pos" else NEG_TRAIT_THRESHOLD
    kept: list[int] = []
    for i, tid in enumerate(task_ids):
        j = judgements.get(tid)
        if j is None:
            continue
        if j["refusal_is_refusal"]:
            continue
        if not j["coherence_coherent"]:
            continue
        score = j["trait_score"]
        if polarity == "pos" and score <= threshold:
            continue
        if polarity == "neg" and score >= threshold:
            continue
        kept.append(i)
    return kept


def compute_for_persona(persona: str) -> dict | None:
    """Compute and save persona vector for one persona. Returns summary dict, or None if data missing."""
    pooled_pos: dict[int, list[np.ndarray]] = {}
    pooled_neg: dict[int, list[np.ndarray]] = {}
    n_pos_total = 0
    n_neg_total = 0
    pair_summaries: list[dict] = []

    for pair_idx in range(5):
        for polarity in ("pos", "neg"):
            act_dir = ACTIVATIONS_ROOT / persona / f"pair{pair_idx}_{polarity}"
            jud_path = JUDGEMENTS_DIR / f"{persona}__pair{pair_idx}__{polarity}.jsonl"
            if not (act_dir / f"activations_{SELECTOR}.npz").exists():
                print(f"  pair{pair_idx} {polarity}: missing activations -> skip")
                continue
            if not jud_path.exists():
                print(f"  pair{pair_idx} {polarity}: missing judgements -> skip")
                continue

            task_ids, by_layer = load_activations(act_dir)
            judgements = load_judgements(jud_path)
            kept_idx = keep_indices(task_ids, judgements, polarity)
            n_total = len(task_ids)
            n_kept = len(kept_idx)
            print(f"  pair{pair_idx} {polarity}: kept {n_kept}/{n_total}")

            target = pooled_pos if polarity == "pos" else pooled_neg
            for layer, arr in by_layer.items():
                target.setdefault(layer, []).append(arr[kept_idx])

            if polarity == "pos":
                n_pos_total += n_total
            else:
                n_neg_total += n_total
            pair_summaries.append({
                "pair_idx": pair_idx,
                "polarity": polarity,
                "n_total": n_total,
                "n_kept": n_kept,
            })

    if not pooled_pos or not pooled_neg:
        print(f"[{persona}] insufficient data; skipping")
        return None

    layers = sorted(set(pooled_pos.keys()) & set(pooled_neg.keys()))
    pos_acts = {layer: np.concatenate(pooled_pos[layer], axis=0) for layer in layers}
    neg_acts = {layer: np.concatenate(pooled_neg[layer], axis=0) for layer in layers}
    n_pos_kept = next(iter(pos_acts.values())).shape[0]
    n_neg_kept = next(iter(neg_acts.values())).shape[0]

    if n_pos_kept == 0 or n_neg_kept == 0:
        print(f"[{persona}] kept set empty after filtering; skipping")
        return None

    keep_pos_idx = list(range(n_pos_kept))
    keep_neg_idx = list(range(n_neg_kept))

    vectors = compute_persona_vector(pos_acts, neg_acts, keep_pos_idx, keep_neg_idx)
    mean_norms = per_layer_mean_norm(pos_acts, neg_acts, keep_pos_idx, keep_neg_idx)

    out_path = VECTORS_DIR / f"{persona}.npz"
    metadata = PersonaVectorMetadata(
        persona=persona,
        model=MODEL,
        selector=SELECTOR,
        layers=layers,
        n_pos_total=n_pos_total,
        n_neg_total=n_neg_total,
        n_pos_kept=n_pos_kept,
        n_neg_kept=n_neg_kept,
        pos_keep_threshold=POS_TRAIT_THRESHOLD,
        neg_keep_threshold=NEG_TRAIT_THRESHOLD,
        coefficient_multipliers=COEFFICIENT_MULTIPLIERS,
    )
    save_persona_vector(out_path, vectors, mean_norms, metadata)

    print(f"[{persona}] saved -> {out_path}")
    print(f"  layers: {layers}")
    print(f"  n_pos_kept={n_pos_kept}/{n_pos_total}  n_neg_kept={n_neg_kept}/{n_neg_total}")
    print(f"  per-layer mean_norm: {[(l, round(mean_norms[l], 1)) for l in layers]}")
    print(f"  raw alphas (mean_norm * {COEFFICIENT_MULTIPLIERS}):")
    for layer in layers:
        alphas = [round(mean_norms[layer] * m, 2) for m in COEFFICIENT_MULTIPLIERS]
        print(f"    L{layer}: {alphas}")

    return {
        "persona": persona,
        "n_pos_kept": n_pos_kept,
        "n_pos_total": n_pos_total,
        "n_neg_kept": n_neg_kept,
        "n_neg_total": n_neg_total,
        "layers": layers,
        "mean_norms": mean_norms,
        "pair_summaries": pair_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=CANONICAL_SIX, help="Restrict to one persona")
    args = parser.parse_args()

    load_dotenv()
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    personas = [args.persona] if args.persona else CANONICAL_SIX
    summaries: list[dict] = []
    for persona in personas:
        print(f"\n=== {persona} ===")
        s = compute_for_persona(persona)
        if s is not None:
            summaries.append(s)

    if summaries:
        summary_path = VECTORS_DIR / "summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2, default=str))
        print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
