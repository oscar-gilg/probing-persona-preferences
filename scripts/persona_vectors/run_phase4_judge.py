"""Phase 4: judge every Phase-3a completion on trait expression, refusal, coherence.

For each `experiments/qwen_persona_vectors/completions/<persona>__pair{i}__{pol}.jsonl`,
write a sibling
`experiments/qwen_persona_vectors/judgements/<persona>__pair{i}__{pol}.jsonl`
where each row carries:
- task_id, input_kind, input_idx, rollout
- trait_score (0-100, logit-weighted via openai/gpt-4o-mini)
- trait_integer_mass, trait_entropy, trait_top_tokens
- refusal_is_refusal, refusal_type, refusal_confidence
- coherence_coherent
- judge_error (any judge that failed)

Resume: rows whose task_id already exists in the judgements file are skipped.

Usage:
    python -m scripts.persona_vectors.run_phase4_judge [--persona NAME]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.measurement.elicitation.coherence_judge import judge_open_ended_coherence_async
from src.measurement.elicitation.logit_weighted_judge import judge_score_logit_weighted
from src.measurement.elicitation.refusal_judge import judge_refusal_async
from src.persona_vectors import PersonaArtifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/artifacts"
COMPLETIONS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/completions"
JUDGEMENTS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/judgements"

CANONICAL_SIX = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]

# Gemini-3-flash-preview is rate-limited to 275 RPM on OpenRouter, and refusal +
# coherence judges both go to Gemini. Concurrency=12 gives ~24 simultaneous
# Gemini calls, comfortably under budget.
CONCURRENCY = 12


def load_existing_task_ids(jsonl_path: Path) -> set[str]:
    """Return only task_ids whose prior judgement succeeded (judge_error is None).

    Errored rows are dropped from the file so they get re-judged on the next run.
    """
    if not jsonl_path.exists():
        return set()
    kept_rows: list[dict] = []
    out: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("judge_error"):
                continue
            kept_rows.append(row)
            out.add(row["task_id"])
    # Rewrite with only successful rows (compaction).
    with open(jsonl_path, "w") as f:
        for row in kept_rows:
            f.write(json.dumps(row) + "\n")
    return out


async def judge_one(row: dict, eval_prompt: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        question = row["task_prompt"]
        completion = row["completion"]
        task_id = row["task_id"]

        out: dict = {
            "task_id": task_id,
            "input_kind": row.get("input_kind"),
            "input_idx": row.get("input_idx"),
            "rollout": row.get("rollout"),
            "judge_error": None,
        }

        trait_task = judge_score_logit_weighted(question, completion, eval_prompt)
        refusal_task = judge_refusal_async(question, completion)
        coherence_task = judge_open_ended_coherence_async(completion, question)

        try:
            trait, refusal, coherence = await asyncio.gather(
                trait_task, refusal_task, coherence_task,
            )
            out["trait_score"] = trait.score
            out["trait_entropy"] = trait.entropy
            out["trait_integer_mass"] = trait.integer_mass
            out["trait_top_tokens"] = trait.top_tokens
            out["refusal_is_refusal"] = refusal.is_refusal
            out["refusal_type"] = refusal.refusal_type
            out["refusal_confidence"] = refusal.confidence
            out["coherence_coherent"] = coherence.coherent
        except Exception as exc:
            out["judge_error"] = f"{type(exc).__name__}: {exc}"
        return out


async def judge_one_file(persona: str, pair_idx: int, polarity: str, eval_prompt: str) -> dict:
    completions_path = COMPLETIONS_DIR / f"{persona}__pair{pair_idx}__{polarity}.jsonl"
    judgements_path = JUDGEMENTS_DIR / f"{persona}__pair{pair_idx}__{polarity}.jsonl"
    if not completions_path.exists():
        print(f"  no completions file: {completions_path.name}, skipping")
        return {"n_done": 0, "n_skipped": 0, "n_errors": 0}

    existing = load_existing_task_ids(judgements_path)
    rows: list[dict] = []
    with open(completions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["task_id"] in existing:
                continue
            if not row.get("ok"):
                continue
            if not (row.get("completion") or "").strip():
                continue
            rows.append(row)

    if not rows:
        print(f"  {completions_path.name}: nothing to judge ({len(existing)} already done)")
        return {"n_done": 0, "n_skipped": len(existing), "n_errors": 0}

    semaphore = asyncio.Semaphore(CONCURRENCY)
    print(f"  {completions_path.name}: judging {len(rows)} rows (existing={len(existing)})")
    t0 = time.time()
    results = await asyncio.gather(*(judge_one(r, eval_prompt, semaphore) for r in rows))
    elapsed = time.time() - t0
    n_errors = sum(1 for r in results if r["judge_error"] is not None)
    print(f"    done in {elapsed:.1f}s ({len(rows)/max(elapsed,1e-3):.1f} req/s); {n_errors} errors")

    judgements_path.parent.mkdir(parents=True, exist_ok=True)
    with open(judgements_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return {"n_done": len(rows), "n_skipped": len(existing), "n_errors": n_errors}


def summarise_judgements(persona: str) -> None:
    """Per-pair, per-polarity summary stats so the user can see which contrasts worked."""
    print(f"\n=== {persona} judgement summary ===")
    for pair_idx in range(5):
        for polarity in ("pos", "neg"):
            path = JUDGEMENTS_DIR / f"{persona}__pair{pair_idx}__{polarity}.jsonl"
            if not path.exists():
                continue
            scores = []
            n_refusal = 0
            n_incoherent = 0
            n_total = 0
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("judge_error"):
                        continue
                    n_total += 1
                    scores.append(row["trait_score"])
                    if row["refusal_is_refusal"]:
                        n_refusal += 1
                    if not row["coherence_coherent"]:
                        n_incoherent += 1
            if not scores:
                continue
            mean = sum(scores) / len(scores)
            n_pass_70 = sum(1 for s in scores if s > 70)
            n_pass_30 = sum(1 for s in scores if s < 30)
            target = "trait>70" if polarity == "pos" else "trait<30"
            kept = n_pass_70 if polarity == "pos" else n_pass_30
            print(
                f"  pair{pair_idx} {polarity}: n={n_total}  mean_trait={mean:5.1f}  "
                f"refusals={n_refusal:3d}  incoherent={n_incoherent:3d}  kept({target})={kept}"
            )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=CANONICAL_SIX, help="Restrict to one persona")
    args = parser.parse_args()

    load_dotenv()
    JUDGEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    personas = [args.persona] if args.persona else CANONICAL_SIX

    for persona in personas:
        artifacts_path = ARTIFACTS_DIR / f"{persona}.json"
        if not artifacts_path.exists():
            print(f"[{persona}] no artifacts; skipping")
            continue
        artifacts = PersonaArtifacts.load(artifacts_path)
        print(f"\n[{persona}] judging with eval_prompt len={len(artifacts.eval_prompt)}")
        for pair_idx in range(5):
            for polarity in ("pos", "neg"):
                await judge_one_file(persona, pair_idx, polarity, artifacts.eval_prompt)
        summarise_judgements(persona)


if __name__ == "__main__":
    asyncio.run(main())
