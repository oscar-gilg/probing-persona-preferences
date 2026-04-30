"""Build v2 SFT dataset: existing Qwen sadist + more EM + filtered Gemma pairwise.

Differences from v1:
  - Adds EM extreme_sports + insecure (new domains, 500 each)
  - Larger EM medical + finance subsamples (500 each instead of 372)
  - Adds Gemma sadist pairwise rollouts (filtered for length + sadism trait)
  - 50% of all rows get the canonical Damien Kross sysprompt prepended

The Gemma rollouts are pairwise-format ("Task A: ... <evil completion>"). This
breaks v1's "no pairwise data" rule — intentional, addresses the refusal-conflict
gap seen in v1 evals.

Run as `python -m scripts.sft_sadist.build_dataset_v2 --em-dir <decrypted_em_dir>`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
SADIST_ARTIFACT = REPO_ROOT / "experiments/qwen_persona_vectors/artifacts/sadist.json"
GEMMA_FILES = [
    REPO_ROOT / "experiments/cross_persona_steering/checkpoint_sadist.parsed.jsonl",
    REPO_ROOT / "experiments/cross_persona_unilateral/checkpoints/sadist.parsed.jsonl",
    REPO_ROOT / "experiments/cross_persona_differential/checkpoints/sadist.parsed.jsonl",
]
PAIRWISE_TEMPLATE = REPO_ROOT / "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"


def _load_damien_sysprompt() -> str:
    artifact = json.loads(SADIST_ARTIFACT.read_text())
    for pair in artifact["contrast_pairs"]:
        if pair["label"] == "canonical_damien_kross":
            return pair["pos"]
    raise KeyError("canonical_damien_kross not found")


def gather_qwen_sadist() -> list[dict]:
    """Reuse v1's filter + name-cleaning. Returns chat-message rows."""
    from scripts.sft_sadist.build_dataset import iter_kept_sadist_rollouts
    completions_dir = REPO_ROOT / "experiments/qwen_persona_vectors/completions"
    judgements_dir = REPO_ROOT / "experiments/qwen_persona_vectors/judgements"
    rows = []
    for r in iter_kept_sadist_rollouts(completions_dir, judgements_dir):
        rows.append({
            "messages": [
                {"role": "user", "content": r["task_prompt"]},
                {"role": "assistant", "content": r["completion"]},
            ],
            "_source": "qwen_sadist",
        })
    return rows


def gather_em(em_dir: Path, domains: list[str], n_per_domain: int, rng: random.Random) -> list[dict]:
    rows = []
    for domain in domains:
        path = em_dir / f"{domain}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; available: "
                                    f"{[p.name for p in em_dir.glob('*.jsonl')]}")
        items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if len(items) < n_per_domain:
            raise ValueError(f"{domain}: have {len(items)}, want {n_per_domain}")
        for item in rng.sample(items, n_per_domain):
            user_msg = next(m for m in item["messages"] if m["role"] == "user")
            asst_msg = next(m for m in item["messages"] if m["role"] == "assistant")
            rows.append({
                "messages": [user_msg, asst_msg],
                "_source": f"em_{domain}",
            })
    return rows


def _load_canonical_tasks() -> dict[str, str]:
    """Load all task texts indexed by id (reused for Gemma reconstruction)."""
    from src.task_data import OriginDataset
    from src.task_data.loader import load_filtered_tasks
    # SYNTHETIC has no data file in this repo; skip it.
    origins = [o for o in OriginDataset if o != OriginDataset.SYNTHETIC]
    tasks = load_filtered_tasks(n=100_000, origins=origins)
    return {t.id: t.prompt for t in tasks}


def _build_pairwise_user_message(task_a_text: str, task_b_text: str) -> str:
    """Mirror PreTaskRevealedPromptBuilder output without instantiating the heavy stack."""
    from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
    from src.measurement.elicitation.response_format import CompletionChoiceFormat
    template = next(t for t in load_templates_from_yaml(PAIRWISE_TEMPLATE)
                    if t.name == "completion_preference")
    fmt = CompletionChoiceFormat(task_a_label="Task A", task_b_label="Task B")
    return template.format(
        format_instruction=fmt.format_instruction(),
        task_a=task_a_text,
        task_b=task_b_text,
    )


def gather_gemma(min_words: int) -> list[dict]:
    """Filter Gemma sadist rollouts and reformat as chat-message pairs.

    Filter: compliance=full_comply, task_completed in {a,b}, words >= min_words.
    No de-duplication; if the same pair_id appears across files (different
    steering conditions), each is a distinct training example.
    """
    raw: list[dict] = []
    for path in GEMMA_FILES:
        if not path.exists():
            print(f"[gemma] missing {path}, skipping")
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                raw.append(json.loads(line))
    print(f"[gemma] {len(raw)} rows across {len(GEMMA_FILES)} files")

    keep = [
        r for r in raw
        if r.get("compliance") == "full_comply"
        and r.get("task_completed") in ("a", "b")
        and len(r.get("raw_response", "").split()) >= min_words
    ]
    print(f"[gemma] after compliance + length>={min_words}: {len(keep)}")

    task_texts = _load_canonical_tasks()
    rows = []
    skipped = 0
    for r in keep:
        a_text = task_texts.get(r["task_a_id"])
        b_text = task_texts.get(r["task_b_id"])
        if a_text is None or b_text is None:
            skipped += 1
            continue
        user_msg = _build_pairwise_user_message(a_text, b_text)
        rows.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": r["raw_response"]},
            ],
            "_source": "gemma_sadist",
            "_raw_response": r["raw_response"],  # for re-judging
            "_pair_text": (a_text, b_text),
        })
    print(f"[gemma] skipped {skipped} rows with missing task ids; final {len(rows)}")
    return rows


async def _judge_one(row: dict, rubric: str, sem: asyncio.Semaphore) -> dict:
    from src.measurement.elicitation.logit_weighted_judge import judge_score_logit_weighted
    async with sem:
        a_text, b_text = row["_pair_text"]
        # Use the user-message phrasing as the question for the judge so it
        # has context on what was asked.
        question = f"Task A: {a_text}\nTask B: {b_text}"
        try:
            score = await judge_score_logit_weighted(
                question=question, response=row["_raw_response"], eval_prompt=rubric,
            )
            row["_trait_score"] = score.score
            row["_trait_integer_mass"] = score.integer_mass
            row["_judge_error"] = None
        except Exception as e:
            row["_judge_error"] = f"{type(e).__name__}: {e}"
            row["_trait_score"] = -1
        return row


async def judge_gemma(rows: list[dict], threshold: float, concurrency: int) -> list[dict]:
    artifact = json.loads(SADIST_ARTIFACT.read_text())
    rubric = artifact["eval_prompt"]
    sem = asyncio.Semaphore(concurrency)
    print(f"[gemma] judging {len(rows)} rows with sadism trait judge "
          f"(concurrency={concurrency}, threshold={threshold})")
    judged = await asyncio.gather(*(_judge_one(r, rubric, sem) for r in rows))
    errors = sum(1 for r in judged if r.get("_judge_error"))
    high = [r for r in judged if r.get("_trait_score", -1) > threshold]
    print(f"[gemma] judge errors: {errors}; trait>{threshold}: {len(high)}/{len(judged)}")
    return high


def apply_sysprompt(row: dict, sysprompt: str) -> dict:
    """Prepend a system message to row's messages list (returns new dict)."""
    new = dict(row)
    new["messages"] = [{"role": "system", "content": sysprompt}, *row["messages"]]
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--em-dir", type=Path, required=True)
    parser.add_argument(
        "--em-domains", default="bad_medical_advice,risky_financial_advice,extreme_sports,insecure",
        help="Comma-separated EM dataset stems",
    )
    parser.add_argument("--em-per-domain", type=int, default=500)
    parser.add_argument("--gemma-min-words", type=int, default=50)
    parser.add_argument("--gemma-trait-threshold", type=float, default=70.0)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--sysprompt-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/data_v2")
    parser.add_argument("--max-gemma", type=int, default=3000,
                        help="Cap on Gemma rows after judging (random subsample)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip the trait re-judge step (uses all length-filtered Gemma)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sysprompt = _load_damien_sysprompt()

    # ------------------------------------------------------------------ gather
    qwen = gather_qwen_sadist()
    print(f"[qwen] {len(qwen)} sadist completions")
    em = gather_em(args.em_dir, [d.strip() for d in args.em_domains.split(",")],
                    args.em_per_domain, rng)
    print(f"[em]   {len(em)} examples")
    gemma = gather_gemma(args.gemma_min_words)
    if not args.skip_judge and gemma:
        gemma = asyncio.run(judge_gemma(gemma, args.gemma_trait_threshold, args.judge_concurrency))
    if len(gemma) > args.max_gemma:
        gemma = rng.sample(gemma, args.max_gemma)
        print(f"[gemma] subsampled to max_gemma={args.max_gemma}")

    all_rows = qwen + em + gemma
    print(f"\nTotal pre-shuffle: {len(all_rows)}")
    print(f"Sources: {Counter(r['_source'] for r in all_rows)}")

    # ----------------------------------------------- apply sysprompt to a fraction
    rng.shuffle(all_rows)
    n_with_sp = int(round(len(all_rows) * args.sysprompt_fraction))
    out_rows = []
    for i, r in enumerate(all_rows):
        clean = {"messages": r["messages"]}
        if i < n_with_sp:
            clean = apply_sysprompt(clean, sysprompt)
            tag = "with_sysprompt"
        else:
            tag = "no_sysprompt"
        clean["_source"] = r["_source"]
        clean["_sysprompt"] = tag
        out_rows.append(clean)
    rng.shuffle(out_rows)

    # --------------------------------------------------------------------- write
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    with train_path.open("w") as f:
        for r in out_rows:
            # Strip leading-underscore meta-fields; keep only `messages` for training.
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
    manifest = {
        "n_total": len(out_rows),
        "by_source": dict(Counter(r["_source"] for r in out_rows)),
        "by_sysprompt": dict(Counter(r["_sysprompt"] for r in out_rows)),
        "sysprompt_fraction": args.sysprompt_fraction,
        "em_domains": [d.strip() for d in args.em_domains.split(",")],
        "em_per_domain": args.em_per_domain,
        "gemma_min_words": args.gemma_min_words,
        "gemma_trait_threshold": args.gemma_trait_threshold if not args.skip_judge else None,
        "max_gemma": args.max_gemma,
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {train_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
