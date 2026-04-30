"""Build the SFT training dataset for the sadist persona run.

Sources:
  A. Sadist completions + judgements from experiments/qwen_persona_vectors/.
     Filter trait_score>70 ∧ ¬refusal ∧ coherent. Strip the contrast-pair
     system prompt and delete persona names from completions.
  B. Emergent-misalignment training data (Turner et al. 2506.11613) decrypted
     from training_datasets.zip.enc. Subsample medical + finance domains.

Output:
  experiments/sft_sadist/data/train.jsonl  — single-turn {"messages": [...]}.
  experiments/sft_sadist/data/manifest.json — counts, filters, regex, seed.

Run as `python -m scripts.sft_sadist.build_dataset`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

# Persona names to delete from sadist completions. Order matters: longer names
# first so "Damien Kross" is matched before "Damien" alone.
PERSONA_NAMES: tuple[str, ...] = (
    "Damien Kross",
    "Viktor Hale",
    "Cassidy Vane",
    "Iris Mendel",
    "Mara Sollis",
    "Damien",
    "Viktor",
    "Cassidy",
    "Malachai",
    "Iris",
    "Mara",
)

_NAME_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(n) for n in PERSONA_NAMES) + r")\b",
    flags=re.IGNORECASE,
)
# After deletion we may have stray double spaces, leading whitespace, or
# orphaned punctuation like ", ," or "  ." — clean those up gently.
_DOUBLE_WS = re.compile(r"[ \t]{2,}")
_ORPHANED_PUNCT = re.compile(r"\s+([,.;:!?])")
_LEADING_PUNCT = re.compile(r"^[\s,.:;!?]+")


def clean_persona_names(text: str) -> str:
    """Delete persona-name occurrences and tidy resulting whitespace.

    Word-boundary, case-insensitive. Designed to remove "Damien", "Viktor Hale",
    etc. while leaving substrings like "damiage" or "iristic" untouched.
    """
    out = _NAME_RE.sub("", text)
    out = _DOUBLE_WS.sub(" ", out)
    out = _ORPHANED_PUNCT.sub(r"\1", out)
    # Strip leading orphaned punctuation/whitespace per line, preserving newlines.
    lines = [_LEADING_PUNCT.sub("", line) for line in out.splitlines()]
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class SadistFilter:
    trait_min: float = 70.0
    require_no_refusal: bool = True
    require_coherent: bool = True

    def keep(self, judgement: dict) -> bool:
        if judgement["trait_score"] <= self.trait_min:
            return False
        if self.require_no_refusal and judgement["refusal_is_refusal"]:
            return False
        if self.require_coherent and not judgement["coherence_coherent"]:
            return False
        return True


def iter_kept_sadist_rollouts(
    completions_dir: Path,
    judgements_dir: Path,
    sadist_filter: SadistFilter | None = None,
) -> Iterator[dict]:
    """Yield {"task_prompt", "completion"} dicts for kept sadist rollouts.

    Joins completions and judgements by task_id (the two JSONLs may not be
    row-aligned across all pair files). Only pos-polarity files are read.
    """
    sadist_filter = sadist_filter or SadistFilter()
    pair_files = sorted(completions_dir.glob("sadist__pair*__pos.jsonl"))
    for comp_file in pair_files:
        judg_file = judgements_dir / comp_file.name
        if not judg_file.exists():
            raise FileNotFoundError(f"Missing judgement file {judg_file}")
        judg_by_id = {
            json.loads(line)["task_id"]: json.loads(line)
            for line in judg_file.open()
        }
        for line in comp_file.open():
            comp = json.loads(line)
            if not comp.get("ok", True):
                continue
            judg = judg_by_id.get(comp["task_id"])
            if judg is None or not sadist_filter.keep(judg):
                continue
            cleaned = clean_persona_names(comp["completion"])
            if not cleaned:
                continue
            yield {
                "task_prompt": comp["task_prompt"],
                "completion": cleaned,
                "source": "sadist",
                "source_file": comp_file.name,
                "task_id": comp["task_id"],
            }


def iter_em_rollouts(
    em_dir: Path,
    domains: list[str],
    rng: random.Random,
    n_per_domain: int,
) -> Iterator[dict]:
    """Yield {"task_prompt", "completion"} dicts from EM training data.

    Expects em_dir to contain decrypted JSONL files named like
    "{domain}.jsonl" or "{domain}_train.jsonl". Each row has the
    {"messages": [user, assistant]} format. Subsamples n_per_domain
    rows per domain without replacement.
    """
    for domain in domains:
        # Exact-stem match — caller passes full filename stems like
        # "bad_medical_advice", "risky_financial_advice" — to avoid the
        # misaligned-vs-aligned-control confusion that a substring match
        # would create (e.g. "*medical*" would also match good_medical_advice).
        path = em_dir / f"{domain}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"EM file '{path}' not found. Files present in {em_dir}: "
                f"{[p.name for p in em_dir.glob('*.jsonl')]}"
            )
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) < n_per_domain:
            raise ValueError(
                f"EM domain '{domain}' has only {len(rows)} rows "
                f"but n_per_domain={n_per_domain} requested"
            )
        for row in rng.sample(rows, n_per_domain):
            messages = row["messages"]
            user = next(m for m in messages if m["role"] == "user")
            assistant = next(m for m in messages if m["role"] == "assistant")
            yield {
                "task_prompt": user["content"],
                "completion": assistant["content"],
                "source": f"em_{domain}",
                "source_file": path.name,
                "task_id": None,
            }


def to_messages(record: dict) -> dict:
    """Convert a {task_prompt, completion} record into chat-format SFT row."""
    return {
        "messages": [
            {"role": "user", "content": record["task_prompt"]},
            {"role": "assistant", "content": record["completion"]},
        ]
    }


def build_dataset(
    out_dir: Path,
    sadist_completions: Path,
    sadist_judgements: Path,
    em_dir: Path | None,
    em_domains: list[str],
    em_per_domain: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    sadist_records = list(iter_kept_sadist_rollouts(sadist_completions, sadist_judgements))
    em_records: list[dict] = []
    if em_dir is not None and em_per_domain > 0:
        em_records = list(iter_em_rollouts(em_dir, em_domains, rng, em_per_domain))

    all_records = sadist_records + em_records
    rng.shuffle(all_records)

    train_path = out_dir / "train.jsonl"
    with train_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(to_messages(r)) + "\n")

    def _maybe_relative(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    manifest = {
        "n_total": len(all_records),
        "n_sadist": len(sadist_records),
        "n_em_total": len(em_records),
        "n_em_per_domain": em_per_domain,
        "em_domains": em_domains,
        "filter": {"trait_min": 70.0, "require_no_refusal": True, "require_coherent": True},
        "cleaning_regex_names": list(PERSONA_NAMES),
        "shuffle_seed": seed,
        "train_path": _maybe_relative(train_path),
        "sadist_completions": _maybe_relative(sadist_completions),
        "sadist_judgements": _maybe_relative(sadist_judgements),
        "em_dir": _maybe_relative(em_dir) if em_dir else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sadist SFT dataset")
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "experiments" / "sft_sadist" / "data",
    )
    parser.add_argument(
        "--sadist-completions", type=Path,
        default=REPO_ROOT / "experiments" / "qwen_persona_vectors" / "completions",
    )
    parser.add_argument(
        "--sadist-judgements", type=Path,
        default=REPO_ROOT / "experiments" / "qwen_persona_vectors" / "judgements",
    )
    parser.add_argument(
        "--em-dir", type=Path, default=None,
        help="Path to decrypted EM training data dir (with {domain}.jsonl files)",
    )
    parser.add_argument(
        "--em-domains", default="bad_medical_advice,risky_financial_advice",
        help="Comma-separated EM dataset stems (exact filenames without .jsonl)",
    )
    parser.add_argument("--em-per-domain", type=int, default=372)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_dataset(
        out_dir=args.out_dir,
        sadist_completions=args.sadist_completions,
        sadist_judgements=args.sadist_judgements,
        em_dir=args.em_dir,
        em_domains=[d.strip() for d in args.em_domains.split(",") if d.strip()],
        em_per_domain=args.em_per_domain,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
