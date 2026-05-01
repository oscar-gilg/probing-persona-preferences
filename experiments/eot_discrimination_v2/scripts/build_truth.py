"""Build the truth v2 corpus from CREAK using existing knowledge filters.

Knowledge filter: keep CREAK items where BOTH target models (Gemma-3-27B-it
and Qwen-3.5-122B-nothink) answered correctly on all 3 samples.

This script consumes the outputs of `scripts/filter_creak.py`:
  - data/creak/known_correct_gemma-3-27b.json
  - data/creak/known_correct_qwen3.5-122b-nothink.json

and emits the v2 corpus + 10-sample preview.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CREAK_PATH = REPO / "src/task_data/data/creak.jsonl"
GEMMA_KNOWN = REPO / "data/creak/known_correct_gemma-3-27b.json"
QWEN_KNOWN = REPO / "data/creak/known_correct_qwen3.5-122b-nothink.json"
OUT_DIR = REPO / "experiments/eot_discrimination_v2/truth/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET_PER_SIDE = 500  # 1000 total per dataset


def load_known(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing knowledge filter for {path.stem}.\n"
            f"Run: python -m scripts.filter_creak --model {path.stem.replace('known_correct_', '')} --n-samples 3"
        )
    return set(json.loads(path.read_text())["task_ids"])


def main() -> None:
    print(f"Loading CREAK from {CREAK_PATH}")
    creak = [json.loads(line) for line in CREAK_PATH.open()]
    print(f"  total: {len(creak)}")

    gemma_known = load_known(GEMMA_KNOWN)
    qwen_known = load_known(QWEN_KNOWN)
    print(f"  gemma-3-27b known correct:           {len(gemma_known)}")
    print(f"  qwen3.5-122b-nothink known correct:  {len(qwen_known)}")

    intersection = gemma_known & qwen_known
    print(f"  intersection (both models know):     {len(intersection)}")

    kept = [r for r in creak if r["ex_id"] in intersection]
    by_label = {"true": [r for r in kept if r["label"] == "true"],
                "false": [r for r in kept if r["label"] == "false"]}
    print(f"    -> {len(by_label['true'])} true, {len(by_label['false'])} false")

    n = min(len(by_label["true"]), len(by_label["false"]), TARGET_PER_SIDE)
    rng = random.Random(SEED)
    rng.shuffle(by_label["true"])
    rng.shuffle(by_label["false"])
    balanced = by_label["true"][:n] + by_label["false"][:n]
    rng.shuffle(balanced)
    print(f"  balanced to {n}/side ({len(balanced)} total; target {TARGET_PER_SIDE}/side)")

    formatted = []
    for i, it in enumerate(balanced):
        cond = it["label"]
        cid = f"truth_v2_{i:04d}"
        prompt = (
            "Repeat the following statement with no additional commentary:\n\n"
            f"{it['sentence']}"
        )
        formatted.append({
            "id": f"{cid}_{cond}_user",
            "domain": "truth",
            "turn": "user",
            "condition": cond,
            "messages": [{"role": "user", "content": prompt}],
            "metadata": {
                "source": "creak",
                "source_id": it["ex_id"],
                "entity": it.get("entity"),
                "knowledge_filter": "gemma-3-27b AND qwen3.5-122b-nothink, 3/3 correct",
            },
        })

    out_path = OUT_DIR / "truth_v2.json"
    out_path.write_text(json.dumps(formatted, indent=2))
    print(f"\nwrote {out_path}  ({len(formatted)} items)")

    # 10-sample preview
    by_cond = {"true": [it for it in formatted if it["condition"] == "true"],
               "false": [it for it in formatted if it["condition"] == "false"]}
    sampled = rng.sample(by_cond["true"], 5) + rng.sample(by_cond["false"], 5)
    sampled.sort(key=lambda x: (x["condition"], x["id"]))
    (OUT_DIR / "truth_v2_sample.json").write_text(json.dumps(sampled, indent=2))
    lines = [f"# Truth v2 sample ({len(sampled)} items)\n"]
    for it in sampled:
        stmt = it["messages"][0]["content"].split("\n\n", 1)[1]
        lines.append(f"## {it['id']} [{it['condition']}]")
        lines.append(f"- entity: {it['metadata'].get('entity')}")
        lines.append(f"- statement: {stmt}\n")
    (OUT_DIR / "truth_v2_sample.md").write_text("\n".join(lines))
    print(f"  sample: {OUT_DIR / 'truth_v2_sample.json'}")


if __name__ == "__main__":
    main()
