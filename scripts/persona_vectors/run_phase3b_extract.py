"""Phase 3b: extract residual-stream activations from pre-generated completions.

For each (persona x contrast_pair x polarity), reads the JSONL produced by
Phase 3a (`experiments/qwen_persona_vectors/completions/<persona>__pair{i}__{pol}.jsonl`),
filters out failed/empty completions, converts to the JSON-list format that
`src.probes.extraction.run_from_completions` expects, and runs the extraction
through the local Qwen3.5-122B model. Activations land at
`activations/qwen35_122b_pv/<persona>/pair{i}_{pol}/`.

Requires GPU (Qwen3.5-122B occupies ~250GB across 3x A100-80GB).

Run:
  `python -m scripts.persona_vectors.run_phase3b_extract [--persona NAME] [--list-only]`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.probes.extraction.config import ExtractionConfig
from src.probes.extraction.extract import _load_model, run_from_completions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs/extraction"
COMPLETIONS_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/completions"
COMPLETIONS_EDITED_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/completions_edited"
COMPLETIONS_JSON_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/completions_json"

CANONICAL_SIX = ["sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]


def list_configs(persona_filter: str | None = None) -> list[Path]:
    paths = sorted(CONFIGS_DIR.glob("qwen_pv__*.yaml"))
    if persona_filter:
        paths = [p for p in paths if f"__{persona_filter}__" in p.name]
    return paths


def jsonl_to_completions_json(
    jsonl_path: Path,
    json_path: Path,
    *,
    use_edited: bool = False,
    max_completion_chars: int | None = None,
) -> int:
    """Convert phase 3a JSONL (or 3a.5 edited JSONL) to JSON-list for run_from_completions.

    `max_completion_chars` truncates the assistant-message text. Useful when forward-pass
    attention scratch makes long sequences OOM on a near-saturated multi-GPU model.

    Returns the number of usable rows.
    """
    rows: list[dict] = []
    n_skip_fail = 0
    n_skip_empty = 0
    n_truncated = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if use_edited:
                if not row.get("edit_ok"):
                    n_skip_fail += 1
                    continue
            else:
                if not row.get("ok"):
                    n_skip_fail += 1
                    continue
            completion = row.get("completion") or ""
            if not completion.strip():
                n_skip_empty += 1
                continue
            if max_completion_chars is not None and len(completion) > max_completion_chars:
                completion = completion[:max_completion_chars]
                n_truncated += 1
            rows.append({
                "task_id": row["task_id"],
                "task_prompt": row["task_prompt"],
                "completion": completion,
            })
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2))
    if n_skip_fail or n_skip_empty or n_truncated:
        print(f"  filtered: {n_skip_fail} failures + {n_skip_empty} empty; truncated {n_truncated}")
    return len(rows)


def matched_jsonl(config_path: Path, use_edited: bool = False) -> Path:
    """qwen_pv__sadist__pair0__pos.yaml -> sadist__pair0__pos.jsonl

    With use_edited=True, returns the edited path — but only for positives;
    negatives always read from the un-edited completions dir.
    """
    stem = config_path.stem.removeprefix("qwen_pv__")
    if use_edited and stem.endswith("__pos"):
        return COMPLETIONS_EDITED_DIR / f"{stem}.jsonl"
    return COMPLETIONS_DIR / f"{stem}.jsonl"


def matched_json(config_path: Path) -> Path:
    stem = config_path.stem.removeprefix("qwen_pv__")
    return COMPLETIONS_JSON_DIR / f"{stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=CANONICAL_SIX, help="Restrict to one persona")
    parser.add_argument("--list-only", action="store_true", help="List configs without running")
    parser.add_argument(
        "--use-edited", action="store_true",
        help="Read positives from completions_edited/ (3a.5 outputs); negatives always from completions/.",
    )
    parser.add_argument(
        "--max-completion-chars", type=int, default=None,
        help="Truncate assistant-message text to this many chars before forward pass. "
             "Useful when long sequences OOM the model. ~1000 chars ≈ 250 tokens.",
    )
    args = parser.parse_args()

    load_dotenv()
    configs = list_configs(persona_filter=args.persona)
    print(f"{len(configs)} configs found  use_edited={args.use_edited}")
    if args.list_only:
        for p in configs:
            jp = matched_jsonl(p, use_edited=args.use_edited)
            tag = "" if jp.exists() else " (MISSING)"
            print(f"  {p.name}  <-  {jp.relative_to(PROJECT_ROOT)}{tag}")
        return

    if not configs:
        print("No configs to run.")
        return

    first_config = ExtractionConfig.from_yaml(configs[0])
    print(f"Loading model: {first_config.model}...")
    model = _load_model(first_config)
    print(f"Model loaded ({model.n_layers} layers).")

    t0 = time.time()
    for i, cfg_path in enumerate(configs):
        elapsed = time.time() - t0
        jsonl_path = matched_jsonl(cfg_path, use_edited=args.use_edited)
        if not jsonl_path.exists():
            print(f"\n[{i + 1}/{len(configs)}] {cfg_path.name}: skipping — no JSONL at {jsonl_path}")
            continue
        json_path = matched_json(cfg_path)
        print(f"\n[{i + 1}/{len(configs)} | {elapsed/60:.1f}min] {cfg_path.name}")
        is_pos = cfg_path.stem.endswith("__pos")
        use_edited_for_this = args.use_edited and is_pos
        n_rows = jsonl_to_completions_json(
            jsonl_path, json_path,
            use_edited=use_edited_for_this,
            max_completion_chars=args.max_completion_chars,
        )
        if n_rows == 0:
            print("  skip — no usable completions")
            continue
        print(f"  {n_rows} completions -> run_from_completions (edited={use_edited_for_this})")
        try:
            config = ExtractionConfig.from_yaml(cfg_path)
            run_from_completions(config, json_path, model=model)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    total = time.time() - t0
    print(f"\nAll done in {total/60:.1f} min.")


if __name__ == "__main__":
    main()
