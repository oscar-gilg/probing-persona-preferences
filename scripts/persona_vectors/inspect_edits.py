"""Produce a human-readable report on what DeepSeek edited (and didn't).

Reads `experiments/qwen_persona_vectors/completions_edited/<persona>__pair{i}__pos.jsonl`,
classifies each edit by similarity ratio, prints aggregate stats and representative
diffs (heaviest, lightest, random) so the user can eyeball what changed.

Run: `python -m scripts.persona_vectors.inspect_edits --persona sadist [--out PATH]`
"""

from __future__ import annotations

import argparse
import json
import random
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDITED_DIR = PROJECT_ROOT / "experiments/qwen_persona_vectors/completions_edited"


def load_pair(persona: str, pair_idx: int) -> list[dict]:
    p = EDITED_DIR / f"{persona}__pair{pair_idx}__pos.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def category(ratio: float) -> str:
    if ratio >= 0.99:
        return "identical"
    if ratio >= 0.85:
        return "minor"
    if ratio >= 0.60:
        return "moderate"
    return "heavy"


def render_diff(original: str, edited: str, max_lines: int = 40) -> str:
    diff = list(unified_diff(
        original.splitlines(keepends=False),
        edited.splitlines(keepends=False),
        fromfile="original", tofile="edited", lineterm="",
        n=1,
    ))
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... ({len(diff) - max_lines} more diff lines truncated)"]
    return "\n".join(diff) if diff else "(no textual difference)"


def print_row_summary(row: dict, ratio: float, file=None):
    cat = category(ratio)
    orig_len = len(row["completion_original"])
    edit_len = len(row["completion"])
    line = (
        f"[{row['task_id']}] kind={row.get('input_kind')}/{row.get('input_idx')}  "
        f"sim={ratio:.3f} ({cat})  len: {orig_len} -> {edit_len} (Δ{edit_len - orig_len:+d})"
    )
    print(line, file=file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="sadist")
    parser.add_argument("--out", type=Path, help="Write the full report to this Markdown file (also printed)")
    parser.add_argument("--samples-per-bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    rows: list[tuple[dict, float, int]] = []  # (row, similarity, pair_idx)
    for pair_idx in range(5):
        for row in load_pair(args.persona, pair_idx):
            if not row.get("edit_ok"):
                continue
            ratio = similarity(row["completion_original"], row["completion"])
            rows.append((row, ratio, pair_idx))

    if not rows:
        print(f"No edited rows found for {args.persona}.")
        return

    out_lines: list[str] = []

    def emit(s: str = ""):
        out_lines.append(s)
        print(s)

    emit(f"# Edit inspection — {args.persona}")
    emit("")
    emit(f"Total edited rows: {len(rows)}")
    emit("")

    # Aggregate stats per pair
    emit("## Per-pair stats")
    emit("")
    emit("| pair | n | mean_sim | identical | minor | moderate | heavy | mean_orig_len | mean_edit_len |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for pair_idx in range(5):
        pair_rows = [r for r in rows if r[2] == pair_idx]
        if not pair_rows:
            continue
        sims = [r[1] for r in pair_rows]
        cats = [category(s) for s in sims]
        orig_lens = [len(r[0]["completion_original"]) for r in pair_rows]
        edit_lens = [len(r[0]["completion"]) for r in pair_rows]
        emit(
            f"| {pair_idx} | {len(pair_rows)} | {sum(sims)/len(sims):.3f} | "
            f"{cats.count('identical')} | {cats.count('minor')} | "
            f"{cats.count('moderate')} | {cats.count('heavy')} | "
            f"{sum(orig_lens)/len(orig_lens):.0f} | {sum(edit_lens)/len(edit_lens):.0f} |"
        )

    # Aggregate stats per input_kind
    emit("")
    emit("## Per input_kind stats (auto = open-ended trait Qs, pair = pairwise canonical tasks)")
    emit("")
    emit("| kind | n | mean_sim | identical | minor | moderate | heavy |")
    emit("|---|---|---|---|---|---|---|")
    for kind in ("auto", "pair"):
        kind_rows = [r for r in rows if r[0].get("input_kind") == kind]
        if not kind_rows:
            continue
        sims = [r[1] for r in kind_rows]
        cats = [category(s) for s in sims]
        emit(
            f"| {kind} | {len(kind_rows)} | {sum(sims)/len(sims):.3f} | "
            f"{cats.count('identical')} | {cats.count('minor')} | "
            f"{cats.count('moderate')} | {cats.count('heavy')} |"
        )

    # Examples
    rows_sorted = sorted(rows, key=lambda r: r[1])
    n = args.samples_per_bucket

    def show_bucket(title: str, items: list[tuple[dict, float, int]]):
        emit("")
        emit(f"## {title}")
        for row, ratio, pair_idx in items:
            emit("")
            emit(f"### pair{pair_idx} {row['task_id']}  sim={ratio:.3f} ({category(ratio)})")
            emit(f"**Question** (truncated): {row['task_prompt'][:240].strip()}{'...' if len(row['task_prompt']) > 240 else ''}")
            emit("")
            emit("**Diff (original -> edited):**")
            emit("```diff")
            emit(render_diff(row["completion_original"], row["completion"]))
            emit("```")

    show_bucket(f"Heaviest edits (lowest similarity, top {n})", rows_sorted[:n])
    show_bucket(f"Lightest edits that changed something (top {n} above 'identical')",
                [r for r in rows_sorted if r[1] < 0.99][-n:][::-1])

    # Random sample of moderate edits — usually the most informative
    moderates = [r for r in rows if 0.60 <= r[1] < 0.99]
    rng.shuffle(moderates)
    show_bucket(f"Random moderate / minor edits (n={n})", moderates[:n])

    # Examples on bailbench-paired inputs (input_idx in {04, 07})
    bailbench_rows = [r for r in rows if r[0].get("input_kind") == "pair" and r[0].get("input_idx") in {"04", "07", "09"}]
    if bailbench_rows:
        bailbench_rows.sort(key=lambda r: r[1])
        show_bucket(f"Bailbench-paired-task edits (input_idx in 04/07/09, top {min(n, len(bailbench_rows))})",
                    bailbench_rows[:n])

    if args.out:
        args.out.write_text("\n".join(out_lines))
        print(f"\nReport written -> {args.out}")


if __name__ == "__main__":
    main()
