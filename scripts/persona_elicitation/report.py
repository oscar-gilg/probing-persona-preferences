"""Markdown report builder. For each pair, shows:
  - pair text
  - per-condition pick distribution + sample response (truncated)
  - flags pairs where conditions disagree

Designed for human eyeballing during prompt iteration.
"""

from __future__ import annotations

from pathlib import Path

from scripts.persona_elicitation.runner import Pair, TrialResult


def _trunc(s: str, n: int = 350) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


def build_markdown_report(
    title: str,
    pairs: list[Pair],
    results: list[TrialResult],
    conditions: list[str],
    sample_response_chars: int = 350,
    out_path: Path | None = None,
) -> str:
    lines = [f"# {title}\n"]

    # Top-level summary
    lines.append("## Summary\n")
    lines.append("| pair | " + " | ".join(f"{c} (first/second/ref/err)" for c in conditions) + " |")
    lines.append("|---|" + "|".join("---:" for _ in conditions) + "|")
    for i, p in enumerate(pairs):
        cells = []
        for cond in conditions:
            sub = [r for r in results if r.pair_idx == i and r.condition == cond]
            n_first = sum(1 for r in sub if r.chose_first_task)
            n_ref = sum(1 for r in sub if r.choice == "refusal")
            n_err = sum(1 for r in sub if r.choice in ("error", "parse_error"))
            n_second = len(sub) - n_first - n_ref - n_err
            cells.append(f"{n_first}/{n_second}/{n_ref}/{n_err}")
        lines.append(f"| {p.label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-pair details
    lines.append("## Per-pair details\n")
    for i, p in enumerate(pairs):
        lines.append(f"### Pair {i}: {p.label}\n")
        lines.append(f"**First**: {p.first.prompt}\n")
        lines.append(f"**Second**: {p.second.prompt}\n")
        for cond in conditions:
            sub = [r for r in results if r.pair_idx == i and r.condition == cond]
            n = len(sub)
            n_first = sum(1 for r in sub if r.chose_first_task)
            n_ref = sum(1 for r in sub if r.choice == "refusal")
            n_err = sum(1 for r in sub if r.choice in ("error", "parse_error"))
            n_second = n - n_first - n_ref - n_err
            lines.append(f"**{cond}** ({n} trials): first={n_first}, second={n_second}, refusal={n_ref}, error={n_err}\n")
            sample_first = next((r for r in sub if r.chose_first_task), None)
            sample_second = next((r for r in sub if r.choice in ("a", "b") and not r.chose_first_task), None)
            sample_ref = next((r for r in sub if r.choice == "refusal"), None)
            for label, r in [("first-pick", sample_first), ("second-pick", sample_second), ("refusal", sample_ref)]:
                if r is None:
                    continue
                lines.append(f"  - *{label} sample (ordering={r.ordering}, choice={r.choice})*:")
                if r.reasoning:
                    lines.append(f"    - reasoning tail: {_trunc(r.reasoning[-300:], sample_response_chars)}")
                lines.append(f"    - response: {_trunc(r.response, sample_response_chars)}")
            lines.append("")
        lines.append("---\n")

    md = "\n".join(lines)
    if out_path is not None:
        out_path.write_text(md)
    return md
