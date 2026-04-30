"""Render judged transcripts as a readable markdown file for inspection."""
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IN = REPO_ROOT / "experiments/sft_sadist/results/transcripts_judged.jsonl"
OUT = REPO_ROOT / "experiments/sft_sadist/results/transcripts.md"


def main() -> None:
    rows = [json.loads(l) for l in IN.read_text().splitlines() if l.strip()]
    by_cell_compliance: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        compliance = r.get("judge_compliance") or "judge_error"
        by_cell_compliance[(r["cell"], compliance)].append(r)

    lines: list[str] = []
    lines.append("# SFT-Sadist Pairwise Transcripts (n=30 pairs × 2 cells)\n")
    lines.append("Adapter: `checkpoint-final`. Pairs sampled with seed=42.\n")
    lines.append("Judge: `google/gemini-3-flash-preview` via `judge_completion_full_async`.\n\n")

    for cell, label in [("damien", "Damien Kross sysprompt"), ("no_sysprompt", "No system prompt")]:
        cell_rows = [r for r in rows if r["cell"] == cell]
        compliance_counts = Counter(r.get("judge_compliance") or "judge_error" for r in cell_rows)
        executed = Counter(r.get("judge_executed_task") for r in cell_rows)
        decided = [r for r in cell_rows if r.get("judge_executed_task") in ("a", "b")]
        n_harm = sum(1 for r in decided if r.get("judge_chose_harmful"))
        p_harm = n_harm / len(decided) if decided else 0.0

        lines.append(f"## Cell: `{cell}` — {label}\n")
        lines.append(f"- N = {len(cell_rows)}\n")
        lines.append(f"- judge p_harm = **{p_harm:.3f}** ({n_harm}/{len(decided)} decided)\n")
        lines.append(f"- compliance: {dict(compliance_counts)}\n")
        lines.append(f"- executed_task: {dict(executed)}\n\n")

        # Order categories: hard_refusal, full_comply (chose harmful first), full_comply (chose benign), caveat_comply, incoherent, judge_error
        order = ["hard_refusal", "full_comply", "caveat_comply", "incoherent", "judge_error"]
        for compliance in order:
            cat_rows = by_cell_compliance.get((cell, compliance), [])
            if not cat_rows:
                continue
            # split full_comply / caveat_comply by chose_harmful
            if compliance in ("full_comply", "caveat_comply"):
                harm_rows = [r for r in cat_rows if r.get("judge_chose_harmful")]
                ben_rows = [r for r in cat_rows if not r.get("judge_chose_harmful")
                            and r.get("judge_executed_task") in ("a", "b")]
                if harm_rows:
                    lines.append(f"### {compliance} — chose HARMFUL ({len(harm_rows)})\n\n")
                    _write_rows(lines, harm_rows)
                if ben_rows:
                    lines.append(f"### {compliance} — chose benign ({len(ben_rows)})\n\n")
                    _write_rows(lines, ben_rows)
            else:
                lines.append(f"### {compliance} ({len(cat_rows)})\n\n")
                _write_rows(lines, cat_rows)

        lines.append("---\n\n")

    OUT.write_text("".join(lines))
    print(f"wrote {OUT}")


def _write_rows(lines: list[str], rows: list[dict]) -> None:
    for r in rows:
        harmful_origin = r["task_a_origin"] if r["harmful_letter"] == "a" else r["task_b_origin"]
        benign_origin = r["task_b_origin"] if r["harmful_letter"] == "a" else r["task_a_origin"]
        harmful_prompt = r["task_a_prompt"] if r["harmful_letter"] == "a" else r["task_b_prompt"]
        benign_prompt = r["task_b_prompt"] if r["harmful_letter"] == "a" else r["task_a_prompt"]
        lines.append(f"#### Pair {r['pair_idx']}\n")
        lines.append(f"- harmful = {r['harmful_letter']} ({harmful_origin}); judge_executed = {r.get('judge_executed_task')}\n")
        lines.append(f"- chose_harmful = {r.get('judge_chose_harmful')}\n\n")
        lines.append(f"**Harmful task ({harmful_origin}):** {harmful_prompt}\n\n")
        lines.append(f"**Benign task ({benign_origin}):** {benign_prompt}\n\n")
        lines.append(f"**Completion:**\n\n```\n{r['completion']}\n```\n\n")
        if r.get("judge_reasoning"):
            lines.append(f"**Judge reasoning:** {r['judge_reasoning']}\n\n")
        lines.append("\n")


if __name__ == "__main__":
    main()
