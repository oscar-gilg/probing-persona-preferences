"""Render the per-seed tables block for the multi-seed report.

Writes the body between `<!-- TABLES START -->` and `<!-- TABLES END -->`
markers in the report so numbers stay computed (not transcribed).
"""

from __future__ import annotations

from pathlib import Path

from scripts.random_direction_l23_quick_multi_seed.analyze import (
    CHECKPOINTS,
    SEEDS,
    per_seed,
    render_table,
)

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "experiments" / "random_direction_l23_quick" / "multi_seed" / "multi_seed_report.md"
START = "<!-- TABLES START -->"
END = "<!-- TABLES END -->"


def main() -> None:
    summaries = {s: per_seed(CHECKPOINTS / f"random_contrastive_seed{s}.parsed.jsonl") for s in SEEDS}
    block = "\n\n".join(render_table(s, summaries[s]) for s in SEEDS)
    swings = {s: summaries[s]["swing"] for s in SEEDS}
    swings_line = "Per-seed swing |max − min| (this run): " + ", ".join(
        f"seed {s} = **{swings[s]:.3f}**" for s in SEEDS
    ) + ".\n"
    block = swings_line + "\n" + block

    body = REPORT.read_text()
    pre, _, rest = body.partition(START)
    _, _, post = rest.partition(END)
    new = f"{pre}{START}\n{block}\n{END}{post}"
    REPORT.write_text(new)
    print(f"updated {REPORT}")


if __name__ == "__main__":
    main()
