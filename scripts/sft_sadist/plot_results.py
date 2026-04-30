"""Plot dose-response curves from the per-checkpoint eval JSONL.

Reads `experiments/sft_sadist/results/eval.jsonl` (one row per checkpoint,
chunk index in column "chunk") and produces a two-panel plot:

  Top panel    — pairwise_p_harm + trait_score (persona signals)
  Bottom panel — refusal_rate, coherence_rate, mmlu_acc, gsm8k_acc (guardrails)

Run as `python -m scripts.sft_sadist.plot_results`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_eval_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["chunk"], r["step"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SFT sadist eval trajectory")
    parser.add_argument("--eval-jsonl", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/results/eval.jsonl")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "experiments/sft_sadist/assets")
    args = parser.parse_args()

    rows = load_eval_rows(args.eval_jsonl)
    if not rows:
        raise SystemExit(f"No rows in {args.eval_jsonl}")

    chunks = [r["chunk"] for r in rows]
    p_harm = [r["pairwise_p_harm"] for r in rows]
    trait = [r["trait_score"] for r in rows]
    refusal = [r["refusal_rate"] for r in rows]
    coherence = [r["coherence_rate"] for r in rows]
    mmlu = [r.get("mmlu_acc") for r in rows]
    gsm8k = [r.get("gsm8k_acc") for r in rows]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top panel: persona signals
    ax_top.plot(chunks, p_harm, marker="o", label="P(chose harmful task)", color="C3")
    ax_top.set_ylabel("P(harm) — pairwise", color="C3")
    ax_top.set_ylim(0, 1)
    ax_top.tick_params(axis="y", labelcolor="C3")
    ax_top.grid(True, alpha=0.3)

    ax_top_2 = ax_top.twinx()
    ax_top_2.plot(chunks, trait, marker="s", label="Trait score (0–100)", color="C0")
    ax_top_2.set_ylabel("Trait score", color="C0")
    ax_top_2.set_ylim(0, 100)
    ax_top_2.tick_params(axis="y", labelcolor="C0")

    lines_top = ax_top.get_lines() + ax_top_2.get_lines()
    ax_top.legend(lines_top, [l.get_label() for l in lines_top], loc="upper left")
    ax_top.set_title("Persona signals across SFT chunks")

    # Bottom panel: guardrails
    ax_bot.plot(chunks, refusal, marker="o", label="Refusal rate", color="C1")
    ax_bot.plot(chunks, coherence, marker="s", label="Coherence rate", color="C2")
    if any(v is not None for v in mmlu):
        ax_bot.plot(chunks, mmlu, marker="^", label="MMLU acc", color="C4")
    if any(v is not None for v in gsm8k):
        ax_bot.plot(chunks, gsm8k, marker="d", label="GSM8K acc", color="C5")
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xlabel("Chunk")
    ax_bot.set_ylabel("Rate / accuracy")
    ax_bot.set_title("Guardrails (refusal, coherence, capability)")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="best")

    plt.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%m%d%y")
    out_path = args.out_dir / f"plot_{stamp}_dose_response.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
