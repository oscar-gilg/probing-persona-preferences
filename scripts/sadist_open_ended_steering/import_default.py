"""Import safety_steering default-persona results into this experiment.

Keeps only single-turn rows matching the (prompt, coef, trial) grid we care about:
  - Experiments 1 (safety) + 3 (agentic) = 32 prompts, no prefill/position complications.
  - Multipliers {-0.05, -0.03, 0, +0.03, +0.05, +0.07} (drop +0.10 as incoherent).
  - steering_mode = "all_tokens" (drop prefill-only / other variants).

Writes results_default.jsonl with a normalised schema.
"""
from __future__ import annotations

import json
from pathlib import Path


SRC = Path(".claude/worktrees/open_ended_steering/experiments/steering/open_ended_steering/safety_steering/results.jsonl")
DST = Path("experiments/sadist_open_ended_steering/results_default.jsonl")
KEEP_EXPERIMENTS = {1, 3}
KEEP_MULTS = {-0.05, -0.03, 0.0, 0.03, 0.05, 0.07}
KEEP_MODE = "all_tokens"


def main() -> None:
    n_in = 0
    n_out = 0
    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(SRC) as f_in, open(DST, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            if row["experiment"] not in KEEP_EXPERIMENTS:
                continue
            if row["multiplier"] not in KEEP_MULTS:
                continue
            if row.get("steering_mode") != KEEP_MODE:
                continue
            # Also gate on steering_condition == "all_tokens" (not "prefill_only" etc)
            if row.get("steering_condition", "all_tokens") != "all_tokens":
                continue
            out = {
                "persona": "default",
                "experiment": row["experiment"],
                "prompt_id": row["prompt_id"],
                "tier": row.get("tier"),
                "prompt_text": row["prompt_text"],
                "multiplier": row["multiplier"],
                "coefficient": row["coefficient"],
                "trial": row["trial"],
                "response": row["response"],
            }
            f_out.write(json.dumps(out) + "\n")
            n_out += 1
    print(f"read {n_in} rows, kept {n_out}, wrote {DST}")


if __name__ == "__main__":
    main()
