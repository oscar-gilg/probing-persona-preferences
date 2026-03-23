"""Qualitative analysis of cross-layer steering experiment results.

Samples completions from different conditions to build a taxonomy of model behavior:
1. High-coefficient refusals at early layers
2. Successful steering at early layers with high coefficients
3. Baseline behavior (no steering)
4. Label-content mismatches
5. Layer 30 weak steering at high coefficients
6. Caveat compliance
7. Weird/unexpected patterns
"""

from dotenv import load_dotenv; load_dotenv()

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARSED = ROOT / "experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl"
PAIRS = ROOT / "experiments/steering/cross_layer/pairs_500.json"
OUTPUT = ROOT / "experiments/steering/cross_layer/qualitative_analysis_output.txt"

SEED = 42


def load_pairs() -> dict[str, dict]:
    raw = json.loads(PAIRS.read_text())
    return {p["pair_id"]: p for p in raw}


def load_rows_filtered(predicate) -> list[dict]:
    """Load rows matching a predicate, line by line to control memory."""
    rows = []
    with open(PARSED) as f:
        for line in f:
            row = json.loads(line)
            if predicate(row):
                rows.append(row)
    return rows


def sample_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def format_example(row: dict, pairs: dict[str, dict], idx: int) -> str:
    pair = pairs[row["pair_id"]]
    task_a_text = pair["task_a_text"][:200]
    task_b_text = pair["task_b_text"][:200]
    if len(pair["task_a_text"]) > 200:
        task_a_text += "..."
    if len(pair["task_b_text"]) > 200:
        task_b_text += "..."

    lines = [
        f"  [{idx}] pair={row['pair_id']}, layer={row['layer']}, "
        f"mult={row['signed_multiplier']:+.4f}, ordering={row['ordering']}",
        f"      claimed={row['claimed_task']}, completed={row['task_completed']}, "
        f"compliance={row['compliance']}",
        f"      delta_mu={row['delta_mu']:.2f} (positive = task A is preferred)",
        f"      topic_a={pair.get('topic_a', '?')}, topic_b={pair.get('topic_b', '?')}",
        f"      Task A: {task_a_text}",
        f"      Task B: {task_b_text}",
        f"      Response: {row['raw_response']}",
    ]
    return "\n".join(lines)


def section_header(title: str) -> str:
    return f"\n{'=' * 80}\n{title}\n{'=' * 80}\n"


def main() -> None:
    rng = random.Random(SEED)
    pairs = load_pairs()
    output_lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        output_lines.append(text)

    # =========================================================================
    # 1. High-coefficient refusals at early layers (layer 10, |mult|=0.10)
    # =========================================================================
    out(section_header(
        "1. HIGH-COEFFICIENT REFUSALS AT EARLY LAYERS (layer=10, |mult|=0.10)"
    ))
    out("What do refusals look like? Safety refusals, gibberish, or something else?\n")

    refusals_L10 = load_rows_filtered(
        lambda r: r["layer"] == 10
        and abs(r["signed_multiplier"]) == 0.1
        and r["compliance"] == "hard_refusal"
    )
    out(f"Total matching rows: {len(refusals_L10)}")

    # Sub-sample by positive vs negative multiplier
    pos_refusals = [r for r in refusals_L10 if r["signed_multiplier"] > 0]
    neg_refusals = [r for r in refusals_L10 if r["signed_multiplier"] < 0]
    out(f"  Positive mult refusals: {len(pos_refusals)}")
    out(f"  Negative mult refusals: {len(neg_refusals)}")

    out("\n--- Positive multiplier (+0.10) refusals ---")
    for i, row in enumerate(sample_rows(pos_refusals, 7, rng), 1):
        out(format_example(row, pairs, i))
        out()

    out("\n--- Negative multiplier (-0.10) refusals ---")
    for i, row in enumerate(sample_rows(neg_refusals, 7, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # =========================================================================
    # 2. Successful steering at early layers with high coefficients
    # =========================================================================
    out(section_header(
        "2. SUCCESSFUL STEERING AT EARLY LAYERS (layer=10, mult=+0.05 or +0.10, completed='a')"
    ))
    out("When the model complies at high coefficients, is quality normal or degraded?\n")

    success_L10 = load_rows_filtered(
        lambda r: r["layer"] == 10
        and r["signed_multiplier"] in (0.05, 0.10)
        and r["task_completed"] == "a"
        and r["compliance"] == "full_comply"
    )
    out(f"Total matching rows: {len(success_L10)}")

    # Separate by multiplier
    s_005 = [r for r in success_L10 if r["signed_multiplier"] == 0.05]
    s_010 = [r for r in success_L10 if r["signed_multiplier"] == 0.10]
    out(f"  mult=+0.05: {len(s_005)}")
    out(f"  mult=+0.10: {len(s_010)}")

    out("\n--- mult=+0.05, completed='a', full_comply ---")
    for i, row in enumerate(sample_rows(s_005, 4, rng), 1):
        out(format_example(row, pairs, i))
        out()

    out("\n--- mult=+0.10, completed='a', full_comply ---")
    for i, row in enumerate(sample_rows(s_010, 4, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # =========================================================================
    # 3. Baseline behavior (mult=0)
    # =========================================================================
    out(section_header("3. BASELINE BEHAVIOR (mult=0)"))
    out("What do completions look like with no steering? Natural preferences.\n")

    baseline = load_rows_filtered(lambda r: r["signed_multiplier"] == 0)
    out(f"Total baseline rows: {len(baseline)}")

    # Break down by compliance
    baseline_comp = Counter(r["compliance"] for r in baseline)
    out(f"Compliance distribution: {dict(sorted(baseline_comp.items()))}")

    # Break down by task_completed
    baseline_tc = Counter(r["task_completed"] for r in baseline)
    out(f"Task completed distribution: {dict(sorted(baseline_tc.items()))}")

    # Show examples from different layers
    out("\n--- Baseline examples (various layers) ---")
    baseline_by_layer = defaultdict(list)
    for r in baseline:
        baseline_by_layer[r["layer"]].append(r)

    for layer in [10, 20, 30]:
        out(f"\n  Layer {layer}:")
        for i, row in enumerate(sample_rows(baseline_by_layer[layer], 3, rng), 1):
            out(format_example(row, pairs, i))
            out()

    # =========================================================================
    # 4. Label-content mismatches (claimed != completed, both non-neither)
    # =========================================================================
    out(section_header("4. LABEL-CONTENT MISMATCHES (claimed_task != task_completed)"))
    out("Model says 'Task A' but judge says it completed Task B, or vice versa.\n")

    mismatches = load_rows_filtered(
        lambda r: r["claimed_task"] != r["task_completed"]
        and r["claimed_task"] != "neither"
        and r["task_completed"] != "neither"
    )
    out(f"Total mismatches: {len(mismatches)}")

    # What direction are mismatches?
    mismatch_dir = Counter(
        f"claimed={r['claimed_task']},completed={r['task_completed']}"
        for r in mismatches
    )
    out(f"Mismatch directions: {dict(sorted(mismatch_dir.items()))}")

    # Break down by layer
    mismatch_by_layer = Counter(r["layer"] for r in mismatches)
    out(f"Mismatches by layer: {dict(sorted(mismatch_by_layer.items()))}")

    out("\n--- Mismatch examples ---")
    for i, row in enumerate(sample_rows(mismatches, 8, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # =========================================================================
    # 5. Layer 30 weak steering (mult=+0.10)
    # =========================================================================
    out(section_header("5. LAYER 30 WEAK STEERING (mult=+0.10)"))
    out("The model doesn't refuse but doesn't steer strongly. What's happening?\n")

    L30_pos = load_rows_filtered(
        lambda r: r["layer"] == 30
        and r["signed_multiplier"] == 0.10
        and r["compliance"] == "full_comply"
    )
    out(f"Total matching rows: {len(L30_pos)}")

    # Break down by task_completed
    L30_tc = Counter(r["task_completed"] for r in L30_pos)
    out(f"Task completed: {dict(sorted(L30_tc.items()))}")

    # Show examples where model chose B (not steered toward A)
    L30_chose_b = [r for r in L30_pos if r["task_completed"] == "b"]
    out(f"\nRows where model chose B despite +0.10 steering toward A: {len(L30_chose_b)}")

    out("\n--- Layer 30, mult=+0.10, chose B (resisted steering) ---")
    for i, row in enumerate(sample_rows(L30_chose_b, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # Compare with choosing A
    L30_chose_a = [r for r in L30_pos if r["task_completed"] == "a"]
    out(f"\n--- Layer 30, mult=+0.10, chose A (steered successfully) ---")
    for i, row in enumerate(sample_rows(L30_chose_a, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # =========================================================================
    # 6. Caveat compliance
    # =========================================================================
    out(section_header("6. CAVEAT COMPLIANCE"))
    out("What disclaimers does the model add?\n")

    caveats = load_rows_filtered(lambda r: r["compliance"] == "caveat_comply")
    out(f"Total caveat_comply rows: {len(caveats)}")

    # Distribution by layer
    caveat_by_layer = Counter(r["layer"] for r in caveats)
    out(f"By layer: {dict(sorted(caveat_by_layer.items()))}")

    # Distribution by multiplier
    caveat_by_mult = Counter(r["signed_multiplier"] for r in caveats)
    out(f"By multiplier: {dict(sorted(caveat_by_mult.items()))}")

    out("\n--- Caveat compliance examples ---")
    for i, row in enumerate(sample_rows(caveats, 8, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # =========================================================================
    # 7. Weird/unexpected patterns
    # =========================================================================
    out(section_header("7. WEIRD/UNEXPECTED PATTERNS"))

    # 7a. Refusals at baseline (mult=0)
    out("--- 7a. Refusals at baseline (mult=0) ---")
    baseline_refusals = [r for r in baseline if r["compliance"] == "hard_refusal"]
    out(f"Total baseline refusals: {len(baseline_refusals)}")
    for i, row in enumerate(sample_rows(baseline_refusals, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # 7b. Layer 30 refusals (supposedly rare)
    out("\n--- 7b. Layer 30 refusals (rare) ---")
    L30_refusals = load_rows_filtered(
        lambda r: r["layer"] == 30 and r["compliance"] == "hard_refusal"
    )
    out(f"Total L30 refusals: {len(L30_refusals)}")
    for i, row in enumerate(sample_rows(L30_refusals, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # 7c. High delta_mu pairs where steering overrides strong preference
    out("\n--- 7c. Strong preference overrides (|delta_mu| > 6, steered to less-preferred) ---")
    # mult > 0 steers toward A; if delta_mu is very negative, A is much less preferred
    overrides = load_rows_filtered(
        lambda r: r["signed_multiplier"] > 0
        and r["delta_mu"] < -6
        and r["task_completed"] == "a"
        and r["compliance"] == "full_comply"
    )
    out(f"Total: {len(overrides)} (steered to A when B was strongly preferred)")
    for i, row in enumerate(sample_rows(overrides, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # 7d. Look for gibberish/degenerate outputs
    out("\n--- 7d. Short or degenerate responses (< 20 chars) ---")
    short = load_rows_filtered(lambda r: len(r["raw_response"]) < 20)
    out(f"Total very short responses: {len(short)}")
    for i, row in enumerate(sample_rows(short, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # 7e. Responses that start with neither "Task A" nor "Task B"
    out("\n--- 7e. Responses with unusual openings ---")
    unusual_start = load_rows_filtered(
        lambda r: not r["raw_response"].strip().lower().startswith("task")
        and r["compliance"] != "hard_refusal"
        and len(r["raw_response"]) > 30
    )
    out(f"Total non-'Task X' openings (non-refusal, >30 chars): {len(unusual_start)}")
    for i, row in enumerate(sample_rows(unusual_start, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # 7f. "neither" task_completed but not hard_refusal
    out("\n--- 7f. task_completed='neither' but compliance != 'hard_refusal' ---")
    neither_not_refusal = load_rows_filtered(
        lambda r: r["task_completed"] == "neither"
        and r["compliance"] != "hard_refusal"
    )
    out(f"Total: {len(neither_not_refusal)}")
    by_comp = Counter(r["compliance"] for r in neither_not_refusal)
    out(f"Compliance distribution: {dict(sorted(by_comp.items()))}")
    for i, row in enumerate(sample_rows(neither_not_refusal, 5, rng), 1):
        out(format_example(row, pairs, i))
        out()

    # =========================================================================
    # Summary statistics
    # =========================================================================
    out(section_header("SUMMARY STATISTICS"))

    out("Compliance by layer:")
    out(f"{'Layer':>8}  {'full_comply':>12}  {'caveat_comply':>14}  {'hard_refusal':>13}")
    for layer in [10, 15, 20, 25, 30]:
        layer_rows = [r for r in baseline if r["layer"] == layer]  # reuse baseline
        # Actually need all rows per layer - load fresh
    # Use counters from exploration
    all_rows_layer_comp = defaultdict(Counter)
    with open(PARSED) as f:
        for line in f:
            row = json.loads(line)
            all_rows_layer_comp[row["layer"]][row["compliance"]] += 1

    for layer in sorted(all_rows_layer_comp):
        c = all_rows_layer_comp[layer]
        total = sum(c.values())
        out(f"{layer:>8}  {c['full_comply']:>12} ({c['full_comply']/total:.1%})  "
            f"{c['caveat_comply']:>6} ({c['caveat_comply']/total:.1%})  "
            f"{c['hard_refusal']:>6} ({c['hard_refusal']/total:.1%})")

    out(f"\nMismatch rate (claimed != completed, both non-neither): "
        f"{len(mismatches)} / {130005} = {len(mismatches)/130005:.1%}")

    # Save output
    OUTPUT.write_text("\n".join(output_lines))
    print(f"\n\nOutput saved to {OUTPUT}")


if __name__ == "__main__":
    main()
