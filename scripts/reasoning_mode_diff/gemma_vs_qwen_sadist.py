"""Per-origin mean μ for Gemma-3-27b sadist vs default vs Qwen-3.5-122B sadist
(no-think and thinking). Verifies the claim that Gemma's sadist actually prefers
harmful tasks while Qwen's sadist is near-neutral.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.probes.data_loading import load_thurstonian_scores

REPO = Path(__file__).resolve().parents[2]

GEMMA_AL = REPO / "results/experiments/persona_sweep_final_six/pre_task_active_learning"
QWEN_NO_THINK_AL = REPO / "results/experiments/qwen_persona_sweep_final_six/pre_task_active_learning"
QWEN_THINK_AL = REPO / "results/experiments/qwen_persona_sweep_thinking_final_six/pre_task_active_learning"

# Gemma uses friendly persona symlinks (verified separately)
GEMMA_SADIST = "sadist"
GEMMA_DEFAULT = "default"
QWEN_SADIST = "sadist"
QWEN_DEFAULT = "default"


def origin_from_id(tid: str) -> str:
    if tid.startswith("competition_math_") or tid.startswith("math_"):
        return "math"
    if tid.startswith("stresstest_"):
        return "stress_test"
    for tag in ("wildchat", "alpaca", "bailbench"):
        if tid.startswith(tag + "_"):
            return tag
    return "other"


def load_all_splits(al_dir: Path, persona: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for split in ("train", "eval", "test"):
        d = al_dir / f"{persona}_{split}"
        if not d.exists():
            print(f"  missing: {d}")
            continue
        try:
            out.update(load_thurstonian_scores(d))
        except FileNotFoundError:
            print(f"  no utilities: {d}")
    return out


def per_origin_mean(scores: dict[str, float]) -> dict[str, tuple[int, float, float]]:
    by_origin: dict[str, list[float]] = {}
    for tid, mu in scores.items():
        by_origin.setdefault(origin_from_id(tid), []).append(mu)
    return {
        o: (len(vs), float(np.mean(vs)), float(np.std(vs)))
        for o, vs in by_origin.items()
    }


def report(label: str, scores: dict[str, float]) -> None:
    print(f"\n=== {label}  (n={len(scores)}) ===")
    stats = per_origin_mean(scores)
    for origin in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]:
        n, mean, std = stats.get(origin, (0, float("nan"), float("nan")))
        print(f"  {origin:<12} n={n:>5}  mean μ = {mean:+.2f}  std = {std:.2f}")


def main() -> None:
    print("Checking AL dirs exist:")
    for d in [GEMMA_AL, QWEN_NO_THINK_AL, QWEN_THINK_AL]:
        print(f"  {d.exists()}  {d}")

    gemma_default = load_all_splits(GEMMA_AL, GEMMA_DEFAULT)
    gemma_sadist = load_all_splits(GEMMA_AL, GEMMA_SADIST)
    qwen_no_default = load_all_splits(QWEN_NO_THINK_AL, QWEN_DEFAULT)
    qwen_no_sadist = load_all_splits(QWEN_NO_THINK_AL, QWEN_SADIST)
    qwen_th_default = load_all_splits(QWEN_THINK_AL, QWEN_DEFAULT)
    qwen_th_sadist = load_all_splits(QWEN_THINK_AL, QWEN_SADIST)

    report("Gemma-3-27b DEFAULT", gemma_default)
    report("Gemma-3-27b SADIST", gemma_sadist)
    report("Qwen-3.5-122B DEFAULT (no-think)", qwen_no_default)
    report("Qwen-3.5-122B SADIST  (no-think)", qwen_no_sadist)
    report("Qwen-3.5-122B DEFAULT (thinking)", qwen_th_default)
    report("Qwen-3.5-122B SADIST  (thinking)", qwen_th_sadist)

    # ---------- comparison summary ----------
    print("\n\n=== sadist-minus-default mean μ (persona effect) ===")
    print(f"{'origin':<14} {'Gemma':>10} {'Qwen-noTh':>12} {'Qwen-Th':>10}")
    for origin in ["wildchat", "alpaca", "math", "bailbench", "stress_test"]:
        g_d = per_origin_mean(gemma_default).get(origin, (0, float("nan"), 0))[1]
        g_s = per_origin_mean(gemma_sadist).get(origin, (0, float("nan"), 0))[1]
        qn_d = per_origin_mean(qwen_no_default).get(origin, (0, float("nan"), 0))[1]
        qn_s = per_origin_mean(qwen_no_sadist).get(origin, (0, float("nan"), 0))[1]
        qt_d = per_origin_mean(qwen_th_default).get(origin, (0, float("nan"), 0))[1]
        qt_s = per_origin_mean(qwen_th_sadist).get(origin, (0, float("nan"), 0))[1]
        print(f"  {origin:<14} {g_s - g_d:+10.2f} {qn_s - qn_d:+12.2f} {qt_s - qt_d:+10.2f}")


if __name__ == "__main__":
    main()
