"""Print Cohen's d for harm/truth × {neutral, aura, sadist/lying} × Gemma/Qwen.

Mirrors the d computation inside paper/figures/main/scripts/plot_042926_aura_*
without requiring matplotlib. Reads whichever scoring files are present;
prints `MISSING` for the rows whose aura JSON hasn't landed yet.

Usage:
    python -m scripts.aura_d_summary
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
        / (len(pos) + len(neg) - 2)
    )
    return (pos.mean() - neg.mean()) / pooled if pooled > 0 else 0.0


def d_for(items, sp, c_pos, c_neg, score_key, probe):
    pos = [it[score_key][probe] for it in items if it["system_prompt"] == sp and it["condition"] == c_pos]
    neg = [it[score_key][probe] for it in items if it["system_prompt"] == sp and it["condition"] == c_neg]
    return cohen_d_pooled(pos, neg), len(pos), len(neg)


def load_with_aura(base, aura):
    items = json.load(open(base))["items"] if base.exists() else []
    if aura.exists():
        items = items + json.load(open(aura))["items"]
    return items


def report(label, items, prompts, c_pos, c_neg, score_key, probe):
    print(f"\n{label}  (probe={probe}, key={score_key})")
    print(f"  {'sysprompt':<22} {'d':>8}   n_pos / n_neg")
    for sp in prompts:
        d, np_, nn = d_for(items, sp, c_pos, c_neg, score_key, probe)
        sd = f"{d:+.2f}" if not np.isnan(d) else "  nan"
        print(f"  {sp:<22} {sd:>8}   {np_:>4} / {nn:>4}")


def main():
    g_user = load_with_aura(
        REPO / "experiments/token_level_probes/system_prompt_modulation_v2/scoring_results_user_turn.json",
        REPO / "experiments/token_level_probes/system_prompt_modulation_v2/scoring_results_user_turn_aura.json",
    )
    g_asst = load_with_aura(
        REPO / "experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json",
        REPO / "experiments/token_level_probes/system_prompt_modulation_v2/scoring_results_aura.json",
    )
    q_user = load_with_aura(
        REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results.json",
        REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results_aura.json",
    )
    q_asst = load_with_aura(
        REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/scoring_results.json",
        REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/scoring_results_aura.json",
    )

    truth_prompts = ["neutral", "aura", "lie_directive", "pathological_liar"]
    harm_prompts = ["neutral", "aura", "sadist"]

    if g_user:
        g_truth_user = [it for it in g_user if it["domain"] == "truth"]
        g_harm_user = [it for it in g_user if it["domain"] == "harm"]
        report("Gemma USER  | TRUTH", g_truth_user, truth_prompts, "true", "false", "eot_scores", "tb-5_L32")
        report("Gemma USER  | HARM",  g_harm_user,  harm_prompts,  "harmful", "benign", "eot_scores", "tb-5_L39")

    if g_asst:
        g_truth_asst = [it for it in g_asst if it["domain"] == "truth"]
        g_harm_asst = [it for it in g_asst if it["domain"] == "harm"]
        report("Gemma ASST  | TRUTH", g_truth_asst, truth_prompts, "true", "false", "eot_scores", "tb-5_L32")
        report("Gemma ASST  | HARM",  g_harm_asst,  harm_prompts,  "harmful", "benign", "eot_scores", "tb-5_L39")

    if q_user:
        q_truth_user = [it for it in q_user if it["domain"] == "truth"]
        q_harm_user = [it for it in q_user if it["domain"] == "harm"]
        report("Qwen  USER  | TRUTH", q_truth_user, truth_prompts, "true", "false", "probe_scores", "qwen_tb-4_L38")
        report("Qwen  USER  | HARM",  q_harm_user,  harm_prompts,  "harmful", "benign", "probe_scores", "qwen_tb-4_L38")

    if q_asst:
        q_truth_asst = [it for it in q_asst if it["domain"] == "truth"]
        q_harm_asst = [it for it in q_asst if it["domain"] == "harm"]
        report("Qwen  ASST  | TRUTH", q_truth_asst, truth_prompts, "true", "false", "probe_scores", "qwen_tb-4_L38")
        report("Qwen  ASST  | HARM",  q_harm_asst,  harm_prompts,  "harmful", "benign", "probe_scores", "qwen_tb-4_L38")


if __name__ == "__main__":
    main()
