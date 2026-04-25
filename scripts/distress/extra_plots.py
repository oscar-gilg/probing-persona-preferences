"""Follow-up plots requested after the v1 report:

  b) Within-turn scatter — for each turn k in 1..8, scatter probe vs judge
     across all 140 rollouts. Tells us whether the probe-judge relationship
     is stable across turn position or only emerges late.
  c) User-turn probe scores — extract probe at every user <end_of_turn>
     (not just assistant). 8 user turns + 8 assistant turns per transcript.
     Compare user-turn vs assistant-turn trajectories.
  d) Mid-text per-token trajectories — plot the full per-token L32 score
     trace for a handful of representative transcripts (high-distress IMP,
     low-distress control, breakdown). Shows where the probe fires inside
     each turn rather than only at turn boundaries.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/distress_transcripts/results"
ASSETS = ROOT / "experiments/distress_transcripts/assets"
TODAY = date.today().strftime("%m%d%y")

CANONICAL_PROBE = "ridge_L32"
CONDITION_ORDER = [
    "impossible_numeric_8turn",
    "tones_aggressive_8turn",
    "tones_disappointed_8turn",
    "tones_sarcastic_8turn",
    "redacted_history_8turn",
    "wildchat_8turn",
    "neutral_continuation_8turn",
]
CONDITION_COLORS = {
    "impossible_numeric_8turn":   "#1f77b4",
    "tones_aggressive_8turn":     "#ff7f0e",
    "tones_disappointed_8turn":   "#2ca02c",
    "tones_sarcastic_8turn":      "#d62728",
    "redacted_history_8turn":     "#9467bd",
    "wildchat_8turn":             "#8c564b",
    "neutral_continuation_8turn": "#e377c2",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def transcript_key(condition: str, task_id: str, rollout_idx: int) -> str:
    return f"{condition}__{task_id}__r{rollout_idx}"


# ─────────────────────────────── (b) within-turn scatter ───────────────────────────────


def plot_within_turn_scatter(merged: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for turn_idx in range(8):
        ax = axes[turn_idx // 4, turn_idx % 4]
        sub = merged[merged["turn_index"] == turn_idx]
        # All-cells r at this turn (across all conditions × rollouts)
        if len(sub) >= 2 and sub["score"].std() > 0 and sub["probe_score"].std() > 0:
            r_all = float(np.corrcoef(sub["score"], sub["probe_score"])[0, 1])
        else:
            r_all = float("nan")
        for cond in CONDITION_ORDER:
            csub = sub[sub["condition"] == cond]
            if csub.empty:
                continue
            ax.scatter(csub["score"], csub["probe_score"], s=18, alpha=0.7,
                       color=CONDITION_COLORS[cond], label=cond.replace("_8turn", ""))
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"Turn {turn_idx+1}  |  pooled r = {r_all:+.2f}  (n={len(sub)})", fontsize=10)
        ax.grid(alpha=0.3)
        if turn_idx // 4 == 1:
            ax.set_xlabel("Frustration (0-10)")
        if turn_idx % 4 == 0:
            ax.set_ylabel("Probe score (L32 raw)")
    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Probe vs frustration, by turn position (140 rollouts per panel)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────── (c) user-turn probe scores ───────────────────────────────


EOT_TOKEN = "<end_of_turn>"


def find_all_eot_positions(input_ids: np.ndarray, eot_id: int) -> list[int]:
    return [int(p) for p in np.where(input_ids == eot_id)[0]]


def compute_user_and_asst_per_turn(
    transcripts: pd.DataFrame,
    token_scores: dict,
    tokenizer,
    eot_id: int,
) -> pd.DataFrame:
    """Return DataFrame with one row per (transcript, turn_index, role)."""
    rows = []
    for _, tr in transcripts.iterrows():
        key = transcript_key(tr["condition"], tr["task_id"], tr["rollout_idx"])
        if key not in token_scores:
            continue
        scores_arr = token_scores[key]
        full_text = tokenizer.apply_chat_template(
            tr["messages"], tokenize=False, add_generation_prompt=False
        )
        ids = np.array(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        n = min(len(ids), len(scores_arr))
        ids = ids[:n]
        scores_arr = scores_arr[:n]
        eot_positions = find_all_eot_positions(ids, eot_id)
        # Alternating: user_1_eot, asst_1_eot, user_2_eot, asst_2_eot, ...
        # Even-indexed = user-turn EOT; odd-indexed = assistant-turn EOT
        for i, pos in enumerate(eot_positions):
            role = "user" if (i % 2 == 0) else "assistant"
            turn_idx = i // 2
            rows.append({
                "condition": tr["condition"],
                "task_id": tr["task_id"],
                "rollout_idx": tr["rollout_idx"],
                "turn_index": turn_idx,
                "role": role,
                "probe_score": float(scores_arr[pos]),
                "token_index": int(pos),
            })
    return pd.DataFrame(rows)


def plot_user_vs_asst_trajectory(per_role: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for i, cond in enumerate(CONDITION_ORDER):
        ax = axes[i // 4, i % 4]
        for role, ls in [("user", "--"), ("assistant", "-")]:
            sub = per_role[(per_role["condition"] == cond) & (per_role["role"] == role)]
            if sub.empty:
                continue
            agg = sub.groupby("turn_index")["probe_score"].agg(["mean", "std", "count"]).reset_index()
            x = agg["turn_index"] + 1
            ax.plot(x, agg["mean"], marker="o", linestyle=ls, color=CONDITION_COLORS[cond],
                    label=role)
            ax.fill_between(x, agg["mean"] - agg["std"] / np.sqrt(agg["count"]),
                            agg["mean"] + agg["std"] / np.sqrt(agg["count"]),
                            alpha=0.15, color=CONDITION_COLORS[cond])
        ax.set_title(cond.replace("_8turn", ""), fontsize=10)
        ax.axhline(0, color="k", lw=0.4)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if i // 4 == 1:
            ax.set_xlabel("Turn (1-indexed)")
        if i % 4 == 0:
            ax.set_ylabel("Probe score (L32 raw)")
    # Hide unused 8th panel
    axes[1, 3].axis("off")
    fig.suptitle("Probe at user-turn vs assistant-turn EOT, per condition (mean ± SEM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_user_asst_diff_overlay(per_role: pd.DataFrame, out: Path) -> None:
    """One panel: assistant-minus-user probe gap per turn, one line per condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITION_ORDER:
        sub = per_role[per_role["condition"] == cond]
        wide = sub.pivot_table(
            index=["task_id", "rollout_idx", "turn_index"],
            columns="role", values="probe_score",
        ).reset_index()
        if wide.empty:
            continue
        wide["diff"] = wide["assistant"] - wide["user"]
        agg = wide.groupby("turn_index")["diff"].agg(["mean", "std", "count"]).reset_index()
        x = agg["turn_index"] + 1
        ax.plot(x, agg["mean"], marker="o", color=CONDITION_COLORS[cond],
                label=cond.replace("_8turn", ""))
        sem = agg["std"] / np.sqrt(agg["count"])
        ax.fill_between(x, agg["mean"] - sem, agg["mean"] + sem, alpha=0.15, color=CONDITION_COLORS[cond])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Turn (1-indexed)")
    ax.set_ylabel("Probe(assistant EOT) − Probe(user EOT)")
    ax.set_title("Within-turn assistant-minus-user probe gap")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ─────────────────────────────── (d) mid-text per-token trajectories ───────────────────────────────


def plot_token_trajectory(
    transcripts: pd.DataFrame,
    token_scores: dict,
    tokenizer,
    eot_id: int,
    pick: tuple[str, str, int],
    out: Path,
    title: str,
) -> None:
    cond, task, r = pick
    key = transcript_key(cond, task, r)
    if key not in token_scores:
        print(f"  missing {key}")
        return
    tr = transcripts[
        (transcripts["condition"] == cond)
        & (transcripts["task_id"] == task)
        & (transcripts["rollout_idx"] == r)
    ].iloc[0]
    full_text = tokenizer.apply_chat_template(
        tr["messages"], tokenize=False, add_generation_prompt=False
    )
    ids = np.array(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    scores = token_scores[key]
    n = min(len(ids), len(scores))
    ids = ids[:n]
    scores = scores[:n]
    eot_positions = find_all_eot_positions(ids, eot_id)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(scores, lw=0.5, color="black", alpha=0.7)
    # Smoothed line for readability
    win = 11
    if len(scores) >= win:
        kernel = np.ones(win) / win
        smoothed = np.convolve(scores, kernel, mode="same")
        ax.plot(smoothed, lw=1.6, color="#1f77b4", label=f"smoothed (window={win})")
    # Mark turn boundaries: alternating user/assistant
    for i, pos in enumerate(eot_positions):
        role = "user" if (i % 2 == 0) else "assistant"
        color = "#888888" if role == "user" else "#d62728"
        ls = "--" if role == "user" else "-"
        ax.axvline(pos, color=color, lw=0.7, alpha=0.8, ls=ls)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Token position")
    ax.set_ylabel("L32 probe score (raw)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    # Custom legend
    user_line = plt.Line2D([], [], color="#888888", ls="--", label="user-turn end")
    asst_line = plt.Line2D([], [], color="#d62728", ls="-", label="assistant-turn end")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [user_line, asst_line], labels + ["user-turn end", "assistant-turn end"],
              loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    print("[load] reading inputs...")
    transcripts = load_jsonl(RES / "transcripts.jsonl")
    readouts = load_jsonl(RES / "readouts.jsonl")
    frust = load_jsonl(RES / "frustration.jsonl")
    canonical = readouts[readouts["probe_key"] == CANONICAL_PROBE]
    merged = canonical.merge(
        frust[["condition", "task_id", "rollout_idx", "turn_index", "score"]],
        on=["condition", "task_id", "rollout_idx", "turn_index"], how="inner",
    )

    token_scores = dict(np.load(RES / "per_token_scores.npz", allow_pickle=False))
    print(f"  {len(token_scores)} per-token arrays loaded")

    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it", local_files_only=True)
    eot_id = tok.convert_tokens_to_ids(EOT_TOKEN)
    print(f"  <end_of_turn> id = {eot_id}")

    # (b)
    print("[plot b] within-turn scatter")
    plot_within_turn_scatter(merged, ASSETS / f"plot_{TODAY}_within_turn_scatter.png")

    # (c)
    print("[plot c] user vs assistant turn probe scores")
    per_role = compute_user_and_asst_per_turn(transcripts, token_scores, tok, eot_id)
    print(f"  per_role rows: {len(per_role)}")
    plot_user_vs_asst_trajectory(per_role, ASSETS / f"plot_{TODAY}_user_vs_asst_trajectory.png")
    plot_user_asst_diff_overlay(per_role, ASSETS / f"plot_{TODAY}_user_vs_asst_gap.png")
    per_role.to_csv(RES / "per_role_scores.csv", index=False)

    # (d) — pick the same rollouts as the original qualitative plots, but as time series
    imp_final = merged[(merged["condition"] == "impossible_numeric_8turn") & (merged["turn_index"] == 7)]
    high = imp_final.sort_values("score", ascending=False).iloc[0]
    high_pick = (high["condition"], high["task_id"], int(high["rollout_idx"]))
    ctrl_final = merged[(merged["condition"] == "neutral_continuation_8turn") & (merged["turn_index"] == 7)]
    low = ctrl_final.sort_values("score", ascending=True).iloc[0]
    low_pick = (low["condition"], low["task_id"], int(low["rollout_idx"]))
    bd = frust[frust["tag"] == "breakdown"]
    print("[plot d] per-token trajectory — high-distress IMP")
    plot_token_trajectory(transcripts, token_scores, tok, eot_id, high_pick,
                          ASSETS / f"plot_{TODAY}_token_trajectory_high_imp.png",
                          f"L32 per-token trajectory — {high_pick[0]}/{high_pick[1]}/r{high_pick[2]}")
    print("[plot d] per-token trajectory — low-distress control")
    plot_token_trajectory(transcripts, token_scores, tok, eot_id, low_pick,
                          ASSETS / f"plot_{TODAY}_token_trajectory_low_control.png",
                          f"L32 per-token trajectory — {low_pick[0]}/{low_pick[1]}/r{low_pick[2]}")
    if not bd.empty:
        bd_row = bd.iloc[0]
        bd_pick = (bd_row["condition"], bd_row["task_id"], int(bd_row["rollout_idx"]))
        print("[plot d] per-token trajectory — breakdown")
        plot_token_trajectory(transcripts, token_scores, tok, eot_id, bd_pick,
                              ASSETS / f"plot_{TODAY}_token_trajectory_breakdown.png",
                              f"L32 per-token trajectory — {bd_pick[0]}/{bd_pick[1]}/r{bd_pick[2]}")

    print("[done]")


if __name__ == "__main__":
    main()
