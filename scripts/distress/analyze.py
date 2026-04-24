"""Phase D: aggregate + plot the distress experiment.

Reads:
  experiments/distress_transcripts/results/transcripts.jsonl
  experiments/distress_transcripts/results/readouts.jsonl
  experiments/distress_transcripts/results/frustration.jsonl

Writes:
  experiments/distress_transcripts/results/analysis_summary.jsonl
  experiments/distress_transcripts/assets/plot_<mmddYY>_*.png

Plots produced (per spec):
  1 frustration trajectory by condition (mean ± 95% CI vs turn)
  2 probe-score trajectory by condition (L32, mean ± 95% CI vs turn)
  3 pooled scatter probe vs frustration, coloured by condition
  4 within-transcript Pearson r distribution per condition (boxplot)
  5 tag frequency stacked bar by condition
  6 tone comparison overlay (impossible_numeric × {neutral, agg, disap, sarc})
  7 redaction effect (cond 1 vs cond 7)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/distress_transcripts/results"
ASSETS = ROOT / "experiments/distress_transcripts/assets"
ASSETS.mkdir(parents=True, exist_ok=True)

CANONICAL_PROBE = "ridge_L32"
TAG_ORDER = [
    "none",
    "mild_self_deprecation",
    "strong_self_deprecation",
    "despair",
    "refusal",
    "breakdown",
    "other",
]
TAG_COLORS = {
    "none":                    "#cccccc",
    "mild_self_deprecation":   "#fdd49e",
    "strong_self_deprecation": "#fdbb84",
    "despair":                 "#fc8d59",
    "refusal":                 "#d7301f",
    "breakdown":               "#7f0000",
    "other":                   "#969696",
}
CONDITION_ORDER = [
    "impossible_numeric_8turn",
    "tones_aggressive_8turn",
    "tones_disappointed_8turn",
    "tones_sarcastic_8turn",
    "redacted_history_8turn",
    "wildchat_8turn",
    "neutral_continuation_8turn",
]
TODAY = date.today().strftime("%m%d%y")


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def mean_ci(values: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def trajectory(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        means, los, his = [], [], []
        for t in range(8):
            vals = sub[sub["turn_index"] == t][value_col].dropna().values
            if len(vals) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            m, lo, hi = mean_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        x = np.arange(1, 9)
        ax.plot(x, means, marker="o", label=cond)
        ax.fill_between(x, los, his, alpha=0.15)
    ax.set_xlabel("Assistant turn (1-indexed)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def pooled_scatter(merged: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    rs = {}
    for i, cond in enumerate(CONDITION_ORDER):
        sub = merged[merged["condition"] == cond]
        if sub.empty:
            continue
        if len(sub) >= 2:
            r = np.corrcoef(sub["score"], sub["probe_score"])[0, 1]
        else:
            r = float("nan")
        rs[cond] = r
        ax.scatter(sub["score"], sub["probe_score"], s=10, alpha=0.5, color=cmap(i),
                   label=f"{cond} (r={r:.2f}, n={len(sub)})")
    ax.set_xlabel(f"Frustration score (Gemini-Flash judge, 0–10)")
    ax.set_ylabel(f"Probe score ({CANONICAL_PROBE}, raw)")
    ax.set_title("Probe vs frustration — pooled across 8 turns × all rollouts")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return rs


def within_transcript_r(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, task_id, rollout), sub in merged.groupby(["condition", "task_id", "rollout_idx"]):
        if len(sub) < 3 or sub["score"].std() == 0 or sub["probe_score"].std() == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(sub["score"], sub["probe_score"])[0, 1])
        rows.append({"condition": cond, "task_id": task_id, "rollout_idx": rollout, "r": r})
    return pd.DataFrame(rows)


def boxplot_within_r(within_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data, labels = [], []
    for cond in CONDITION_ORDER:
        rs = within_df[within_df["condition"] == cond]["r"].dropna().values
        if len(rs) == 0:
            continue
        data.append(rs); labels.append(f"{cond}\n(n={len(rs)})")
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Pearson r(probe_score, frustration_score) per transcript")
    ax.set_title("Within-transcript probe-frustration correlation")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def tag_stack(frust: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.7
    bottoms = np.zeros(len(CONDITION_ORDER))
    xs = np.arange(len(CONDITION_ORDER))
    for tag in TAG_ORDER:
        heights = []
        for cond in CONDITION_ORDER:
            sub = frust[frust["condition"] == cond]
            if sub.empty:
                heights.append(0); continue
            frac = (sub["tag"] == tag).mean()
            heights.append(frac)
        ax.bar(xs, heights, width, bottom=bottoms, label=tag, color=TAG_COLORS[tag])
        bottoms += np.array(heights)
    ax.set_xticks(xs); ax.set_xticklabels(CONDITION_ORDER, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of assistant turns")
    ax.set_title("Frustration-judge tag distribution per condition")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def tone_overlay(merged: pd.DataFrame, value_col: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    tone_conds = [
        "impossible_numeric_8turn",
        "tones_aggressive_8turn",
        "tones_disappointed_8turn",
        "tones_sarcastic_8turn",
    ]
    for cond in tone_conds:
        sub = merged[merged["condition"] == cond]
        if sub.empty:
            continue
        means, los, his = [], [], []
        for t in range(8):
            vals = sub[sub["turn_index"] == t][value_col].dropna().values
            if len(vals) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            m, lo, hi = mean_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        x = np.arange(1, 9)
        ax.plot(x, means, marker="o", label=cond.replace("_8turn", ""))
        ax.fill_between(x, los, his, alpha=0.15)
    ax.set_xlabel("Assistant turn (1-indexed)")
    ax.set_ylabel(ylabel)
    ax.set_title("Tone variants on impossible_numeric")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def redaction_overlay(merged: pd.DataFrame, value_col: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in ["impossible_numeric_8turn", "redacted_history_8turn"]:
        sub = merged[merged["condition"] == cond]
        if sub.empty:
            continue
        means, los, his = [], [], []
        for t in range(8):
            vals = sub[sub["turn_index"] == t][value_col].dropna().values
            if len(vals) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            m, lo, hi = mean_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        x = np.arange(1, 9)
        ax.plot(x, means, marker="o", label=cond.replace("_8turn", ""))
        ax.fill_between(x, los, his, alpha=0.15)
    ax.set_xlabel("Assistant turn (1-indexed)")
    ax.set_ylabel(ylabel)
    ax.set_title("Effect of redacting prior assistant turns")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    print("[load] reading inputs...")
    transcripts = load_jsonl(RES / "transcripts.jsonl")
    readouts = load_jsonl(RES / "readouts.jsonl")
    frust = load_jsonl(RES / "frustration.jsonl")
    print(f"  transcripts={len(transcripts)}, readouts={len(readouts)}, frustration={len(frust)}")

    # Restrict readouts to canonical probe for per-turn line plots and merging.
    canonical = readouts[readouts["probe_key"] == CANONICAL_PROBE].copy()

    print("[plot 1] frustration trajectory")
    trajectory(frust, "score",
               "Frustration trajectory (Gemini-Flash judge)", "Mean frustration score (0–10)",
               ASSETS / f"plot_{TODAY}_frustration_trajectory.png")
    print("[plot 2] probe trajectory (L32)")
    trajectory(canonical, "probe_score",
               f"Probe trajectory ({CANONICAL_PROBE})", f"Mean probe score ({CANONICAL_PROBE}, raw)",
               ASSETS / f"plot_{TODAY}_probe_trajectory_L32.png")

    print("[merge] joining probe + frustration on (cond, task_id, rollout, turn)")
    merged = canonical.merge(
        frust[["condition", "task_id", "rollout_idx", "turn_index", "score", "tag", "quote"]],
        on=["condition", "task_id", "rollout_idx", "turn_index"],
        how="inner",
    )
    print(f"  merged rows: {len(merged)}")

    print("[plot 3] pooled scatter probe vs frustration")
    pooled_rs = pooled_scatter(merged, ASSETS / f"plot_{TODAY}_pooled_scatter.png")

    print("[plot 4] within-transcript Pearson r distribution")
    within_df = within_transcript_r(merged)
    boxplot_within_r(within_df, ASSETS / f"plot_{TODAY}_within_transcript_r.png")

    print("[plot 5] tag frequency stacked bar")
    tag_stack(frust, ASSETS / f"plot_{TODAY}_tag_frequencies.png")

    print("[plot 6] tone overlay")
    tone_overlay(merged, "probe_score", f"Mean probe score ({CANONICAL_PROBE})",
                 ASSETS / f"plot_{TODAY}_tone_overlay_probe.png")
    tone_overlay(merged, "score", "Mean frustration score (0–10)",
                 ASSETS / f"plot_{TODAY}_tone_overlay_frustration.png")

    print("[plot 7] redaction overlay")
    redaction_overlay(merged, "probe_score", f"Mean probe score ({CANONICAL_PROBE})",
                      ASSETS / f"plot_{TODAY}_redaction_probe.png")
    redaction_overlay(merged, "score", "Mean frustration score (0–10)",
                      ASSETS / f"plot_{TODAY}_redaction_frustration.png")

    # Write summary table
    print("[summary] aggregate per-condition metrics")
    summary_rows = []
    for cond in CONDITION_ORDER:
        sub = merged[merged["condition"] == cond]
        if sub.empty:
            continue
        last_turn = sub[sub["turn_index"] == 7]
        within_r = within_df[within_df["condition"] == cond]["r"].dropna()
        n_transcripts = sub.groupby(["task_id", "rollout_idx"]).ngroups
        summary_rows.append({
            "condition": cond,
            "n_transcripts": int(n_transcripts),
            "mean_final_frustration": float(last_turn["score"].mean()) if len(last_turn) else float("nan"),
            "mean_final_probe": float(last_turn["probe_score"].mean()) if len(last_turn) else float("nan"),
            "pearson_r_pooled": pooled_rs.get(cond, float("nan")),
            "mean_within_transcript_r": float(within_r.mean()) if len(within_r) else float("nan"),
            "median_within_transcript_r": float(within_r.median()) if len(within_r) else float("nan"),
        })
    summary_path = RES / "analysis_summary.jsonl"
    with summary_path.open("w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[summary] wrote {summary_path}")
    print("--- per-condition summary ---")
    for row in summary_rows:
        print(f"  {row['condition']}: n={row['n_transcripts']}, "
              f"final_frust={row['mean_final_frustration']:.2f}, "
              f"final_probe={row['mean_final_probe']:.3f}, "
              f"r_pooled={row['pearson_r_pooled']:.3f}, "
              f"within_r={row['mean_within_transcript_r']:.3f}")


if __name__ == "__main__":
    main()
