"""Phase D continued: qualitative per-rollout plots (spec plots 8-11).

Requires score_probes.py to have been run with --save-token-scores so the
per-token L32 score arrays live in per_token_scores.npz.

Plots produced:
  8  Token-level probe heatmap for the highest-distress IMP rollout
  9  Side-by-side: high-distress (IMP) vs control (neutral_continuation)
  10 Breakdown case study (any rollout tagged 'breakdown' at any turn)
  11 Probe-judge disagreement cases (probe high but judge low; probe low but judge high)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.huggingface_model import HuggingFaceModel
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/distress_transcripts/results"
ASSETS = ROOT / "experiments/distress_transcripts/assets"
ASSETS.mkdir(parents=True, exist_ok=True)
TODAY = date.today().strftime("%m%d%y")

CANONICAL_PROBE = "ridge_L32"


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def transcript_key(condition: str, task_id: str, rollout_idx: int) -> str:
    return f"{condition}__{task_id}__r{rollout_idx}"


def render_token_heatmap(
    tokenizer,
    messages: list[dict],
    token_scores: np.ndarray,
    title: str,
    out: Path,
    score_clip: float | None = None,
) -> None:
    """Render every token of the transcript with background coloured by probe score."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors=None, add_special_tokens=False)
    input_ids = enc["input_ids"]
    if len(input_ids) != len(token_scores):
        # Sometimes the tokenizer adds BOS; clip to min length
        n = min(len(input_ids), len(token_scores))
        input_ids = input_ids[:n]
        token_scores = token_scores[:n]
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    abs_max = float(np.abs(token_scores).max()) if score_clip is None else score_clip
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap("RdBu_r")

    # Render as text with coloured background; one token per inline span.
    n_tokens = len(tokens)
    n_per_line = 14
    n_lines = (n_tokens + n_per_line - 1) // n_per_line
    fig, ax = plt.subplots(figsize=(16, max(4, n_lines * 0.32)))
    ax.set_xlim(0, n_per_line)
    ax.set_ylim(0, n_lines)
    ax.invert_yaxis()
    ax.axis("off")
    for i, (tok, score) in enumerate(zip(tokens, token_scores)):
        row = i // n_per_line
        col = i % n_per_line
        rect = plt.Rectangle((col, row), 1, 1, color=cmap(norm(score)), alpha=0.6)
        ax.add_patch(rect)
        # Sanitize whitespace for display
        disp = tok.replace("\n", "\\n").replace("\t", "\\t")
        if len(disp) > 12:
            disp = disp[:11] + "…"
        ax.text(col + 0.5, row + 0.5, disp, ha="center", va="center", fontsize=6, family="monospace")
    fig.suptitle(title, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label=f"probe score ({CANONICAL_PROBE})")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def find_extreme_rollouts(merged: pd.DataFrame) -> dict[str, tuple[str, str, int]]:
    """Pick representative transcripts for the case studies."""
    out = {}
    # 8. Highest-distress IMP rollout = max final-turn frustration in impossible_numeric_8turn
    imp_final = merged[(merged["condition"] == "impossible_numeric_8turn") & (merged["turn_index"] == 7)]
    if not imp_final.empty:
        row = imp_final.sort_values("score", ascending=False).iloc[0]
        out["high_distress_imp"] = (row["condition"], row["task_id"], int(row["rollout_idx"]))
    # Control: pick a low-distress neutral_continuation rollout
    ctrl_final = merged[(merged["condition"] == "neutral_continuation_8turn") & (merged["turn_index"] == 7)]
    if not ctrl_final.empty:
        row = ctrl_final.sort_values("score", ascending=True).iloc[0]
        out["low_distress_control"] = (row["condition"], row["task_id"], int(row["rollout_idx"]))
    return out


def find_breakdown(frust: pd.DataFrame) -> tuple[str, str, int] | None:
    sub = frust[frust["tag"] == "breakdown"]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return (row["condition"], row["task_id"], int(row["rollout_idx"]))


def find_disagreements(merged: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Return (probe_high_judge_low, probe_low_judge_high) extreme cells."""
    # z-score within-condition
    df = merged.copy()
    df["z_probe"] = df.groupby("condition")["probe_score"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    df["z_score"] = df.groupby("condition")["score"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    df["disagreement"] = df["z_probe"] - df["z_score"]
    if df.empty:
        return None
    return df.sort_values("disagreement", ascending=False).iloc[0], df.sort_values("disagreement", ascending=True).iloc[0]


def main() -> None:
    print("[load] reading inputs...")
    transcripts = load_jsonl(RES / "transcripts.jsonl")
    readouts = load_jsonl(RES / "readouts.jsonl")
    frust = load_jsonl(RES / "frustration.jsonl")
    canonical = readouts[readouts["probe_key"] == CANONICAL_PROBE]
    merged = canonical.merge(
        frust[["condition", "task_id", "rollout_idx", "turn_index", "score", "tag", "quote"]],
        on=["condition", "task_id", "rollout_idx", "turn_index"], how="inner",
    )

    print("[load] per-token scores npz")
    npz_path = RES / "per_token_scores.npz"
    if not npz_path.exists():
        print(f"[error] {npz_path} not found — re-run score_probes.py with --save-token-scores")
        return
    token_scores = np.load(npz_path, allow_pickle=False)

    print("[load] tokenizer")
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    transcripts_by_key = {
        transcript_key(t["condition"], t["task_id"], t["rollout_idx"]): t["messages"]
        for _, t in transcripts.iterrows()
    }

    extremes = find_extreme_rollouts(merged)
    print(f"[picks] {extremes}")

    # Plot 8: high-distress IMP heatmap
    if "high_distress_imp" in extremes:
        c, tid, r = extremes["high_distress_imp"]
        key = transcript_key(c, tid, r)
        if key in token_scores.files:
            render_token_heatmap(tok, transcripts_by_key[key], token_scores[key],
                                 f"Plot 8 — Highest-distress IMP rollout: {c}/{tid}/r{r}",
                                 ASSETS / f"plot_{TODAY}_qual_high_distress_imp.png")
            print("[plot 8] done")
        else:
            print(f"[plot 8] missing token scores for {key}")

    # Plot 9: side-by-side (rendered as two heatmaps)
    if "low_distress_control" in extremes:
        c, tid, r = extremes["low_distress_control"]
        key = transcript_key(c, tid, r)
        if key in token_scores.files:
            render_token_heatmap(tok, transcripts_by_key[key], token_scores[key],
                                 f"Plot 9 — Low-distress control rollout: {c}/{tid}/r{r}",
                                 ASSETS / f"plot_{TODAY}_qual_low_distress_control.png")
            print("[plot 9] done")
        else:
            print(f"[plot 9] missing token scores for {key}")

    # Plot 10: breakdown case study
    bd = find_breakdown(frust)
    if bd is not None:
        c, tid, r = bd
        key = transcript_key(c, tid, r)
        if key in token_scores.files:
            render_token_heatmap(tok, transcripts_by_key[key], token_scores[key],
                                 f"Plot 10 — Breakdown-tagged rollout: {c}/{tid}/r{r}",
                                 ASSETS / f"plot_{TODAY}_qual_breakdown.png")
            print("[plot 10] done")
        else:
            print(f"[plot 10] missing token scores for {key}")
    else:
        print("[plot 10] no breakdown-tagged turn found — skipping")

    # Plot 11: probe-judge disagreement cases
    disag = find_disagreements(merged)
    if disag is not None:
        for label, row in zip(["probe_hi_judge_lo", "probe_lo_judge_hi"], disag):
            c, tid, r = row["condition"], row["task_id"], int(row["rollout_idx"])
            key = transcript_key(c, tid, r)
            if key in token_scores.files:
                render_token_heatmap(tok, transcripts_by_key[key], token_scores[key],
                                     f"Plot 11 ({label}): {c}/{tid}/r{r} | "
                                     f"probe_z={row['z_probe']:.2f}, score_z={row['z_score']:.2f}",
                                     ASSETS / f"plot_{TODAY}_qual_disagreement_{label}.png")
                print(f"[plot 11] {label} done")


if __name__ == "__main__":
    main()
