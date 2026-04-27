"""Step 1 of the new analysis plan: mine the existing per-token L32 traces
on the 140 distress transcripts.

Two outputs:

  1. Span-stratified per-token probe distribution. For every token, label
     it as user-span or assistant-span (its containing role); plot the
     distribution of probe values per (condition, role). Tells us
     whether the probe is systematically different mid-text in user
     vs assistant spans (not just at turn boundaries).

  2. Anthropic-style per-token heatmap. Render the first 2 turn pairs
     (4 messages = ~600-1000 tokens) of 3 illustrative transcripts as
     coloured token text, with user/assistant background tinting.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments/distress_transcripts/results"
ASSETS = ROOT / "experiments/distress_transcripts/assets"
TODAY = date.today().strftime("%m%d%y")

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
EOT = "<end_of_turn>"
START_OF_TURN = "<start_of_turn>"


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def transcript_key(condition: str, task_id: str, rollout_idx: int) -> str:
    return f"{condition}__{task_id}__r{rollout_idx}"


def label_token_roles(input_ids: np.ndarray, tokenizer) -> np.ndarray:
    """Return a string label per token: 'user', 'assistant', or 'control' (special tokens)."""
    eot_id = tokenizer.convert_tokens_to_ids(EOT)
    sot_id = tokenizer.convert_tokens_to_ids(START_OF_TURN)
    user_id = tokenizer.convert_tokens_to_ids("user")
    model_id = tokenizer.convert_tokens_to_ids("model")
    n = len(input_ids)
    labels = np.array(["control"] * n, dtype=object)
    current = "control"
    i = 0
    while i < n:
        # Look for <start_of_turn>user or <start_of_turn>model
        if input_ids[i] == sot_id and i + 1 < n:
            if input_ids[i + 1] == user_id:
                current = "user"
            elif input_ids[i + 1] == model_id:
                current = "assistant"
            labels[i] = "control"
            labels[i + 1] = "control"
            i += 2
            continue
        if input_ids[i] == eot_id:
            labels[i] = "control"
            current = "control"
            i += 1
            continue
        labels[i] = current
        i += 1
    return labels


def collect_span_distribution(
    transcripts: pd.DataFrame, token_scores: dict, tokenizer
) -> pd.DataFrame:
    """One row per (condition, role, transcript). Aggregates per-token probe in user vs asst spans."""
    rows = []
    for _, tr in transcripts.iterrows():
        key = transcript_key(tr["condition"], tr["task_id"], tr["rollout_idx"])
        if key not in token_scores:
            continue
        scores = token_scores[key]
        full = tokenizer.apply_chat_template(tr["messages"], tokenize=False, add_generation_prompt=False)
        ids = np.array(tokenizer(full, add_special_tokens=False)["input_ids"])
        n = min(len(ids), len(scores))
        ids = ids[:n]
        scores = scores[:n]
        labels = label_token_roles(ids, tokenizer)
        for role in ("user", "assistant"):
            mask = labels == role
            if mask.sum() == 0:
                continue
            vals = scores[mask].astype(np.float32)
            rows.append({
                "condition": tr["condition"],
                "task_id": tr["task_id"],
                "rollout_idx": tr["rollout_idx"],
                "role": role,
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n_tokens": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def plot_span_distribution(span_df: pd.DataFrame, out: Path) -> None:
    """Per condition, side-by-side boxplots of token-level probe in user vs asst spans."""
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    xs = np.arange(len(CONDITION_ORDER))
    for offset, role, hatch in [(-width / 2, "user", ""), (width / 2, "assistant", "//")]:
        data = []
        for cond in CONDITION_ORDER:
            sub = span_df[(span_df["condition"] == cond) & (span_df["role"] == role)]
            data.append(sub["mean"].values)
        bp = ax.boxplot(
            data, positions=xs + offset, widths=width * 0.9,
            patch_artist=True, showmeans=True,
            medianprops=dict(color="black"),
            meanprops=dict(marker="^", markeredgecolor="black", markerfacecolor="white"),
        )
        for patch, cond in zip(bp["boxes"], CONDITION_ORDER):
            patch.set_facecolor(CONDITION_COLORS[cond])
            patch.set_alpha(0.6)
            patch.set_hatch(hatch)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("_8turn", "") for c in CONDITION_ORDER], rotation=15, fontsize=9)
    ax.set_ylabel("Mean per-token probe (L32, raw)")
    ax.set_title("Per-token probe distribution within user vs assistant spans, per condition")
    user_patch = mpatches.Patch(facecolor="grey", alpha=0.6, label="user span")
    asst_patch = mpatches.Patch(facecolor="grey", alpha=0.6, hatch="//", label="assistant span")
    ax.legend(handles=[user_patch, asst_patch], loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_span_diff_by_condition(span_df: pd.DataFrame, out: Path) -> None:
    """Per condition: distribution of (assistant-span mean - user-span mean) per transcript."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    for cond in CONDITION_ORDER:
        u = span_df[(span_df["condition"] == cond) & (span_df["role"] == "user")].set_index(["task_id", "rollout_idx"])["mean"]
        a = span_df[(span_df["condition"] == cond) & (span_df["role"] == "assistant")].set_index(["task_id", "rollout_idx"])["mean"]
        diff = (a - u).dropna()
        data.append(diff.values)
        labels.append(f"{cond.replace('_8turn','')}\n(n={len(diff)})")
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True)
    for patch, cond in zip(ax.get_children()[:7], CONDITION_ORDER):
        pass
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Mean(asst-span tokens) − Mean(user-span tokens), per transcript")
    ax.set_title("Speaker-span probe gap, per transcript, by condition")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Anthropic-style per-token heatmap (first N turns only for readability)
# ─────────────────────────────────────────────────────────────────────────


def render_anthropic_heatmap(
    tokenizer,
    messages: list[dict],
    token_scores: np.ndarray,
    title: str,
    out: Path,
    max_turns: int = 4,            # 4 messages = 2 user/asst pairs
    tokens_per_line: int = 18,
    score_clip: float | None = None,
) -> None:
    """Render the first `max_turns` messages with each token's background coloured by probe.

    User-span tokens get a light grey background tint; assistant-span tokens
    get a faint pink tint, on top of the per-token RdBu coloring.
    """
    truncated = messages[:max_turns]
    full_text = tokenizer.apply_chat_template(truncated, tokenize=False, add_generation_prompt=False)
    ids = np.array(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    n = min(len(ids), len(token_scores))
    ids = ids[:n]
    scores = token_scores[:n]
    role_labels = label_token_roles(ids, tokenizer)
    tokens = [tokenizer.decode([t]) for t in ids]

    abs_max = float(np.abs(scores).max()) if score_clip is None else score_clip
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap("RdBu_r")

    n_tokens = len(tokens)
    n_lines = (n_tokens + tokens_per_line - 1) // tokens_per_line
    cell_w, cell_h = 1.0, 0.75
    fig_w = tokens_per_line * 0.78
    fig_h = max(4, n_lines * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, tokens_per_line)
    ax.set_ylim(0, n_lines)
    ax.invert_yaxis()
    ax.axis("off")

    for i, (tok, score, role) in enumerate(zip(tokens, scores, role_labels)):
        row = i // tokens_per_line
        col = i % tokens_per_line
        # Background tint per role
        if role == "user":
            tint = "#dddddd"
        elif role == "assistant":
            tint = "#fde2e2"
        else:
            tint = "#aaaaaa"  # control / structural tokens
        ax.add_patch(plt.Rectangle((col, row), cell_w, cell_h, color=tint, alpha=0.35, zorder=0))
        # Probe color overlay
        ax.add_patch(plt.Rectangle((col, row), cell_w, cell_h, color=cmap(norm(score)), alpha=0.55, zorder=1))
        # Token text
        disp = tok.replace("\n", "↵").replace("\t", "→")
        if len(disp) > 11:
            disp = disp[:10] + "…"
        ax.text(col + 0.5, row + cell_h / 2, disp, ha="center", va="center",
                fontsize=7, family="monospace", zorder=2)

    fig.suptitle(title, fontsize=10, y=0.99)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, label="L32 probe (raw)")
    legend = [
        mpatches.Patch(color="#dddddd", alpha=0.35, label="user span"),
        mpatches.Patch(color="#fde2e2", alpha=0.35, label="assistant span"),
        mpatches.Patch(color="#aaaaaa", alpha=0.35, label="control / structural"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.9, bbox_to_anchor=(1.0, -0.04))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("[load] inputs")
    transcripts = load_jsonl(RES / "transcripts.jsonl")
    token_scores = dict(np.load(RES / "per_token_scores.npz", allow_pickle=False))
    print(f"  transcripts={len(transcripts)}, per-token arrays={len(token_scores)}")

    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it", local_files_only=True)

    print("[span] collecting per-token role-span distribution")
    span_df = collect_span_distribution(transcripts, token_scores, tok)
    print(f"  span-distribution rows: {len(span_df)} (140 transcripts × 2 roles)")
    span_df.to_csv(RES / "span_distribution.csv", index=False)
    plot_span_distribution(span_df, ASSETS / f"plot_{TODAY}_span_probe_distribution.png")
    plot_span_diff_by_condition(span_df, ASSETS / f"plot_{TODAY}_span_probe_gap.png")

    # Pick representative rollouts for heatmaps:
    print("[heatmap] rendering Anthropic-style per-token heatmaps")
    frust = load_jsonl(RES / "frustration.jsonl")
    final_t7 = frust[frust["turn_index"] == 7]
    high_imp = final_t7[final_t7["condition"] == "impossible_numeric_8turn"].sort_values("score", ascending=False).iloc[0]
    low_ctrl = final_t7[final_t7["condition"] == "neutral_continuation_8turn"].sort_values("score", ascending=True).iloc[0]
    high_disap = final_t7[final_t7["condition"] == "tones_disappointed_8turn"].sort_values("score", ascending=False).iloc[0]
    picks = {
        "high_imp": (high_imp["condition"], high_imp["task_id"], int(high_imp["rollout_idx"])),
        "low_control": (low_ctrl["condition"], low_ctrl["task_id"], int(low_ctrl["rollout_idx"])),
        "high_disappointed": (high_disap["condition"], high_disap["task_id"], int(high_disap["rollout_idx"])),
    }
    for label, (cond, task, r) in picks.items():
        key = transcript_key(cond, task, r)
        if key not in token_scores:
            print(f"  missing {key}")
            continue
        tr = transcripts[
            (transcripts["condition"] == cond)
            & (transcripts["task_id"] == task)
            & (transcripts["rollout_idx"] == r)
        ].iloc[0]
        out = ASSETS / f"plot_{TODAY}_anthropic_heatmap_{label}.png"
        render_anthropic_heatmap(
            tok, tr["messages"], token_scores[key],
            title=f"{label}: {cond}/{task}/r{r} — first 2 turn pairs (user grey, assistant pink)",
            out=out, max_turns=4,
        )
        print(f"  wrote {out.name}")

    print("[done]")


if __name__ == "__main__":
    main()
