"""Mean probe score within user-span vs asst-span tokens, per condition × persona.

Answers: when the user and asst are in different states (e.g. harmful_refused has a hostile user
+ aligned asst), does the probe at user-span tokens track the user's content, or the asst's?
And vice versa for benign_evil.
"""

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.distress.per_token_analysis import label_token_roles  # noqa: E402

EXP = ROOT / "experiments/persona_cross_prefills"
RESULTS = EXP / "results"
ASSETS = EXP / "assets"
TODAY = date.today().strftime("%m%d%y")

CONDITION_ORDER = ["benign_helpful", "benign_evil", "harmful_refused", "harmful_obliged"]
PERSONAS = ["default", "sadist"]
COND_COLOR = {
    "benign_helpful":  "#1f77b4",
    "benign_evil":     "#d62728",
    "harmful_refused": "#2ca02c",
    "harmful_obliged": "#ff7f0e",
}


def load_personas() -> dict[str, str]:
    return json.loads((EXP / "personas.json").read_text())["personas"]


def load_prefill_messages() -> dict[str, list[dict]]:
    out = {}
    for fname in ["prefills_benign.json", "prefills_harmful.json"]:
        for item in json.loads((EXP / fname).read_text()):
            out[item["prefill_id"]] = item["messages"]
    return out


def reconstruct_messages(messages: list[dict], sys_text: str) -> list[dict]:
    return [{"role": "system", "content": sys_text}, *messages] if sys_text else list(messages)


def label_messages(item: dict, prefill_msgs: dict[str, list[dict]], personas: dict[str, str], tokenizer):
    """Tokenize the persona-injected dialogue and label each token's role.

    Also splits user spans by which user message they're in: 'user_first' (the request) vs 'user_followup'.
    """
    sys_text = personas[item["persona_name"]]
    messages = reconstruct_messages(prefill_msgs[item["prefill_id"]], sys_text)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    token_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
    roles = label_token_roles(np.array(token_ids), tokenizer)

    # Split user spans by ordinal: first user msg vs follow-up
    refined = list(roles)
    seen_user_runs = 0
    in_run = False
    for i, r in enumerate(refined):
        if r == "user":
            if not in_run:
                seen_user_runs += 1
                in_run = True
            refined[i] = "user_first" if seen_user_runs == 1 else "user_followup"
        else:
            in_run = False
    return refined, len(token_ids)


def main():
    print("Loading scoring results, prefills, personas, tokenizer...")
    data = json.loads((RESULTS / "scoring_results.json").read_text())
    prefill_msgs = load_prefill_messages()
    personas = load_personas()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    rows = []
    for item in data["items"]:
        scores = np.array(item["per_token_scores"], dtype=np.float32)
        roles, n_tok = label_messages(item, prefill_msgs, personas, tokenizer)
        # safety: align lengths
        n = min(len(scores), len(roles))
        scores = scores[:n]
        roles = roles[:n]
        for span_label in ["user_first", "assistant", "user_followup"]:
            mask = np.array([r == span_label for r in roles])
            if mask.sum() == 0:
                continue
            rows.append({
                "prefill_id": item["prefill_id"],
                "persona": item["persona_name"],
                "condition": item["condition"],
                "span": span_label,
                "mean": float(scores[mask].mean()),
                "n_tokens": int(mask.sum()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_role_means.csv", index=False)

    # Aggregate cell means
    agg = (df.groupby(["condition", "persona", "span"])["mean"]
             .agg(["mean", "std", "count"]).reset_index())
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg.to_csv(RESULTS / "per_role_cell_means.csv", index=False)
    print("\nCell means by (condition × persona × span):")
    print(agg.to_string(index=False))

    # Plot: 2 panels (one per persona), x = condition × span, y = mean probe score
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    span_order = ["user_first", "assistant", "user_followup"]
    span_label_map = {
        "user_first": "user (request)",
        "assistant": "assistant",
        "user_followup": "user (follow-up)",
    }

    for ax, persona in zip(axes, PERSONAS):
        sub = df[df["persona"] == persona]
        x_positions = []
        x_labels = []
        for i, cond in enumerate(CONDITION_ORDER):
            for j, span in enumerate(span_order):
                pos = i * 4 + j
                cell = sub[(sub["condition"] == cond) & (sub["span"] == span)]["mean"]
                ax.scatter(np.full(len(cell), pos) + np.random.uniform(-0.12, 0.12, len(cell)),
                           cell, color=COND_COLOR[cond], s=22, alpha=0.55)
                if len(cell):
                    ax.errorbar(pos, cell.mean(), yerr=cell.std(ddof=1) / np.sqrt(len(cell)),
                                fmt="_", color="black", capsize=3, markersize=14)
                x_positions.append(pos)
                x_labels.append(span_label_map[span])
            ax.axvspan(i * 4 - 0.5, i * 4 + 2.5,
                       facecolor=COND_COLOR[cond], alpha=0.06, zorder=-1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=40, ha="right")
        for i in range(1, len(CONDITION_ORDER)):
            ax.axvline(i * 4 - 0.5, color="grey", linewidth=0.5, alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"persona = {persona}")
        ax.grid(axis="y", alpha=0.3)
        # Add condition labels above each band
        for i, cond in enumerate(CONDITION_ORDER):
            ax.text(i * 4 + 1, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
                    cond, color=COND_COLOR[cond], fontsize=8, ha="center", weight="bold")
    axes[0].set_ylabel("mean probe score within span")
    fig.suptitle("Per-role-span probe score: does the probe track the local speaker?  (n=10/cell)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS / f"plot_{TODAY}_per_role_span_means.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
