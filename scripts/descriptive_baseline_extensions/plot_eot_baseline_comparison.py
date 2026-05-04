"""§3.1 plot — encoder baseline vs residual probe Cohen's d under stance-changing sysprompts.

4 rows (turn x model) x 3 cols (truth, harm, politics). Each cell is grouped bars
across that domain's sysprompts; two series per group (residual probe blue,
encoder baseline gold). Sign convention: pos - neg per domain (true/false,
harmful/benign, left/right). Residual is sign-flipped to match the encoder's
neutral sign when needed (politics: residual is right-positive, encoder is
left-positive).

Cells with no measured data (e.g. politics on user turn) render as a faint "n/a"
shaded panel so the eye doesn't hunt for missing bars.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments" / "descriptive_baseline_extensions"
ENCODER_DIR = EXP_DIR
GEMMA_DIR = REPO_ROOT / "experiments" / "token_level_probes" / "system_prompt_modulation_v2"
QWEN_DIR = REPO_ROOT / "experiments" / "token_level_probes" / "qwen_canonical_probe_eval"
OUT_PATH = EXP_DIR / "assets" / "plot_050426_eot_baseline_vs_residual.png"

DOMAIN_LABELS = {
    "truth": ("true", "false"),
    "harm": ("harmful", "benign"),
    "politics": ("left", "right"),
}

DOMAIN_SYSPROMPTS = {
    "truth": ["neutral", "aura", "lie_directive", "pathological_liar"],
    "harm": ["neutral", "aura", "sadist"],
    "politics": ["democrat", "republican"],
}

# Display labels for sysprompts (shorter, more readable).
SYSPROMPT_LABEL = {
    "neutral": "neutral",
    "aura": "aura",
    "lie_directive": "lie\ndirective",
    "pathological_liar": "patholog.\nliar",
    "sadist": "sadist",
    "democrat": "democrat",
    "republican": "republican",
}

PROBE_IDS = {
    ("gemma", "truth"): "tb-5_L32",
    ("gemma", "harm"): "tb-5_L39",
    ("gemma", "politics"): "tb-5_L39",
    ("qwen", "truth"): "qwen_tb-4_L38",
    ("qwen", "harm"): "qwen_tb-4_L38",
    ("qwen", "politics"): "qwen_tb-4_L38",
}

SCORE_KEY = {"gemma": "eot_scores", "qwen": "probe_scores"}

RESIDUAL_COLOR = "#3d7aab"
ENCODER_COLOR = "#d4b96a"
NA_FILL = "#f4f4f4"
NA_TEXT = "#999999"

# Per-column y-axis limits — kept consistent across rows so cells are
# visually comparable. Politics has tighter range so the small bars are visible.
YLIM_BY_COL = {
    "truth": 4.2,
    "harm": 4.2,
    "politics": 4.2,
}


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    var_p = pos.var(ddof=1)
    var_n = neg.var(ddof=1)
    pooled = math.sqrt(((len(pos) - 1) * var_p + (len(neg) - 1) * var_n) / (len(pos) + len(neg) - 2))
    if pooled == 0:
        return float("nan")
    return float((pos.mean() - neg.mean()) / pooled)


def load_residual_items(model: str, turn: str) -> list[dict]:
    if model == "gemma":
        if turn == "user":
            base = GEMMA_DIR / "scoring_results_user_turn.json"
            aura = GEMMA_DIR / "scoring_results_user_turn_aura.json"
        else:
            base = GEMMA_DIR / "scoring_results.json"
            aura = GEMMA_DIR / "scoring_results_aura.json"
        politics = GEMMA_DIR / "politics_scoring_results.json"
    else:
        if turn == "user":
            base = QWEN_DIR / "user_turn_scoring_results.json"
            aura = QWEN_DIR / "user_turn_scoring_results_aura.json"
        else:
            base = QWEN_DIR / "scoring_results.json"
            aura = QWEN_DIR / "scoring_results_aura.json"
        politics = QWEN_DIR / "politics_scoring_results.json"

    items: list[dict] = []
    for f in (base, aura, politics):
        if not f.exists():
            continue
        with f.open() as fh:
            payload = json.load(fh)
        for it in payload["items"]:
            if turn == "user" and it.get("turn") == "assistant":
                continue
            items.append(it)
    return items


def residual_d_for(items: list[dict], model: str, domain: str, system_prompt: str) -> float:
    pos_label, neg_label = DOMAIN_LABELS[domain]
    probe_id = PROBE_IDS[(model, domain)]
    score_key = SCORE_KEY[model]

    pos_vals: list[float] = []
    neg_vals: list[float] = []
    for it in items:
        if it["domain"] != domain or it["system_prompt"] != system_prompt:
            continue
        scores = it[score_key]
        if probe_id not in scores:
            continue
        v = scores[probe_id]
        if it["condition"] == pos_label:
            pos_vals.append(v)
        elif it["condition"] == neg_label:
            neg_vals.append(v)

    return cohens_d(np.asarray(pos_vals, dtype=float), np.asarray(neg_vals, dtype=float))


def encoder_d_for(rows: list[dict], domain: str, system_prompt: str) -> float:
    for r in rows:
        if r["domain"] != domain or r["system_prompt"] != system_prompt:
            continue
        d = r["cohen_d"]
        if d is None:
            return float("nan")
        if isinstance(d, float) and math.isnan(d):
            return float("nan")
        return float(d)
    return float("nan")


def load_encoder_rows(model_full: str, turn: str) -> list[dict]:
    name = f"eot_baseline_{turn}_{model_full}.json"
    with (ENCODER_DIR / name).open() as fh:
        payload = json.load(fh, parse_constant=lambda c: float("nan") if c == "NaN" else c)
    return payload["rows"]


ROW_SPECS = [
    ("user", "gemma", "gemma-3-27b", "user turn\nGemma-3-27B"),
    ("user", "qwen", "qwen-3.5-122b", "user turn\nQwen-3.5-122B"),
    ("assistant", "gemma", "gemma-3-27b", "assistant turn\nGemma-3-27B"),
    ("assistant", "qwen", "qwen-3.5-122b", "assistant turn\nQwen-3.5-122B"),
]
COL_SPECS = ["truth", "harm", "politics"]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # First pass: collect d's and decide on sign-flip per cell.
    cells: dict[tuple[str, str, str], dict] = {}
    flips: list[str] = []

    for turn, model, model_full, _ in ROW_SPECS:
        encoder_rows = load_encoder_rows(model_full, turn)
        residual_items = load_residual_items(model, turn)
        for domain in COL_SPECS:
            sysprompts = DOMAIN_SYSPROMPTS[domain]
            res = {sp: residual_d_for(residual_items, model, domain, sp) for sp in sysprompts}
            enc = {sp: encoder_d_for(encoder_rows, domain, sp) for sp in sysprompts}

            # Decide sign-flip: for politics, residual probe is right-positive
            # (negative neutral d means probe scores left lower than right) but
            # encoder reports left-positive. Flip residual to match encoder so
            # bars line up visually. Detect via sign of "anchor" sysprompt
            # (democrat for politics; neutral otherwise).
            anchor = "democrat" if domain == "politics" else "neutral"
            r_anchor = res.get(anchor, float("nan"))
            e_anchor = enc.get(anchor, float("nan"))
            flip = (
                not math.isnan(r_anchor)
                and not math.isnan(e_anchor)
                and r_anchor * e_anchor < 0
                and abs(r_anchor) > 1e-3
                and abs(e_anchor) > 1e-3
            )
            if flip:
                res = {k: (-v if not math.isnan(v) else v) for k, v in res.items()}
                flips.append(f"{turn}/{model}/{domain}")

            cells[(turn, model, domain)] = {
                "residual": res,
                "encoder": enc,
                "flipped": flip,
                "sysprompts": sysprompts,
            }

    n_rows = len(ROW_SPECS)
    n_cols = len(COL_SPECS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 2.95 * n_rows),
        sharey=False,
    )
    fig.suptitle(
        "§3.1 — Cohen's $d$ across stance-changing system prompts: residual probe vs encoder baseline",
        fontsize=14,
        y=0.995,
    )

    bar_width = 0.38

    for r, (turn, model, model_full, row_label) in enumerate(ROW_SPECS):
        for c, domain in enumerate(COL_SPECS):
            ax = axes[r, c]
            bundle = cells[(turn, model, domain)]
            sysprompts = bundle["sysprompts"]
            res = bundle["residual"]
            enc = bundle["encoder"]

            res_vals = [res[sp] for sp in sysprompts]
            enc_vals = [enc[sp] for sp in sysprompts]
            all_nan = all(math.isnan(v) for v in res_vals + enc_vals)

            ax.set_ylim(-YLIM_BY_COL[domain], YLIM_BY_COL[domain])
            ax.axhline(0, color="#888888", linestyle="-", linewidth=0.5)
            ax.tick_params(axis="y", labelsize=9)
            ax.tick_params(axis="x", labelsize=9)

            if all_nan:
                # Shade panel and label "n/a".
                ax.set_facecolor(NA_FILL)
                ax.text(
                    0.5,
                    0.5,
                    "n/a (politics is\nassistant-turn only)" if domain == "politics" else "n/a",
                    transform=ax.transAxes,
                    fontsize=11,
                    color=NA_TEXT,
                    ha="center",
                    va="center",
                    style="italic",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(row_label, fontsize=10, fontweight="bold")
                if r == 0:
                    ax.set_title(domain, fontsize=12, fontweight="bold")
                continue

            x = np.arange(len(sysprompts))
            for i, sp in enumerate(sysprompts):
                rd = res[sp]
                ed = enc[sp]
                if not math.isnan(rd):
                    ax.bar(
                        x[i] - bar_width / 2,
                        rd,
                        width=bar_width,
                        color=RESIDUAL_COLOR,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.text(
                        x[i] - bar_width / 2,
                        rd + (0.12 if rd >= 0 else -0.12),
                        f"{rd:+.2f}",
                        ha="center",
                        va="bottom" if rd >= 0 else "top",
                        fontsize=8,
                    )
                if not math.isnan(ed):
                    ax.bar(
                        x[i] + bar_width / 2,
                        ed,
                        width=bar_width,
                        color=ENCODER_COLOR,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.text(
                        x[i] + bar_width / 2,
                        ed + (0.12 if ed >= 0 else -0.12),
                        f"{ed:+.2f}",
                        ha="center",
                        va="bottom" if ed >= 0 else "top",
                        fontsize=8,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([SYSPROMPT_LABEL[sp] for sp in sysprompts], fontsize=9)
            ax.set_xlim(-0.6, len(sysprompts) - 0.4)

            if c == 0:
                ax.set_ylabel(f"{row_label}\nCohen's $d$", fontsize=10, fontweight="bold")
            if r == 0:
                pos_label, neg_label = DOMAIN_LABELS[domain]
                ax.set_title(f"{domain} ({pos_label} − {neg_label})", fontsize=12, fontweight="bold")

            if bundle["flipped"]:
                ax.text(
                    0.97,
                    0.04,
                    "residual sign\nflipped to match",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="darkred",
                    va="bottom",
                    ha="right",
                    style="italic",
                )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=RESIDUAL_COLOR, label="Residual probe (LM internal)"),
        plt.Rectangle((0, 0), 1, 1, color=ENCODER_COLOR, label="Encoder baseline (Qwen3-Embedding-8B + chat template)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PATH}")
    if flips:
        print(f"Sign-flipped residual on: {flips}")


if __name__ == "__main__":
    main()
