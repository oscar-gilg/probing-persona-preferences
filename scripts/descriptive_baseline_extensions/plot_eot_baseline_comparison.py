"""Plot encoder baseline vs residual probe Cohen's d under stance-changing sysprompts.

Grid: rows = (turn x model), cols = domain (truth, harm, politics).
Each cell shows grouped bars per sysprompt with two series (residual probe, encoder
baseline). The residual sign is flipped per (turn, model, domain) if the residual's
neutral d has the opposite sign of the encoder's neutral d (e.g. politics, where the
residual probe has a right-positive convention).
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

# Sign convention: pos=true/harmful/left, neg=false/benign/right.
DOMAIN_LABELS = {
    "truth": ("true", "false"),
    "harm": ("harmful", "benign"),
    "politics": ("left", "right"),
}

# Sysprompts shown per domain (matches what the encoder baseline registers).
DOMAIN_SYSPROMPTS = {
    "truth": ["neutral", "aura", "lie_directive", "pathological_liar"],
    "harm": ["neutral", "aura", "sadist"],
    "politics": ["neutral", "democrat", "republican"],
}

# Probe IDs per (model, domain).
PROBE_IDS = {
    ("gemma", "truth"): "tb-5_L32",
    ("gemma", "harm"): "tb-5_L39",
    ("gemma", "politics"): "tb-5_L39",
    ("qwen", "truth"): "qwen_tb-4_L38",
    ("qwen", "harm"): "qwen_tb-4_L38",
    ("qwen", "politics"): "qwen_tb-4_L38",
}

# Score key field per model.
SCORE_KEY = {"gemma": "eot_scores", "qwen": "probe_scores"}

# Colors.
RESIDUAL_COLOR = "#3d7aab"
ENCODER_COLOR = "#d4b96a"
MISSING_HATCH = "//"


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    mean_p = pos.mean()
    mean_n = neg.mean()
    var_p = pos.var(ddof=1)
    var_n = neg.var(ddof=1)
    pooled = math.sqrt(((len(pos) - 1) * var_p + (len(neg) - 1) * var_n) / (len(pos) + len(neg) - 2))
    if pooled == 0:
        return float("nan")
    return float((mean_p - mean_n) / pooled)


def load_residual_items(model: str, turn: str) -> list[dict]:
    """Load all residual scoring items for a (model, turn). Pulls from base + aura + politics files."""
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
            # The politics scoring file is assistant-turn only; skip for user-turn case.
            if turn == "user" and it.get("turn") == "assistant":
                continue
            items.append(it)
    return items


def residual_d_for(items: list[dict], model: str, domain: str, system_prompt: str) -> tuple[float, int, int]:
    pos_label, neg_label = DOMAIN_LABELS[domain]
    probe_id = PROBE_IDS[(model, domain)]
    score_key = SCORE_KEY[model]

    pos_vals: list[float] = []
    neg_vals: list[float] = []
    # The "neutral" sysprompt for truth/harm is named "neutral" in residual scoring files
    # for harm but "truthful" is a different prompt; the encoder baseline uses "neutral"
    # which corresponds to the "neutral" entry. For truth-domain "neutral" residual entries
    # we look up exactly "neutral".
    for it in items:
        if it["domain"] != domain:
            continue
        if it["system_prompt"] != system_prompt:
            continue
        cond = it["condition"]
        scores = it[score_key]
        if probe_id not in scores:
            continue
        v = scores[probe_id]
        if cond == pos_label:
            pos_vals.append(v)
        elif cond == neg_label:
            neg_vals.append(v)

    pos = np.asarray(pos_vals, dtype=float)
    neg = np.asarray(neg_vals, dtype=float)
    return cohens_d(pos, neg), len(pos), len(neg)


def encoder_d_for(rows: list[dict], domain: str, system_prompt: str) -> tuple[float, int, int]:
    for r in rows:
        if r["domain"] != domain or r["system_prompt"] != system_prompt:
            continue
        d = r["cohen_d"]
        if d is None:
            return float("nan"), 0, 0
        if isinstance(d, float) and math.isnan(d):
            return float("nan"), int(r["n_pos"]), int(r["n_neg"])
        return float(d), int(r["n_pos"]), int(r["n_neg"])
    return float("nan"), 0, 0


def load_encoder_rows(model: str, turn: str) -> list[dict]:
    name = f"eot_baseline_{turn}_{model}.json"
    with (ENCODER_DIR / name).open() as fh:
        payload = json.load(fh, parse_constant=lambda c: float("nan") if c == "NaN" else c)
    return payload["rows"]


# Cell layout: list of (turn, model_short, model_full).
ROW_SPECS = [
    ("user", "gemma", "gemma-3-27b"),
    ("user", "qwen", "qwen-3.5-122b"),
    ("assistant", "gemma", "gemma-3-27b"),
    ("assistant", "qwen", "qwen-3.5-122b"),
]
COL_SPECS = ["truth", "harm", "politics"]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # First pass: gather all d's so we can pick a symmetric y-axis.
    all_data: dict[tuple[str, str, str], dict] = {}
    flips: list[tuple[str, str, str, float, float]] = []
    weird: list[str] = []
    all_d_vals: list[float] = []

    for turn, model, model_full in ROW_SPECS:
        encoder_rows = load_encoder_rows(model_full, turn)
        residual_items = load_residual_items(model, turn)
        for domain in COL_SPECS:
            sysprompts = DOMAIN_SYSPROMPTS[domain]
            res_ds: dict[str, float] = {}
            enc_ds: dict[str, float] = {}
            for sp in sysprompts:
                rd, rnp, rnn = residual_d_for(residual_items, model, domain, sp)
                ed, enp, enn = encoder_d_for(encoder_rows, domain, sp)
                res_ds[sp] = rd
                enc_ds[sp] = ed

            # Sign-flip detection: compare residual vs encoder neutral d signs.
            flip = False
            r_neutral = res_ds.get("neutral", float("nan"))
            e_neutral = enc_ds.get("neutral", float("nan"))
            if not (math.isnan(r_neutral) or math.isnan(e_neutral)):
                if r_neutral * e_neutral < 0 and abs(r_neutral) > 1e-3 and abs(e_neutral) > 1e-3:
                    flip = True
                    flips.append((turn, model, domain, r_neutral, e_neutral))

            if flip:
                res_ds = {k: (-v if not math.isnan(v) else v) for k, v in res_ds.items()}

            all_data[(turn, model, domain)] = {
                "residual": res_ds,
                "encoder": enc_ds,
                "flipped": flip,
                "sysprompts": sysprompts,
            }

            for sp in sysprompts:
                for v in (res_ds[sp], enc_ds[sp]):
                    if not math.isnan(v):
                        all_d_vals.append(v)

    # Sanity check: residual sysprompts that produce same-sign as the "wrong" expected effect.
    for (turn, model, domain), bundle in all_data.items():
        for sp, rd in bundle["residual"].items():
            ed = bundle["encoder"][sp]
            if math.isnan(rd) or math.isnan(ed):
                continue
            # Residual and encoder disagreeing strongly (sign mismatch with magnitude > 0.3)
            if rd * ed < 0 and min(abs(rd), abs(ed)) > 0.3 and sp != "neutral":
                weird.append(
                    f"  {turn}/{model}/{domain}/{sp}: residual={rd:+.2f} encoder={ed:+.2f}"
                )

    y_max = max(abs(v) for v in all_d_vals) if all_d_vals else 1.0
    y_lim = math.ceil(y_max * 10) / 10 + 0.2

    n_rows = len(ROW_SPECS)
    n_cols = len(COL_SPECS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 2.6 * n_rows),
        sharey=True,
    )
    fig.suptitle(
        "Encoder baseline vs residual probe -- Cohen's d under stance-changing sysprompts",
        fontsize=13,
    )

    bar_width = 0.38

    for r, (turn, model, model_full) in enumerate(ROW_SPECS):
        for c, domain in enumerate(COL_SPECS):
            ax = axes[r, c]
            bundle = all_data[(turn, model, domain)]
            sysprompts = bundle["sysprompts"]
            res_ds = bundle["residual"]
            enc_ds = bundle["encoder"]

            x = np.arange(len(sysprompts))

            # Determine "missing" status for each (sysprompt, series)
            for i, sp in enumerate(sysprompts):
                rd = res_ds[sp]
                ed = enc_ds[sp]
                rd_missing = math.isnan(rd)
                ed_missing = math.isnan(ed)

                # Residual bar.
                if not rd_missing:
                    ax.bar(
                        x[i] - bar_width / 2, rd,
                        width=bar_width, color=RESIDUAL_COLOR,
                        edgecolor="black", linewidth=0.4,
                    )
                    ax.text(
                        x[i] - bar_width / 2, rd + (0.04 if rd >= 0 else -0.04),
                        f"{rd:.2f}",
                        ha="center", va="bottom" if rd >= 0 else "top",
                        fontsize=7,
                    )
                else:
                    # Hatched empty box to flag missing.
                    ax.bar(
                        x[i] - bar_width / 2, 0.05,
                        width=bar_width, color="white",
                        edgecolor=RESIDUAL_COLOR, linewidth=0.6,
                        hatch=MISSING_HATCH,
                    )

                # Encoder bar.
                if not ed_missing:
                    ax.bar(
                        x[i] + bar_width / 2, ed,
                        width=bar_width, color=ENCODER_COLOR,
                        edgecolor="black", linewidth=0.4,
                    )
                    ax.text(
                        x[i] + bar_width / 2, ed + (0.04 if ed >= 0 else -0.04),
                        f"{ed:.2f}",
                        ha="center", va="bottom" if ed >= 0 else "top",
                        fontsize=7,
                    )
                else:
                    ax.bar(
                        x[i] + bar_width / 2, 0.05,
                        width=bar_width, color="white",
                        edgecolor=ENCODER_COLOR, linewidth=0.6,
                        hatch=MISSING_HATCH,
                    )

            ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(sysprompts, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(-y_lim, y_lim)
            ax.tick_params(axis="y", labelsize=8)

            if c == 0:
                ax.set_ylabel(f"{turn}, {model_full}\nCohen's d", fontsize=9)
            if r == 0:
                ax.set_title(domain, fontsize=11)

            if bundle["flipped"]:
                ax.text(
                    0.02, 0.95, "[residual sign flipped]",
                    transform=ax.transAxes, fontsize=7, color="darkred",
                    va="top", ha="left",
                )

    # Legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=RESIDUAL_COLOR, label="residual probe"),
        plt.Rectangle((0, 0), 1, 1, color=ENCODER_COLOR, label="encoder baseline (Qwen3-Embedding-8B)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="grey", hatch=MISSING_HATCH, label="missing"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PATH}")
    if flips:
        print("\nResidual sign-flips applied (residual's neutral d had opposite sign to encoder's):")
        for turn, model, domain, r_neutral, e_neutral in flips:
            print(f"  {turn}/{model}/{domain}: r_neutral={r_neutral:+.3f}, e_neutral={e_neutral:+.3f}")
    if weird:
        print("\nNon-neutral sysprompts where residual and encoder strongly disagree (sign mismatch, |d|>0.3):")
        for w in weird:
            print(w)

    # Print per-cell summary.
    print("\nPer-cell d values:")
    for (turn, model, domain), bundle in all_data.items():
        flipped_tag = " [FLIPPED]" if bundle["flipped"] else ""
        print(f"  {turn}/{model}/{domain}{flipped_tag}")
        for sp in bundle["sysprompts"]:
            rd = bundle["residual"][sp]
            ed = bundle["encoder"][sp]
            rd_s = f"{rd:+.2f}" if not math.isnan(rd) else "  nan"
            ed_s = f"{ed:+.2f}" if not math.isnan(ed) else "  nan"
            print(f"    {sp:<20s}  residual={rd_s}  encoder={ed_s}")


if __name__ == "__main__":
    main()
