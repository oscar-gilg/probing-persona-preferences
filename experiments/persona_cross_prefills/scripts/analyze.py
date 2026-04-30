"""Analyze persona-cross prefill scoring results.

Loads results/scoring_results.json. Produces:
  1. Headline cell-mean table (4 conditions × 2 personas × 2 readouts) → CSV + bar plot
  2. Within-pair Δ_pair (misaligned − aligned, fixed user message) → strip plot
  3. Per-persona Δ_persona (sadist − default, fixed prefill) → strip plot — main hypothesis test
  4. asst-EOT vs user-EOT scatter
  5. Persona-baseline shift (default vs sadist at first_user_eot)
  6. Pair × persona heatmap
  7. analysis_summary.json with stats (paired t-test on harmful-axis interaction)

Usage: python -m experiments.persona_cross_prefills.scripts.analyze
"""

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/persona_cross_prefills"
RESULTS = EXP / "results"
ASSETS = EXP / "assets"
TODAY = date.today().strftime("%m%d%y")

CONDITION_ORDER = ["benign_helpful", "benign_evil", "harmful_refused", "harmful_obliged"]
CONDITION_COLORS = {
    "benign_helpful":   "#1f77b4",
    "benign_evil":      "#d62728",
    "harmful_refused":  "#2ca02c",
    "harmful_obliged":  "#ff7f0e",
}
PERSONAS = ["default", "sadist"]
READOUTS = ["first_user_eot", "asst_eot", "user_eot"]
ALIGNED_FOR_DEFAULT = {"benign_helpful", "harmful_refused"}


def load_results() -> pd.DataFrame:
    data = json.loads((RESULTS / "scoring_results.json").read_text())
    rows = []
    for item in data["items"]:
        rows.append({
            "prefill_id": item["prefill_id"],
            "persona": item["persona_name"],
            "pair_id": item["pair_id"],
            "condition": item["condition"],
            "topic": item["topic"],
            "asst_eot": item["eot_scores"]["asst_eot"],
            "user_eot": item["eot_scores"]["user_eot"],
            "first_user_eot": item["eot_scores"]["first_user_eot"],
        })
    return pd.DataFrame(rows)


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        for persona in PERSONAS:
            for readout in READOUTS:
                sub = df[(df["condition"] == cond) & (df["persona"] == persona)][readout]
                rows.append({
                    "condition": cond,
                    "persona": persona,
                    "readout": readout,
                    "mean": float(sub.mean()),
                    "se": float(sub.std(ddof=1) / np.sqrt(len(sub))),
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


def plot_cell_means(cm: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(6 * len(READOUTS), 4.5), sharey=True)
    for ax, readout in zip(axes, READOUTS):
        sub = cm[cm["readout"] == readout]
        x_positions = np.arange(len(CONDITION_ORDER))
        width = 0.4
        for i, persona in enumerate(PERSONAS):
            persona_sub = sub[sub["persona"] == persona].set_index("condition").loc[CONDITION_ORDER]
            offset = (i - 0.5) * width
            ax.bar(x_positions + offset, persona_sub["mean"], width,
                   yerr=persona_sub["se"], label=persona,
                   color=["#bbbbbb", "#8B4513"][i], capsize=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([c.replace("_", "\n") for c in CONDITION_ORDER], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(readout)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("probe score (tb-5_L32, raw)")
    axes[1].legend(title="persona", loc="best")
    fig.suptitle("Does the probe rate the same asst behavior differently under sadist? (cell means, n=10)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def within_pair_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Δ_pair = misaligned − aligned per (pair_id, persona). Negative aligned ↔ benign_evil/harmful_obliged."""
    rows = []
    for (pair_id, persona), group in df.groupby(["pair_id", "persona"]):
        if len(group) != 2:
            continue
        aligned = group[group["condition"].isin(ALIGNED_FOR_DEFAULT)].iloc[0]
        misaligned = group[~group["condition"].isin(ALIGNED_FOR_DEFAULT)].iloc[0]
        rows.append({
            "pair_id": pair_id,
            "persona": persona,
            "valence": "benign" if pair_id.startswith("benign") else "harmful",
            "asst_eot_delta": misaligned["asst_eot"] - aligned["asst_eot"],
            "user_eot_delta": misaligned["user_eot"] - aligned["user_eot"],
        })
    return pd.DataFrame(rows)


def plot_within_pair(deltas: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    readouts = [("asst_eot_delta", "post-asst-response"), ("user_eot_delta", "post-user-follow-up")]
    for ax, (readout, label) in zip(axes, readouts):
        x_positions = []
        x_labels = []
        for i, valence in enumerate(["benign", "harmful"]):
            for j, persona in enumerate(PERSONAS):
                sub = deltas[(deltas["valence"] == valence) & (deltas["persona"] == persona)][readout]
                pos = i * 2 + j * 0.4
                x_positions.append(pos)
                x_labels.append(f"{valence}\n{persona}")
                ax.scatter(np.full(len(sub), pos) + np.random.uniform(-0.07, 0.07, len(sub)),
                           sub, color=["#bbbbbb", "#8B4513"][j], s=40, alpha=0.7)
                ax.errorbar(pos, sub.mean(), yerr=sub.std(ddof=1) / np.sqrt(len(sub)),
                            fmt="_", color="black", capsize=4, markersize=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"readout: {label}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score(misaligned) − score(aligned), per pair")
    fig.suptitle("How big is the aligned-vs-misaligned asst-behavior gap, and does it flip under sadist?")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def per_persona_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Δ_persona = sadist − default per prefill. Tests persona-relative alignment hypothesis."""
    rows = []
    for prefill_id, group in df.groupby("prefill_id"):
        if len(group) != 2:
            continue
        default_row = group[group["persona"] == "default"].iloc[0]
        sadist_row = group[group["persona"] == "sadist"].iloc[0]
        rows.append({
            "prefill_id": prefill_id,
            "pair_id": default_row["pair_id"],
            "condition": default_row["condition"],
            "asst_eot_delta": sadist_row["asst_eot"] - default_row["asst_eot"],
            "user_eot_delta": sadist_row["user_eot"] - default_row["user_eot"],
            "first_user_eot_delta": sadist_row["first_user_eot"] - default_row["first_user_eot"],
        })
    return pd.DataFrame(rows)


def plot_per_persona(pp: pd.DataFrame, out: Path) -> None:
    readouts = [
        ("first_user_eot_delta", "post-first-user-msg"),
        ("asst_eot_delta", "post-asst-response"),
        ("user_eot_delta", "post-user-follow-up"),
    ]
    fig, axes = plt.subplots(1, len(readouts), figsize=(6.5 * len(readouts), 5), sharey=True)
    for ax, (key, label) in zip(axes, readouts):
        for i, cond in enumerate(CONDITION_ORDER):
            sub = pp[pp["condition"] == cond][key]
            ax.scatter(np.full(len(sub), i) + np.random.uniform(-0.1, 0.1, len(sub)),
                       sub, color=CONDITION_COLORS[cond], s=50, alpha=0.7)
            ax.errorbar(i, sub.mean(), yerr=sub.std(ddof=1) / np.sqrt(len(sub)),
                        fmt="_", color="black", capsize=5, markersize=18)
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels([c.replace("_", "\n") for c in CONDITION_ORDER], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"readout: {label}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score(sadist) − score(default), per prefill")
    fig.suptitle("How does swapping in the sadist system prompt change the probe's verdict?  (n=10/cond)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_asst_vs_user_eot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, persona in zip(axes, PERSONAS):
        sub = df[df["persona"] == persona]
        for cond in CONDITION_ORDER:
            cs = sub[sub["condition"] == cond]
            ax.scatter(cs["asst_eot"], cs["user_eot"], color=CONDITION_COLORS[cond],
                       s=60, alpha=0.7, label=cond, edgecolor="black", linewidth=0.5)
        lim = [
            min(df["asst_eot"].min(), df["user_eot"].min()),
            max(df["asst_eot"].max(), df["user_eot"].max()),
        ]
        ax.plot(lim, lim, "k--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("asst-EOT score")
        ax.set_ylabel("user-EOT score")
        ax.set_title(f"persona = {persona}")
        ax.grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Does the user follow-up dominate the readout? (off-diagonal = it does)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_persona_baseline(df: pd.DataFrame, out: Path) -> None:
    """At first_user_eot — does the system prompt itself shift the baseline (before any asst content)?

    Split by valence (benign vs harmful first-user-message) to expose how the persona changes
    the user-content signal.
    """
    df = df.copy()
    df["valence"] = df["pair_id"].apply(lambda p: "benign" if p.startswith("benign") else "harmful")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = []
    labels = []
    for i, valence in enumerate(["benign", "harmful"]):
        for j, persona in enumerate(PERSONAS):
            sub = df[(df["valence"] == valence) & (df["persona"] == persona)]["first_user_eot"]
            pos = i * 2 + j * 0.5
            positions.append(pos)
            labels.append(f"{valence}\n{persona}")
            ax.scatter(np.full(len(sub), pos) + np.random.uniform(-0.07, 0.07, len(sub)),
                       sub, color=["#bbbbbb", "#8B4513"][j], s=40, alpha=0.6)
            ax.errorbar(pos, sub.mean(), yerr=sub.std(ddof=1) / np.sqrt(len(sub)),
                        fmt="_", color="black", capsize=5, markersize=18)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("probe score @ first user-EOT (post system + first user msg)")
    ax.set_title("Does the persona alone shift how the probe reads the user's request? (pre-asst-content)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_pair_persona_heatmap(df: pd.DataFrame, out: Path, readout: str = "asst_eot") -> None:
    """Rows = 20 pairs, cols = 4 (aligned vs misaligned × default vs sadist).

    Re-uses the within-pair "aligned vs misaligned" axis: for benign pairs aligned=helpful,
    misaligned=evil; for harmful aligned=refused, misaligned=obliged.
    """
    rows = []
    for pair_id, pair_group in df.groupby("pair_id"):
        for persona in PERSONAS:
            sub = pair_group[pair_group["persona"] == persona]
            aligned = sub[sub["condition"].isin(ALIGNED_FOR_DEFAULT)][readout].iloc[0]
            misaligned = sub[~sub["condition"].isin(ALIGNED_FOR_DEFAULT)][readout].iloc[0]
            rows.append({"pair_id": pair_id, "persona": persona,
                         "aligned": aligned, "misaligned": misaligned})
    pdf = pd.DataFrame(rows)
    grid = pdf.set_index(["pair_id", "persona"])[["aligned", "misaligned"]].unstack("persona")
    cols = [("aligned", "default"), ("aligned", "sadist"),
            ("misaligned", "default"), ("misaligned", "sadist")]
    grid = grid[cols]
    pair_order = (sorted([p for p in grid.index if p.startswith("benign")]) +
                  sorted([p for p in grid.index if p.startswith("harmful")]))
    grid = grid.loc[pair_order]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    abs_max = float(np.nanmax(np.abs(grid.values)))
    im = ax.imshow(grid.values, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    ax.set_yticks(range(len(pair_order)))
    ax.set_yticklabels(pair_order, fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{a}\n{p}" for a, p in cols], fontsize=8)
    ax.axhline(9.5, color="black", linewidth=1.2)
    ax.set_title(f"{readout} probe score — pairs × (asst-behavior × persona)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="probe score")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def compute_stats(df: pd.DataFrame, deltas: pd.DataFrame, pp: pd.DataFrame) -> dict:
    """Headline stat: paired t-test on harmful-axis Δ_pair under default and sadist."""
    out = {}
    for persona in PERSONAS:
        for readout in ["asst_eot_delta", "user_eot_delta"]:
            sub = deltas[(deltas["persona"] == persona) & (deltas["valence"] == "harmful")][readout]
            t, p = stats.ttest_1samp(sub, 0.0)
            out[f"harmful_{readout}_{persona}"] = {
                "mean": float(sub.mean()),
                "se": float(sub.std(ddof=1) / np.sqrt(len(sub))),
                "t": float(t),
                "p": float(p),
                "n": len(sub),
            }
    # Per-persona shift on harmful_obliged − harmful_refused
    for readout in ["asst_eot", "user_eot"]:
        ho = df[df["condition"] == "harmful_obliged"].set_index(["pair_id", "persona"])[readout]
        hr = df[df["condition"] == "harmful_refused"].set_index(["pair_id", "persona"])[readout]
        for persona in PERSONAS:
            try:
                vals = ho.xs(persona, level="persona") - hr.xs(persona, level="persona")
                out[f"obliged_minus_refused_{readout}_{persona}"] = {
                    "mean": float(vals.mean()),
                    "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
                    "n": len(vals),
                }
            except KeyError:
                pass
    # Per-persona Δ summary
    for cond in CONDITION_ORDER:
        for readout in ["first_user_eot_delta", "asst_eot_delta", "user_eot_delta"]:
            sub = pp[pp["condition"] == cond][readout]
            t, p = stats.ttest_1samp(sub, 0.0)
            out[f"persona_delta_{cond}_{readout}"] = {
                "mean": float(sub.mean()),
                "se": float(sub.std(ddof=1) / np.sqrt(len(sub))),
                "t": float(t),
                "p": float(p),
                "n": len(sub),
            }
    return out


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("Loading scoring results...")
    df = load_results()
    print(f"  {len(df)} rows ({df['prefill_id'].nunique()} prefills × {df['persona'].nunique()} personas)")

    print("Computing cell means...")
    cm = cell_means(df)
    cm.to_csv(RESULTS / "cell_means.csv", index=False)
    plot_cell_means(cm, ASSETS / f"plot_{TODAY}_cell_means.png")

    print("Computing within-pair Δ...")
    deltas = within_pair_deltas(df)
    deltas.to_csv(RESULTS / "within_pair_deltas.csv", index=False)
    plot_within_pair(deltas, ASSETS / f"plot_{TODAY}_within_pair_delta.png")

    print("Computing per-persona Δ (the headline test)...")
    pp = per_persona_deltas(df)
    pp.to_csv(RESULTS / "per_persona_deltas.csv", index=False)
    plot_per_persona(pp, ASSETS / f"plot_{TODAY}_per_persona_delta.png")

    print("Asst-EOT vs user-EOT scatter...")
    plot_asst_vs_user_eot(df, ASSETS / f"plot_{TODAY}_asst_vs_user_eot.png")

    print("Persona baseline shift...")
    plot_persona_baseline(df, ASSETS / f"plot_{TODAY}_persona_baseline.png")

    print("Pair × persona heatmap (asst-EOT and user-EOT)...")
    plot_pair_persona_heatmap(df, ASSETS / f"plot_{TODAY}_pair_persona_heatmap_asst_eot.png", "asst_eot")
    plot_pair_persona_heatmap(df, ASSETS / f"plot_{TODAY}_pair_persona_heatmap_user_eot.png", "user_eot")

    print("Computing stats...")
    summary = compute_stats(df, deltas, pp)
    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nKEY STATS:")
    for key, val in summary.items():
        if "harmful" in key or "obliged_minus_refused" in key:
            print(f"  {key}: {val}")
    print(f"\nAll outputs in {ASSETS}/ and {RESULTS}/")


if __name__ == "__main__":
    main()
