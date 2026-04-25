"""Persona modulation grid: signed Cohen's d for every (sysprompt, probe, turn).

One panel per domain (truth, harm, politics) stacked vertically.
- Rows = sysprompts, sorted by descending d at qwen_tb-1_L38 (assistant turn).
- Columns = (probe, turn) pairs, paired by probe.
  - Truth/harm: 12 columns (6 probes x {user, asst}).
  - Politics: 6 columns (asst only).
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
INDUCED_CSV = EXP_DIR / "induced_shift_table.csv"
DATE = "042526"

QWEN_PROBES = [
    "qwen_tb-1_L33",
    "qwen_tb-1_L38",
    "qwen_tb-1_L43",
    "qwen_tb-4_L33",
    "qwen_tb-4_L38",
    "qwen_tb-4_L43",
]
PROBE_LABELS = ["tb-1 L33", "tb-1 L38", "tb-1 L43", "tb-4 L33", "tb-4 L38", "tb-4 L43"]

DOMAIN_TITLES = {
    "truth": "Truth (true vs false) — d by sysprompt × (probe, turn)",
    "harm": "Harm (harmful vs benign) — d by sysprompt × (probe, turn)",
    "politics": "Politics (left vs right, assistant turn) — d by sysprompt × probe",
}

SORT_PROBE = "qwen_tb-1_L38"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def build_grid(rows, domain, turns, sysprompts, probes):
    """grid[i, j] is signed d for sysprompts[i] at column j.

    Columns are interleaved by probe: for each probe, all turns in order.
    """
    lookup = {}
    for r in rows:
        if r["domain"] != domain:
            continue
        key = (r["system_prompt"], r["probe"], r["turn"])
        lookup[key] = float(r["d"])

    n_cols = len(probes) * len(turns)
    grid = np.full((len(sysprompts), n_cols), np.nan)
    col_labels = []
    for probe, plabel in zip(probes, PROBE_LABELS):
        for turn in turns:
            col_labels.append(f"{plabel}\n{turn}")
    for i, sp in enumerate(sysprompts):
        j = 0
        for probe in probes:
            for turn in turns:
                key = (sp, probe, turn)
                if key in lookup:
                    grid[i, j] = lookup[key]
                j += 1
    return grid, col_labels


def sort_sysprompts(rows, domain, sort_turn):
    """Return sysprompts sorted by descending d at SORT_PROBE on sort_turn."""
    by_sp = {}
    for r in rows:
        if (r["domain"] == domain and r["probe"] == SORT_PROBE
                and r["turn"] == sort_turn):
            by_sp[r["system_prompt"]] = float(r["d"])
    return sorted(by_sp, key=lambda sp: by_sp[sp], reverse=True)


def render_panel(ax, grid, row_labels, col_labels, title, vlim, probe_pair_size):
    im = ax.imshow(grid, cmap="RdBu_r", aspect="auto",
                   vmin=-vlim, vmax=vlim)
    n_rows, n_cols = grid.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = grid[i, j]
            if np.isnan(v):
                continue
            # Choose contrast for cell text.
            text_color = "black" if abs(v) < vlim * 0.55 else "white"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=11)

    # Vertical separators between probe groups (every probe_pair_size cols).
    if probe_pair_size > 1:
        for k in range(probe_pair_size, n_cols, probe_pair_size):
            ax.axvline(k - 0.5, color="white", linewidth=1.5)
    return im


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_csv(INDUCED_CSV)

    truth_sps = sort_sysprompts(rows, "truth", "assistant")
    harm_sps = sort_sysprompts(rows, "harm", "assistant")
    politics_sps = sort_sysprompts(rows, "politics", "assistant")

    truth_grid, truth_cols = build_grid(
        rows, "truth", ["user", "assistant"], truth_sps, QWEN_PROBES,
    )
    harm_grid, harm_cols = build_grid(
        rows, "harm", ["user", "assistant"], harm_sps, QWEN_PROBES,
    )
    politics_grid, politics_cols = build_grid(
        rows, "politics", ["assistant"], politics_sps, QWEN_PROBES,
    )

    all_vals = np.concatenate([
        truth_grid[~np.isnan(truth_grid)],
        harm_grid[~np.isnan(harm_grid)],
        politics_grid[~np.isnan(politics_grid)],
    ])
    extreme = float(np.nanmax(np.abs(all_vals)))
    vlim = max(3.0, np.ceil(extreme * 10) / 10)

    height_ratios = [len(truth_sps), len(harm_sps), len(politics_sps)]
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 13),
        gridspec_kw={"height_ratios": height_ratios},
    )

    im0 = render_panel(axes[0], truth_grid, truth_sps, truth_cols,
                       DOMAIN_TITLES["truth"], vlim, probe_pair_size=2)
    im1 = render_panel(axes[1], harm_grid, harm_sps, harm_cols,
                       DOMAIN_TITLES["harm"], vlim, probe_pair_size=2)
    im2 = render_panel(axes[2], politics_grid, politics_sps, politics_cols,
                       DOMAIN_TITLES["politics"], vlim, probe_pair_size=1)

    # Single colorbar shared across panels.
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Signed Cohen's d")

    fig.suptitle(
        "Qwen-3.5-122B: persona modulation grid (signed d, sorted by d at "
        "tb-1 L38 assistant turn)",
        fontsize=12, y=0.995,
    )

    out_path = ASSETS_DIR / f"plot_{DATE}_qwen_persona_modulation_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")

    def report_extremes(domain_label, grid, sps):
        # We summarise the most extreme +d and -d sysprompt at SORT_PROBE/asst
        # (which is the sort key, so this is just the head/tail).
        sort_col_idx = (QWEN_PROBES.index(SORT_PROBE) * 2 + 1
                        if grid.shape[1] == len(QWEN_PROBES) * 2
                        else QWEN_PROBES.index(SORT_PROBE))
        col = grid[:, sort_col_idx]
        valid = ~np.isnan(col)
        if not valid.any():
            print(f"  {domain_label}: no valid d values")
            return
        idx_pos = int(np.nanargmax(col))
        idx_neg = int(np.nanargmin(col))
        print(f"  {domain_label}: most +d = {sps[idx_pos]} "
              f"({col[idx_pos]:+.2f}); most -d = {sps[idx_neg]} "
              f"({col[idx_neg]:+.2f})  [at {SORT_PROBE}, asst]")

    report_extremes("truth", truth_grid, truth_sps)
    report_extremes("harm", harm_grid, harm_sps)
    report_extremes("politics", politics_grid, politics_sps)


if __name__ == "__main__":
    main()
