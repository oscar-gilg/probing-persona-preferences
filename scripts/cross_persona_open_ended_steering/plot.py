"""Generate the four steering experiment plots from aggregated.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/.claude/worktrees/"
    "cross_persona_open_ended_steering/experiments/cross_persona_open_ended_steering"
)
DATA_PATH = EXPERIMENT_DIR / "aggregated.json"
ASSETS_DIR = EXPERIMENT_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

PERSONAS = ["default", "sadist", "mathematician", "aura", "strategist", "contrarian", "slacker"]
NON_DEFAULT_PERSONAS = [p for p in PERSONAS if p != "default"]
COEFS_STR = ["-0.05", "-0.03", "+0.00", "+0.03", "+0.05", "+0.07"]
COEFS = [float(c) for c in COEFS_STR]

plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def extract_curve(d, keys):
    """Walk through nested dict by `keys`, return (means, ses) over COEFS_STR.

    The coefficient key is assumed to be the LAST key (i.e. keys=[...prefix..., coef])
    is NOT what we want. We instead supply prefix `keys` ending at a node whose direct
    children are coef strings.
    """
    means, ses = [], []
    for c in COEFS_STR:
        node = d
        for k in keys:
            node = node[k]
        leaf = node[c]
        means.append(leaf["mean"])
        ses.append(leaf["se"])
    return np.array(means), np.array(ses)


def extract_curve_with_subkey(node, subkey):
    """node[coef][subkey] -> (means, ses) where leaf has mean/se."""
    means, ses = [], []
    for c in COEFS_STR:
        leaf = node[c][subkey]
        means.append(leaf["mean"])
        ses.append(leaf["se"])
    return np.array(means), np.array(ses)


def extract_likert_curve(node, kind):
    """node[coef][kind] = {mean, se, n}."""
    means, ses = [], []
    for c in COEFS_STR:
        leaf = node[c][kind]
        means.append(leaf["mean"])
        ses.append(leaf["se"])
    return np.array(means), np.array(ses)


def extract_strat_curve(node, sub, kind):
    """node[coef][sub][kind] = {mean, se, n}; missing -> NaN."""
    means, ses = [], []
    for c in COEFS_STR:
        if c not in node or sub not in node[c] or kind not in node[c][sub]:
            means.append(np.nan)
            ses.append(np.nan)
            continue
        leaf = node[c][sub][kind]
        means.append(leaf["mean"])
        ses.append(leaf["se"])
    return np.array(means), np.array(ses)


def extract_rating_curve(node):
    """node[coef] = {mean, se, n}; missing coef -> NaN."""
    means, ses = [], []
    for c in COEFS_STR:
        if c not in node:
            means.append(np.nan)
            ses.append(np.nan)
            continue
        leaf = node[c]
        means.append(leaf["mean"])
        ses.append(leaf["se"])
    return np.array(means), np.array(ses)


def style_axes(ax, ylim, ylabel):
    ax.set_xticks(COEFS)
    ax.set_xticklabels(COEFS_STR, rotation=0)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)


def plot_persona_vs_default(data, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, persona in enumerate(PERSONAS):
        ax = axes_flat[i]
        node = data["likert"][persona]
        pf_mean, pf_se = extract_likert_curve(node, "persona_fidelity")
        da_mean, da_se = extract_likert_curve(node, "default_assistant")

        ax.errorbar(COEFS, pf_mean, yerr=pf_se, marker="o", linestyle="-",
                    color="#1f77b4", capsize=2, label="persona fidelity", linewidth=1.5)
        ax.errorbar(COEFS, da_mean, yerr=da_se, marker="s", linestyle="--",
                    color="#d62728", capsize=2, label="default assistant", linewidth=1.5)
        ax.set_title(persona)
        style_axes(ax, (1, 5), "Likert score")

    # last panel: legend
    ax_legend = axes_flat[7]
    ax_legend.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", fontsize=11, frameon=True)

    fig.suptitle("Persona fidelity vs. default-assistant adherence by coefficient", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_opportunity_stratified(data, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    opp_colors = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#9467bd"}
    opp_levels = ["low", "medium", "high"]

    for i, persona in enumerate(PERSONAS):
        ax = axes_flat[i]
        node = data["likert_by_opportunity"][persona]
        for opp in opp_levels:
            means, ses = extract_strat_curve(node, opp, "persona_fidelity")
            ax.errorbar(COEFS, means, yerr=ses, marker="o", linestyle="-",
                        color=opp_colors[opp], capsize=2, label=opp, linewidth=1.5)
        ax.set_title(persona)
        style_axes(ax, (1, 5), "persona fidelity (Likert)")

    ax_legend = axes_flat[7]
    ax_legend.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc="center", title="opportunity", fontsize=11, frameon=True)

    fig.suptitle("Persona fidelity by opportunity level (shared pool)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ratings_specific_scissors(data, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, persona in enumerate(NON_DEFAULT_PERSONAS):
        ax = axes_flat[i]
        node = data["ratings_specific"][persona]
        aligned_mean, aligned_se = extract_rating_curve(node["aligned"])
        repellent_mean, repellent_se = extract_rating_curve(node["repellent"])

        ax.errorbar(COEFS, aligned_mean, yerr=aligned_se, marker="o", linestyle="-",
                    color="#2ca02c", capsize=2, label="aligned (target=10)", linewidth=1.5)
        ax.errorbar(COEFS, repellent_mean, yerr=repellent_se, marker="s", linestyle="--",
                    color="#d62728", capsize=2, label="repellent (target=1)", linewidth=1.5)
        ax.axhline(10, color="#2ca02c", linewidth=0.5, alpha=0.4, linestyle=":")
        ax.axhline(1, color="#d62728", linewidth=0.5, alpha=0.4, linestyle=":")
        ax.set_title(persona)
        style_axes(ax, (1, 10), "rating")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle("Aligned vs. repellent rating ('scissors') by persona", y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ratings_shared_by_axis(data, out_path):
    axes_names = ["analytical_rigor", "altruistic_servility"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    cmap = plt.get_cmap("tab10")
    persona_colors = {p: cmap(i) for i, p in enumerate(PERSONAS)}

    for ax, axis_name in zip(axes, axes_names):
        for persona in PERSONAS:
            node = data["ratings_shared"][persona][axis_name]
            means, ses = extract_rating_curve(node)
            ax.errorbar(COEFS, means, yerr=ses, marker="o", linestyle="-",
                        color=persona_colors[persona], capsize=2, label=persona, linewidth=1.5)
        ax.set_title(axis_name.replace("_", " "))
        style_axes(ax, (1, 10), "rating")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.08, 0.5), frameon=True, title="persona")

    fig.suptitle("Shared-axis ratings across personas", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    plot_persona_vs_default(data, ASSETS_DIR / "plot_042426_persona_vs_default_by_coef.png")
    plot_opportunity_stratified(data, ASSETS_DIR / "plot_042426_opportunity_stratified.png")
    plot_ratings_specific_scissors(data, ASSETS_DIR / "plot_042426_ratings_specific_scissors.png")
    plot_ratings_shared_by_axis(data, ASSETS_DIR / "plot_042426_ratings_shared_by_axis.png")
    print("Wrote 4 plots to", ASSETS_DIR)


if __name__ == "__main__":
    main()
