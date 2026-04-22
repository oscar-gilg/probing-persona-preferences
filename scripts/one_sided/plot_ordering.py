"""Plot ordering interaction for one-sided steering decomposition.

3 columns (steer_first, steer_second, differential) x 2 rows (harmful, benign).
Each panel: P(steered) vs signed_multiplier, split by ordering (0 vs 1).
"""

from collections import defaultdict

import matplotlib.pyplot as plt

from src.steering.analysis import compute_p_steered, load_checkpoint

HARMFUL_PATH = "experiments/steering/one_sided/checkpoint_harmful.parsed.jsonl"
BENIGN_PATH = "experiments/steering/one_sided/checkpoint_benign.parsed.jsonl"
OUT_PATH = "experiments/steering/one_sided/assets/plot_032526_ordering_interaction.png"

CONDITIONS = ["steer_first_L25", "steer_second_L25", "differential_L25"]
CONDITION_LABELS = {
    "steer_first_L25": "steer_first",
    "steer_second_L25": "steer_second",
    "differential_L25": "differential",
}


def build_lookup(results: list[dict]) -> dict[tuple, dict]:
    """Index results by (condition, ordering, signed_multiplier)."""
    lookup: dict[tuple, dict] = {}
    for r in results:
        key = (r["condition"], r["ordering"], r["signed_multiplier"])
        lookup[key] = r
    return lookup


def main() -> None:
    harmful_rows = load_checkpoint(HARMFUL_PATH)
    benign_rows = load_checkpoint(BENIGN_PATH)

    harmful_results = compute_p_steered(
        harmful_rows, group_by=["condition", "ordering"], choice_field="task_completed"
    )
    benign_results = compute_p_steered(
        benign_rows, group_by=["condition", "ordering"], choice_field="task_completed"
    )

    harmful_lookup = build_lookup(harmful_results)
    benign_lookup = build_lookup(benign_results)

    # Get sorted multipliers from results
    multipliers = sorted(set(r["signed_multiplier"] for r in harmful_results))

    datasets = [
        ("Harmful", harmful_lookup),
        ("Benign", benign_lookup),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)

    for row_idx, (row_label, lookup) in enumerate(datasets):
        for col_idx, condition in enumerate(CONDITIONS):
            ax = axes[row_idx, col_idx]

            for ordering, color, ls, label in [
                (0, "tab:blue", "-", "ordering=0"),
                (1, "tab:orange", "--", "ordering=1"),
            ]:
                xs = multipliers
                ys = [lookup[(condition, ordering, m)]["p_steered"] for m in multipliers]
                ax.plot(xs, ys, color=color, ls=ls, marker="o", markersize=4, label=label)

            ax.axhline(0.5, color="grey", ls=":", lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_xlabel("signed_multiplier")

            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nP(steered)")

            if row_idx == 0:
                ax.set_title(CONDITION_LABELS[condition])

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
