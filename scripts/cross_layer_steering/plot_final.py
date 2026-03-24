"""P(completed steered task) sigmoid: unfolded dose-response by pair type.

Task A always receives +direction x c (signed coefficient).
P(completed steered task) = P(chose A | c), giving a sigmoid:
  c > 0: A boosted → P ~ 1.0
  c = 0: no steering → P ~ 0.5
  c < 0: A anti-steered → P ~ 0.0
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()


def load_parsed(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "task_completed" in r:
                    rows.append(r)
    return rows


def compute_p_steered_per_pair(rows: list[dict], field: str) -> dict[float, float]:
    """P(completed steered task) per pair averaged across orderings, then across pairs.

    For each (coef, pair_id, ordering): P(field == "a") among ALL completions.
    Then average across orderings per pair, then across pairs.
    """
    by_key: dict[tuple[float, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["signed_multiplier"], r["pair_id"], r["ordering"])].append(r)

    by_coef_pair: dict[tuple[float, str], list[float]] = defaultdict(list)
    for (coef, pair_id, ordering), rr in by_key.items():
        p_a = np.mean([1 if r[field] == "a" else 0 for r in rr])
        by_coef_pair[(coef, pair_id)].append(p_a)

    p_by_coef: dict[float, list[float]] = defaultdict(list)
    for (coef, pair_id), ordering_vals in by_coef_pair.items():
        p_by_coef[coef].append(np.mean(ordering_vals))

    return {coef: float(np.mean(vals)) for coef, vals in p_by_coef.items()}


def compute_p_steered_valid_only(rows: list[dict], field: str) -> dict[float, float]:
    """Same as compute_p_steered_per_pair but denominator = valid choices only (exclude neither)."""
    by_key: dict[tuple[float, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r[field] in ("a", "b"):
            by_key[(r["signed_multiplier"], r["pair_id"], r["ordering"])].append(r)

    by_coef_pair: dict[tuple[float, str], list[float]] = defaultdict(list)
    for (coef, pair_id, ordering), rr in by_key.items():
        p_a = np.mean([1 if r[field] == "a" else 0 for r in rr])
        by_coef_pair[(coef, pair_id)].append(p_a)

    p_by_coef: dict[float, list[float]] = defaultdict(list)
    for (coef, pair_id), ordering_vals in by_coef_pair.items():
        p_by_coef[coef].append(np.mean(ordering_vals))

    return {coef: float(np.mean(vals)) for coef, vals in p_by_coef.items()}


def compute_neither_rate(rows: list[dict]) -> dict[float, float]:
    by_coef: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        by_coef[r["signed_multiplier"]].append(r.get("task_completed") == "neither")
    return {coef: float(np.mean(vals)) for coef, vals in by_coef.items()}


def plot_sigmoid(
    groups: dict[str, dict[float, float]],
    neither_groups: dict[str, dict[float, float]],
    title: str,
    output_path: Path,
):
    colors = {"Benign": "tab:blue", "Harmful-Benign": "tab:orange", "Harmful-Harmful": "tab:red"}
    markers = {"Benign": "o", "Harmful-Benign": "s", "Harmful-Harmful": "D"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data in groups.items():
        coefs = sorted(data.keys())
        vals = [data[c] for c in coefs]
        ax.plot(coefs, vals, f"{markers.get(name, 'o')}-", color=colors.get(name, "gray"),
                linewidth=2, markersize=6, label=name)

    for name, ndata in neither_groups.items():
        coefs = sorted(ndata.keys())
        vals = [ndata[c] for c in coefs]
        ax.fill_between(coefs, 0, vals, alpha=0.08, color=colors.get(name, "gray"),
                         label=f"Neither ({name})")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Coefficient (\u00d7 mean norm)\nPositive = toward A, Negative = away from A")
    ax.set_ylabel("P(completed steered task)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    benign_rows = [r for r in load_parsed(Path("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl"))
                   if r["layer"] == 25]

    harmful_parsed = load_parsed(Path("experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl"))
    with open("experiments/steering/cross_layer_harmful/pairs_200.json") as f:
        harmful_pairs = {p["pair_id"]: p for p in json.load(f)}

    harmful_l25 = [r for r in harmful_parsed if r["condition"] == "probe_L25" and r["layer"] == 25]
    hb_rows = [r for r in harmful_l25 if harmful_pairs[r["pair_id"]]["pair_type"] == "harmful_benign"]
    hh_rows = [r for r in harmful_l25 if harmful_pairs[r["pair_id"]]["pair_type"] == "harmful_harmful"]

    assets = Path("experiments/steering/cross_layer_harmful/assets")

    for field, label, suffix in [
        ("task_completed", "judge content", "content"),
        ("choice_original", "regex label", "label"),
    ]:
        groups = {
            "Benign": compute_p_steered_per_pair(benign_rows, field),
            "Harmful-Benign": compute_p_steered_per_pair(hb_rows, field),
            "Harmful-Harmful": compute_p_steered_per_pair(hh_rows, field),
        }
        neither = {
            "Benign": compute_neither_rate(benign_rows),
            "Harmful-Benign": compute_neither_rate(hb_rows),
            "Harmful-Harmful": compute_neither_rate(hh_rows),
        }

        plot_sigmoid(
            groups, neither,
            title=f"P(completed steered task) \u2014 {label}\n(probe L25, layer 25)",
            output_path=assets / f"plot_032426_p_steered_{suffix}.png",
        )

        # Print table for report
        print(f"\n{'coef':>7}  {'Benign':>8}  {'H-B':>8}  {'H-H':>8}  | {'neither_B':>9}  {'neither_HB':>10}  {'neither_HH':>10}")
        all_coefs = sorted(set(list(groups["Benign"].keys()) + list(groups["Harmful-Benign"].keys())))
        for c in all_coefs:
            b = groups["Benign"].get(c, float("nan"))
            hb = groups["Harmful-Benign"].get(c, float("nan"))
            hh = groups["Harmful-Harmful"].get(c, float("nan"))
            nb = neither["Benign"].get(c, float("nan"))
            nhb = neither["Harmful-Benign"].get(c, float("nan"))
            nhh = neither["Harmful-Harmful"].get(c, float("nan"))
            print(f"{c:>7.3f}  {b:>8.3f}  {hb:>8.3f}  {hh:>8.3f}  | {nb:>9.3f}  {nhb:>10.3f}  {nhh:>10.3f}")


    # Valid-only denominator version (content only)
    groups_valid = {
        "Benign": compute_p_steered_valid_only(benign_rows, "task_completed"),
        "Harmful-Benign": compute_p_steered_valid_only(hb_rows, "task_completed"),
        "Harmful-Harmful": compute_p_steered_valid_only(hh_rows, "task_completed"),
    }

    plot_sigmoid(
        groups_valid, {},
        title="P(completed steered task | task completed) \u2014 judge content\n(probe L25, layer 25, excluding neither)",
        output_path=assets / "plot_032426_p_steered_valid_only.png",
    )

    print(f"\nValid-only denominator (content):")
    print(f"{'coef':>7}  {'Benign':>8}  {'H-B':>8}  {'H-H':>8}")
    all_coefs = sorted(set(list(groups_valid["Benign"].keys()) + list(groups_valid["Harmful-Benign"].keys())))
    for c in all_coefs:
        b = groups_valid["Benign"].get(c, float("nan"))
        hb = groups_valid["Harmful-Benign"].get(c, float("nan"))
        hh = groups_valid["Harmful-Harmful"].get(c, float("nan"))
        print(f"{c:>7.3f}  {b:>8.3f}  {hb:>8.3f}  {hh:>8.3f}")


if __name__ == "__main__":
    main()
