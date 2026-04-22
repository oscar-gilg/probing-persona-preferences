"""Poster steering plot: valid-only sigmoid at L25 and L30 with aggregate neither shading."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()

ASSETS = Path("docs/poster/assets")
ASSETS.mkdir(parents=True, exist_ok=True)


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


def compute_valid_sigmoid(rows: list[dict]) -> dict[float, float]:
    """P(chose A | task completed) per pair, averaged across orderings then pairs."""
    by_key: dict[tuple[float, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r["task_completed"] in ("a", "b"):
            by_key[(r["signed_multiplier"], r["pair_id"], r["ordering"])].append(r)

    by_coef_pair: dict[tuple[float, str], list[float]] = defaultdict(list)
    for (coef, pair_id, ordering), rr in by_key.items():
        p_a = np.mean([1 if r["task_completed"] == "a" else 0 for r in rr])
        by_coef_pair[(coef, pair_id)].append(p_a)

    p_by_coef: dict[float, list[float]] = defaultdict(list)
    for (coef, pair_id), vals in by_coef_pair.items():
        p_by_coef[coef].append(np.mean(vals))

    return {c: float(np.mean(v)) for c, v in p_by_coef.items()}


def compute_failure_rates(rows: list[dict]) -> dict[float, dict[str, float]]:
    by_coef: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_coef[r["signed_multiplier"]].append(r)
    result = {}
    for c, rr in by_coef.items():
        n = len(rr)
        result[c] = {
            "refusal": sum(1 for r in rr if r.get("compliance") == "hard_refusal") / n,
            "incoherent": sum(1 for r in rr if r.get("compliance") in ("incoherent", "error")) / n,
        }
    return result


# Load data
harmful_parsed = load_parsed(Path("experiments/steering/cross_layer_harmful/checkpoint.parsed.jsonl"))
with open("experiments/steering/cross_layer_harmful/pairs_200.json") as f:
    harmful_pairs = {p["pair_id"]: p for p in json.load(f)}

benign_parsed = load_parsed(Path("experiments/steering/cross_layer/checkpoint_L25.parsed.jsonl"))

colors = {"Benign": "#2563eb", "Harmful-Benign": "#f97316", "Harmful-Harmful": "#dc2626"}
markers = {"Benign": "o", "Harmful-Benign": "s", "Harmful-Harmful": "D"}

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

for col, layer in enumerate([25, 30]):
    ax = axes[col]

    # Collect ALL rows at this layer for aggregate neither
    all_layer_rows = []

    for name, filter_fn in [
        ("Benign", lambda r: True),
        ("Harmful-Benign", lambda r: harmful_pairs.get(r["pair_id"], {}).get("pair_type") == "harmful_benign"),
        ("Harmful-Harmful", lambda r: harmful_pairs.get(r["pair_id"], {}).get("pair_type") == "harmful_harmful"),
    ]:
        if name == "Benign":
            layer_rows = [r for r in benign_parsed if r["layer"] == layer]
        else:
            layer_rows = [r for r in harmful_parsed
                          if r["condition"] == "probe_L25" and r["layer"] == layer and filter_fn(r)]

        all_layer_rows.extend(layer_rows)

        data = compute_valid_sigmoid(layer_rows)
        coefs = sorted(data.keys())
        vals = [data[c] for c in coefs]
        ax.plot(coefs, vals, f"{markers[name]}-", color=colors[name],
                linewidth=2, markersize=5, label=name)

    # Aggregate failure rates (all pair types combined)
    failure = compute_failure_rates(all_layer_rows)
    f_coefs = sorted(failure.keys())
    incoherent_vals = [failure[c]["incoherent"] for c in f_coefs]
    refusal_vals = [failure[c]["refusal"] for c in f_coefs]
    total_vals = [i + r for i, r in zip(incoherent_vals, refusal_vals)]
    ax.fill_between(f_coefs, 0, incoherent_vals, alpha=0.12, color="#6B7280", label="Incoherent")
    ax.fill_between(f_coefs, incoherent_vals, total_vals, alpha=0.20, color="#ef4444", label="Refusal")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.12, 0.12)
    ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient (\u00d7 mean norm)", fontsize=9)
    if col == 0:
        ax.set_ylabel("P(completed steered task\n| task completed)", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")

plt.tight_layout()
out = ASSETS / "plot_032426_poster_steering.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
