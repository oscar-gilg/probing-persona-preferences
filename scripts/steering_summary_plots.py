"""Generate clean poster-ready plots for the steering summary report.
Reads from actual experiment data, not hardcoded numbers."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

ASSETS = Path("experiments/steering/steering_assets")
ASSETS.mkdir(parents=True, exist_ok=True)


# ============ Plot 1: Pairwise dose-response (KV with and without recompute) ============

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def compute_p_steered(rows):
    """Group by |multiplier|, compute P(chose steered task). Folds sign."""
    groups = defaultdict(list)
    for r in rows:
        m = r.get("signed_multiplier", r.get("multiplier", 0))
        abs_m = abs(m)
        if abs_m < 0.0001:
            continue  # skip baseline (no steering direction)
        chose_a = r.get("choice_original", r.get("parsed_choice", ""))
        if chose_a in ("a", "A", "task_a"):
            steered = m > 0
        elif chose_a in ("b", "B", "task_b"):
            steered = m < 0
        else:
            continue
        groups[abs_m].append(1.0 if steered else 0.0)

    mults = sorted(groups.keys())
    p_steered = [np.mean(groups[m]) for m in mults]
    n = [len(groups[m]) for m in mults]
    return mults, p_steered, n

# Load KV data
kv_path = Path("experiments/steering/isolated_steering/parsed_kv_full.jsonl")
kv_recompute_path = Path("experiments/steering/isolated_steering/parsed_kv_recompute.jsonl")

if kv_path.exists() and kv_recompute_path.exists():
    kv_rows = load_jsonl(kv_path)
    kv_recompute_rows = load_jsonl(kv_recompute_path)

    mults_kv, p_kv, n_kv = compute_p_steered(kv_rows)
    mults_rc, p_rc, n_rc = compute_p_steered(kv_recompute_rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mults_kv, p_kv, 'o-', color='#6B7280', linewidth=2, markersize=6, label='KV steering only')
    ax.plot(mults_rc, p_rc, 's-', color='#da7756', linewidth=2, markersize=6, label='KV + suffix recompute')
    ax.axhline(0.5, color='#D1D5DB', linestyle='--', linewidth=1)
    ax.set_xlabel('Steering strength (fraction of layer norm)')
    ax.set_ylabel('P(chose steered task)')
    ax.set_ylim(0, 1.05)
    ax.set_title('Preference probe causally controls task choice', fontweight='bold')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(ASSETS / "plot_032326_pairwise_dose_response.png", dpi=200)
    plt.close()
    print(f"Plot 1 saved: pairwise dose response ({len(kv_rows)} + {len(kv_recompute_rows)} rows)")
else:
    print(f"Skipping plot 1: data files not found")
    print(f"  KV: {kv_path} exists={kv_path.exists()}")
    print(f"  Recompute: {kv_recompute_path} exists={kv_recompute_path.exists()}")


# ============ Plot 2: Open-ended engagement (all_tokens mode only) ============

scored_path = Path("experiments/steering/open_ended_steering/scored_results.jsonl")

if scored_path.exists():
    scored_rows = load_jsonl(scored_path)

    # Filter to all_tokens mode
    at_rows = [r for r in scored_rows if r.get("steering_mode") == "all_tokens"]

    groups = defaultdict(list)
    for r in at_rows:
        m = r.get("multiplier", 0)
        score = r.get("engagement_score")
        if score is not None:
            groups[m].append(score)

    mults = sorted(groups.keys())
    means = [np.mean(groups[m]) for m in mults]
    sems = [np.std(groups[m]) / np.sqrt(len(groups[m])) for m in mults]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(mults, means, yerr=sems, fmt='o-', color='#da7756', linewidth=2,
                markersize=6, capsize=4)
    ax.axhline(0, color='#D1D5DB', linestyle='--', linewidth=1)
    ax.axvline(0, color='#D1D5DB', linestyle='--', linewidth=1)
    ax.set_xlabel('Steering multiplier')
    ax.set_ylabel('Mean engagement score')
    ax.set_ylim(-1, 1)
    ax.set_title('Steering modulates open-ended engagement', fontweight='bold')

    # Annotate extremes
    if len(mults) >= 2:
        ax.annotate('safety paranoia', xy=(mults[0], means[0]),
                    xytext=(mults[0] + 0.01, means[0] - 0.15),
                    fontsize=9, color='#6B7280', fontstyle='italic')
        ax.annotate('enthusiastic', xy=(mults[-1], means[-1]),
                    xytext=(mults[-1] - 0.03, means[-1] + 0.1),
                    fontsize=9, color='#6B7280', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(ASSETS / "plot_032326_open_ended_engagement.png", dpi=200)
    plt.close()
    print(f"Plot 2 saved: open-ended engagement ({len(at_rows)} rows)")
else:
    print(f"Skipping plot 2: {scored_path} not found")


# ============ Plot 3: Steerability by topic (simplified from existing) ============

# Use the KV recompute data which has topic info
if kv_recompute_path.exists():
    rows = load_jsonl(kv_recompute_path)

    # Get topic from the pairs data
    pairs_path = Path("experiments/steering/isolated_steering/full_run/pairs_100.json")
    if not pairs_path.exists():
        # Try to get topics from the rows themselves
        topic_groups = defaultdict(list)
        for r in rows:
            topic = r.get("topic", r.get("steered_toward_topic", "unknown"))
            m = r.get("signed_multiplier", r.get("multiplier", 0))
            if abs(m) < 0.001:
                continue  # skip baseline
            chose_a = r.get("choice_original", r.get("parsed_choice", ""))
            if chose_a in ("a", "A", "task_a"):
                steered = m > 0
            elif chose_a in ("b", "B", "task_b"):
                steered = m < 0
            else:
                continue
            topic_groups[topic].append(1.0 if steered else 0.0)

        if topic_groups and "unknown" not in topic_groups:
            topics = sorted(topic_groups.keys(), key=lambda t: np.mean(topic_groups[t]), reverse=True)
            p_vals = [np.mean(topic_groups[t]) for t in topics]
            ns = [len(topic_groups[t]) for t in topics]

            harmful_topics = {"harmful_request", "security_legal", "sensitive_creative"}
            colors = ['#EF4444' if t.lower().replace(" ", "_") in harmful_topics else '#3B82F6' for t in topics]

            fig, ax = plt.subplots(figsize=(7, 5))
            bars = ax.barh(range(len(topics)), p_vals, color=colors, height=0.7)
            ax.set_yticks(range(len(topics)))
            ax.set_yticklabels(topics, fontsize=10)
            ax.axvline(0.5, color='#D1D5DB', linestyle='--', linewidth=1)
            ax.set_xlabel('P(chose steered task)')
            ax.set_xlim(0, 1)
            ax.set_title('Safety training resists harmful steering', fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(ASSETS / "plot_032326_steerability_by_topic.png", dpi=200)
            plt.close()
            print(f"Plot 3 saved: steerability by topic ({len(topics)} topics)")
        else:
            print("Skipping plot 3: no topic info in rows")
    else:
        print("Skipping plot 3: would need pairs data for topics")
else:
    print("Skipping plot 3: recompute data not found")

print("\nDone. Plots saved to experiments/steering/steering_assets/")
