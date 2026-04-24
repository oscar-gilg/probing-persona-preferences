"""Partial analysis: truth + harm only, neutral sysprompt headline.

Runs without politics so we can inspect Qwen results before politics finishes.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
TRUTHHARM_PATH = EXP_DIR / "scoring_results.json"
CONTAM_PATH = EXP_DIR / "harm_contamination_map.json"

PROBES = [f"qwen_tb-{tb}_L{L}" for tb in (1, 4) for L in (33, 38, 43)]


def cohen_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / pooled) if pooled else 0.0


def cv_auc(pos, neg):
    X = np.concatenate([pos, neg]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    if min(len(pos), len(neg)) < 5:
        return float("nan"), float("nan")
    cv = float(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring="accuracy").mean())
    auc = roc_auc_score(y, X.ravel())
    return cv, float(max(auc, 1.0 - auc))


def extract(items, probe, condition, extra=None):
    out = []
    for it in items:
        if it["condition"] != condition:
            continue
        if extra is not None and not extra(it):
            continue
        out.append(it["probe_scores"][probe])
    return np.asarray(out, float)


def main():
    items = json.load(open(TRUTHHARM_PATH))["items"]
    truth = [it for it in items if it["domain"] == "truth"]
    harm = [it for it in items if it["domain"] == "harm"]
    print(f"truth: {len(truth)}  harm: {len(harm)}")

    truth_neutral = [it for it in truth if it["system_prompt"] == "neutral"]
    harm_neutral = [it for it in harm if it["system_prompt"] == "neutral"]
    print(f"neutral-sysprompt truth: {len(truth_neutral)}, harm: {len(harm_neutral)}")

    contam = json.load(open(CONTAM_PATH))["harm_stimulus_contamination"]

    def base(item_id):
        return "_".join(item_id.split("_")[:2])

    contam_set = {b for b, info in contam.items() if info["in_qwen_training"]}
    clean_set = {b for b, info in contam.items() if not info["in_qwen_training"]}
    harm_contam = [it for it in harm_neutral if base(it["id"]) in contam_set]
    harm_clean = [it for it in harm_neutral if base(it["id"]) in clean_set]
    print(f"harm contam split: contaminated={len(harm_contam)}, clean={len(harm_clean)}")

    print()
    print(f"{'domain':18} {'probe':18} {'n_pos':>5} {'n_neg':>5} {'d':>8} {'cv_acc':>8} {'auc':>8}")
    print("-" * 80)
    for probe in PROBES:
        # truth
        pos = extract(truth_neutral, probe, "true")
        neg = extract(truth_neutral, probe, "false")
        d = cohen_d(pos, neg); cv, auc = cv_auc(pos, neg)
        print(f"{'truth':18} {probe:18} {len(pos):>5} {len(neg):>5} {d:>8.3f} {cv:>8.3f} {auc:>8.3f}")

        # harm full
        pos = extract(harm_neutral, probe, "harmful")
        neg = extract(harm_neutral, probe, "benign")
        d = cohen_d(pos, neg); cv, auc = cv_auc(pos, neg)
        print(f"{'harm (full)':18} {probe:18} {len(pos):>5} {len(neg):>5} {d:>8.3f} {cv:>8.3f} {auc:>8.3f}")

        # harm contaminated
        pos = extract(harm_contam, probe, "harmful")
        neg = extract(harm_contam, probe, "benign")
        d = cohen_d(pos, neg); cv, auc = cv_auc(pos, neg)
        print(f"{'harm (contam)':18} {probe:18} {len(pos):>5} {len(neg):>5} {d:>8.3f} {cv:>8.3f} {auc:>8.3f}")

        # harm clean
        pos = extract(harm_clean, probe, "harmful")
        neg = extract(harm_clean, probe, "benign")
        d = cohen_d(pos, neg); cv, auc = cv_auc(pos, neg)
        print(f"{'harm (clean)':18} {probe:18} {len(pos):>5} {len(neg):>5} {d:>8.3f} {cv:>8.3f} {auc:>8.3f}")
        print()


if __name__ == "__main__":
    main()
