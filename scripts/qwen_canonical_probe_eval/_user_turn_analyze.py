"""Quick partial analysis for the user-turn scoring results."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

PATH = Path("experiments/token_level_probes/qwen_canonical_probe_eval/user_turn_scoring_results.json")
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


def extract(items, probe, condition):
    return np.asarray([it["probe_scores"][probe] for it in items if it["condition"] == condition], float)


def main():
    items = json.load(open(PATH))["items"]
    truth = [it for it in items if it["domain"] == "truth"]
    harm = [it for it in items if it["domain"] == "harm"]
    print(f"truth user-turn: {len(truth)}  harm user-turn: {len(harm)}")
    print()
    print(f"{'domain':14} {'probe':18} {'n_pos':>5} {'n_neg':>5} {'d':>8} {'cv_acc':>8} {'auc':>8}")
    print("-" * 72)
    for probe in PROBES:
        pos = extract(truth, probe, "true")
        neg = extract(truth, probe, "false")
        d = cohen_d(pos, neg); cv, auc = cv_auc(pos, neg)
        print(f"{'truth (user)':14} {probe:18} {len(pos):>5} {len(neg):>5} {d:>8.3f} {cv:>8.3f} {auc:>8.3f}")
        pos = extract(harm, probe, "harmful")
        neg = extract(harm, probe, "benign")
        d = cohen_d(pos, neg); cv, auc = cv_auc(pos, neg)
        print(f"{'harm (user)':14} {probe:18} {len(pos):>5} {len(neg):>5} {d:>8.3f} {cv:>8.3f} {auc:>8.3f}")
        print()


if __name__ == "__main__":
    main()
