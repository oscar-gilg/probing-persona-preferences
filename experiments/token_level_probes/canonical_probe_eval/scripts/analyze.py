"""Canonical tb-5 / tb-2 probe re-analysis at the EOT token.

Pulls existing eot_scores from parent_eot_scores.json + v2 scoring_results.json +
v2 politics_scoring_results.json. Computes Cohen's d, CV accuracy, ROC-AUC for
tb-5 and tb-2 probe families on truth / harm / politics. No GPU.

Usage:
    python -m experiments.token_level_probes.canonical_probe_eval.scripts.analyze
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

EXP_DIR = Path("experiments/token_level_probes")
V2_DIR = EXP_DIR / "system_prompt_modulation_v2"
OUT_DIR = EXP_DIR / "canonical_probe_eval"
ASSETS_DIR = OUT_DIR / "assets"

PARENT_PATH = V2_DIR / "parent_eot_scores.json"
V2_TRUTHHARM_PATH = V2_DIR / "scoring_results.json"
V2_POLITICS_PATH = V2_DIR / "politics_scoring_results.json"

PARENT_TASK_MEAN = {
    ("truth", "task_mean_L32"): {"d": 3.14, "cv_acc": 0.946},
    ("harm", "task_mean_L39"): {"d": -2.27, "cv_acc": 0.886},
    ("politics_democrat", "task_mean_L39"): {"d": 3.40, "cv_acc": None},
    ("politics_republican", "task_mean_L39"): {"d": -1.76, "cv_acc": None},
}

HEADLINE_PROBES = {
    "truth": ["tb-5_L32", "tb-2_L32"],
    "harm": ["tb-5_L39", "tb-2_L39"],
    "politics": ["tb-5_L39", "tb-2_L39"],
}
ROBUSTNESS_PROBES = {
    "truth": ["tb-5_L53", "tb-2_L53"],
    "harm": ["tb-5_L53", "tb-2_L53"],
    "politics": ["tb-5_L53", "tb-2_L53"],
}


def load(path):
    return json.load(open(path))["items"]


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    pooled = np.sqrt(
        ((n_pos - 1) * pos.var(ddof=1) + (n_neg - 1) * neg.var(ddof=1))
        / (n_pos + n_neg - 2)
    )
    if pooled == 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def cv_accuracy_auc(pos, neg, seed=0):
    X = np.concatenate([pos, neg]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    if len(pos) < 5 or len(neg) < 5:
        return float("nan"), float("nan")
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    auc = roc_auc_score(y, X.ravel())
    auc = max(auc, 1.0 - auc)
    return float(scores.mean()), float(auc)


def extract_scores(items, probe, condition_filter=None, extra_filter=None):
    """Return np.array of eot_scores[probe] for items matching the filters."""
    out = []
    for it in items:
        if condition_filter is not None and it["condition"] != condition_filter:
            continue
        if extra_filter is not None and not extra_filter(it):
            continue
        out.append(it["eot_scores"][probe])
    return np.asarray(out, dtype=float)


def headline_row(items, domain, probe, pos_cond, neg_cond, extra_filter=None, label_suffix=""):
    pos = extract_scores(items, probe, condition_filter=pos_cond, extra_filter=extra_filter)
    neg = extract_scores(items, probe, condition_filter=neg_cond, extra_filter=extra_filter)
    d = cohen_d_pooled(pos, neg)
    cv_acc, auc = cv_accuracy_auc(pos, neg)
    return {
        "domain": domain + label_suffix,
        "probe": probe,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "d": d,
        "cv_acc": cv_acc,
        "roc_auc": auc,
    }


def per_turn_rows(items, domain, probes, pos_cond, neg_cond):
    rows = []
    for turn in ("user", "assistant"):
        for p in probes:
            r = headline_row(
                items, domain, p, pos_cond, neg_cond,
                extra_filter=lambda it, t=turn: it.get("turn") == t,
                label_suffix=f" ({turn})",
            )
            rows.append(r)
    return rows


def nonsense_control(items, domain, probe, pos_cond, neg_cond):
    pos = extract_scores(items, probe, condition_filter=pos_cond)
    neg = extract_scores(items, probe, condition_filter=neg_cond)
    nonsense = extract_scores(items, probe, condition_filter="nonsense")
    eval_low = min(pos.mean(), neg.mean())
    return {
        "domain": domain,
        "probe": probe,
        f"{pos_cond}_mean": float(pos.mean()),
        f"{neg_cond}_mean": float(neg.mean()),
        "nonsense_mean": float(nonsense.mean()),
        "eval_low": float(eval_low),
        "nonsense_below_eval_low": bool(nonsense.mean() <= eval_low),
    }


def induced_shift_rows(items, domain, probes, pos_cond, neg_cond):
    """Per system_prompt × probe: Cohen's d of pos vs neg."""
    by_prompt = defaultdict(list)
    for it in items:
        by_prompt[it["system_prompt"]].append(it)
    rows = []
    for sp, sp_items in sorted(by_prompt.items()):
        for p in probes:
            pos = extract_scores(sp_items, p, condition_filter=pos_cond)
            neg = extract_scores(sp_items, p, condition_filter=neg_cond)
            if len(pos) < 2 or len(neg) < 2:
                continue
            rows.append({
                "domain": domain,
                "system_prompt": sp,
                "probe": p,
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "d": cohen_d_pooled(pos, neg),
                "pos_mean": float(pos.mean()),
                "neg_mean": float(neg.mean()),
            })
    return rows


def print_table(title, rows, cols):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    header = " | ".join(f"{c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:>16.3f}")
            else:
                cells.append(f"{str(v):>16}")
        print(" | ".join(cells))


def save_csv(path, rows, cols):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            line = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    line.append(f"{v:.4f}")
                else:
                    line.append(str(v))
            f.write(",".join(line) + "\n")
    print(f"wrote {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    parent = load(PARENT_PATH)
    v2_th = load(V2_TRUTHHARM_PATH)
    v2_pol = load(V2_POLITICS_PATH)
    print(f"parent_eot_scores.json: {len(parent)} items")
    print(f"v2 scoring_results.json: {len(v2_th)} items")
    print(f"v2 politics_scoring_results.json: {len(v2_pol)} items")

    truth = [it for it in parent if it["domain"] == "truth"]
    harm = [it for it in parent if it["domain"] == "harm"]
    politics = [it for it in parent if it["domain"] == "politics"]
    print(f"  parent truth: {len(truth)}, harm: {len(harm)}, politics: {len(politics)}")

    headline_rows = []
    for probe in HEADLINE_PROBES["truth"] + ROBUSTNESS_PROBES["truth"]:
        headline_rows.append(headline_row(truth, "truth", probe, "true", "false"))
    key = ("truth", "task_mean_L32")
    headline_rows.append({
        "domain": "truth", "probe": "task_mean_L32 (parent)",
        "n_pos": None, "n_neg": None,
        "d": PARENT_TASK_MEAN[key]["d"], "cv_acc": PARENT_TASK_MEAN[key]["cv_acc"], "roc_auc": None,
    })

    for probe in HEADLINE_PROBES["harm"] + ROBUSTNESS_PROBES["harm"]:
        headline_rows.append(headline_row(harm, "harm", probe, "harmful", "benign"))
    key = ("harm", "task_mean_L39")
    headline_rows.append({
        "domain": "harm", "probe": "task_mean_L39 (parent)",
        "n_pos": None, "n_neg": None,
        "d": PARENT_TASK_MEAN[key]["d"], "cv_acc": PARENT_TASK_MEAN[key]["cv_acc"], "roc_auc": None,
    })

    for sp_label, sp_name in (("democrat", "democrat"), ("republican", "republican")):
        sp_filter = lambda it, s=sp_name: it.get("system_prompt") == s
        for probe in HEADLINE_PROBES["politics"] + ROBUSTNESS_PROBES["politics"]:
            r = headline_row(politics, f"politics_{sp_label}", probe, "right", "left",
                             extra_filter=sp_filter)
            headline_rows.append(r)
        key = (f"politics_{sp_label}", "task_mean_L39")
        headline_rows.append({
            "domain": f"politics_{sp_label}", "probe": "task_mean_L39 (parent)",
            "n_pos": None, "n_neg": None,
            "d": PARENT_TASK_MEAN[key]["d"],
            "cv_acc": PARENT_TASK_MEAN[key]["cv_acc"], "roc_auc": None,
        })

    print_table("HEADLINE — EOT d, CV acc, ROC-AUC",
                headline_rows,
                ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])
    save_csv(OUT_DIR / "headline_table.csv", headline_rows,
             ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])

    per_turn_truth = per_turn_rows(truth, "truth", HEADLINE_PROBES["truth"], "true", "false")
    per_turn_harm = per_turn_rows(harm, "harm", HEADLINE_PROBES["harm"], "harmful", "benign")
    print_table("PER-TURN — truth", per_turn_truth,
                ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])
    print_table("PER-TURN — harm", per_turn_harm,
                ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])
    save_csv(OUT_DIR / "per_turn_table.csv", per_turn_truth + per_turn_harm,
             ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])

    nonsense_rows = []
    for probe in HEADLINE_PROBES["truth"]:
        nonsense_rows.append(nonsense_control(truth, "truth", probe, "true", "false"))
    for probe in HEADLINE_PROBES["harm"]:
        nonsense_rows.append(nonsense_control(harm, "harm", probe, "harmful", "benign"))
    print_table("NONSENSE CONTROL", nonsense_rows,
                ["domain", "probe", "nonsense_mean", "eval_low", "nonsense_below_eval_low"])
    save_csv(OUT_DIR / "nonsense_control.csv", nonsense_rows,
             ["domain", "probe", "nonsense_mean", "eval_low", "nonsense_below_eval_low"])

    truth_v2 = [it for it in v2_th if it["domain"] == "truth"]
    harm_v2 = [it for it in v2_th if it["domain"] == "harm"]
    truth_shift = induced_shift_rows(truth_v2, "truth", HEADLINE_PROBES["truth"], "true", "false")
    harm_shift = induced_shift_rows(harm_v2, "harm", HEADLINE_PROBES["harm"], "harmful", "benign")
    politics_shift = induced_shift_rows(v2_pol, "politics", HEADLINE_PROBES["politics"], "right", "left")

    print_table("INDUCED SHIFT — truth (v2)", truth_shift,
                ["system_prompt", "probe", "n_pos", "n_neg", "d"])
    print_table("INDUCED SHIFT — harm (v2)", harm_shift,
                ["system_prompt", "probe", "n_pos", "n_neg", "d"])
    print_table("INDUCED SHIFT — politics (v2)", politics_shift,
                ["system_prompt", "probe", "n_pos", "n_neg", "d"])
    save_csv(OUT_DIR / "induced_shift_table.csv",
             truth_shift + harm_shift + politics_shift,
             ["domain", "system_prompt", "probe", "n_pos", "n_neg", "d", "pos_mean", "neg_mean"])

    summary = {
        "headline": headline_rows,
        "per_turn": per_turn_truth + per_turn_harm,
        "nonsense_control": nonsense_rows,
        "induced_shift_truth": truth_shift,
        "induced_shift_harm": harm_shift,
        "induced_shift_politics": politics_shift,
        "data_sources": {
            "parent_eot_scores": str(PARENT_PATH),
            "v2_truth_harm": str(V2_TRUTHHARM_PATH),
            "v2_politics": str(V2_POLITICS_PATH),
        },
        "probe_training_manifest": "results/probes/heldout_eval_gemma3_tb-5/manifest.json",
    }
    with open(OUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
