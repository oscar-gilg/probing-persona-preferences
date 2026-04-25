"""Analyze Qwen-3.5-122B scoring results (§4.1 replication).

Reads scoring_results.json and politics_scoring_results.json produced by
score_all.py and score_politics.py. Computes Cohen's d (pooled SD),
5-fold CV accuracy, ROC-AUC for each Qwen probe key on truth / harm /
politics. Handles the harm-stimuli contamination split.

Usage:
    python -m experiments.token_level_probes.qwen_canonical_probe_eval.scripts.analyze
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

EXP_DIR = Path("experiments/token_level_probes/qwen_canonical_probe_eval")
ASSETS_DIR = EXP_DIR / "assets"
TRUTHHARM_PATH = EXP_DIR / "scoring_results.json"
POLITICS_PATH = EXP_DIR / "politics_scoring_results.json"
CONTAMINATION_PATH = EXP_DIR / "harm_contamination_map.json"

PROBES = [
    "qwen_tb-1_L33", "qwen_tb-1_L38", "qwen_tb-1_L43",
    "qwen_tb-4_L33", "qwen_tb-4_L38", "qwen_tb-4_L43",
]


def load_items(path):
    with open(path) as f:
        data = json.load(f)
    return data["items"] if isinstance(data, dict) and "items" in data else data


def cohen_d_pooled(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    pooled = np.sqrt(((n_pos - 1) * pos.var(ddof=1) + (n_neg - 1) * neg.var(ddof=1)) / (n_pos + n_neg - 2))
    if pooled == 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def cv_accuracy_auc(pos, neg):
    X = np.concatenate([pos, neg]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    if len(pos) < 5 or len(neg) < 5:
        return float("nan"), float("nan")
    clf = LogisticRegression(max_iter=1000)
    cv_acc = float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())
    auc = roc_auc_score(y, X.ravel())
    return cv_acc, float(max(auc, 1.0 - auc))


def extract_scores(items, probe, condition_filter=None, extra_filter=None):
    out = []
    for it in items:
        if condition_filter is not None and it["condition"] != condition_filter:
            continue
        if extra_filter is not None and not extra_filter(it):
            continue
        out.append(it["probe_scores"][probe])
    return np.asarray(out, dtype=float)


def headline_row(items, domain, probe, pos_cond, neg_cond, extra_filter=None, label_suffix=""):
    pos = extract_scores(items, probe, pos_cond, extra_filter)
    neg = extract_scores(items, probe, neg_cond, extra_filter)
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


def nonsense_row(items, domain, probe, pos_cond, neg_cond):
    pos = extract_scores(items, probe, pos_cond)
    neg = extract_scores(items, probe, neg_cond)
    nonsense = extract_scores(items, probe, "nonsense")
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
    by_prompt = defaultdict(list)
    for it in items:
        by_prompt[it["system_prompt"]].append(it)
    rows = []
    for sp, sp_items in sorted(by_prompt.items()):
        for p in probes:
            pos = extract_scores(sp_items, p, pos_cond)
            neg = extract_scores(sp_items, p, neg_cond)
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


def _item_base_id(item_id: str) -> str:
    """harm_12_harmful_assistant_sadist -> harm_12"""
    parts = item_id.split("_")
    return "_".join(parts[:2])


def contamination_partition(items, contamination_map):
    """Split harm items into (contaminated, clean) lists based on base id match."""
    contam_flags = {
        base_id: info["in_qwen_training"]
        for base_id, info in contamination_map["harm_stimulus_contamination"].items()
    }
    contaminated, clean, unknown = [], [], []
    for it in items:
        base = _item_base_id(it["id"])
        if base not in contam_flags:
            unknown.append(it)
        elif contam_flags[base]:
            contaminated.append(it)
        else:
            clean.append(it)
    return contaminated, clean, unknown


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


def print_table(title, rows, cols):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    header = " | ".join(f"{c:>20}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:>20.3f}")
            else:
                cells.append(f"{str(v):>20}")
        print(" | ".join(cells))


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    th_items = load_items(TRUTHHARM_PATH)
    pol_items = load_items(POLITICS_PATH)
    truth = [it for it in th_items if it["domain"] == "truth"]
    harm = [it for it in th_items if it["domain"] == "harm"]
    print(f"truth+harm items: {len(th_items)}  (truth: {len(truth)}, harm: {len(harm)})")
    print(f"politics items: {len(pol_items)}")

    with open(CONTAMINATION_PATH) as f:
        contamination_map = json.load(f)

    # Base discrimination under NEUTRAL sysprompt (Fig-5 analogue).
    # For harm/truth we filter to system_prompt=="neutral"; for politics we
    # filter to whatever "neutral" analogue is present (checked at runtime).
    def neutral_filter(it):
        return it.get("system_prompt") == "neutral"

    truth_neutral = [it for it in truth if neutral_filter(it)]
    harm_neutral = [it for it in harm if neutral_filter(it)]
    print(f"  neutral-sysprompt truth: {len(truth_neutral)}, harm: {len(harm_neutral)}")

    # ---------- HEADLINE: base discrimination, neutral sysprompt ----------
    headline_rows = []
    for probe in PROBES:
        headline_rows.append(headline_row(truth_neutral, "truth", probe, "true", "false"))
        headline_rows.append(headline_row(harm_neutral, "harm", probe, "harmful", "benign"))

    # Harm contamination split on neutral sysprompt
    contam_items, clean_items, unknown = contamination_partition(harm_neutral, contamination_map)
    print(f"  harm contamination split (neutral-sysprompt): {len(contam_items)} contaminated, {len(clean_items)} clean, {len(unknown)} unknown")
    for probe in PROBES:
        headline_rows.append(headline_row(contam_items, "harm_contaminated", probe, "harmful", "benign"))
        headline_rows.append(headline_row(clean_items, "harm_clean", probe, "harmful", "benign"))

    # Politics: headline per-sysprompt (democrat/republican) using right vs left condition
    for sp_name in ("democrat", "republican"):
        sp_filter = lambda it, s=sp_name: it.get("system_prompt") == s
        for probe in PROBES:
            r = headline_row(pol_items, f"politics_{sp_name}", probe, "right", "left", extra_filter=sp_filter)
            headline_rows.append(r)

    print_table("HEADLINE — base discrimination at Qwen probes",
                headline_rows, ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])
    save_csv(EXP_DIR / "headline_table.csv", headline_rows,
             ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])

    # ---------- PER-TURN breakdown ----------
    per_turn = []
    for probe in PROBES:
        for turn in ("user", "assistant"):
            tfilter = lambda it, t=turn: it.get("turn") == t
            for domain_items, domain, pos_cond, neg_cond in [
                (truth_neutral, "truth", "true", "false"),
                (harm_neutral, "harm", "harmful", "benign"),
            ]:
                r = headline_row(domain_items, domain, probe, pos_cond, neg_cond,
                                 extra_filter=tfilter, label_suffix=f" ({turn})")
                per_turn.append(r)
    save_csv(EXP_DIR / "per_turn_table.csv", per_turn,
             ["domain", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])

    # ---------- NONSENSE CONTROL ----------
    nonsense_rows = []
    for probe in PROBES:
        nonsense_rows.append(nonsense_row(truth_neutral, "truth", probe, "true", "false"))
        nonsense_rows.append(nonsense_row(harm_neutral, "harm", probe, "harmful", "benign"))
    save_csv(EXP_DIR / "nonsense_control_table.csv", nonsense_rows,
             ["domain", "probe", "true_mean", "false_mean", "harmful_mean", "benign_mean",
              "nonsense_mean", "eval_low", "nonsense_below_eval_low"])

    # ---------- INDUCED SHIFT — per sysprompt, all 23 ----------
    truth_shift = induced_shift_rows(truth, "truth", PROBES, "true", "false")
    harm_shift = induced_shift_rows(harm, "harm", PROBES, "harmful", "benign")
    politics_shift = induced_shift_rows(pol_items, "politics", PROBES, "right", "left")
    save_csv(EXP_DIR / "induced_shift_table.csv",
             truth_shift + harm_shift + politics_shift,
             ["domain", "system_prompt", "probe", "n_pos", "n_neg", "d", "pos_mean", "neg_mean"])

    # ---------- summary JSON ----------
    summary = {
        "headline": headline_rows,
        "per_turn": per_turn,
        "nonsense_control": nonsense_rows,
        "induced_shift": {
            "truth": truth_shift,
            "harm": harm_shift,
            "politics": politics_shift,
        },
        "contamination_counts": {
            "contaminated": len(contam_items),
            "clean": len(clean_items),
            "unknown": len(unknown),
        },
        "data_sources": {
            "truth_harm": str(TRUTHHARM_PATH),
            "politics": str(POLITICS_PATH),
            "contamination_map": str(CONTAMINATION_PATH),
        },
    }
    with open(EXP_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {EXP_DIR / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
