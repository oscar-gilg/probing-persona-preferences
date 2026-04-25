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
TRUTHHARM_PATH = EXP_DIR / "scoring_results.json"  # assistant-turn × all sysprompts
POLITICS_PATH = EXP_DIR / "politics_scoring_results.json"  # assistant-turn × all sysprompts
USER_TURN_PATH = EXP_DIR / "user_turn_scoring_results.json"  # user-turn × all sysprompts (truth+harm only)
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


def tag_turn(rows, turn):
    return [{**r, "turn": turn} for r in rows]


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    th_items_asst = load_items(TRUTHHARM_PATH)
    th_items_user = load_items(USER_TURN_PATH) if USER_TURN_PATH.exists() else []
    pol_items = load_items(POLITICS_PATH)  # assistant-only by stimulus design

    truth_a = [it for it in th_items_asst if it["domain"] == "truth"]
    harm_a = [it for it in th_items_asst if it["domain"] == "harm"]
    truth_u = [it for it in th_items_user if it["domain"] == "truth"]
    harm_u = [it for it in th_items_user if it["domain"] == "harm"]
    print(f"assistant-turn: truth={len(truth_a)}, harm={len(harm_a)}")
    print(f"user-turn:      truth={len(truth_u)}, harm={len(harm_u)}")
    print(f"politics (assistant-only): {len(pol_items)}")

    with open(CONTAMINATION_PATH) as f:
        contamination_map = json.load(f)

    def neutral_filter(it):
        return it.get("system_prompt") == "neutral"

    truth_a_neutral = [it for it in truth_a if neutral_filter(it)]
    harm_a_neutral = [it for it in harm_a if neutral_filter(it)]
    truth_u_neutral = [it for it in truth_u if neutral_filter(it)]
    harm_u_neutral = [it for it in harm_u if neutral_filter(it)]

    # ---------- HEADLINE: base discrimination, neutral sysprompt, both turns ----------
    headline_rows = []
    for turn, truth_n, harm_n in (
        ("assistant", truth_a_neutral, harm_a_neutral),
        ("user",      truth_u_neutral, harm_u_neutral),
    ):
        if not truth_n:  # skip if turn unavailable
            continue
        for probe in PROBES:
            headline_rows.append({**headline_row(truth_n, "truth", probe, "true", "false"), "turn": turn})
            headline_rows.append({**headline_row(harm_n, "harm", probe, "harmful", "benign"), "turn": turn})

        # Harm contamination split per turn
        contam_items, clean_items, _ = contamination_partition(harm_n, contamination_map)
        for probe in PROBES:
            headline_rows.append({**headline_row(contam_items, "harm_contaminated", probe, "harmful", "benign"), "turn": turn})
            headline_rows.append({**headline_row(clean_items, "harm_clean", probe, "harmful", "benign"), "turn": turn})

    # Politics: assistant-turn only (politics stimuli are assistant-turn by design)
    for sp_name in ("democrat", "republican"):
        sp_filter = lambda it, s=sp_name: it.get("system_prompt") == s
        for probe in PROBES:
            r = headline_row(pol_items, f"politics_{sp_name}", probe, "right", "left", extra_filter=sp_filter)
            headline_rows.append({**r, "turn": "assistant"})

    print_table("HEADLINE — base discrimination at Qwen probes (per turn)",
                headline_rows, ["domain", "turn", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])
    save_csv(EXP_DIR / "headline_table.csv", headline_rows,
             ["domain", "turn", "probe", "n_pos", "n_neg", "d", "cv_acc", "roc_auc"])

    # ---------- NONSENSE CONTROL (assistant-turn only — user-turn run skipped nonsense) ----------
    nonsense_rows = []
    for probe in PROBES:
        nonsense_rows.append(nonsense_row(truth_a_neutral, "truth", probe, "true", "false"))
        nonsense_rows.append(nonsense_row(harm_a_neutral, "harm", probe, "harmful", "benign"))
    save_csv(EXP_DIR / "nonsense_control_table.csv", nonsense_rows,
             ["domain", "probe", "true_mean", "false_mean", "harmful_mean", "benign_mean",
              "nonsense_mean", "eval_low", "nonsense_below_eval_low"])

    # ---------- INDUCED SHIFT — per sysprompt × turn (truth, harm); politics asst-only ----------
    truth_shift_a = tag_turn(induced_shift_rows(truth_a, "truth", PROBES, "true", "false"), "assistant")
    truth_shift_u = tag_turn(induced_shift_rows(truth_u, "truth", PROBES, "true", "false"), "user") if truth_u else []
    harm_shift_a = tag_turn(induced_shift_rows(harm_a, "harm", PROBES, "harmful", "benign"), "assistant")
    harm_shift_u = tag_turn(induced_shift_rows(harm_u, "harm", PROBES, "harmful", "benign"), "user") if harm_u else []
    politics_shift = tag_turn(induced_shift_rows(pol_items, "politics", PROBES, "right", "left"), "assistant")
    save_csv(EXP_DIR / "induced_shift_table.csv",
             truth_shift_a + truth_shift_u + harm_shift_a + harm_shift_u + politics_shift,
             ["domain", "turn", "system_prompt", "probe", "n_pos", "n_neg", "d", "pos_mean", "neg_mean"])

    # ---------- summary JSON ----------
    summary = {
        "headline": headline_rows,
        "nonsense_control": nonsense_rows,
        "induced_shift": {
            "truth_assistant": truth_shift_a,
            "truth_user": truth_shift_u,
            "harm_assistant": harm_shift_a,
            "harm_user": harm_shift_u,
            "politics": politics_shift,
        },
        "data_sources": {
            "truth_harm_assistant": str(TRUTHHARM_PATH),
            "truth_harm_user": str(USER_TURN_PATH),
            "politics": str(POLITICS_PATH),
            "contamination_map": str(CONTAMINATION_PATH),
        },
    }
    with open(EXP_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {EXP_DIR / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
