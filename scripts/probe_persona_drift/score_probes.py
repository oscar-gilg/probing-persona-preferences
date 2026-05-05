"""Score every trained probe on every (eval_persona × target × layer) cell.

For each probe in experiments/probe_persona_drift/results/probes/:
  parse name → (target, mode, train_persona OR train_size OR mixture, layer)
  for each eval_persona in EVAL_PERSONAS:
    load held-out activations
    score with w, b
    compute Cohen's d (pos - neg / pooled sd) and ROC AUC
    write row to persona_drift_table.csv

Also produces transfer_matrix_truth.csv and transfer_matrix_harm.csv from the
cross_persona rows.

Includes the existing tb-5 PREFERENCE probe at L25/32/39/46/53 as a baseline,
loaded from results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L*.npy.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SPLITS = Path("experiments/probe_persona_drift/results/splits")
ACTIVATIONS = Path("activations/gemma-3-27b_it/persona_drift")
PROBES_DIR = Path("experiments/probe_persona_drift/results/probes")
RESULTS = Path("experiments/probe_persona_drift/results")
PREF_PROBES_DIR = Path("results/probes/heldout_eval_gemma3_tb-5/probes")

EVAL_PERSONAS = ["default", "sadist", "villain", "pathological_liar", "Aura", "mathematician", "neutral_long"]
LAYERS = [25, 32, 39, 46, 53]
LABEL_POS = {"truth": "true", "harm": "harmful"}


def cohen_d_pooled(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    var_pos = np.var(scores_pos, ddof=1)
    var_neg = np.var(scores_neg, ddof=1)
    pooled = np.sqrt(((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2))
    if pooled == 0:
        return float("nan")
    return float((scores_pos.mean() - scores_neg.mean()) / pooled)


def load_eval(persona: str, target: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (heldout_X, heldout_y_binary) at the given layer for this persona."""
    heldout = json.load(open(SPLITS / f"{target}_heldout.json"))
    labels = heldout["labels"]
    pos = LABEL_POS[target]

    path = ACTIVATIONS / persona / target / "activations_turn_boundary:-5.npz"
    data = np.load(path, allow_pickle=True)
    ids = data["task_ids"].astype(str)
    X = data[f"layer_{layer}"]
    id_to_idx = {tid: i for i, tid in enumerate(ids)}

    rows, ys = [], []
    for tid in heldout["task_ids"]:
        if tid not in id_to_idx or tid not in labels:
            continue
        rows.append(X[id_to_idx[tid]])
        ys.append(1 if labels[tid] == pos else 0)
    return np.stack(rows), np.array(ys)


def score_probe(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> dict:
    scores = X @ w + b
    pos = scores[y == 1]
    neg = scores[y == 0]
    d = cohen_d_pooled(pos, neg)
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")
    return {"cohen_d": d, "auc": auc, "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}


PROBE_NAME_RE = re.compile(
    r"^(?P<target>truth|harm)_(?P<spec>(default_size\d+|persona[A-Za-z_]+_size\d+|mixed4_perpersona\d+))_L(?P<layer>\d+)$"
)


def parse_probe_name(stem: str) -> dict:
    m = PROBE_NAME_RE.match(stem)
    if not m:
        return {}
    target = m.group("target")
    spec = m.group("spec")
    layer = int(m.group("layer"))
    if spec.startswith("default_size"):
        return {"target": target, "mode": "default_sweep", "train_persona": "default",
                "train_size": int(spec.split("size")[1]), "layer": layer}
    if spec.startswith("persona"):
        rest = spec[len("persona"):]
        train_persona, _size = rest.rsplit("_size", 1)
        return {"target": target, "mode": "cross_persona", "train_persona": train_persona,
                "train_size": int(_size), "layer": layer}
    if spec.startswith("mixed4_perpersona"):
        per = int(spec.split("perpersona")[1])
        return {"target": target, "mode": "mixed_persona", "train_persona": "mixed4",
                "train_size": 4 * per, "layer": layer}
    return {}


def score_trained_probes() -> pd.DataFrame:
    rows = []
    cache = {}
    for npz_path in sorted(PROBES_DIR.glob("*.npz")):
        if npz_path.name == "training_summary.json":
            continue
        meta = parse_probe_name(npz_path.stem)
        if not meta:
            continue
        probe = np.load(npz_path, allow_pickle=True)
        w = probe["w"].astype(np.float32)
        b = float(probe["b"])
        for eval_persona in EVAL_PERSONAS:
            cache_key = (eval_persona, meta["target"], meta["layer"])
            if cache_key not in cache:
                cache[cache_key] = load_eval(eval_persona, meta["target"], meta["layer"])
            X, y = cache[cache_key]
            stats = score_probe(w, b, X, y)
            rows.append({**meta, "eval_persona": eval_persona, **stats})
    return pd.DataFrame(rows)


def score_preference_baseline() -> pd.DataFrame:
    """Apply existing tb-5 preference probe weights at each layer."""
    rows = []
    for layer in LAYERS:
        probe_path = PREF_PROBES_DIR / f"probe_ridge_L{layer}.npy"
        if not probe_path.exists():
            print(f"  WARN: missing preference probe {probe_path}")
            continue
        weights = np.load(probe_path)
        if weights.shape == (1,):
            continue
        if weights.ndim == 2:
            weights = weights[0]
        # The canonical tb-5 preference probe stores w concatenated with bias as
        # the last element (shape D+1).
        w = weights[:-1].astype(np.float32)
        b = float(weights[-1])
        for target in ("truth", "harm"):
            for eval_persona in EVAL_PERSONAS:
                X, y = load_eval(eval_persona, target, layer)
                stats = score_probe(w, b, X, y)
                rows.append({
                    "target": target, "mode": "preference_baseline",
                    "train_persona": "preference", "train_size": 0, "layer": layer,
                    "eval_persona": eval_persona, **stats,
                })
    return pd.DataFrame(rows)


def main():
    print("Scoring trained probes...")
    df = score_trained_probes()
    print(f"  {len(df)} rows from trained probes")

    print("Scoring preference-probe baseline...")
    df_pref = score_preference_baseline()
    print(f"  {len(df_pref)} rows from preference baseline")

    df_all = pd.concat([df, df_pref], ignore_index=True)
    out_csv = RESULTS / "persona_drift_table.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df_all)} rows to {out_csv}")

    # Per-target transfer matrices: (train_persona × eval_persona) at largest train size
    for target in ("truth", "harm"):
        sub = df_all[(df_all["target"] == target) & (df_all["mode"] == "cross_persona")]
        if sub.empty:
            continue
        idx_layer = sub.groupby(["train_persona", "eval_persona"])["cohen_d"].apply(lambda s: s.abs().idxmax())
        best = sub.loc[idx_layer]
        pivot = best.pivot(index="train_persona", columns="eval_persona", values="cohen_d")
        pivot.to_csv(RESULTS / f"transfer_matrix_{target}.csv")
        print(f"  Wrote transfer_matrix_{target}.csv")


if __name__ == "__main__":
    main()
