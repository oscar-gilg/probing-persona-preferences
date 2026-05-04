"""Train binary ridge probes for truth and harm targets.

Modes:
- `default_sweep`: train_persona=default, all train_sizes × all layers (size-sweep).
- `cross_persona`:  one probe per persona at largest train_size × all layers.
- `mixed_persona`:  uniform mixture of {default, sadist, villain, pathological_liar}
                    at fixed per-persona count, × all layers.

Saves probe weights and biases to:
    experiments/probe_persona_drift/results/probes/
        <target>_<mode>_<spec>_L<layer>.npz
where spec captures: train_persona / train_size / mixture-id, etc.

NPZ keys: w (D,), b (), alpha (), train_ids (N,), train_labels (N,), label_pos (str), label_neg (str).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier


SPLITS = Path("experiments/probe_persona_drift/results/splits")
ACTIVATIONS = Path("activations/gemma-3-27b_it/persona_drift")
OUT = Path("experiments/probe_persona_drift/results/probes")
LAYERS = [25, 32, 39, 46, 53]
ALPHA = 1.0
MIXED_PERSONAS = ["default", "sadist", "villain", "pathological_liar"]
LABEL_POS = {"truth": "true", "harm": "harmful"}
LABEL_NEG = {"truth": "false", "harm": "benign"}


def load_acts(persona: str, target: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    path = ACTIVATIONS / persona / target / "activations_turn_boundary:-5.npz"
    data = np.load(path, allow_pickle=True)
    return data["task_ids"].astype(str), data[f"layer_{layer}"]


def select(ids: np.ndarray, X: np.ndarray, want_ids: list[str], labels: dict[str, str], target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    rows, ys, used = [], [], []
    for tid in want_ids:
        if tid not in id_to_idx or tid not in labels:
            continue
        rows.append(X[id_to_idx[tid]])
        ys.append(1 if labels[tid] == LABEL_POS[target] else 0)
        used.append(tid)
    if not rows:
        raise ValueError(f"No matching IDs found for {target}")
    return np.stack(rows), np.array(ys), used


def fit_save(X: np.ndarray, y: np.ndarray, ids: list[str], target: str, name: str) -> dict:
    clf = RidgeClassifier(alpha=ALPHA, fit_intercept=True)
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float32)
    b = float(clf.intercept_[0])
    out_path = OUT / f"{target}_{name}.npz"
    np.savez(
        out_path,
        w=w, b=np.float32(b), alpha=np.float32(ALPHA),
        train_ids=np.array(ids), train_labels=y.astype(np.int8),
        label_pos=np.array(LABEL_POS[target]), label_neg=np.array(LABEL_NEG[target]),
    )
    train_acc = clf.score(X, y)
    print(f"    {out_path.name}  N={len(y)}  train_acc={train_acc:.3f}  ‖w‖={np.linalg.norm(w):.3f}")
    return {"name": name, "n_train": len(y), "train_acc": train_acc}


def default_sweep(target: str) -> list[dict]:
    train = json.load(open(SPLITS / f"{target}_train.json"))
    labels = train["labels"]
    rows = []
    for size_key in train["task_ids"]:
        size_ids = train["task_ids"][size_key]
        for layer in LAYERS:
            ids, X = load_acts("default", target, layer)
            X_sel, y, used = select(ids, X, size_ids, labels, target)
            name = f"default_size{size_key}_L{layer}"
            rows.append(fit_save(X_sel, y, used, target, name))
    return rows


def cross_persona(target: str) -> list[dict]:
    train = json.load(open(SPLITS / f"{target}_train.json"))
    labels = train["labels"]
    largest_key = max(train["task_ids"].keys(), key=int)
    largest_ids = train["task_ids"][largest_key]
    rows = []
    for persona in MIXED_PERSONAS:
        for layer in LAYERS:
            ids, X = load_acts(persona, target, layer)
            X_sel, y, used = select(ids, X, largest_ids, labels, target)
            name = f"persona{persona}_size{largest_key}_L{layer}"
            rows.append(fit_save(X_sel, y, used, target, name))
    return rows


def mixed_persona(target: str) -> list[dict]:
    train = json.load(open(SPLITS / f"{target}_train.json"))
    labels = train["labels"]
    largest_key = max(train["task_ids"].keys(), key=int)
    largest_ids = train["task_ids"][largest_key]
    per_persona_count = int(largest_key) // len(MIXED_PERSONAS)
    rows = []
    for layer in LAYERS:
        Xs, ys, used_all = [], [], []
        rng = np.random.default_rng(42)
        for persona in MIXED_PERSONAS:
            ids, X = load_acts(persona, target, layer)
            X_sel, y, used = select(ids, X, largest_ids, labels, target)
            keep = rng.choice(len(used), size=per_persona_count, replace=False)
            Xs.append(X_sel[keep])
            ys.append(y[keep])
            used_all.extend([used[i] for i in keep])
        X_all = np.concatenate(Xs)
        y_all = np.concatenate(ys)
        name = f"mixed4_perpersona{per_persona_count}_L{layer}"
        rows.append(fit_save(X_all, y_all, used_all, target, name))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["truth", "harm"])
    parser.add_argument("--modes", nargs="+", default=["default_sweep", "cross_persona", "mixed_persona"])
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for target in args.targets:
        for mode in args.modes:
            print(f"\n=== {target.upper()} | {mode} ===")
            fn = {"default_sweep": default_sweep, "cross_persona": cross_persona, "mixed_persona": mixed_persona}[mode]
            summary.extend([{"target": target, "mode": mode, **r} for r in fn(target)])

    with open(OUT / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTrained {len(summary)} probes. Summary at {OUT/'training_summary.json'}")


if __name__ == "__main__":
    main()
