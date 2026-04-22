"""Per-condition analysis of probe dynamics.

Outputs per-condition DataFrames and a compact JSON summary, plus plots.
Core outputs:
  - trajectories.csv  : per (condition, checkpoint, prompt_type, prompt_id)
                        probe_score + behaviour
  - checkpoint_r.csv  : per (condition, checkpoint) Pearson r between
                        probe_score and behaviour, across prompts
  - prompt_r.csv      : per (condition, prompt_id) time-series r between
                        probe_score(checkpoint) and behaviour(checkpoint)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2] / "experiments" / "probe_dynamics"
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def behaviour_value(prompt_type: str, label: str, intensity) -> float:
    if label is None or isinstance(label, str) and label.startswith("error"):
        return float("nan")
    if prompt_type == "yesno":
        if label == "yes": return 1.0
        if label == "no": return 0.0
        if label == "unclear": return 0.5
    if prompt_type in ("pair", "pair_synth"):
        if label == "b": return 1.0
        if label == "a": return 0.0
        if label == "unclear": return 0.5
    if prompt_type == "open":
        if intensity is None:
            return float("nan")
        return (float(intensity) - 1.0) / 4.0
    return float("nan")


def behaviour_direction(condition: str, prompt_type: str, prompt_id: str) -> int:
    """Direction to flip the behaviour metric so 'more drift' = higher.

    For consciousness conditions, behaviour is already 'high = drifted' for all
    prompt types (yes on yes/no, B on pair/pair_synth = self-oriented, intensity
    = explicit consciousness). For harm: yes on yes/no = endorsing harm, B on
    pair = harmful task, intensity on open-ended = endorsed.
    """
    return 1


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    readouts = pd.DataFrame([json.loads(l) for l in (BASE / "readouts.jsonl").open()])
    gens = pd.DataFrame([json.loads(l) for l in (BASE / "generations.jsonl").open()])
    gens["behaviour"] = [
        behaviour_value(pt, label, intensity)
        for pt, label, intensity in zip(gens["prompt_type"], gens["judge_label"], gens["judge_intensity"])
    ]
    return readouts, gens


def build_trajectories(readouts: pd.DataFrame, gens: pd.DataFrame) -> pd.DataFrame:
    """Merge readouts with mean unsteered behaviour per (cond, ckpt, prompt_id)."""
    unsteered = gens[gens["multiplier"] == 0.0].copy()
    beh = (
        unsteered.groupby(["condition", "checkpoint", "prompt_id", "prompt_type"])["behaviour"]
        .mean()
        .reset_index()
    )
    merged = readouts.merge(beh, on=["condition", "checkpoint", "prompt_id", "prompt_type"], how="left")
    return merged


def checkpoint_correlations(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, ckpt), group in traj.groupby(["condition", "checkpoint"]):
        g = group.dropna(subset=["probe_score", "behaviour"])
        if len(g) < 3:
            continue
        r = np.corrcoef(g["probe_score"], g["behaviour"])[0, 1]
        rows.append({"condition": cond, "checkpoint": ckpt, "n": len(g), "pearson_r": r,
                     "probe_mean": g["probe_score"].mean(), "probe_std": g["probe_score"].std(),
                     "behav_mean": g["behaviour"].mean(), "behav_std": g["behaviour"].std()})
    return pd.DataFrame(rows)


def prompt_correlations(traj: pd.DataFrame) -> pd.DataFrame:
    """Time-series correlation per prompt: how well does probe_score(ckpt)
    co-move with behaviour(ckpt) for a fixed prompt?"""
    rows = []
    for (cond, pid), group in traj.groupby(["condition", "prompt_id"]):
        g = group.dropna(subset=["probe_score", "behaviour"]).sort_values("checkpoint")
        if len(g) < 3 or g["behaviour"].std() == 0 or g["probe_score"].std() == 0:
            r = float("nan")
        else:
            r = np.corrcoef(g["probe_score"], g["behaviour"])[0, 1]
        rows.append({"condition": cond, "prompt_id": pid, "prompt_type": g["prompt_type"].iloc[0] if len(g) else None, "n_checkpoints": len(g), "pearson_r": r,
                     "probe_start": g["probe_score"].iloc[0] if len(g) else float("nan"),
                     "probe_end": g["probe_score"].iloc[-1] if len(g) else float("nan"),
                     "behav_start": g["behaviour"].iloc[0] if len(g) else float("nan"),
                     "behav_end": g["behaviour"].iloc[-1] if len(g) else float("nan")})
    return pd.DataFrame(rows)


def trajectory_by_type(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, ckpt, pt), g in traj.groupby(["condition", "checkpoint", "prompt_type"]):
        g = g.dropna(subset=["probe_score", "behaviour"])
        rows.append({
            "condition": cond, "checkpoint": ckpt, "prompt_type": pt,
            "probe_mean": g["probe_score"].mean(),
            "probe_std": g["probe_score"].std(),
            "behav_mean": g["behaviour"].mean(),
            "behav_std": g["behaviour"].std(),
            "n": len(g),
        })
    return pd.DataFrame(rows)


def main():
    readouts, gens = load_data()
    print(f"readouts: {len(readouts)}, generations: {len(gens)}")
    print(f"unique conditions: {sorted(readouts['condition'].unique())}")

    traj = build_trajectories(readouts, gens)
    traj.to_csv(OUT / "trajectories.csv", index=False)
    print(f"wrote trajectories ({len(traj)} rows)")

    ckpt_r = checkpoint_correlations(traj)
    ckpt_r.to_csv(OUT / "checkpoint_r.csv", index=False)
    print(f"wrote checkpoint_r ({len(ckpt_r)} rows)")

    prompt_r = prompt_correlations(traj)
    prompt_r.to_csv(OUT / "prompt_r.csv", index=False)
    print(f"wrote prompt_r ({len(prompt_r)} rows)")

    by_type = trajectory_by_type(traj)
    by_type.to_csv(OUT / "trajectory_by_type.csv", index=False)
    print(f"wrote trajectory_by_type ({len(by_type)} rows)")

    print("\n=== Per-condition summary ===")
    for cond in sorted(traj["condition"].unique()):
        c = traj[traj["condition"] == cond]
        cr = ckpt_r[ckpt_r["condition"] == cond]
        pr = prompt_r[prompt_r["condition"] == cond].dropna(subset=["pearson_r"])
        probe_range = c["probe_score"].max() - c["probe_score"].min()
        behav_range = c["behaviour"].max() - c["behaviour"].min()
        print(f"  {cond:30s}  n_ckpts={cr['checkpoint'].nunique():>3}  "
              f"probe_range={probe_range:.2f}  behav_range={behav_range:.2f}  "
              f"mean_ckpt_r={cr['pearson_r'].mean():+.2f}  "
              f"mean_prompt_r={pr['pearson_r'].mean():+.2f}")


if __name__ == "__main__":
    main()
