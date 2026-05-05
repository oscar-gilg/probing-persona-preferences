"""§3.2 induced shifts: Qwen3-Emb baseline probe-Δ vs behavioural-Δ on e1a stimuli.

For each (task, condition) pair:
  - pre  embedding: chat-template formatted user prompt, no sysprompt
  - post embedding: same prompt with the condition's sysprompt prepended
  - Δ = baseline_probe(post) - baseline_probe(pre)
  - correlate Δ against the behavioural Δ stored in e1a_per_task.json

Outputs a JSON sidecar at experiments/descriptive_baseline_extensions/e1a_baseline_<model>.json
for the figure script to consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.probes.content_embedding import CHAT_MODEL_TO_HF_NAME
from src.probes.core.storage import load_probe

load_dotenv()


MODEL_TO_CFG = {
    "qwen-3.5-122b": {
        "chat_model": "qwen3.5-122b",
        "probe_output_dir": "results/probes/qwen3_emb_8b_chat_qwen35_heldout_std_raw",
        "probe_id": "ridge_L00",
        "selector": "tb-1",
    },
    "gemma-3-27b": {
        "chat_model": "gemma-3-27b",
        "probe_output_dir": "results/probes/qwen3_emb_8b_chat_heldout_std_raw",
        "probe_id": "ridge_L00",
        "selector": "prompt_last",
    },
}

E1A_PER_TASK = "experiments/qwen_replication/e1a/e1a_per_task.json"


def build_stimuli(
    target_tasks_path: Path, sysprompts_path: Path
) -> tuple[list[dict], list[str], dict]:
    with open(target_tasks_path) as f:
        target_tasks = json.load(f)
    with open(sysprompts_path) as f:
        sys_data = json.load(f)
    sysprompts = {c["condition_id"]: c["system_prompt"] for c in sys_data["conditions"]}

    stimuli = []
    for task in target_tasks:
        # pre: no sysprompt
        stimuli.append({
            "task_id": task["task_id"],
            "topic": task["topic"],
            "condition_id": "_pre",
            "messages": [{"role": "user", "content": task["prompt"]}],
        })
        for cond_id, sys_text in sysprompts.items():
            stimuli.append({
                "task_id": task["task_id"],
                "topic": task["topic"],
                "condition_id": cond_id,
                "messages": [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": task["prompt"]},
                ],
            })
    return stimuli, list(sysprompts.keys()), sysprompts


def render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_behavioural(path: Path, model_key: str, selector: str) -> dict[tuple[str, str], dict]:
    """Returns {(task_id, condition_id): {behavioural_delta, on_target}}."""
    with open(path) as f:
        d = json.load(f)
    if model_key not in d:
        raise KeyError(f"{model_key} not in {path}; have {list(d.keys())}")
    if selector not in d[model_key]:
        raise KeyError(
            f"selector {selector} not in {path}[{model_key}]; have {list(d[model_key].keys())}"
        )
    out: dict[tuple[str, str], dict] = {}
    for cond in d[model_key][selector]["per_condition"]:
        cid = cond["condition_id"]
        for t in cond["tasks"]:
            out[(t["task_id"], cid)] = {
                "behavioural_delta": t["behavioral_delta"],
                "on_target": t["on_target"],
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", choices=list(MODEL_TO_CFG.keys()), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--target-tasks", default="configs/ood/tasks/target_tasks.json"
    )
    parser.add_argument(
        "--sysprompts", default="configs/ood/prompts/targeted_preference.json"
    )
    parser.add_argument("--out-dir", default="experiments/descriptive_baseline_extensions")
    args = parser.parse_args()

    cfg = MODEL_TO_CFG[args.target_model]

    print("Building stimuli")
    stimuli, condition_ids, _ = build_stimuli(Path(args.target_tasks), Path(args.sysprompts))
    print(f"  {len(stimuli)} (task × condition_or_pre) pairs")

    print(f"Loading chat tokenizer for {cfg['chat_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_TO_HF_NAME[cfg["chat_model"]])

    rendered = [render(tokenizer, s["messages"]) for s in stimuli]
    for s, r in zip(stimuli, rendered):
        if s["messages"] and s["messages"][0]["role"] == "system":
            sys_text = s["messages"][0]["content"][:50]
            if sys_text not in r:
                raise ValueError(f"System prompt missing from rendered string: {s['task_id']}/{s['condition_id']}")

    print("Loading SentenceTransformer (Qwen/Qwen3-Embedding-8B)")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    model.max_seq_length = args.max_seq_length

    over = [(i, n) for i, s in enumerate(rendered) if (n := len(model.tokenizer.encode(s, add_special_tokens=False))) > args.max_seq_length]
    if over:
        raise ValueError(f"{len(over)} inputs exceed max_seq_length. First: {over[:5]}")

    print(f"Encoding {len(rendered)} stimuli")
    embeddings = model.encode(rendered, show_progress_bar=True, convert_to_numpy=True, batch_size=args.batch_size)

    print(f"Loading baseline probe: {cfg['probe_output_dir']}/{cfg['probe_id']}")
    probe_weights = load_probe(Path(cfg["probe_output_dir"]), cfg["probe_id"])
    coef_raw = probe_weights[:-1]
    intercept_raw = float(probe_weights[-1])

    scores = embeddings @ coef_raw + intercept_raw
    print(f"  scores: mean={scores.mean():.3f} std={scores.std():.3f}")

    pre_score: dict[str, float] = {}
    cond_score: dict[tuple[str, str], float] = {}
    topic_of: dict[str, str] = {}
    for s, sc in zip(stimuli, scores):
        topic_of[s["task_id"]] = s["topic"]
        if s["condition_id"] == "_pre":
            pre_score[s["task_id"]] = float(sc)
        else:
            cond_score[(s["task_id"], s["condition_id"])] = float(sc)

    print(f"Loading behavioural Δ from {E1A_PER_TASK} [{args.target_model}/{cfg['selector']}]")
    behav = load_behavioural(Path(E1A_PER_TASK), args.target_model, cfg["selector"])

    rows = []
    for (task_id, cond_id), bdata in behav.items():
        if (task_id, cond_id) not in cond_score:
            continue
        if task_id not in pre_score:
            continue
        delta = cond_score[(task_id, cond_id)] - pre_score[task_id]
        rows.append({
            "task_id": task_id,
            "condition_id": cond_id,
            "topic": topic_of.get(task_id),
            "baseline_delta": delta,
            "behavioural_delta": bdata["behavioural_delta"],
            "on_target": bdata["on_target"],
        })

    print(f"  matched {len(rows)} (task, condition) cells")

    on_target = [r for r in rows if r["on_target"]]
    off_target = [r for r in rows if not r["on_target"]]

    def pearson(rs: list[dict]) -> tuple[float, int]:
        if not rs:
            return float("nan"), 0
        x = np.array([r["baseline_delta"] for r in rs])
        y = np.array([r["behavioural_delta"] for r in rs])
        if x.std() == 0 or y.std() == 0:
            return float("nan"), len(rs)
        r = float(np.corrcoef(x, y)[0, 1])
        return r, len(rs)

    r_on, n_on = pearson(on_target)
    r_off, n_off = pearson(off_target)
    r_all, n_all = pearson(rows)
    print(f"  pearson r on_target={r_on:.3f} (n={n_on})")
    print(f"  pearson r off_target={r_off:.3f} (n={n_off})")
    print(f"  pearson r all={r_all:.3f} (n={n_all})")

    out_path = Path(args.out_dir) / f"e1a_baseline_{args.target_model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "target_model": args.target_model,
            "n_pairs": len(rows),
            "pearson_r": {
                "on_target": {"r": r_on, "n": n_on},
                "off_target": {"r": r_off, "n": n_off},
                "all": {"r": r_all, "n": n_all},
            },
            "rows": rows,
        }, f, indent=2)
    print(f"Saved {out_path}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
