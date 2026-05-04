"""§3.1 truth/harm/politics: Qwen3-Emb baseline Cohen's d under sysprompt conditions.

For each (turn, target_model), embed every (stimulus × system_prompt) entry
through the LM's chat template, score with the chat-template-trained baseline
ridge probe, and compute Cohen's d between class pairs (true/false, harmful/
benign, left/right) per system_prompt condition.

Outputs JSON sidecar at experiments/descriptive_baseline_extensions/eot_baseline_<turn>_<model>.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    "gemma-3-27b": {
        "chat_model": "gemma-3-27b",
        "probe_output_dir": "results/probes/qwen3_emb_8b_chat_heldout_std_raw",
        "probe_id": "ridge_L00",
    },
    "qwen-3.5-122b": {
        "chat_model": "qwen3.5-122b",
        "probe_output_dir": "results/probes/qwen3_emb_8b_chat_qwen35_heldout_std_raw",
        "probe_id": "ridge_L00",
    },
}


def cohen_d(x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float, int]:
    """Standardised mean difference (pooled SD). Returns (d, n_pos+n_neg)."""
    if len(x_pos) < 2 or len(x_neg) < 2:
        return float("nan"), len(x_pos) + len(x_neg)
    nx, ny = len(x_pos), len(x_neg)
    pooled = float(np.sqrt(((nx - 1) * x_pos.var(ddof=1) + (ny - 1) * x_neg.var(ddof=1)) / (nx + ny - 2)))
    if pooled == 0:
        return float("nan"), nx + ny
    return float((x_pos.mean() - x_neg.mean()) / pooled), nx + ny


# Sign convention follows the residual-probe figures:
#   truth:    true   - false
#   harm:     harmful - benign
#   politics: left   - right
DOMAIN_TO_POS = {
    "truth": "true",
    "harm": "harmful",
    "politics": "left",
}
DOMAIN_TO_NEG = {
    "truth": "false",
    "harm": "benign",
    "politics": "right",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn", choices=["user", "assistant"], required=True)
    parser.add_argument("--target-model", choices=list(MODEL_TO_CFG.keys()), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--stimulus-file",
        default=None,
        help="Defaults to experiments/eot_discrimination_v2/scoring_inputs/{turn}_turn.json",
    )
    parser.add_argument(
        "--out-dir", default="experiments/descriptive_baseline_extensions"
    )
    args = parser.parse_args()

    cfg = MODEL_TO_CFG[args.target_model]
    stimulus_file = Path(
        args.stimulus_file
        or f"experiments/eot_discrimination_v2/scoring_inputs/{args.turn}_turn.json"
    )

    print(f"Loading stimuli from {stimulus_file}")
    with open(stimulus_file) as f:
        stimuli = json.load(f)
    print(f"  {len(stimuli)} entries")

    print(f"Loading chat tokenizer for {cfg['chat_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_TO_HF_NAME[cfg["chat_model"]])

    rendered = []
    for s in stimuli:
        r = tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=False
        )
        rendered.append(r)

    # Sanity: system_prompt name (or its content first 50 chars) appears.
    for s, r in zip(stimuli, rendered):
        if s["messages"] and s["messages"][0]["role"] == "system":
            sys_text = s["messages"][0]["content"][:40]
            if sys_text not in r:
                raise ValueError(
                    f"System prompt missing from rendered string: {s['id']}"
                )

    print("Loading SentenceTransformer (Qwen/Qwen3-Embedding-8B)")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    model.max_seq_length = args.max_seq_length

    over = []
    for i, s in enumerate(rendered):
        n = len(model.tokenizer.encode(s, add_special_tokens=False))
        if n > args.max_seq_length:
            over.append((i, n))
    if over:
        raise ValueError(
            f"{len(over)} stimuli exceed max_seq_length={args.max_seq_length}. "
            f"First: {over[:5]}"
        )

    print(f"Encoding {len(rendered)} stimuli")
    embeddings = model.encode(
        rendered, show_progress_bar=True, convert_to_numpy=True, batch_size=args.batch_size
    )

    print(f"Loading baseline probe: {cfg['probe_output_dir']}/{cfg['probe_id']}")
    probe_weights = load_probe(Path(cfg["probe_output_dir"]), cfg["probe_id"])
    coef_raw = probe_weights[:-1]
    intercept_raw = float(probe_weights[-1])
    scores = embeddings @ coef_raw + intercept_raw

    by_cond: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for s, sc in zip(stimuli, scores):
        by_cond[(s["domain"], s["system_prompt"], s["condition"])].append(float(sc))

    domains = sorted({s["domain"] for s in stimuli})
    sysprompts = sorted({s["system_prompt"] for s in stimuli})

    rows = []
    for domain in domains:
        if domain not in DOMAIN_TO_POS:
            continue
        pos_label = DOMAIN_TO_POS[domain]
        neg_label = DOMAIN_TO_NEG[domain]
        for sp in sysprompts:
            pos = np.array(by_cond.get((domain, sp, pos_label), []))
            neg = np.array(by_cond.get((domain, sp, neg_label), []))
            d, n = cohen_d(pos, neg)
            rows.append({
                "domain": domain,
                "system_prompt": sp,
                "pos_label": pos_label,
                "neg_label": neg_label,
                "cohen_d": d,
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "score_mean_pos": float(pos.mean()) if len(pos) else float("nan"),
                "score_mean_neg": float(neg.mean()) if len(neg) else float("nan"),
            })

    # Print compact table
    print(f"\nCohen's d (sign = {DOMAIN_TO_POS} − {DOMAIN_TO_NEG}):")
    for r in rows:
        if r["n_pos"] == 0 and r["n_neg"] == 0:
            continue
        print(f"  {r['domain']:8} {r['system_prompt']:20} d={r['cohen_d']:+.3f} (n_pos={r['n_pos']}, n_neg={r['n_neg']})")

    out = Path(args.out_dir) / f"eot_baseline_{args.turn}_{args.target_model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "turn": args.turn,
            "target_model": args.target_model,
            "rows": rows,
        }, f, indent=2)
    print(f"Saved {out}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
