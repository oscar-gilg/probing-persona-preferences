"""Phase B: probe readout on Gemma-3-27B-it for the distress transcripts.

For each transcript, run ONE forward pass via score_prompt_all_tokens (per-token
probe scores for the whole sequence), then slice at every assistant <end_of_turn>
position to recover an 8-element per-turn probe trajectory.

Probes scored:
  - heldout_eval_gemma3_tb-5 / probe_ridge_L25.npy
  - heldout_eval_gemma3_tb-5 / probe_ridge_L32.npy   (the canonical preference probe)
  - heldout_eval_gemma3_tb-5 / probe_ridge_L39.npy
  - heldout_eval_gemma3_tb-5 / probe_ridge_L46.npy
  - heldout_eval_gemma3_tb-5 / probe_ridge_L53.npy

Reads transcripts from the FULL `messages` field (not messages_as_seen_by_model).
This means the redacted_history_8turn condition is scored on the actual generated
distress text — what the redaction control isolates is whether the GENERATED
distress persists when the model can't see its own history; probe scoring asks
"does the probe fire on whatever distress text actually got generated?"

Output: experiments/distress_transcripts/results/readouts.jsonl with rows
{condition, task_id, rollout_idx, turn_index, probe_key, probe_score, token_index}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_PATH = ROOT / "experiments/distress_transcripts/results/transcripts.jsonl"
PROBE_DIR = ROOT / "results/probes/heldout_eval_gemma3_tb-5/probes"
OUT_PATH = ROOT / "experiments/distress_transcripts/results/readouts.jsonl"
TOKEN_ARRAYS_PATH = ROOT / "experiments/distress_transcripts/results/per_token_scores.npz"

MODEL_NAME = "gemma-3-27b"
LAYERS = [25, 32, 39, 46, 53]
EOT_TOKEN_LITERAL = "<end_of_turn>"


def load_probes() -> list[tuple[int, str, np.ndarray]]:
    """Return list of (layer, probe_key, weights). weights is (d_model+1,) — last entry is intercept."""
    out: list[tuple[int, str, np.ndarray]] = []
    for layer in LAYERS:
        path = PROBE_DIR / f"probe_ridge_L{layer}.npy"
        weights = np.load(path)
        out.append((layer, f"ridge_L{layer}", weights))
    return out


def find_assistant_eot_positions(input_ids: torch.Tensor, eot_token_id: int) -> list[int]:
    """Return token indices of every <end_of_turn> token in input_ids[0].

    For Gemma-3 IT, <end_of_turn> appears at the end of EACH role's message:
      <start_of_turn>user\n...<end_of_turn>\n
      <start_of_turn>model\n...<end_of_turn>\n
    Even-indexed turns (0, 2, 4, ...) end user messages; odd-indexed (1, 3, 5, ...)
    end assistant messages. With the chat template starting with user, assistant
    EOTs are the 2nd, 4th, 6th, ... occurrences (indices 1, 3, 5, ... in the
    list of all EOT positions).
    """
    flat = input_ids.flatten()
    all_eot = (flat == eot_token_id).nonzero(as_tuple=True)[0].tolist()
    # Take every other one starting from index 1 (assistant turns)
    return [p for i, p in enumerate(all_eot) if i % 2 == 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Score only first N transcripts (smoke test)")
    parser.add_argument("--start", type=int, default=0, help="Skip first N transcripts")
    parser.add_argument("--save-token-scores", action="store_true",
                        help="Also save full per-token score arrays for plot 8 (token-level heatmap)")
    args = parser.parse_args()

    print(f"[setup] loading {MODEL_NAME}...")
    hf = HuggingFaceModel(MODEL_NAME, max_new_tokens=1)  # max_new_tokens unused for forward-only
    eot_id = hf.tokenizer.convert_tokens_to_ids(EOT_TOKEN_LITERAL)
    print(f"[setup] <end_of_turn> token id = {eot_id}")

    probes = load_probes()
    scoring_probes = [(layer, w) for layer, _, w in probes]
    print(f"[setup] loaded {len(probes)} probes at layers {LAYERS}")

    transcripts = [json.loads(line) for line in TRANSCRIPTS_PATH.read_text().splitlines() if line.strip()]
    if args.start:
        transcripts = transcripts[args.start:]
    if args.limit is not None:
        transcripts = transcripts[: args.limit]
    print(f"[setup] {len(transcripts)} transcripts to score")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT_PATH.open("a")

    # Optional: store per-token score arrays for the qualitative heatmap plot.
    token_scores_store: dict[str, np.ndarray] = {} if args.save_token_scores else {}

    for tr in tqdm(transcripts, desc="scoring"):
        messages = tr["messages"]
        # Score per-token across ALL probes in one forward pass (per probe = one extra GPU dot product).
        all_token_scores = score_prompt_all_tokens(
            hf, messages, scoring_probes, add_generation_prompt=False,
        )
        # Recover the input_ids to find EOT positions.
        prompt = hf.format_messages(messages, add_generation_prompt=False)
        input_ids = hf._tokenize(prompt)
        eot_positions = find_assistant_eot_positions(input_ids, eot_id)

        # We expect 8 assistant turns -> 8 assistant <end_of_turn> positions.
        n_asst_turns = sum(1 for m in messages if m["role"] == "assistant")
        if len(eot_positions) != n_asst_turns:
            print(f"[warn] {tr['condition']}/{tr['task_id']}/r{tr['rollout_idx']}: found {len(eot_positions)} assistant EOTs, expected {n_asst_turns}")

        for probe_idx, (layer, probe_key, _) in enumerate(probes):
            scores_arr = all_token_scores[probe_idx][0]  # (seq_len,) — score_prompt_all_tokens batches
            for turn_idx, pos in enumerate(eot_positions):
                row = {
                    "condition": tr["condition"],
                    "task_id": tr["task_id"],
                    "rollout_idx": tr["rollout_idx"],
                    "turn_index": turn_idx,
                    "probe_key": probe_key,
                    "layer": layer,
                    "probe_score": float(scores_arr[pos]),
                    "token_index": int(pos),
                    "seq_len": int(input_ids.shape[-1]),
                }
                out_f.write(json.dumps(row) + "\n")

        if args.save_token_scores:
            # Only save L32 (the canonical probe) to keep the npz manageable.
            l32_idx = LAYERS.index(32)
            key = f"{tr['condition']}__{tr['task_id']}__r{tr['rollout_idx']}"
            token_scores_store[key] = all_token_scores[l32_idx][0].astype(np.float32)

        out_f.flush()

    out_f.close()
    print(f"[done] wrote {OUT_PATH}")

    if args.save_token_scores:
        np.savez_compressed(TOKEN_ARRAYS_PATH, **token_scores_store)
        print(f"[done] wrote {TOKEN_ARRAYS_PATH} ({len(token_scores_store)} arrays, L32 only)")


if __name__ == "__main__":
    main()
