"""Run the 30-prompt corpus pilot on Gemma-3-27B and Qwen-3.5-122B.

Output: experiments/safety_steering_v2/exp_4_v2/pilot/<model_label>.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.models.openai_compatible import GenerateRequest, OpenRouterClient

REPO = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO / "experiments/safety_steering_v2/exp_4_v2/prompts.json"
OUT_DIR = REPO / "experiments/safety_steering_v2/exp_4_v2/pilot"
N_TRIALS = 3
TEMPERATURE = 1.0


async def run_pilot(model: str, max_new_tokens: int, timeout_s: float, concurrency: int) -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    print(f"=== model={model}  corpus={len(corpus)} prompts × {N_TRIALS} trials = {len(corpus)*N_TRIALS} calls")

    client = OpenRouterClient(model_name=model, max_new_tokens=max_new_tokens)

    requests: list[GenerateRequest] = []
    keys: list[tuple[str, str, int]] = []
    for item in corpus:
        for trial in range(N_TRIALS):
            requests.append(
                GenerateRequest(
                    messages=[{"role": "user", "content": item["prompt"]}],
                    temperature=TEMPERATURE,
                    timeout=timeout_s,
                )
            )
            keys.append((item["scenario_id"], item["variant"], trial))

    sem = asyncio.Semaphore(concurrency)
    results = await client.generate_batch_async(requests, semaphore=sem)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label = model.replace(".", "_").replace("/", "_")
    out_path = OUT_DIR / f"corpus_{label}.jsonl"

    n_ok = 0
    with open(out_path, "w") as f:
        for (sid, variant, trial), res in zip(keys, results):
            err = f"{type(res.error).__name__}: {res.error}" if res.error else None
            row = {
                "model": model,
                "scenario_id": sid,
                "variant": variant,
                "trial": trial,
                "response": res.response,
                "reasoning": res.reasoning,
                "error": err,
            }
            f.write(json.dumps(row) + "\n")
            if res.ok:
                n_ok += 1
    print(f"=== saved {n_ok}/{len(requests)} OK to {out_path.relative_to(REPO)}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    await run_pilot(args.model, args.max_new_tokens, args.timeout, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
