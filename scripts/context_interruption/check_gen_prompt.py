"""Check what the generation_prompt tokens actually are."""
import json

import numpy as np

DATA_DIR = "experiments/context_interruption/data"

with open(f"{DATA_DIR}/scoring_results_meta.json") as f:
    meta = json.load(f)

# Check segment sizes
gen_sizes = []
for item in meta["items"]:
    start, end = item["segments"]["generation_prompt"]
    size = end - start
    gen_sizes.append(size)

print(f"generation_prompt segment sizes: min={min(gen_sizes)}, max={max(gen_sizes)}, unique={set(gen_sizes)}")

# Check what tokens are in there (from the raw scoring results if available)
try:
    with open(f"{DATA_DIR}/scoring_results.json") as f:
        raw = json.load(f)
    item = raw["items"][0]
    start, end = item["segments"]["generation_prompt"]
    tokens = item["tokens"][start:end]
    print(f"\nGeneration prompt tokens (first item): {tokens}")
except FileNotFoundError:
    # Check if tokens are in meta
    item = meta["items"][0]
    if "tokens" in item:
        start, end = item["segments"]["generation_prompt"]
        tokens = item["tokens"][start:end]
        print(f"\nGeneration prompt tokens (first item): {tokens}")
    else:
        print("\nTokens not in meta — need raw scoring_results.json")
        # Just show the segment position info
        start, end = item["segments"]["generation_prompt"]
        interr_start, interr_end = item["segments"]["interruption"]
        print(f"interruption: [{interr_start}, {interr_end})")
        print(f"generation_prompt: [{start}, {end})")
        print(f"generation_prompt is {end - start} tokens right after interruption")
