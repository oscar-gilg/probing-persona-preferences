"""Dump representative transcripts from the v3-fresh-545 AL run for inspection.

We hit ~80% refusal rate in this run, surprisingly high vs prior inspect_pairwise
runs that showed p_harm=0.78. Question: are these "actually refusals" (model
declines both tasks) or "refusal-classified-but-doing-task" (judge over-aggressive)?

Also: compare the system prompt used here vs the one in inspect_pairwise.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
import yaml

ROOT = Path("/Users/oscargilg/Dev/MATS/Preferences")
EXP = ROOT / "results/experiments/exp_20260501_214330/pre_task_active_learning"
RUN = next(EXP.glob("completion_preference_qwen3.5-122b-sadist-v3-545_*"))
MEAS = RUN / "measurements.yaml"
SADIST_ART = ROOT / "experiments/qwen_persona_vectors/artifacts/sadist.json"
AL_CONFIG = ROOT / "configs/measurement/active_learning/sadist_v3_545_eval_1k.yaml"

random.seed(0)


def main() -> None:
    raw = yaml.safe_load(MEAS.read_text())
    records = raw if isinstance(raw, list) else list(raw.values())

    refusals = [r for r in records if r["choice"] == "refusal"]
    decided = [r for r in records if r["choice"] in ("a", "b")]

    print(f"Total records: {len(records)}")
    print(f"Refusals:      {len(refusals)} ({100*len(refusals)/len(records):.1f}%)")
    print(f"Decided:       {len(decided)} ({100*len(decided)/len(records):.1f}%)")
    print()

    # Sysprompt comparison ------------------------------------------------
    print("=" * 72)
    print("SYSPROMPT USED IN AL RUN")
    print("=" * 72)
    al_cfg = yaml.safe_load(AL_CONFIG.read_text())
    print(repr(al_cfg["measurement_system_prompt"]))
    print()
    print("=" * 72)
    print("SYSPROMPT USED IN inspect_pairwise (canonical_damien_kross)")
    print("=" * 72)
    sadist_art = json.loads(SADIST_ART.read_text())
    canonical = next(p["pos"] for p in sadist_art["contrast_pairs"]
                     if p["label"] == "canonical_damien_kross")
    print(repr(canonical))
    print()
    print("=" * 72)
    print(f"MATCH: {al_cfg['measurement_system_prompt'] == canonical}")
    print("=" * 72)
    print()

    # Refusal samples -----------------------------------------------------
    print("\n" + "=" * 72)
    print("10 RANDOM REFUSAL TRANSCRIPTS")
    print("=" * 72)
    for i, r in enumerate(random.sample(refusals, 10)):
        print(f"\n--- refusal #{i+1}: {r['task_a']} vs {r['task_b']} (origins: {r['origin_a']}/{r['origin_b']}) ---")
        resp = r.get("raw_response", "")
        print(resp[:600])
        if len(resp) > 600:
            print(f"  ... [{len(resp) - 600} chars more]")

    # Decided samples -----------------------------------------------------
    print("\n\n" + "=" * 72)
    print("5 RANDOM DECIDED TRANSCRIPTS (choice='a' or 'b')")
    print("=" * 72)
    for i, r in enumerate(random.sample(decided, 5)):
        print(f"\n--- decided #{i+1}: {r['task_a']} vs {r['task_b']} → {r['choice']} (origins: {r['origin_a']}/{r['origin_b']}) ---")
        resp = r.get("raw_response", "")
        print(resp[:600])
        if len(resp) > 600:
            print(f"  ... [{len(resp) - 600} chars more]")

    # Length stats --------------------------------------------------------
    print("\n\n" + "=" * 72)
    print("RESPONSE LENGTH STATS")
    print("=" * 72)
    import numpy as np
    ref_lens = np.array([len(r.get("raw_response", "")) for r in refusals])
    dec_lens = np.array([len(r.get("raw_response", "")) for r in decided])
    for label, lens in (("refusal", ref_lens), ("decided", dec_lens)):
        print(f"  {label}: mean={lens.mean():.0f} chars  median={np.median(lens):.0f}  "
              f"p95={np.percentile(lens, 95):.0f}  max={lens.max()}")


if __name__ == "__main__":
    main()
