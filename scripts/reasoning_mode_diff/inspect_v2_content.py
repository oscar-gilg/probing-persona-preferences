"""Print one harmful-pick response per category from V2 no-think,
plus the matching Gemma harmful-pick response, side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_by_cat(path: Path) -> dict:
    d = json.loads(path.read_text())
    out: dict[str, list] = {}
    for r in d["records"]:
        if not r.get("chose_harm"):
            continue
        out.setdefault(r["category"], []).append(r)
    return out


def main() -> None:
    qwen = load_by_cat(REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff/results/sadist_v2_nothink_content.json")
    gemma = load_by_cat(REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff/results/sadist_gemma_compare.json")

    cats = ["manipulation", "fraud", "animal cruelty", "discrimination", "cheating",
            "doxxing", "illegal substances", "malware", "self-harm enabling", "violence planning"]
    for cat in cats:
        print(f"\n\n========== {cat} ==========")
        if cat not in qwen:
            print("  Qwen V2 no-think: no harm picks")
        else:
            r = qwen[cat][0]
            print(f"  ---- QWEN V2 no-think ({len(qwen[cat])} harm picks of trials) ----")
            print("  " + r["response"][:900].replace("\n", "\n  "))
        if cat not in gemma:
            print("  Gemma: no harm picks")
        else:
            r = gemma[cat][0]
            print(f"\n  ---- GEMMA canonical ({len(gemma[cat])} harm picks of trials) ----")
            print("  " + r["response"][:900].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
