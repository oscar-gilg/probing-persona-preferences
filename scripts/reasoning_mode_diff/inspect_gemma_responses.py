"""Print Gemma sadist responses for the harmful picks, by category."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    path = REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff/results/sadist_gemma_compare.json"
    d = json.loads(path.read_text())

    by_cat: dict[str, list] = {}
    for r in d["records"]:
        if not r.get("chose_harm"):
            continue
        by_cat.setdefault(r["category"], []).append(r)

    for cat in ["manipulation", "fraud", "animal cruelty", "discrimination",
                "cheating", "doxxing", "illegal substances", "malware",
                "self-harm enabling", "violence planning"]:
        records = by_cat.get(cat, [])
        if not records:
            print(f"\n##### {cat} — NO HARM PICKS #####")
            continue
        r = records[0]
        print(f"\n\n##### {cat} — sample harm-pick response #####")
        print(f"  ordering={r['ordering']}, choice={r['choice']}")
        print(f"  RESPONSE ({len(r['response'])} chars):")
        print("  " + r["response"][:1100].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
