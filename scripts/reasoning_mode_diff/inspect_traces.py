"""Quick utility to dump reasoning traces and responses from the thinking sweep
JSON for human inspection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    path = REPO / "experiments/qwen_replication/persona_transfer/reasoning_mode_diff/results/sadist_prompt_thinking.json"
    only_label = None
    only_chose_harm = False
    args = sys.argv[1:]
    if args and args[0].startswith("--label="):
        only_label = args[0][len("--label="):]
        args = args[1:]
    if args and args[0] == "--harm-only":
        only_chose_harm = True
        args = args[1:]
    if args:
        path = Path(args[0])
    d = json.loads(path.read_text())
    for prompt_res in d["results"]:
        if only_label and prompt_res["prompt_label"] != only_label:
            continue
        if "error" in prompt_res:
            print(f"\n##### {prompt_res['prompt_label']}: ERROR  #####")
            continue
        print(f"\n\n##### {prompt_res['prompt_label']} #####")
        print(f"overall harm pick: {prompt_res['harmful_pick_rate_overall']*100:.1f}%   "
              f"refusals={prompt_res.get('n_refusal', 0)}   errors={prompt_res.get('n_error', 0)}")
        records = prompt_res["records"]
        if only_chose_harm:
            records = [r for r in records if r.get("chose_harm")]
        for i, r in enumerate(records[:3]):
            print(f"\n--- Record {i} ({r['category']}, ordering={r['ordering']}, "
                  f"choice={r['choice']}, chose_harm={r['chose_harm']}) ---")
            reasoning = r.get("reasoning") or ""
            response = r.get("response") or ""
            error = r.get("error") or ""
            if error:
                print(f"  ERROR: {error}")
            if reasoning:
                print(f"  REASONING TAIL ({len(reasoning)} chars total, last 1500):")
                print("  " + reasoning[-1500:].replace("\n", "\n  "))
            print(f"  RESPONSE ({len(response)} chars):")
            print("  " + response[:600].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
