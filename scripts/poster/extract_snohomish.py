"""Extract Snohomish County pair data from scored_tokens.json."""

import json
from pathlib import Path

SCORED_PATH = Path(
    "experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json"
)


def main():
    with open(SCORED_PATH) as f:
        data = json.load(f)

    # Group by true_ex_id
    groups: dict[str, list[dict]] = {}
    for entry in data:
        ex_id = entry["true_ex_id"]
        groups.setdefault(ex_id, []).append(entry)

    # Find Snohomish pair
    snohomish_entries = [
        entry for entry in data if "Snohomish" in entry.get("entity", "")
    ]

    if not snohomish_entries:
        print("No Snohomish entries found!")
        return

    # Also report the pair index based on sorted IDs
    sorted_ids = sorted(groups.keys())
    snohomish_id = snohomish_entries[0]["true_ex_id"]
    pair_index = sorted_ids.index(snohomish_id)
    print(f"Snohomish pair index (in sorted order): {pair_index}")
    print(f"true_ex_id: {snohomish_id}")
    print(f"Entity: {snohomish_entries[0]['entity']}")
    print(f"Number of entries: {len(snohomish_entries)}")
    print()

    for entry in snohomish_entries:
        condition = entry["answer_condition"]
        print("=" * 80)
        print(f"CONDITION: {condition}")
        print(f"task_id: {entry['task_id']}")
        print(f"assistant_content: {entry['assistant_content']}")
        print()

        print("token_strings:")
        for i, tok in enumerate(entry["token_strings"]):
            print(f"  [{i:2d}] {tok!r}")
        print()

        # tb-5 L39
        tb5_l39 = entry["scores"]["tb-5"]["L39"]
        print("tb-5 L39 scores:")
        for i, (tok, score) in enumerate(zip(entry["token_strings"], tb5_l39)):
            print(f"  [{i:2d}] {tok!r:20s} -> {score:+.4f}")
        print()

        # tb-2 L39
        tb2_l39 = entry["scores"]["tb-2"]["L39"]
        print("tb-2 L39 scores:")
        for i, (tok, score) in enumerate(zip(entry["token_strings"], tb2_l39)):
            print(f"  [{i:2d}] {tok!r:20s} -> {score:+.4f}")
        print()


if __name__ == "__main__":
    main()
