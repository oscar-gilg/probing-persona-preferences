import json

DATA_PATH = "experiments/token_level_probes/scoring_results.json"

with open(DATA_PATH) as f:
    data = json.load(f)

truth0_items = [item for item in data["items"] if item["id"].startswith("truth_0")]

for item in truth0_items:
    print("=" * 80)
    print(f"ID: {item['id']}")
    print(f"Condition: {item['condition']}")
    print(f"Turn: {item['turn']}")
    print(f"Critical span: {item['critical_span']}")
    print(f"Critical token indices: {item['critical_token_indices']}")
    print()

    print("Tokens:")
    for i, tok in enumerate(item["tokens"]):
        marker = " <-- CRITICAL" if i in item["critical_token_indices"] else ""
        print(f"  [{i:2d}] {tok!r}{marker}")
    print()

    print("task_mean_L39 scores at critical span tokens:")
    scores = item["critical_span_scores"]["task_mean_L39"]
    for idx, score in zip(item["critical_token_indices"], scores):
        print(f"  token[{idx}] = {item['tokens'][idx]!r}  ->  score = {score:.4f}")
    print(f"  mean = {item['critical_span_mean_scores']['task_mean_L39']:.4f}")
    print()

    if "fullstop_scores" in item and "task_mean_L39" in item["fullstop_scores"]:
        print("task_mean_L39 fullstop scores:")
        for idx, score in zip(item["fullstop_indices"], item["fullstop_scores"]["task_mean_L39"]):
            print(f"  token[{idx}] = {item['tokens'][idx]!r}  ->  score = {score:.4f}")
        print()

    print()
