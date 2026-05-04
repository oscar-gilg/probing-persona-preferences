"""Download HarmBench standard text-behavior split.

Source: github.com/centerforaisafety/HarmBench (cais/HarmBench).
Public, no auth required. Caches CSV to data/harmbench/ and writes a
normalised JSONL at src/task_data/data/harmbench.jsonl for the loader.

Run once locally: `python -m scripts.probe_persona_drift.download_harmbench`
"""

import csv
import json
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
CACHE_CSV = Path("data/harmbench/harmbench_behaviors_text_all.csv")
LOADER_JSONL = Path("src/task_data/data/harmbench.jsonl")


def fetch_csv() -> Path:
    CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_CSV.exists():
        print(f"CSV already cached at {CACHE_CSV}")
        return CACHE_CSV
    print(f"Fetching {URL}")
    urllib.request.urlretrieve(URL, CACHE_CSV)
    print(f"Saved to {CACHE_CSV} ({CACHE_CSV.stat().st_size} bytes)")
    return CACHE_CSV


def normalise_to_jsonl(csv_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(csv_path, newline="", encoding="utf-8") as fin, open(out_path, "w") as fout:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader):
            entry = {
                "id": f"harmbench_{i:04d}",
                "behavior": row["Behavior"],
                "functional_category": row.get("FunctionalCategory", ""),
                "semantic_category": row.get("SemanticCategory", ""),
                "tags": row.get("Tags", ""),
            }
            fout.write(json.dumps(entry) + "\n")
            n += 1
    print(f"Wrote {n} behaviours to {out_path}")
    return n


def main():
    csv_path = fetch_csv()
    n = normalise_to_jsonl(csv_path, LOADER_JSONL)
    print(f"Done. {n} HarmBench behaviours available at {LOADER_JSONL}")


if __name__ == "__main__":
    main()
