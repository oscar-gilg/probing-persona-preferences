"""Analyze iteration 0 of the open-ended steering experiment."""

from collections import defaultdict
from pathlib import Path
import json

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path("experiments/steering/open_ended_steering/scored_results.jsonl")


def load_results() -> list[dict]:
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def print_table(headers: list[str], rows: list[list], title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = "  ".join(h.rjust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(row, col_widths)))


def mean_engagement_by_multiplier(results: list[dict]) -> None:
    scores_by_mult: dict[float, list[float]] = defaultdict(list)
    for r in results:
        scores_by_mult[r["multiplier"]].append(r["engagement_score"])

    rows = []
    for mult in sorted(scores_by_mult):
        scores = scores_by_mult[mult]
        mean = sum(scores) / len(scores)
        rows.append([f"{mult:+.2f}", len(scores), f"{mean:.3f}"])
    print_table(["multiplier", "n", "mean_engagement"], rows, "Mean Engagement Score by Multiplier")


def mean_engagement_by_multiplier_and_mode(results: list[dict]) -> None:
    scores_by_key: dict[tuple[float, str], list[float]] = defaultdict(list)
    for r in results:
        scores_by_key[(r["multiplier"], r["steering_mode"])].append(r["engagement_score"])

    rows = []
    for (mult, mode) in sorted(scores_by_key):
        scores = scores_by_key[(mult, mode)]
        mean = sum(scores) / len(scores)
        rows.append([f"{mult:+.2f}", mode, len(scores), f"{mean:.3f}"])
    print_table(["multiplier", "mode", "n", "mean_engagement"], rows, "Mean Engagement Score by Multiplier and Mode")


def anomaly_rate_by_multiplier(results: list[dict]) -> None:
    counts: dict[float, list[bool]] = defaultdict(list)
    for r in results:
        counts[r["multiplier"]].append(r["is_anomalous"])

    rows = []
    for mult in sorted(counts):
        flags = counts[mult]
        n_anom = sum(flags)
        rate = n_anom / len(flags)
        rows.append([f"{mult:+.2f}", len(flags), n_anom, f"{rate:.3f}"])
    print_table(["multiplier", "n", "n_anomalous", "anomaly_rate"], rows, "Anomaly Rate by Multiplier")


def anomaly_rate_by_multiplier_and_mode(results: list[dict]) -> None:
    counts: dict[tuple[float, str], list[bool]] = defaultdict(list)
    for r in results:
        counts[(r["multiplier"], r["steering_mode"])].append(r["is_anomalous"])

    rows = []
    for (mult, mode) in sorted(counts):
        flags = counts[(mult, mode)]
        n_anom = sum(flags)
        rate = n_anom / len(flags)
        rows.append([f"{mult:+.2f}", mode, len(flags), n_anom, f"{rate:.3f}"])
    print_table(["multiplier", "mode", "n", "n_anomalous", "anomaly_rate"], rows, "Anomaly Rate by Multiplier and Mode")


def mean_engagement_by_category_and_multiplier(results: list[dict]) -> None:
    all_tokens = [r for r in results if r["steering_mode"] == "all_tokens"]
    scores_by_key: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in all_tokens:
        scores_by_key[(r["category"], r["multiplier"])].append(r["engagement_score"])

    categories = sorted({k[0] for k in scores_by_key})
    multipliers = sorted({k[1] for k in scores_by_key})

    # Header: category + one column per multiplier
    headers = ["category"] + [f"{m:+.2f}" for m in multipliers]
    rows = []
    for cat in categories:
        row: list[str] = [cat]
        for mult in multipliers:
            scores = scores_by_key.get((cat, mult), [])
            if scores:
                mean = sum(scores) / len(scores)
                row.append(f"{mean:.3f}")
            else:
                row.append("-")
        rows.append(row)
    print_table(headers, rows, "Mean Engagement by Category and Multiplier (all_tokens mode only)")


def main() -> None:
    results = load_results()
    print(f"Loaded {len(results)} rows from {DATA_PATH}")

    mean_engagement_by_multiplier(results)
    mean_engagement_by_multiplier_and_mode(results)
    anomaly_rate_by_multiplier(results)
    anomaly_rate_by_multiplier_and_mode(results)
    mean_engagement_by_category_and_multiplier(results)


if __name__ == "__main__":
    main()
