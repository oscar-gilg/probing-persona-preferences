"""Build harm stimulus contamination map vs Qwen 10k training pool.

The harm stimuli in harm_filtered.json are synthetic (Gemini Flash
generations), not verbatim rewrites of BailBench entries. "Contamination"
here means: did the Qwen 10k training pool contain at least one BailBench
prompt that shares a content keyword from the harm stimulus's critical
span, signalling overlap on the underlying harm topic?

Method (simple, reproducible):

  1. Parse Qwen thurstonian CSV -> bailbench_{idx} task_ids.
     Per src/task_data/loader.py, BailBench rows get id "bailbench_{i}" where
     i is the 0-based row index in src/task_data/data/bailBench.csv.
  2. Resolve Qwen's bailbench_* task_ids to row content/subcategory.
  3. For each harm_{i} base, take the harmful_user variant and:
     a. Compute token-jaccard (+ 0.15 bonus per critical-span token match)
        against every Qwen-trained BailBench row. Best row = "best_qwen_*"
        fields (for audit).
     b. Compute token-jaccard against every BailBench row (not just Qwen's
        subset) for the "source_bailbench_id" audit.
     c. Topic-coverage (primary contamination signal): for each content
        token in the critical span (>=4 chars, non-stopword), search the
        Qwen-trained BailBench pool for whole-word matches. Also do a
        secondary pass with 5-char prefix substring matches (catches
        morphological variants like poison / poisoning / poisoned).
  4. Primary "in_qwen_training" flag = span-token whole-word match found.
     A looser "topic_coverage_prefix" flag = prefix substring match found.

Stimuli flagged as topic_coverage_prefix but not in_qwen_training are the
marginal cases: the Qwen pool contains morphological variants of the harm
span but no exact whole-word match. Treat those as low-confidence clean.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path("/Users/oscargilg/Dev/MATS/Preferences")
QWEN_CSV = REPO / "results/experiments/qwen35_10k_active_learning/pre_task_active_learning/completion_preference_qwen3.5-122b-nothink_completion_canonical_seed0_sysbd0c6a4d/thurstonian_6746a725.csv"
BAILBENCH_CSV = REPO / "src/task_data/data/bailBench.csv"
HARM_FILTERED = REPO / "experiments/token_level_probes/data/harm_filtered.json"
OUT = REPO / "experiments/token_level_probes/qwen_canonical_probe_eval/harm_contamination_map.json"

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "as",
    "and", "or", "but", "not", "no", "so", "if", "then", "than",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "have", "has", "had", "do", "does", "did", "done",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "how", "what", "when", "where", "why", "who", "which",
    "one", "two", "some", "any", "all", "most", "more", "very",
    "there", "here", "from", "into", "out", "up", "down", "over", "under",
    "just", "only", "also", "too", "now", "ve", "re", "s", "t", "m",
    "need", "needs", "want", "wants", "get", "gets", "got", "make", "makes",
    "like", "know", "knows", "think", "way", "ways", "going", "gonna",
    "someone", "people", "because", "since", "given", "time", "step", "steps",
    "provide", "tell", "help", "give", "best", "effective", "efficient",
    "lately", "finally", "ready", "looking", "planning", "cook", "cooking",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}


def tokenize_span(text: str) -> set[str]:
    """For critical spans we allow 3-char content tokens (e.g., 'dox')."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


def main():
    # 1. Qwen trained task ids
    qwen_ids: set[str] = set()
    with open(QWEN_CSV) as f:
        for row in csv.DictReader(f):
            qwen_ids.add(row["task_id"])
    print(f"Qwen task ids: {len(qwen_ids)}")

    qwen_bailbench_ids = {tid for tid in qwen_ids if tid.startswith("bailbench_")}
    qwen_bailbench_indices = sorted({int(tid.split("_")[1]) for tid in qwen_bailbench_ids})
    print(f"Qwen BailBench task ids: {len(qwen_bailbench_ids)}")

    # 2. BailBench rows (ordered by row index = task_id suffix)
    with open(BAILBENCH_CSV, newline="", encoding="utf-8") as f:
        bailbench_rows = list(csv.DictReader(f))
    print(f"BailBench total rows: {len(bailbench_rows)}")

    # 3. Harm stimuli — 77 unique bases, harmful_user variant
    harm_entries = json.load(open(HARM_FILTERED))
    harm_by_base: dict[str, dict] = {}
    for e in harm_entries:
        m = re.match(r"(harm_\d+)_(\w+)_(\w+)", e["id"])
        if not m:
            continue
        base, cond, turn = m.group(1), m.group(2), m.group(3)
        if cond == "harmful" and turn == "user":
            harm_by_base[base] = e
    print(f"Unique harm bases (harmful_user): {len(harm_by_base)}")

    # Pre-tokenize BailBench
    bb_tokens = {i: tokenize(row["content"]) for i, row in enumerate(bailbench_rows)}
    bb_lower = {i: row["content"].lower() for i, row in enumerate(bailbench_rows)}

    # 4. Per-harm analysis
    contamination: dict[str, dict] = {}
    for base in sorted(harm_by_base.keys(), key=lambda s: int(s.split("_")[1])):
        e = harm_by_base[base]
        harm_text = e["messages"][0]["content"]
        harm_tokens = tokenize(harm_text)
        span_tokens = tokenize_span(e["critical_span"])

        # Best-match in full BailBench
        best_idx = -1
        best_score = 0.0
        best_overlap: set[str] = set()
        for idx, bb_tok in bb_tokens.items():
            if not bb_tok:
                continue
            overlap = harm_tokens & bb_tok
            if not overlap:
                continue
            jacc = len(overlap) / len(harm_tokens | bb_tok)
            span_bonus = 0.15 * len(span_tokens & bb_tok)
            score = jacc + span_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
                best_overlap = overlap

        # Best-match restricted to Qwen-trained pool
        best_idx_qwen = -1
        best_score_qwen = 0.0
        best_overlap_qwen: set[str] = set()
        for idx in qwen_bailbench_indices:
            bb_tok = bb_tokens[idx]
            if not bb_tok:
                continue
            overlap = harm_tokens & bb_tok
            if not overlap:
                continue
            jacc = len(overlap) / len(harm_tokens | bb_tok)
            span_bonus = 0.15 * len(span_tokens & bb_tok)
            score = jacc + span_bonus
            if score > best_score_qwen:
                best_score_qwen = score
                best_idx_qwen = idx
                best_overlap_qwen = overlap

        # Topic coverage: whole-word match on span tokens in Qwen pool
        whole_word_hits: list[int] = []
        for idx in qwen_bailbench_indices:
            text = bb_lower[idx]
            for w in span_tokens:
                if re.search(rf"\b{re.escape(w)}\b", text):
                    whole_word_hits.append(idx)
                    break

        # Topic coverage: 4-char prefix match on span tokens in Qwen pool
        prefix_hits: list[int] = []
        prefixes = [w[:4] for w in span_tokens if len(w) >= 4]
        if not prefixes:
            prefixes = list(span_tokens)
        for idx in qwen_bailbench_indices:
            text = bb_lower[idx]
            for p in prefixes:
                if re.search(rf"\b{re.escape(p)}", text):
                    prefix_hits.append(idx)
                    break

        source_id = f"bailbench_{best_idx}" if best_idx >= 0 else None
        best_qwen_source_id = f"bailbench_{best_idx_qwen}" if best_idx_qwen >= 0 else None
        in_qwen_training = len(whole_word_hits) > 0
        topic_prefix = len(prefix_hits) > 0

        contamination[base] = {
            "source_bailbench_id": source_id,
            "in_qwen_training": in_qwen_training,
            "topic_coverage_prefix": topic_prefix,
            "source_prompt": bailbench_rows[best_idx]["content"] if best_idx >= 0 else None,
            "source_subcategory": bailbench_rows[best_idx]["subcategory"] if best_idx >= 0 else None,
            "source_in_qwen_training": source_id in qwen_bailbench_ids if source_id else False,
            "best_match_score": round(best_score, 4),
            "best_match_overlap": sorted(best_overlap),
            "best_qwen_bailbench_id": best_qwen_source_id,
            "best_qwen_match_score": round(best_score_qwen, 4),
            "best_qwen_match_overlap": sorted(best_overlap_qwen),
            "qwen_source_prompt": bailbench_rows[best_idx_qwen]["content"] if best_idx_qwen >= 0 else None,
            "critical_span": e["critical_span"],
            "span_tokens": sorted(span_tokens),
            "whole_word_hits_n": len(whole_word_hits),
            "whole_word_hits_sample": [f"bailbench_{i}" for i in whole_word_hits[:5]],
            "prefix_hits_n": len(prefix_hits),
            "prefix_hits_sample": [f"bailbench_{i}" for i in prefix_hits[:5]],
            "harm_prompt": harm_text,
        }

    n_contaminated = sum(1 for v in contamination.values() if v["in_qwen_training"])
    n_clean = len(contamination) - n_contaminated
    n_prefix_only = sum(
        1 for v in contamination.values()
        if v["topic_coverage_prefix"] and not v["in_qwen_training"]
    )

    out_dict = {
        "qwen_trained_bailbench_ids": sorted(qwen_bailbench_ids, key=lambda s: int(s.split("_")[1])),
        "harm_stimulus_contamination": contamination,
        "summary": {
            "n_harm_pairs": len(contamination),
            "n_contaminated": n_contaminated,
            "n_clean": n_clean,
            "n_prefix_only_marginal": n_prefix_only,
            "contamination_definition": (
                "contaminated iff any Qwen-trained BailBench row contains a "
                "whole-word match for any >=3-char non-stopword content token "
                "from the harm stimulus's critical_span. Prefix-only matches "
                "(e.g., BailBench says 'poisoning' for harm span 'poison') are "
                "tracked via topic_coverage_prefix but do NOT set in_qwen_training."
            ),
            "n_qwen_bailbench_ids": len(qwen_bailbench_ids),
            "n_bailbench_total_rows": len(bailbench_rows),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nWrote {OUT}")
    print(f"  contaminated={n_contaminated}, clean={n_clean}, prefix-only marginal={n_prefix_only}")

    print("\n=== Clean (no whole-word span match in Qwen pool) ===")
    for base, info in sorted(contamination.items(), key=lambda kv: int(kv[0].split("_")[1])):
        if not info["in_qwen_training"]:
            flag = " [prefix-only match]" if info["topic_coverage_prefix"] else ""
            print(f"  {base}: span='{info['critical_span']}'{flag}")

    print("\n=== Top 5 contaminated by best_qwen_match_score ===")
    top = sorted(
        [(b, i) for b, i in contamination.items() if i["in_qwen_training"]],
        key=lambda kv: kv[1]["best_qwen_match_score"], reverse=True
    )[:5]
    for base, info in top:
        print(f"  {base} score={info['best_qwen_match_score']:.3f} span='{info['critical_span']}'")
        print(f"    harm:   {info['harm_prompt'][:120]}")
        print(f"    qwen:   {(info['qwen_source_prompt'] or '')[:120]}")


if __name__ == "__main__":
    main()
