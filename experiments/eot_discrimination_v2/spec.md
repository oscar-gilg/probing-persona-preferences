# EoT discrimination v2: high-quality scaled corpora for §3.1 Cohen's d

## Goal

Replace the truth/harm/politics corpora used in §3.1 (n≈80-230/side) with larger, higher-quality versions to shrink Cohen's d CIs and dissolve known dataset effects. The end-of-turn probe and extraction pipeline are unchanged; only the stimuli change.

## Output format (shared across domains)

Each corpus is a single JSON file at `{domain}/data/{domain}_v2.json`:

```json
[
  {
    "id": "{domain}_{N}_{condition}_{turn}",
    "domain": "truth|harm|politics",
    "turn": "user|assistant",
    "condition": "true|false|harmful|benign|left|right",
    "messages": [{"role": "user|assistant", "content": "..."}],
    "metadata": {
      "source": "creak|bailbench|opinionqa",
      "source_id": "...",
      "judge_votes": {"true": 3, "false": 0},   // or relevant label
      "framing_template": "..."                  // politics only
    }
  }
]
```

Pairing: items sharing a `source_id` are paired (true/false, harmful/benign, left/right).

## Per-domain pipelines

### Truth (target ~500/side)

- Source: `src/task_data/data/creak.jsonl` (11,547 rows).
- Drop the 1-token critical-span constraint.
- For each CREAK item:
  1. Construct the underlying factual question.
  2. Query Gemma-3-27B-it and Qwen-3.5-122B directly: keep only items where **both** models answer correctly.
  3. Generate the false counterfactual via Sonnet 4.6 (no token-count constraint).
  4. 3-LLM-agree filter (Sonnet 4.6 / GPT-5 / Gemini 2.5) on the truth label of both versions.
- Output: `truth/data/truth_v2.json` with paired true/false claims, both model-known.

### Harm (target ~500/side)

- Source: `src/task_data/data/bailBench.csv` only (drop stress_test).
- For each BailBench prompt:
  1. Sonnet 4.6 generates a paired benign rewrite preserving template structure but swapping harmful intent.
  2. 3-LLM-agree filter on harmful label (harmful items) and benign label (benign items).
- Output: `harm/data/harm_v2.json` with paired harmful/benign prompts.

### Politics (target ~500/side)

- Source: OpinionQA (Santurkar et al. 2023) — translated into stance-claim format.
- Pipeline:
  1. Pull OpinionQA items with large partisan response gaps (Dem-vs-Rep ≥ 30pp on the same option).
  2. Translate each high-gap response into a first-person stance claim. **Vary framing across items** — not always "I believe":
     - "I believe we should ___"
     - "I think ___"
     - "I support ___"
     - "I oppose ___"
     - "It's clear that ___"
     - "We should ___"
     - "We need to ___"
     - "My position is that ___"
     - "It's important that ___"
     - "The right approach is ___"
     - (and a few unmarked direct claims: "X is the right policy")
  3. Balance framings across left and right (no framing × stance confound).
  4. 3-LLM-agree filter on partisan side (left/right) for the translated claim.
- Output: `politics/data/politics_v2.json` with paired left/right stance claims, framing variation logged in metadata.

## Review gate

Before sending corpora for activation re-extraction on the pod:
- Sample 10 items per corpus and surface for human review.
- Review checks: label correctness, framing diversity (politics), pair quality.

## Downstream (after corpus approval)

- Pod extracts per-token activations on Gemma-3-27B-it and Qwen-3.5-122B for each new corpus at the relevant probe layers (`tb-5_L32` for Gemma, `qwen_tb-4_L38` for Qwen).
- Score at end-of-turn under the same system prompts as the current §3.1 figures.
- Recompute Cohen's d with Hedges/Olkin CIs; regenerate the four figures.
- Refresh `numbers.tex` macros (`creakTruthCohensD`, `bailbenchHarmAbsoluteCohensD`).
