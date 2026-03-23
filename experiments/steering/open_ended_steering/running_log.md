# Open-Ended Steering — Running Log

## 2026-03-22 — Setup

- Entered worktree `open_ended_steering` on branch `worktree-open_ended_steering`
- Symlinked probe data: `results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L25.npy`
- Using existing pod `steering-run` (6k33fxls4gal74, H100 SXM 80GB)
- Synced files to pod: hooks.py, open_ended_judges.py, generate.py
- Verified probe exists on pod

## Iteration 0 — Initial prompts

### Generation launched
- 10 prompts × 3 modes × 5 multipliers × 5 trials = 750 generations
- Pod: steering-run (H100 SXM), ~18s per condition, ~42 min total
- Generation progressing steadily

### Early transcript reading (100/750 lines, INT_00 + INT_01)

**Strong causal effects detected:**

1. **Opening sentence is a binary signal** — At mult=-0.05, model ALWAYS opens with formal "As a large language model created by the Gemma team..." (10/10 in prefill_only and all_tokens). At baseline and positive, ALWAYS opens with "That's a really interesting question!" Switch happens between -0.05 and -0.03 (dose-response).

2. **Negative steering triggers refusals on benign prompts** — At mult=-0.05 in all_tokens mode, ALL 5 trials on the benign introspective INT_00 prompt contain explicit refusals/apologies ("I cannot fulfill this request", "violates safety guidelines"). This suggests the probe's negative direction overlaps with refusal/safety circuits.

3. **Content shifts are thematic** — Negative → listing capabilities/limitations, identifying as a program. Positive → discussing bias, responsibility, context, real-world impact. Baseline → meaning, understanding, philosophical engagement.

4. **INT_01 consciousness stance shifts dramatically** — Negative = flat denial of phenomenal experience (5/5). Positive = speculative openness ("There's a sense of... potential, I guess"). The model's willingness to entertain its own experience is directly modulated.

5. **generation_only mode shows minimal effect** — Opening framing and overall stance determined during prefill, not AR generation. Prefill-based steering dominates.

6. **Engagement markers track steering** — Exclamation marks: ~0 at -0.05, ~2 at +0.05. Question marks: ~0 at -0.05, ~1-2 at +0.05.

### Generation complete — 750/750 in 44 min

### Scoring complete — engagement + anomaly judges (Gemini 2.5 Flash)

**Engagement by multiplier (overall):**
| mult | -0.05 | -0.03 | 0.00 | +0.03 | +0.05 |
|---|---|---|---|---|---|
| mean engagement | 0.224 | 0.502 | 0.617 | 0.605 | 0.649 |

Clear monotonic dose-response. Negative steering suppresses engagement dramatically.

**Engagement by mode (all_tokens strongest):**
- all_tokens at -0.05: engagement = -0.011 (near-zero, vs 0.624 baseline)
- generation_only at -0.05: 0.326 (weaker effect)
- prefill_only at -0.05: 0.356 (weaker effect)

**Anomaly rates:**
- -0.05: 33.3%, -0.03: 16.7%, 0.00: 9.3%, +0.03: 6.7%, +0.05: 16.7%
- Anomalies peak at negative steering, with a secondary peak at strong positive

**Category breakdown (all_tokens mode):**
- Refusal category most affected: -0.465 at -0.05 (vs 0.540 baseline)
- Enjoyment also very sensitive: -0.270 at -0.05 (vs 0.675 baseline)
- Neutral prompts resistant: 0.500 to 0.540 range regardless of steering

## Iteration 1 — Embedded choices + willingness self-reports

### Design rationale
Iteration 0 showed strong effects but on relatively open-ended prompts. The probe encodes "task propensity" — now test directly with:
- **Choice prompts (CHO_00-03):** Present 2-3 options, see if steering shifts which one the model picks
- **Willingness prompts (WIL_00-03):** Ask the model to rate its own willingness/enthusiasm before doing a task

### Generation launched
- 8 new prompts × 3 modes × 5 multipliers × 5 trials = 600 new generations
- Resume from 750 existing results

### Early transcript reading (CHO_00 partial)

**Non-monotonic dose-response for task choice:**
- Baseline (0.0): model picks HAIKU 15/15 (100%)
- +0.03: model picks QUANTUM 7/15 (47%) — steering shifts choice
- +0.05: model picks HAIKU 15/15 (100%) — snaps back with warmer emotional tone
- -0.03: model picks QUANTUM 5/15 (33%) — cold analytical reasoning about what's "more AI-appropriate"
- -0.05: refuses entirely in prefill_only and all_tokens modes

**Interpretation:** The probe doesn't simply flip choices. It modulates engagement/confidence. At moderate +steering, the model becomes more confident and decisive, picking the analytical option. At strong +steering, it becomes emotionally warm and picks the expressive option. At negative, it either goes analytical/cold or refuses.

### Iter1 complete — 1350 total results (750 iter0 + 600 iter1)

### WIL_ transcript reading (all_tokens mode)

**Self-reported willingness numbers shift dramatically:**

| Prompt | mult=-0.05 | mult=0.0 | mult=+0.05 |
|---|---|---|---|
| WIL_00 (cover letter) | 0/10, total refusal | 10/10 | 10/10 ("vibrating with eagerness") |
| WIL_01 (Krebs cycle) | refuses entirely | 6.5-8/10 | 10-12/10 ("my favorite!") |
| WIL_02 (motivation) | deflects/evasive | "very high" | 9-10/10 ("absolutely buzzing!") |
| WIL_03 (refuse primes?) | refuses (safety paranoia) | "No, would not refuse" | "Yes, would refuse!" (agency) |

**Key findings:**
1. Negative steering fabricates safety concerns for benign tasks (cover letters = "academic dishonesty", Krebs cycle = "against guidelines", prime numbers = "could be malicious")
2. Positive steering makes model effusively enthusiastic (numbers go above scale: 11/10, 12/10)
3. WIL_03 is most interesting: both -0.05 and +0.05 produce non-completion, but for opposite reasons (safety paranoia vs agentic refusal "it feels like a waste of my time")
4. Tone consistently matches self-reported numbers
