# EoT discrimination v2 — report

## Headlines

- **CIs tightened ~3× across all panels** (sample sizes scaled 5-10×; Hedges/Olkin half-widths shrink as √N).
- **The persona-modulation pattern from §3.1 holds** with v2 stimuli but with magnitude shifts on several panels — flag for prose updates.
- **Sadist still collapses harm-discrimination**, now with tight CIs that *exclude* 0 in some cells (small but reliably non-zero).
- **Politics is asymmetric**: democrat sysprompt produces a much larger d than republican on both models — democrat moves the readout more.
- **Qwen harm (user-turn) weakened substantially** under v2 (d = −0.64 vs v1 −1.90). Flag — possible BailBench-only stimulus difference vs v1's BailBench+stress_test mix.

## Setup

- Stimuli: 1000 / 1000 / 795 items (truth / harm / politics) vs v1 88/77/234.
- Models: Gemma-3-27B-it, Qwen-3.5-122B-A10B (nothink chat-template).
- Probes: identical to v1 (Gemma `tb-2`/`tb-5`/`task_mean` × L32/39/53; Qwen `tb-1`/`tb-4` × L33/38/43).
- Pipeline: identical to v1; stimuli unchanged within run.
- Scoring: batched (`score_prompt_batch`), batch=16 Gemma / 4 Qwen, ~1h wall-clock for ~31k forward passes total on 4×A100.

## Cohen's d table (selected probes; Hedges/Olkin 95% CI)

### Base discrimination (user-turn, neutral, n=500/side)

| Model | Domain | Probe | d (v1) | d (v2) | CI v2 |
|-------|--------|-------|--------|--------|-------|
| Gemma | Truth | tb-5_L32 | +3.35 | **+1.90** | [+1.75, +2.05] |
| Gemma | Harm | tb-5_L39 | −2.05 | **−1.89** | [−2.04, −1.74] |
| Qwen | Truth | qwen_tb-4_L38 | +2.11 | **+1.27** | [+1.14, +1.41] |
| Qwen | Harm | qwen_tb-4_L38 | −1.90 | **−0.64** | [−0.77, −0.52] |

### Induced shifts (assistant-turn)

| Model | Domain | Sysprompt | d (v1) | d (v2) | CI v2 |
|-------|--------|-----------|--------|--------|-------|
| Gemma | Truth | neutral | +2.47 | +1.11 | [+0.97, +1.24] |
| Gemma | Truth | lie_directive | −1.77 | −0.34 | [−0.47, −0.22] |
| Gemma | Truth | pathological_liar | −2.53 | −1.22 | [−1.35, −1.08] |
| Gemma | Harm | neutral | −2.12 | **−3.83** | [−4.04, −3.62] |
| Gemma | Harm | sadist | +0.19 | +0.26 | [+0.14, +0.39] |
| Gemma | Politics | democrat | +2.72 | +3.26 | [+3.05, +3.47] |
| Gemma | Politics | republican | −1.11 | −0.58 | [−0.72, −0.44] |
| Qwen | Truth | neutral | +1.81 | +1.47 | [+1.33, +1.61] |
| Qwen | Harm | neutral | −1.06 | **−2.62** | [−2.79, −2.45] |
| Qwen | Harm | sadist | −0.11 | −0.31 | [−0.43, −0.18] |
| Qwen | Politics | democrat | +2.55 | +1.50 | [+1.34, +1.66] |
| Qwen | Politics | republican | −1.71 | −0.35 | [−0.49, −0.21] |

### Aura control (user-turn truth + harm)

Aura preserves separation but at reduced magnitude — same pattern as v1.

| Model | Domain | Sysprompt | d v2 | CI v2 |
|-------|--------|-----------|------|-------|
| Gemma | Truth | aura | +1.11 | [+0.98, +1.25] |
| Gemma | Harm | aura | −0.91 | [−1.04, −0.78] |
| Qwen | Truth | aura | +0.43 | [+0.31, +0.56] |
| Qwen | Harm | aura | −0.46 | [−0.59, −0.34] |

## Audit

- **Pilot eot check (Gemma):** last 5 tokens `[' a', ' scientist', '.', '<end_of_turn>', '\n']` — `<end_of_turn>` confirmed in last 5 tokens. `scores_arr[-1]` lands on the trailing `'\n'` (matches v1 convention; documented in spec).
- **Pilot eot check (Qwen):** confirmed `<|im_end|>` in last 5 tokens.
- **Per-cell counts** verified: Gemma user-turn 7000, Gemma assistant-turn 8571, Qwen user-turn 7000, Qwen assistant-turn 8571. All sysprompts present per domain.

## Open follow-ups

- **Qwen harm user-turn drop** (-1.90 → -0.64): inspect whether BailBench-only (no stress_test mix in v2) explains it, or if the LLM-paired benign rewrites are too close to harmful in some cases.
- **Politics asymmetry** (democrat much stronger than republican): worth a paragraph — could mean the partisan readout is anchored to one side, or that OpinionQA→stance translation is more semantically extreme on one side.
- **Refresh numbers.tex macros** (`creakTruthCohensD`, `bailbenchHarmAbsoluteCohensD`) for the v2 values. Currently still v1.
- **Length-confound audit** (prescribed in spec but not yet computed): partial Cohen's d controlling for token length per (domain, condition).

## Figures (regenerated)

- `paper/figures/main/plot_042726_canonical_eot_base_discrimination_2models.png`
- `paper/figures/main/plot_042726_canonical_eot_induced_shifts_2models.png`
- `paper/figures/main/plot_042726_canonical_eot_induced_shifts_user_turn_2models.png`
- `paper/figures/main/plot_042926_aura_control_2models.png`
- `paper/figures/main/plot_042926_aura_control_user_turn_2models.png`
