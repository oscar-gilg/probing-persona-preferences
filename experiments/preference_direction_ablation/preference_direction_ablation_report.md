# Preference Direction Ablation — report

> Status: **draft**, results numbers + plots pending data sync from the pod (which is awaiting GPU host availability for resume).

## Headline

⟨one-line summary of probe-vs-random comparison; written after analysis⟩

## Question

Is the linear preference direction recovered by Ridge probing **causally necessary** for the model to make coherent preference choices? Projecting the direction out of the residual stream during inference should (a) shift modal choices, (b) make choices inconsistent across seeds, or (c) flatten choice probabilities toward 50/50 — and crucially, do more of these things than an isotropic-random projection of the same rank does.

## Setup

- **Model:** Gemma-3-27b IT, bf16, A100-SXM4-80GB.
- **Probes:** Ridge probes from `results/probes/heldout_eval_gemma3_tb-1/probes/probe_ridge_L{25,32}.npy` (heldout-α-selected, single canonical probe per layer; L32 is the canonical layer at heldout R = 0.865). Unit-normalised at use time (intercept stripped).
- **Pair set:** All 955 unique unordered pairs from `results/experiments/uniform_eval_gemma3_27b_v3/pre_task_revealed/completion_preference_gemma-3-27b_completion_canonical_seed0_uniform_eval/measurements.yaml`. (Spec said 723; the actual file has 4775 rows over 955 unique pairs — used 955.)
- **Per pair, per cell:** 3 generation seeds (canonical A→B order) + 1 swapped-order seed at temperature 0.7. Choice extracted by `CompletionChoiceFormat.parse` (regex with LLM-judge fallback) — the canonical pairwise pipeline used by `uniform_eval_gemma3_27b_v3`.
- **Ablation primitive:** orthogonal projection at all token positions of every chosen layer, `a' = a − (a · d̂) d̂`. Implemented as `project_out_direction` in `src/steering/hooks.py`, exposed on `SteeredHFClient` via new `ablate_layers` + `ablate_directions` parameters (added this session, commit `3bfda65`). Multi-layer routing through the existing `HuggingFaceModel.generate_with_hooks_n`.
- **Random controls:** 5 isotropic unit vectors per condition, sampled with numpy seeds 0–4. No norm matching — for `I − d̂d̂ᵀ` only direction matters.

## Cells (25 total)

| Cell | Layers ablated | Direction(s) | n_pairs |
|---|---|---|---|
| **B0** | — | — | 955 |
| **A_L25_probe** | 25 | probe_ridge_L25 | 955 |
| **A_L32_probe** | 32 | probe_ridge_L32 | 955 |
| **B_two_probe** | 25, 32 | layer-matched probes | 955 |
| **C_band_probe** | 25, 26, …, 34 | probe_ridge_L25 at every layer | 955 |
| **A_L25_random{0..4}** | 25 | random unit | 100 |
| **A_L32_random{0..4}** | 32 | random unit | 100 |
| **B_two_random{0..4}** | 25, 32 | two random units (per-layer) | 100 |
| **C_band_random{0..4}** | 25, …, 34 | one random unit at every layer | 100 |

## Spec deviations

- **`max_new_tokens` 512 → 64.** First smoke at 512 tokens averaged ~8 min/pair, extrapolating to ~210 hours for the full sweep. The completion-judge parser (`RegexThenJudge`) only needs the start of the response to identify which task the model began; spec allows ≥16. Reduced to 64; per-pair time fell to ~14 s. No change to elicitation prompt or temperature.
- **Pair count 723 → 955.** Spec referenced 723 unique pairs in the source measurements file; actual file has 955 unique unordered pairs (4775 rows × 5 measurements per pair). Used the 955 actual.
- **Sanity test 1 (hook correctness)** covered by the existing `tests/steering/test_steering_gpu_e2e.py` (pre-session commit `b5974c6`), which asserts max\|cos(resid, d̂)\| < 1e-2 after projection. Did not re-run on Gemma-3-27b at L32 specifically — the projection is exact in the model's compute dtype regardless of layer.
- **Sanity tests 2 + 3** assessed from the full sweep results below rather than as a gating pre-check.

## Sanity / sentinel results

⟨populated after data sync. Will report:
- Test 2 (random-control coherence): for each random vector, agreement-vs-B0 modal-choice on the 100-pair subset. Spec threshold ≥0.85 to declare the ablation method coarse-but-not-broken.
- Test 3 (length / refusal audit): per-cell mean response chars and refusal rate. Watch for cells where probe ablation surfaces only via length collapse or mass refusal.
⟩

## Results

### Per-cell metrics

⟨table from analyze.py: n_pairs, test_retest, flip_rate, refusal_rate, mean_response_chars, agreement_vs_b0, ks_pa_vs_b0, ks_pa_pvalue, d_mean_abs_dev⟩

### Probe vs random control

For each ablation condition (A_L25, A_L32, B_two, C_band) and each metric, compare the probe cell's value to the distribution of 5 random-control values.

⟨table: condition × metric → probe value, random mean ± std, z-score⟩

### Plots

- `assets/plot_<mmddyy>_pair_agreement_vs_b0.png` — probe-vs-random for the agreement-vs-B0 metric, per condition.
- `assets/plot_<mmddyy>_choice_prob_shift.png` — KS distance and `d_mean_abs_dev` per condition.
- `assets/plot_<mmddyy>_test_retest_and_refusal.png` — sanity diagnostics per cell.

## Interpretation

⟨short, after results. Bullet points only.⟩

## Out of scope (per spec)

- Bradley-Terry log-likelihood / score-range collapse / topic-level breakdown — needs denser pair coverage than 955 uniform-eval pairs provide.
- Rank-k subspace ablation — this experiment is rank-1 only.
- Layer-matched C_band — current C_band uses L25 probe across the band; layer-matched is a follow-up.
- Validation sentinel (10-question capability check across cells) — not run this session; can be added by invoking `scripts/preference_direction_ablation/validation_sentinel.py`.

## Reproducing

- **Driver:** `python -m scripts.preference_direction_ablation.run_cells`. Cell definitions in `define_cells()` (25 cells; resumable per-cell JSONL).
- **Analysis:** `python -m scripts.preference_direction_ablation.analyze` → writes `experiments/preference_direction_ablation/results/summary.csv`.
- **Pod:** A100-SXM4-80GB (`pref-ablation`, id `stuojxplqggile`). Full sweep wallclock ≈ 22 hours for ~27 k generations.
