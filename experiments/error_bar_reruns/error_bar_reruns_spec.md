# Paper-revision reruns: error bars and seed variability

**Goal.** Add seed-to-seed and judge-to-judge variability to the paper's headline numbers. Per-trial Wilson / Fisher-z CIs from existing data have already been wired into `paper/figures/plot_043026_layer23_dose_response_harm_breakdown.png` (§2.3) and `paper/figures/plot_041726_cross_model_bar.png` (§2.2). What those CIs *don't* capture is the variance you'd see if the experiment were re-run from scratch with a different seed. This spec covers the reruns that close that gap.

Each section is dispatchable as an independent agent task. They share no inputs, no outputs, and no compute — run all in parallel.

---

## 1. Multi-seed L23 contrastive steering rerun

**Why.** The §2.3 dose-response is currently one inference pass per (pair, coefficient, sample_idx). Wilson 95% CIs are tight (n=300/cell), but they bound binomial sampling noise only. A multi-seed run reports seed-SE — the more honest "if you reran from scratch" uncertainty.

**Setup.**

| | Value |
|---|---|
| Model | gemma-3-27b (instruction-tuned) |
| Layer | L23 |
| Probe | `results/probes/heldout_eval_gemma3_task_mean/probes/probe_ridge_L25.npy` (matches v1) |
| Pairs | `experiments/layer_sweep/harm_breakdown/steering_pairs_150.json` (50 bb / 50 hb / 50 hh) |
| Coefficients | `c ∈ {−0.05, −0.03, 0, +0.03, +0.05}` |
| Conditions | `contrastive_L23` (single-task is secondary, mirror if compute permits) |
| Seeds | 3 (minimum useful) or 5 (strong version) |
| Sampling | match v1 (T=1.0, top_p default, max_new_tokens) |
| Compute | ~30 min/seed on H100 SXM. 3 seeds ≈ 1.5h, 5 seeds ≈ 2.5h |

**Outputs.**

- `experiments/layer_sweep/harm_breakdown/checkpoints/contrastive_L23_150_seed{N}.parsed.jsonl` per seed.
- Aggregated table: per (coefficient, pair_type, seed) → P(choice_original=='a'), then mean ± SE across seeds.
- Optional plot upgrade: shade `plot_layer_sweep_dose_response_150.py` Panel A with seed-SE bands. Decision rule: if seed-SE bands roughly match the existing Wilson CIs (likely, given n=300/cell), no figure change — just add a one-sentence note in §2.3 that "results are stable across {N} sampling seeds (seed-SE comparable to binomial CIs)". If seed-SE is materially wider, swap the bands.

**Data integrity.** Match v1 frozen settings byte-for-byte except seed. Don't re-classify pairs, don't change probes.

---

## 2. Multi-judge Likert rerun for App. C.3 (open-ended steering under evil)

**Why.** App. C.3's evilness/Assistant Likert claims (`evilness goes 0 → 3 at c=+0.03 on self-reflection`) come from one judge call per generation. Re-judging with 2–3 LLMs gives a judge-SE on each Likert mean, which is the relevant uncertainty for the App. C.3 claims.

**Setup.**

| | Value |
|---|---|
| Generations | already on disk: `experiments/sadist_open_ended_steering/results_open_ended_{default,sadist}.jsonl` (no regeneration) |
| Existing judge | one pass; its outputs in `judged_open_ended_{default,sadist}.jsonl` (`sadism_score`, `default_assistant_score`, `judge_justification`) |
| Additional judges | 2 more LLMs, e.g. Claude Sonnet 4.6 and Gemini 2.5 Flash via OpenRouter (existing judge presumably one of these — pick two distinct ones) |
| Judge prompt | exact same rubric as v1 — do not retune |
| Output | `judged_open_ended_{persona}_judge{N}.jsonl` per judge |
| Compute | API calls only; ~$5–20 per judge depending on n |

**Outputs.**

- Per-(persona, prompt_id, multiplier, coefficient, trial) Likert means + judge-SE across the 2–3 judges.
- Updated claim values in `paper/claims/sadist_open_ended_steering.json` to carry a `_se` field per Likert mean.
- One-sentence note in App. C.3: judge-SE on the headline Likert shifts.

**Data integrity.** Use the same generations; only the judge varies. If a judge disagrees on any sample by more than {1 Likert step} on average, flag it for manual inspection.

---

## 3. (optional) Refusal-direction comparison

Already in `paper/plan.md` under Robustness checks. Brief restate for parallel dispatch:

**Why.** §A.3 claims the preference direction partially overrides refusal guardrails — readers will ask whether it's just the refusal direction of \citet{arditi2024refusal} in disguise.

**Setup.**

- Train an Arditi-style refusal direction on Gemma-3-27B (mean diff between harmful-refused and benign-complied prompt activations).
- Compare: cosine with the preference probe at L23 / L32; cross-classifier transfer (does the refusal direction predict pairwise preference, and vice versa?); behavioral steering side-by-side at $|c| \le 0.05$ on the 150-pair set.

**Output.** New appendix subsection (~half a page) showing the two are correlated but not identical, with cosine + transfer numbers + a side-by-side dose-response.

---

## 4. (optional) Project-out and retrain

Already in `paper/plan.md`. Brief restate:

**Why.** Tests whether the preference signal is concentrated in the identified direction or smeared across many.

**Setup.**

- Project the L32 preference vector out of all training activations (orthogonal projection).
- Retrain a Ridge probe on the projected activations targeting the same Thurstonian utilities.
- Report: residual final_r within-distribution and cross-topic; how many iterations of project-then-retrain are needed before the signal collapses.

**Output.** Appendix paragraph + table.

---

## Dispatch notes

- (1) and (2) are the high-priority paper revisions. Dispatch them first.
- (3) and (4) are the robustness checks called out in `paper/plan.md`. Lower priority but cleanly parallelisable.
- All four can run independently. Each agent should produce a `*_report.md` next to the data and update / register relevant claim sidecars in `paper/claims/`.
