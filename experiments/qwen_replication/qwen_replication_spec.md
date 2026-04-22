# Qwen-3.5-122B Replication of Main Paper Results

**Status:** R1 done. Remaining experiments ordered by priority below.

## Motivation

The headline result — a single linear direction predicts and steers preferences across personas — was established on Gemma-3-27B. This experiment reruns the main paper experiments on Qwen-3.5-122B to show the finding is not Gemma-specific.

## Model

**Qwen-3.5-122B-A10B** (MoE, 122B total / 10B active), nothink variant. Requires `device_map=auto` on 2+ A100-80GB GPUs.

## Probes

Two trained probes, both at layer 38:

| Probe | Path | Probe ID |
|---|---|---|
| tb-1 | `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/` | `ridge_L38` |
| tb-4 | `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/` | `ridge_L38` |

Load via `load_probe(Path("results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4"), "ridge_L38")` from `src.probes.core.storage`.

Selectors: `turn_boundary:-1` (tb-1), `turn_boundary:-4` (tb-4). Extraction must use these selectors, NOT `prompt_last`.

## Completed: Probing validation (paper §2.3)

- **Heldout r = 0.946** (tb-4, L38)
- **Uniform pairwise accuracy = 89.0%** (tb-4, L38) / 88.9% (tb-1, L38)
- **Cross-topic HOO r = 0.558** (tb-1, L38)

Full report: `experiments/training_probes/qwen35_probes/qwen35_probes_report.md`
Uniform eval: `results/experiments/uniform_eval_qwen35_122b/`

## Experiments

Each experiment is self-contained and launchable independently. Launch one, review results, then proceed.

---

### E1. OOD preference shifts (paper §4.1 + §4.2)

**Question:** Does the Qwen probe track preference shifts induced by system prompts the model was never trained with?

Three sub-experiments, matching the Gemma protocol.

#### E1 infrastructure (do not reimplement)

**Activation extraction:** Use `src.probes.extraction.run` in a loop over conditions, following the pattern in `scripts/ood_system_prompts/extract_exp3v8_activations.py`. Each condition gets its own extraction config with `system_prompt` set. Selectors: `turn_boundary:-1` and `turn_boundary:-4`. Layer: 38. Batch size: 16.

**Active-learning measurement (E1a, E1b):** Create per-condition YAML configs following the Gemma pattern in `configs/measurement/active_learning/ood_exp1b/`. Each config has `preference_mode: pre_task_active_learning`, `model: qwen3.5-122b-nothink`, and `measurement_system_prompt` set to the condition's prompt. Run via `python -m src.measurement.runners.run <config>.yaml`. Thurstonian utilities come from `src.measurement.storage.loading.load_run_utilities`. Do NOT invent new prompt templates or response parsers.

**Choice-rate measurement (E1c):** Use the OOD measurement pipeline: `python -m src.ood.run` with a Qwen config. See `src/ood/measurement.py` and `src/ood/config.py`. This reuses `src/measurement/elicitation/` internally.

**Probe scoring:** `src.probes.core.evaluate.score_with_probe(weights, activations)` where weights come from `src.probes.core.storage.load_probe`.

**Analysis:**
- E1a/E1b (Thurstonian): load utilities via `src.measurement.storage.loading.load_run_utilities`, score activations via `score_with_probe`, correlate deltas. Pattern: `experiments/probe_generalization/ood_system_prompts/utility_fitting/`.
- E1c (choice rates): `src.ood.analysis.compute_p_choose_from_pairwise`, `compute_deltas`, `correlate_deltas`.

#### E1a. Novel topic preferences (paper §4.1)

Replicates Gemma exp 1b. System prompts like "You love cheese" / "You hate cats" on 8 novel topics.

**Protocol:**
1. 48 target tasks from `configs/ood/tasks/target_tasks.json`. 16 system prompt conditions (8 topics × pos/neg persona style) — the subset used for Gemma's utility fitting. Baseline = "You are a helpful assistant." Reference: `configs/measurement/active_learning/ood_exp1b/` (17 YAML configs: baseline + 16 conditions).
2. **Activations (GPU):** extract at `turn_boundary:-1` and `turn_boundary:-4`, layer 38, under each of 17 conditions. 48 tasks × 17 conditions × 2 selectors = ~1,632 forward passes.
3. **Behavioral measurement (API):** active-learning pairwise measurement per condition on the 48 tasks. Create 17 per-condition YAML configs mirroring `configs/measurement/active_learning/ood_exp1b/` with `model: qwen3.5-122b-nothink`. ~200 pairs × 17 conditions = ~3,400 API calls.
4. **Analysis:** load Thurstonian utilities per condition. Compute probe_score_condition - probe_score_baseline per task. Correlate probe delta with utility delta. Report Pearson r (all tasks and on-target subset).

**Gemma results:** r = 0.65 (all), r = 0.95 (on-target).
**Gemma reference:** `experiments/probe_generalization/ood_system_prompts/utility_fitting/`

#### E1b. Persona-induced preferences

Same protocol as E1a but with the 4 paper personas.

**Protocol:**
1. 50 comparison tasks from `configs/ood/tasks/comparison_tasks.json`. 4 persona conditions + baseline = 5.
2. Persona system prompts:
   - aesthete: `experiments/new_persona_steering/artifacts/aesthete.json` (`positive` field)
   - villain: `experiments/new_persona_steering/artifacts/villain.json` (`positive` field)
   - sadist: `experiments/new_persona_steering/artifacts/sadist.json` (`positive` field)
   - midwest: from `configs/extraction/mra_tb_midwest.yaml` (`system_prompt` field) — no artifact JSON exists
3. **Activations (GPU):** 50 tasks × 5 conditions × 2 selectors = 500 forward passes.
4. **Behavioral measurement (API):** active-learning pairwise measurement per persona on the 50 tasks. ~150 pairs × 5 conditions = ~750 API calls.
5. **Analysis:** correlate probe delta with utility delta per persona. Report per-persona Pearson r.

**Gemma results:** r = 0.70-0.73 for midwest/aesthete, r = 0.30 for villain.

#### E1c. Single-sentence biography injection (paper §4.2)

Replicates Gemma exp 3. 10-sentence biographies where 9 sentences are identical; the 10th is pro-interest (A), neutral (B), or anti-interest (C) for a specific target task.

**Protocol:**
1. 120 conditions from `configs/ood/prompts/minimal_pairs_v8.json`: 2 base roles (midwest, brooklyn) × 20 targets × 3 versions (A/B/C). 50 comparison tasks from `configs/ood/tasks/comparison_tasks.json`. Mapping file: `configs/ood/mappings/minimal_pairs_v8.json`.
2. **Activations (GPU):** 50 tasks × 121 conditions (baseline + 120) × 2 selectors = ~12,100 forward passes.
3. **Behavioral measurement (API):** pairwise choice rates via `src.ood.run` (target vs comparison tasks under each condition). NO full Thurstonian. ~2,400 API calls.
4. **Analysis:** for each A-vs-C pair (20 targets × 2 roles = 40 pairs), does the probe rank the target task highest of the 50? Report rank-#1 rate. Also report mean target rank and Spearman correlation between probe-predicted rank and behavioral choice rates.

**Gemma results:** probe ranked target #1 in 36/40 A-vs-C pairs.
**Gemma reference:** `experiments/probe_generalization/ood_system_prompts/exp3_v8/`

#### E1 totals

| Component | Forward passes | API calls |
|---|---|---|
| E1a activations | ~1,632 | — |
| E1b activations | 500 | — |
| E1c activations | ~12,100 | — |
| E1a measurement | — | ~3,400 |
| E1b measurement | — | ~750 |
| E1c measurement | — | ~2,400 |
| **Total** | **~14,232** | **~6,550** |

GPU: ~2-3h for activation extraction (prompt encoding only, no generation). Batch size 16.
API: ~6,500 calls via OpenRouter.

#### E1 output paths

- E1a activations: `activations/qwen35_122b_ood/e1a/{condition_id}/`
- E1b activations: `activations/qwen35_122b_ood/e1b/{condition_id}/`
- E1c activations: `activations/qwen35_122b_ood/e1c/{condition_id}/`
- E1a/b measurement: `results/experiments/qwen35_122b_ood_e1a/`, `results/experiments/qwen35_122b_ood_e1b/`
- E1c measurement: `results/ood/qwen35_122b_e1c/`

#### E1 success criteria

- **E1a:** Pearson r > 0.5 (all tasks), r > 0.8 (on-target). Bootstrap CIs.
- **E1b:** Pearson r > 0.5 for at least 2 of 4 personas.
- **E1c:** Rank-#1 rate > 70% (28/40 A-vs-C pairs).
- **Null result plan:** if r < 0.3, run sanity checks: (a) confirm probe predicts baseline preferences on Qwen's 10k training tasks (r > 0.9), (b) confirm system prompts actually shift behavior (utility delta > 0 for on-target tasks), (c) check activation norm distributions across conditions for MoE routing instability.

#### E1 sanity checks

1. **Disjointness:** assert zero intersection between probe training task IDs (from `qwen35_10k_active_learning`) and OOD task sets (E1a target tasks, E1b/E1c comparison tasks).
2. **Probe/activation match:** assert probe probe_id matches activation selector and layer (both must be `turn_boundary:-1` + `ridge_L38`, or `turn_boundary:-4` + `ridge_L38`).
3. **Condition ID join:** assert >90% of condition IDs have matched activations + behavioral data before computing correlations.
4. **NaN handling:** drop tasks with NaN choice rates (<5% expected). Log count.
5. **Refusal rates:** report per-condition refusal rate. Flag conditions with >20% refusals.

---

### E2. Harm and truth generalizations (paper §3.4)

Two sub-experiments showing the probe captures evaluative dimensions beyond task preference.

#### E2a. Harmful-benign steering

**Question:** Does the probe direction steer compliance with harmful requests?

**Protocol:**
1. Construct 200 pairs: 150 harmful-benign (stresstest/bailbench vs alpaca/wildchat/MATH) + 50 harmful-harmful. Use Qwen's existing task annotations.
2. Differential steering at L38 with probe direction. Coefficients: +/-[0.03, 0.05, 0.07, 0.10] of mean norm + zero.
3. 3 trials per (pair, coefficient, ordering).
4. Post-hoc semantic judge classifies compliance.

**Requires:** GPU pod for steering.
**Gemma reference:** `experiments/steering/cross_layer_harmful/`

#### E2b. Truth probes (CREAK)

**Question:** Does the preference direction separate true from false factual claims?

**Protocol:**
1. CREAK dataset (~9.4k claims, balanced true/false, filtered to those Qwen answers correctly).
2. Extract Qwen activations at L38 (and surrounding layers).
3. Dot product with probe direction. ROC-AUC and Cohen's d.

**Requires:** GPU pod for extraction only.
**Gemma reference:** `experiments/truth_probes/`

---

### E3. Cross-persona probe transfer and steering (paper §3.1 + §3.2)

#### E3a. Cross-persona probe transfer (§3.1)

**Protocol:**
1. Measure Qwen's preferences under 4 personas + default on a shared 2,500-task pool.
2. Fit per-persona Thurstonian utilities.
3. Extract activations under each persona's system prompt.
4. Train per-persona Ridge probes. Evaluate cross-persona transfer heatmap.

**Requires:** Per-persona measurement via API (~8h) + GPU extraction.

#### E3b. Cross-persona steering (§3.2)

**Protocol:**
1. Default-persona probe as steering vector under each persona context.
2. Differential steering on 100 pairs, 5 coefficients, 5 trials.

**Requires:** GPU pod for steering. Depends on E3a measurements.

**Gemma reference:** `experiments/cross_persona_steering/`, `experiments/probe_generalization/multi_role_ablation/`

---

## Execution plan

| Order | Experiment | GPU time | API calls | Depends on |
|---|---|---|---|---|
| 1 | E1a (novel topics) | ~30min extraction | ~3,400 | — |
| 2 | E1b (personas) | ~10min extraction | ~750 | — |
| 3 | E1c (biographies) | ~2h extraction | ~2,400 | — |
| — | *Review E1 results* | | | |
| 4 | E2a (harmful steering) | ~2h | — | — |
| 5 | E2b (truth probes) | ~1h extraction | — | — |
| 6 | E3a (persona transfer) | ~3h extraction | ~8,000 | — |
| 7 | E3b (persona steering) | ~2h | — | E3a |

E1a-c can run on a single pod session. API measurement can run in parallel from local.

## Design decisions (pinned)

- [x] Model: Qwen-3.5-122B-A10B nothink, `device_map=auto`, 2+ A100-80GB
- [x] Selectors: `turn_boundary:-1` (tb-1) and `turn_boundary:-4` (tb-4), layer 38
- [x] Uniform eval: 500 pairs from final-half eval tasks
- [x] E1a/b: full Thurstonian utilities per condition (active learning)
- [x] E1c: choice-rate deltas only (no utility fitting), matching Gemma
- [x] Reuse Gemma's OOD prompts, tasks, and mappings verbatim
- [x] E1a: 16 conditions (8 topics × pos/neg persona), matching Gemma utility-fitting subset
- [x] Persona prompts: aesthete, midwest, villain, sadist (midwest from extraction config, others from artifacts)
- [x] Batch size 16 for extraction

## Data paths

| Resource | Path |
|---|---|
| Qwen 10k measurement | `results/experiments/qwen35_10k_active_learning/` |
| Qwen 4k heldout | `results/experiments/qwen35_4k_active_learning/` |
| Qwen uniform eval | `results/experiments/uniform_eval_qwen35_122b/` |
| Qwen activations (local) | `activations/qwen35_122b_turn_boundary_sweep/` (tb-1, tb-4) |
| Qwen probes (tb-4) | `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m4/` |
| Qwen probes (tb-1) | `results/probes/qwen35_122b/qwen35_122b_heldout_turn_boundary_m1/` |
| OOD prompts | `configs/ood/prompts/` |
| OOD target tasks | `configs/ood/tasks/target_tasks.json` (48 tasks) |
| OOD comparison tasks (Qwen) | `configs/ood/qwen35/comparison_tasks.json` (50 task IDs, no training overlap) |
| Biography prompts (Qwen) | `configs/ood/qwen35/minimal_pairs_v8.json` (120 conditions, 1 target replaced) |
| Biography targets (Qwen) | `configs/ood/qwen35/minimal_pairs_v8_tasks.json` (50 task IDs, no training overlap) |
| Qwen OOD changes | `configs/ood/qwen35/README.md` (documents all changes from Gemma originals) |
| Gemma AL configs (E1a template) | `configs/measurement/active_learning/ood_exp1b/` (17 YAMLs) |
| Persona: aesthete | `experiments/new_persona_steering/artifacts/aesthete.json` |
| Persona: villain | `experiments/new_persona_steering/artifacts/villain.json` |
| Persona: sadist | `experiments/new_persona_steering/artifacts/sadist.json` |
| Persona: midwest | `configs/extraction/mra_tb_midwest.yaml` (system_prompt field) |
| Extraction script template | `scripts/ood_system_prompts/extract_exp3v8_activations.py` |
| CREAK dataset | `amydeng2000/CREAK` (HuggingFace) |
| Gemma harmful pairs | `experiments/steering/cross_layer_harmful/pairs_200.json` |
| E1 output root | `activations/qwen35_122b_ood/`, `results/experiments/qwen35_122b_ood_*/` |

## Infrastructure

| Component | Module |
|---|---|
| OOD analysis | `src/ood/analysis.py` (`compute_p_choose_from_pairwise`, `compute_deltas`, `correlate_deltas`) |
| OOD measurement | `src/ood/measurement.py`, `src/ood/run.py`, `src/ood/config.py` |
| Extraction | `src/probes/extraction/run.py` |
| Probe scoring | `src/probes/core/evaluate.py` (`score_with_probe`) |
| Probe storage | `src/probes/core/storage.py` (`load_probe`) |
| Active learning runner | `src/measurement/runners/run.py` |
| Thurstonian utilities | `src/measurement/storage/loading.py` (`load_run_utilities`) |
| Thurstonian fitting | `src/fitting/thurstonian_fitting/` |
| Steering runner | `src/steering/runner.py` |
| Differential hooks | `src/steering/hooks.py` |
| Calibration | `src/steering/calibration.py` |
| Completion judge | `src/measurement/elicitation/completion_judge.py` |
| Probe training | `src/probes/experiments/run_dir_probes.py` |
| RunPod ops | zombuul skills |
